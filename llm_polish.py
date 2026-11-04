"""Stage 2A Layer 2 - optional LLM polish for the Layer 1 diagnosis draft.

Sends the deterministic Layer 1 draft to an LLM and asks it to rewrite the
technical diagnosis as short, farmer-friendly advice.  Everything about this
layer is optional by construction:

* no third-party SDK is imported (HTTP via the standard library only);
* provider discovery is automatic (Ollama local -> OpenAI -> Anthropic) and
  driven by environment variables;
* ``polish()`` never raises and never returns empty on failure - it returns
  ``None`` so the caller keeps Layer 1's text, exactly as the spec requires
  ("if LLM unavailable, fall back to Layer 1's template output, never fail").

Environment variables
---------------------
    KISANGPT_LLM_PROVIDER   "ollama" | "openai" | "anthropic" | "auto" (default)
    KISANGPT_LLM_MODEL       model name override (otherwise: llama3 if
                             installed, else the smallest local Ollama model)
    KISANGPT_LLM_TIMEOUT     seconds (default 120; a cold Ollama model load
                             plus generation easily exceeds a 20s budget)
    OLLAMA_HOST              default http://127.0.0.1:11434
    OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL (default gpt-4o-mini)
    ANTHROPIC_API_KEY / ANTHROPIC_MODEL (default claude-3-5-haiku-latest)

Usage
-----
    from llm_polish import polish, describe_provider
    text = polish(layer1_draft)          # str, or None to keep Layer 1
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

#: System prompt mandated by the Stage 2A spec.
SYSTEM_PROMPT = (
    "You are an agricultural extension officer. Rewrite this technical "
    "diagnosis into simple, actionable advice for a farmer. Keep it under "
    "150 words. Use local crop terminology. Do not invent facts not in the "
    "input."
)


@dataclass(frozen=True)
class LLMConfig:
    """Resolved provider settings for Layer 2."""

    provider: str            # "ollama" | "openai" | "anthropic"
    model: str
    base_url: str
    api_key: str | None
    timeout: float


def _env(name: str, default: str | None = None) -> str | None:
    """Read an environment variable (None when unset/empty)."""
    value = os.environ.get(name)
    return value if value else default


def _ollama_installed_models(host: str, timeout: float = 2.0) -> list[dict[str, Any]]:
    """Return the models an Ollama daemon reports (empty list on failure)."""
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        return list(data.get("models", []))
    except Exception:                                    # noqa: BLE001
        return []


def _pick_ollama_model(host: str) -> str:
    """Choose an installed model: env override, then llama3, then the
    smallest local model (fastest on CPU; cloud-linked models skipped)."""
    override = _env("KISANGPT_LLM_MODEL")
    if override:
        return override
    models = _ollama_installed_models(host)
    names = [str(m.get("name", "")) for m in models]
    if "llama3" in names:
        return "llama3"
    local = [
        (int(m.get("size", 0)), str(m.get("name", "")))
        for m in models
        if "cloud" not in str(m.get("name", "")).lower()
    ]
    if local:
        return min(local)[1]
    return "llama3"


def detect_config(provider: str | None = None) -> LLMConfig | None:
    """Resolve which LLM to use, or ``None`` when no provider is configured.

    Order (``auto``): Ollama on the default local port, then OpenAI, then
    Anthropic.  An explicitly requested provider that lacks its credentials
    also yields ``None`` - the caller falls back to Layer 1.
    """
    choice = (provider or _env("KISANGPT_LLM_PROVIDER", "auto") or "auto").lower()
    timeout = float(_env("KISANGPT_LLM_TIMEOUT", "120") or 120)

    if choice in ("auto", "ollama"):
        host = _env("OLLAMA_HOST", "http://127.0.0.1:11434")
        base = (host or "").rstrip("/")
        cfg = LLMConfig(
            provider="ollama",
            model=_pick_ollama_model(base),
            base_url=base,
            api_key=None,
            timeout=timeout,
        )
        if choice == "ollama" or _probe_reachable(cfg.base_url, 1.0):
            return cfg
        if choice == "ollama":
            return None

    if choice in ("auto", "openai"):
        key = _env("OPENAI_API_KEY")
        if key:
            return LLMConfig(
                provider="openai",
                model=_env("KISANGPT_LLM_MODEL", "gpt-4o-mini") or "gpt-4o-mini",
                base_url=(
                    _env("OPENAI_BASE_URL", "https://api.openai.com/v1") or ""
                ).rstrip("/"),
                api_key=key,
                timeout=timeout,
            )
        if choice == "openai":
            return None

    if choice in ("auto", "anthropic"):
        key = _env("ANTHROPIC_API_KEY")
        if key:
            return LLMConfig(
                provider="anthropic",
                model=_env("KISANGPT_LLM_MODEL", "claude-3-5-haiku-latest")
                or "claude-3-5-haiku-latest",
                base_url="https://api.anthropic.com",
                api_key=key,
                timeout=timeout,
            )
        if choice == "anthropic":
            return None

    return None


def _probe_reachable(base_url: str, timeout: float) -> bool:
    """Cheap reachability check (does an Ollama daemon answer?)."""
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=timeout):
            return True
    except Exception:                                    # noqa: BLE001
        return False


def _post_json(
    url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float
) -> dict[str, Any]:
    """POST JSON and parse the response (raises on any network/HTTP error)."""
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json", **headers}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _extract_text(data: dict[str, Any], provider: str) -> str:
    """Pull the assistant text out of a provider-specific response shape."""
    if provider == "ollama":
        return str(data.get("message", {}).get("content", "")).strip()
    if provider == "openai":
        return str(
            data.get("choices", [{}])[0].get("message", {}).get("content", "")
        ).strip()
    if provider == "anthropic":
        blocks = data.get("content", [])
        return "".join(
            str(b.get("text", "")) for b in blocks if b.get("type") == "text"
        ).strip()
    return ""


def _call(config: LLMConfig, user_prompt: str) -> str:
    """Dispatch one completion request to the configured provider."""
    system = {"role": "system", "content": SYSTEM_PROMPT}
    user = {"role": "user", "content": user_prompt}

    if config.provider == "ollama":
        data = _post_json(
            f"{config.base_url}/api/chat",
            {
                "model": config.model,
                "messages": [system, user],
                "stream": False,
                "options": {"temperature": 0.2},
            },
            {},
            config.timeout,
        )
        return _extract_text(data, "ollama")

    if config.provider == "openai":
        data = _post_json(
            f"{config.base_url}/chat/completions",
            {
                "model": config.model,
                "messages": [system, user],
                "temperature": 0.2,
                "max_tokens": 400,
            },
            {"Authorization": f"Bearer {config.api_key}"},
            config.timeout,
        )
        return _extract_text(data, "openai")

    if config.provider == "anthropic":
        data = _post_json(
            f"{config.base_url}/v1/messages",
            {
                "model": config.model,
                "system": SYSTEM_PROMPT,
                "messages": [user],
                "max_tokens": 400,
                "temperature": 0.2,
            },
            {
                "x-api-key": str(config.api_key),
                "anthropic-version": "2023-06-01",
            },
            config.timeout,
        )
        return _extract_text(data, "anthropic")

    raise ValueError(f"unknown provider: {config.provider!r}")


def polish(
    draft: str,
    config: LLMConfig | None = None,
    provider: str | None = None,
) -> str | None:
    """Rewrite a Layer 1 draft with an LLM, or return ``None`` to keep it.

    Contract: never raises.  Any configuration gap, network error, timeout,
    non-200 response or empty completion returns ``None``, and the caller
    falls back to Layer 1's text.
    """
    try:
        cfg = config if config is not None else detect_config(provider)
        if cfg is None:
            return None
        prompt = (
            "Technical diagnosis draft:\n\n"
            f"{draft}\n\n"
            "Rewrite it for the farmer as specified."
        )
        text = _call(cfg, prompt)
        return text or None
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError,
            KeyError, json.JSONDecodeError, OSError):
        return None
    except Exception:                                    # noqa: BLE001
        # Layer 2 must never break Layer 1.
        return None


def describe_provider(provider: str | None = None) -> str:
    """Human-readable status of Layer 2 for CLI headers."""
    cfg = detect_config(provider)
    if cfg is None:
        return "LLM polish: unavailable (Layer 1 template output will be used)"
    return f"LLM polish: {cfg.provider}/{cfg.model} @ {cfg.base_url}"
