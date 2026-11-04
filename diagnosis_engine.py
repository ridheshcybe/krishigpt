"""Stage 2A Layer 1 - structured, deterministic diagnosis engine.

Turns the unified model's structured output into a farmer-facing diagnosis
draft using only two sources of truth:

1. the model output it is given (predicted class, confidence, attention
   weights, tabular attributions, Grad-CAM summary, NDVI trend), and
2. ``knowledge_base.json`` (per-class description, symptoms, favourable
   conditions, actions, escalation and numeric severity rules).

It never invents facts: every sentence is either a number from the model
output or text from the knowledge base.  Layer 1 has no network, no LLM and
no optional dependency, so it works offline and is the guaranteed fallback
for ``llm_polish.py``.

Severity is computed, not asserted:
* start from the class's ``severity_rules.base``;
* escalate one level (low -> medium -> high) for every matched
  ``critical_features`` rule (numeric op against a feature value);
* escalate one level when confidence falls below
  ``high_if_confidence_below``;
* every escalation appends its reason, so the output can justify itself.

Usage
-----
    from diagnosis_engine import ModelOutput, diagnose, load_knowledge_base
    diagnosis = diagnose(ModelOutput(predicted_class="disease_stress", ...))
    print(diagnosis.draft)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_KNOWLEDGE_PATH = Path("knowledge_base.json")

#: Escalation ladder used by the severity rules (index = level).
SEVERITY_LEVELS: tuple[str, ...] = ("low", "medium", "high")

#: Attention below this share means no single modality dominated the decision.
DOMINANCE_THRESHOLD = 0.45

_REQUIRED_KEYS = (
    "description",
    "visual_symptoms",
    "favorable_conditions",
    "actions",
    "escalation",
    "severity_rules",
)


# ---------------------------------------------------------------------------
# Data contract
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TopFeature:
    """One tabular attribution: how much a feature moved the prediction."""

    feature: str
    value: float
    contribution: float          # >0 supports the predicted class, <0 argues against


@dataclass(frozen=True)
class ModelOutput:
    """Structured output of the unified multi-modal model for one sample."""

    predicted_class: str
    confidence: float
    attention_weights: dict[str, float]
    tabular_top_features: list[TopFeature] = field(default_factory=list)
    gradcam_summary: str = "no image attribution available"
    ndvi_trend: str = "no NDVI trend available"
    ndvi_is_proxy: bool = True
    true_class: str | None = None
    sample_id: str = ""


@dataclass(frozen=True)
class Diagnosis:
    """Layer 1 result: structured fields plus the rendered template draft."""

    sample_id: str
    predicted_class: str
    confidence: float
    severity: str
    severity_reasons: list[str]
    driver_modality: str
    driver_share: float
    factors: list[str]
    temporal_context: str
    symptoms: list[str]
    actions: list[str]
    escalation: str
    structured: dict[str, Any]
    draft: str


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------
def load_knowledge_base(path: Path = DEFAULT_KNOWLEDGE_PATH) -> dict[str, Any]:
    """Load and validate the knowledge base (all 6 classes required).

    Raises:
        FileNotFoundError: if the JSON file is missing.
        ValueError: if a class or a required section is missing/incomplete.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Knowledge base missing: {path}. Stage 2A cannot run without it."
        )
    kb = json.loads(path.read_text(encoding="utf-8"))
    classes = kb.get("meta", {}).get("classes", [])
    missing = [c for c in classes if c not in kb]
    if missing:
        raise ValueError(f"knowledge_base.json is missing classes: {missing}")
    for name in classes:
        entry = kb[name]
        absent = [k for k in _REQUIRED_KEYS if k not in entry]
        if absent:
            raise ValueError(
                f"knowledge_base.json[{name!r}] missing sections: {absent}"
            )
    return kb


def feature_label(feature: str, kb: dict[str, Any]) -> str:
    """Human-readable name for a model feature (falls back to the raw name)."""
    labels = kb.get("meta", {}).get("feature_labels", {})
    return labels.get(feature, feature.replace("_", " "))


# ---------------------------------------------------------------------------
# Severity
# ---------------------------------------------------------------------------
def _match(value: float, op: str, threshold: float) -> bool:
    """Evaluate one numeric severity rule (operators: < <= > >=)."""
    if op == "<":
        return value < threshold
    if op == "<=":
        return value <= threshold
    if op == ">":
        return value > threshold
    if op == ">=":
        return value >= threshold
    raise ValueError(f"unsupported severity operator: {op!r}")


def compute_severity(
    model_output: ModelOutput, kb_entry: dict[str, Any], kb: dict[str, Any]
) -> tuple[str, list[str]]:
    """Return ``(severity, reasons)`` from the KB rules + model numbers.

    Rules (deterministic, documented in knowledge_base.json):
    start at the class base level, escalate one step per matched critical
    feature, and one more step if confidence is below the class threshold.
    Level caps at "high" and reasons accumulate in order.
    """
    rules = kb_entry["severity_rules"]
    level = SEVERITY_LEVELS.index(rules.get("base", "low"))
    reasons: list[str] = [f"baseline for this class: {SEVERITY_LEVELS[level]}"]

    values = {f.feature: f.value for f in model_output.tabular_top_features}
    for rule in rules.get("critical_features", []):
        feature = rule["feature"]
        if feature in values and _match(values[feature], rule["op"], rule["value"]):
            if level < len(SEVERITY_LEVELS) - 1:
                level += 1
            reasons.append(
                f"{feature_label(feature, kb)} is {values[feature]:g} "
                f"(rule {rule['op']}{rule['value']:g}: {rule['note']})"
            )

    threshold = rules.get("high_if_confidence_below")
    if threshold is not None and model_output.confidence < threshold:
        if level < len(SEVERITY_LEVELS) - 1:
            level += 1
        reasons.append(
            f"low model confidence ({model_output.confidence:.2f} < {threshold:g})"
        )

    return SEVERITY_LEVELS[level], reasons


# ---------------------------------------------------------------------------
# Narrative pieces
# ---------------------------------------------------------------------------
def describe_driver(model_output: ModelOutput) -> tuple[str, float, str]:
    """Return ``(modality, share, sentence)`` describing who drove the call."""
    weights = model_output.attention_weights
    if not weights:
        return "unknown", 0.0, "No attention weights were recorded for this sample."
    driver = max(weights, key=weights.get)
    share = float(weights[driver])
    if share < DOMINANCE_THRESHOLD:
        joined = ", ".join(f"{k}={v:.2f}" for k, v in weights.items())
        return (
            "mixed",
            share,
            f"No single modality dominated (attention {joined}).",
        )
    return (
        driver,
        share,
        f"The {driver} modality drove this decision (attention {share:.2f}).",
    )


def describe_factors(
    model_output: ModelOutput, kb: dict[str, Any], limit: int = 3
) -> list[str]:
    """Render the strongest tabular attributions as human-readable lines."""
    ordered = sorted(
        model_output.tabular_top_features, key=lambda f: abs(f.contribution), reverse=True
    )[:limit]
    lines: list[str] = []
    for feat in ordered:
        label = feature_label(feat.feature, kb)
        if feat.contribution >= 0:
            relation = "supports the prediction"
        else:
            relation = "argues against the prediction"
        lines.append(
            f"{label} = {feat.value:g} ({relation}, contribution {feat.contribution:+.2f})"
        )
    return lines


def build_structured(
    model_output: ModelOutput, kb: dict[str, Any]
) -> dict[str, Any]:
    """Assemble the Layer 1 structured diagnosis (JSON-serialisable)."""
    class_name = model_output.predicted_class
    entry = kb.get(class_name)
    if entry is None:
        raise ValueError(
            f"predicted class {class_name!r} is not in the knowledge base "
            f"(known: {sorted(k for k in kb if k != 'meta')})"
        )
    severity, reasons = compute_severity(model_output, entry, kb)
    driver, share, driver_sentence = describe_driver(model_output)
    factors = describe_factors(model_output, kb)

    temporal = model_output.ndvi_trend
    if model_output.ndvi_is_proxy and temporal:
        temporal += " [PROXY series - prototype data, not a real satellite readout]"

    return {
        "sample_id": model_output.sample_id,
        "predicted_class": class_name,
        "confidence": round(float(model_output.confidence), 4),
        "true_class": model_output.true_class,
        "severity": severity,
        "severity_reasons": reasons,
        "attention_weights": {
            k: round(float(v), 4) for k, v in model_output.attention_weights.items()
        },
        "driver_modality": driver,
        "driver_share": round(share, 4),
        "driver_sentence": driver_sentence,
        "tabular_top_features": [asdict(f) for f in model_output.tabular_top_features],
        "contributing_factors": factors,
        "gradcam_summary": model_output.gradcam_summary,
        "temporal_context": temporal,
        "knowledge": {
            "description": entry["description"],
            "visual_symptoms": list(entry["visual_symptoms"]),
            "favorable_conditions": dict(entry["favorable_conditions"]),
            "actions": list(entry["actions"]),
            "escalation": entry["escalation"],
        },
        "evidence_sources": ["model output", "knowledge_base.json"],
    }


def render_draft(structured: dict[str, Any]) -> str:
    """Render the structured diagnosis as a deterministic text draft."""
    kb = structured["knowledge"]
    lines: list[str] = []
    lines.append(
        f"PREDICTION: {structured['predicted_class']} "
        f"(confidence {structured['confidence'] * 100:.0f}%) | "
        f"severity: {structured['severity'].upper()}"
    )
    lines.append(structured["driver_sentence"])
    lines.append(f"What it means: {kb['description']}")
    if structured["contributing_factors"]:
        lines.append("Contributing factors:")
        lines.extend(f"  - {f}" for f in structured["contributing_factors"])
    lines.append(
        f"Visual findings (image branch): {structured['gradcam_summary']}"
    )
    lines.append(f"Temporal context: {structured['temporal_context']}")
    lines.append("Expected symptoms: " + "; ".join(kb["visual_symptoms"]))
    lines.append("Conditions that favour it: " + "; ".join(
        f"{k}: {v}" for k, v in kb["favorable_conditions"].items()
    ))
    lines.append("Recommended actions:")
    lines.extend(f"  {i}. {a}" for i, a in enumerate(kb["actions"], start=1))
    lines.append(f"When to escalate: {kb['escalation']}")
    lines.append("Severity because:")
    lines.extend(f"  - {r}" for r in structured["severity_reasons"])
    if structured["true_class"] is not None:
        lines.append(
            f"(ground truth for this sample: {structured['true_class']})"
        )
    lines.append(
        "Evidence basis: model output + knowledge_base.json only; "
        "no external facts added."
    )
    return "\n".join(lines)


def diagnose(
    model_output: ModelOutput, knowledge_base: dict[str, Any] | None = None
) -> Diagnosis:
    """Run Layer 1 end to end: severity, factors, template draft.

    Args:
        model_output: structured output of the unified model.
        knowledge_base: pre-loaded knowledge base (loaded from
            ``knowledge_base.json`` when omitted).

    Returns:
        A :class:`Diagnosis` with both the structured dict and the draft.
    """
    kb = knowledge_base if knowledge_base is not None else load_knowledge_base()
    structured = build_structured(model_output, kb)
    draft = render_draft(structured)
    return Diagnosis(
        sample_id=structured["sample_id"],
        predicted_class=structured["predicted_class"],
        confidence=structured["confidence"],
        severity=structured["severity"],
        severity_reasons=structured["severity_reasons"],
        driver_modality=structured["driver_modality"],
        driver_share=structured["driver_share"],
        factors=structured["contributing_factors"],
        temporal_context=structured["temporal_context"],
        symptoms=list(structured["knowledge"]["visual_symptoms"]),
        actions=list(structured["knowledge"]["actions"]),
        escalation=structured["knowledge"]["escalation"],
        structured=structured,
        draft=draft,
    )
