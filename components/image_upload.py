"""Stage 2B Component - Leaf Image Input.

Supports:
1. Modern card container with tabbed input:
   - "📁 Instant Presets" (1-click test cards from all stress/healthy classes)
   - "📤 Upload Image" (drag-and-drop file uploader with preview & remove)
   - "🎲 Random Test Sample" (shuffle through held-out test splits)
2. Interactive thumbnails previewing actual dataset samples.
3. Clean metadata display and full error handling.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import streamlit as st
from PIL import Image

ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png", "bmp", "webp"]
DEFAULT_TEST_DIR = Path("data/paddy_images/test")

CLASS_DISPLAY_NAMES = {
    "healthy": "🌱 Healthy Paddy",
    "Blast": "🍂 Rice Blast Disease",
    "BLB": "🌾 Bacterial Leaf Blight",
    "leaf_spot": "🟤 Brown Leaf Spot",
    "hispa": "🐛 Rice Hispa Pest/Damage",
}


@st.cache_data(show_spinner=False)
def load_test_image_pool(test_dir: str = "data/paddy_images/test") -> dict[str, list[str]]:
    """Scan and index held-out test set images grouped by category folder."""
    root = Path(test_dir)
    if not root.exists():
        alt_root = Path("data/paddy_images")
        if alt_root.exists():
            root = alt_root

    category_images: dict[str, list[str]] = {}
    if not root.exists():
        return category_images

    for category_dir in root.iterdir():
        if category_dir.is_dir():
            imgs = [
                str(p.resolve())
                for p in category_dir.glob("*.*")
                if p.suffix.lower() in [f".{ext}" for ext in ALLOWED_EXTENSIONS]
            ]
            if imgs:
                category_images[category_dir.name] = sorted(imgs)

    return category_images


def collect_image(
    key: str = "leaf_upload",
    accept_multiple: bool = False,
) -> tuple[Image.Image | None, str, str]:
    """Render the polished leaf image input interface."""
    category_pools = load_test_image_pool(str(DEFAULT_TEST_DIR))
    all_images = [img for pool in category_pools.values() for img in pool]

    st.markdown(
        """
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
            <h4 style="margin: 0; font-size: 1.15rem; color: #1e293b; font-weight: 700;">
                📸 1. Crop Leaf Visual Input
            </h4>
            <span style="font-size: 0.8rem; background: #e2e8f0; color: #475569; padding: 2px 8px; border-radius: 9999px; font-weight: 600;">
                MobileNetV2 Stream
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    source_tab = st.radio(
        "Source Mode",
        options=["⚡ Preset Samples", "📤 Upload Custom Photo", "🎲 Random Test Pool"],
        horizontal=True,
        label_visibility="collapsed",
        key=f"{key}_mode_select",
    )

    image: Image.Image | None = None
    filename: str = ""
    source: str = "none"

    if source_tab == "⚡ Preset Samples":
        st.caption("Select a real test sample across distinct visual pathologies:")

        avail_classes = list(category_pools.keys())
        if not avail_classes:
            st.warning("No test sample folders located in `data/paddy_images/test`.")
            return None, "", "none"

        selected_cat = st.selectbox(
            "Select Condition Type:",
            options=avail_classes,
            format_func=lambda c: CLASS_DISPLAY_NAMES.get(c, c),
            key=f"{key}_preset_cat",
        )

        samples_in_cat = category_pools[selected_cat]
        sample_labels = [f"Sample #{i+1} ({Path(p).stem})" for i, p in enumerate(samples_in_cat)]

        selected_idx_label = st.selectbox(
            "Select Image Instance:",
            options=range(len(samples_in_cat)),
            format_func=lambda i: sample_labels[i],
            key=f"{key}_preset_img_idx",
        )

        chosen_path = Path(samples_in_cat[selected_idx_label])
        try:
            image = Image.open(chosen_path).convert("RGB")
            filename = chosen_path.name
            source = "preset"
        except Exception as e:
            st.error(f"Failed to load sample image: {e}")

        if image is not None:
            c_preview, c_meta = st.columns([1, 1], gap="medium")
            with c_preview:
                st.image(image, width="stretch")
            with c_meta:
                st.markdown(
                    f"""
                    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; font-size: 0.88rem;">
                        <div style="color: #64748b; font-size: 0.75rem; text-transform: uppercase; font-weight: 700;">Ground Truth Class</div>
                        <div style="font-weight: 700; color: #0f172a; margin-bottom: 8px; font-size: 1rem;">
                            {CLASS_DISPLAY_NAMES.get(selected_cat, selected_cat)}
                        </div>
                        <div style="color: #64748b; font-size: 0.75rem; text-transform: uppercase; font-weight: 700;">File Path</div>
                        <div style="font-family: monospace; color: #334155; font-size: 0.8rem; word-break: break-all;">
                            data/paddy_images/test/{selected_cat}/{filename}
                        </div>
                        <div style="margin-top: 8px; color: #16a34a; font-weight: 600; font-size: 0.8rem;">
                            ✓ Verified Test-Split Imagery
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    elif source_tab == "📤 Upload Custom Photo":
        uploaded_file = st.file_uploader(
            "Drag & drop leaf photo (JPG / PNG)",
            type=ALLOWED_EXTENSIONS,
            key=f"{key}_file_uploader",
            help="High-contrast, well-lit crop leaf photos yield the sharpest Grad-CAM saliency.",
        )

        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                filename = uploaded_file.name
                source = "upload"
                c_prev, c_txt = st.columns([1, 1])
                with c_prev:
                    st.image(image, caption=f"Uploaded: {filename}", width="stretch")
                with c_txt:
                    st.success(f"✓ Image loaded: {image.width}×{image.height} px")
                    st.caption("Ready for MobileNetV2 feature embedding extraction.")
            except Exception as err:
                st.error(f"Error reading image: {err}")
        else:
            st.info("👆 Drag & drop a crop photo above or browse files from your device.")

    else:
        if not all_images:
            st.warning("No test images found in dataset.")
            return None, "", "none"

        if f"{key}_rand_idx" not in st.session_state:
            st.session_state[f"{key}_rand_idx"] = random.randint(0, len(all_images) - 1)

        c_roll, c_pool = st.columns([1, 1])
        with c_roll:
            if st.button("🎲 Shuffle Test Sample", key=f"{key}_shuffle_btn", width="stretch"):
                st.session_state[f"{key}_rand_idx"] = random.randint(0, len(all_images) - 1)
                st.rerun()
        with c_pool:
            st.caption(f"Randomly indexing across {len(all_images)} verified held-out images.")

        cur_idx = st.session_state[f"{key}_rand_idx"] % len(all_images)
        chosen_path = Path(all_images[cur_idx])
        filename = chosen_path.name
        source = "test-random"

        try:
            image = Image.open(chosen_path).convert("RGB")
            st.image(image, caption=f"Sample: {chosen_path.parent.name}/{filename}", width="stretch")
        except Exception as e:
            st.error(f"Could not load image: {e}")

    return image, filename, source
