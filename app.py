"""
ASCII Art Studio — a Streamlit app for converting images and videos to ASCII.

Run locally:   streamlit run app.py
Deploy:        see README.md (Streamlit Community Cloud / Hugging Face Spaces).
"""

from __future__ import annotations

import os
import tempfile
import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from ascii_engine import DEFAULT_CHARSET, adjust_tonemap, ascii_to_text, image_to_ascii
from fx import apply_glow, apply_noise, apply_scanlines
from renderer import AsciiRenderer
from themes import THEMES
from utils import (
    MAX_IMAGE_MB,
    MAX_VIDEO_MB,
    hex_to_rgb,
    pil_to_bytes,
    read_image_bytes,
    rgb_to_hex,
)
from video_processor import process_video


# =============================================================================
# Page config + global CSS
# =============================================================================
st.set_page_config(
    page_title="ASCII Art Studio",
    page_icon="▓",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
/* Background gradient */
.stApp {
    background:
        radial-gradient(ellipse at top left, rgba(102,255,136,0.05), transparent 45%),
        radial-gradient(ellipse at bottom right, rgba(255,102,204,0.05), transparent 45%),
        #07070c;
}
section[data-testid="stSidebar"] {
    background-color: #0a0a12;
    border-right: 1px solid #181824;
}
h1, h2, h3 { letter-spacing: -0.01em; }

/* Buttons */
.stDownloadButton button, .stButton button {
    border: 1px solid #2a2a3a;
    background-color: #10101a;
    color: #f0f0f0;
    transition: all .15s ease;
}
.stDownloadButton button:hover, .stButton button:hover {
    border-color: #66ff88;
    color: #66ff88;
    box-shadow: 0 0 0 1px rgba(102,255,136,0.25);
}

/* Hero header */
.hero { padding: 1.25rem 0 0.25rem; }
.hero .title {
    font-size: 2.3rem; font-weight: 800; line-height: 1.05;
    background: linear-gradient(90deg, #66ff88 0%, #3ddbff 50%, #ff66cc 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero .subtitle { color: #8a8a98; font-size: 0.95rem; margin-top: 6px; }
.badge {
    display: inline-block; padding: 2px 10px; margin: 6px 6px 0 0;
    border: 1px solid #2a2a3a; border-radius: 999px;
    font-size: 0.72rem; color: #a0a0b0;
}

/* Tabs */
button[data-baseweb="tab"] { font-weight: 600; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
      <div class="title">▓▒░ ASCII Art Studio ░▒▓</div>
      <div class="subtitle">Turn images and video into luminous ASCII — monochrome, true-color, or retro-themed.</div>
      <div>
        <span class="badge">Vectorized NumPy</span>
        <span class="badge">True Color</span>
        <span class="badge">CRT FX</span>
        <span class="badge">MP4 Export</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# Sidebar controls
# =============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Conversion")
    output_width = st.slider(
        "Output width (characters)", 60, 320, 160, 10,
        help="Wider = more detail, slower to render.",
    )
    aspect_corr = st.slider(
        "Aspect-ratio correction", 0.30, 0.90, 0.50, 0.05,
        help="Compensates for tall monospace glyphs. 0.5 is the classic default — "
             "lower values squash vertically, higher values stretch.",
    )

    st.markdown("### 🎚️ Tuning")
    brightness = st.slider("Brightness", -0.50, 0.50, 0.00, 0.05)
    contrast = st.slider("Contrast", 0.50, 2.00, 1.10, 0.05)
    gamma = st.slider("Gamma", 0.40, 2.50, 1.00, 0.05)

    st.markdown("### 🎨 Theme")
    theme_name = st.selectbox("Color engine", list(THEMES.keys()), index=0)
    theme = THEMES[theme_name]
    st.caption(theme.get("description", ""))

    if theme["mode"] == "mono":
        with st.expander("Custom palette"):
            bg_hex = st.color_picker("Background", rgb_to_hex(theme["bg"]))
            fg_hex = st.color_picker("Foreground", rgb_to_hex(theme["fg"]))
            bg_color = hex_to_rgb(bg_hex)
            fg_color = hex_to_rgb(fg_hex)
    else:
        bg_hex = st.color_picker("Backdrop", "#000000")
        bg_color = hex_to_rgb(bg_hex)
        fg_color = None

    invert_lum = st.checkbox(
        "Invert luminance mapping",
        value=theme.get("invert_luminance", False),
        help="On dark themes leave this off. For dark-on-light output, turn it on.",
    )

    st.markdown("### 🔤 Character ramp")
    charset = st.text_area(
        "Dense → sparse",
        value=DEFAULT_CHARSET,
        height=80,
        help="Order glyphs from densest (lots of ink) to sparsest (space). "
             "The first character maps to the brightest pixel on dark themes.",
    )
    if len(charset) < 2:
        st.warning("Charset too short; using the default.")
        charset = DEFAULT_CHARSET

    st.markdown("### 🌀 Visual FX")
    fx_scanlines = st.slider("CRT scanlines", 0.00, 0.60, 0.00, 0.05)
    fx_noise = st.slider("Grain / noise", 0.00, 0.20, 0.00, 0.01)
    fx_glow = st.slider("Phosphor glow", 0.00, 0.80, 0.00, 0.05)

    st.markdown("### 🔡 Render")
    font_size = st.slider("Font size (px)", 8, 22, 12, 1)

    st.markdown("---")
    st.caption(
        "Tip: Matrix Green + CRT scanlines + Glow = instant terminal vibe. "
        "True Color + Noise = lo-fi Dreamcore."
    )


# =============================================================================
# Cached resources
# =============================================================================
@st.cache_resource(max_entries=8, show_spinner=False)
def get_renderer(charset_str: str, font_size_: int) -> AsciiRenderer:
    """Renderer is expensive to construct — cache by (charset, font_size)."""
    return AsciiRenderer(charset=charset_str, font_size=font_size_)


def convert_frame(rgb: np.ndarray, renderer: AsciiRenderer) -> tuple[np.ndarray, np.ndarray]:
    """Full pipeline: tonemap → ascii → render → FX. Returns (RGB array, char_grid)."""
    adjusted = adjust_tonemap(rgb, brightness, contrast, gamma)
    char_grid, color_grid = image_to_ascii(
        adjusted,
        output_width=output_width,
        charset=charset,
        aspect_ratio_correction=aspect_corr,
        invert=invert_lum,
    )
    img = renderer.render(
        char_grid,
        color_grid if theme["mode"] == "color" else None,
        bg_color,
        fg_color,
        mode=theme["mode"],
    )
    arr = np.array(img)
    if fx_glow > 0:
        arr = apply_glow(arr, strength=fx_glow)
    if fx_scanlines > 0:
        arr = apply_scanlines(arr, intensity=fx_scanlines)
    if fx_noise > 0:
        arr = apply_noise(arr, amount=fx_noise)
    return arr, char_grid


# =============================================================================
# Main stage: tabs
# =============================================================================
tab_img, tab_vid, tab_help = st.tabs(["🖼️  Image", "🎬  Video", "📖  About"])

# -----------------------------------------------------------------------------
# IMAGE TAB
# -----------------------------------------------------------------------------
with tab_img:
    col_src, col_out = st.columns([0.4, 0.6], gap="large")

    with col_src:
        st.markdown("#### Source")
        img_file = st.file_uploader(
            f"JPG / PNG / WEBP (max {MAX_IMAGE_MB} MB)",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            accept_multiple_files=False,
            label_visibility="collapsed",
            key="image_uploader",
        )

        use_sample = st.checkbox("Use sample image", value=(img_file is None))

        source_rgb: np.ndarray | None = None
        if img_file is not None:
            raw = img_file.read()
            if len(raw) > MAX_IMAGE_MB * 1024 * 1024:
                st.error(f"Image too large (>{MAX_IMAGE_MB} MB).")
            else:
                try:
                    source_rgb = read_image_bytes(raw)
                except ValueError as e:
                    st.error(str(e))
        elif use_sample:
            # Synthetic gradient + rings so users see color + luminance behaviour
            yy, xx = np.meshgrid(np.arange(360), np.arange(640), indexing="ij")
            r = ((xx * 255) // 640).astype(np.uint8)
            g = ((yy * 255) // 360).astype(np.uint8)
            b = (180 - ((xx + yy) % 180)).astype(np.uint8)
            source_rgb = np.dstack([r, g, b])
            st.caption("Showing a generated sample. Upload an image for real results.")

        if source_rgb is not None:
            st.image(source_rgb, caption="Source preview", use_container_width=True)

    with col_out:
        st.markdown("#### ASCII output")
        if source_rgb is None:
            st.info("Upload an image or tick **Use sample image** to see the live preview.")
        else:
            renderer = get_renderer(charset, font_size)
            with st.spinner("Rendering…"):
                out_arr, char_grid = convert_frame(source_rgb, renderer)
            st.image(out_arr, use_container_width=True)

            st.markdown("#### 📥 Export")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.download_button(
                    "📝  .txt",
                    data=ascii_to_text(char_grid),
                    file_name="ascii_art.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            with c2:
                png_bytes = pil_to_bytes(Image.fromarray(out_arr), fmt="PNG")
                st.download_button(
                    "🖼️  .png",
                    data=png_bytes,
                    file_name="ascii_art.png",
                    mime="image/png",
                    use_container_width=True,
                )
            with c3:
                if st.button("🔍  Render 2× .png", use_container_width=True):
                    with st.spinner("Rendering at 2× size…"):
                        hi_renderer = get_renderer(charset, font_size * 2)
                        hi_arr, _ = convert_frame(source_rgb, hi_renderer)
                        hi_bytes = pil_to_bytes(Image.fromarray(hi_arr), fmt="PNG")
                    st.download_button(
                        "⬇️  Download 2×",
                        data=hi_bytes,
                        file_name="ascii_art_2x.png",
                        mime="image/png",
                        use_container_width=True,
                    )

# -----------------------------------------------------------------------------
# VIDEO TAB
# -----------------------------------------------------------------------------
with tab_vid:
    st.markdown("#### Source")
    st.caption(
        f"MP4 / MOV / WEBM / MKV / AVI, max {MAX_VIDEO_MB} MB. "
        "Short clips (under 15s) render fastest."
    )
    vid_file = st.file_uploader(
        "Upload video",
        type=["mp4", "mov", "webm", "mkv", "avi"],
        accept_multiple_files=False,
        label_visibility="collapsed",
        key="video_uploader",
    )

    max_seconds = st.slider(
        "Max duration to process (s)", 2, 60, 10, 1,
        help="Caps how many frames are rendered regardless of source length.",
    )

    c_a, c_b = st.columns([0.4, 0.6])
    process_btn = c_a.button(
        "▶  Process and Preview",
        type="primary",
        use_container_width=True,
        disabled=vid_file is None,
    )
    c_b.caption(
        "Output resolution is set by **Output width** and **Font size** in the sidebar."
    )

    if process_btn and vid_file is not None:
        raw = vid_file.read()
        if len(raw) > MAX_VIDEO_MB * 1024 * 1024:
            st.error(f"Video too large. Please keep it under {MAX_VIDEO_MB} MB.")
            st.stop()

        suffix = os.path.splitext(vid_file.name)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as in_tmp:
            in_tmp.write(raw)
            input_path = in_tmp.name
        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

        # Count how many frames we'll actually touch
        probe = cv2.VideoCapture(input_path)
        src_fps = probe.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        probe.release()
        max_frames = (
            min(total, int(max_seconds * src_fps)) if total else int(max_seconds * src_fps)
        )

        progress = st.progress(0.0, text="Warming up…")
        status = st.empty()
        t0 = time.time()
        renderer = get_renderer(charset, font_size)

        def _fn(rgb: np.ndarray) -> np.ndarray:
            arr, _ = convert_frame(rgb, renderer)
            return arr

        def _on_progress(frac: float) -> None:
            progress.progress(
                frac, text=f"Rendering ASCII frames… {frac * 100:.0f}%"
            )

        try:
            stats = process_video(
                input_path=input_path,
                output_path=out_path,
                frame_fn=_fn,
                progress_cb=_on_progress,
                max_frames=max_frames,
            )
        except Exception as e:
            st.error(f"Video processing failed: {e}")
            st.stop()
        finally:
            try:
                os.unlink(input_path)
            except OSError:
                pass

        dt = time.time() - t0
        progress.progress(1.0, text=f"Done in {dt:.1f}s")
        status.success(
            f"Rendered **{stats['frames']}** frames @ {stats['fps']:.1f} fps → "
            f"**{stats['size'][0]}×{stats['size'][1]}** px."
        )

        with open(out_path, "rb") as f:
            video_bytes = f.read()
        st.video(video_bytes)
        st.download_button(
            "📥  Download .mp4",
            data=video_bytes,
            file_name="ascii_art.mp4",
            mime="video/mp4",
            use_container_width=True,
        )

# -----------------------------------------------------------------------------
# ABOUT TAB
# -----------------------------------------------------------------------------
with tab_help:
    st.markdown(
        """
### How it works
Every pixel's luminance is mapped to a character from a customizable ramp
of glyphs ordered by visual density. Contrast, gamma, and brightness are
applied first; then the image is downscaled (with an aspect-ratio
correction factor so output isn't vertically stretched); finally the
character grid is rendered by compositing pre-rasterized glyph masks.

All steps use vectorized NumPy. The renderer's glyph lookup is done via
a uint32 reinterpret of the '<U1' character grid, so there are no Python
loops over pixels — which is why video works at interactive speeds.

### Color engines
- **Monochrome** — pick any background and foreground pair (Matrix, Amber,
  Synthwave are presets; the palette is editable).
- **True Color** — each character inherits the matching source pixel's RGB.
  Shines on photography, illustrations, and anything with vivid palettes.

### Visual FX
Stack CRT scanlines, Gaussian grain, and phosphor glow for lo-fi / Dreamcore
aesthetics. They work independently and play well with any theme.

### Tips
- For **detail**, increase Output Width and decrease Font Size.
- For **print**, use the **2× .png** button — it re-renders at double size.
- For **video**, keep width ≤ 180 and clips under 15s for the snappiest
  processing on free hosts.
- If the output looks vertically stretched, nudge **aspect-ratio correction**
  down. If it looks squashed, nudge it up.

### Limits
- Images: {img_mb} MB, longest side auto-scaled to 4096 px
- Video:  {vid_mb} MB, user-selectable duration cap

Built with Streamlit, OpenCV, NumPy, and Pillow.
""".format(img_mb=MAX_IMAGE_MB, vid_mb=MAX_VIDEO_MB)
    )
