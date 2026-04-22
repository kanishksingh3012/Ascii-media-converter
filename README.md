# ▓▒░ ASCII Art Studio ░▒▓

A production-grade Streamlit app that converts images and videos into high-quality
ASCII art — with monochrome, true-color, and retro terminal themes, plus lo-fi
CRT FX for that Dreamcore aesthetic.

- **Vectorized NumPy pipeline** — even the per-glyph compositing. Video works at
  interactive speeds.
- **Aspect-ratio correction** — output never looks vertically stretched.
- **Pro tuning** — brightness, contrast, gamma, output width, custom charset.
- **Six color engines** — Monochrome Dark/Light, Matrix Green, Amber Terminal,
  Synthwave Pink, and full True Color.
- **Visual FX** — CRT scanlines, grain, phosphor glow.
- **Export suite** — `.txt`, `.png` (1× and 2×), and `.mp4`.

---

## 📁 Project layout

```
ascii_art_studio/
├── app.py                  # Streamlit UI (main stage + sidebar + tabs)
├── ascii_engine.py         # Vectorized image → character-grid converter
├── renderer.py             # Character-grid → PIL image renderer
├── fx.py                   # Scanlines, grain, phosphor glow
├── themes.py               # Color engine presets
├── video_processor.py      # Streaming frame pipeline → MP4
├── utils.py                # Safe I/O helpers
├── requirements.txt        # Python deps
├── packages.txt            # apt deps (DejaVu fonts for PIL)
├── .streamlit/config.toml  # Dark theme + upload size
└── README.md
```

Each module does one thing and is independently testable. The renderer caches
the expensive glyph-rasterization step keyed by `(charset, font_size)`, so
adjusting tonemap sliders never rebuilds it.

---

## 🚀 Run locally

Requires **Python 3.10+**.

```bash
git clone <your-fork-url> ascii_art_studio
cd ascii_art_studio

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Streamlit will open `http://localhost:8501` in your browser.

> **Fonts on Linux**: the renderer looks for DejaVu Sans Mono at standard paths.
> On Debian/Ubuntu: `sudo apt-get install fonts-dejavu-core`. On macOS it falls
> back to Menlo/Monaco. On Windows it uses Consolas / Courier.

---

## ☁️ Deploy to Streamlit Community Cloud

Streamlit Community Cloud is free, and picks up `requirements.txt`, `packages.txt`,
and `.streamlit/config.toml` automatically.

1. **Push this directory to a public GitHub repo** (any name works, e.g.
   `ascii-art-studio`).
2. Go to **<https://share.streamlit.io>** and sign in with GitHub.
3. Click **New app** → pick your repo, branch (`main`), and set the main file to
   `app.py`.
4. Hit **Deploy**. First build takes ~2 min (installs OpenCV + Pillow + DejaVu fonts).
5. Your app is live at `https://<your-username>-<repo>-app-<hash>.streamlit.app`.
   Share that URL.

To update: push to the tracked branch and Streamlit Cloud rebuilds within a minute.

**Tuning knobs** (optional):

- Streamlit Cloud free tier gives each app ~1 GB RAM. If you want to support
  larger videos, lower `MAX_VIDEO_MB` in `utils.py` or reduce the default
  `output_width` in `app.py`.
- You can change the brand palette in `.streamlit/config.toml`.

---

## 🤗 Deploy to Hugging Face Spaces

Hugging Face Spaces supports Streamlit natively and tends to have more generous
resources on the free tier.

1. Sign in at **<https://huggingface.co>** and click **New Space**.
2. Fill in:
   - **Owner** — your account
   - **Space name** — `ascii-art-studio`
   - **License** — your choice (e.g. MIT)
   - **SDK** — **Streamlit**
   - **Hardware** — CPU basic (free) is enough
3. Create the Space. It's initialised as a Git repo.
4. Clone it and add the project files:

   ```bash
   git clone https://huggingface.co/spaces/<user>/ascii-art-studio
   cd ascii-art-studio

   # Copy every file from this project into the repo (including hidden .streamlit/)
   cp -r /path/to/ascii_art_studio/. .

   git add .
   git commit -m "Initial commit"
   git push
   ```

5. The Space rebuilds automatically. You'll see a public URL like
   `https://huggingface.co/spaces/<user>/ascii-art-studio`.

> Spaces reads `requirements.txt` and `packages.txt` the same way Streamlit Cloud
> does — no extra config required. If you prefer, you can also add a short
> `README.md` header with YAML front-matter for Spaces metadata (title, emoji,
> `app_file: app.py`), but it's optional.

---

## 🛠️ Architecture notes

**Why it's fast.** The expensive part of ASCII rendering is the per-glyph draw
call. The renderer rasterizes every glyph once into a `(N, char_h, char_w)` mask
stack, then at render time:

1. The character grid (`'<U1'` dtype, i.e. one Unicode code point per element)
   is reinterpreted as a `uint32` array via `.view()` — zero-copy.
2. A pre-computed LUT maps code points → mask indices with a single fancy index.
3. `self.masks[idx_grid]` gathers `(H, W, ch, cw)` tiles in one NumPy call.
4. A `transpose + reshape` lays them out into the final mask canvas.
5. The final composite is a uint16 multiply-add over two `(H·ch, W·cw, 3)`
   buffers — all vectorized, no Python loops over pixels or chars.

**Video is streaming.** `process_video` never holds all frames in memory — it
reads from `cv2.VideoCapture`, runs each frame through the pipeline, and writes
to `cv2.VideoWriter` immediately. RAM use is bounded by a few frames regardless
of clip length.

**Aspect correction.** The ASCII-grid row count is computed as
`int(src_h * (out_w / src_w) * aspect_ratio_correction)`, where `0.5` is the
sweet spot for typical monospace fonts. Exposing it as a slider lets users tune
for their font of choice.

---

## 🧪 Quick sanity tests

```python
from ascii_engine import image_to_ascii, ascii_to_text
from renderer import AsciiRenderer
import numpy as np

# Random RGB image
img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

chars, colors = image_to_ascii(img, output_width=80)
print(chars.shape, colors.shape)          # (~30, 80), (~30, 80, 3)

r = AsciiRenderer(charset="@%#*+=-:. ", font_size=14)
out = r.render(chars, colors, bg_color=(0, 0, 0), fg_color=(255, 255, 255), mode="mono")
out.save("test.png")

print(ascii_to_text(chars)[:200])
```

---

## 🎨 Credits

- Character ramp adapted from Paul Bourke's classic density ramp.
- Fonts: DejaVu Sans Mono (Bitstream Vera-derived; permissive license).

Feedback and PRs welcome.
