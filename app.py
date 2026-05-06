"""
Edge Detection Studio — Streamlit + OpenCV
Run: streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Edge Detection Studio",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

/* ── Root palette ── */
:root {
    --bg:       #080b0f;
    --surface:  #0f1318;
    --card:     #141920;
    --border:   #1e2730;
    --accent:   #00f0c8;
    --accent2:  #8b5cf6;
    --accent3:  #f97316;
    --text:     #e2e8f0;
    --muted:    #64748b;
    --danger:   #ef4444;
    --glow:     rgba(0, 240, 200, 0.15);
}

/* ── Global resets ── */
html, body, .stApp {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Space Mono', monospace !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div {
    padding-top: 1rem !important;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── Sliders ── */
[data-testid="stSlider"] > div > div > div {
    background: var(--accent) !important;
}
[data-testid="stSlider"] label {
    color: var(--muted) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Radio buttons ── */
[data-testid="stRadio"] label {
    color: var(--text) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
}
[data-testid="stRadio"] > div {
    gap: 0.25rem;
}

/* ── Select boxes ── */
[data-testid="stSelectbox"] label {
    color: var(--muted) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="stSelectbox"] > div > div {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'Space Mono', monospace !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--card) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 6px !important;
}
[data-testid="stFileUploader"] label {
    color: var(--muted) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--card) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    padding: 0.4rem 0.8rem !important;
    transition: all 0.2s ease;
    width: 100%;
}
.stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    box-shadow: 0 0 12px var(--glow) !important;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #00f0c8, #0ea5e9) !important;
    color: #080b0f !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    width: 100%;
    transition: all 0.2s;
}
[data-testid="stDownloadButton"] > button:hover {
    opacity: 0.9 !important;
    box-shadow: 0 0 20px rgba(0,240,200,0.3) !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    padding: 0.75rem !important;
}
[data-testid="stMetric"] label {
    color: var(--muted) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.65rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
[data-testid="stMetricValue"] {
    color: var(--accent) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
}

/* ── Dividers ── */
hr {
    border-color: var(--border) !important;
    margin: 0.75rem 0 !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    background: var(--card) !important;
}
[data-testid="stExpander"] summary {
    color: var(--muted) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Image containers ── */
[data-testid="stImage"] {
    border-radius: 4px;
    overflow: hidden;
}
[data-testid="stImage"] img {
    border-radius: 4px;
}

/* ── Columns gap ── */
[data-testid="column"] {
    padding: 0 0.4rem !important;
}

/* ── Info / warning / error boxes ── */
[data-testid="stAlert"] {
    background: var(--card) !important;
    border-left-color: var(--accent) !important;
    color: var(--text) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER BANNER
# ─────────────────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(135deg, #0f1318 0%, #0a1628 50%, #0f1318 100%);
    border-bottom: 1px solid #1e2730;
    padding: 1rem 2rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    position: relative;
    overflow: hidden;
">
  <div style="
    position:absolute; inset:0;
    background: radial-gradient(ellipse at 20% 50%, rgba(0,240,200,0.06) 0%, transparent 60%),
                radial-gradient(ellipse at 80% 50%, rgba(139,92,246,0.06) 0%, transparent 60%);
    pointer-events:none;
  "></div>
  <div style="
    font-family:'Syne',sans-serif;
    font-size:1.6rem;
    font-weight:800;
    background: linear-gradient(90deg, #00f0c8, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing:-0.02em;
  ">⬡ EDGE DETECTION STUDIO</div>
  <div style="
    font-family:'Space Mono',monospace;
    font-size:0.72rem;
    color:#64748b;
    border: 1px solid #1e2730;
    border-radius:3px;
    padding: 2px 8px;
    margin-left:auto;
  ">Python · OpenCV 4 · Streamlit</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# EDGE ENGINE
# ─────────────────────────────────────────────
def load_image(uploaded) -> np.ndarray | None:
    data = np.frombuffer(uploaded.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    if w > 1200:
        scale = 1200 / w
        img = cv2.resize(img, (1200, int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def process_edges(bgr: np.ndarray, method: str, params: dict):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    if method == "Canny":
        blur = params.get("blur_ksize", 1)
        if blur > 0:
            ksize = blur * 2 + 1
            gray = cv2.GaussianBlur(gray, (ksize, ksize), 0)
        edges = cv2.Canny(gray,
                          params.get("low_thresh", 50),
                          params.get("high_thresh", 150),
                          apertureSize=params.get("aperture", 3))

    elif method == "Sobel":
        ksize = params.get("ksize", 3)
        scale = params.get("scale", 1.0)
        dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize, scale=scale)
        dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize, scale=scale)
        mag = np.sqrt(dx**2 + dy**2)
        mag = np.clip(mag, 0, 255).astype(np.uint8)
        _, edges = cv2.threshold(mag, params.get("thresh", 50), 255, cv2.THRESH_BINARY)

    elif method == "Laplacian":
        ksize = params.get("ksize", 3)
        scale = params.get("scale", 1.0)
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize, scale=scale)
        lap = np.clip(np.abs(lap), 0, 255).astype(np.uint8)
        _, edges = cv2.threshold(lap, params.get("thresh", 20), 255, cv2.THRESH_BINARY)
    else:
        edges = np.zeros_like(gray)

    return edges


COLOR_MAP = {
    "Cyan":    (0,   255, 215),
    "Violet":  (139, 92,  246),
    "Orange":  (249, 115, 22),
    "Red":     (239, 68,  68),
    "White":   (255, 255, 255),
    "Yellow":  (253, 224, 71),
    "Green":   (34,  197, 94),
    "Hot Pink":(236, 72,  153),
}


def build_overlay(bgr: np.ndarray, edges: np.ndarray, color_name: str) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    overlay = rgb.copy()
    mask = edges > 0
    overlay[mask] = COLOR_MAP.get(color_name, (0, 255, 215))
    return overlay


def edge_density(edges: np.ndarray) -> float:
    return np.count_nonzero(edges) / edges.size * 100


def to_png_bytes(img_rgb: np.ndarray) -> bytes:
    pil = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def edges_to_rgb(edges: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

# ─────────────────────────────────────────────
# SIDEBAR — CONTROLS
# ─────────────────────────────────────────────
with st.sidebar:
    # ── Section header helper ──
    def sh(label):
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;margin:1rem 0 0.5rem;">
          <span style="font-family:'Space Mono',monospace;font-size:0.6rem;
                       color:#00f0c8;letter-spacing:0.15em;text-transform:uppercase;">{label}</span>
          <div style="flex:1;height:1px;background:#1e2730;"></div>
        </div>""", unsafe_allow_html=True)

    # ── Studio logo in sidebar ──
    st.markdown("""
    <div style="text-align:center;padding:0.5rem 0 1rem;">
      <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:800;
                  color:#00f0c8;letter-spacing:0.05em;">⬡ STUDIO</div>
    </div>""", unsafe_allow_html=True)

    # ── Upload ──
    sh("Input Image")
    uploaded = st.file_uploader("", type=["jpg","jpeg","png","bmp","tiff","webp"])

    # ── Method ──
    sh("Detection Method")
    method = st.radio("", ["Canny", "Sobel", "Laplacian"], horizontal=True, label_visibility="collapsed")

    # ── Method params ──
    params = {}

    if method == "Canny":
        sh("Canny Parameters")
        params["low_thresh"]  = st.slider("Low threshold",  0, 255, 50, 1)
        params["high_thresh"] = st.slider("High threshold", 0, 255, 150, 1)
        params["blur_ksize"]  = st.slider("Blur radius",    0, 10,  1, 1)
        params["aperture"]    = st.radio("Aperture size", [3, 5, 7], horizontal=True)

    elif method == "Sobel":
        sh("Sobel Parameters")
        params["thresh"] = st.slider("Threshold", 0, 255, 50, 1)
        params["scale"]  = st.slider("Scale", 0.1, 5.0, 1.0, 0.1)
        params["ksize"]  = st.radio("Kernel size", [1, 3, 5, 7], horizontal=True)

    elif method == "Laplacian":
        sh("Laplacian Parameters")
        params["thresh"] = st.slider("Threshold", 0, 255, 20, 1)
        params["scale"]  = st.slider("Scale", 0.1, 5.0, 1.0, 0.1)
        params["ksize"]  = st.radio("Kernel size", [1, 3, 5, 7], horizontal=True)

    # ── Display ──
    sh("Display")
    edge_color   = st.selectbox("Edge color", list(COLOR_MAP.keys()))
    show_overlay = st.checkbox("Overlay on original", value=True)

# ─────────────────────────────────────────────
# MAIN CANVAS
# ─────────────────────────────────────────────
pad = st.container()

with pad:
    st.markdown('<div style="padding:1.2rem 1.5rem 0;">', unsafe_allow_html=True)

    if uploaded is None:
        # ── Empty state ──
        st.markdown("""
        <div style="
            display:flex; flex-direction:column; align-items:center; justify-content:center;
            min-height:55vh; gap:1rem;
            background: linear-gradient(135deg, #0f1318, #0a1628);
            border: 1px dashed #1e2730; border-radius:8px;
            margin-top:1rem;
        ">
          <div style="font-size:3rem; filter:grayscale(0.4);">⬡</div>
          <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:700;
                      color:#1e2730;letter-spacing:-0.02em;">
            Upload an image to begin
          </div>
          <div style="font-family:'Space Mono',monospace;font-size:0.7rem;color:#334155;">
            JPG · PNG · BMP · TIFF · WEBP
          </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        bgr = load_image(uploaded)

        if bgr is None:
            st.error("Could not decode the image. Please try another file.")
        else:
            h, w = bgr.shape[:2]
            rgb_orig = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            # ── Run detection ──
            edges   = process_edges(bgr, method, params)
            overlay = build_overlay(bgr, edges, edge_color)
            density = edge_density(edges)
            display = overlay if show_overlay else edges_to_rgb(edges)

            # ── Stats row ──
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Resolution", f"{w}×{h}")
            with c2:
                st.metric("Edge Density", f"{density:.2f}%")
            with c3:
                st.metric("Method", method)
            with c4:
                st.metric("Edge Color", edge_color)

            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

            # ── Three-panel layout ──
            col_orig, col_edges, col_ov = st.columns(3)

            def panel_header(title, accent_color="#64748b"):
                st.markdown(f"""
                <div style="
                    font-family:'Space Mono',monospace;
                    font-size:0.62rem;
                    color:{accent_color};
                    text-transform:uppercase;
                    letter-spacing:0.15em;
                    border-bottom:1px solid #1e2730;
                    padding-bottom:0.4rem;
                    margin-bottom:0.5rem;
                ">▶ {title}</div>
                """, unsafe_allow_html=True)

            with col_orig:
                panel_header("Original Image", "#64748b")
                st.image(rgb_orig, use_container_width=True)

            with col_edges:
                panel_header("Edge Map", "#8b5cf6")
                st.image(edges_to_rgb(edges), use_container_width=True)

            with col_ov:
                panel_header("Result Overlay", "#00f0c8")
                st.image(display, use_container_width=True)

            st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

            # ── Full-width overlay ──
            st.markdown("""
            <div style="font-family:'Space Mono',monospace;font-size:0.62rem;
                        color:#00f0c8;text-transform:uppercase;letter-spacing:0.15em;
                        border-bottom:1px solid #1e2730;padding-bottom:0.4rem;
                        margin-bottom:0.5rem;">
              ◈ Full Resolution Preview
            </div>""", unsafe_allow_html=True)
            st.image(display, use_container_width=True)

            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

            # ── Export ──
            st.markdown("""
            <div style="font-family:'Space Mono',monospace;font-size:0.62rem;
                        color:#f97316;text-transform:uppercase;letter-spacing:0.15em;
                        border-bottom:1px solid #1e2730;padding-bottom:0.4rem;
                        margin-bottom:0.75rem;">
              ↓ Export
            </div>""", unsafe_allow_html=True)

            exp1, exp2, exp3 = st.columns(3)
            with exp1:
                st.download_button(
                    "⬇ Save Overlay PNG",
                    data=to_png_bytes(overlay),
                    file_name="edge_overlay.png",
                    mime="image/png",
                )
            with exp2:
                st.download_button(
                    "⬇ Save Edges PNG",
                    data=to_png_bytes(edges_to_rgb(edges)),
                    file_name="edge_map.png",
                    mime="image/png",
                )
            with exp3:
                st.download_button(
                    "⬇ Save Original PNG",
                    data=to_png_bytes(rgb_orig),
                    file_name="original.png",
                    mime="image/png",
                )

    st.markdown('</div>', unsafe_allow_html=True)
