import streamlit as st
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import imageio
import io
import zipfile
import time
import random

# PAGE CONFIG
st.set_page_config(page_title="Lights of Time Generator", page_icon="🔦", layout="wide")

# ADVANCED UI STYLING (DARK STUDIO)
st.markdown("""
    <style>
    .stApp { background-color: #050505 !important; color: #e0e0e0 !important; }
    [data-testid="stSidebar"] { background-color: #0f0f0f !important; border-right: 1px solid #222; }
    .metadata-card { 
        background-color: rgba(255, 255, 255, 0.03); 
        padding: 12px; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.08);
        font-size: 0.8rem; color: #aaa; margin-bottom: 10px;
    }
    .hero-container {
        border-radius: 15px; overflow: hidden; border: 1px solid #333;
        margin-top: 1rem; margin-bottom: 2rem; background-color: #000;
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        min-height: 400px;
    }
    h1, h2, h3 { color: #fff !important; font-weight: 700 !important; }
    div.stButton > button { background-color: #007bff !important; color: white !important; border-radius: 6px !important; }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION STATE ---
if 'history' not in st.session_state: st.session_state.history = []
if 'rendering' not in st.session_state: st.session_state.rendering = False

# Widget Keys and Defaults
keys_defaults = {
    'render_mode': "Still Light", 'seed': 42, 'complexity': 4,
    'exposure': 1.8, 'glow': 1.0, 'aberration': 0.5, 'line_width': 2.5
}
for k, v in keys_defaults.items():
    if f"lt_{k}" not in st.session_state: st.session_state[f"lt_{k}"] = v

# --- CALLBACKS ---
def callback_randomize():
    st.session_state["lt_seed"] = random.randint(1, 999999)
    st.session_state["lt_exposure"] = round(random.uniform(1.2, 3.5), 1)
    st.session_state["lt_glow"] = round(random.uniform(0.6, 1.4), 2)
    st.session_state["lt_aberration"] = round(random.uniform(0.0, 2.5), 2)
    st.toast("The lights have shifted! 🔦")

def callback_restore(meta):
    for k in keys_defaults.keys():
        if k in meta: st.session_state[f"lt_{k}"] = meta[k]

def reset_app():
    st.session_state.history = []
    st.rerun()

# --- LIGHT ENGINE ---


def generate_light_path(t, complexity, seed, prog):
    np.random.seed(seed)
    x, y = np.zeros_like(t), np.zeros_like(t)
    angle = prog * 2 * np.pi
    for i in range(1, complexity + 1):
        # The 'Competition' logic: overlapping sine waves
        amp = np.random.uniform(0.5, 1.5) / (i**0.7)
        phase = np.random.uniform(0, 2*np.pi)
        # Nudging the path based on animation progress
        x += amp * np.cos(i * t + phase + (np.cos(angle) * (i/complexity)))
        y += amp * np.sin(i * t + phase + (np.sin(angle) * (i/complexity)))
    return x, y

def render_frame(complexity, seed, exposure, glow, aberration, line_width, prog):
    w, h = 1080, 1080
    canvas = np.zeros((h, w, 3), dtype=np.float32)
    # Higher resolution linspace for smoother lines
    t = np.linspace(0, 2 * np.pi, 3000)
    
    # Chromatic Aberration Logic: Split RGB paths
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)] 
    offsets = [aberration * -2.5, 0, aberration * 2.5]
    
    for color, offset in zip(colors, offsets):
        x, y = generate_light_path(t, complexity, seed, prog)
        # Scale to canvas size
        pts = np.stack([x * 350 + (w/2) + offset, y * 350 + (h/2)], axis=1).astype(np.int32)
        # Additive blending: lines get brighter where they cross
        cv2.polylines(canvas, [pts], False, np.array(color) * exposure, thickness=int(line_width), lineType=cv2.LINE_AA)
    
    # Glow Layer (Stacked Gaussian Blur)
    if glow > 0:
        glow_layer = gaussian_filter(canvas, sigma=glow * 8)
        canvas = canvas + (glow_layer * 0.6)
    
    return np.clip(canvas * 255, 0, 255).astype(np.uint8)

# --- MAIN PAGE ---
st.title("🔦 Lights of Time Generator")

with st.expander("📖 Studio Quick Start"):
    st.markdown("""
    - **Path Complexity:** Controls how many 'competitor' waves battle for the line's shape.
    - **Exposure:** The core brightness of the light filaments.
    - **Chromatic Aberration:** Simulates lens distortion by splitting the RGB colors.
    - **Neon Glow:** Creates the atmospheric light falloff.
    """)

preview_placeholder = st.empty()

# SIDEBAR
with st.sidebar:
    st.header("Studio Controls")
    mode = st.radio("Output Type", ["Still Light", "Animated Loop"], key="lt_render_mode")
    
    complexity = st.slider("Path Complexity", 2, 12, key="lt_complexity", help="Higher values create more tangled, intricate light paths.")
    seed = st.number_input("Seed", step=1, key="lt_seed", help="The unique identifier for this specific light path.")
    
    with st.expander("Optics Styling", expanded=True):
        st.button("🎲 Surprise Me!", on_click=callback_randomize, use_container_width=True)
        st.divider()
        exposure = st.slider("Exposure", 0.5, 5.0, key="lt_exposure")
        glow = st.slider("Neon Glow", 0.0, 3.0, key="lt_glow")
        aberration = st.slider("Chromatic Aberration", 0.0, 10.0, key="lt_aberration")
        line_width = st.slider("Line Weight", 1.0, 15.0, key="lt_line_width")

    st.divider()
    gen_btn = st.button("EXECUTE LIGHT RENDER", type="primary", use_container_width=True)
    st.button("Clear Studio", on_click=reset_app, use_container_width=True)

# --- RENDER EXECUTION ---
if gen_btn:
    with preview_placeholder.container():
        st.markdown('<div class="hero-container">', unsafe_allow_html=True)
        if mode == "Still Light":
            with st.spinner("Exposing Film..."):
                img = render_frame(complexity, seed, exposure, glow, aberration, line_width, 0)
                st.image(img, use_container_width=True)
                data, fmt = img, "png"
        else:
            frames = []
            bar = st.progress(0, text="Capturing Light Trails...")
            for i in range(60):
                f = render_frame(complexity, seed, exposure, glow, aberration, line_width, i/60)
                frames.append(f)
                bar.progress((i+1)/60)
            
            b = io.BytesIO()
            imageio.mimsave(b, frames, format='GIF', fps=30, loop=0)
            data, fmt = b.getvalue(), "gif"
            st.image(data, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        meta = {'complexity': complexity, 'seed': seed, 'exposure': exposure, 'glow': glow, 'aberration': aberration, 'line_width': line_width, 'Mode': mode}
        st.session_state.history.insert(0, {"data": data, "fmt": fmt, "time": time.strftime("%H:%M:%S"), "meta": meta})
        st.rerun()

# HERO PREVIEW (STATIC)
elif st.session_state.history:
    latest = st.session_state.history[0]
    with preview_placeholder.container():
        st.markdown('<div class="hero-container">', unsafe_allow_html=True)
        st.image(latest['data'], use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# GALLERY & ZIP EXPORT
if st.session_state.history:
    st.divider()
    g_col1, g_col2 = st.columns([3, 1])
    with g_col1: st.subheader("Light Gallery")
    with g_col2:
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            for i, item in enumerate(st.session_state.history):
                zf.writestr(f"light_{i}.{item['fmt']}", item['data'] if isinstance(item['data'], bytes) else imageio.imwrite("<bytes>", item['data'], format='PNG'))
        st.download_button("📦 DOWNLOAD ZIP", data=zip_buf.getvalue(), file_name="lights_of_time.zip", use_container_width=True)

    cols = st.columns(3)
    for idx, item in enumerate(st.session_state.history):
        with cols[idx % 3]:
            st.image(item['data'], use_container_width=True)
            m = item['meta']
            st.markdown(f"""<div class="metadata-card">Seed: {m['seed']} | Exp: {m['exposure']} | Aberr: {m['aberration']}</div>""", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1: st.download_button("💾", item['data'], f"light_{idx}.{item['fmt']}", key=f"dl_{idx}")
            with c2: st.button("🔄", key=f"res_{idx}", on_click=callback_restore, args=(m,))
            with c3: st.button("🗑️", key=f"del_{idx}", on_click=lambda i=idx: (st.session_state.history.pop(i), st.rerun()))