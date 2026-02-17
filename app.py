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

# UI STYLING
st.markdown("""
    <style>
    .stApp { background-color: #050505 !important; color: #e0e0e0 !important; }
    [data-testid="stSidebar"] { background-color: #0f0f0f !important; border-right: 1px solid #222; }
    .metadata-card { 
        background-color: rgba(255, 255, 255, 0.03); 
        padding: 12px; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.08);
        font-size: 0.8rem; color: #aaa; margin-top: 8px; margin-bottom: 12px;
    }
    .hero-container {
        border-radius: 15px; overflow: hidden; border: 1px solid #333;
        margin-top: 1rem; margin-bottom: 2rem; background-color: #000;
        display: flex; flex-direction: column; align-items: center; justify-content: center;
    }
    h1, h2, h3 { color: #fff !important; font-weight: 700 !important; }
    div.stButton > button { background-color: #238636 !important; color: white !important; border-radius: 6px !important; }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION STATE & CALLBACKS ---
if 'history' not in st.session_state: st.session_state.history = []

def callback_randomize():
    st.session_state['seed_val'] = random.randint(1, 999999)
    st.session_state['exposure'] = round(random.uniform(1.5, 3.5), 1)
    st.session_state['blur_slider'] = round(random.uniform(0.0, 2.0), 1)
    st.toast("Optics adjusted! 🔦")

def callback_restore(meta):
    st.session_state['render_mode_radio'] = meta["Mode"]
    st.session_state['seed_val'] = meta["Seed"]
    st.session_state['exposure'] = meta["Exp"]
    st.session_state['complexity'] = meta["Complexity"]
    st.session_state['aberration'] = meta["Aberr"]
    st.session_state['strand_count'] = meta.get("Strands", 15)
    st.session_state['blur_slider'] = meta.get("Blur", 0.5)

def reset_app():
    st.session_state.history = []
    st.rerun()

# --- LIGHT ENGINE ---

def generate_photographic_path(complexity, seed, prog, bundle_offset=0):
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 1, 1500)
    x, y = np.zeros_like(t), np.zeros_like(t)
    for i in range(1, complexity + 1):
        freq_x = rng.uniform(0.1, 0.7) * i
        freq_y = rng.uniform(0.1, 0.7) * i
        amp = rng.uniform(0.6, 1.3) / (i**0.9)
        phase = rng.uniform(0, 2*np.pi) + (prog * 2 * np.pi)
        x += amp * np.cos(freq_x * t * 2 * np.pi + phase + bundle_offset)
        y += amp * np.sin(freq_y * t * 1.5 * np.pi + (phase * 0.5) + bundle_offset)
    return x, y

def render_light_frame(complexity, strands, seed, exposure, glow, aberration, line_width, blur_val, prog):
    w, h = 1080, 1080
    canvas = np.zeros((h, w, 3), dtype=np.float32)
    
    for s_idx in range(strands):
        b_offset = (s_idx / strands) * 0.08
        x, y = generate_photographic_path(complexity, seed, prog, b_offset)
        
        colors = [(0.9, 0.1, 0.1), (0.1, 0.9, 0.2), (0.1, 0.3, 0.9)] 
        offsets = [aberration * -1.8, 0, aberration * 1.8]
        
        for color, offset in zip(colors, offsets):
            pts = np.stack([x * 400 + (w/2) + offset, y * 400 + (h/2)], axis=1).astype(np.int32)
            # Soft atmospheric layer
            cv2.polylines(canvas, [pts], False, np.array(color) * exposure * 0.08, 
                          thickness=int(line_width * 10), lineType=cv2.LINE_AA)
            # Brilliant core
            cv2.polylines(canvas, [pts], False, np.array(color) * exposure, 
                          thickness=int(line_width), lineType=cv2.LINE_AA)

    # 1. Global Bloom (Atmospheric falloff)
    if glow > 0:
        canvas += gaussian_filter(canvas, sigma=glow * 5) * 0.4
    
    # 2. Global Blur (Lens softness/Depth of Field)
    if blur_val > 0:
        canvas = gaussian_filter(canvas, sigma=blur_val)
    
    return np.clip(canvas * 255, 0, 255).astype(np.uint8)

# --- UI ---

st.title("🔦 Lights of Time Generator")

with st.expander("📖 Studio Quick Start Guide", expanded=False):
    st.markdown("""
    - **Complexity:** 1-2 for sweeping Jaime Gorospe arcs.
    - **Strands:** Increases the density of the light bundle.
    - **Blur:** Softens the edges of the lines for a lens-realistic feel.
    - **Prism Aberration:** Creates the iconic rainbow edges.
    """)

preview_placeholder = st.empty()

# SIDEBAR
with st.sidebar:
    st.header("Studio Controls")
    mode = st.radio("Output Type", ["Still Light", "Animated Loop"], key="render_mode_radio", 
                    help="Still generates a PNG; Animated creates a seamless loop GIF.")
    
    complexity = st.slider("Gesture Complexity", 1, 6, key="complexity", 
                           help="Lower values create sweeping arcs. Higher values create knots.")
    
    strand_count = st.slider("Strand Count", 1, 40, value=15, key="strand_count",
                             help="Number of individual light filaments in the bundle.")
    
    seed = st.number_input("Seed", step=1, key="seed_val", help="Unique ID for the gesture.")
    
    with st.expander("Optics Styling", expanded=True):
        st.button("🎲 Surprise Me!", on_click=callback_randomize, use_container_width=True, 
                  help="Randomize Seeds and visual aesthetics.")
        st.divider()
        exposure = st.slider("Exposure", 0.5, 5.0, key="exposure", help="Brightness of the light core.")
        aberration = st.slider("Prism Aberration", 0.0, 15.0, key="aberration", help="Splits RGB paths.")
        glow = st.slider("Bloom / Glow", 0.0, 3.0, value=1.0, help="Soft atmospheric falloff.")
        blur_val = st.slider("Lens Blur", 0.0, 5.0, value=0.5, key="blur_slider", 
                             help="Softens the final output to mimic real lens focus.")
        line_width = st.slider("Core Weight", 0.5, 4.0, value=1.2, help="Thickness of filaments.")

    st.divider()
    gen_btn = st.button("EXECUTE LIGHT RENDER", type="primary", use_container_width=True, help="Start render.")
    st.button("Clear Studio", on_click=reset_app, use_container_width=True, help="Clear history.")

# --- RENDER ENGINE ---
if gen_btn:
    with preview_placeholder.container():
        bar = st.progress(0, text="Developing Exposure...")
        if mode == "Still Light":
            img = render_light_frame(complexity, strand_count, seed, exposure, glow, aberration, line_width, blur_val, 0)
            is_success, buffer = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            data, fmt = buffer.tobytes(), "png"
        else:
            frames = []
            for i in range(60):
                f = render_light_frame(complexity, strand_count, seed, exposure, glow, aberration, line_width, blur_val, i/60)
                frames.append(f)
                bar.progress((i+1)/60)
            b = io.BytesIO()
            imageio.mimsave(b, frames, format='GIF', fps=30, loop=0)
            data, fmt = b.getvalue(), "gif"
        
        meta = {'Complexity': complexity, 'Strands': strand_count, 'Seed': seed, 'Exp': exposure, 'Aberr': aberration, 'Blur': blur_val, 'Mode': mode}
        st.session_state.history.insert(0, {"data": data, "fmt": fmt, "time": time.strftime("%H:%M:%S"), "meta": meta})
        st.rerun()

elif st.session_state.history:
    latest = st.session_state.history[0]
    with preview_placeholder.container():
        st.markdown('<div class="hero-container">', unsafe_allow_html=True)
        st.image(latest['data'], use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# GALLERY
if st.session_state.history:
    st.divider()
    st.subheader("Light Gallery")
    cols = st.columns(3)
    for idx, item in enumerate(st.session_state.history):
        with cols[idx % 3]:
            st.image(item['data'], use_container_width=True)
            m = item['meta']
            st.markdown(f"""<div class="metadata-card"><b>{m['Mode']}</b><br>Seed: {m['Seed']} | Blur: {m.get('Blur', 0.5)}<br>Comp: {m['Complexity']} | Aberr: {m['Aberr']}</div>""", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1: st.download_button("💾", item['data'], f"light_{idx}.{item['fmt']}", key=f"dl_{idx}")
            with c2: st.button("🔄", key=f"res_{idx}", on_click=callback_restore, args=(m,), help="Restore settings.")
            with c3: st.button("🗑️", key=f"del_{idx}", on_click=lambda i=idx: (st.session_state.history.pop(i), st.rerun()))
