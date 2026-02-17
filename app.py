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
        min-height: 400px;
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
    st.toast("New light paths discovered! 🔦")

def callback_restore(meta):
    # Using the key-based approach to prevent StreamlitAPIException
    st.session_state['render_mode_radio'] = meta["Mode"]
    st.session_state['seed_val'] = meta["Seed"]
    st.session_state['exposure'] = meta["Exp"]
    st.session_state['complexity'] = meta["Complexity"]
    st.session_state['aberration'] = meta["Aberr"]

def reset_app():
    st.session_state.history = []
    st.rerun()

# --- LIGHT ENGINE ---

def generate_photographic_path(complexity, seed, prog, bundle_offset=0):
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 1, 1500)
    x, y = np.zeros_like(t), np.zeros_like(t)
    
    # Jaime Gorospe's paths are low-frequency 'Gestures'
    # We use a base path + tiny harmonic variations
    for i in range(1, complexity + 1):
        # We slow down the frequencies significantly (0.1 - 1.0 range)
        freq_x = rng.uniform(0.1, 0.8) * i
        freq_y = rng.uniform(0.1, 0.8) * i
        amp = rng.uniform(0.5, 1.2) / (i**0.8)
        phase = rng.uniform(0, 2*np.pi) + (prog * 2 * np.pi)
        
        # Add slight bundle variation
        x += amp * np.cos(freq_x * t * 2 * np.pi + phase + bundle_offset)
        y += amp * np.sin(freq_y * t * 1.5 * np.pi + phase * 0.5 + bundle_offset)
    return x, y

def render_light_frame(complexity, seed, exposure, glow, aberration, line_width, prog):
    w, h = 1080, 1080
    canvas = np.zeros((h, w, 3), dtype=np.float32)
    
    num_strands = 15 
    
    for s_idx in range(num_strands):
        # Create a bundle offset so lines follow the same 'movement'
        b_offset = (s_idx / num_strands) * 0.05
        x, y = generate_photographic_path(complexity, seed, prog, b_offset)
        
        # Color shifting (Prismatic edges)
        colors = [(0.9, 0.1, 0.1), (0.1, 0.9, 0.2), (0.1, 0.3, 0.9)] 
        offsets = [aberration * -1.8, 0, aberration * 1.8]
        
        for color, offset in zip(colors, offsets):
            pts = np.stack([x * 400 + (w/2) + offset, y * 400 + (h/2)], axis=1).astype(np.int32)
            
            # Draw the soft glow first (Wide, low opacity)
            cv2.polylines(canvas, [pts], False, np.array(color) * exposure * 0.1, 
                          thickness=int(line_width * 8), lineType=cv2.LINE_AA)
            
            # Draw the bright core (Thin, high intensity)
            cv2.polylines(canvas, [pts], False, np.array(color) * exposure, 
                          thickness=int(line_width), lineType=cv2.LINE_AA)

    # Final Bloom Pass
    if glow > 0:
        canvas += gaussian_filter(canvas, sigma=glow * 6) * 0.4
    
    return np.clip(canvas * 255, 0, 255).astype(np.uint8)

# --- UI ---

st.title("🔦 Lights of Time Generator")

with st.sidebar:
    st.header("Studio Controls")
    mode = st.radio("Output Type", ["Still Light", "Animated Loop"], key="render_mode_radio")
    
    # Setting complexity low (2-5) creates those nice sweeping arcs
    complexity = st.slider("Gesture Complexity", 1, 6, key="complexity", help="1-2 for sweeping arcs, 5+ for complex knots.")
    seed = st.number_input("Seed", step=1, key="seed_val")
    
    with st.expander("Optics Styling", expanded=True):
        st.button("🎲 Surprise Me!", on_click=callback_randomize, use_container_width=True)
        st.divider()
        exposure = st.slider("Exposure", 0.5, 5.0, key="exposure")
        aberration = st.slider("Prism Aberration", 0.0, 15.0, key="aberration")
        glow = st.slider("Bloom / Glow", 0.0, 3.0, value=1.0)
        line_width = st.slider("Core Weight", 0.5, 4.0, value=1.2)

    st.divider()
    gen_btn = st.button("EXECUTE LIGHT RENDER", type="primary", use_container_width=True)
    st.button("Clear Studio", on_click=reset_app, use_container_width=True)

preview_placeholder = st.empty()

if gen_btn:
    with preview_placeholder.container():
        st.markdown('<div class="hero-container">', unsafe_allow_html=True)
        if mode == "Still Light":
            img = render_light_frame(complexity, seed, exposure, glow, aberration, line_width, 0)
            st.image(img, use_container_width=True)
            # Encode for download
            is_success, buffer = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            data, fmt = buffer.tobytes(), "png"
        else:
            frames = []
            bar = st.progress(0, text="Capturing Light...")
            for i in range(60):
                f = render_light_frame(complexity, seed, exposure, glow, aberration, line_width, i/60)
                frames.append(f)
                bar.progress((i+1)/60)
            b = io.BytesIO()
            imageio.mimsave(b, frames, format='GIF', fps=30, loop=0)
            data, fmt = b.getvalue(), "gif"
            st.image(data, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        meta = {'Complexity': complexity, 'Seed': seed, 'Exp': exposure, 'Aberr': aberration, 'Mode': mode}
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
            st.markdown(f"""<div class="metadata-card"><b>{m['Mode']}</b> • Seed: {m['Seed']}<br>Complexity: {m['Complexity']} | Exp: {m['Exp']}</div>""", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1: st.download_button("💾", item['data'], f"light_{idx}.{item['fmt']}", key=f"dl_{idx}")
            with c2: st.button("🔄", key=f"res_{idx}", on_click=callback_restore, args=(m,))
            with c3: st.button("🗑️", key=f"del_{idx}", on_click=lambda i=idx: (st.session_state.history.pop(i), st.rerun()))
