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
    .stApp { background-color: #020202 !important; color: #e0e0e0 !important; }
    [data-testid="stSidebar"] { background-color: #080808 !important; border-right: 1px solid #1a1a1a; }
    .metadata-card { 
        background-color: rgba(255, 255, 255, 0.02); 
        padding: 12px; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.05);
        font-size: 0.8rem; color: #888; margin-top: 8px;
    }
    .hero-container {
        border-radius: 12px; overflow: hidden; border: 1px solid #222;
        margin-top: 1rem; margin-bottom: 2rem; background-color: #000;
        display: flex; align-items: center; justify-content: center;
    }
    h1, h2, h3 { color: #fff !important; }
    div.stButton > button { background-color: #1a1a1a !important; color: #fff !important; border: 1px solid #333 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION STATE & CALLBACKS ---
if 'history' not in st.session_state: st.session_state.history = []

def callback_randomize():
    st.session_state['seed_val'] = random.randint(1, 999999)
    st.session_state['exposure'] = round(random.uniform(2.0, 5.0), 1)
    st.toast("New physical light sources configured! 🔦")

def callback_restore(meta):
    st.session_state['render_mode_radio'] = meta["Mode"]
    st.session_state['seed_val'] = meta["Seed"]
    st.session_state['exposure'] = meta["Exp"]
    st.session_state['complexity_slider'] = meta["Complexity"]

def reset_app():
    st.session_state.history = []
    st.rerun()

# --- THE PHYSICAL LIGHT ENGINE ---

def get_physical_path(complexity, seed, prog, jitter_amp=0.0):
    """Generates paths with organic 'hand-drawn' jitter."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 1, 2500)
    x, y = np.zeros_like(t), np.zeros_like(t)
    
    for i in range(1, complexity + 1):
        freq = rng.uniform(0.1, 0.5) * i
        amp = rng.uniform(0.8, 1.5) / (i**0.7)
        phase = rng.uniform(0, 2*np.pi) + (prog * 2 * np.pi)
        x += amp * np.cos(freq * t * 2 * np.pi + phase)
        y += amp * np.sin(freq * t * 1.6 * np.pi + phase * 0.3)
    
    if jitter_amp > 0:
        x += rng.normal(0, jitter_amp, x.shape)
        y += rng.normal(0, jitter_amp, y.shape)
        
    return x, y

def render_gorospe_frame(complexity, seed, exposure, glow, aberration, weight_mult, blur, prog):
    w, h = 1080, 1080
    canvas = np.zeros((h, w, 3), dtype=np.float32)
    rng = np.random.RandomState(seed)
    
    # We define 4 distinct "Light Objects" to draw
    # Object structure: (Color_RGB, Thickness, Jitter, Intensity_Mult, Strand_Count)
    light_objects = [
        (np.array([1.0, 0.95, 0.8]), 2.0 * weight_mult, 0.001, 1.2, 1),   # The Hot Power Filament
        (np.array([1.0, 0.4, 0.1]), 8.0 * weight_mult, 0.005, 0.4, 1),    # The Amber Glow Ribbon
        (np.array([0.1, 0.6, 1.0]), 1.5 * weight_mult, 0.012, 0.3, 8),    # The Blue Fiber Bundle
        (np.array([0.9, 0.9, 1.0]), 0.5 * weight_mult, 0.0, 0.2, 3)      # The Ghost Veils
    ]

    for color, thickness, jitter, intensity, count in light_objects:
        obj_seed = seed + int(color[0]*100)
        
        for c_idx in range(count):
            # Each strand in a bundle gets a tiny unique offset
            strand_seed = obj_seed + c_idx
            x, y = get_physical_path(complexity, strand_seed, prog, jitter)
            
            # Chromatic Aberration shift (only on edges)
            offsets = [aberration * -1.0, 0, aberration * 1.0]
            channel_colors = [color * np.array([1,0,0]), color * np.array([0,1,0]), color * np.array([0,0,1])]
            
            for c_color, offset in zip(channel_colors, offsets):
                pts = np.stack([x * 380 + (w/2) + offset, y * 380 + (h/2)], axis=1).astype(np.int32)
                
                # Draw core
                cv2.polylines(canvas, [pts], False, (c_color * intensity * exposure), 
                              thickness=int(thickness), lineType=cv2.LINE_AA)
                
                # Draw surrounding volumetric glow for this specific object
                if glow > 0:
                    cv2.polylines(canvas, [pts], False, (c_color * intensity * exposure * 0.1), 
                                  thickness=int(thickness * 5), lineType=cv2.LINE_AA)

    # Final Bloom and Lens Softness
    if glow > 0:
        canvas += gaussian_filter(canvas, sigma=glow * 8) * 0.3
    if blur > 0:
        canvas = gaussian_filter(canvas, sigma=blur)
        
    return np.clip(canvas * 255, 0, 255).astype(np.uint8)

# --- UI ---
st.title("🔦 Lights of Time Generator")

with st.expander("📖 Studio Quick Start Guide", expanded=False):
    st.markdown("""
    - **Complexity:** 1-2 for Gorospe-style sweeping gestures.
    - **Filament Weight:** Controls the thickness of the 'light painting' tools.
    - **Surprise Me:** Swaps out the physical light sources.
    """)

preview_area = st.empty()

with st.sidebar:
    st.header("Camera Settings")
    mode = st.radio("Output Type", ["Still Light", "Animated Loop"], key="render_mode_radio")
    
    complexity = st.slider("Gesture Complexity", 1, 4, key="complexity_slider", help="1-2 for clean sweeping arcs.")
    seed = st.number_input("Seed", step=1, key="seed_val")
    
    with st.expander("Optics & Film", expanded=True):
        st.button("🎲 Surprise Me!", on_click=callback_randomize, use_container_width=True)
        st.divider()
        exposure = st.slider("Luminosity / Exp", 1.0, 10.0, key="exposure", help="Sensor sensitivity.")
        aberration = st.slider("Prism Diffraction", 0.0, 20.0, key="aberration", help="Lens edge fringing.")
        glow = st.slider("Atmospheric Bloom", 0.0, 5.0, value=2.0)
        blur_val = st.slider("Lens Blur", 0.0, 5.0, key="blur_slider", value=1.0)
        line_weight = st.slider("Filament Weight", 0.5, 5.0, value=2.0)

    st.divider()
    gen_btn = st.button("EXECUTE EXPOSURE", type="primary", use_container_width=True)
    if st.button("Reset Studio", use_container_width=True): reset_app()

if gen_btn:
    with preview_area.container():
        bar = st.progress(0, text="Capturing Physical Light...")
        if mode == "Still Light":
            img = render_gorospe_frame(complexity, seed, exposure, glow, aberration, line_weight, blur_val, 0)
            is_success, buffer = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            data, fmt = buffer.tobytes(), "png"
        else:
            frames = []
            for i in range(60):
                f = render_gorospe_frame(complexity, seed, exposure, glow, aberration, line_weight, blur_val, i/60)
                frames.append(f)
                bar.progress((i+1)/60)
            b = io.BytesIO()
            imageio.mimsave(b, frames, format='GIF', fps=30, loop=0)
            data, fmt = b.getvalue(), "gif"
        
        meta = {'Complexity': complexity, 'Seed': seed, 'Exp': exposure, 'Mode': mode}
        st.session_state.history.insert(0, {"data": data, "fmt": fmt, "time": time.strftime("%H:%M:%S"), "meta": meta})
        st.rerun()

elif st.session_state.history:
    latest = st.session_state.history[0]
    with preview_area.container():
        st.markdown('<div class="hero-container">', unsafe_allow_html=True)
        st.image(latest['data'], use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# GALLERY
if st.session_state.history:
    st.divider()
    cols = st.columns(3)
    for idx, item in enumerate(st.session_state.history):
        with cols[idx % 3]:
            st.image(item['data'], use_container_width=True)
            m = item['meta']
            st.markdown(f"""<div class="metadata-card"><b>{m['Mode']}</b> • {item['time']}<br>Seed: {m['Seed']} | Exp: {m['Exp']}</div>""", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1: st.download_button("💾", item['data'], f"light_{idx}.{item['fmt']}", key=f"dl_{idx}")
            with c2: st.button("🔄", key=f"res_{idx}", on_click=callback_restore, args=(m,))
            with c3:
                if st.button("🗑️", key=f"del_{idx}"):
                    st.session_state.history.pop(idx)
                    st.rerun()
