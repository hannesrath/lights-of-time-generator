import streamlit as st
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import imageio
import io
import time
import random

# PAGE CONFIG
st.set_page_config(page_title="Lights of Time Generator", page_icon="🔦", layout="wide")

# UI STYLING
st.markdown("""
    <style>
    .stApp { background-color: #020202 !important; color: #e0e0e0 !important; }
    [data-testid="stSidebar"] { background-color: #080808 !important; border-right: 1px solid #1a1a1a; }
    .hero-container {
        border-radius: 12px; overflow: hidden; border: 1px solid #222;
        margin-top: 1rem; margin-bottom: 2rem; background-color: #000;
        display: flex; align-items: center; justify-content: center;
    }
    div.stButton > button { background-color: #1a1a1a !important; color: #fff !important; border: 1px solid #333 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION STATE ---
if 'history' not in st.session_state: st.session_state.history = []

# --- CORE MATH ---
def get_physical_path(complexity, seed, prog, jitter_amp=0.0):
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

def render_pro_frame(complexity, seed, exposure, glow, aberration, weight_mult, blur, prog):
    w, h = 1080, 1080
    canvas = np.zeros((h, w, 3), dtype=np.float32)
    
    # Define distinct "Light Objects" per reference analysis
    # Format: (Color_RGB, Base_Thickness, Jitter, Intensity, Count)
    light_objects = [
        (np.array([1.0, 0.95, 0.8]), 1.5, 0.002, 1.5, 1),   # Warm White Filament
        (np.array([1.0, 0.5, 0.1]), 10.0, 0.005, 0.3, 1),   # Amber Glow Ribbon
        (np.array([0.1, 0.4, 1.0]), 1.2, 0.015, 0.4, 6),    # Blue Fiber Bundle
        (np.array([0.8, 0.9, 1.0]), 0.8, 0.0, 0.2, 2)       # Ghostly Veils
    ]

    for color, thickness, jitter, intensity, count in light_objects:
        for c_idx in range(count):
            obj_seed = seed + int(color[0]*1000) + c_idx
            x, y = get_physical_path(complexity, obj_seed, prog, jitter)
            
            # Prismatic Offset Logic
            offsets = [aberration * -1.0, 0, aberration * 1.0]
            channel_colors = [color * np.array([1,0,0]), color * np.array([0,1,0]), color * np.array([0,0,1])]
            
            for c_color, offset in zip(channel_colors, offsets):
                pts = np.stack([x * 380 + (w/2) + offset, y * 380 + (h/2)], axis=1).astype(np.int32)
                # Primary Light Core
                cv2.polylines(canvas, [pts], False, (c_color * intensity * exposure), 
                              thickness=int(thickness * weight_mult), lineType=cv2.LINE_AA)
                # Volumetric Light Envelope
                if glow > 0:
                    cv2.polylines(canvas, [pts], False, (c_color * intensity * exposure * 0.1), 
                                  thickness=int(thickness * 6 * weight_mult), lineType=cv2.LINE_AA)

    # Multi-pass Bloom
    if glow > 0:
        canvas += gaussian_filter(canvas, sigma=glow * 3) * 0.5
        canvas += gaussian_filter(canvas, sigma=glow * 10) * 0.3
    if blur > 0:
        canvas = gaussian_filter(canvas, sigma=blur)
        
    return np.clip(canvas * 255, 0, 255).astype(np.uint8)

# --- UI & EXECUTION ---
st.title("🔦 Lights of Time Generator")

with st.sidebar:
    st.header("Camera Settings")
    mode = st.radio("Output Type", ["Still Light", "Animated Loop"], key="mode")
    complexity = st.slider("Gesture Complexity", 1, 4, value=2, key="comp")
    seed = st.number_input("Seed", step=1, value=42, key="seed")
    exposure = st.slider("Luminosity / Exp", 1.0, 10.0, value=5.0)
    aberration = st.slider("Prism Diffraction", 0.0, 20.0, value=5.0)
    glow = st.slider("Atmospheric Bloom", 0.0, 5.0, value=2.0)
    blur_val = st.slider("Lens Blur", 0.0, 5.0, value=0.5)
    line_weight = st.slider("Object Weight", 0.5, 5.0, value=1.0)
    gen_btn = st.button("EXECUTE EXPOSURE", type="primary", use_container_width=True)

preview_area = st.empty()

if gen_btn:
    with preview_area.container():
        bar = st.progress(0, text="Developing Exposure...")
        if mode == "Still Light":
            img = render_pro_frame(complexity, seed, exposure, glow, aberration, line_weight, blur_val, 0)
            st.image(img, use_container_width=True)
            is_success, buffer = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            data = buffer.tobytes()
        else:
            frames = []
            for i in range(60):
                f = render_pro_frame(complexity, seed, exposure, glow, aberration, line_weight, blur_val, i/60)
                frames.append(f)
                bar.progress((i+1)/60)
            b = io.BytesIO()
            imageio.mimsave(b, frames, format='GIF', fps=30, loop=0)
            data = b.getvalue()
            st.image(data, use_container_width=True)
        st.session_state.history.insert(0, data)
