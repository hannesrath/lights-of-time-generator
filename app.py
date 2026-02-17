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
    .stApp { background-color: #000000 !important; color: #e0e0e0 !important; }
    [data-testid="stSidebar"] { background-color: #0a0a0a !important; border-right: 1px solid #222; }
    div.stButton > button { background-color: #222 !important; color: #fff !important; border: 1px solid #444 !important; }
    </style>
    """, unsafe_allow_html=True)

if 'history' not in st.session_state: st.session_state.history = []

# --- MATH ENGINE ---

def get_master_spine(complexity, seed, prog):
    """Generates the main path that the bundle follows."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 1, 3000)
    x, y = np.zeros_like(t), np.zeros_like(t)
    
    # Jaime Gorospe style: Low frequency sweeping arcs
    for i in range(1, complexity + 1):
        freq = rng.uniform(0.1, 0.4) * i
        amp = rng.uniform(0.8, 1.5) / (i**0.6)
        phase = rng.uniform(0, 2*np.pi) + (prog * 2 * np.pi)
        
        x += amp * np.cos(freq * t * 2 * np.pi + phase)
        y += amp * np.sin(freq * t * 1.5 * np.pi + phase * 0.5)
        
    return x, y, t

def render_organic_bundle(complexity, seed, exposure, glow, aberration, bundle_spread, blur, prog):
    w, h = 1080, 1080
    canvas = np.zeros((h, w, 3), dtype=np.float32)
    
    # 1. GET MASTER PATH
    mx, my, t_vals = get_master_spine(complexity, seed, prog)
    
    # 2. DEFINE THE FIBER PALETTE (The "Gorospe" Mix)
    # We create a list of "Fiber Definitions" that will be instanced many times
    # (Color_BGR, Thickness, Probability)
    fiber_types = [
        ((5, 120, 255), 2, 0.4),    # Amber/Orange (Dominant)
        ((255, 255, 255), 1, 0.3),  # Pure White (Highlights)
        ((255, 200, 0), 1, 0.2),    # Cyan/Blue (Accent)
        ((50, 50, 200), 1, 0.1)     # Deep Red/Purple (Depth)
    ]
    
    # Generate 50-80 individual strands
    rng = np.random.RandomState(seed)
    total_strands = 60
    
    for i in range(total_strands):
        # Pick a fiber type based on probability
        r = rng.rand()
        cumulative = 0
        chosen_color = fiber_types[0][0]
        chosen_thick = fiber_types[0][1]
        
        for f_color, f_thick, f_prob in fiber_types:
            cumulative += f_prob
            if r <= cumulative:
                chosen_color = f_color
                chosen_thick = f_thick
                break
        
        # 3. CALCULATE ORGANIC OFFSET (The "Breathing" Effect)
        # Each strand has its own unique sine-wave distance from the center
        # This creates the "tighten and loosen" effect seen in the reference
        offset_freq = rng.uniform(1.0, 5.0)
        offset_phase = rng.uniform(0, 2*np.pi)
        max_dist = rng.uniform(0.01, 0.08) * bundle_spread
        
        # Calculate perpendicular offset direction
        # Simple approximation: just offset x/y with distinct phases
        dx = np.cos(t_vals * offset_freq + offset_phase) * max_dist
        dy = np.sin(t_vals * offset_freq + offset_phase) * max_dist
        
        # Final Strand Path
        sx = mx + dx
        sy = my + dy
        
        # Scale to canvas
        px = sx * 380 + (w/2)
        py = sy * 380 + (h/2)
        pts = np.stack([px, py], axis=1).astype(np.int32)
        
        # 4. DRAW STRAND
        # Random intensity variation per strand
        intensity = rng.uniform(0.5, 1.2) * exposure
        
        # Base color
        base_color = np.array(chosen_color) * intensity
        
        # Draw Core
        cv2.polylines(canvas, [pts], False, base_color, 
                      thickness=chosen_thick, lineType=cv2.LINE_AA)
        
        # Draw Glow (Wider, fainter line)
        if glow > 0:
            cv2.polylines(canvas, [pts], False, base_color * 0.15, 
                          thickness=chosen_thick * 6, lineType=cv2.LINE_AA)

    # 5. POST-PROCESSING
    if aberration > 0:
        shift = int(aberration)
        if shift > 0:
            canvas_copy = canvas.copy()
            canvas[:, :-shift, 0] = canvas_copy[:, shift:, 0] # Shift Blue
            canvas[:, shift:, 2] = canvas_copy[:, :-shift, 2] # Shift Red

    if blur > 0:
        canvas = gaussian_filter(canvas, sigma=blur)
        
    return np.clip(canvas, 0, 255).astype(np.uint8)

# --- UI ---
st.title("🔦 Lights of Time: Organic Bundle")

with st.sidebar:
    st.header("Generator Settings")
    mode = st.radio("Mode", ["Still Light", "Animated Loop"], key="mode")
    complexity = st.slider("Path Complexity", 1, 4, value=2, key="comp")
    seed = st.number_input("Seed", step=1, value=42, key="seed")
    
    st.divider()
    
    exposure = st.slider("Exposure", 0.5, 3.0, value=1.5)
    bundle_spread = st.slider("Bundle Spread", 0.5, 5.0, value=2.0, help="How loose/tight the fibers are.")
    aberration = st.slider("Prism Shift", 0.0, 10.0, value=3.0)
    glow = st.slider("Glow Intensity", 0.0, 2.0, value=1.0)
    blur = st.slider("Lens Blur", 0.0, 3.0, value=0.5)
    
    gen_btn = st.button("EXPOSE FILM", type="primary", use_container_width=True)

preview = st.empty()

if gen_btn:
    with preview.container():
        bar = st.progress(0, text="Simulating Fibers...")
        if mode == "Still Light":
            img_bgr = render_organic_bundle(complexity, seed, exposure, glow, aberration, bundle_spread, blur, 0)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, use_container_width=True)
            is_success, buffer = cv2.imencode(".png", img_bgr)
            data = buffer.tobytes()
        else:
            frames = []
            for i in range(60):
                f_bgr = render_organic_bundle(complexity, seed, exposure, glow, aberration, bundle_spread, blur, i/60)
                frames.append(cv2.cvtColor(f_bgr, cv2.COLOR_BGR2RGB))
                bar.progress((i+1)/60)
            b = io.BytesIO()
            imageio.mimsave(b, frames, format='GIF', fps=30, loop=0)
            data = b.getvalue()
            st.image(data, use_container_width=True)
            
        st.session_state.history.insert(0, data)

if st.session_state.history:
    st.divider()
    cols = st.columns(4)
    for idx, item in enumerate(st.session_state.history):
        with cols[idx % 4]:
            st.image(item, use_container_width=True) 
            st.download_button("💾", item, f"light_{idx}.png", key=f"dl_{idx}")
