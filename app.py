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

def get_smooth_spine(complexity, seed, prog):
    """Generates a very smooth, low-frequency master path."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 1, 3000)
    x, y = np.zeros_like(t), np.zeros_like(t)
    
    # Very low frequency for that "Elegant Sweep" look
    for i in range(1, complexity + 1):
        freq = rng.uniform(0.1, 0.3) * i 
        amp = rng.uniform(0.8, 1.4) / (i**0.5)
        phase = rng.uniform(0, 2*np.pi) + (prog * 2 * np.pi)
        
        x += amp * np.cos(freq * t * 2 * np.pi + phase)
        y += amp * np.sin(freq * t * 1.5 * np.pi + phase * 0.5)
        
    return x, y, t

def render_luminous_frame(complexity, seed, exposure, glow, aberration, spread, blur, prog):
    w, h = 1080, 1080
    
    # 1. USE FLOAT32 FOR ADDITIVE BLENDING (Crucial for "White Hot" look)
    # We start with a black void.
    accumulation_buffer = np.zeros((h, w, 3), dtype=np.float32)
    
    mx, my, t_vals = get_smooth_spine(complexity, seed, prog)
    
    # 2. THE GOROSPE PALETTE
    # (B, G, R) Normalized 0-1
    palette = [
        {'color': (0.05, 0.6, 1.0), 'weight': 1.0, 'thick': 2},   # Golden Amber (Dominant)
        {'color': (0.9, 0.95, 1.0), 'weight': 0.8, 'thick': 4},   # Cool White (Core)
        {'color': (1.0, 0.8, 0.0), 'weight': 0.6, 'thick': 1},    # Deep Blue (Edge)
        {'color': (0.2, 0.4, 1.0), 'weight': 0.5, 'thick': 1}     # Orange/Red (Accent)
    ]
    
    rng = np.random.RandomState(seed)
    
    # 3. DRAW STRANDS
    total_strands = 60
    
    for i in range(total_strands):
        # Pick color
        p_idx = rng.randint(0, len(palette))
        p = palette[p_idx]
        
        # Smooth Breathing Offset (Not random noise)
        # Strands move in/out slowly like fibers in a muscle
        offset_phase = rng.uniform(0, 2*np.pi)
        offset_freq = rng.uniform(1.0, 3.0)
        
        # Calculate offset magnitude
        # We modulate it by the progress 't' so it creates twisting shapes
        dist = np.sin(t_vals * offset_freq + offset_phase) * spread * 0.08
        
        sx = mx + np.cos(t_vals * 5.0 + offset_phase) * dist
        sy = my + np.sin(t_vals * 5.0 + offset_phase) * dist
        
        # Scale to canvas
        px = sx * 380 + (w/2)
        py = sy * 380 + (h/2)
        pts = np.stack([px, py], axis=1).astype(np.int32)
        
        # 4. ADDITIVE DRAWING
        # Create a temporary layer for this single strand
        strand_layer = np.zeros((h, w, 3), dtype=np.float32)
        
        # Calculate color intensity
        # Exposure * Palette Weight * Random Variation
        intensity = exposure * p['weight'] * rng.uniform(0.8, 1.2)
        color = np.array(p['color']) * intensity
        
        # Draw the line onto the temp layer
        cv2.polylines(strand_layer, [pts], False, color, 
                      thickness=p['thick'], lineType=cv2.LINE_AA)
        
        # ADD to the master buffer (This is where the magic happens)
        # Overlapping lines will naturally sum up towards white (1.0, 1.0, 1.0)
        accumulation_buffer += strand_layer

    # 5. POST-PROCESSING
    
    # Bloom/Glow (Add a blurred version of the buffer to itself)
    if glow > 0:
        bloom = gaussian_filter(accumulation_buffer, sigma=glow * 10)
        accumulation_buffer += bloom * 0.4

    # Chromatic Aberration
    if aberration > 0:
        shift = int(aberration)
        if shift > 0:
            accumulation_buffer[:, :-shift, 0] = accumulation_buffer[:, shift:, 0] # Blue
            accumulation_buffer[:, shift:, 2] = accumulation_buffer[:, :-shift, 2] # Red

    # Lens Blur
    if blur > 0:
        accumulation_buffer = gaussian_filter(accumulation_buffer, sigma=blur)
        
    # 6. TONE MAPPING (HDR -> LDR)
    # We simply clip at 1.0 (White) and convert to 0-255
    final_image = np.clip(accumulation_buffer, 0, 1.0) * 255
    return final_image.astype(np.uint8)

# --- UI ---
st.title("🔦 Lights of Time: Luminous Flow")

with st.sidebar:
    st.header("Generator Settings")
    mode = st.radio("Mode", ["Still Light", "Animated Loop"], key="mode")
    complexity = st.slider("Flow Complexity", 1, 4, value=2, key="comp")
    seed = st.number_input("Seed", step=1, value=99, key="seed")
    
    st.divider()
    
    exposure = st.slider("Light Intensity", 0.05, 0.5, value=0.15, help="Low values work best for additive blending!")
    spread = st.slider("Bundle Tightness", 0.5, 4.0, value=1.5, help="How much the strands separate.")
    aberration = st.slider("Prism Shift", 0.0, 10.0, value=4.0)
    glow = st.slider("Glow Amount", 0.0, 3.0, value=1.0)
    blur = st.slider("Softness", 0.0, 3.0, value=0.6)
    
    gen_btn = st.button("EXPOSE FILM", type="primary", use_container_width=True)

preview = st.empty()

if gen_btn:
    with preview.container():
        bar = st.progress(0, text="Accumulating Light...")
        if mode == "Still Light":
            # Generate
            img_bgr = render_luminous_frame(complexity, seed, exposure, glow, aberration, spread, blur, 0)
            # Display (Convert BGR to RGB)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, use_container_width=True)
            # Save
            is_success, buffer = cv2.imencode(".png", img_bgr)
            data = buffer.tobytes()
        else:
            frames = []
            for i in range(60):
                f_bgr = render_luminous_frame(complexity, seed, exposure, glow, aberration, spread, blur, i/60)
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
