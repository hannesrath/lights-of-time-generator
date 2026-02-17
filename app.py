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

def get_centered_spine(complexity, seed, scale_factor):
    """Generates a static, centered master path (0 to 1)."""
    rng = np.random.RandomState(seed)
    # We generate fixed points so the shape doesn't wiggle during draw-on
    t = np.linspace(0, 1, 4000) 
    x, y = np.zeros_like(t), np.zeros_like(t)
    
    for i in range(1, complexity + 1):
        freq = rng.uniform(0.1, 0.4) * i 
        amp = rng.uniform(0.5, 1.2) / (i**0.5)
        phase = rng.uniform(0, 2*np.pi)
        
        x += amp * np.cos(freq * t * 2 * np.pi + phase)
        y += amp * np.sin(freq * t * 1.5 * np.pi + phase * 0.5)
        
    # Center and Scale
    x = (x - np.mean(x)) * scale_factor
    y = (y - np.mean(y)) * scale_factor
    
    return x, y, t

def render_write_on_frame(params, prog):
    w, h = 1080, 1080
    accumulation_buffer = np.zeros((h, w, 3), dtype=np.float32)
    rng = np.random.RandomState(params['seed'])
    
    # PALETTES
    palettes = {
        "Gorospe Classic": [(0.05, 0.6, 1.0), (0.9, 0.95, 1.0), (1.0, 0.4, 0.0)],
        "Neon Cyber": [(0.5, 0.0, 1.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)],
        "Warm Tungsten": [(0.1, 0.4, 1.0), (0.2, 0.2, 1.0), (0.8, 0.9, 1.0)],
        "Deep Ocean": [(1.0, 0.5, 0.0), (0.8, 0.8, 0.0), (0.8, 0.2, 0.4)]
    }
    current_palette = palettes.get(params['palette_name'], palettes["Gorospe Classic"])
    
    # LOOP LOGIC (Fade out at the very end to loop smoothly)
    # If prog is > 0.95, we fade the whole image to black
    global_alpha = 1.0
    if params['mode'] == "Animated Loop":
        if prog > 0.9:
            global_alpha = 1.0 - ((prog - 0.9) * 10.0) # Fade out in last 10%
    
    # Draw limits based on progress
    # In 'Still' mode, we draw 100%. In 'Animated', we draw up to 'prog'.
    draw_limit = 1.0 if params['mode'] == "Still Light" else prog
    
    # Avoid drawing nothing at frame 0
    if draw_limit < 0.001: draw_limit = 0.001

    # --- MULTI-STROKE LOOP ---
    for s_idx in range(params['num_strokes']):
        stroke_seed = params['seed'] + (s_idx * 500)
        
        # Get the FULL path first
        full_mx, full_my, t_vals = get_centered_spine(params['complexity'], stroke_seed, params['scale'] * rng.uniform(0.8, 1.2))
        
        # SLICE THE PATH based on progress
        # We only take points where t <= draw_limit
        mask = t_vals <= draw_limit
        if np.sum(mask) < 2: continue # Skip if not enough points yet
        
        mx = full_mx[mask]
        my = full_my[mask]
        current_t = t_vals[mask]
        
        # Dominant Color for this stroke
        dominant_color = current_palette[rng.randint(0, len(current_palette))]
        
        # Position Offset (Center the composition)
        stroke_offset_x = rng.uniform(-0.15, 0.15) * 500
        stroke_offset_y = rng.uniform(-0.15, 0.15) * 500
        
        # --- FIBER BUNDLE ---
        stroke_strands = 40
        for i in range(stroke_strands):
            # Fiber Color
            base_c = dominant_color if rng.rand() > 0.3 else current_palette[rng.randint(0, len(current_palette))]
            if rng.rand() > 0.6: base_c = (0.98, 0.98, 0.98) # White highlights
            
            # Fiber Offset (Breathing)
            # We calculate this for the *current slice* of t
            offset_phase = rng.uniform(0, 2*np.pi)
            offset_freq = rng.uniform(1.0, 3.0)
            
            dist = np.sin(current_t * offset_freq + offset_phase) * params['spread'] * 0.08
            
            # Calculate path for this fiber
            sx = mx + np.cos(current_t * 5.0 + offset_phase) * dist
            sy = my + np.sin(current_t * 5.0 + offset_phase) * dist
            
            px = sx * 450 + (w/2) + stroke_offset_x
            py = sy * 450 + (h/2) + stroke_offset_y
            
            pts = np.stack([px, py], axis=1).astype(np.int32)
            
            # DRAWING
            temp_layer = np.zeros((h, w, 3), dtype=np.float32)
            intensity = params['exposure'] * rng.uniform(0.6, 1.2)
            thick = 1 if rng.rand() > 0.3 else 2
            
            # 1. DRAW THE TRAIL
            color = np.array(base_c) * intensity
            cv2.polylines(temp_layer, [pts], False, color, thickness=thick, lineType=cv2.LINE_AA)
            
            # 2. DRAW THE "TIP" (The Light Source)
            # Only in animation mode: make the end of the line brighter
            if params['mode'] == "Animated Loop" and len(pts) > 1:
                # Get the last few points (the "head")
                head_pts = pts[-15:] 
                # Draw white-hot tip
                cv2.polylines(temp_layer, [head_pts], False, (2.0, 2.0, 2.0), thickness=thick+2, lineType=cv2.LINE_AA)
                
            accumulation_buffer += temp_layer

    # --- POST PROCESSING ---
    if params['glow'] > 0:
        bloom = gaussian_filter(accumulation_buffer, sigma=params['glow'] * 10)
        accumulation_buffer += bloom * 0.4

    if params['aberration'] > 0:
        shift = int(params['aberration'])
        if shift > 0:
            accumulation_buffer[:, :-shift, 0] = accumulation_buffer[:, shift:, 0]
            accumulation_buffer[:, shift:, 2] = accumulation_buffer[:, :-shift, 2]

    if params['blur'] > 0:
        accumulation_buffer = gaussian_filter(accumulation_buffer, sigma=params['blur'])
        
    # Apply Global Alpha (Fade out at end of loop)
    accumulation_buffer *= global_alpha
        
    final_image = np.clip(accumulation_buffer, 0, 1.0) * 255
    return final_image.astype(np.uint8)

# --- UI ---
st.title("🔦 Lights of Time: Multi-Stroke Studio")

with st.sidebar:
    st.header("Composition")
    mode = st.radio("Mode", ["Still Light", "Animated Loop"], key="mode", help="Animated Loop creates a 'Write-On' effect.")
    palette_name = st.selectbox("Color Palette", ["Gorospe Classic", "Neon Cyber", "Warm Tungsten", "Deep Ocean"], key="pal")
    
    col1, col2 = st.columns(2)
    with col1:
        num_strokes = st.number_input("Light Sources", 1, 10, 3, help="Number of separate light paths.")
    with col2:
        scale = st.slider("Zoom", 0.5, 2.0, 1.1, help="Scale of the light paths.")
        
    st.divider()
    
    st.header("Physics")
    complexity = st.slider("Curve Complexity", 1, 4, value=2)
    spread = st.slider("Fiber Spread", 0.5, 4.0, value=1.5)
    exposure = st.slider("Exposure", 0.05, 0.8, value=0.25)
    
    with st.expander("Lens Optics"):
        aberration = st.slider("Prism Shift", 0.0, 10.0, value=3.0)
        glow = st.slider("Glow Amount", 0.0, 3.0, value=1.0)
        blur = st.slider("Softness", 0.0, 3.0, value=0.6)
    
    seed = st.number_input("Seed", step=1, value=101)
    gen_btn = st.button("EXPOSE FILM", type="primary", use_container_width=True)

preview = st.empty()

if gen_btn:
    params = {
        "seed": seed, "complexity": complexity, "exposure": exposure,
        "glow": glow, "aberration": aberration, "spread": spread,
        "blur": blur, "num_strokes": num_strokes, "scale": scale,
        "palette_name": palette_name, "mode": mode
    }
    
    with preview.container():
        bar = st.progress(0, text="Integrating Light Paths...")
        if mode == "Still Light":
            img_bgr = render_write_on_frame(params, 1.0)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, use_container_width=True)
            is_success, buffer = cv2.imencode(".png", img_bgr)
            data = buffer.tobytes()
        else:
            frames = []
            # Generate 60 frames for the write-on
            for i in range(60):
                prog = i / 60.0
                f_bgr = render_write_on_frame(params, prog)
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
