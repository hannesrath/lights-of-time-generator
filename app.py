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

def get_centered_spine(complexity, seed, prog, scale_factor):
    """Generates a centered, scaled master path."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 1, 3000)
    x, y = np.zeros_like(t), np.zeros_like(t)
    
    # Generate the abstract shape
    for i in range(1, complexity + 1):
        freq = rng.uniform(0.1, 0.4) * i 
        amp = rng.uniform(0.5, 1.2) / (i**0.5)
        phase = rng.uniform(0, 2*np.pi) + (prog * 2 * np.pi)
        
        x += amp * np.cos(freq * t * 2 * np.pi + phase)
        y += amp * np.sin(freq * t * 1.5 * np.pi + phase * 0.5)
        
    # CENTERING LOGIC
    # We normalize the path to be roughly -1 to 1, then scale to canvas
    x = (x - np.mean(x)) * scale_factor
    y = (y - np.mean(y)) * scale_factor
    
    return x, y, t

def render_multistroke_frame(params, prog):
    w, h = 1080, 1080
    accumulation_buffer = np.zeros((h, w, 3), dtype=np.float32)
    
    rng = np.random.RandomState(params['seed'])
    
    # --- COLOR PALETTE LOGIC ---
    # Define color sets (Normalized 0-1 BGR)
    palettes = {
        "Gorospe Classic": [
            (0.05, 0.6, 1.0),   # Golden Amber
            (0.9, 0.95, 1.0),   # Cool White
            (1.0, 0.4, 0.0)     # Deep Blue
        ],
        "Neon Cyber": [
            (0.5, 0.0, 1.0),    # Magenta
            (1.0, 1.0, 0.0),    # Cyan
            (0.0, 1.0, 0.0)     # Lime
        ],
        "Warm Tungsten": [
            (0.1, 0.4, 1.0),    # Orange
            (0.2, 0.2, 1.0),    # Red
            (0.8, 0.9, 1.0)     # White
        ],
        "Deep Ocean": [
            (1.0, 0.5, 0.0),    # Deep Blue
            (0.8, 0.8, 0.0),    # Teal
            (0.8, 0.2, 0.4)     # Violet
        ]
    }
    
    current_palette = palettes.get(params['palette_name'], palettes["Gorospe Classic"])
    
    # --- MULTI-STROKE LOOP ---
    # We draw 'num_strokes' separate bundles to create the multi-source look
    for s_idx in range(params['num_strokes']):
        
        # Each stroke gets a unique sub-seed
        stroke_seed = params['seed'] + (s_idx * 500)
        
        # 1. Generate Spine for this specific stroke
        # We vary the complexity slightly per stroke for organic feel
        mx, my, t_vals = get_centered_spine(
            params['complexity'], 
            stroke_seed, 
            prog, 
            params['scale'] * rng.uniform(0.8, 1.2)
        )
        
        # 2. Assign a "Dominant" color for this stroke
        # In real light painting, one tool might be mostly Amber, another mostly Blue
        dom_color_idx = rng.randint(0, len(current_palette))
        dominant_color = current_palette[dom_color_idx]
        
        # 3. DRAW THE BUNDLE FOR THIS STROKE
        # Each stroke has 40-60 fibers
        stroke_strands = 50
        
        for i in range(stroke_strands):
            # Fiber color mixing: 70% chance to use stroke's dominant color, 30% chance for white/accent
            if rng.rand() > 0.3:
                base_c = dominant_color
            else:
                # Pick a random accent or white
                base_c = current_palette[rng.randint(0, len(current_palette))]
                if rng.rand() > 0.5: base_c = (0.95, 0.95, 0.95) # Add white highlights
            
            # Fiber Offset (Breathing)
            offset_phase = rng.uniform(0, 2*np.pi)
            offset_freq = rng.uniform(1.0, 3.0)
            dist = np.sin(t_vals * offset_freq + offset_phase) * params['spread'] * 0.08
            
            # Path Calculation
            sx = mx + np.cos(t_vals * 5.0 + offset_phase) * dist
            sy = my + np.sin(t_vals * 5.0 + offset_phase) * dist
            
            # Position on Canvas (Centered)
            # Add random translation to the whole stroke so they aren't all perfectly on top of each other
            stroke_offset_x = rng.uniform(-0.2, 0.2) * 500
            stroke_offset_y = rng.uniform(-0.2, 0.2) * 500
            
            px = sx * 450 + (w/2) + stroke_offset_x
            py = sy * 450 + (h/2) + stroke_offset_y
            
            # Bounds check to prevent drawing way off canvas (optimization)
            # (Optional, but good for performance)
            
            pts = np.stack([px, py], axis=1).astype(np.int32)
            
            # Draw
            temp_layer = np.zeros((h, w, 3), dtype=np.float32)
            intensity = params['exposure'] * rng.uniform(0.5, 1.0)
            
            # Thinner lines look more like fiber optics
            thick = 1 if rng.rand() > 0.2 else 3
            
            color = np.array(base_c) * intensity
            cv2.polylines(temp_layer, [pts], False, color, thickness=thick, lineType=cv2.LINE_AA)
            
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
        
    final_image = np.clip(accumulation_buffer, 0, 1.0) * 255
    return final_image.astype(np.uint8)

# --- UI ---
st.title("🔦 Lights of Time: Multi-Stroke Studio")

with st.sidebar:
    st.header("Composition")
    mode = st.radio("Mode", ["Still Light", "Animated Loop"], key="mode")
    palette_name = st.selectbox("Color Palette", ["Gorospe Classic", "Neon Cyber", "Warm Tungsten", "Deep Ocean"], key="pal")
    
    col1, col2 = st.columns(2)
    with col1:
        num_strokes = st.number_input("Light Sources", 1, 10, 3, help="Number of separate light paths.")
    with col2:
        scale = st.slider("Zoom / Fill", 0.5, 2.0, 1.2, help="Scale of the light paths.")
        
    st.divider()
    
    st.header("Physics")
    complexity = st.slider("Curve Complexity", 1, 4, value=2)
    spread = st.slider("Fiber Spread", 0.5, 4.0, value=2.0)
    exposure = st.slider("Exposure", 0.05, 0.8, value=0.15)
    
    with st.expander("Lens Optics"):
        aberration = st.slider("Prism Shift", 0.0, 10.0, value=4.0)
        glow = st.slider("Glow Amount", 0.0, 3.0, value=1.0)
        blur = st.slider("Softness", 0.0, 3.0, value=0.6)
    
    seed = st.number_input("Seed", step=1, value=42)
    gen_btn = st.button("EXPOSE FILM", type="primary", use_container_width=True)

preview = st.empty()

if gen_btn:
    params = {
        "seed": seed, "complexity": complexity, "exposure": exposure,
        "glow": glow, "aberration": aberration, "spread": spread,
        "blur": blur, "num_strokes": num_strokes, "scale": scale,
        "palette_name": palette_name
    }
    
    with preview.container():
        bar = st.progress(0, text="Integrating Light Paths...")
        if mode == "Still Light":
            img_bgr = render_multistroke_frame(params, 0)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, use_container_width=True)
            is_success, buffer = cv2.imencode(".png", img_bgr)
            data = buffer.tobytes()
        else:
            frames = []
            for i in range(60):
                f_bgr = render_multistroke_frame(params, i/60)
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
