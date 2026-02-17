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

# --- MATH HELPERS ---

def get_centered_spine(complexity, seed, scale_factor):
    """Generates the master path center-lines."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 1, 4000) 
    x, y = np.zeros_like(t), np.zeros_like(t)
    
    for i in range(1, complexity + 1):
        freq = rng.uniform(0.1, 0.4) * i 
        amp = rng.uniform(0.5, 1.2) / (i**0.5)
        phase = rng.uniform(0, 2*np.pi)
        x += amp * np.cos(freq * t * 2 * np.pi + phase)
        y += amp * np.sin(freq * t * 1.5 * np.pi + phase * 0.5)
        
    x = (x - np.mean(x)) * scale_factor
    y = (y - np.mean(y)) * scale_factor
    return x, y, t

def get_path_normals(x, y):
    """Calculates perpendicular vectors for ribbon width."""
    dx = np.gradient(x)
    dy = np.gradient(y)
    dist = np.sqrt(dx**2 + dy**2) + 1e-6
    nx = -dy / dist
    ny = dx / dist
    return nx, ny

# --- MAIN RENDERER (HYBRID TOOL ENGINE) ---

def render_hybrid_frame(params, prog):
    w, h = 1080, 1080
    accumulation_buffer = np.zeros((h, w, 3), dtype=np.float32)
    rng = np.random.RandomState(params['seed'])
    
    # Define distinct palettes for different tools
    ribbon_palette = [(0.05, 0.5, 1.0), (0.0, 0.8, 1.0), (0.2, 0.4, 0.8)] # Gold/Amber
    cluster_palette = [(1.0, 1.0, 1.0), (0.9, 0.9, 1.0), (1.0, 0.8, 0.4)] # White/Cool Blue
    
    # Animation Logic
    global_alpha = 1.0
    if params['mode'] == "Animated Loop" and prog > 0.9:
        global_alpha = 1.0 - ((prog - 0.9) * 10.0)
    draw_limit = 1.0 if params['mode'] == "Still Light" else prog
    if draw_limit < 0.002: draw_limit = 0.002

    # --- MULTI-STROKE LOOP ---
    for s_idx in range(params['num_strokes']):
        stroke_seed = params['seed'] + (s_idx * 500)
        s_rng = np.random.RandomState(stroke_seed)
        
        # 1. Generate Full Path
        full_mx, full_my, t_vals = get_centered_spine(params['complexity'], stroke_seed, params['scale'] * s_rng.uniform(0.9, 1.1))
        
        # 2. Slice Path based on animation progress
        mask = t_vals <= draw_limit
        if np.sum(mask) < 5: continue
        
        mx, my = full_mx[mask], full_my[mask]
        current_t = t_vals[mask]
        
        # Position Offset
        off_x = s_rng.uniform(-0.1, 0.1) * 500
        off_y = s_rng.uniform(-0.1, 0.1) * 500
        
        temp_layer = np.zeros((h, w, 3), dtype=np.float32)
        
        # DECIDE TOOL TYPE FOR THIS STROKE
        # 40% chance of Ribbon, 60% chance of Fiber Cluster
        tool_type = "ribbon" if s_rng.rand() < 0.4 else "cluster"
        
        if tool_type == "ribbon":
            # === TOOL A: THE GOLDEN RIBBON ===
            # Wide, variable thickness based on twist
            
            # Calculate normals for width expansion
            nx, ny = get_path_normals(mx, my)
            
            # Twist math: varies width from thick to thin
            twist_freq = s_rng.uniform(2.0, 5.0)
            twist_phase = s_rng.uniform(0, 2*np.pi)
            # Width factor goes from 0.1 (thin edge) to 1.0 (flat face)
            twist = np.abs(np.sin(current_t * twist_freq + twist_phase)) * 0.9 + 0.1
            
            current_width = twist * params['spread'] * 30.0 * params['line_weight']
            
            # Construct Polygon strip
            lx = mx + nx * (current_width / 2.0)
            ly = my + ny * (current_width / 2.0)
            rx = mx - nx * (current_width / 2.0)
            ry = my - ny * (current_width / 2.0)
            
            px_l = lx * 450 + (w/2) + off_x
            py_l = ly * 450 + (h/2) + off_y
            px_r = rx * 450 + (w/2) + off_x
            py_r = ry * 450 + (h/2) + off_y
            
            pts_l = np.stack([px_l, py_l], axis=1)
            pts_r = np.stack([px_r, py_r], axis=1)[::-1]
            poly = np.concatenate([pts_l, pts_r]).astype(np.int32)
            
            # Color & Draw
            base_c = ribbon_palette[s_rng.randint(0, len(ribbon_palette))]
            intensity = params['exposure'] * s_rng.uniform(0.3, 0.6) # Ribbons are dimmer
            cv2.fillPoly(temp_layer, [poly], np.array(base_c) * intensity, lineType=cv2.LINE_AA)
            
        else:
            # === TOOL B: THE FIBER CLUSTER ===
            # Thin, bright, parallel strands
            
            num_fibers = s_rng.randint(3, 8)
            base_c = cluster_palette[s_rng.randint(0, len(cluster_palette))]
            
            for f_idx in range(num_fibers):
                # slight parallel offset
                fiber_off = (f_idx - num_fibers/2) * params['spread'] * 0.005
                
                sx = mx + fiber_off
                sy = my + fiber_off
                
                px = sx * 450 + (w/2) + off_x
                py = sy * 450 + (h/2) + off_y
                pts = np.stack([px, py], axis=1).astype(np.int32)
                
                # Color & Draw
                # Clusters are brighter and thinner
                intensity = params['exposure'] * s_rng.uniform(0.8, 1.5)
                thick = max(1, int(params['line_weight'] * s_rng.uniform(1.0, 2.5)))
                
                cv2.polylines(temp_layer, [pts], False, np.array(base_c) * intensity, thickness=thick, lineType=cv2.LINE_AA)

        # Add this stroke to master buffer
        accumulation_buffer += temp_layer

    # --- POST PROCESSING ---
    if params['glow'] > 0:
        bloom = gaussian_filter(accumulation_buffer, sigma=params['glow'] * 12)
        accumulation_buffer += bloom * 0.5

    if params['aberration'] > 0:
        shift = int(params['aberration'])
        if shift > 0:
            accumulation_buffer[:, :-shift, 0] = accumulation_buffer[:, shift:, 0]
            accumulation_buffer[:, shift:, 2] = accumulation_buffer[:, :-shift, 2]

    if params['blur'] > 0:
        accumulation_buffer = gaussian_filter(accumulation_buffer, sigma=params['blur'])
        
    accumulation_buffer *= global_alpha
    final_image = np.clip(accumulation_buffer, 0, 1.0) * 255
    return final_image.astype(np.uint8)

# --- UI ---
st.title("🔦 Lights of Time: Hybrid Studio")

with st.sidebar:
    st.header("Composition")
    mode = st.radio("Mode", ["Still Light", "Animated Loop"], key="mode")
    
    col1, col2 = st.columns(2)
    with col1:
        num_strokes = st.number_input("Light Sources", 1, 8, 5, help="Distinct light tools.")
    with col2:
        scale = st.slider("Zoom", 0.5, 1.5, 0.9)
        
    st.divider()
    
    st.header("Physics")
    complexity = st.slider("Gesture Complexity", 1, 4, value=2)
    line_weight = st.slider("Base Thickness", 0.5, 3.0, value=1.5, help="Scales both ribbons and fibers.")
    spread = st.slider("Tool Width/Spread", 0.5, 4.0, value=2.0, help="Width of ribbons, tightness of clusters.")
    exposure = st.slider("Exposure", 0.1, 1.0, value=0.4)
    
    with st.expander("Optics"):
        aberration = st.slider("Prism Shift", 0.0, 10.0, value=5.0)
        glow = st.slider("Glow Amount", 0.0, 3.0, value=1.2)
        blur = st.slider("Softness", 0.0, 3.0, value=0.7)
    
    seed = st.number_input("Seed", step=1, value=888)
    gen_btn = st.button("EXPOSE FILM", type="primary", use_container_width=True)

preview = st.empty()

if gen_btn:
    params = {
        "seed": seed, "complexity": complexity, "exposure": exposure,
        "glow": glow, "aberration": aberration, "spread": spread,
        "blur": blur, "num_strokes": num_strokes, "scale": scale,
        "line_weight": line_weight, "mode": mode
    }
    
    with preview.container():
        bar = st.progress(0, text="Integrating Light Tools...")
        if mode == "Still Light":
            img_bgr = render_hybrid_frame(params, 1.0)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, use_container_width=True)
            is_success, buffer = cv2.imencode(".png", img_bgr)
            data = buffer.tobytes()
        else:
            frames = []
            for i in range(60):
                f_bgr = render_hybrid_frame(params, i/60)
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
