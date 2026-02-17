import streamlit as st
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import imageio
import io
import random

# PAGE CONFIG
st.set_page_config(page_title="Lights of Time: Scatter Studio", page_icon="🔦", layout="wide")

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

def get_smooth_spine(complexity, seed, scale_factor):
    """Generates the master path center-lines."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 1, 6000) 
    x, y = np.zeros_like(t), np.zeros_like(t)
    
    # Low frequency sweeping arcs
    for i in range(1, complexity + 1):
        freq = rng.uniform(0.1, 0.4) * i 
        amp = rng.uniform(0.5, 1.2) / (i**0.5)
        phase = rng.uniform(0, 2*np.pi)
        x += amp * np.cos(freq * t * 2 * np.pi + phase)
        y += amp * np.sin(freq * t * 1.5 * np.pi + phase * 0.5)
        
    x = (x - np.mean(x)) * scale_factor
    y = (y - np.mean(y)) * scale_factor
    
    # Compute normal vectors for the scatter brush direction
    dx = np.gradient(x)
    dy = np.gradient(y)
    dist = np.sqrt(dx**2 + dy**2) + 1e-6
    nx = -dy / dist
    ny = dx / dist
    
    return x, y, nx, ny, t

# --- PALETTE DEFINITIONS ---
PALETTES = {
    "Gorospe Gold/Ice": {
        "ribbons": [(0.0, 0.6, 1.0), (0.0, 0.8, 1.0), (0.1, 0.4, 0.9)], 
        "clusters": [(0.95, 0.95, 1.0), (0.9, 1.0, 1.0)], 
    },
    "RGB Chaos": {
        "ribbons": [(0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)], 
        "clusters": [(0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0)], 
    },
    "Neon Cyber": {
        "ribbons": [(1.0, 0.2, 0.0), (0.8, 0.0, 1.0)], 
        "clusters": [(0.0, 1.0, 0.5), (0.0, 1.0, 0.0)], 
    }
}

def render_frame(params, prog):
    w, h = 1080, 1080
    buffer = np.zeros((h, w, 3), dtype=np.float32)
    rng = np.random.RandomState(params['seed'])
    
    palette = PALETTES[params['palette']]
    
    # Draw Limit & Fade
    draw_limit = 1.0 if params['mode'] == "Still Light" else prog
    if draw_limit < 0.005: draw_limit = 0.005
    global_alpha = 1.0
    if params['mode'] == "Animated Loop" and prog > 0.9:
        global_alpha = 1.0 - ((prog - 0.9) * 10.0)

    for s_idx in range(params['num_strokes']):
        s_seed = params['seed'] + (s_idx * 500)
        s_rng = np.random.RandomState(s_seed)
        
        # Z-Depth scale
        z_depth = s_rng.uniform(0.7, 1.3)
        
        # 1. Path
        fx, fy, fnx, fny, t_vals = get_smooth_spine(params['complexity'], s_seed, params['scale'] * z_depth)
        mask = t_vals <= draw_limit
        if np.sum(mask) < 10: continue
        
        mx, my = fx[mask], fy[mask]
        nx, ny = fnx[mask], fny[mask]
        ct = t_vals[mask]
        
        # Global position offset
        off_x = s_rng.uniform(-0.1, 0.1) * 500
        off_y = s_rng.uniform(-0.1, 0.1) * 500
        
        # 2. Tool Type
        is_ribbon = s_rng.rand() < 0.5
        
        temp = np.zeros((h, w, 3), dtype=np.float32)
        
        if is_ribbon:
            # === THE SCATTER RIBBON ===
            # We draw 20-40 individual lines ("bristles") per ribbon
            num_bristles = 30
            
            # Twist Physics
            twist_freq = s_rng.uniform(3.0, 8.0)
            twist_phase = s_rng.uniform(0, 2*np.pi)
            twist = np.abs(np.sin(ct * twist_freq + twist_phase))
            
            base_col = np.array(palette["ribbons"][s_rng.randint(0, len(palette["ribbons"]))]) * params['exposure']
            
            for b in range(num_bristles):
                # SCATTER LOGIC:
                # 1. Random offset from center (Scatter)
                # 2. Random thickness (Width Jitter)
                
                # Offset: Most are near center, some scatter wide
                scatter_amount = s_rng.normal(0, 0.5) * params['width_scale'] * 12.0
                
                # Modulate scatter by Twist (thin sections have less scatter)
                current_offset = scatter_amount * twist * z_depth
                
                bx = mx + nx * (current_offset * 0.002)
                by = my + ny * (current_offset * 0.002)
                
                # Random Thickness per bristle
                # Some are 1px, some are 3px
                thick = max(1, int(s_rng.uniform(0.5, 2.5)))
                
                px = bx*450 + w/2 + off_x
                py = by*450 + h/2 + off_y
                pts = np.stack([px, py], axis=1).astype(np.int32)
                
                # Lower opacity for scatter brush effect
                cv2.polylines(temp, [pts], False, base_col * 0.4, thickness=thick, lineType=cv2.LINE_AA)

        else:
            # === THE SCATTER CLUSTER ===
            # Fewer lines, but brighter and sharper
            num_fibers = s_rng.randint(5, 12)
            base_col = np.array(palette["clusters"][s_rng.randint(0, len(palette["clusters"]))]) * params['exposure']
            
            for f in range(num_fibers):
                # Spread
                fo = s_rng.normal(0, 1.0) * params['spread'] * 0.002 * z_depth
                sx = mx + fo; sy = my + fo
                
                px = sx*450 + w/2 + off_x
                py = sy*450 + h/2 + off_y
                pts = np.stack([px, py], axis=1).astype(np.int32)
                
                # Random thickness
                thick = 1 if s_rng.rand() > 0.3 else 2
                
                # Higher intensity
                cv2.polylines(temp, [pts], False, base_col * 1.2, thickness=thick, lineType=cv2.LINE_AA)
                
                # Draw Tip
                if params['mode'] == "Animated Loop" and len(pts) > 5:
                     cv2.polylines(temp, [pts[-15:]], False, (2,2,2), thickness=thick+2, lineType=cv2.LINE_AA)

        buffer += temp

    # --- POST ---
    if params['glow'] > 0:
        bloom = gaussian_filter(buffer, sigma=params['glow'] * 5)
        buffer += bloom * 0.3
        
    if params['aberration'] > 0:
        s = int(params['aberration'])
        if s > 0:
            buffer[:, :-s, 0] = buffer[:, s:, 0]
            buffer[:, s:, 2] = buffer[:, :-s, 2]
            
    buffer *= global_alpha
    return np.clip(buffer, 0, 1.0) * 255

# --- UI ---
st.title("🔦 Lights of Time: Scatter Studio")

with st.sidebar:
    st.header("Art Direction")
    mode = st.radio("Mode", ["Still Light", "Animated Loop"], key="mode")
    palette = st.selectbox("Color Palette", list(PALETTES.keys()), key="pal")
    
    col1, col2 = st.columns(2)
    with col1:
        num_strokes = st.number_input("Light Sources", 1, 20, 10)
    with col2:
        scale = st.slider("Zoom", 0.5, 1.5, 0.85)
        
    st.divider()
    st.header("Brush Physics")
    complexity = st.slider("Complexity", 1, 4, 2)
    width_scale = st.slider("Scatter Width", 0.1, 3.0, 1.0, help="How wide the brush scatters.")
    spread = st.slider("Cluster Spread", 0.1, 3.0, 1.0)
    exposure = st.slider("Exposure", 0.1, 2.0, 0.6)
    
    with st.expander("Lens Optics"):
        aberration = st.slider("Prism Shift", 0.0, 10.0, 3.0)
        glow = st.slider("Glow", 0.0, 3.0, 1.0)
    
    seed = st.number_input("Seed", step=1, value=501)
    gen = st.button("EXPOSE FILM", type="primary", use_container_width=True)

main = st.empty()

if gen:
    p = {
        "seed": seed, "complexity": complexity, "exposure": exposure,
        "glow": glow, "aberration": aberration, "spread": spread,
        "num_strokes": num_strokes, "scale": scale, "width_scale": width_scale,
        "mode": mode, "palette": palette
    }
    
    with main.container():
        bar = st.progress(0, text="Scattering Light Particles...")
        if mode == "Still Light":
            res = render_frame(p, 1.0)
            st.image(cv2.cvtColor(res.astype(np.uint8), cv2.COLOR_BGR2RGB), use_container_width=True)
            is_success, buffer = cv2.imencode(".png", res)
            data = buffer.tobytes()
        else:
            frames = []
            for i in range(60):
                res = render_frame(p, i/60)
                frames.append(cv2.cvtColor(res.astype(np.uint8), cv2.COLOR_BGR2RGB))
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
