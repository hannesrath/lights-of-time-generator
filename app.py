import streamlit as st
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import imageio
import io
import random

# PAGE CONFIG
st.set_page_config(page_title="Lights of Time: Gorospe Studio", page_icon="🔦", layout="wide")

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
    t = np.linspace(0, 1, 6000) # Higher res for thin lines
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
    return x, y, t

def get_path_normals(x, y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    dist = np.sqrt(dx**2 + dy**2) + 1e-6
    nx = -dy / dist
    ny = dx / dist
    return nx, ny

# --- PALETTE DEFINITIONS ---
PALETTES = {
    "Gorospe Gold/Ice": {
        "ribbons": [(0.05, 0.5, 1.0), (0.0, 0.7, 1.0), (0.1, 0.3, 0.9)], 
        "clusters": [(1.0, 1.0, 1.0), (0.9, 0.9, 1.0), (0.8, 0.8, 1.0)], 
    },
    "Electric Neon": {
        "ribbons": [(1.0, 0.2, 0.0), (1.0, 0.0, 0.5), (0.8, 0.0, 1.0)], 
        "clusters": [(1.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 1.0)], 
    },
    "Tungsten Wire": {
        "ribbons": [(0.1, 0.2, 0.8), (0.1, 0.1, 0.6), (0.3, 0.4, 0.9)], 
        "clusters": [(0.5, 0.6, 1.0), (0.4, 0.4, 0.9), (0.8, 0.8, 0.8)], 
    },
    "Prism Minimal": {
        "ribbons": [(0.9, 0.9, 0.9), (0.8, 0.8, 0.8)], 
        "clusters": [(1.0, 1.0, 1.0)], 
    }
}

def render_frame(params, prog):
    w, h = 1080, 1080
    buffer = np.zeros((h, w, 3), dtype=np.float32)
    rng = np.random.RandomState(params['seed'])
    
    palette = PALETTES[params['palette']]
    
    # Animation: Draw up to 'prog' %
    draw_limit = 1.0 if params['mode'] == "Still Light" else prog
    if draw_limit < 0.005: draw_limit = 0.005
    
    # Loop Fade Out
    global_alpha = 1.0
    if params['mode'] == "Animated Loop" and prog > 0.9:
        global_alpha = 1.0 - ((prog - 0.9) * 10.0)

    for s_idx in range(params['num_strokes']):
        s_seed = params['seed'] + (s_idx * 500)
        s_rng = np.random.RandomState(s_seed)
        
        # --- Z-DEPTH SIMULATION ---
        # Vary the scale significantly to push some strokes "back"
        z_depth = s_rng.uniform(0.6, 1.4) 
        
        # 1. Path Generation
        fx, fy, t_vals = get_smooth_spine(params['complexity'], s_seed, params['scale'] * z_depth)
        
        # 2. Slicing
        mask = t_vals <= draw_limit
        if np.sum(mask) < 10: continue
        mx, my = fx[mask], fy[mask]
        ct = t_vals[mask]
        
        # 3. Tool Selection
        is_ribbon = s_rng.rand() < 0.5
        
        # Offset position
        off_x = s_rng.uniform(-0.1, 0.1) * 500
        off_y = s_rng.uniform(-0.1, 0.1) * 500
        
        temp = np.zeros((h, w, 3), dtype=np.float32)
        
        if is_ribbon:
            # === RIBBON TOOL ===
            nx, ny = get_path_normals(mx, my)
            
            twist_freq = s_rng.uniform(2.0, 6.0)
            twist_phase = s_rng.uniform(0, 2*np.pi)
            twist = np.abs(np.sin(ct * twist_freq + twist_phase))
            
            # --- THE FIX: TINY BASE WIDTH ---
            # Reduced to 0.4 for a "distant tape" look
            # We also scale width by z_depth (further away = thinner)
            base_w = 0.4 * params['width_scale'] * z_depth
            width = (twist * base_w * 20.0) + 1.0 
            
            # Geometry
            lx = mx + nx * (width/2); ly = my + ny * (width/2)
            rx = mx - nx * (width/2); ry = my - ny * (width/2)
            
            pl = np.stack([lx*450 + w/2 + off_x, ly*450 + h/2 + off_y], axis=1)
            pr = np.stack([rx*450 + w/2 + off_x, ry*450 + h/2 + off_y], axis=1)[::-1]
            poly = np.concatenate([pl, pr]).astype(np.int32)
            
            # Color
            c = palette["ribbons"][s_rng.randint(0, len(palette["ribbons"]))]
            # Further away = Dimmer
            dist_dim = 1.0 if z_depth > 1.0 else 0.7
            col = np.array(c) * params['exposure'] * 0.6 * dist_dim
            
            cv2.fillPoly(temp, [poly], col, lineType=cv2.LINE_AA)
            
            # Hot core (very thin)
            core_pts = np.stack([mx*450 + w/2 + off_x, my*450 + h/2 + off_y], axis=1).astype(np.int32)
            cv2.polylines(temp, [core_pts], False, col * 1.8, thickness=1, lineType=cv2.LINE_AA)

        else:
            # === CLUSTER TOOL ===
            num_fibers = s_rng.randint(3, 7) # Fewer fibers for cleaner look
            c = palette["clusters"][s_rng.randint(0, len(palette["clusters"]))]
            
            for f in range(num_fibers):
                # Very tight spread
                fo = (f - num_fibers/2) * params['spread'] * 0.002 * z_depth
                sx = mx + fo; sy = my + fo
                
                pts = np.stack([sx*450 + w/2 + off_x, sy*450 + h/2 + off_y], axis=1).astype(np.int32)
                
                # Fiber color
                intensity = params['exposure'] * s_rng.uniform(1.2, 2.0)
                # Keep thickness minimal
                thick = 1
                
                cv2.polylines(temp, [pts], False, np.array(c) * intensity, thickness=thick, lineType=cv2.LINE_AA)
                
                if params['mode'] == "Animated Loop" and len(pts) > 5:
                     cv2.polylines(temp, [pts[-10:]], False, (2,2,2), thickness=2, lineType=cv2.LINE_AA)

        buffer += temp

    # --- POST ---
    if params['glow'] > 0:
        # Tighter glow for sharper look
        bloom = gaussian_filter(buffer, sigma=params['glow'] * 6)
        buffer += bloom * 0.4
        
    if params['aberration'] > 0:
        s = int(params['aberration'])
        if s > 0:
            buffer[:, :-s, 0] = buffer[:, s:, 0]
            buffer[:, s:, 2] = buffer[:, :-s, 2]
            
    buffer *= global_alpha
    return np.clip(buffer, 0, 1.0) * 255

# --- UI ---
st.title("🔦 Lights of Time: Gorospe Studio")

with st.sidebar:
    st.header("Art Direction")
    mode = st.radio("Mode", ["Still Light", "Animated Loop"], key="mode")
    palette = st.selectbox("Color Palette", list(PALETTES.keys()), key="pal")
    
    col1, col2 = st.columns(2)
    with col1:
        num_strokes = st.number_input("Light Sources", 1, 15, 10)
    with col2:
        # Reduced default zoom for "distant" look
        scale = st.slider("Zoom", 0.5, 1.5, 0.8)
        
    st.divider()
    st.header("Light Physics")
    complexity = st.slider("Complexity", 1, 4, 2)
    # Reduced default width scale
    width_scale = st.slider("Tool Width", 0.5, 3.0, 1.0, help="Scales ribbon width.")
    spread = st.slider("Fiber Spread", 0.5, 4.0, 1.5)
    exposure = st.slider("Exposure", 0.1, 2.0, 0.7)
    
    with st.expander("Lens Optics"):
        aberration = st.slider("Prism Shift", 0.0, 10.0, 3.0)
        glow = st.slider("Glow", 0.0, 3.0, 1.0)
    
    seed = st.number_input("Seed", step=1, value=999)
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
        bar = st.progress(0, text="Painting Light...")
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
