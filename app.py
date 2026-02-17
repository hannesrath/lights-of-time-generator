import streamlit as st
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import imageio
import io
import random
import time

# PAGE CONFIG
st.set_page_config(page_title="Lights of Time: Gorospe Studio", page_icon="🔦", layout="wide")

# UI STYLING
st.markdown("""
    <style>
    .stApp { background-color: #000000 !important; color: #e0e0e0 !important; }
    [data-testid="stSidebar"] { background-color: #0a0a0a !important; border-right: 1px solid #222; }
    div.stButton > button { background-color: #222 !important; color: #fff !important; border: 1px solid #444 !important; }
    .gallery-card { border: 1px solid #333; padding: 10px; border-radius: 5px; background: #111; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# INITIALIZE HISTORY
if 'history' not in st.session_state: st.session_state.history = []

# --- MATH ENGINE ---

def get_smooth_spine(complexity, seed, scale_factor, aspect_ratio):
    """Generates the master path center-lines, adjusted for AR."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 1, 6000) 
    x, y = np.zeros_like(t), np.zeros_like(t)
    
    # Aspect Ratio Correction
    # If portrait (9:16), we need to squeeze X and stretch Y slightly
    ar_scale_x = 1.0
    ar_scale_y = 1.0
    if aspect_ratio < 1.0: ar_scale_x = aspect_ratio # Portrait
    else: ar_scale_y = 1.0 / aspect_ratio # Landscape

    for i in range(1, complexity + 1):
        freq = rng.uniform(0.1, 0.4) * i 
        amp = rng.uniform(0.5, 1.2) / (i**0.5)
        phase = rng.uniform(0, 2*np.pi)
        x += amp * np.cos(freq * t * 2 * np.pi + phase) * ar_scale_x
        y += amp * np.sin(freq * t * 1.5 * np.pi + phase * 0.5) * ar_scale_y
        
    x = (x - np.mean(x)) * scale_factor
    y = (y - np.mean(y)) * scale_factor
    
    # Normals for width
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
    },
    "Tungsten Wire": {
        "ribbons": [(0.1, 0.3, 1.0), (0.2, 0.2, 0.8)], 
        "clusters": [(0.6, 0.8, 1.0), (0.5, 0.5, 0.9)], 
    }
}

# --- RENDER ENGINE ---

def render_frame(params, prog):
    # Set Canvas Dimensions based on Aspect Ratio
    h = 1080
    w = int(1080 * params['ar_val'])
    
    buffer = np.zeros((h, w, 3), dtype=np.float32)
    rng = np.random.RandomState(params['seed'])
    
    palette = PALETTES[params['palette']]
    
    # Draw Limit
    draw_limit = 1.0 if params['mode'] == "Still Light" else prog
    if draw_limit < 0.005: draw_limit = 0.005
    
    # Loop Fade
    global_alpha = 1.0
    if params['mode'] == "Animated Loop" and prog > 0.9:
        global_alpha = 1.0 - ((prog - 0.9) * 10.0)

    # --- MULTI-STROKE LOOP ---
    # We create a list of layers so we can blur them individually for DoF
    layers = [] # (z_depth, image_layer)

    for s_idx in range(params['num_strokes']):
        s_seed = params['seed'] + (s_idx * 500)
        s_rng = np.random.RandomState(s_seed)
        
        # Z-Depth: 0.5 (Far) to 1.5 (Close)
        # Focus Plane is around 1.0
        z_depth = s_rng.uniform(0.5, 1.5)
        
        # 1. Path
        fx, fy, fnx, fny, t_vals = get_smooth_spine(params['complexity'], s_seed, params['scale'] * (1.0/z_depth), params['ar_val'])
        mask = t_vals <= draw_limit
        if np.sum(mask) < 10: continue
        
        mx, my = fx[mask], fy[mask]
        nx, ny = fnx[mask], fny[mask]
        ct = t_vals[mask]
        
        # Position Offset
        off_x = s_rng.uniform(-0.1, 0.1) * (w/2)
        off_y = s_rng.uniform(-0.1, 0.1) * (h/2)
        
        # 2. Tool Type
        is_ribbon = s_rng.rand() < 0.5
        
        stroke_layer = np.zeros((h, w, 3), dtype=np.float32)
        
        if is_ribbon:
            # === SCATTER RIBBON ===
            num_bristles = 25
            twist_freq = s_rng.uniform(3.0, 8.0)
            twist_phase = s_rng.uniform(0, 2*np.pi)
            twist = np.abs(np.sin(ct * twist_freq + twist_phase))
            
            base_col = np.array(palette["ribbons"][s_rng.randint(0, len(palette["ribbons"]))]) * params['exposure']
            
            for b in range(num_bristles):
                scatter_amount = s_rng.normal(0, 0.5) * params['width_scale'] * 12.0
                current_offset = scatter_amount * twist
                
                bx = mx + nx * (current_offset * 0.002)
                by = my + ny * (current_offset * 0.002)
                
                thick = max(1, int(s_rng.uniform(0.5, 2.5)))
                
                px = bx*(h*0.4) + w/2 + off_x
                py = by*(h*0.4) + h/2 + off_y
                pts = np.stack([px, py], axis=1).astype(np.int32)
                
                cv2.polylines(stroke_layer, [pts], False, base_col * 0.4, thickness=thick, lineType=cv2.LINE_AA)

        else:
            # === SCATTER CLUSTER ===
            num_fibers = s_rng.randint(5, 10)
            base_col = np.array(palette["clusters"][s_rng.randint(0, len(palette["clusters"]))]) * params['exposure']
            
            for f in range(num_fibers):
                fo = s_rng.normal(0, 1.0) * params['spread'] * 0.002
                sx = mx + fo; sy = my + fo
                px = sx*(h*0.4) + w/2 + off_x
                py = sy*(h*0.4) + h/2 + off_y
                pts = np.stack([px, py], axis=1).astype(np.int32)
                
                thick = 1 if s_rng.rand() > 0.3 else 2
                cv2.polylines(stroke_layer, [pts], False, base_col * 1.2, thickness=thick, lineType=cv2.LINE_AA)
                
                if params['mode'] == "Animated Loop" and len(pts) > 5:
                     cv2.polylines(stroke_layer, [pts[-15:]], False, (2,2,2), thickness=thick+2, lineType=cv2.LINE_AA)

        layers.append((z_depth, stroke_layer))

    # --- COMPOSITING WITH DEPTH OF FIELD ---
    # Sort layers by depth (far to near) - painter's algorithm
    # Actually, additive doesn't care about order, but blur does.
    
    for z, layer in layers:
        # DoF Logic:
        # Focus plane is 1.0. Distance from 1.0 determines blur.
        dist_from_focus = abs(z - 1.0)
        
        # Blur amount based on distance (f/1.4 simulation)
        blur_amount = dist_from_focus * params['dof'] * 10.0
        
        if blur_amount > 0.5:
            layer = gaussian_filter(layer, sigma=blur_amount)
            
        buffer += layer

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

# --- UI HANDLERS ---
def restore_settings(meta):
    st.session_state['seed'] = meta['seed']
    st.session_state['complexity'] = meta['complexity']
    st.session_state['exposure'] = meta['exposure']
    st.session_state['glow'] = meta['glow']
    st.session_state['aberration'] = meta['aberration']
    st.session_state['spread'] = meta['spread']
    st.session_state['num_strokes'] = meta['num_strokes']
    st.session_state['scale'] = meta['scale']
    st.session_state['width_scale'] = meta['width_scale']
    st.session_state['mode'] = meta['mode']
    st.session_state['pal'] = meta['palette']
    st.session_state['dof'] = meta['dof']
    
    # Map AR string back to index
    ar_map = {"1:1 (Square)": 0, "9:16 (Story)": 1, "16:9 (Cinema)": 2, "4:3 (Classic)": 3}
    # We can't set index directly on radio easily if key is used, but we can try updating session state
    # A workaround for radio index is usually not needed if we just render with the restored params.
    # But to update UI:
    st.session_state['ar_select'] = meta['ar_name']

def delete_item(idx):
    st.session_state.history.pop(idx)
    st.rerun()

# --- SIDEBAR UI ---
st.title("🔦 Lights of Time")

with st.sidebar:
    st.header("1. Canvas & Mode")
    mode = st.radio("Output Mode", ["Still Light", "Animated Loop"], key="mode", help="Still image or 60-frame GIF write-on.")
    ar_options = {"1:1 (Square)": 1.0, "9:16 (Story)": 9/16, "16:9 (Cinema)": 16/9, "4:3 (Classic)": 4/3}
    ar_name = st.selectbox("Aspect Ratio", list(ar_options.keys()), key="ar_select", help="Shape of the final output.")
    ar_val = ar_options[ar_name]

    st.divider()
    
    st.header("2. Art Direction")
    palette = st.selectbox("Color Palette", list(PALETTES.keys()), key="pal", help="The color scheme of the light tools.")
    scale = st.slider("Zoom", 0.5, 1.5, 0.85, key="scale", help="How close the camera is to the light subject.")
    num_strokes = st.number_input("Light Sources", 1, 30, 12, key="num_strokes", help="Number of individual light strokes in the scene.")

    st.divider()
    
    st.header("3. Brush Physics")
    complexity = st.slider("Gesture Complexity", 1, 4, 2, key="complexity", help="1 = Simple arcs, 4 = Complex knots.")
    width_scale = st.slider("Scatter Width", 0.1, 3.0, 1.0, key="width_scale", help="How wide the ribbon brushes scatter.")
    spread = st.slider("Cluster Spread", 0.1, 3.0, 1.0, key="spread", help="How tight the fiber clusters are.")
    
    st.divider()
    
    st.header("4. Lens & Optics")
    exposure = st.slider("Exposure", 0.1, 2.0, 0.6, key="exposure", help="Brightness of the sensor.")
    dof = st.slider("Depth of Field (Blur)", 0.0, 2.0, 1.0, key="dof", help="Simulates Z-depth blur. 0 = Infinite focus, 2 = Macro focus.")
    glow = st.slider("Glow Intensity", 0.0, 3.0, 1.0, key="glow", help="Soft atmospheric bloom.")
    aberration = st.slider("Prism Shift", 0.0, 10.0, 3.0, key="aberration", help="Chromatic aberration on the edges.")
    
    st.divider()
    seed = st.number_input("Seed", step=1, value=501, key="seed", help="Random number generator seed.")
    gen = st.button("EXPOSE FILM", type="primary", use_container_width=True)

# --- MAIN PREVIEW AREA ---
main = st.empty()

if gen:
    p = {
        "seed": seed, "complexity": complexity, "exposure": exposure,
        "glow": glow, "aberration": aberration, "spread": spread,
        "num_strokes": num_strokes, "scale": scale, "width_scale": width_scale,
        "mode": mode, "palette": palette, "dof": dof, "ar_val": ar_val, "ar_name": ar_name
    }
    
    with main.container():
        bar = st.progress(0, text="Developing Exposure...")
        if mode == "Still Light":
            res = render_frame(p, 1.0)
            # Display
            st.image(cv2.cvtColor(res.astype(np.uint8), cv2.COLOR_BGR2RGB), use_container_width=True)
            # Encode
            is_success, buffer = cv2.imencode(".png", res)
            data = buffer.tobytes()
            fmt = "png"
        else:
            frames = []
            for i in range(60):
                res = render_frame(p, i/60)
                frames.append(cv2.cvtColor(res.astype(np.uint8), cv2.COLOR_BGR2RGB))
                bar.progress((i+1)/60)
            b = io.BytesIO()
            imageio.mimsave(b, frames, format='GIF', fps=30, loop=0)
            data = b.getvalue()
            fmt = "gif"
            st.image(data, use_container_width=True)
            
        # Add to history with metadata
        st.session_state.history.insert(0, {"data": data, "fmt": fmt, "meta": p, "time": time.strftime("%H:%M:%S")})

# --- GALLERY SECTION ---
if st.session_state.history:
    st.divider()
    st.subheader(f"Film Roll ({len(st.session_state.history)})")
    
    # Grid layout for gallery
    cols = st.columns(3)
    for idx, item in enumerate(st.session_state.history):
        with cols[idx % 3]:
            # Card container
            with st.container():
                st.image(item['data'], use_container_width=True)
                m = item['meta']
                
                # Metadata tag
                st.caption(f"**{m['mode']}** • {m['ar_name']} • Seed: {m['seed']}")
                
                # Action Buttons
                c1, c2, c3 = st.columns([1, 1, 1])
                with c1:
                    st.download_button("💾", item['data'], f"light_{idx}.{item['fmt']}", key=f"dl_{idx}")
                with c2:
                    if st.button("Apply", key=f"res_{idx}", help="Restore these settings"):
                        restore_settings(m)
                        st.rerun()
                with c3:
                    if st.button("🗑️", key=f"del_{idx}", help="Delete image"):
                        delete_item(idx)
            st.divider()
