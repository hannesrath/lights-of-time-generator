import streamlit as st
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import imageio
import io
import random
import time
import threading
from streamlit.runtime.scriptrunner import add_script_run_ctx

# PAGE CONFIG
st.set_page_config(page_title="Lights of Time: Cinematic Studio", page_icon="🔦", layout="wide")

# UI STYLING
st.markdown("""
    <style>
    .stApp { background-color: #000000 !important; color: #e0e0e0 !important; }
    [data-testid="stSidebar"] { background-color: #0a0a0a !important; border-right: 1px solid #222; }
    div.stButton > button { background-color: #222 !important; color: #fff !important; border: 1px solid #444 !important; }
    </style>
    """, unsafe_allow_html=True)

# SESSION STATE
if 'history' not in st.session_state: st.session_state.history = []
if 'is_generating' not in st.session_state: st.session_state.is_generating = False
if 'progress' not in st.session_state: st.session_state.progress = 0.0

# --- MATH ENGINE ---

def get_cinematic_spine(complexity, seed, scale_factor, aspect_ratio):
    """Generates a path that GUARANTEES entry and exit from the frame."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 1, 12000) # Ultra high res
    
    # 1. Base Oscillation (The "Dance")
    x_osc = np.zeros_like(t)
    y_osc = np.zeros_like(t)
    
    ar_scale_x = 1.0; ar_scale_y = 1.0
    if aspect_ratio < 1.0: ar_scale_x = aspect_ratio 
    else: ar_scale_y = 1.0 / aspect_ratio

    for i in range(1, complexity + 1):
        freq = rng.uniform(0.1, 0.4) * i 
        amp = rng.uniform(0.5, 1.2) / (i**0.5)
        phase = rng.uniform(0, 2*np.pi)
        x_osc += amp * np.cos(freq * t * 2 * np.pi + phase) * ar_scale_x
        y_osc += amp * np.sin(freq * t * 1.5 * np.pi + phase * 0.5) * ar_scale_y
    
    # 2. Linear Transport (The "Walk Through")
    # We pick a random angle and force the path to travel along it
    # Magnitude 3.0 ensures it crosses the screen (which is size 1.0 to 2.0)
    angle = rng.uniform(0, 2*np.pi)
    transport_mag = 3.5 
    
    # t goes 0->1. We map it to -0.5 -> 0.5 to center the movement
    t_centered = t - 0.5
    
    x_trans = np.cos(angle) * t_centered * transport_mag
    y_trans = np.sin(angle) * t_centered * transport_mag
    
    # Combine: The gesture rides on top of the transport
    x = (x_osc * 0.4 * scale_factor) + x_trans
    y = (y_osc * 0.4 * scale_factor) + y_trans
    
    # Normals
    dx = np.gradient(x); dy = np.gradient(y)
    dist = np.sqrt(dx**2 + dy**2) + 1e-6
    nx = -dy / dist; ny = dx / dist
    
    return x, y, nx, ny, t

# --- PALETTES ---
PALETTES = {
    "RGB Chaos": {
        "ribbons": [(0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 1.0), (1.0, 0.0, 1.0)],
        "clusters": [(0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0), (1.0, 0.5, 0.0), (0.5, 0.0, 1.0)],
    },
    "Gorospe Gold/Ice": {
        "ribbons": [(0.0, 0.6, 1.0), (0.0, 0.8, 1.0), (0.1, 0.4, 0.9)], 
        "clusters": [(0.95, 0.95, 1.0), (0.9, 1.0, 1.0)], 
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

# --- RENDERER ---

def render_frame_math(params, prog):
    h = 1080
    w = int(1080 * params['ar_val'])
    
    buffer = np.zeros((h, w, 3), dtype=np.float32)
    rng = np.random.RandomState(params['seed'])
    palette = PALETTES[params['palette']]
    
    # Animation Physics
    if params['mode'] == "Still Light":
        t_start = 0.0; t_end = 1.0
    else:
        trail_len = params['trail_len']
        # Prog goes 0->1.
        # We need the window [start, end] to slide from "Before 0" to "After 1"
        # Window size is trail_len.
        # Total travel distance is roughly 1.0 + trail_len
        
        travel_span = 1.2 + trail_len
        current_head = (prog * travel_span) - 0.1 # Start slightly before 0
        
        t_end = current_head
        t_start = current_head - trail_len

    layers = [] 

    for s_idx in range(params['num_strokes']):
        s_seed = params['seed'] + (s_idx * 500)
        s_rng = np.random.RandomState(s_seed)
        z_depth = s_rng.uniform(0.6, 1.4)
        
        # Use the new Cinematic Spine
        fx, fy, fnx, fny, t_vals = get_cinematic_spine(params['complexity'], s_seed, params['scale'] * (1.0/z_depth), params['ar_val'])
        
        # Temporal Slicing
        mask = (t_vals >= t_start) & (t_vals <= t_end)
        if np.sum(mask) < 2: continue
        
        mx, my = fx[mask], fy[mask]
        nx, ny = fnx[mask], fny[mask]
        ct = t_vals[mask]
        
        # Opacity Gradient (Tail Fade)
        if params['mode'] == "Animated Loop":
            # Distance from tail start
            dist_from_tail = ct - t_start
            # Normalize by trail length
            op = dist_from_tail / (trail_len + 1e-6)
            op = np.clip(op, 0.0, 1.0)
            opacity_curve = np.power(op, 0.5) # Soft curve
        else:
            opacity_curve = np.ones_like(ct)
        
        off_x = s_rng.uniform(-0.05, 0.05) * w
        off_y = s_rng.uniform(-0.05, 0.05) * h
        
        is_ribbon = s_rng.rand() < 0.5
        stroke_layer = np.zeros((h, w, 3), dtype=np.float32)
        
        if is_ribbon:
            num_bristles = 40
            twist_freq = s_rng.uniform(3.0, 8.0)
            twist_phase = s_rng.uniform(0, 2*np.pi)
            twist = np.abs(np.sin(ct * twist_freq + twist_phase))
            
            base_col = np.array(palette["ribbons"][s_rng.randint(0, len(palette["ribbons"]))]) * params['exposure']
            if params['palette'] == "RGB Chaos": base_col *= 1.5
            
            for b in range(num_bristles):
                scatter_amount = s_rng.normal(0, 0.5) * params['width_scale'] * 35.0
                current_offset = scatter_amount * twist
                
                bx = mx + nx * (current_offset * 0.002)
                by = my + ny * (current_offset * 0.002)
                thick = max(1, int(s_rng.uniform(0.5, 3.0)))
                
                px = bx*(h*0.4) + w/2 + off_x
                py = by*(h*0.4) + h/2 + off_y
                pts = np.stack([px, py], axis=1).astype(np.int32)
                
                # Check bounds optimization
                if np.all(px < 0) or np.all(px > w) or np.all(py < 0) or np.all(py > h): continue
                
                avg_op = np.mean(opacity_curve)
                cv2.polylines(stroke_layer, [pts], False, base_col * 0.4 * avg_op, thickness=thick, lineType=cv2.LINE_AA)

        else:
            num_fibers = s_rng.randint(6, 12)
            base_col = np.array(palette["clusters"][s_rng.randint(0, len(palette["clusters"]))]) * params['exposure']
            if params['palette'] == "RGB Chaos": base_col *= 1.3
            
            for f in range(num_fibers):
                fo = s_rng.normal(0, 1.0) * params['spread'] * 0.002
                sx = mx + fo; sy = my + fo
                px = sx*(h*0.4) + w/2 + off_x
                py = sy*(h*0.4) + h/2 + off_y
                pts = np.stack([px, py], axis=1).astype(np.int32)
                
                if np.all(px < 0) or np.all(px > w) or np.all(py < 0) or np.all(py > h): continue

                thick = 1 if s_rng.rand() > 0.3 else 2
                avg_op = np.mean(opacity_curve)
                cv2.polylines(stroke_layer, [pts], False, base_col * 1.5 * avg_op, thickness=thick, lineType=cv2.LINE_AA)
                
                if params['mode'] == "Animated Loop" and len(pts) > 2 and t_end < 0.99:
                    tip_pt = pts[-1:]
                    cv2.circle(stroke_layer, tuple(tip_pt[0]), thick+2, (1.0, 1.0, 1.0), -1)

        layers.append((z_depth, stroke_layer))

    # Compositing
    for z, layer in layers:
        dist_from_focus = abs(z - 1.0)
        blur_amount = dist_from_focus * params['dof'] * 10.0
        if blur_amount > 0.5:
            layer = gaussian_filter(layer, sigma=blur_amount)
        buffer += layer

    if params['glow'] > 0:
        bloom = gaussian_filter(buffer, sigma=params['glow'] * 5)
        buffer += bloom * 0.3
        
    if params['aberration'] > 0:
        s = int(params['aberration'])
        if s > 0:
            buffer[:, :-s, 0] = buffer[:, s:, 0]
            buffer[:, s:, 2] = buffer[:, :-s, 2]
            
    return np.clip(buffer, 0, 1.0) * 255

# --- BACKGROUND WORKER ---

def background_generation(params):
    try:
        if params['mode'] == "Still Light":
            st.session_state.progress = 0.5
            res = render_frame_math(params, 0.5) # Capture middle of stroke for still
            is_success, buffer = cv2.imencode(".png", res)
            data = buffer.tobytes()
            fmt = "png"
            st.session_state.history.insert(0, {"data": data, "fmt": fmt, "meta": params, "time": time.strftime("%H:%M:%S")})
            st.session_state.progress = 1.0
            
        else:
            frames = []
            total_frames = 90
            for i in range(total_frames):
                res = render_frame_math(params, i/total_frames)
                frames.append(cv2.cvtColor(res.astype(np.uint8), cv2.COLOR_BGR2RGB))
                st.session_state.progress = (i + 1) / total_frames
            
            b = io.BytesIO()
            imageio.mimsave(b, frames, format='GIF', fps=30, loop=0)
            data = b.getvalue()
            fmt = "gif"
            st.session_state.history.insert(0, {"data": data, "fmt": fmt, "meta": params, "time": time.strftime("%H:%M:%S")})
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        st.session_state.is_generating = False

def start_generation(params):
    st.session_state.is_generating = True
    st.session_state.progress = 0.0
    t = threading.Thread(target=background_generation, args=(params,))
    add_script_run_ctx(t)
    t.start()

# --- SIDEBAR & UI ---
def delete_item(idx):
    st.session_state.history.pop(idx)
    st.rerun()

def restore_settings(meta):
    if not st.session_state.is_generating:
        for k, v in meta.items():
            if k in st.session_state: continue 
            # We skip direct session state set for widget keys to avoid warnings,
            # but in this simplified script we just rerun which resets widgets to defaults
            # unless we use specific key management. For now, simple restore:
            pass 
        # Ideally, we map meta to st.session_state values here
        st.rerun()

st.title("🔦 Lights of Time")

with st.sidebar:
    st.header("1. Canvas & Mode")
    mode = st.radio("Output Mode", ["Still Light", "Animated Loop"], key="mode")
    ar_options = {"1:1": 1.0, "9:16": 9/16, "16:9": 16/9}
    ar_name = st.selectbox("Aspect Ratio", list(ar_options.keys()), key="ar_select")
    
    trail_len = 1.0
    if mode == "Animated Loop":
        trail_len = st.slider("Trail Length", 0.1, 0.8, 0.4, help="Length of the light snake.")

    st.divider()
    st.header("2. Art Direction")
    palette = st.selectbox("Palette", list(PALETTES.keys()), index=0, key="pal")
    num_strokes = st.number_input("Sources", 1, 30, 15, key="num_strokes")
    
    st.divider()
    st.header("3. Physics")
    complexity = st.slider("Complexity", 1, 4, 3, key="complexity")
    width_scale = st.slider("Ribbon Width", 0.1, 3.0, 1.5, key="width_scale")
    exposure = st.slider("Exposure", 0.1, 2.0, 0.8, key="exposure")
    dof = st.slider("Blur (DoF)", 0.0, 2.0, 1.0, key="dof")
    
    st.divider()
    seed = st.number_input("Seed", value=99, key="seed")
    
    if st.button("EXPOSE FILM", type="primary", disabled=st.session_state.is_generating):
        p = {
            "seed": seed, "complexity": complexity, "exposure": exposure,
            "num_strokes": num_strokes, "width_scale": width_scale, "mode": mode, 
            "palette": palette, "dof": dof, "ar_val": ar_options[ar_name], "ar_name": ar_name,
            "trail_len": trail_len, "scale": 0.85, "glow": 1.0, "aberration": 3.0, "spread": 1.0
        }
        start_generation(p)
        st.rerun()

# --- MAIN ---
if st.session_state.is_generating:
    st.info("Generating cinematic exposure...")
    st.progress(st.session_state.progress)
    time.sleep(0.5)
    st.rerun()

if st.session_state.history:
    st.divider()
    cols = st.columns(3)
    for idx, item in enumerate(st.session_state.history):
        with cols[idx % 3]:
            st.image(item['data'], use_container_width=True)
            c1, c2 = st.columns(2)
            with c1: st.download_button("💾", item['data'], f"img_{idx}.{item['fmt']}")
            with c2: 
                if st.button("🗑️", key=f"del_{idx}"): delete_item(idx)
