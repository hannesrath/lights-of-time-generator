import streamlit as st
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import imageio
import io
import random
import time

# PAGE CONFIG
st.set_page_config(page_title="Lights of Time: Motion Studio", page_icon="🔦", layout="wide")

# UI STYLING
st.markdown("""
    <style>
    .stApp { background-color: #000000 !important; color: #e0e0e0 !important; }
    [data-testid="stSidebar"] { background-color: #0a0a0a !important; border-right: 1px solid #222; }
    div.stButton > button { background-color: #222 !important; color: #fff !important; border: 1px solid #444 !important; }
    .gallery-card { border: 1px solid #333; padding: 10px; border-radius: 5px; background: #111; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

if 'history' not in st.session_state: st.session_state.history = []

# --- MATH ENGINE ---

def get_smooth_spine(complexity, seed, scale_factor, aspect_ratio):
    """Generates the master path center-lines."""
    rng = np.random.RandomState(seed)
    # Extremely high resolution to ensure smooth "melting" tails
    t = np.linspace(0, 1, 10000) 
    x, y = np.zeros_like(t), np.zeros_like(t)
    
    # AR Correction
    ar_scale_x = 1.0
    ar_scale_y = 1.0
    if aspect_ratio < 1.0: ar_scale_x = aspect_ratio 
    else: ar_scale_y = 1.0 / aspect_ratio

    for i in range(1, complexity + 1):
        freq = rng.uniform(0.1, 0.4) * i 
        amp = rng.uniform(0.5, 1.2) / (i**0.5)
        phase = rng.uniform(0, 2*np.pi)
        x += amp * np.cos(freq * t * 2 * np.pi + phase) * ar_scale_x
        y += amp * np.sin(freq * t * 1.5 * np.pi + phase * 0.5) * ar_scale_y
        
    x = (x - np.mean(x)) * scale_factor
    y = (y - np.mean(y)) * scale_factor
    
    # Normals
    dx = np.gradient(x)
    dy = np.gradient(y)
    dist = np.sqrt(dx**2 + dy**2) + 1e-6
    nx = -dy / dist
    ny = dx / dist
    
    return x, y, nx, ny, t

# --- PALETTE DEFINITIONS ---
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

# --- RENDER ENGINE ---

def render_frame(params, prog):
    h = 1080
    w = int(1080 * params['ar_val'])
    
    buffer = np.zeros((h, w, 3), dtype=np.float32)
    rng = np.random.RandomState(params['seed'])
    palette = PALETTES[params['palette']]
    
    # --- ANIMATION PHYSICS ---
    if params['mode'] == "Still Light":
        t_start = 0.0
        t_end = 1.0
    else:
        trail_len = params['trail_len']
        # The 'effective progress' must go from 0 to (1.0 + trail_len)
        # to allow the tail to fully exit the screen.
        effective_prog = prog * (1.0 + trail_len * 1.2) # 1.2 padding ensures full exit
        
        t_end = min(1.0, effective_prog)
        t_start = max(0.0, effective_prog - trail_len)
        
        # If t_start reached 1.0, the line is gone.
        if t_start >= 1.0: return np.zeros((h, w, 3), dtype=np.uint8)

    layers = [] 

    for s_idx in range(params['num_strokes']):
        s_seed = params['seed'] + (s_idx * 500)
        s_rng = np.random.RandomState(s_seed)
        z_depth = s_rng.uniform(0.5, 1.5)
        
        fx, fy, fnx, fny, t_vals = get_smooth_spine(params['complexity'], s_seed, params['scale'] * (1.0/z_depth), params['ar_val'])
        
        # --- GRADIENT TAIL LOGIC ---
        # Instead of a hard chop, we include a buffer zone for fading.
        fade_length = 0.05 # 5% of path length is used for fading the tail
        safe_start = max(0.0, t_start)
        
        # We grab points that are visible
        mask = (t_vals >= safe_start) & (t_vals <= t_end)
        
        if np.sum(mask) < 2: continue
        
        mx, my = fx[mask], fy[mask]
        nx, ny = fnx[mask], fny[mask]
        ct = t_vals[mask]
        
        # Calculate Opacity Curve for this segment
        # 1.0 at the Head (t_end), 0.0 at the Tail (t_start)
        if params['mode'] == "Animated Loop":
            # Linear fade from tail to head
            opacity_curve = (ct - safe_start) / (t_end - safe_start + 1e-6)
            opacity_curve = np.clip(opacity_curve, 0.0, 1.0)
            # Power curve for smoother "comet" look
            opacity_curve = np.power(opacity_curve, 0.5)
        else:
            opacity_curve = np.ones_like(ct)
        
        # Global position offset
        off_x = s_rng.uniform(-0.1, 0.1) * (w/2)
        off_y = s_rng.uniform(-0.1, 0.1) * (h/2)
        
        is_ribbon = s_rng.rand() < 0.5
        stroke_layer = np.zeros((h, w, 3), dtype=np.float32)
        
        if is_ribbon:
            # === SCATTER RIBBON ===
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
                
                # Draw solid line first (base geometry)
                # To simulate gradient fade on a polyline, we multiply the color by average opacity of the segment
                # For better results in high-motion, we rely on the frame updates to dissolve it, 
                # but applying the average opacity of this chunk helps smoothness.
                avg_op = np.mean(opacity_curve)
                
                cv2.polylines(stroke_layer, [pts], False, base_col * 0.4 * avg_op, thickness=thick, lineType=cv2.LINE_AA)

        else:
            # === SCATTER CLUSTER ===
            num_fibers = s_rng.randint(6, 12)
            base_col = np.array(palette["clusters"][s_rng.randint(0, len(palette["clusters"]))]) * params['exposure']
            if params['palette'] == "RGB Chaos": base_col *= 1.3
            
            for f in range(num_fibers):
                fo = s_rng.normal(0, 1.0) * params['spread'] * 0.002
                sx = mx + fo; sy = my + fo
                px = sx*(h*0.4) + w/2 + off_x
                py = sy*(h*0.4) + h/2 + off_y
                pts = np.stack([px, py], axis=1).astype(np.int32)
                
                thick = 1 if s_rng.rand() > 0.3 else 2
                avg_op = np.mean(opacity_curve)
                
                cv2.polylines(stroke_layer, [pts], False, base_col * 1.5 * avg_op, thickness=thick, lineType=cv2.LINE_AA)
                
                # TIP
                if params['mode'] == "Animated Loop" and len(pts) > 2:
                    tip_pt = pts[-1:] 
                    # Tip stays bright until the very end of the line
                    tip_bright = 1.0 if t_end < 0.99 else 0.0
                    if tip_bright > 0:
                        cv2.circle(stroke_layer, tuple(tip_pt[0]), thick+2, (1.0, 1.0, 1.0), -1)

        layers.append((z_depth, stroke_layer))

    # --- COMPOSITING ---
    for z, layer in layers:
        dist_from_focus = abs(z - 1.0)
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
    st.session_state['trail_len'] = meta.get('trail_len', 0.6)
    st.session_state['ar_select'] = meta['ar_name']

def delete_item(idx):
    st.session_state.history.pop(idx)
    st.rerun()

# --- SIDEBAR UI ---
st.title("🔦 Lights of Time")

with st.sidebar:
    st.header("1. Canvas & Mode")
    mode = st.radio("Output Mode", ["Still Light", "Animated Loop"], key="mode")
    ar_options = {"1:1 (Square)": 1.0, "9:16 (Story)": 9/16, "16:9 (Cinema)": 16/9, "4:3 (Classic)": 4/3}
    ar_name = st.selectbox("Aspect Ratio", list(ar_options.keys()), key="ar_select")
    ar_val = ar_options[ar_name]

    if mode == "Animated Loop":
        st.subheader("Animation Physics")
        # 0.6 to 0.8 is the sweet spot for a long, unraveling tail
        trail_len = st.slider("Trail Duration", 0.1, 1.0, 0.7, key="trail_len")
    else:
        trail_len = 1.0 

    st.divider()
    
    st.header("2. Art Direction")
    palette = st.selectbox("Color Palette", list(PALETTES.keys()), index=0, key="pal") 
    scale = st.slider("Zoom", 0.5, 1.5, 0.85, key="scale")
    num_strokes = st.number_input("Light Sources", 1, 30, 15, key="num_strokes") 

    st.divider()
    
    st.header("3. Brush Physics")
    complexity = st.slider("Gesture Complexity", 1, 4, 3, key="complexity") 
    width_scale = st.slider("Scatter Width", 0.1, 3.0, 1.5, key="width_scale")
    spread = st.slider("Cluster Spread", 0.1, 3.0, 1.0, key="spread")
    
    st.divider()
    
    st.header("4. Lens & Optics")
    exposure = st.slider("Exposure", 0.1, 2.0, 0.6, key="exposure")
    dof = st.slider("Depth of Field", 0.0, 2.0, 1.0, key="dof")
    glow = st.slider("Glow Intensity", 0.0, 3.0, 1.0, key="glow")
    aberration = st.slider("Prism Shift", 0.0, 10.0, 3.0, key="aberration")
    
    st.divider()
    seed = st.number_input("Seed", step=1, value=501, key="seed")
    gen = st.button("EXPOSE FILM", type="primary", use_container_width=True)

# --- MAIN ---
main = st.empty()

if gen:
    p = {
        "seed": seed, "complexity": complexity, "exposure": exposure,
        "glow": glow, "aberration": aberration, "spread": spread,
        "num_strokes": num_strokes, "scale": scale, "width_scale": width_scale,
        "mode": mode, "palette": palette, "dof": dof, "ar_val": ar_val, "ar_name": ar_name,
        "trail_len": trail_len
    }
    
    with main.container():
        bar = st.progress(0, text="Developing Exposure...")
        if mode == "Still Light":
            res = render_frame(p, 1.0)
            st.image(cv2.cvtColor(res.astype(np.uint8), cv2.COLOR_BGR2RGB), use_container_width=True)
            is_success, buffer = cv2.imencode(".png", res)
            data = buffer.tobytes()
            fmt = "png"
        else:
            frames = []
            total_frames = 90
            for i in range(total_frames):
                res = render_frame(p, i/total_frames)
                frames.append(cv2.cvtColor(res.astype(np.uint8), cv2.COLOR_BGR2RGB))
                bar.progress((i+1)/total_frames)
            b = io.BytesIO()
            imageio.mimsave(b, frames, format='GIF', fps=30, loop=0)
            data = b.getvalue()
            fmt = "gif"
            st.image(data, use_container_width=True)
            
        st.session_state.history.insert(0, {"data": data, "fmt": fmt, "meta": p, "time": time.strftime("%H:%M:%S")})

# --- GALLERY ---
if st.session_state.history:
    st.divider()
    st.subheader(f"Film Roll ({len(st.session_state.history)})")
    cols = st.columns(3)
    for idx, item in enumerate(st.session_state.history):
        with cols[idx % 3]:
            with st.container(border=True):
                st.image(item['data'], use_container_width=True)
                m = item['meta']
                st.caption(f"**{m['mode']}** • {m['ar_name']} • Seed: {m['seed']}")
                c1, c2, c3 = st.columns([1, 1, 1])
                with c1: st.download_button("💾", item['data'], f"light_{idx}.{item['fmt']}", key=f"dl_{idx}")
                with c2: 
                    if st.button("Apply", key=f"res_{idx}"): 
                        restore_settings(m)
                        st.rerun()
                with c3: 
                    if st.button("🗑️", key=f"del_{idx}"): 
                        delete_item(idx)
