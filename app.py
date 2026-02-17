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

def get_ribbon_spine(complexity, seed, prog):
    """Generates the spine path and the normal vectors for width calculation."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 1, 1500) # Reduced count slightly for loop performance
    x, y = np.zeros_like(t), np.zeros_like(t)
    
    # Generate Low-Frequency Arcs
    for i in range(1, complexity + 1):
        freq = rng.uniform(0.1, 0.4) * i
        amp = rng.uniform(0.8, 1.4) / (i**0.6)
        phase = rng.uniform(0, 2*np.pi) + (prog * 2 * np.pi)
        
        x += amp * np.cos(freq * t * 2 * np.pi + phase)
        y += amp * np.sin(freq * t * 1.5 * np.pi + phase * 0.5)
    
    # Scale to canvas roughly
    x = x * 380 + 540 # 1080/2
    y = y * 380 + 540
    
    # Calculate Normal Vectors (for expanding the ribbon width)
    dx = np.gradient(x)
    dy = np.gradient(y)
    dist = np.sqrt(dx**2 + dy**2) + 1e-6
    nx = -dy / dist # Perpendicular X
    ny = dx / dist  # Perpendicular Y
    
    return x, y, nx, ny, t

def render_ribbon_tool(complexity, seed, exposure, glow, aberration, tool_width, twist_speed, blur, prog):
    w, h = 1080, 1080
    canvas = np.zeros((h, w, 3), dtype=np.float32)
    
    # 1. GET GEOMETRY
    mx, my, nx, ny, t_vals = get_ribbon_spine(complexity, seed, prog)
    
    # 2. DEFINE EMITTERS (Amber Body, White Core, Blue Edge)
    # Alpha controls opacity, Width controls thickness
    emitters = [
        {'offset': 0.0, 'color': (5, 120, 255), 'width': 25.0, 'alpha': 0.4},  # Wide Amber
        {'offset': 0.0, 'color': (200, 255, 255), 'width': 5.0, 'alpha': 1.2}, # White Core
        {'offset': 0.6, 'color': (255, 200, 0), 'width': 3.0, 'alpha': 0.9},   # Cyan Edge
        {'offset': -0.5, 'color': (50, 50, 255), 'width': 3.0, 'alpha': 0.8}   # Red/Blue Edge
    ]
    
    # 3. CALCULATE TWIST
    # Twist factor: 1.0 = Flat (Wide), 0.0 = Sideways (Thin)
    twist = np.sin(t_vals * np.pi * twist_speed + (prog * np.pi))
    
    for em in emitters:
        # A. Calculate the Center Path of this specific emitter
        # We offset it from the master spine based on the twist
        # "offset" shifts it left/right relative to the ribbon center
        current_offset = em['offset'] * tool_width * 15.0 * twist
        
        cx = mx + nx * current_offset
        cy = my + ny * current_offset
        
        # B. Calculate the Width at every point
        # The physical width of this stripe varies by twist
        current_width = np.abs(twist) * em['width'] * tool_width + 1.0
        
        # C. Construct the Polygon Strip (Left side and Right side)
        # We expand the center path outward along the normal
        lx = cx + nx * (current_width / 2.0)
        ly = cy + ny * (current_width / 2.0)
        rx = cx - nx * (current_width / 2.0)
        ry = cy - ny * (current_width / 2.0)
        
        # D. Combine into a filled polygon array [Left_Points ... Right_Points_Reversed]
        # This creates one continuous shape for the ribbon
        pts_left = np.stack([lx, ly], axis=1)
        pts_right = np.stack([rx, ry], axis=1)[::-1] # Reverse right side to close the loop
        poly = np.concatenate([pts_left, pts_right]).astype(np.int32)
        
        # E. DRAW
        base_color = np.array(em['color']) * exposure * em['alpha']
        
        # Fill the core shape
        cv2.fillPoly(canvas, [poly], base_color, lineType=cv2.LINE_AA)
        
        # Add Glow (draw slightly wider, simpler line for bloom)
        if glow > 0:
            # For glow, we can just use polylines on the center path since it's blurry anyway
            pts_center = np.stack([cx, cy], axis=1).astype(np.int32)
            cv2.polylines(canvas, [pts_center], False, base_color * 0.2, 
                          thickness=int(em['width'] * tool_width * 4), lineType=cv2.LINE_AA)

    # 4. POST-PROCESSING
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
st.title("🔦 Lights of Time: Ribbon Studio")

with st.sidebar:
    st.header("Tool Configuration")
    mode = st.radio("Mode", ["Still Light", "Animated Loop"], key="mode")
    complexity = st.slider("Gesture Complexity", 1, 4, value=2, key="comp")
    seed = st.number_input("Seed", step=1, value=555, key="seed")
    
    st.divider()
    
    exposure = st.slider("Brightness", 0.5, 3.0, value=1.5)
    tool_width = st.slider("Ribbon Width", 0.5, 3.0, value=1.5)
    twist_speed = st.slider("Twist Rate", 1.0, 10.0, value=4.0)
    aberration = st.slider("Prism Edge", 0.0, 20.0, value=6.0)
    glow = st.slider("Atmosphere", 0.0, 2.0, value=0.8)
    blur = st.slider("Softness", 0.0, 3.0, value=0.6)
    
    gen_btn = st.button("PAINT RIBBON", type="primary", use_container_width=True)

preview = st.empty()

if gen_btn:
    with preview.container():
        bar = st.progress(0, text="Painting Ribbons...")
        if mode == "Still Light":
            img_bgr = render_ribbon_tool(complexity, seed, exposure, glow, aberration, tool_width, twist_speed, blur, 0)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, use_container_width=True)
            is_success, buffer = cv2.imencode(".png", img_bgr)
            data = buffer.tobytes()
        else:
            frames = []
            for i in range(60):
                f_bgr = render_ribbon_tool(complexity, seed, exposure, glow, aberration, tool_width, twist_speed, blur, i/60)
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
            st.download_button("💾", item, f"ribbon_{idx}.png", key=f"dl_{idx}")
