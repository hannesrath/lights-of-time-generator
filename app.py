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
    .hero-container {
        border-radius: 0px; overflow: hidden; border: 1px solid #333;
        margin-top: 1rem; margin-bottom: 2rem; background-color: #000;
        display: flex; align-items: center; justify-content: center;
    }
    div.stButton > button { background-color: #222 !important; color: #fff !important; border: 1px solid #444 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION STATE ---
if 'history' not in st.session_state: st.session_state.history = []

# --- RIBBON PATH MATH ---
def get_ribbon_path(complexity, seed, prog):
    """Generates the 'Master Spine' of the ribbon."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 1, 3000)
    x, y = np.zeros_like(t), np.zeros_like(t)
    
    # Low frequency, high amplitude for "Sweeping" gestures
    for i in range(1, complexity + 1):
        freq = rng.uniform(0.1, 0.4) * i
        amp = rng.uniform(0.8, 1.4) / (i**0.6)
        phase = rng.uniform(0, 2*np.pi) + (prog * 2 * np.pi)
        
        x += amp * np.cos(freq * t * 2 * np.pi + phase)
        y += amp * np.sin(freq * t * 1.5 * np.pi + phase * 0.5)
        
    return x, y

def render_ribbon_frame(complexity, seed, exposure, glow, aberration, tool_width, blur, prog):
    w, h = 1080, 1080
    # FLOAT32 is critical for bright light accumulation
    canvas = np.zeros((h, w, 3), dtype=np.float32)
    
    # --- THE "TOOL HEAD" DEFINITION ---
    # We define the physical tool as a list of emitters attached to the master spine.
    # Structure: (Offset_Multiplier, Color_RGB, Thickness_Px, Intensity_Mult)
    
    # This "Gorospe Tool" has a wide Warm strip, a bright White core, and a Cool edge.
    tool_emitters = [
        # 1. The Wide Amber/Gold Backing (The "Ribbon Body")
        {'offset': 0.0, 'color': np.array([1.0, 0.6, 0.05]), 'width': 25.0, 'int': 0.8},
        
        # 2. The Bright White Core (The "Hot filament")
        {'offset': 0.0, 'color': np.array([1.0, 0.9, 0.8]), 'width': 6.0, 'int': 2.5},
        
        # 3. The Cyan Edge (Parallel detail line)
        {'offset': 0.6, 'color': np.array([0.0, 0.8, 1.0]), 'width': 4.0, 'int': 1.5},
        
        # 4. The Thin Blue Guard (Parallel detail line on other side)
        {'offset': -0.5, 'color': np.array([0.1, 0.2, 1.0]), 'width': 3.0, 'int': 1.2}
    ]

    master_x, master_y = get_ribbon_path(complexity, seed, prog)

    for emitter in tool_emitters:
        # Calculate Parallel Offset
        # We simulate ribbon width by offsetting X/Y slightly based on the emitter's position on the tool
        # Ideally we'd use normals, but a simple linear offset creates the cool "twisting" artifacts seen in light painting
        
        off_x = emitter['offset'] * tool_width * 15.0
        off_y = emitter['offset'] * tool_width * 15.0 * 0.5 # Isometric skew
        
        path_x = master_x * 380 + (w/2) + off_x
        path_y = master_y * 380 + (h/2) + off_y
        
        points = np.stack([path_x, path_y], axis=1).astype(np.int32)
        
        # Base Color Calculation
        base_color = emitter['color'] * emitter['int'] * exposure
        
        # 1. DRAW THE MAIN BODY (Solid Light)
        # Convert to tuple for OpenCV
        color_tuple = (float(base_color[2]), float(base_color[1]), float(base_color[0])) # BGR order for CV2
        
        # We draw multiple times with slight jitter to create "Texture"
        cv2.polylines(canvas, [points], False, color_tuple, 
                      thickness=int(emitter['width'] * tool_width), lineType=cv2.LINE_AA)
        
        # 2. DRAW THE GLOW (Atmosphere)
        if glow > 0:
            glow_color = (float(base_color[2]*0.2), float(base_color[1]*0.2), float(base_color[0]*0.2))
            cv2.polylines(canvas, [points], False, glow_color, 
                          thickness=int(emitter['width'] * tool_width * 4), lineType=cv2.LINE_AA)

    # --- CHROMATIC ABERRATION (Global Lens Effect) ---
    if aberration > 0:
        # Shift Red and Blue channels
        shift = int(aberration)
        if shift > 0:
            # Shift Red Channel left
            canvas[:, :-shift, 2] = canvas[:, shift:, 2]
            # Shift Blue Channel right
            canvas[:, shift:, 0] = canvas[:, :-shift, 0]

    # --- FINAL OPTICS ---
    if blur > 0:
        canvas = gaussian_filter(canvas, sigma=blur)
        
    # Tone Map: Compress the HDR values to visible range (0-255)
    # This prevents the "Dark Image" issue by normalizing bright spots
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    return canvas

# --- UI ---
st.title("🔦 Lights of Time: Ribbon Engine")

with st.sidebar:
    st.header("Tool Settings")
    mode = st.radio("Output", ["Still Light", "Animated Loop"], key="mode")
    complexity = st.slider("Gesture Complexity", 1, 4, value=2, key="comp")
    seed = st.number_input("Seed", step=1, value=1024, key="seed")
    
    st.divider()
    
    exposure = st.slider("Light Intensity", 1.0, 5.0, value=2.0, help="Brightness of the tool.")
    tool_width = st.slider("Ribbon Width", 0.5, 4.0, value=1.5, help="Physical width of the light tool.")
    glow = st.slider("Atmosphere", 0.0, 2.0, value=1.0)
    aberration = st.slider("Prism Edge", 0.0, 20.0, value=5.0)
    blur_val = st.slider("Lens Softness", 0.0, 5.0, value=0.5)
    
    gen_btn = st.button("PAINT LIGHT", type="primary", use_container_width=True)

preview = st.empty()

if gen_btn:
    with preview.container():
        bar = st.progress(0, text="Painting Ribbons...")
        if mode == "Still Light":
            img = render_ribbon_frame(complexity, seed, exposure, glow, aberration, tool_width, blur_val, 0)
            # OpenCV uses BGR, Streamlit uses RGB. We generated BGR in engine for CV2 compatibility.
            # So we convert BGR -> RGB for display
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(rgb_img, use_container_width=True)
            
            # Save format
            is_success, buffer = cv2.imencode(".png", img) 
            data = buffer.tobytes()
        else:
            frames = []
            for i in range(60):
                f = render_ribbon_frame(complexity, seed, exposure, glow, aberration, tool_width, blur_val, i/60)
                frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                bar.progress((i+1)/60)
            b = io.BytesIO()
            imageio.mimsave(b, frames, format='GIF', fps=30, loop=0)
            data = b.getvalue()
            st.image(data, use_container_width=True)
            
        st.session_state.history.insert(0, data)
