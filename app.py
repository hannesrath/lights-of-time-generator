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
        border-radius: 12px; overflow: hidden; border: 1px solid #333;
        margin-top: 1rem; margin-bottom: 2rem; background-color: #000;
        display: flex; align-items: center; justify-content: center;
    }
    div.stButton > button { background-color: #222 !important; color: #fff !important; border: 1px solid #444 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION STATE ---
if 'history' not in st.session_state: st.session_state.history = []

# --- MATH ENGINE ---
def get_ribbon_spine(complexity, seed, prog):
    """Generates the master path for the ribbon."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 1, 4000) # High point count for smooth curves
    x, y = np.zeros_like(t), np.zeros_like(t)
    
    # Jaime Gorospe style: Low frequency sweeping arcs
    for i in range(1, complexity + 1):
        freq = rng.uniform(0.1, 0.4) * i
        amp = rng.uniform(0.8, 1.5) / (i**0.6)
        phase = rng.uniform(0, 2*np.pi) + (prog * 2 * np.pi)
        
        x += amp * np.cos(freq * t * 2 * np.pi + phase)
        y += amp * np.sin(freq * t * 1.5 * np.pi + phase * 0.5)
        
    return x, y, t

def render_ribbon_tool(complexity, seed, exposure, glow, aberration, tool_width, twist_speed, blur, prog):
    w, h = 1080, 1080
    # Canvas is float32 to accumulate light, but we will draw with 0-255 values
    canvas = np.zeros((h, w, 3), dtype=np.float32)
    
    # 1. GENERATE MASTER SPINE
    mx, my, t_vals = get_ribbon_spine(complexity, seed, prog)
    
    # 2. DEFINE THE PHYSICAL TOOL (The "Brush")
    # This defines the stripes seen in the reference: Amber body, White core, Blue edge.
    # Colors are BGR (Blue, Green, Red) for OpenCV
    emitters = [
        # Wide Amber Body
        {'offset': 0.0, 'color': (5, 120, 255), 'width': 30, 'alpha': 0.6}, 
        # Bright White Core
        {'offset': 0.1, 'color': (200, 255, 255), 'width': 8, 'alpha': 1.0},
        # Cyan Edge Detail
        {'offset': 0.6, 'color': (255, 200, 0), 'width': 4, 'alpha': 0.9},
        # Deep Blue Under-glow
        {'offset': -0.5, 'color': (255, 50, 0), 'width': 4, 'alpha': 0.8}
    ]
    
    # 3. CALCULATE TWIST (Simulates hand rotation)
    # The ribbon gets wider and narrower as it twists along the path
    twist = np.sin(t_vals * np.pi * twist_speed + (prog*np.pi)) 
    
    for emitter in emitters:
        # Calculate the parallel offset based on the twist
        # When twist is 0, the ribbon is "sideways" (thin). When 1, it's "flat" (wide).
        current_offset = emitter['offset'] * tool_width * 20.0 * twist
        
        # Apply offset to master path
        ex = mx * 380 + (w/2) + current_offset
        ey = my * 380 + (h/2) + (current_offset * 0.5) # Slight skew for 3D feel
        
        pts = np.stack([ex, ey], axis=1).astype(np.int32)
        
        # COLOR MATH: Scale 0-255 by exposure
        base_color = np.array(emitter['color']) * exposure * emitter['alpha']
        
        # Draw the solid core of the line
        cv2.polylines(canvas, [pts], False, base_color, 
                      thickness=max(1, int(emitter['width'] * abs(twist) * tool_width + 1)), 
                      lineType=cv2.LINE_AA)
        
        # Draw the atmospheric glow (Bloom)
        if glow > 0:
            cv2.polylines(canvas, [pts], False, base_color * 0.3, 
                          thickness=int(emitter['width'] * 4 * tool_width), 
                          lineType=cv2.LINE_AA)

    # 4. POST-PROCESSING
    
    # Chromatic Aberration (Shift Red/Blue channels)
    if aberration > 0:
        shift = int(aberration)
        if shift > 0:
            # Shift Blue channel left, Red channel right
            canvas_copy = canvas.copy()
            canvas[:, :-shift, 0] = canvas_copy[:, shift:, 0] # Blue
            canvas[:, shift:, 2] = canvas_copy[:, :-shift, 2] # Red

    # Lens Blur
    if blur > 0:
        canvas = gaussian_filter(canvas, sigma=blur)
        
    # Tone Mapping: Clamp values to 255 (White)
    # This ensures "overexposed" areas become pure white instead of wrapping around to black
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    
    return canvas

# --- UI ---
st.title("🔦 Lights of Time: Ribbon Studio")

with st.sidebar:
    st.header("Tool Configuration")
    mode = st.radio("Mode", ["Still Light", "Animated Loop"], key="mode")
    complexity = st.slider("Gesture Complexity", 1, 4, value=2, key="comp")
    seed = st.number_input("Seed", step=1, value=555, key="seed")
    
    with st.expander("Light Physics", expanded=True):
        exposure = st.slider("Brightness", 0.5, 3.0, value=1.2, help="Overall light intensity.")
        tool_width = st.slider("Ribbon Width", 0.5, 3.0, value=1.0, help="Space between the colored stripes.")
        twist_speed = st.slider("Twist Rate", 1.0, 10.0, value=3.0, help="How fast the tool rotates.")
        aberration = st.slider("Prism Shift", 0.0, 10.0, value=4.0)
        glow = st.slider("Atmosphere", 0.0, 2.0, value=1.0)
        blur = st.slider("Softness", 0.0, 3.0, value=0.5)

    gen_btn = st.button("PAINT RIBBON", type="primary", use_container_width=True)

preview = st.empty()

if gen_btn:
    with preview.container():
        bar = st.progress(0, text="Exposing Sensor...")
        if mode == "Still Light":
            # Generate one frame
            img_bgr = render_ribbon_tool(complexity, seed, exposure, glow, aberration, tool_width, twist_speed, blur, 0)
            # Convert BGR to RGB for Streamlit display
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, use_container_width=True)
            
            # Encode PNG for history
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

# GALLERY
if st.session_state.history:
    st.divider()
    cols = st.columns(4)
    for idx, item in enumerate(st.session_state.history):
        with cols[idx % 4]:
            st.image(item, use_container_width=True) 
            st.download_button("💾", item, f"ribbon_{idx}.png", key=f"dl_{idx}")
