import streamlit as st
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import imageio
import io
import zipfile
import time
import random

# PAGE CONFIG
st.set_page_config(page_title="Lights of Time Generator", page_icon="🔦", layout="wide")

# ADVANCED UI STYLING (DARK STUDIO)
st.markdown("""
    <style>
    .stApp { background-color: #050505 !important; color: #e0e0e0 !important; }
    [data-testid="stSidebar"] { background-color: #0f0f0f !important; border-right: 1px solid #222; }
    .metadata-card { 
        background-color: rgba(255, 255, 255, 0.03); 
        padding: 12px; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.08);
        font-size: 0.8rem; color: #aaa; margin-bottom: 10px;
    }
    .hero-container {
        border-radius: 15px; overflow: hidden; border: 1px solid #333;
        margin-top: 1rem; margin-bottom: 2rem; background-color: #000;
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        min-height: 400px;
    }
    h1, h2, h3 { color: #fff !important; font-weight: 700 !important; }
    div.stButton > button { background-color: #238636 !important; color: white !important; border-radius: 6px !important; border: none !important; }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if 'history' not in st.session_state: st.session_state.history = []

keys_defaults = {
    'render_mode_radio': "Still Light", 'seed_val': 42, 'complexity': 6,
    'exposure': 1.8, 'glow': 1.2, 'aberration': 2.5, 'line_width': 1.5
}
for k, v in keys_defaults.items():
    if k not in st.session_state: st.session_state[k] = v

# --- CALLBACKS ---
def callback_randomize():
    st.session_state['seed_val'] = random.randint(1, 999999)
    st.session_state['exposure'] = round(random.uniform(1.5, 3.0), 1)
    st.session_state['glow'] = round(random.uniform(0.8, 1.6), 2)
    st.session_state['aberration'] = round(random.uniform(1.0, 4.0), 2)
    st.toast("New light gestures discovered! 🔦")

def callback_restore(meta):
    st.session_state['render_mode_radio'] = meta["Mode"]
    st.session_state['seed_val'] = meta["Seed"]
    st.session_state['exposure'] = meta["Exp"]
    st.session_state['glow'] = meta["Glow"]
    st.session_state['aberration'] = meta["Aberr"]
    st.session_state['line_width'] = meta["Width"]
    st.session_state['complexity'] = meta["Complexity"]

def reset_app():
    st.session_state.history = []
    st.rerun()

# --- PHOTOGRAPHIC LIGHT ENGINE ---

def generate_gesture_path(complexity, seed, prog):
    """Creates organic, non-circular light gestures."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 1, 1000)
    x, y = np.zeros_like(t), np.zeros_like(t)
    
    # Use multiple frequencies to create a 'wandering' path
    for i in range(1, complexity + 1):
        freq = rng.uniform(0.5, 2.5) * i
        amp = rng.uniform(0.2, 1.0) / (i**0.6)
        phase = rng.uniform(0, 2*np.pi) + (prog * 2 * np.pi)
        
        x += amp * np.cos(freq * t * 2 * np.pi + phase)
        y += amp * np.sin(freq * t * 1.5 * np.pi + phase * 0.5)
        
    return x, y

def render_light_frame(complexity, seed, exposure, glow, aberration, line_width, prog):
    w, h = 1080, 1080
    # Canvas uses float32 to handle additive 'overexposure' math
    canvas = np.zeros((h, w, 3), dtype=np.float32)
    
    num_strands = 25  # A bundle of filaments
    
    for s_idx in range(num_strands):
        # Slightly vary each strand for a natural look
        s_seed = seed + (s_idx * 100)
        s_exposure = exposure * random.uniform(0.4, 1.0)
        s_width = max(1, int(line_width * random.uniform(0.5, 2.0)))
        
        x, y = generate_gesture_path(complexity, s_seed, prog)
        
        # Chromatic aberration colors (more natural white core with colored fringes)
        colors = [
            (1.0, 0.2, 0.1), # Warm Red
            (0.1, 1.0, 0.4), # Electric Green
            (0.1, 0.4, 1.0)  # Deep Blue
        ]
        
        # Prism offsets
        offsets = [aberration * -1.5, 0, aberration * 1.5]
        
        for color, offset in zip(colors, offsets):
            # Scale coordinates to frame
            pts = np.stack([
                x * 400 + (w/2) + offset, 
                y * 400 + (h/2)
            ], axis=1).astype(np.int32)
            
            # Additive blending: canvas += color
            cv2.polylines(canvas, [pts], False, np.array(color) * s_exposure, 
                          thickness=s_width, lineType=cv2.LINE_AA)
    
    # Apply high-end Photographic Glow (Multi-pass Blur)
    if glow > 0:
        canvas += gaussian_filter(canvas, sigma=glow * 3) * 0.5
        canvas += gaussian_filter(canvas, sigma=glow * 10) * 0.3
    
    # Tone mapping: bring HDR float values back to 0-255
    canvas = np.clip(canvas * 255, 0, 255).astype(np.uint8)
    return canvas

# --- MAIN PAGE ---
st.title("🔦 Lights of Time Generator")

with st.expander("📖 Studio Quick Start"):
    st.markdown("""
    - **Complexity:** Higher values create more random, chaotic 'gestures'.
    - **Chromatic Aberration:** Controls the 'prism' split on the edges of light trails.
    - **Exposure:** Intensity of the core light filaments.
    - **Neon Glow:** Creates atmospheric falloff.
    """)

preview_placeholder = st.empty()

# SIDEBAR
with st.sidebar:
    st.header("Studio Controls")
    mode = st.radio("Output Type", ["Still Light", "Animated Loop"], key="render_mode_radio")
    
    complexity = st.slider("Gesture Complexity", 2, 12, key="complexity", help="Complexity of the light Source movement.")
    seed = st.number_input("Seed", step=1, key="seed_val")
    
    with st.expander("Optics Styling", expanded=True):
        st.button("🎲 Surprise Me!", on_click=callback_randomize, use_container_width=True)
        st.divider()
        exposure = st.slider("Exposure", 0.5, 5.0, key="exposure")
        glow = st.slider("Neon Glow", 0.0, 3.0, key="glow")
        aberration = st.slider("Chromatic Aberration", 0.0, 10.0, key="aberration")
        line_width = st.slider("Core Line Weight", 0.5, 5.0, key="line_width")

    st.divider()
    gen_btn = st.button("EXECUTE LIGHT RENDER", type="primary", use_container_width=True)
    st.button("Clear Studio", on_click=reset_app, use_container_width=True)

# --- RENDER EXECUTION ---
if gen_btn:
    with preview_placeholder.container():
        st.markdown('<div class="hero-container">', unsafe_allow_html=True)
        if mode == "Still Light":
            with st.spinner("Exposing Film..."):
                img_array = render_light_frame(complexity, seed, exposure, glow, aberration, line_width, 0)
                st.image(img_array, use_container_width=True)
                
                is_success, buffer = cv2.imencode(".png", cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
                data = buffer.tobytes()
                fmt = "png"
        else:
            frames = []
            bar = st.progress(0, text="Capturing Light Trails...")
            for i in range(60):
                f = render_light_frame(complexity, seed, exposure, glow, aberration, line_width, i/60)
                frames.append(f)
                bar.progress((i+1)/60)
            
            b = io.BytesIO()
            imageio.mimsave(b, frames, format='GIF', fps=30, loop=0)
            data, fmt = b.getvalue(), "gif"
            st.image(data, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        meta = {'Complexity': complexity, 'Seed': seed, 'Exp': exposure, 'Glow': glow, 'Aberr': aberration, 'Width': line_width, 'Mode': mode}
        st.session_state.history.insert(0, {"data": data, "fmt": fmt, "time": time.strftime("%H:%M:%S"), "meta": meta})
        st.rerun()

# HERO PREVIEW
elif st.session_state.history:
    latest = st.session_state.history[0]
    with preview_placeholder.container():
        st.markdown('<div class="hero-container">', unsafe_allow_html=True)
        st.image(latest['data'], use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# GALLERY
if st.session_state.history:
    st.divider()
    g_col1, g_col2 = st.columns([3, 1])
    with g_col1: st.subheader("Light Gallery")
    with g_col2:
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            for i, item in enumerate(st.session_state.history):
                zf.writestr(f"light_{i}.{item['fmt']}", item['data'])
        st.download_button("📦 DOWNLOAD ZIP", data=zip_buf.getvalue(), file_name="lights_of_time.zip", use_container_width=True)

    cols = st.columns(3)
    for idx, item in enumerate(st.session_state.history):
        with cols[idx % 3]:
            st.image(item['data'], use_container_width=True)
            m = item['meta']
            st.markdown(f"""<div class="metadata-card"><b>{m['Mode']}</b> • {item['time']}<br>Seed: {m['Seed']} | Exp: {m['Exp']} | Aberr: {m['Aberr']}</div>""", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1: st.download_button("💾", item['data'], f"light_{idx}.{item['fmt']}", key=f"dl_{idx}")
            with c2: st.button("🔄", key=f"res_{idx}", on_click=callback_restore, args=(m,))
            with c3: st.button("🗑️", key=f"del_{idx}", on_click=lambda i=idx: (st.session_state.history.pop(i), st.rerun()))
