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

# PRO STUDIO STYLING
st.markdown("""
    <style>
    .stApp { background-color: #020202 !important; color: #e0e0e0 !important; }
    [data-testid="stSidebar"] { background-color: #080808 !important; border-right: 1px solid #1a1a1a; }
    .metadata-card { 
        background-color: rgba(255, 255, 255, 0.02); 
        padding: 12px; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.05);
        font-size: 0.8rem; color: #888; margin-top: 8px;
    }
    .hero-container {
        border-radius: 12px; overflow: hidden; border: 1px solid #222;
        margin-top: 1rem; margin-bottom: 2rem; background-color: #000;
        display: flex; align-items: center; justify-content: center;
    }
    h1, h2, h3 { color: #fff !important; letter-spacing: -1px; }
    div.stButton > button { background-color: #1a1a1a !important; color: #fff !important; border: 1px solid #333 !important; }
    div.stButton > button:hover { border-color: #555 !important; background-color: #222 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if 'history' not in st.session_state: st.session_state.history = []

# --- CALLBACKS (Fixed Logic) ---
def callback_randomize():
    st.session_state['seed_val'] = random.randint(1, 999999)
    st.session_state['exposure'] = round(random.uniform(2.5, 4.5), 1)
    st.session_state['aberration'] = round(random.uniform(5.0, 15.0), 1)
    st.toast("Optics shifted! 🔦")

def callback_restore(meta):
    st.session_state['render_mode_radio'] = meta["Mode"]
    st.session_state['seed_val'] = meta["Seed"]
    st.session_state['exposure'] = meta["Exp"]
    st.session_state['complexity_slider'] = meta["Complexity"]
    st.session_state['strand_count_slider'] = meta.get("Strands", 25)
    st.session_state['blur_slider'] = meta.get("Blur", 0.8)
    st.session_state['aberration'] = meta["Aberr"]

def reset_app():
    st.session_state.history = []
    st.rerun()

# --- THE PHOTOGRAPHIC ENGINE ---

def get_path(complexity, seed, prog, b_idx):
    """Generates organic gestural arcs with micro-variations per strand."""
    rng = np.random.RandomState(seed + b_idx)
    t = np.linspace(0, 1, 2000)
    x, y = np.zeros_like(t), np.zeros_like(t)
    
    for i in range(1, complexity + 1):
        freq = rng.uniform(0.1, 0.5) * i
        amp = rng.uniform(0.7, 1.4) / (i**0.9)
        phase = rng.uniform(0, 2*np.pi) + (prog * 2 * np.pi)
        drift = b_idx * 0.003
        x += amp * np.cos(freq * t * 2 * np.pi + phase + drift)
        y += amp * np.sin(freq * t * 1.8 * np.pi + phase * 0.4 + drift)
    return x, y

def render_pro_light(complexity, strands, seed, exposure, glow, aberration, weight, blur, prog):
    w, h = 1080, 1080
    canvas = np.zeros((h, w, 3), dtype=np.float32)
    
    for s_idx in range(strands):
        x, y = get_path(complexity, seed, prog, s_idx)
        
        # Base spectral primaries
        colors = [np.array([1.0, 0.1, 0.05]), np.array([0.1, 1.0, 0.2]), np.array([0.05, 0.2, 1.0])]
        offsets = [aberration * -1.5, 0, aberration * 1.5]
        
        strand_energy = exposure * np.random.RandomState(seed + s_idx).uniform(0.4, 1.0)
        
        for color_arr, offset in zip(colors, offsets):
            pts = np.stack([x * 380 + (w/2) + offset, y * 380 + (h/2)], axis=1).astype(np.int32)
            
            # Helper to convert numpy color math to a standard float tuple for OpenCV
            def to_cv_color(arr, intensity):
                scaled = arr * intensity
                return (float(scaled[0]), float(scaled[1]), float(scaled[2]))

            # Layer 1: Atmospheric Envelope
            cv2.polylines(canvas, [pts], False, to_cv_color(color_arr, strand_energy * 0.03), 
                          thickness=int(weight * 25), lineType=cv2.LINE_AA)
            
            # Layer 2: Prismatic Diffused Light
            cv2.polylines(canvas, [pts], False, to_cv_color(color_arr, strand_energy * 0.2), 
                          thickness=int(weight * 6), lineType=cv2.LINE_AA)
            
            # Layer 3: Incandescent Filament (White-hot core)
            # We add all channels to ensure the center is white
            white_val = float(strand_energy * 0.85)
            cv2.polylines(canvas, [pts], False, (white_val, white_val, white_val), 
                          thickness=int(weight), lineType=cv2.LINE_AA)

    if glow > 0:
        canvas += gaussian_filter(canvas, sigma=glow * 3) * 0.4
        canvas += gaussian_filter(canvas, sigma=glow * 10) * 0.2
    
    if blur > 0:
        canvas = gaussian_filter(canvas, sigma=blur)
        
    return np.clip(canvas * 255, 0, 255).astype(np.uint8)

# --- UI ---
st.title("🔦 Lights of Time Generator")

with st.sidebar:
    st.header("Camera Settings")
    mode = st.radio("Output Type", ["Still Light", "Animated Loop"], key="render_mode_radio")
    
    complexity = st.slider("Gesture Complexity", 1, 6, key="complexity_slider")
    strand_count = st.slider("Bundle Strands", 1, 50, key="strand_count_slider")
    seed = st.number_input("Seed", step=1, key="seed_val")
    
    with st.expander("Optics & Film", expanded=True):
        st.button("🎲 Surprise Me!", on_click=callback_randomize, use_container_width=True)
        st.divider()
        exposure = st.slider("Luminosity / Exp", 0.5, 8.0, key="exposure")
        aberration = st.slider("Prism Diffraction", 0.0, 20.0, key="aberration")
        glow = st.slider("Atmospheric Bloom", 0.0, 4.0, value=1.5)
        blur_val = st.slider("Lens Blur", 0.0, 5.0, key="blur_slider")
        line_weight = st.slider("Filament Weight", 0.5, 5.0, value=1.0)

    st.divider()
    gen_btn = st.button("EXECUTE EXPOSURE", type="primary", use_container_width=True)
    if st.button("Reset Studio", use_container_width=True): reset_app()

preview_area = st.empty()

# --- EXECUTION ---
if gen_btn:
    with preview_area.container():
        bar = st.progress(0, text="Developing Exposure...")
        if mode == "Still Light":
            img = render_pro_light(complexity, strand_count, seed, exposure, glow, aberration, line_weight, blur_val, 0)
            is_success, buffer = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            data, fmt = buffer.tobytes(), "png"
        else:
            frames = []
            for i in range(60):
                f = render_pro_light(complexity, strand_count, seed, exposure, glow, aberration, line_weight, blur_val, i/60)
                frames.append(f)
                bar.progress((i+1)/60)
            b = io.BytesIO()
            imageio.mimsave(b, frames, format='GIF', fps=30, loop=0)
            data, fmt = b.getvalue(), "gif"
        
        meta = {'Complexity': complexity, 'Strands': strand_count, 'Seed': seed, 'Exp': exposure, 'Glow': glow, 'Aberr': aberration, 'Blur': blur_val, 'Width': line_weight, 'Mode': mode}
        st.session_state.history.insert(0, {"data": data, "fmt": fmt, "time": time.strftime("%H:%M:%S"), "meta": meta})
        st.rerun()

elif st.session_state.history:
    latest = st.session_state.history[0]
    with preview_area.container():
        st.markdown('<div class="hero-container">', unsafe_allow_html=True)
        st.image(latest['data'], use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# GALLERY (Displays after the Hero section)
if st.session_state.history:
    st.divider()
    st.subheader("Light Gallery")
    cols = st.columns(3)
    for idx, item in enumerate(st.session_state.history):
        with cols[idx % 3]:
            st.image(item['data'], use_container_width=True)
            m = item['meta']
            st.markdown(f"""<div class="metadata-card"><b>{m['Mode']}</b> • {item['time']}<br>Seed: {m['Seed']} | Aberr: {m['Aberr']}<br>Exp: {m['Exp']} | Strands: {m.get('Strands', 25)}</div>""", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1: st.download_button("💾", item['data'], f"light_{idx}.{item['fmt']}", key=f"dl_{idx}")
            with c2: st.button("🔄", key=f"res_{idx}", on_click=callback_restore, args=(m,))
            with c3:
                if st.button("🗑️", key=f"del_{idx}"):
                    st.session_state.history.pop(idx)
                    st.rerun()
