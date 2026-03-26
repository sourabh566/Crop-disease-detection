import streamlit as st
from PIL import Image
import numpy as np
import time
import os
from config import DISEASE_DB, APP_TITLE, THEME_COLOR, CONFIDENCE_THRESHOLD, SAVED_MODEL_PATH
from utils import preprocess_image
from model import load_disease_model, predict_disease

# ─── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(page_title=APP_TITLE, page_icon="🌿", layout="wide")

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
    .main {{ background-color: #f4f6f0; }}
    .result-card {{
        padding: 20px;
        border-radius: 12px;
        background-color: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-left: 6px solid {THEME_COLOR};
        margin-bottom: 20px;
    }}
    .confidence-high {{ color: #2E7D32; font-weight: bold; font-size: 1.1em; }}
    .confidence-low  {{ color: #C62828; font-weight: bold; font-size: 1.1em; }}
    .confidence-mid  {{ color: #E65100; font-weight: bold; font-size: 1.1em; }}
    .badge-real {{ background:#1B5E20; color:white; border-radius:6px; padding:2px 10px; font-size:0.8em; }}
    .badge-fallback {{ background:#E65100; color:white; border-radius:6px; padding:2px 10px; font-size:0.8em; }}
    </style>
    """, unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/leaf.png", width=80)
    st.title("AgroGuard AI")
    st.info("Identify crop diseases instantly using Deep Learning. Designed for farmers and agricultural experts.")
    st.divider()
    st.subheader("How it works")
    st.write("1. Upload or snap a photo of a leaf.")
    st.write("2. AI analyzes the visual patterns.")
    st.write("3. Get diagnosis and treatment advice.")
    st.divider()
    trained_exists = os.path.exists(SAVED_MODEL_PATH)
    if trained_exists:
        st.success("✅ Using trained PlantVillage model")
    else:
        st.warning("⚠️ Run `python train.py` to enable the deep learning model.")
        st.info("Currently using color-feature analysis as fallback.")

# ─── Header ───────────────────────────────────────────────────────────────────
st.title(f"🌿 {APP_TITLE}")
st.write("Ensuring a sustainable future through intelligent farming.")

# ─── Model Loading (Cached) ───────────────────────────────────────────────────
@st.cache_resource
def get_model():
    return load_disease_model()

model_tuple = get_model()
model_mode = model_tuple[0]

# ─── Input Tabs ───────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📤 Upload Image", "📸 Take Photo"])

uploaded_file = None

with tab1:
    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"])

with tab2:
    camera_file = st.camera_input("Capture leaf image")
    if camera_file:
        uploaded_file = camera_file

# ─── Analysis ─────────────────────────────────────────────────────────────────
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input Image")
        st.image(image, use_container_width=True, caption="Original Leaf Image")

    with col2:
        st.subheader("Analysis Progress")
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Step 1: Preprocess
        status_text.text("🔬 Preprocessing image (noise reduction & normalization)...")
        processed_image = preprocess_image(image)
        time.sleep(0.4)
        progress_bar.progress(40)

        # Step 2: Inference
        status_text.text("🤖 Running AI diagnostics...")
        predicted_class, confidence = predict_disease(model_tuple, processed_image)
        time.sleep(0.4)
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")

        # ─── Results ──────────────────────────────────────────────────────────
        st.divider()

        # Model mode badge
        if model_mode == "nn":
            st.markdown('<span class="badge-real">🧠 PlantVillage Model</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-fallback">🎨 Color-Feature Fallback</span>', unsafe_allow_html=True)

        # Confidence warning
        if confidence < CONFIDENCE_THRESHOLD:
            st.warning("⚠️ Low confidence. Please use a clearer, well-lit image.")

        disease_info = DISEASE_DB.get(predicted_class, DISEASE_DB["Tomato___healthy"])

        # Confidence color class
        if confidence >= 0.75:
            conf_class = "confidence-high"
        elif confidence >= 0.5:
            conf_class = "confidence-mid"
        else:
            conf_class = "confidence-low"

        st.markdown(f"""
            <div class="result-card">
                <h3>🦠 {disease_info['disease']}</h3>
                <p><b>Plant:</b> {disease_info['plant']} — <i>{disease_info['botanical_name']}</i></p>
                <p><b>Confidence:</b>
                    <span class="{conf_class}">{confidence*100:.1f}%</span>
                </p>
            </div>
        """, unsafe_allow_html=True)

        # Confidence bar
        st.metric("Confidence Score", f"{confidence*100:.1f}%")
        st.progress(confidence)

        st.subheader("📋 Recommended Treatment")
        st.success(disease_info['treatment'])

        st.subheader("🛡️ Prevention Strategy")
        st.info(disease_info['prevention'])

else:
    st.info("👆 Upload a leaf image above to begin analysis.")
    with st.expander("📋 Detectable Diseases (38 classes)"):
        for k, v in DISEASE_DB.items():
            emoji = "✅" if "healthy" in k.lower() else "🦠"
            st.write(f"{emoji} **{v['plant']}** — {v['disease']}")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
if model_mode == "nn":
    st.caption("🧠 Powered by MobileNetV2 fine-tuned on PlantVillage (38 classes).")
else:
    st.caption("🎨 Using color-feature analysis. Run `python train.py` for the full deep learning model.")
