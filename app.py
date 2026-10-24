from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Panel AI | Image Classifier",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = Path(__file__).resolve().parent / "panel_classifier.h5"


CLASS_NAMES = [
    "Bird-drop",
    "Clean",
    "Dusty",
    "Electrical-damage",
    "Physical-Damage",
    "Snow-Covered",
]

CLASS_DESCRIPTIONS = {
    "Bird-drop": "Bird droppings detected",
    "Clean": "Panel appears clean",
    "Dusty": "Dust accumulation detected",
    "Electrical-damage": "Possible electrical damage detected",
    "Physical-Damage": "Possible physical damage detected",
    "Snow-Covered": "Snow or ice coverage detected",
}


# -------------------------------------------------------------------
# Styling
# -------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    .stApp {
        background:
            radial-gradient(circle at 8% 0%, rgba(61, 124, 255, 0.09), transparent 28rem),
            radial-gradient(circle at 95% 12%, rgba(15, 185, 129, 0.08), transparent 26rem),
            #f6f8fc;
        font-family: 'DM Sans', sans-serif;
    }
    .block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem; }
    h1, h2, h3 { font-family: 'Manrope', sans-serif !important; letter-spacing: -0.035em; }
    .eyebrow {
        color: #2563eb; font-size: 0.78rem; font-weight: 800;
        letter-spacing: 0.13em; text-transform: uppercase; margin-bottom: 0.6rem;
    }
    .hero-title {
        color: #14213d; font-family: 'Manrope', sans-serif;
        font-size: clamp(2.2rem, 4vw, 3.35rem); line-height: 1.08;
        font-weight: 800; letter-spacing: -0.055em; margin: 0;
    }
    .hero-copy { color: #5c6b82; font-size: 1.03rem; line-height: 1.7; max-width: 700px; margin-top: 0.8rem; }
    .panel-card {
        background: rgba(255, 255, 255, 0.94); border: 1px solid #e3eaf4;
        border-radius: 22px; padding: 1.35rem 1.45rem;
        box-shadow: 0 14px 40px rgba(31, 52, 88, 0.055);
    }
    .section-title { color: #17233c; font-family: 'Manrope', sans-serif; font-weight: 800; font-size: 1.1rem; margin-bottom: 0.35rem; }
    .section-subtitle { color: #708096; font-size: 0.9rem; line-height: 1.55; margin-bottom: 1rem; }
    .result-kicker { color: #64748b; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.12em; font-weight: 800; margin-bottom: 0.55rem; }
    .result-name { color: #10213f; font-family: 'Manrope', sans-serif; font-size: clamp(1.55rem, 2.5vw, 2.1rem); line-height: 1.2; font-weight: 800; overflow-wrap: anywhere; }
    .result-description { color: #65758b; font-size: 0.95rem; margin-top: 0.45rem; line-height: 1.5; }
    .status-pill { display: inline-block; background: #e8f7ef; color: #137547; border: 1px solid #c8ecd8; border-radius: 999px; padding: 0.33rem 0.68rem; font-weight: 700; font-size: 0.77rem; margin-bottom: 1rem; }
    .footer-note { color: #8290a4; font-size: 0.8rem; text-align: center; margin-top: 2.25rem; }
    div[data-testid="stFileUploader"] { background: #f8faff; border: 1px dashed #b8c7dc; border-radius: 16px; padding: 0.75rem; }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(100deg, #2563eb, #3478f6); border: 0;
        border-radius: 12px; min-height: 3rem; font-weight: 800;
        box-shadow: 0 8px 18px rgba(37, 99, 235, 0.2);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div.stButton > button[kind="primary"]:hover { box-shadow: 0 10px 24px rgba(37, 99, 235, 0.28); transform: translateY(-1px); }
    div[data-testid="stMetric"] { background: #f7faff; padding: 0.9rem 1rem; border: 1px solid #e5edf8; border-radius: 14px; }
    div[data-testid="stProgressBar"] > div > div { background: linear-gradient(90deg, #2563eb, #20b486); }
    hr { border-color: #e6ecf4; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    
    return tf.keras.models.load_model(model_path, compile=False)


def get_input_size(model):
    
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    height = int(shape[1]) if len(shape) > 2 and shape[1] else 256
    width = int(shape[2]) if len(shape) > 2 and shape[2] else 256
    return height, width


def predict_image(model, image: Image.Image):
    height, width = get_input_size(model)
    resized = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    image_array = np.asarray(resized, dtype=np.float32)
    image_batch = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_batch, verbose=0)
    return np.asarray(predictions[0], dtype=np.float64).reshape(-1)


# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------
st.markdown('<div class="eyebrow">AI-powered visual inspection</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-title">Panel inspection, <span style="color:#2563eb">made simple.</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero-copy">Upload a panel image and let your trained EfficientNet model estimate '
    'which condition it belongs to. View the predicted class and scores for all six categories.</div>',
    unsafe_allow_html=True,
)

pill_cols = st.columns(3)
for col, icon, label in zip(
    pill_cols,
    ["🖼️", "⚡", "📊"],
    ["Image upload", "Live inference", "Class probabilities"],
):
    with col:
        st.markdown(
            f'<div style="color:#53647c;font-size:0.88rem;font-weight:700;padding:0.65rem 0.2rem;">'
            f'{icon}&nbsp;&nbsp; {label}</div>',
            unsafe_allow_html=True,
        )

st.write("")
left, right = st.columns([0.94, 1.06], gap="large")

# -------------------------------------------------------------------
# Upload panel
# -------------------------------------------------------------------
with left:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">1. Upload an image</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Choose a clear JPG, JPEG, PNG, or WebP image. '
        'Use an image similar to those in your training dataset for best results.</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose a panel image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
        key="panel_upload",
    )

    image = None
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Image ready for inspection", use_container_width=True)
            st.caption(f"Image dimensions: {image.width} × {image.height} px")
        except (UnidentifiedImageError, OSError):
            st.error("This file could not be read as an image. Try another JPG or PNG.")

    analyze_clicked = st.button(
        "✦  Analyze image",
        type="primary",
        use_container_width=True,
        disabled=(image is None),
    )
    st.markdown(
        '<div style="color:#8290a4;font-size:0.78rem;margin-top:0.7rem;">'
        'The image is processed locally by the model configured in this app.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

# Keep the currently selected upload associated with the displayed result.
file_signature = None
if uploaded_file is not None:
    import hashlib
    file_signature = hashlib.sha256(uploaded_file.getvalue()).hexdigest()

if st.session_state.get("file_signature") != file_signature:
    st.session_state["prediction_result"] = None
    st.session_state["file_signature"] = file_signature

# -------------------------------------------------------------------
# Prediction panel
# -------------------------------------------------------------------
with right:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">2. Prediction result</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Your result will appear here, including the predicted condition '
        'and the model scores for each class.</div>',
        unsafe_allow_html=True,
    )

    model = None
    if not MODEL_PATH.exists():
        st.error(f"Model file not found: `{MODEL_PATH.name}`")
        st.info(f"Place `panel_classifier.h5` next to `app.py`.\n\nExpected path: `{MODEL_PATH}`")
    else:
        try:
            with st.spinner("Loading trained model…"):
                model = load_model(str(MODEL_PATH))
        except Exception as exc:
            st.error("The model could not be loaded. Check that the H5 file is valid and compatible with your TensorFlow/Keras installation.")
            with st.expander("Technical details"):
                st.code(str(exc))

    if analyze_clicked and image is not None and model is not None:
        try:
            with st.spinner("Analyzing image… extracting features and calculating class scores."):
                probabilities = predict_image(model, image)
                if len(probabilities) != len(CLASS_NAMES):
                    raise ValueError(
                        f"The model returned {len(probabilities)} scores, but CLASS_NAMES contains "
                        f"{len(CLASS_NAMES)} classes. Update CLASS_NAMES to match the training output order."
                    )
                predicted_index = int(np.argmax(probabilities))
                st.session_state["prediction_result"] = {
                    "probabilities": probabilities.tolist(),
                    "index": predicted_index,
                    "confidence": float(probabilities[predicted_index]),
                    "file_signature": file_signature,
                }
        except Exception as exc:
            st.session_state["prediction_result"] = None
            st.error("Prediction failed.")
            with st.expander("Technical details"):
                st.code(str(exc))

    result = st.session_state.get("prediction_result")
    if result is None:
        st.markdown(
            """
            <div style="border:1px dashed #d5dfed;border-radius:18px;padding:2.6rem 1.2rem;
                        text-align:center;background:#fafcff;margin-top:0.6rem;">
              <div style="font-size:2.3rem;margin-bottom:0.65rem;">🔎</div>
              <div style="font-weight:800;color:#2b3b55;font-size:1.02rem;">Ready when you are</div>
              <div style="font-size:0.87rem;color:#8290a4;margin-top:0.4rem;">
                Upload an image, then click Analyze image.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        pred_index = int(result["index"])
        pred_class = CLASS_NAMES[pred_index]
        confidence = float(result["confidence"])

        st.markdown('<div class="status-pill">✓ Analysis complete</div>', unsafe_allow_html=True)
        st.markdown('<div class="result-kicker">Predicted condition</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-name">{pred_class}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="result-description">{CLASS_DESCRIPTIONS.get(pred_class, "Predicted image category")}</div>',
            unsafe_allow_html=True,
        )
        st.write("")

        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Top class score", f"{confidence:.1%}")
        with metric_col2:
            st.metric("Classes evaluated", str(len(CLASS_NAMES)))

        st.progress(float(np.clip(confidence, 0.0, 1.0)), text="Score for the predicted class")
        st.caption("A model score is not a guarantee of correctness. Verify important inspection decisions.")

        st.markdown("---")
        st.markdown('<div class="section-title">Class probability breakdown</div>', unsafe_allow_html=True)
        probs = np.asarray(result["probabilities"], dtype=np.float64)
        for idx in np.argsort(probs)[::-1]:
            name = CLASS_NAMES[int(idx)]
            score = float(np.clip(probs[idx], 0.0, 1.0))
            is_top = int(idx) == pred_index
            name_col, score_col = st.columns([0.75, 0.25])
            with name_col:
                star = "★ " if is_top else ""
                color = "#1d4ed8" if is_top else "#53647c"
                weight = "800" if is_top else "600"
                st.markdown(
                    f'<div style="font-weight:{weight};color:{color};padding-top:0.25rem;">{star}{name}</div>',
                    unsafe_allow_html=True,
                )
            with score_col:
                st.markdown(
                    f'<div style="text-align:right;font-weight:800;color:#31415b;">{score:.1%}</div>',
                    unsafe_allow_html=True,
                )
            st.progress(score)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    '<div class="footer-note">Panel AI · EfficientNetB0 image classification · '
    'Predictions are model estimates and should be verified for safety-critical decisions.</div>',
    unsafe_allow_html=True,
)
