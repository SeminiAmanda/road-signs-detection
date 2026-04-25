"""
GTSRB - German Traffic Sign Recognition
Streamlit Web App
=================
Usage:
    streamlit run scripts/app.py
"""

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

# ─────────────────────────────────────────────
# CLASS NAMES (GTSRB 43 classes)
# ─────────────────────────────────────────────
CLASS_NAMES = {
    0:  "Speed limit (20km/h)",
    1:  "Speed limit (30km/h)",
    2:  "Speed limit (50km/h)",
    3:  "Speed limit (60km/h)",
    4:  "Speed limit (70km/h)",
    5:  "Speed limit (80km/h)",
    6:  "End of speed limit (80km/h)",
    7:  "Speed limit (100km/h)",
    8:  "Speed limit (120km/h)",
    9:  "No passing",
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 metric tons",
}

IMG_SIZE   = 32
MODEL_PATH = "./scripts/best_model.keras"

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Road Sign Classifier",
    page_icon="🚦",
    layout="centered"
)

st.title("🚦 Road Sign Classifier")
st.markdown("Upload a road sign image and the model will identify it.")
st.markdown("---")

# ─────────────────────────────────────────────
# LOAD MODEL (cached so it only loads once)
# ─────────────────────────────────────────────
@st.cache_resource
def load_trained_model():
    return load_model(MODEL_PATH)

with st.spinner("Loading model..."):
    model = load_trained_model()

# ─────────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Choose a road sign image",
    type=["jpg", "jpeg", "png", "ppm"]
)

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_column_width=True)

    # Preprocess
    img_resized = image.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    img_array  = np.array(img_resized, dtype=np.float32) / 255.0
    img_array  = np.expand_dims(img_array, axis=0)  # (1, 32, 32, 3)

    # Predict
    predictions = model.predict(img_array, verbose=0)[0]
    top3_idx    = predictions.argsort()[-3:][::-1]

    predicted_class = top3_idx[0]
    confidence      = predictions[predicted_class] * 100

    with col2:
        st.subheader("Prediction")
        st.success(f"**{CLASS_NAMES[predicted_class]}**")
        st.metric("Confidence", f"{confidence:.1f}%")

    # Top 3 predictions
    st.markdown("---")
    st.subheader("Top 3 Predictions")
    for i, idx in enumerate(top3_idx):
        prob = predictions[idx] * 100
        st.write(f"**{i+1}. {CLASS_NAMES[idx]}**")
        st.progress(int(prob), text=f"{prob:.1f}%")

else:
    st.info("👆 Upload an image to get started!")
    st.markdown("### Example signs you can test:")
    st.markdown("- Speed limit signs (circular, red border)")
    st.markdown("- Stop sign (octagonal, red)")
    st.markdown("- Yield sign (triangular, red border)")
    st.markdown("- Warning signs (triangular, yellow)")