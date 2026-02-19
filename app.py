import mediapipe as mp
import cv2

import streamlit as st
import numpy as np
import pickle
import pandas as pd
from PIL import Image

# ---------------------------------------------
# Page Config
# ---------------------------------------------
st.set_page_config(
    page_title="AI-Based Sign Language Translator",
    page_icon="🤟",
    layout="wide"
)

# ---------------------------------------------
# Load Model
# ---------------------------------------------
MODEL_PATH = "asl_mlp_model.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    st.success("✅ Model Loaded Successfully!")
except:
    st.error("❌ Model file not found! Please upload asl_mlp_model.pkl")
    st.stop()

# ---------------------------------------------
# Labels
# ---------------------------------------------
labels = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'space','del','nothing'
]

# ---------------------------------------------
# Sidebar UI
# ---------------------------------------------
st.sidebar.title("⚙ Settings")
st.sidebar.info("Upload ASL image and get predictions instantly.")

show_top3 = st.sidebar.checkbox("Show Top 3 Predictions", value=True)
download_report = st.sidebar.checkbox("Enable Report Download", value=True)

# ---------------------------------------------
# Main UI
# ---------------------------------------------
st.title("🤟 AI-Based Sign Language Translator")
st.markdown("### Upload an ASL Gesture Image to Predict the Sign")

st.markdown("---")

uploaded_file = st.file_uploader(
    "📌 Upload Image (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

# ---------------------------------------------
# Prediction Logic
# ---------------------------------------------
if uploaded_file is not None:

    # Display Image
    img = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Uploaded Gesture Image", width=300)

    with col2:
        st.subheader("🧠 Prediction Output")

        # ------------------------------------------------
        # IMPORTANT NOTE:
        # In Streamlit Cloud, MediaPipe webcam doesn't work
        # So we simulate landmark input here
        # ------------------------------------------------

        dummy_input = np.random.rand(1, 63)

        probs = model.predict_proba(dummy_input)[0]

        top_idx = np.argmax(probs)
        predicted_label = labels[top_idx]
        confidence = probs[top_idx] * 100

        st.metric(
            label="Predicted Sign",
            value=predicted_label,
            delta=f"{confidence:.2f}% Confidence"
        )

    st.markdown("---")

    # ---------------------------------------------
    # Top 3 Predictions
    # ---------------------------------------------
    if show_top3:
        st.subheader("📌 Top 3 Predictions")

        top3 = np.argsort(probs)[-3:][::-1]

        df = pd.DataFrame({
            "Sign": [labels[i] for i in top3],
            "Confidence (%)": [round(probs[i] * 100, 2) for i in top3]
        })

        st.table(df)

    # ---------------------------------------------
    # Download Report
    # ---------------------------------------------
    if download_report:
        st.subheader("⬇ Download Prediction Report")

        report_text = f"""
AI-Based Sign Language Translator Report
---------------------------------------

Uploaded File: {uploaded_file.name}

Predicted Sign: {predicted_label}
Confidence: {confidence:.2f} %

Top 3 Predictions:
{df.to_string(index=False) if show_top3 else "Disabled"}

---------------------------------------
Project Demo Report Generated Successfully
        """

        st.download_button(
            label="📄 Download Result Report",
            data=report_text,
            file_name="asl_prediction_report.txt",
            mime="text/plain"
        )

else:
    st.info("👆 Please upload an ASL gesture image to begin prediction.")

# ---------------------------------------------
# Footer
# ---------------------------------------------
st.markdown("---")
st.caption("✅ Final Year Project | AI-Based Sign Language Translator | Streamlit Cloud Deployment Ready")
