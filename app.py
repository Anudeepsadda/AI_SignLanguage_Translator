import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
from PIL import Image

# -------------------------------
# Load trained MLP model
# -------------------------------
with open("asl_mlp_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("AI-Based Sign Language Translator")
st.write("Upload an ASL hand gesture image to get prediction")

# -------------------------------
# MediaPipe Hands Setup
# -------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

# -------------------------------
# Landmark Extraction Function
# -------------------------------
def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            row = []
            for lm in hand_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])
            return np.array(row)

    return None

# -------------------------------
# Upload Image Option
# -------------------------------
uploaded_file = st.file_uploader("Upload a Hand Gesture Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    # Show uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Convert PIL image → OpenCV format
    img_cv = np.array(img)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    # Extract landmarks
    landmarks = extract_landmarks(img_cv)

    if landmarks is not None:
        landmarks = landmarks.reshape(1, -1)

        # Prediction + Confidence
        probs = model.predict_proba(landmarks)
        confidence = np.max(probs)
        prediction = model.classes_[np.argmax(probs)]

        st.success(f"Predicted Sign: {prediction}")
        st.info(f"Confidence Score: {confidence:.2f}")

    else:
        st.error("No hand detected in the uploaded image. Try another image.")
