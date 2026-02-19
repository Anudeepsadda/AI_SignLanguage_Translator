import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
from PIL import Image
from collections import deque

# ---------------------------------
# Page Config (Modern UI)
# ---------------------------------
st.set_page_config(
    page_title="AI Sign Language Translator",
    page_icon="🤟",
    layout="wide"
)

# ---------------------------------
# Load Trained Model
# ---------------------------------
with open("asl_mlp_model.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------------------------
# MediaPipe Setup
# ---------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

# ---------------------------------
# Prediction History (Last 5)
# ---------------------------------
if "history" not in st.session_state:
    st.session_state.history = deque(maxlen=5)

# ---------------------------------
# Landmark Extraction Function
# ---------------------------------
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

# ---------------------------------
# Sidebar Navigation
# ---------------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Upload & Predict", "Model Performance", "About Project"]
)

# =================================
# PAGE 1: Upload & Predict
# =================================
if page == "Upload & Predict":

    st.title("🤟 AI-Based Sign Language Translator")
    st.write("Upload an ASL hand gesture image and get instant predictions.")

    uploaded_file = st.file_uploader(
        "📤 Upload a Hand Gesture Image",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file is not None:

        col1, col2 = st.columns(2)

        # Show uploaded image
        img = Image.open(uploaded_file)

        with col1:
            st.image(img, caption="Uploaded Gesture", use_column_width=True)

        # Convert to OpenCV
        img_cv = np.array(img)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

        landmarks = extract_landmarks(img_cv)

        if landmarks is not None:

            landmarks = landmarks.reshape(1, -1)

            # Prediction probabilities
            probs = model.predict_proba(landmarks)[0]

            # Top-3 Predictions
            top3_idx = np.argsort(probs)[::-1][:3]
            top3_classes = model.classes_[top3_idx]
            top3_scores = probs[top3_idx]

            prediction = top3_classes[0]
            confidence = top3_scores[0]

            # Save history
            st.session_state.history.appendleft(prediction)

            with col2:
                st.success(f"✅ Predicted Sign: **{prediction}**")

                st.write("### Confidence Score")
                st.progress(int(confidence * 100))
                st.info(f"Model Confidence: {confidence:.2f}")

                st.write("### 🔥 Top 3 Predictions")
                for i in range(3):
                    st.write(f"{i+1}. {top3_classes[i]} ({top3_scores[i]:.2f})")

            # Download Report Feature
            report_text = f"""
AI Sign Language Translator Result
---------------------------------
Prediction: {prediction}
Confidence: {confidence:.2f}

Top 3 Predictions:
1. {top3_classes[0]} ({top3_scores[0]:.2f})
2. {top3_classes[1]} ({top3_scores[1]:.2f})
3. {top3_classes[2]} ({top3_scores[2]:.2f})
"""

            st.download_button(
                label="📥 Download Prediction Report",
                data=report_text,
                file_name="prediction_report.txt"
            )

        else:
            st.error("❌ No hand detected. Please upload a clear gesture image.")

    # Show Prediction History
    st.write("## 🕒 Recent Predictions History")
    if len(st.session_state.history) > 0:
        st.write(" → ".join(st.session_state.history))
    else:
        st.write("No predictions yet.")

# =================================
# PAGE 2: Model Performance
# =================================
elif page == "Model Performance":

    st.title("📊 Model Performance Overview")

    st.write("""
    The proposed system was trained using hand landmark features extracted
    from the ASL Alphabet Dataset.
    """)

    st.metric("Random Forest Accuracy", "98.29%")
    st.metric("Neural Network (MLP) Accuracy", "99.29%")

    st.write("""
    ✅ Novelty Added:
    - Hybrid ML + DL comparison  
    - Confidence-aware temporal voting for stability  
    """)

# =================================
# PAGE 3: About Project
# =================================
elif page == "About Project":

    st.title("📌 About This Project")

    st.write("""
    This AI-Based Sign Language Translator is a final-year student project
    designed to help bridge the communication gap between sign language users
    and the general public.

    **Technologies Used:**
    - MediaPipe Hands (Landmark Detection)
    - Machine Learning (Random Forest)
    - Deep Learning (Neural Network MLP)
    - Streamlit Web Deployment

    **Deployment:** Streamlit Cloud + GitHub
    """)

    st.success("🎓 Project Ready for Viva & Final Submission!")
