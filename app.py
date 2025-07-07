import streamlit as st
import numpy as np
import tempfile
import tensorflow as tf
from utils import preprocess_audio

# Load the model
model = tf.keras.models.load_model("SEA_model.h5")

# Emotion labels (customize as per your model's output order)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# UI Title
st.set_page_config(page_title="Speech Emotion Analysis", page_icon="🎙️")
st.title("🎙️ Speech Emotion Analysis (LSTM Model)")
st.markdown("Upload an audio file to predict the **emotion** from speech.")

# File uploader
uploaded_file = st.file_uploader("Choose an audio file (.wav, .mp3)", type=["wav", "mp3"])

# Process and predict
if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    with st.spinner("Analyzing emotion..."):
        features = preprocess_audio(tmp_path)
        if features is not None:
            prediction = model.predict(features)
            predicted_emotion = emotion_labels[np.argmax(prediction)]
            confidence = np.max(prediction) * 100

            st.success(f"🎯 **Predicted Emotion:** {predicted_emotion}")
            st.write(f"🧠 Confidence: `{confidence:.2f}%`")
        else:
            st.error("❌ Failed to extract features from the audio.")