# app.py
import streamlit as st
import librosa
import numpy as np
from keras.models import load_model

model = load_model('your_trained_model.h5')

st.title("Speech Emotion Analyzer")
audio_file = st.file_uploader("Upload Audio", type=["wav"])

if audio_file is not None:
    st.audio(audio_file)
    features = extract_features(audio_file)
    features = np.expand_dims(features, axis=(0, -1))
    prediction = model.predict(features)
    emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    st.write("Predicted Emotion:", emotions[np.argmax(prediction)])