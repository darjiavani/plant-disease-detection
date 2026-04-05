import streamlit as st
import numpy as np
from PIL import Image
import joblib

# Load model
model = joblib.load("model/model.pkl")

class_names = ['Healthy', 'Rust', 'Blight']

st.title("🌿 Plant Disease Detection")

file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

if file:
    img = Image.open(file).resize((64, 64))
    st.image(img, caption="Uploaded Image")

    img = np.array(img).flatten().reshape(1, -1)

    pred = model.predict(img)[0]

    st.success(f"Disease: {class_names[pred]}")