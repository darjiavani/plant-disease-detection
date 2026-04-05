
import streamlit as st
from PIL import Image
import numpy as np
import joblib
import time

# ---------------- LOGIN SYSTEM ----------------
users = {
    "avani": "1234",
    "admin": "admin"
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.title("🔐 Login System")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in users and users[username] == password:
            st.session_state.logged_in = True
            st.success("Login successful ✅")
        else:
            st.error("Invalid credentials ❌")

if not st.session_state.logged_in:
    login()
    st.stop()

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿", layout="centered")

# ---------------- STYLING ----------------
st.markdown("""
<style>
body {
    background-image: url("https://images.unsplash.com/photo-1501004318641-b39e6451bec6");
    background-size: cover;
}
.title {
    text-align: center;
    font-size: 40px;
    color: #00FFAA;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: #ddd;
}
.card {
    background: rgba(0,0,0,0.6);
    padding: 20px;
    border-radius: 15px;
}
.result {
    text-align: center;
    font-size: 28px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Mobile responsive
st.markdown("""
<style>
@media (max-width: 600px) {
    .title {font-size: 26px;}
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown('<div class="title">🌿 Plant Disease Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered smart analyzer 🤖</div>', unsafe_allow_html=True)

# ---------------- MODEL LOAD ----------------
model = joblib.load("model/model.pkl")
classes = ['Healthy', 'Rust', 'Blight']

# ---------------- UI CARD ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)

# ---------------- MULTIPLE IMAGE UPLOAD ----------------
uploaded_files = st.file_uploader(
    "📤 Upload Leaf Images",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        img = Image.open(file)

        st.image(img, use_container_width=True)

        # Preprocess
        img_resized = img.resize((64, 64))
        img_array = np.array(img_resized).flatten().reshape(1, -1)

        # Loading animation
        with st.spinner("🔍 Analyzing..."):
            time.sleep(1)
            prediction = model.predict(img_array)

        result = classes[prediction[0]]

        # Confidence
        try:
            probs = model.predict_proba(img_array)
            confidence = np.max(probs) * 100
        except:
            confidence = None

        # Color result
        if result == "Healthy":
            color = "lightgreen"
        elif result == "Rust":
            color = "orange"
        else:
            color = "red"

        st.markdown(f"<div class='result' style='color:{color};'>🌱 {result}</div>", unsafe_allow_html=True)

        if confidence:
            st.write(f"📊 Confidence: {confidence:.2f}%")

        # Suggestions
        if result == "Healthy":
            st.success("✅ Plant is healthy")
        elif result == "Rust":
            st.warning("⚠️ Rust detected - use fungicide")
        else:
            st.error("🚨 Blight detected - immediate action needed")

        st.divider()

st.markdown('</div>', unsafe_allow_html=True)