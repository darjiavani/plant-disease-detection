import streamlit as st
import numpy as np
from PIL import Image
import joblib
import json
import os

# =======================
# 🎨 PAGE CONFIG
# =======================
st.set_page_config(page_title="Plant Disease Detection", layout="centered")

# =======================
# 🎨 BACKGROUND STYLE
# =======================
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
background-image: url("https://images.unsplash.com/photo-1501004318641-b39e6451bec6");
background-size: cover;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# =======================
# 🔐 USER FILE
# =======================
USER_FILE = "users.json"

if not os.path.exists(USER_FILE):
    with open(USER_FILE, "w") as f:
        json.dump({}, f)

def load_users():
    with open(USER_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f)

# =======================
# 🔐 SESSION STATE
# =======================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "history" not in st.session_state:
    st.session_state.history = []

# =======================
# 🔐 LOGIN / SIGNUP
# =======================
if not st.session_state.logged_in:

    st.title("🔐 Login System")

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    # LOGIN
    with tab1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            users = load_users()
            if username in users and users[username] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid credentials ❌")

    # SIGNUP
    with tab2:
        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")

        if st.button("Create Account"):
            users = load_users()
            if new_user in users:
                st.warning("User already exists")
            else:
                users[new_user] = new_pass
                save_users(users)
                st.success("Account created! Now login")

# =======================
# 🌿 MAIN APP
# =======================
else:

    st.title("🌿 Plant Disease Detection")

    st.success(f"👤 Welcome {st.session_state.username}")

    # LOGOUT
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    # LOAD MODEL
    model = joblib.load("model/model.pkl")
    class_names = ["Healthy", "Rust", "Blight"]

    # FILE UPLOAD
    file = st.file_uploader("📤 Upload Leaf Image", type=["jpg", "png", "jpeg"])

    if file:
        img = Image.open(file).resize((64, 64))
        st.image(img, caption="Uploaded Image")

        # PREPROCESS
        img_array = np.array(img).flatten().reshape(1, -1)

        # PREDICTION
        probs = model.predict_proba(img_array)[0]
        pred = np.argmax(probs)
        final_class = class_names[pred]

        # SAVE HISTORY
        st.session_state.history.append(final_class)

        # =======================
        # 🎯 RESULT CARD
        # =======================
        st.markdown(f"""
        <div style="background:#1e293b;padding:20px;border-radius:12px">
        <h3 style="color:#38bdf8;">🔍 Prediction Result</h3>
        <p>🌿 Healthy: {probs[0]*100:.2f}%</p>
        <p>🍂 Rust: {probs[1]*100:.2f}%</p>
        <p>⚠ Blight: {probs[2]*100:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

        # FINAL RESULT
        st.success(f"🌟 Final Prediction: {final_class}")

        # =======================
        # 📊 PROGRESS BARS
        # =======================
        st.subheader("📊 Confidence Level")

        for i, cls in enumerate(class_names):
            st.write(cls)
            st.progress(int(probs[i]*100))

    # =======================
    # 📜 HISTORY
    # =======================
    if st.session_state.history:
        st.subheader("📜 Prediction History")
        st.write(st.session_state.history)