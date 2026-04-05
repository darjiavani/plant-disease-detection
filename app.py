import streamlit as st
import numpy as np
from PIL import Image
import joblib
import json
import os

# ---------------------------
# LOAD MODEL
# ---------------------------
model = joblib.load("model/model.pkl")
class_names = ['Healthy', 'Rust', 'Blight']

# ---------------------------
# USER DATABASE
# ---------------------------
if not os.path.exists("users.json"):
    with open("users.json", "w") as f:
        json.dump({}, f)

with open("users.json", "r") as f:
    users = json.load(f)

# ---------------------------
# SESSION STATE
# ---------------------------
if "login" not in st.session_state:
    st.session_state.login = False
if "user" not in st.session_state:
    st.session_state.user = ""

# ---------------------------
# UI STYLE
# ---------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}
.big-title {
    text-align:center;
    font-size:40px;
    color:#00ffcc;
}
.card {
    padding:20px;
    border-radius:15px;
    background:#1c1c1c;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# SIGNUP
# ---------------------------
def signup():
    st.subheader("🆕 Create Account")
    new_user = st.text_input("Username")
    new_pass = st.text_input("Password", type="password")

    if st.button("Signup"):
        if new_user in users:
            st.error("User already exists")
        else:
            users[new_user] = new_pass
            with open("users.json", "w") as f:
                json.dump(users, f)
            st.success("Account created!")

# ---------------------------
# LOGIN
# ---------------------------
def login():
    st.markdown("<h1 class='big-title'>🔐 Login System</h1>", unsafe_allow_html=True)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in users and users[username] == password:
            st.session_state.login = True
            st.session_state.user = username
            st.success("Login successful")
        else:
            st.error("Invalid credentials")

# ---------------------------
# MAIN APP
# ---------------------------
def main_app():
    st.markdown("<h1 class='big-title'>🌿 Plant Disease Detection</h1>", unsafe_allow_html=True)
    st.write(f"👤 Welcome: {st.session_state.user}")

    file = st.file_uploader("Upload Leaf Image", type=["jpg","png","jpeg"])

    if file:
        img = Image.open(file).resize((64,64))
        st.image(img, caption="Uploaded Image")

        img_array = np.array(img).flatten().reshape(1, -1)

        # Prediction
        probs = model.predict_proba(img_array)[0]
        top3 = np.argsort(probs)[-3:][::-1]

        st.subheader("🔍 Predictions")

        for i in top3:
            st.write(f"{class_names[i]} → {round(probs[i]*100,2)}%")

# ---------------------------
# NAVIGATION
# ---------------------------
menu = st.sidebar.selectbox("Menu", ["Login", "Signup"])

if not st.session_state.login:
    if menu == "Login":
        login()
    else:
        signup()
else:
    main_app()