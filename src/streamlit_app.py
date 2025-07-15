import streamlit as st
import requests
import time
import os
import streamlit.components.v1 as components
import base64
from utils import load_path

st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("Fake News Classifier")

# loading the path for assets (uses the load_path utility function)
fake_path = load_path('assets', 'fake.gif')
notfake_path = load_path('assets', 'notfake.jpg')
video_path = load_path('assets', 'loader.mp4')

# converts video into base64 format for better customization
def get_base64_video(path):
    with open(path, "rb") as file:
        return base64.b64encode(file.read()).decode()

# logic to autoplay the video and unmute it on page load
def autoplay_video_html(video_data_base64):
    return f"""
    <video width="700" autoplay>
        <source src="data:video/mp4;base64,{video_data_base64}" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    """

# loading a form on the screen
with st.form(key="news_form"):
    text = st.text_area("Paste a news article here:")
    submit_button = st.form_submit_button(label="🎬 Start Prediction")

# action upon submit button click
if submit_button:
    if not text or len(text.strip()) < 10:
        st.warning("Please enter a valid news article.")
    else:
        video_base64 = get_base64_video(video_path)
        loading_text_placeholder = st.empty()
        video_placeholder = st.empty()
        
        loading_text_placeholder.markdown(
            "<h2 style='text-align: center; color: #FF8800;'>Analyzing article... Please wait</h2>",
            unsafe_allow_html=True
        )
        with video_placeholder:
            components.html(autoplay_video_html(video_base64), height=400)

        # placeholder time to await result and video completion before displaying output
        start_time = time.time()
        try:
            response = requests.get("http://localhost:8000/predict", params={"text": text})
            result = response.json()
            label = result["label"]
            confidence = result["confidence"]
        except Exception as e:
            st.error("Prediction failed. Is FastAPI running?")
            st.stop()

        VIDEO_DURATION = 10 
        elapsed = time.time() - start_time
        if elapsed < VIDEO_DURATION:
            time.sleep(VIDEO_DURATION - elapsed)

        loading_text_placeholder.empty()
        video_placeholder.empty()

        # show dynamic output based on the type of the model prediction
        success_text = st.text(f"")
        success_text.markdown(
            f"<h2 style='text-align: center; color: #FF8800;'>Prediction: {label} News</h2>",
            unsafe_allow_html=True
        )
        if label == "Fake":
            st.image(fake_path, caption="This might be fake!", use_container_width=True)
        else:
            st.image(notfake_path, caption="Looks real!", use_container_width=True)
