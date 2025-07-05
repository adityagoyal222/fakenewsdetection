import streamlit as st
import requests
import time
import os
import streamlit.components.v1 as components
import base64

st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("Fake News Classifier")

def load_path(directory, filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, f"../{directory}/{filename}")
    model_path = os.path.normpath(model_path)
    return model_path

fake_path = load_path('assets', 'fake.gif')
notfake_path = load_path('assets', 'notfake.jpg')
video_path = load_path('assets', 'loader.mp4')

def autoplay_video_html(video_path):
    with open(video_path, "rb") as video_file:
        video_bytes = video_file.read()
        encoded = base64.b64encode(video_bytes).decode()
    return f"""
    <div id="video-container">
        <video id="loadingVideo" width="700" autoplay muted>
            <source src="data:video/mp4;base64,{encoded}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
    </div>
    """

def get_base64_video(path):
    with open(path, "rb") as file:
        return base64.b64encode(file.read()).decode()


def autoplay_video_html(video_data_base64):
    return f"""
    <video width="700" autoplay>
        <source src="data:video/mp4;base64,{video_data_base64}" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    """

with st.form(key="news_form"):
    text = st.text_area("Paste a news article here:")
    submit_button = st.form_submit_button(label="🎬 Start Prediction")

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

        start_time = time.time()
        try:
            response = requests.get("http://localhost:8000/predict", params={"text": text})
            result = response.json()
            label = result["label"]
            confidence = result["confidence"]
        except Exception as e:
            st.error("Prediction failed. Is FastAPI running?")
            st.stop()

        VIDEO_DURATION = 19 
        elapsed = time.time() - start_time
        if elapsed < VIDEO_DURATION:
            time.sleep(VIDEO_DURATION - elapsed)

        loading_text_placeholder.empty()
        video_placeholder.empty()

        success_text = st.text(f"")
        success_text.markdown(
            f"<h2 style='text-align: center; color: #FF8800;'>Prediction: {label} News</h2>",
            unsafe_allow_html=True
        )
        if label == "Fake":
            st.image(fake_path, caption="This might be fake!", use_container_width=True)
        else:
            st.image(notfake_path, caption="Looks real!", use_container_width=True)
