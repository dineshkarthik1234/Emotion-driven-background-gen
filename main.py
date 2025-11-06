import cv2
import streamlit as st
from emotion_module import detect_emotion
from generative_module import generate_background, replace_background
from PIL import Image
import numpy as np

st.set_page_config(page_title="Emotion-Driven AI Art Camera", layout="wide")
st.title("Emotion-Driven AI Art Camera with Background Generation")

prompt_input = st.text_input("Enter background prompt for generation:", value="beautiful fantasy landscape, vibrant colors")
run_cam = st.button("Start Camera")

if run_cam:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    artframe = st.empty()
    stop_cam = False
    st.write("Press 'Stop' button to end the session.")
    
    # Generate background image once per session from user prompt
    bg_image = generate_background(prompt_input)

    while not stop_cam:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access camera.")
            break
        emotions = detect_emotion(frame)
        emotion = "neutral"
        for (x, y, w, h, e) in emotions:
            emotion = e
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        # Replace background with generated one
        composed_frame = replace_background(frame, bg_image)

        # Display camera feed with replaced background
        stframe.image(composed_frame, channels="BGR", caption="Live Feed with Generated Background", use_column_width=True)

        stop_cam = st.button("Stop")
    cap.release()
    st.success("Camera stopped.")
