import streamlit as st
import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import applications
from efficientnet.tfkeras import EfficientNetB0 #EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
import numpy as np
import time
import cv2
import shutil  # Import the shutil library for file removal
from PIL import Image

# Load the deepfake detection model
deepfake_model = load_model('best_model.h5')


st.title("Deepfake Detection and Highlighting")

uploaded_video = st.file_uploader("Upload an MP4 video...", type=["mp4"])

if uploaded_video is not None:
    with st.spinner("Processing video..."):
        video_path = "uploaded_video.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())

        # Create a folder to save the frames
        frames_folder = "temp_frames"
        os.makedirs(frames_folder, exist_ok=True)

        # Process the video
        video = cv2.VideoCapture(video_path)
        frame_count = 0
        fake_frame_count = 0
        real_frame_count = 0

        # Get the frame rate of the video
        frame_rate = int(video.get(cv2.CAP_PROP_FPS))

        while True:
            ret, frame = video.read()
            if not ret:
                break

            frame_count += 1

            # Skip frames based on frame rate (e.g., 2 frames per second)
            if frame_count % frame_rate != 0:
                continue

            frame_filename = os.path.join(frames_folder, f"frame_{frame_count}.jpg")

            # Convert the frame to RGB and save it
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imwrite(frame_filename, frame_rgb)

            # Perform deepfake detection on the frame
            frame_image = cv2.resize(frame_rgb, (128, 128)) / 255.0
            frame_image = np.expand_dims(frame_image, axis=0)
            prediction = deepfake_model.predict(frame_image)[0]

            if prediction >= 0.5:
                result = "Fake"
                fake_frame_count += 1
            else:
                result = "Real"
                real_frame_count += 1

            # Highlight the result in red for Fake and green for Real
            if result == "Fake":
                frame_rgb = cv2.rectangle(frame_rgb, (0, 0), (frame_rgb.shape[1], frame_rgb.shape[0]), (255, 0, 0), 5)
            else:
                frame_rgb = cv2.rectangle(frame_rgb, (0, 0), (frame_rgb.shape[1], frame_rgb.shape[0]), (0, 255, 0), 5)

            # Display the processed frame
            st.image(frame_rgb, caption=f"Frame {frame_count} - {result}", use_column_width=True)

        st.warning("Uploaded video file and temporary frames have been deleted for privacy.")

        # Determine the final result based on the majority of frames
        if fake_frame_count > real_frame_count:
            final_result = "Fake"
        else:
            final_result = "Real"

        st.subheader("Final Result:")
        st.info(f"Given video is {final_result}.")