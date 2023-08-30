import os
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import dlib
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import applications
from efficientnet.tfkeras import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Function to extract frames from a video
def extract_frames(video_path, frame_rate=2):
    # Create a temporary directory named "temp_frames"
    temp_dir = "temp_frames"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Get the video file name without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Create a directory for frames within the temp directory
    frames_dir = os.path.join(temp_dir, video_name)
    
    # Handle existing directory by adding a number suffix
    dir_suffix = 1
    while os.path.exists(frames_dir):
        frames_dir = os.path.join(temp_dir, f"{video_name}_{dir_suffix}")
        dir_suffix += 1
    
    os.makedirs(frames_dir)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error opening video file")
        return
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    interval = int(fps / frame_rate)
    
    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_number % interval == 0:
            frame_filename = os.path.join(frames_dir, f"frame_{frame_number:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
        
        frame_number += 1
    
    cap.release()
    st.success(f"Extracted {frame_number} frames at {frame_rate} frames per second")
    
    return frames_dir

# Function to process video, detect deepfake, and highlight frames
def process_video(model_path, frame_rate=2):
    st.title("Deepfake Detection")

    # Load the deepfake detection model
    deepfake_model = load_model(best_model.h5)

    uploaded_video = st.file_uploader("Upload an MP4 video...", type=["mp4"])

    if uploaded_video is not None:
        with st.spinner("Processing video..."):
            video_path = "uploaded_video.mp4"
            with open(video_path, "wb") as f:
                f.write(uploaded_video.read())

            # Extract frames from the video
            frames_dir = extract_frames(video_path, frame_rate)

            # Process the frames for deepfake detection
            frame_count = 0
            fake_frame_count = 0
            real_frame_count = 0

            # Get the frame rate of the video
            video = cv2.VideoCapture(video_path)
            frame_rate = int(video.get(cv2.CAP_PROP_FPS))

            while True:
                ret, frame = video.read()
                if not ret:
                    break

                frame_count += 1

                # Skip frames based on frame rate (e.g., 2 frames per second)
                if frame_count % frame_rate != 0:
                    continue

                frame_filename = os.path.join(frames_dir, f"frame_{frame_count:04d}.jpg")

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
                frame_with_boxes = frame_rgb.copy()

                # Load the dlib face detection model
                detector = dlib.get_frontal_face_detector()

                # Convert frame to grayscale for face detection
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect faces in the frame
                faces = detector(gray_frame)

                # Draw bounding boxes on the faces
                for face in faces:
                    x, y, w, h = face.left(), face.top(), face.width(), face.height()
                    cv2.rectangle(frame_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if result == "Fake":
                    frame_with_boxes = cv2.rectangle(frame_with_boxes, (0, 0), (frame_with_boxes.shape[1], frame_with_boxes.shape[0]), (255, 0, 0), 5)
                else:
                    frame_with_boxes = cv2.rectangle(frame_with_boxes, (0, 0), (frame_with_boxes.shape[1], frame_with_boxes.shape[0]), (0, 255, 0), 5)

                # Display the processed frame
                st.image(frame_with_boxes, caption=f"Frame {frame_count} - {result}", use_column_width=True)

            st.warning("Uploaded video file and temporary frames have been deleted for privacy.")

            # Determine the final result based on the majority of frames
            if fake_frame_count > real_frame_count:
                final_result = "Fake"
            else:
                final_result = "Real"

            st.subheader("Final Result:")
            st.info(f"Given video is {final_result}.")

# Specify the path to your deepfake detection model
model_path = 'best_model.h5'

# Run the Streamlit app
if __name__ == "__main__":
    process_video(model_path)
