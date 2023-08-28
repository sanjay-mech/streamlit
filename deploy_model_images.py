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


import cv2
import numpy as np


# Load the saved model
model_path = "best_model.h5"
model = load_model(model_path)

# Define a function to make predictions
def predict_fake(image):
    # Preprocess the input image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = cv2.resize(image, (128, 128))  # Resize to match model input dimensions
    image = image / 255.0  # Normalize pixel values to [0, 1]

    # Make a prediction
    prediction = model.predict(np.expand_dims(image, axis=0))[0]

    # Decide whether it's a deepfake or not
    if prediction >= 0.5:
        result = "Real"
    else:
        result = "Fake"

    return result

# Streamlit app
st.title("Deepfake Detection")

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Detect"):
        prediction = predict_fake(image)
        st.write(f"Prediction: {prediction}")

st.text("Upload an image and click 'Detect' to check if it's a deepfake.")