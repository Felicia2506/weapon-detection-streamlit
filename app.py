import os
import requests
import tensorflow.lite as tflite
import numpy as np
from PIL import Image
import streamlit as st

# Google Drive file ID (replace with your actual ID)
FILE_ID = "1A2B3C4D5E6F7G8H"

# Construct the Google Drive direct download URL
MODEL_URL = f"https://drive.google.com/file/d/1-BCKd-ssavT3O8HQ-NSeP1fuswuMDeMa/view?usp=drive_link"
MODEL_PATH = "vgg_golden_model.tflite"

# Function to download model if not available locally
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading TFLite model...")
        response = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        st.success("Model downloaded successfully!")

# Call the download function before running inference
download_model()

# Load the model
@st.cache_resource
def load_model():
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

# Preprocess image
def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image = np.array(image).astype(np.float32) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Run inference
def predict(image, interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    image = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    classes = ["Gun", "Knife", "Safe"]
    predicted_class = classes[np.argmax(output)]
    confidence = np.max(output) * 100

    return predicted_class, confidence

# Streamlit UI
st.title("Weapon Detection in X-ray Images")
uploaded_file = st.file_uploader("Upload an X-ray Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    interpreter = load_model()
    prediction, confidence = predict(image, interpreter)

    st.write(f"Prediction: **{prediction}**")
    st.write(f"Confidence: **{confidence:.2f}%**")
