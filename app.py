import streamlit as st
import gdown
import tensorflow as tf
import os

# Define the Google Drive link (Replace with your actual link)
gdrive_link = "https://drive.google.com/file/d/1-BCKd-ssavT3O8HQ-NSeP1fuswuMDeMa/view?usp=drive_link"

# Download the model file
model_path = "vgg_finetuned_full.tflite"

if not os.path.exists(model_path):
    st.info("Downloading the model file... (This happens only once)")
    gdown.download(gdrive_link, model_path, quiet=False)

# Load the model
st.info("Loading model...")
model = tf.keras.models.load_model(model_path)
st.success("Model loaded successfully!")

# Sample UI
st.title("Weapon Detection Model")
st.write("Upload an image to classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")

    # Preprocess the image
    from PIL import Image
    import numpy as np

    image = Image.open(uploaded_file)
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions, axis=-1)

    class_names = ["gun", "knife", "safe"]
    st.write(f"Predicted Class: {class_names[predicted_class[0]]}")
