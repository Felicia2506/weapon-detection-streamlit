import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from PIL import Image

# Load the model
model = load_model("vgg_finetuned_full.h5")

# Define class names
class_names = ['gun', 'knife', 'safe']

# Streamlit App Title
st.title("Weapon Detection Model")
st.write("Upload an image to detect if it is a gun, knife, or a safe object.")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict using the model
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100

    # Show prediction
    st.write(f"Predicted Class: **{predicted_class}**")
    st.write(f"Confidence: **{confidence:.2f}%**")
