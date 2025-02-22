import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load TFLite Model
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Path to your TFLite model in Google Drive
tflite_model_path = "/content/drive/MyDrive/models/vgg_golden_model.tflite"

interpreter = load_tflite_model(tflite_model_path)

# Get model input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to preprocess image and make predictions
def predict(image):
    image = image.resize((224, 224))  # Adjust size based on your model
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    return np.argmax(output)  # Modify based on your class labels

# Streamlit UI
st.title("VGG Golden Model Classifier")
st.write("Upload an image to classify.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    prediction = predict(image)
    st.write(f"Predicted Class: {prediction}")
