import streamlit as st
import tensorflow.lite as tflite
import numpy as np
from PIL import Image

# Load the TFLite model
@st.cache_resource
def load_model():
    interpreter = tflite.Interpreter(model_path="vgg_finetuned_full.tflite")
    interpreter.allocate_tensors()
    return interpreter

model = load_model()

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to match the model input
    image = np.array(image) / 255.0   # Normalize
    image = np.expand_dims(image, axis=0).astype(np.float32)
    return image

# Function to make predictions
def predict(image):
    input_tensor_index = model.get_input_details()[0]['index']
    output_tensor_index = model.get_output_details()[0]['index']
    
    model.set_tensor(input_tensor_index, image)
    model.invoke()
    
    predictions = model.get_tensor(output_tensor_index)
    class_index = np.argmax(predictions)
    class_labels = ["Gun", "Knife", "Safe"]  # Change based on your dataset
    return class_labels[class_index]

# Streamlit UI
st.title("Weapon Detection with Deep Learning")
st.write("Upload an image to detect if it contains a gun, knife, or is safe.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess and predict
    processed_image = preprocess_image(image)
    prediction = predict(processed_image)
    
    st.write(f"### Predicted Class: {prediction}")
