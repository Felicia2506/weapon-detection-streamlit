import os
import requests
import tensorflow.lite as tflite
import numpy as np
from PIL import Image
import streamlit as st

# ------------------------------
# 🔹 Google Drive Model Download
# ------------------------------
FILE_ID = "1-BCKd-ssavT3O8HQ-NSeP1fuswuMDeMa"
MODEL_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
MODEL_PATH = "vgg_golden_model.tflite"

# Function to download model if not available locally
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading TFLite model (this may take a moment)...")
        try:
            response = requests.get(MODEL_URL, stream=True)
            if response.status_code == 200:
                with open(MODEL_PATH, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
                st.success("TFLite model downloaded successfully! ✅")
            else:
                st.error(f"Error downloading model! HTTP Status: {response.status_code}")
        except Exception as e:
            st.error(f"Download failed: {str(e)}")

# Call the function before running inference
download_model()

# ------------------------------
# 🔹 Load TFLite Model
# ------------------------------
@st.cache_resource
def load_model():
    try:
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

# ------------------------------
# 🔹 Preprocess Image Function
# ------------------------------
def preprocess_image(image, target_size=(224, 224)):
    image = image.convert("RGB")  # Ensure image is RGB
    image = image.resize(target_size)  # Resize to model input size
    image = np.array(image).astype(np.float32) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# ------------------------------
# 🔹 Run Inference on Image
# ------------------------------
def predict(image, interpreter):
    if interpreter is None:
        return "Error", 0.0

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    image = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    classes = ["Gun", "Knife", "Safe"]
    predicted_class = classes[np.argmax(output)]
    confidence = np.max(output) * 100  # Convert to percentage

    return predicted_class, confidence

# ------------------------------
# 🔹 Streamlit UI Setup
# ------------------------------
st.title("🔍 X-ray Weapon Detection")
st.write("Upload an X-ray image to classify whether it contains a **Gun, Knife, or is Safe**.")

uploaded_file = st.file_uploader("📤 Upload an Image (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    interpreter = load_model()
    prediction, confidence = predict(image, interpreter)

    st.markdown(f"### 🎯 Prediction: **{prediction}**")
    st.markdown(f"### 🔥 Confidence: **{confidence:.2f}%**")

    # Additional insights
    if prediction == "Safe":
        st.success("✅ No weapon detected. The X-ray is safe.")
    else:
        st.warning("⚠️ Weapon detected! Further inspection recommended.")

# ------------------------------
# 🔹 Footer
# ------------------------------
st.markdown("---")
st.write("🚀 **Built with Streamlit & TensorFlow Lite**")
