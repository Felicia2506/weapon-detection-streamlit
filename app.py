import os
import requests
import tensorflow as tf
import numpy as np
from PIL import Image
import streamlit as st
import gdown
# ------------------------------
# 🔹 Google Drive Model Download
# ------------------------------
FILE_ID = "1-73vXzdGyEPuFz4BikMpHrlOA5PnGtU0"  # Updated to your H5 model ID
MODEL_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
MODEL_PATH = "vgg_finetuned_model.h5"
EXPECTED_SIZE_MB = 100  # Adjust based on actual model size

def download_model():
    """Download the H5 model from Google Drive if it does not exist."""
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading H5 model (this may take a few minutes)...")
        try:
            response = requests.get(MODEL_URL, stream=True)
            if response.status_code == 200:
                with open(MODEL_PATH, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
                
                # Check file size after download
                downloaded_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
                if downloaded_size < (EXPECTED_SIZE_MB * 0.9):  # Allow 10% margin
                    st.error(f"Model download failed! File too small ({downloaded_size:.2f} MB)")
                    os.remove(MODEL_PATH)  # Delete corrupt file
                    return
                
                st.success(f"Model downloaded successfully! ✅ ({downloaded_size:.2f} MB)")
            else:
                st.error(f"Error downloading model! HTTP Status: {response.status_code}")
        except Exception as e:
            st.error(f"Download failed: {str(e)}")

download_model()

# ------------------------------
# 🔹 Load the H5 Model in Streamlit
# ------------------------------
@st.cache_resource
def load_model():
    """Load the TensorFlow Keras H5 model."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

# ------------------------------
# 🔹 Preprocess Image Function
# ------------------------------
def preprocess_image(image, target_size=(224, 224)):
    """Convert image to model-compatible format."""
    image = image.convert("RGB")  # Ensure it's RGB
    image = image.resize(target_size)  # Resize to model input size
    image = np.array(image).astype(np.float32) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# ------------------------------
# 🔹 Run Inference on Image
# ------------------------------
def predict(image, model):
    """Perform inference using the Keras model."""
    if model is None:
        return "Error", 0.0

    image = preprocess_image(image)
    predictions = model.predict(image)
    
    classes = ["Gun", "Knife", "Safe"]  # Update class labels based on your model output
    predicted_class = classes[np.argmax(predictions)]
    confidence = np.max(predictions) * 100  # Convert to percentage

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

    model = load_model()  # Load H5 model
    prediction, confidence = predict(image, model)

    st.markdown(f"### 🎯 Prediction: **{prediction}**")
    st.markdown(f"### 🔥 Confidence: **{confidence:.2f}%**")

    # Additional insights
    if prediction == "Safe":
        st.success("✅ No weapon detected. The X-ray is safe.")
    else:
        st.warning("⚠️ Weapon detected! Further inspection recommended.")

st.markdown("---")
st.write("🚀 **Built with Streamlit & TensorFlow**")
