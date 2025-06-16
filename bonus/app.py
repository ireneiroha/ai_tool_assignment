import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_model.h5")

model = load_model()

st.title("MNIST Digit Classifier")
st.write("Upload a 28x28 image of a handwritten digit (0–9). The model will predict the digit.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        # Load and preprocess image
        image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
        image = ImageOps.invert(image)  # Invert colors
        image = image.resize((28, 28))  # Resize to match model input
        img_array = np.array(image).astype("float32") / 255.0  # Normalize
        img_array = img_array.reshape(1, 28, 28, 1)

        st.image(image, caption="Uploaded Digit", width=150)

        # Predict
        prediction = model.predict(img_array)
        pred_label = np.argmax(prediction, axis=1)[0]
        st.success(f"Predicted Digit: {pred_label}")
    except Exception as e:
        st.error(f"Error processing image: {e}")
else:
    st.info("Please upload a digit image.")

st.caption("Model: Simple CNN trained on MNIST.")


