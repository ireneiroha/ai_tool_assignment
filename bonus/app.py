import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Load the trained model
model = tf.keras.models.load_model("mnist_model.h5")

st.title("🧠 MNIST Digit Classifier")
st.write("Upload an image of a handwritten digit (0–9), and the model will predict it.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load and preprocess image
    image = Image.open(uploaded_file).convert("L")  # convert to grayscale
    image = ImageOps.invert(image)  # invert for white digits on black background
    image = image.resize((28, 28))
    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28)

    # If your model expects a channel dimension, uncomment the next line:
    # img_array = img_array[..., np.newaxis]

    st.image(image, caption="Uploaded Image", use_container_width=False)

    # Predict
    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction)

    st.subheader(f"Predicted Digit: {predicted_label}")
    st.bar_chart(prediction[0])

