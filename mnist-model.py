import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Load the trained model
model = tf.keras.models.load_model("mnist_model.keras")

st.title("MNIST Digit Classifier")
st.write("Upload a digit image (0–9) or draw one below to see what the model predicts.")

uploaded_file = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # convert to grayscale
    image = ImageOps.invert(image)  # invert colors if background is white
    image = image.resize((28, 28))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    st.image(image, caption="Uploaded Digit", use_container_width=True)

    prediction = model.predict(img_array)
    pred_class = np.argmax(prediction)
    confidence = np.max(prediction)

    st.subheader(f"Prediction: **{pred_class}**")
    st.write(f"Confidence: **{confidence:.2f}**")
