import streamlit as st
import tensorflow as tf
from keras import layers, losses
from PIL import Image
import numpy as np
from model import UNet

WIDTH = 256
HEIGHT = 256
SIZE = (WIDTH, HEIGHT)
WEIGHTS_FILE = "weights/weights.h5"


model = UNet()

inputs = layers.Input((WIDTH, HEIGHT, 3))
model(inputs)
model.load_weights(WEIGHTS_FILE)

# Function to preprocess the uploaded image
def preprocess_image(uploaded_image):
    image = Image.open(uploaded_image).resize(SIZE)
    image_array = np.array(image) / 255.0  # Scale pixel values if required
    image_array = image_array[np.newaxis, ..., :3]  # Add batch dimension and remove alpha channel if present
    return image_array

# Function to postprocess the prediction
def postprocess_prediction(prediction, threshold):
    mask = prediction > threshold  # Threshold the predictions to get binary mask
    mask = mask.squeeze() * 255  # Remove batch dimension and convert to uint8
    return Image.fromarray(mask.astype(np.uint8))


image_1 = st.image('path_to_your_image.png', use_column_width=True)

# Check if the button is clicked
if image_1:
    run_model(image_1)


def run_model(uploaded_file):

    if uploaded_file is not None:

        image = preprocess_image(uploaded_file)
        st.image(image.squeeze(), caption='Uploaded Skin Image', use_column_width=True)

        x1, s1 = model.encoder.block_1(image)
        x2, s2 = model.encoder.block_2(x1)
        x3, s3 = model.encoder.block_3(x2)


