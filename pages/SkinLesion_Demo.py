import streamlit as st
import tensorflow as tf
from keras import layers, losses
from PIL import Image
import numpy as np
from model import UNet

WIDTH = 256
HEIGHT = 256
SIZE = (WIDTH, HEIGHT)
WEIGHTS_FILE = "saved/v1/weights.h5"


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
def postprocess_prediction(prediction):
    mask = prediction > 0.8  # Threshold the predictions to get binary mask
    mask = mask.squeeze() * 255  # Remove batch dimension and convert to uint8
    return Image.fromarray(mask.astype(np.uint8))

# Streamlit interface
st.title('Skin Lesion Identification App')
uploaded_file = st.file_uploader("Choose a skin image...", type="jpg")

if uploaded_file is not None:
    image = preprocess_image(uploaded_file)
    st.image(image.squeeze(), caption='Uploaded Skin Image', use_column_width=True)
    
    prediction = model.predict(image)
    mask_image = postprocess_prediction(prediction)
    st.image(mask_image, caption='Predicted Lesion Mask', use_column_width=True)
    st.image(prediction)