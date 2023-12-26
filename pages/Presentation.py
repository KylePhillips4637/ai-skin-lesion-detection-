import streamlit as st
import tensorflow as tf
from keras import layers, losses
from PIL import Image
import numpy as np
from model import UNet
from matplotlib import pyplot

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

def display_matrices(mat, rows, cols):
    fig, axs = pyplot.subplots(rows, cols, squeeze=False)
    for idx in range(rows*cols):
        axs[idx%rows, idx//rows].matshow(mat[0, :, :, idx])
        axs[idx%rows, idx//rows].tick_params(
            axis='both', which='both',
            bottom=False, labelbottom=False,
            top=False, labeltop=False,
            left=False, labelleft=False,
            right=False, labelright=False,
        )
    return fig


image = st.file_uploader("Choose a skin image...", type="jpg")

if image is not None:
    st.text('Original image:')
    st.image(image)

    st.text('Scaled image:')
    processed_image = preprocess_image(image)
    st.image(processed_image)

    st.text('Feature images after encoder\'s first block:')
    x1, s1 = model.encoder.block_1(processed_image)
    fig = display_matrices(s1, 2, 4)
    st.pyplot(fig)

    st.text('Feature images after encoder\'s second block:')
    x2, s2 = model.encoder.block_2(x1)
    fig = display_matrices(s2, 3, 4)
    st.pyplot(fig)

    st.text('Feature images after encoder\'s third block:')
    x3, s3 = model.encoder.block_3(x2)
    fig = display_matrices(s3, 4, 4)
    st.pyplot(fig)

    st.text('Feature images after encoder\'s fourth block:')
    x4, s4 = model.encoder.block_4(x3)
    fig = display_matrices(s4, 4, 8)
    st.pyplot(fig)

    st.text('Feature images after decoder\'s first block')
    x5 = model.decoder.block_1([x4, s3])
    fig = display_matrices(x5, 4, 4)
    st.pyplot(fig)

    st.text('Feature images after decoder\'s second block')
    x6 = model.decoder.block_2([x5, s2])
    fig = display_matrices(x6, 3, 4)
    st.pyplot(fig)

    st.text('Feature images after decoder\'s third block')
    x7 = model.decoder.block_3([x6, s1])
    fig = display_matrices(x7, 2, 4)
    st.pyplot(fig)
            
    st.text('Output segmented image:')
    x8 = model.decoder.conv_out(x7)
    fig = display_matrices(x8, 1, 1)
    st.pyplot(fig)
