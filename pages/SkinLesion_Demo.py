import streamlit as st
import tensorflow as tf
from keras import layers, losses
from PIL import Image
import numpy as np

WIDTH = 256
HEIGHT = 256
SIZE = (WIDTH, HEIGHT)
MODEL_FILE = "models/trained"
WEIGHTS_FILE = "saved/v1/weights.h5"

class EncoderBlock(layers.Layer):
    def __init__(self, n_filters, max_pooling=True):
        super().__init__()
        self.max_pooling = max_pooling
        self.conv_1 = layers.Conv2D(
            n_filters,
            kernel_size=3,
            activation='relu',
            padding='same'
        )
        self.conv_2 = layers.Conv2D(
            n_filters,
            kernel_size=3,
            activation='relu',
            padding='same'
        )
        if max_pooling:
            self.max_pool = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
    
    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.conv_2(x)
        skip = x
        if self.max_pooling:
            x = self.max_pool(x)
        return x, skip

class DecoderBlock(layers.Layer):
    def __init__(self, n_filters):
        super().__init__()
        self.up_sampling = layers.UpSampling2D(size=(2, 2))
        self.concat = layers.Concatenate(axis=3)
        self.conv_1 = layers.Conv2DTranspose(
            n_filters,
            kernel_size=3,
            activation='relu',
            padding='same'
        )
        self.conv_2 = layers.Conv2DTranspose(
            n_filters,
            kernel_size=3,
            activation='relu',
            padding='same'
        )
    
    def call(self, inputs):
        inputs, skip = inputs
        x = self.up_sampling(inputs)
        x = self.concat([x, skip])
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x

class Encoder(layers.Layer):
    def __init__(self):
        super().__init__()
        self.block_1 = EncoderBlock(8)
        self.block_2 = EncoderBlock(16)
        self.block_3 = EncoderBlock(32)
        self.block_4 = EncoderBlock(64, max_pooling=False)
    
    def call(self, inputs):
        x, skip_1 = self.block_1(inputs)
        x, skip_2 = self.block_2(x)
        x, skip_3 = self.block_3(x)
        x, _ = self.block_4(x)
        return x, skip_1, skip_2, skip_3

class Decoder(layers.Layer):
    def __init__(self):
        super().__init__()
        self.block_1 = DecoderBlock(32)
        self.block_2 = DecoderBlock(16)
        self.block_3 = DecoderBlock(8)
        self.conv_out = layers.Conv2DTranspose(1, kernel_size=3, activation='sigmoid', padding='same')
    
    def call(self, inputs):
        encoded, skip_1, skip_2, skip_3 = inputs
        x = self.block_1((encoded, skip_3))
        x = self.block_2((x, skip_2))
        x = self.block_3((x, skip_1))
        x = self.conv_out(x)
        return x

class UNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def call(self, inputs, training=False):
        x = self.encoder(inputs)
        outputs = self.decoder(x)
        return outputs

model = UNet()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=losses.BinaryFocalCrossentropy(
        apply_class_balancing=True
    ),
    metrics=[
        "BinaryAccuracy", 
        "BinaryIoU"
    ]
)

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