# %% Imports

import gc
import random

import numpy
import tensorflow
from tensorflow import keras, data
from keras import layers, datasets, utils, losses, metrics
import matplotlib.pyplot as pyplot
from tqdm import tqdm

# Check for GPU
print(tensorflow.config.list_physical_devices("GPU"))

# For seed sync
random.seed()

WIDTH = 256
HEIGHT = 256
SIZE = (WIDTH, HEIGHT)
INPUTS_DIR = "../training_input_images/"
LABELS_DIR = "../training_label_images/"
BATCH_SIZE = 100

# %% Load dataset

seed = 100

input_images = utils.image_dataset_from_directory(
    INPUTS_DIR,
    labels=None,
    label_mode=None,
    image_size=SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    seed=seed
)

label_images = utils.image_dataset_from_directory(
    LABELS_DIR,
    labels=None,
    label_mode=None,
    image_size=SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    seed=seed
)

# %% Model

class EncoderBlock(layers.Layer):
    def __init__(self, n_filters, max_pooling=True):
        super().__init__()
        self.max_pooling = max_pooling
        self.conv_1 = layers.Conv2D(n_filters, kernel_size=3, activation='relu', padding='same')
        self.conv_2 = layers.Conv2D(n_filters, kernel_size=3, activation='relu', padding='same')
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
    def __init__(self, n_filters, width, height):
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
    
    def call(self, inputs, skip):
        x = self.up_sampling(inputs)
        x = self.concat([x, skip])
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x

class Encoder(layers.Layer):
    def __init__(self):
        super().__init__()
        self.block_1 = EncoderBlock(16)
        self.block_2 = EncoderBlock(32)
        self.block_3 = EncoderBlock(64)
        self.block_4 = EncoderBlock(128, max_pooling=False)
    
    def call(self, inputs):
        x, skip_1 = self.block_1(inputs)
        x, skip_2 = self.block_2(x)
        x, skip_3 = self.block_3(x)
        x, _ = self.block_4(x)
        return x, skip_1, skip_2, skip_3

class Decoder(layers.Layer):
    def __init__(self):
        super().__init__()
        self.block_1 = DecoderBlock(64, WIDTH//4, HEIGHT//4)
        self.block_2 = DecoderBlock(32, WIDTH//2, HEIGHT//2)
        self.block_3 = DecoderBlock(16, WIDTH, HEIGHT)
        self.conv_out = layers.Conv2DTranspose(1, kernel_size=3, activation='sigmoid', padding='same')
    
    def call(self, inputs, skip_1, skip_2, skip_3):
        x = self.block_1(inputs, skip_3)
        x = self.block_2(x, skip_2)
        x = self.block_3(x, skip_1)
        x = self.conv_out(x)
        return x

class UNet(keras.Model):
    def __init__(self):
        super().__init__()
        self.rescaling = layers.Rescaling(1./255)
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def call(self, inputs, training=False):
        x = self.rescaling(inputs)
        x, skip_1, skip_2, skip_3 = self.encoder(x)
        outputs = self.decoder(x, skip_1, skip_2, skip_3)
        return outputs

# %%

model = UNet()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss=losses.BinaryFocalCrossentropy(
        apply_class_balancing=True
    ),
    metrics=[
        "BinaryAccuracy", 
        "BinaryIoU"
    ]
)

# %%

EPOCH = 10

for epoch in range(EPOCH):
    print(f"=== Epoch {epoch} ===")
    for (inputs, labels) in data.Dataset.zip((input_images, label_images)):
        labels = tensorflow.cast(labels < 0.5, dtype="int8")
        model.fit(inputs, labels, verbose=1)
