# %% Imports

import gc

import numpy
import tensorflow
from tensorflow import keras, data
from keras import layers
import matplotlib.pyplot as pyplot
from tqdm import tqdm

# Check for GPU
print(tensorflow.config.list_physical_devices("GPU"))

BATCH_COUNT = 100
WIDTH = 256
HEIGHT = 256
SIZE = (WIDTH, HEIGHT)
INPUT_BATCHES_DIR = f"training_inputs_{WIDTH}_{HEIGHT}_batched/"
LABEL_BATCHES_DIR = f"training_labels_{WIDTH}_{HEIGHT}_batched/"
INPUTS_DIR = f"training_inputs_{WIDTH}_{HEIGHT}/"
LABELS_DIR = f"training_labels_{WIDTH}_{HEIGHT}/"

# %% Load index

index = []
for idx in range(BATCH_COUNT):
    index.append("batch_" + str(idx))
    
print(index)

# %% Load dataset



# %% Model

class EncoderBlock(layers.Layer):
    def __init__(self, n_filters, max_pooling=True):
        super().__init__()
        self.max_pooling = max_pooling
        self.conv_1 = layers.Conv2D(n_filters, kernel_size=3, activation='relu', padding='same')
        if max_pooling:
            self.max_pool = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
    
    def call(self, inputs):
        x = self.conv_1(inputs)
        skip = x
        if self.max_pooling:
            x = self.max_pool(x)
        return x, skip

class DecoderBlock(layers.Layer):
    def __init__(self, n_filters, width, height):
        super().__init__()
        self.up_sampling = layers.UpSampling2D(size=(2, 2))
        self.crop = layers.CenterCrop(width, height)
        self.concat = layers.Concatenate(axis=3)
        self.conv_1 = layers.Conv2DTranspose(n_filters, kernel_size=3, activation='relu', padding='same')
    
    def call(self, inputs, skip):
        x = self.up_sampling(inputs)
        x = self.crop(x)
        x = self.concat([x, skip])
        x = self.conv_1(x)
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
        self.block_1 = DecoderBlock(32, WIDTH//4, HEIGHT//4)
        self.block_2 = DecoderBlock(16, WIDTH//2, HEIGHT//2)
        self.block_3 = DecoderBlock(8, WIDTH, HEIGHT)
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
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def call(self, inputs, training=False):
        x, skip_1, skip_2, skip_3 = self.encoder(inputs)
        outputs = self.decoder(x, skip_1, skip_2, skip_3)
        return outputs

# %%

model = UNet()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["binary_accuracy"]
)

# %%

hists = []
for epoch in range(10):
    for batch in tqdm(range(BATCH_COUNT)):
        inputs = numpy.load(INPUT_BATCHES_DIR + "batch_" + str(batch) + ".npy")
        labels = numpy.load(LABEL_BATCHES_DIR + "batch_" + str(batch) + ".npy")
        hist = model.fit(inputs, labels, verbose=0)
        hists.append(hist)
    gc.collect()
