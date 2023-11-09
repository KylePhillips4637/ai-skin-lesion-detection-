# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random

import streamlit as st
from streamlit.logger import get_logger

import numpy
import tensorflow as tf
from keras import layers, utils, losses
import matplotlib.pyplot as pyplot
from tqdm import tqdm


LOGGER = get_logger(__name__)

random.seed()

RAW_WIDTH = 400
RAW_HEIGHT = 400
RAW_SIZE = (RAW_WIDTH, RAW_HEIGHT)
WIDTH = 256
HEIGHT = 256
SIZE = (WIDTH, HEIGHT)
INPUTS_DIR = "../training_input_images/"
LABELS_DIR = "../training_label_images/"
BATCH_SIZE = 50


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


def run():
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
    
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )

    st.write("# Welcome to our beautiful website")

    st.sidebar.success("options: ")

    st.markdown(
        """
        Streamlit is an open-source app framework built specifically for
        Machine Learning and Data Science projects.
        **👈 Select a demo from the sidebar** to see some examples
        of what Streamlit can do!
        ### Want to learn more?
        - Check out [streamlit.io](https://streamlit.io)
        - Jump into our [documentation](https://docs.streamlit.io)
        - Ask a question in our [community
          forums](https://discuss.streamlit.io)
        ### See more complex demos
        - Use a neural net to [analyze the Udacity Self-driving Car Image
          Dataset](https://github.com/streamlit/demo-self-driving)
        - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
    """
    )


if __name__ == "__main__":
    run()