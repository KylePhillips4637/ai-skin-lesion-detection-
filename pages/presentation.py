import streamlit as st
import tensorflow as tf
import keras
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

image = keras.utils.load_img("__", target_size=(WIDTH, HEIGHT))
image = image.reshape((1, WIDTH, HEIGHT, 3))

x1, s1 = model.encoder.block_1(image)
x2, s2 = model.encoder.block_2(x1)
x3, s3 = model.encoder.block_3(x2)


