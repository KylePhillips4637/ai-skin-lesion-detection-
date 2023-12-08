import os

import streamlit as st
import tensorflow as tf
from keras import layers, losses
from PIL import Image
import numpy

from model import ClassifierNet

model = ClassifierNet()
model((layers.Input((256, 256, 3)), layers.Input(3)))
model.load_weights('weights/weights_part_3.h5')

cols = st.columns(7)
images = os.listdir('Images/')

for i in range(7):
    image_file = 'Images/' + images[i]
    
    with cols[i]:
        st.image(image_file)
        button = st.button('Choose', key=str(i))
    
    if button:
        image = Image.open(image_file)
        image = image.resize((256, 256))
        image = numpy.array(image.getdata()).reshape((1, 256, 256, 3)) / 255
        outputs = model((image, numpy.array(((25., 1., 0.),))))
        st.text('None: ' + str(outputs[0][0].numpy()))
        st.text('Type 1: ' + str(outputs[0][1].numpy()))
        st.text('Type 2: ' + str(outputs[0][2].numpy()))
