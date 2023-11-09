# %% Imports

import csv
import multiprocessing.dummy as dummy

import numpy
from PIL import Image

WIDTH = 256
HEIGHT = 256
SIZE = (WIDTH, HEIGHT)
BATCH_SIZE = 20
META_DATA = "ISIC-2017_Training_Data_metadata.csv"
TRAINING_DIR = "training_images/"
INPUTS_DIR = f"training_inputs_{WIDTH}_{HEIGHT}/"
BATCHES_DIR = f"training_inputs_{WIDTH}_{HEIGHT}_batched/"

# %% Load index

index = []
with open(META_DATA, mode="r", encoding="utf-8") as meta_file:
    meta_data = csv.reader(meta_file)
    next(meta_data)  # skip header
    for row in meta_data:
        index.append(row[0])

# %% Save images as numpy arrays

def processing_1(image_file):
    image = Image.open(TRAINING_DIR + image_file + ".jpg").resize(SIZE)
    data = numpy.array(image.getdata(), dtype="float16").reshape((
        WIDTH, HEIGHT, 3))
    numpy.save(INPUTS_DIR + image_file, data)

pools = dummy.Pool(4)
pools.map(processing_1, index)

# %% Combine images into batches

def processing_2(batch_num):
    start = BATCH_SIZE * batch_num
    end = min(BATCH_SIZE * (batch_num + 1), len(index))
    batch = []
    for image_index in range(start, end):
        data = numpy.load(INPUTS_DIR + index[image_index] + ".npy")
        batch.append(data)
    batch = numpy.array(batch)
    numpy.save(BATCHES_DIR + "batch_" + str(batch_num), batch)

pools = dummy.Pool(4)
pools.map(processing_2, range(len(index) // BATCH_SIZE))
