import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from skimage.transform import resize
import xml
import xml.etree.ElementTree as ET
import warnings
import math
from scipy import ndimage

# Satck train data
train_blocks = ['stack_0101.npy', 'stack_0102.npy', 'stack_0203.npy', 'stack_0301.npy']

stack_loc = 'data/Intermediate_train_valid'

# stack all train data together?
all_stacked_train_data = [np.load(os.path.join(stack_loc, file)) for file in train_blocks]

train_stack = np.vstack(all_stacked_train_data)

# save this stack for the use in training the model?
np.save("data/train_input_sub_images/all_train_data.npy", train_stack)