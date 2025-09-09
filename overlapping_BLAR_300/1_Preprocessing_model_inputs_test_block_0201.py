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

def chose_xml_and_jpeg(file_location):
    # list all files in location
    list_of_all_files = os.listdir(file_location)
    # sort files
    list_of_all_files.sort()
    # separate xml and jpeg files
    all_xml_files = [file for file in list_of_all_files if file.split('.')[-1] == 'xml']
    all_xml_files.sort()
    all_jpeg_files = [file for file in list_of_all_files if file not in all_xml_files]
    all_jpeg_files.sort()
    # get the final 20 files
    chosen_xml_files = all_xml_files[-20:]
    chosen_jpeg_files = all_jpeg_files[-20:]
    # make sure the xml and jpeg files correspond to each other?
    mean = np.mean([file.split('.')[0] for file in chosen_xml_files] == [file.split('.')[0] for file in chosen_jpeg_files])

    # chose the required files only - notice that for the inputs we do not need to create their density maps - therefore we do not need the xml files
    task_specific_image_files = chosen_jpeg_files[:13]
    return(task_specific_image_files, mean)

# write a function for reading the images

def read_all_images(file_names, im_path):
    all_images = []
    for file in file_names:
        joined_path = os.path.join(im_path, file)
        read_image = plt.imread(joined_path)
        plt.imshow(read_image)
        plt.show()
        all_images.append(read_image)

    return all_images

# Do a function for this

def create_and_stack_subwindows(im_height, im_width, stride, kernel_size, n_channels, im_list):
    # initiate an empty list
    catch_all_subwindows = []
    for i in range(0, im_height, stride):
        for j in range(0, im_width, stride):
            local_list_at_subimage_sequence_level= []
            for im_file in im_list:
                chosen_window = im_file[i:i+kernel_size, j:j+kernel_size, :]
                # resize the window
                chosen_window = resize(chosen_window, (kernel_size, kernel_size ,n_channels))
                local_list_at_subimage_sequence_level.append(chosen_window)
            catch_all_subwindows.append(local_list_at_subimage_sequence_level)
    # stack all these together
    Stacked_subwindows = np.stack(catch_all_subwindows)
    return(Stacked_subwindows)

block_0103 = '../../../Spring_2024/S_lab_TasselNet/Block_7_TN/Block_7_images_and_xml'

# get all image files
all_im_files_0103, mean_0103 =  chose_xml_and_jpeg(block_0103)

print(all_im_files_0103, mean_0103)

# read the images and plot them
read_images_0103 = read_all_images(all_im_files_0103, block_0103)
print(read_images_0103[0].shape)

im_height = read_images_0103[0].shape[0]
im_width = read_images_0103[0].shape[1]
stride = 24
kernel_size = 300
n_channels = 3
stack_block_0103 = create_and_stack_subwindows(im_height, im_width, stride, kernel_size, n_channels, read_images_0103)

# examine the shape for this
print(stack_block_0103.shape)

np.save("data/test_input_sub_images/test_data_blk_0201.npy", stack_block_0103)

# sanity check
load_test_blk_0103 = np.load("data/test_input_sub_images/test_data_blk_0201.npy")
np.mean(load_test_blk_0103 == stack_block_0103)