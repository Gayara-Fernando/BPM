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

block_0101 = '../../../Spring_2024/S_lab_TasselNet/Block_2_TN/Block_2_images_and_xml'

im_height = 768
im_width = 1024
stride = 24
kernel_size = 300
n_channels = 3

all_im_files_0101, mean_0101 =  chose_xml_and_jpeg(block_0101)
print(all_im_files_0101)
print(mean_0101)

read_images = read_all_images(all_im_files_0101, block_0101)

for file in read_images:
    print(file.shape)
    
stack_block_0101 = create_and_stack_subwindows(im_height, im_width, stride, kernel_size, n_channels, read_images)
print(stack_block_0101.shape)

stack_block_0101 = stack_block_0101.astype(np.float16) 
# save this to stack later?
np.save("data/Intermediate_train_valid/stack_0102.npy", stack_block_0101)






