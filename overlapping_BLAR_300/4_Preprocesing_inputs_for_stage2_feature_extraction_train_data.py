import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from data_generator import DataGenerator, batch_predict

# where is this model?
fine_tuned_model = tf.keras.models.load_model("../../../Spring_2024/Bayes_for_comps/TS_bayes_implementation_for_TN/models/trained_gmp_model_dense_32_new.h5")

fine_tuned_model.summary()
# Define the feature extractor model

# feature extractor input
feat_ext_input = fine_tuned_model.input

# feature extractor output 
feat_ext_output = fine_tuned_model.layers[-4].output

# define the model
feature_extractor_model = tf.keras.models.Model(inputs = feat_ext_input, outputs = feat_ext_output)

feature_extractor_model.summary()

# Okay - now what do we need to do?

# I think we do have the inputs stored and arranged in a previous exercise, may be we can use these?

# Where is this location?

sub_windows_of_images_loc = 'data/test_input_sub_images/'

contents_here = os.listdir(sub_windows_of_images_loc)
contents_here.sort()

print(contents_here)

train_save_path = 'data/train_features_non_overlapping/'

def store_train_extracted_features(path_to_inputs, input_feature_file, save_path):
    # load the file
    loaded_input_file = np.load(os.path.join(path_to_inputs, input_feature_file))
    # Let's get the predictions across all time points in a for loop?
    catch_all_preds = []
    for i in range(loaded_input_file.shape[1]):
        time_wise_data = loaded_input_file[:,i,:,:,:]
        extracted_features = feature_extractor_model.predict(time_wise_data)
        catch_all_preds.append(extracted_features)

    # stack these predictions?
    stacked_features = np.stack(catch_all_preds, axis = 1)
    # save the stack of extracted features?
    save_name = 'train_features_block_' + input_feature_file.split('.')[0][-4:] + '.npy'
    np.save(os.path.join(save_path, save_name), stacked_features)
    # also do the sanity check?
    print(np.mean(np.load(os.path.join(save_path, save_name)) == stacked_features))
    return stacked_features

# Easier to do it in a for loop - but verify this tomorrow
all_stacks = []
for i in range(len(contents_here)):
    stack = store_train_extracted_features(sub_windows_of_images_loc, contents_here[i], train_save_path)
    all_stacks.append(stack)
