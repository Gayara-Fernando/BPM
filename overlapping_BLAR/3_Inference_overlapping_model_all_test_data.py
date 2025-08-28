# In this script we will stack all data together and get the metrics printed. Should we save these values JIC?

import tensorflow as tf
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from data_generator import DataGenerator, batch_predict

# get the trained model?
overlapping_model = tf.keras.models.load_model("models/CNN_seq2seq_overlapping.keras")

# input features
input_features_loc = 'data/test_input_sub_images'
input_contents = os.listdir(input_features_loc)
input_contents.sort()

# test targets
out_targets_loc = 'data/test_out_targets'
out_contents = os.listdir(out_targets_loc)
out_contents.sort()

# list of all test features across the blocks
all_test_features_list = [np.load(os.path.join(input_features_loc, file)) for file in input_contents]

# stack all data together
all_test_features = np.vstack(all_test_features_list)
print(all_test_features.shape)

# load and stack targets
all_test_targets_list = [np.load(os.path.join(out_targets_loc, file)) for file in out_contents]
# stack all data together
all_test_targets = np.vstack(all_test_targets_list)
print(all_test_targets.shape)

# create the test data generator
all_test_data_gen = DataGenerator(all_test_features, all_test_targets, batch_size = 32, shuffle = False)

# get all test predictions
all_test_preds, all_test_targets_alt = batch_predict(overlapping_model, all_test_data_gen, flatten=True, verbose=False)

print("Shape of predicted values and targets: ", all_test_preds.shape, all_test_targets_alt.shape)

print("verify the targets through the batch predict function and original targets are the same: ", np.mean(all_test_targets_alt == all_test_targets))

# flatten these vectors
all_test_targets_flatten = all_test_targets_alt.flatten()
all_test_preds_flatten = all_test_preds.flatten()

# compute metrics
all_test_rmse = np.sqrt(mean_squared_error(all_test_targets_flatten, all_test_preds_flatten))
all_test_mae = mean_absolute_error(all_test_targets_flatten, all_test_preds_flatten)
all_test_r2_score = r2_score(all_test_targets_flatten, all_test_preds_flatten)
all_test_pearsonr = pearsonr(all_test_targets_flatten, all_test_preds_flatten)

# print the metrics
print("rmse: ", all_test_rmse)
print("mae: ", all_test_mae)
print("r2_score: ", all_test_r2_score)
print("pearson corr: ", all_test_pearsonr)

