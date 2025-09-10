import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from data_generator import DataGenerator, batch_predict

overlapping_model = tf.keras.models.load_model("models/CNN_seq2seq_overlapping_300.keras") 
overlapping_model.summary()

# input features
input_features_loc = 'data/test_input_sub_images/'
input_contents = os.listdir(input_features_loc)
input_contents.sort()

print(input_contents)

# small sanity check
try_shape = np.load(os.path.join(input_features_loc, input_contents[0]))
print(try_shape.shape)

# test targets
out_targets_loc = 'data/test_out_targets'
out_contents = os.listdir(out_targets_loc)
out_contents.sort()

print(out_contents)

for_sanity_check = []
for i in range(len(input_contents)):
    # load the np file
    test_features = np.load(os.path.join(input_features_loc, input_contents[i]))
    # load targets
    test_targets = np.load(os.path.join(out_targets_loc, out_contents[i]))
    # print shape of the loaded file
    print(test_features.shape)
    # predicted_values
    # define the test data generator here
    test_data_gen = DataGenerator(test_features, test_targets, batch_size, shuffle=False)
    # Use the batch predictions to generate the predictions
    test_preds, test_targets_alt = batch_predict(overlapping_model, test_data_gen, flatten=True, verbose=True)
    print(np.mean(test_targets == test_targets_alt))
    print(test_preds.shape)
    for_sanity_check.append(test_preds)
    # save these values?
    # name
    loc_name = 'data/predicted_sequences_from_stage_1/' + 'pred_values_blk_' + input_contents[i].split('.')[0][-4:] + '.npy'
    np.save(loc_name, test_preds)