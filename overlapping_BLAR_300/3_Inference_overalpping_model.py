# Data is too large we cannot get the outputs on the notebook itself - submit as a job

# Import all libraries
import tensorflow as tf
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from data_generator import DataGenerator, batch_predict

# import the model here?
model = tf.keras.models.load_model("models/CNN_seq2seq_overlapping_300.keras")
model.summary()

# input features
input_features_loc = 'data/test_input_sub_images'
input_contents = os.listdir(input_features_loc)
input_contents.sort()

print(input_contents)

# test targets
out_targets_loc = 'data/test_out_targets'
out_contents = os.listdir(out_targets_loc)
out_contents.sort()

print(out_contents)

trial_test_features = np.load(os.path.join(input_features_loc, input_contents[0]))
print(trial_test_features.shape)

trial_test_targets = np.load(os.path.join(out_targets_loc, out_contents[0]))
print(trial_test_targets.shape)


# Get preds in a loop
# now how to proceed? We may need to do a batch predict now using the generator 

batch_size = 32
test_rmse = []
test_mae = []
test_r2 = []
test_pearsonr = []
preds = []
for i in range(len(input_contents)):
    # load the features
    test_features = np.load(os.path.join(input_features_loc, input_contents[i]))
    # load targets
    test_targets = np.load(os.path.join(out_targets_loc, out_contents[i]))
    test_data_gen = DataGenerator(test_features, test_targets, batch_size, shuffle=False)
    # Use the batch predictions to generate the predictions
    test_preds, test_targets_alt = batch_predict(model, test_data_gen, flatten=True, verbose=True)
    preds.append(test_preds)
    print(np.mean(test_targets == test_targets_alt))
    # compute the test scores, I think we need to flatten these before computing the scores - or can use tf, but the answers are going to be the same
    test_preds_flatten = test_preds.flatten()
    test_targets_flatten = test_targets_alt.flatten()
    mae = mean_absolute_error(test_targets_flatten, test_preds_flatten)
    test_mae.append(mae)
    rmse = np.sqrt(mean_squared_error(test_targets_flatten, test_preds_flatten))
    test_rmse.append(rmse)
    rsquare = r2_score(test_targets_flatten, test_preds_flatten)
    test_r2.append(rsquare)
    pearsonr_score = pearsonr(test_targets_flatten, test_preds_flatten)[0]
    test_pearsonr.append(pearsonr_score)

print(test_rmse)
print(test_mae)
print(test_r2)
print(test_pearsonr)

np.save('data/3_Inference_overlapping/test_rmse.npy', np.array(test_rmse))
np.save('data/3_Inference_overlapping/test_mae.npy', np.array(test_mae))
np.save('data/3_Inference_overlapping/test_r2.npy', np.array(test_r2))
np.save('data/3_Inference_overlapping/test_pearsonr.npy', np.array(test_pearsonr))




