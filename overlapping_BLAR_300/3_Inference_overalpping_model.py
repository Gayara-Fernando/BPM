#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Data is too large we cannot get the outputs on the notebook itself - submit as a job


# In[ ]:


# Okay, in this work, we will still not be storing the extracted features that we will be using in the stage 2 model, but rather focus on the model performnace, incase we need to report this somewhere.


# In[ ]:


import tensorflow as tf
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from data_generator import DataGenerator, batch_predict


# In[ ]:


# import the model here?


# In[ ]:


model = tf.keras.models.load_model("models/CNN_seq2seq_overlapping_300.keras")


# In[ ]:


model.summary()


# In[ ]:


# we might even need the data generator for the predictions - hold on to this thought for now


# In[ ]:


# locate the test data


# In[ ]:


# input features
input_features_loc = 'data/test_input_sub_images'
input_contents = os.listdir(input_features_loc)
input_contents.sort()


# In[ ]:


input_contents


# In[ ]:


# test targets
out_targets_loc = 'data/test_out_targets'
out_contents = os.listdir(out_targets_loc)
out_contents.sort()


# In[ ]:


out_contents


# In[ ]:


trial_test_features = np.load(os.path.join(input_features_loc, input_contents[0]))


# In[ ]:


trial_test_features.shape


# In[ ]:


trial_test_targets = np.load(os.path.join(out_targets_loc, out_contents[0]))


# In[ ]:


trial_test_targets.shape


# In[ ]:


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


# In[ ]:


print(test_rmse)


# In[ ]:


print(test_mae)


# In[ ]:


print(test_r2)


# In[ ]:


print(test_pearsonr)


# In[ ]:


# save these?


# In[ ]:


np.save(np.array(test_rmse), 'data/3_Inference_overlapping/test_rmse.npy')
np.save(np.array(test_mae), 'data/3_Inference_overlapping/test_mae.npy')
np.save(np.array(test_r2), 'data/3_Inference_overlapping/test_r2.npy')
np.save(np.array(test_pearsonr), 'data/3_Inference_overlapping/test_pearsonr.npy')


# In[ ]:


# We have reported the metrics considering the entire test dataset together as well, let's work on that. Pretty sure we will not be able to get it done in a notebook, let's move to a py script.


# In[ ]:


# Okay, I donot think even HCC has enough memory to deal with all the data we have - so maybe at this point, let's just let it go?

