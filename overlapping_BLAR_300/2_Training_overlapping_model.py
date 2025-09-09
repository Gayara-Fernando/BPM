#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_generator import DataGenerator, batch_predict 


# In[2]:


# import the train andvalidation data?


# In[ ]:


X_train = np.load("data/train_input_sub_images/all_train_data.npy")
X_valid = np.load("data/valid_input_sub_images/all_valid_data.npy")


# In[ ]:


X_train.shape, X_valid.shape


# In[ ]:


y_train = np.load("data/train_out_targets/all_train_targets.npy")
y_valid = np.load("data/valid_out_targets/all_valid_targets.npy")


# In[ ]:


y_train.shape, y_valid.shape


# In[7]:


# We need our CNN model
CNN_model = tf.keras.models.load_model('../../../Spring_2024/Bayes_for_comps/TS_bayes_implementation_for_TN/models/trained_gmp_model_dense_32_new.h5')


# In[8]:


# Define the reduced model

# last layer
output_reduced = CNN_model.layers[-11].output

reduced_model = tf.keras.models.Model(inputs = CNN_model.input, outputs = output_reduced)

reduced_model.summary()


# In[9]:


# okay, now need to add back the dropout, the dense and activation

# add dropout
added_dropout = tf.keras.layers.Dropout(0.5, name = "New_dropout_0")(CNN_model.layers[-11].output)

# add global max pooling
added_flatten = tf.keras.layers.GlobalMaxPooling2D()(added_dropout)

# add dense
added_dense = tf.keras.layers.Dense(64, name = "New_Dense_0")(added_flatten)

# add activation
added_Act = tf.keras.layers.Activation('relu', name = "New_Activation_0")(added_dense)

# add dropout
added_dropout2 = tf.keras.layers.Dropout(0.5, name = "New_dropout_1")(added_Act)

# # add another dense
added_dense_1 = tf.keras.layers.Dense(32, name = "New_Dense_1")(added_dropout2)

# # add activation
added_Act_1 = tf.keras.layers.Activation('relu', name = "New_Activation_1")(added_dense_1)

new_model = tf.keras.models.Model(CNN_model.input, added_Act_1)


# In[10]:


new_model.summary()


# In[11]:


# Pass this thorugh the TD layer, and add the rest of the encoder decoder model for our exercise
input_time = 13
feature_size = 32
output_time = 7

# Encoder
# define encoder input
encoder_input = tf.keras.layers.Input(shape = (input_time, None, None, 3)) 

# pass the feature extractor model through a TD layer
td_model = tf.keras.layers.TimeDistributed(new_model)

td_out = td_model(encoder_input)

# add an lstm to process the input sequence
lstm_layer = tf.keras.layers.LSTM(64, activation = "relu", return_state = True, return_sequences = False)

encoder_outputs, state_h, state_c = lstm_layer(td_out)

# Decoder

# repeat the context vector 7 times
decoder_inp = tf.keras.layers.RepeatVector(output_time)(encoder_outputs)

# define an LSTM for the output sequence
decoder_lstm = tf.keras.layers.LSTM(64, return_sequences = True, activation = 'relu')

decoder_out = decoder_lstm(decoder_inp, initial_state = [state_h, state_c])

# TD dense layer to generate the output sequnces
dense_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(feature_size, activation = 'relu'))

dense_out = dense_layer(decoder_out)

# define the model
CNN_seq2seq_model = tf.keras.models.Model(inputs = encoder_input, outputs = dense_out)


# In[12]:


CNN_seq2seq_model.summary()


# In[13]:


# now, we can freeze the laeyr weights of the CNN model (the CNN layers at least)

# freeze the layers
for layer in CNN_model.layers:
    layer.trainable = False


# In[14]:


# Final model before unfreezing - verify the required weights are frozen?
CNN_seq2seq_model.summary()


# In[15]:


# Okay, now let's train the frozen model

# compile the model
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
CNN_seq2seq_model.compile(loss='mean_squared_error', optimizer=opt, metrics = ['mean_absolute_error'])


# In[16]:


# add early stopping
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', restore_best_weights = True, verbose=1, patience=10)


# In[17]:


# Okay, kernel keeps restarting, I think we need a data generator?


# In[18]:


# # Prepare the train data generator
# train_gen_alt = DataGenerator(train_gcn_feats, train_adj_list, train_keep["Cell_Line"].values.reshape(-1,1), train_keep["Cell_Line"].values.reshape(-1,1),
#                                   train_keep["Cell_Line"].values.reshape(-1,1), train_keep["AUC"].values.reshape(-1,1), batch_size=32)

# # Prepare the validation data generator
# val_gen_alt = DataGenerator(valid_gcn_feats, valid_adj_list, valid_keep["Cell_Line"].values.reshape(-1,1), valid_keep["Cell_Line"].values.reshape(-1,1),
#                                 valid_keep["Cell_Line"].values.reshape(-1,1), valid_keep["AUC"].values.reshape(-1,1), batch_size=32, shuffle = False)


# In[19]:


Train_data_gen = DataGenerator(X_train, y_train, batch_size=32)
Val_data_gen = DataGenerator(X_valid, y_valid, batch_size=32, shuffle = False)


# In[20]:


# Train the model - we need a generator now
history = CNN_seq2seq_model.fit(Train_data_gen, validation_data = Val_data_gen, epochs = 100, callbacks = [es])


# In[21]:


# plot the losses
plt.figure(figsize = (15,7))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Learning curves for the losses")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
# history.history['loss']


# In[22]:


# plot the maes
plt.figure(figsize = (15,7))
plt.plot(history.history['mean_absolute_error'], label='Training MAE')
plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
plt.title("Learning curves for the MAE")
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.show()


# In[23]:


# Now unfreeze the last CNN layer
CNN_model.trainable = True

set_trainable = False

for layer in CNN_model.layers:
    if layer.name == 'conv2d_3':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False


# In[24]:


CNN_seq2seq_model.summary()


# In[25]:


# compile the model
opt = tf.keras.optimizers.Adam(learning_rate=0.0009)
CNN_seq2seq_model.compile(loss='mean_squared_error', optimizer=opt, metrics = ['mean_absolute_error'])


# In[26]:


# Early stopping
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', restore_best_weights = True, verbose=1, patience=10)


# In[27]:


Train_data_gen = DataGenerator(X_train, y_train, batch_size=32)
Val_data_gen = DataGenerator(X_valid, y_valid, batch_size=32, shuffle = False)


# In[28]:


# Train the model - Maybe we do not need a generator for now, let's see
history_new = CNN_seq2seq_model.fit(Train_data_gen, validation_data = Val_data_gen, epochs = 100, callbacks = [es])


# In[29]:


# save this model as keras?

# save this model
model_name_finetuned = 'CNN_seq2seq_overlapping_300.keras'
CNN_seq2seq_model.save('models' + '/' + model_name_finetuned)


# In[30]:


# plot the losses
plt.figure(figsize = (15,7))
plt.plot(history_new.history['loss'], label='Training Loss')
plt.plot(history_new.history['val_loss'], label='Validation Loss')
plt.title("Learning curves for the losses")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
# history.history['loss']


# In[31]:


# plot the maes
plt.figure(figsize = (15,7))
plt.plot(history_new.history['mean_absolute_error'], label='Training MAE')
plt.plot(history_new.history['val_mean_absolute_error'], label='Validation MAE')
plt.title("Learning curves for the MAE")
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.show()


# In[32]:


# The kernel keeps restarting, maybe we need to use a data generator? With tensorflow sequential class

