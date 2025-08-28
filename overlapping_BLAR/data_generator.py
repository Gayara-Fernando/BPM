import tensorflow as tf
import numpy as np

# Define an instance for generating batches with Sequence class
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, feature1, target, batch_size=32, shuffle=True):
        """
        Initializes the data generator.

        :param feature1, feature2, feature3, feature4, feature5: The 5 different input features.
        :param labels: The target variable.
        :param batch_size: The size of the batch to generate.
        :param shuffle: Whether to shuffle data after each epoch.
        """
        self.feature1 = feature1
        self.labels = target
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.labels))
        self.on_epoch_end()

    def __len__(self):
        # Number of batches per epoch
        return int(np.ceil(len(self.labels) / self.batch_size))

    def on_epoch_end(self):
        # Shuffling the indexes after each epoch
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        # Generate one batch of data
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Extract the individual features for the batch
        batch_feature1 = self.feature1[batch_indexes]
        
        # Get the corresponding labels for the batch
        batch_labels = self.labels[batch_indexes]

        # Return the individual features and labels
        return (batch_feature1, batch_labels)


# now for prediction 
def batch_predict(model, generator, flatten=True, verbose=False):
    predictions = []
    true_values = []

    for batch_features, batch_labels in generator:
        # Get model predictions for the batch
        batch_predictions = model.predict(batch_features, batch_size=generator.batch_size, verbose=0)
        # pred = batch_predictions.flatten()
        # y = batch_labels.flatten()
        predictions.extend(batch_predictions)
        true_values.extend(batch_labels)
        # print("Batch Predict:")
    print(f"Predictions: {len(predictions)}")
    print(f"True: {len(true_values)}")
    return np.array(predictions), np.array(true_values)