from tensorflow import keras
import tensorflow as tf
from matplotlib import pyplot as plt
from functools import reduce
from get_data import get_data
from utils import plot_svd
import numpy as np
import datetime
import random

NUM_REPRESENTATION_UNITS = 6
LEARNING_RATE = 0.1
SAVE_INTERVAL = 100  # save every {SAVE_INTERVAL} epochs
NUM_EPOCHS = 500
BATCH_SIZE = 1

# Read the data
[items, attributes, df] = get_data()

num_items = len(items)
num_attributes = len(attributes)

features = keras.utils.to_categorical(range(num_items))
targets = df.values

# features = np.reshape(features, (features.shape[0], 1, features.shape[1]))

# Create the layers
input_layer = keras.Input(shape=(num_items,), batch_size=BATCH_SIZE, name="items")

reshaped = keras.layers.Reshape((1, num_items))(input_layer)

recurrent_layer = keras.layers.SimpleRNN(
    NUM_REPRESENTATION_UNITS,
    activation="sigmoid",
    stateful=True,  # allow state to persist between batches (since they are size 1 rn)
    return_sequences=False,
    kernel_initializer=keras.initializers.RandomUniform(minval=-1, maxval=1),
    recurrent_initializer="random_uniform",
    bias_initializer="zeros",
)(reshaped)

output_layer = keras.layers.Dense(
    num_attributes,
    activation="sigmoid",
    name="attributes",
    kernel_initializer=keras.initializers.RandomUniform(minval=-1, maxval=1),
    bias_initializer="zeros",
)(recurrent_layer)

# Compile the model
model = keras.Model(inputs=input_layer, outputs=output_layer)
model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=0.0),
    loss="mean_squared_error",
    metrics=["accuracy"],
)

# Tensorboard things
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%d-%H%M%S")
file_writer = tf.summary.create_file_writer(log_dir)
file_writer.set_as_default()

tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1, embeddings_freq=50
)

model.summary()

# Fit the model
history = model.fit(
    features,
    targets,
    epochs=NUM_EPOCHS,
    callbacks=[
        tensorboard_callback,
        # Metrics()
    ],
    batch_size=BATCH_SIZE,
    shuffle=True,
    verbose=2,
)
