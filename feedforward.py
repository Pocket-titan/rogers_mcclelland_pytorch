from matplotlib import pyplot as plt
from functools import reduce
from tensorflow import keras
from get_data import get_data
from utils import plot_svd
import tensorflow as tf
import numpy as np
import datetime

NUM_REPRESENTATION_UNITS = 6
LEARNING_RATE = 0.1
SAVE_INTERVAL = 100  # save every {SAVE_INTERVAL} epochs
NUM_EPOCHS = 500

# Read the data
[items, attributes, df] = get_data()

num_items = len(items)
num_attributes = len(attributes)

u, s, vT = np.linalg.svd(df.transpose(), full_matrices=False)
# plot_svd(u, s, vT)

features = keras.utils.to_categorical(range(num_items))
targets = df.values

# Create the layers
input_layer = keras.Input(shape=(num_items,), name="items")

hidden_layer = keras.layers.Dense(
    NUM_REPRESENTATION_UNITS,
    activation="sigmoid",
    kernel_initializer=tf.random_uniform_initializer(minval=-1, maxval=1),
    use_bias=False,
    name="representations",
)(input_layer)

output_layer = keras.layers.Dense(
    num_attributes,
    activation="sigmoid",
    name="attributes",
    kernel_initializer=tf.random_uniform_initializer(minval=-1, maxval=1),
    use_bias=False,
)(hidden_layer)

# Compile the model
model = keras.Model(inputs=input_layer, outputs=output_layer)
model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=0.0),
    loss="mean_squared_error",
    metrics=["binary_accuracy", "categorical_crossentropy", "mean_absolute_error",],
)

# Tensorboard things
# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%d-%H%M%S")
# file_writer = tf.summary.create_file_writer(log_dir)
# file_writer.set_as_default()

# tensorboard_callback = keras.callbacks.TensorBoard(
#     log_dir=log_dir, histogram_freq=1, embeddings_freq=50
# )

# Custom metrics
class Metrics(keras.callbacks.Callback):
    def on_train_begin(self, logs={}) -> None:
        self._data = []

    def on_epoch_end(self, epoch, logs={}) -> None:
        if epoch % SAVE_INTERVAL == 0 or epoch == NUM_EPOCHS - 1:
            w_1, *_ = model.get_layer(name="representations").get_weights()
            w_2, *_ = model.get_layer(name="attributes").get_weights()

            a = reduce(
                np.matmul,
                [np.transpose(u), np.transpose(w_2), np.transpose(w_1), np.transpose(vT)],
            )

            plt.imshow(reduce(np.matmul, [u, a, vT]), cmap="bwr")
            plt.show()
        pass

    def on_train_end(self, logs={}) -> None:
        # for singular_values in np.transpose(self._data):
        #     plt.plot(singular_values)
        # plt.show()
        pass


# Fit the model
history = model.fit(
    features,
    targets,
    epochs=NUM_EPOCHS,
    callbacks=[
        # tensorboard_callback ,
        Metrics()
    ],
    batch_size=1,
    shuffle=True,
    verbose=0,
)
