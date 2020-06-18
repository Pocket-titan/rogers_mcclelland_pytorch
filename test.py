import matplotlib.pyplot as plt
from get_data import get_data
from tensorflow import keras
from functools import reduce
import tensorflow as tf
import pandas as pd
import numpy as np
import datetime

NUM_REPRESENTATION_UNITS = 6
LEARNING_RATE = 0.01
NUM_EPOCHS = 5000

[items, attributes, df] = get_data()

U, S, VT = np.linalg.svd(df.transpose(), full_matrices=False)

num_items = len(items)
num_attributes = len(attributes)


def plot_SVD(u, s, vT) -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, dpi=90)

    s_is_diagonal_matrix = True if type(s[0]) is np.ndarray else False

    im1 = ax1.imshow(u, cmap="bwr", vmin=-1, vmax=1)
    im2 = ax2.imshow(
        s if s_is_diagonal_matrix else np.diag(s),
        cmap="bwr",
        vmin=-np.max(s),
        vmax=np.max(s),
    )
    im3 = ax3.imshow(vT, cmap="bwr", vmin=-1, vmax=1)
    fig.colorbar(im1, ax=ax1)
    fig.colorbar(im2, ax=ax2)
    fig.colorbar(im3, ax=ax3)
    plt.tight_layout()
    plt.show()


features = keras.utils.to_categorical(range(num_items))
targets = df.values

input_layer = keras.Input(shape=(num_items,), name="items")
x = keras.layers.Dense(
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
)(x)

model = keras.Model(inputs=input_layer, outputs=output_layer)
model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=0.0),
    loss="mean_squared_error",
    metrics=["binary_accuracy", "categorical_crossentropy", "mean_absolute_error",],
)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%d-%H%M%S")
file_writer = tf.summary.create_file_writer(log_dir)
file_writer.set_as_default()

tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1, embeddings_freq=50
)


class Metrics(keras.callbacks.Callback):
    def on_train_begin(self, logs={}) -> None:
        self._data = []

    def on_epoch_end(self, epoch, logs={}) -> None:
        if epoch == NUM_EPOCHS - 1:
            W_1, *_ = model.get_layer(name="representations").get_weights()
            W_2, *_ = model.get_layer(name="attributes").get_weights()

            A = reduce(
                np.matmul,
                [np.transpose(U), np.transpose(W_2), np.transpose(W_1), np.transpose(VT)],
            )

            plt.imshow(reduce(np.matmul, [U, A, VT]), cmap="bwr")
            plt.show()
            pass

    def on_train_end(self, logs={}) -> None:
        # for singular_values in np.transpose(self._data):
        #     plt.plot(singular_values)
        # plt.show()
        pass


history = model.fit(
    features,
    targets,
    epochs=NUM_EPOCHS,
    callbacks=[Metrics()],
    batch_size=1,
    shuffle=True,
    verbose=2,
)
