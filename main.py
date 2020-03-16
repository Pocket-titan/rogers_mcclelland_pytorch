"""
Implementation of the semantic network first described in McClelland, McNaughton & O'Reilly (1995) and in
the book "Semantic Cognition" by Rogers & McClelland, using TensorFlow with a Keras backend.
- Inspired by the Pytorch implementation from https://github.com/jeffreyallenbrooks/rogers-and-mcclelland.
- Based mainly on "Semantic Cognition: A Parallel Distributed Processing Approach", Rogers, T. and McClelland, J., 2003
  (henceforth referred to as (2003) or "the paper from 2003")
"""

from tensorflow import keras
from sklearn.decomposition import PCA
from matplotlib import rcParams
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import numpy as np
import scipy as sp
import pandas as pd
import datetime
import io
import os

plt.style.use("plotstyle.mplstyle")

# Model parameters
NUM_EPOCHS = 2000
SAVE_INTERVAL = 500  # save the representaton layer every ${SAVE_INTERVAL} epochs
LEARNING_RATE = 0.1
NUM_HIDDEN_UNITS = 15

# Read and format the data
data = pd.read_csv("data/Rumelhart_livingthings.csv", sep=",")

items = sorted(data.Item.unique())
relations = sorted(data.Relation.unique())
attributes = sorted(data.Attribute.unique())

num_items = len(items)
num_relations = len(relations)
num_attributes = len(attributes)

# Make inputs and outputs
data_table = pd.pivot_table(
    data, values="TRUE", index=["Item", "Relation"], columns=["Attribute"], fill_value=0
).astype(float)
targets = tf.convert_to_tensor(data_table.values)  # output tensor

input_items = keras.utils.to_categorical(range(num_items))  # input tensor [0]
input_relations = keras.utils.to_categorical(range(num_relations))  # input tensor [1]
inputs = [[], []]  # input tensor
for item in input_items:
    for relation in input_relations:
        inputs[0].append(item)
        inputs[1].append(relation)

# Initializing the model
# Item input layer
items_layer = keras.Input(shape=(num_items,), name="items")

# Representations intermediate layer
representations_layer = keras.layers.Dense(
    num_items,
    activation="sigmoid",
    # Pick weights from a random uniform distribution with μ=0, σ^2=0.9 (2003, p. 46)
    kernel_initializer=tf.random_uniform_initializer(minval=-1, maxval=1),
    # While (2003, p. 46) mentions using an untrainable, constant bias of -2, this gave poor results in this model
    # bias_initializer=tf.constant_initializer(-2),
    name="representations",
)(items_layer)

# Relaton input layer
relations_layer = keras.Input(shape=(num_relations,), name="relations")

# Combine the relations input & representations layer
combined_layer = keras.layers.concatenate([representations_layer, relations_layer])

# Hidden intermediate layer
hidden_layer = keras.layers.Dense(
    NUM_HIDDEN_UNITS,
    input_shape=(num_items + num_relations,),
    kernel_initializer=tf.random_uniform_initializer(minval=-1, maxval=1),
    # bias_initializer=tf.constant_initializer(-2),
    activation="sigmoid",
    name="hidden",
)(combined_layer)

# Attributes output layer
attributes_layer = keras.layers.Dense(
    num_attributes,
    activation="sigmoid",
    name="attributes",
    kernel_initializer=tf.random_uniform_initializer(minval=-1, maxval=1),
    # bias_initializer=tf.constant_initializer(-2),
)(hidden_layer)

# Use an SSE (Sum of Squared Error) loss function instead of an MSE (Mean Squared Error) one
# MSE := keras.mean(keras.square(y_pred - y_true), axis=-1)
# This was found to accelerate the process of learning; TODO long term differences MSE/SSE?
def euclidean_distance(y_actual, y_predicted):
    return keras.backend.sum(((y_actual - y_predicted) ** 2), axis=-1)


# Compile the model
model = keras.Model(inputs=[items_layer, relations_layer], outputs=attributes_layer)
model.compile(
    # Use stochastic gradient descent as optimizer, without weight decay or momentum, as mentioned in (2003, p. 46)
    optimizer=keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=0.0),
    # loss="mean_squared_error",
    loss=euclidean_distance,
    metrics=[
        "accuracy",
        "binary_accuracy",
        "categorical_crossentropy",
        "mean_absolute_error",
    ],
)

# Custom callbacks for viewing the model in TensorBoard; which is launched with command:
# $ tensorboard --logdir logs/fit
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(log_dir)
file_writer.set_as_default()


def plot_to_image(figure):
    """Plots the provided representation, then converts it to a PNG image and
    returns it as a tensor. The created figure is closed and inaccessible after this call."""
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.clf()
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def reshape(t):
    """Reshape an array of images to tensors, for viewing in TensorBoard"""
    t = tf.convert_to_tensor(t)
    if t.shape[0] == 1:
        t = tf.squeeze(t, [0])
    elif len(t.shape) > 1 and t.shape[1] == 1:
        t = tf.squeeze(t, [1])

    return tf.stack(t, axis=0)


reps = []  # representation layer array


class SaveRepresentation(keras.callbacks.Callback):
    """Save the representation layer every ${SAVE_INTERVAL} epochs"""

    def on_epoch_end(self, epoch, logs=None):
        # The second condition guarantees this runs on the last epoch (actually NUM_EPOCHS - 1)!
        if epoch % SAVE_INTERVAL == 0 or epoch == NUM_EPOCHS - 1:
            # For every input, find the corresponding activation of the representation layer
            output = model.get_layer(name="representations").call(input_items)
            current_rep = output.numpy()
            reps.append(current_rep)
            # NOTE: this is what I _used_ to do; but the above is correct, I believe:
            # # Get the weights of the representation layer
            # [weights, biases] = model.get_layer(name="representations").get_weights()
            # print(tf.convert_to_tensor(weights))
            # current_rep = tf.squeeze(weights).numpy()
            # reps.append(current_rep)


class LogDistanceMatrix(keras.callbacks.Callback):
    """Callback for logging the distance matrix"""

    def on_train_end(self, logs=None):
        imgs = []
        for i, rep in enumerate(reps):
            df = pd.DataFrame(rep, columns=[i for i in range(len(rep))], index=items)
            rep_df = pd.DataFrame(
                sp.spatial.distance_matrix(df.values, df.values),
                index=df.index,
                columns=df.index,
            )
            fig = plt.figure(figsize=(8, 8))
            plt.title("Distance matrix epoch {}".format(i * SAVE_INTERVAL))
            sns.heatmap(rep_df, cmap="coolwarm", square=True)
            # cmap = sns.cubehelix_palette(8, reverse=True, as_cmap=True)
            plt.xticks(rotation=45)
            img = plot_to_image(fig)
            imgs.append(img)

        # Create image summaries all at once so they are show side-by-side (so step 0 is false; look at plot titles instead)
        with file_writer.as_default():
            distance_matrices = reshape(imgs)
            tf.summary.image(
                "Distance Matrix",
                distance_matrices,
                step=0,
                max_outputs=len(distance_matrices),
            )


class LogDendrogram(keras.callbacks.Callback):
    """Callback for logging the dendrogram"""

    def on_train_end(self, logs=None):
        imgs = []
        for i, rep in enumerate(reps):
            df = pd.DataFrame(rep, columns=[i for i in range(len(rep))], index=items)
            rep_df = pd.DataFrame(
                sp.spatial.distance_matrix(df.values, df.values),
                index=df.index,
                columns=df.index,
            )
            fig = plt.figure(figsize=(8, 8))
            plt.title("Dendrogram epoch {}".format(i * SAVE_INTERVAL))
            linkage = sp.cluster.hierarchy.linkage(
                sp.spatial.distance.squareform(rep_df.values)
            )
            sp.cluster.hierarchy.dendrogram(
                linkage, labels=items, leaf_rotation=90.0, show_contracted=True
            )
            img = plot_to_image(fig)
            imgs.append(img)

        with file_writer.as_default():
            dendrograms = reshape(imgs)
            tf.summary.image(
                "Dendrogram", dendrograms, step=0, max_outputs=len(dendrograms)
            )


class LogPCA(keras.callbacks.Callback):
    """Callback for logging the PCA"""

    def on_train_end(self, logs=None):
        pca = PCA(n_components=3)
        df = pd.DataFrame(
            reps[-1], columns=[i for i in range(len(reps[-1]))], index=items
        )
        pca.fit(df)

        reduced = pca.transform(reps[-1])
        reduced = pd.DataFrame(data=reduced, index=items, columns=["PC1", "PC2", "PC3"])

        D1 = 0
        D2 = 1

        x = reduced.iloc[:, D1]
        y = reduced.iloc[:, D2]
        n = reduced.index

        fig = plt.figure(figsize=(8, 8))
        plt.title("PCA epoch {}".format(NUM_EPOCHS))
        plt.tight_layout()
        plt.scatter(x, y)

        for i, txt in enumerate(n):
            plt.annotate(txt, (x[i], y[i]))

        plt.xlabel("PC{}".format(D1 + 1))
        plt.ylabel("PC{}".format(D2 + 1))

        img = plot_to_image(fig)

        with file_writer.as_default():
            tf.summary.image("PCA", img, step=NUM_EPOCHS, max_outputs=1)


# Create the TensorBoard callback
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1, embeddings_freq=50
)

# Finally, train the model!
history = model.fit(
    inputs,
    targets,
    epochs=NUM_EPOCHS,
    callbacks=[
        tensorboard_callback,
        SaveRepresentation(),
        LogDistanceMatrix(),
        LogDendrogram(),
        LogPCA(),
    ],
    # Still unsure about batch size; should it be 1 for stochastic gradient descent?
    # 32 (the whole set of [input, target] vectors) seems to give better results
    # batch_size=1,
    batch_size=32,
    # steps_per_epoch=???, TODO: find out how steps work; esp. in TensorBoard plots
    shuffle=True,
    verbose=2,
)
