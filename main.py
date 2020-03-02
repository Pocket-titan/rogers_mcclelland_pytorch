"""
Implementation of the semantic network described in McClelland, McNaughton & O'Reilly (1995) and in
the book "Semantic Cognition" by Rogers & McClelland, using TensorFlow with a Keras backend.
Adapted from Pytorch from https://github.com/jeffreyallenbrooks/rogers-and-mcclelland.
"""

from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, concatenate
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import numpy as np
import scipy as sp
import pandas as pd
import datetime
import io

# Model parameters
NUM_EPOCHS = 3500
LEARNING_RATE = 0.1
NUM_HIDDEN_UNITS = 15

# Read and format the data
data = pd.read_csv('data/Rumelhart_livingthings.csv', sep=',')

items = sorted(data.Item.unique())
relations = sorted(data.Relation.unique())
attributes = sorted(data.Attribute.unique())

num_items = len(items)
num_relations = len(relations)
num_attributes = len(attributes)

# Make inputs and outputs
data_table = pd.pivot_table(data, values='TRUE', index=[
                            'Item', 'Relation'], columns=['Attribute'], fill_value=0).astype(float)
output_tensor = tf.convert_to_tensor(data_table.values)

# Combine inputs in array, [input_items_tensor, input_relations_tensor]
input_tensor = [np.zeros([num_items*num_relations, num_items]),
                np.zeros([num_items*num_relations, num_relations])]

# Create our input tensor for training
count = 0
for living_thing_id in tf.eye(num_items):
    for relation_id in tf.eye(num_relations):
        input_tensor[0][count, :] = living_thing_id  # 1 or 0
        input_tensor[1][count, :] = relation_id  # 1 or 0
        count += 1
input_tensor = [tf.convert_to_tensor(x) for x in input_tensor]

# Initializing the model
input_items = Input(shape=(num_items,), name='items')
x = Dense(num_items, activation="sigmoid",
          bias_initializer=tf.constant_initializer(-2), kernel_initializer=tf.keras.initializers.RandomUniform(-0.9, 0.9),
          name='representations')(input_items)
input_relations = Input(shape=(num_relations,), name='relations')
x = concatenate([x, input_relations])
x = Dense(NUM_HIDDEN_UNITS, input_shape=(num_items+num_relations,), bias_initializer=tf.constant_initializer(-2), kernel_initializer=tf.keras.initializers.RandomUniform(-0.9, 0.9),
          activation="sigmoid", name='hidden')(x)
predictions = Dense(num_attributes, activation='sigmoid', name='attributes', kernel_initializer=tf.keras.initializers.RandomUniform(-0.9, 0.9),
                    bias_initializer=tf.constant_initializer(-2))(x)

model = keras.Model(inputs=[input_items, input_relations], outputs=predictions)

model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=LEARNING_RATE,),
              loss='mean_squared_error', metrics=['accuracy'])

# Custom callbacks for viewing the model in TensorBoard; which is launched with command:
# $ tensorboard --logdir logs/fit
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def data_to_image(rep_df, epoch):
  """Plots the provided representation, then converts it to a PNG image and
  returns it. The created figure is closed and inaccessible after this call."""
  figure = plt.figure(figsize=(8, 8))
  cmap = sns.diverging_palette(220, 10, as_cmap=True)
  image = sns.heatmap(rep_df, cmap=cmap, square=True)
  plt.title('Distance matrix epoch {}'.format(epoch))
  plt.tight_layout()
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image

class CustomLogger(keras.callbacks.Callback):
    """
    Custom callback which runs during the model training session and logs custom metrics
    """
    def __init__(self, tensorboard_callback):
        self.tensorboard_callback = tensorboard_callback
        self.reps = []

    def on_train_begin(self, logs=None):
        writer = self.tensorboard_callback._get_writer(
            self.tensorboard_callback._train_run_name)
        self.writer = writer

    def on_epoch_end(self, epoch, logs=None):
        # On the last epoch, this isn't run - weirdly - so we need the second condition to log a summary
        if epoch % 500 == 0 or epoch == NUM_EPOCHS - 1:
            [weights, biases] = model.get_layer(
                name='representations').get_weights()
            current_representation = tf.squeeze(weights)
            df = pd.DataFrame(current_representation.numpy(), columns=[
                              i for i in range(len(current_representation))], index=items)
            rep_df = pd.DataFrame(sp.spatial.distance_matrix(
                df.values, df.values), index=df.index, columns=df.index)

            # Plot it
            final_image = data_to_image(rep_df, epoch)

            # And write it
            with self.writer.as_default():
                summary = tf.summary.image("Distance Matrix", final_image, step=epoch)
                self.writer.flush()

        def on_train_end(self, logs=None):
            self.writer.close()

tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1, embeddings_freq=50)

my_callback = CustomLogger(tensorboard_callback)

history = model.fit(input_tensor,
                    output_tensor, epochs=NUM_EPOCHS, callbacks=[tensorboard_callback, my_callback])
