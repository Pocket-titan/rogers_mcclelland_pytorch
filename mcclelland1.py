"""
Attempt at implementation of the semantic network described in McClelland, McNaughton & O'Reilly (1995) and in
the book "Semantic Cognition" by Rogers & McClelland, using TensorFlow with a Keras backend
"""
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, concatenate
import tensorflow as tf
import datetime
import numpy as np
import pandas as pd
from data import items, attributes, relations, data

# Model parameters
NUM_EPOCHS = 3500
LEARNING_RATE = 0.1
NUM_HIDDEN_UNITS = 15

# Encoding and decoding; tensorflow uses vectors (tensors) and we are using text data -> we need a coder
# Example: the word "pine" maps to the vector (1, 0, 0, 0, 0, 0, 0, 0)
# NOTE: Dimensionality is large this way; efficient enough?
#       Other options is to use unique numbers for each: "pine" |-> 1, "oak" |-> 2, etc.
def encode(word, kind):
    arr = items if kind == 'items' else relations if kind == 'relations' else attributes if kind == 'attributes' else None
    return [1 if w == word else 0 for w in arr]


def decode(vector, kind):
    arr = items if kind == 'items' else relations if kind == 'relations' else attributes if kind == 'attributes' else None
    return [arr[i] for i, n in enumerate(vector) if n == 1][0] or 'Wrong!'


def encode_array(array, kind):
    return [encode(word, kind) for word in array]


def decode_array(array, kind):
    return [decode(vector, kind) for vector in array]

# Helper
def flat(l):
    return [item for sublist in l for item in sublist]

# Initializing the model based on parameters derived from p. 56 of "Semantic Cognition"
items = Input(shape=(8,), name='items')
representations = Dense(8, activation="sigmoid",
                     bias_initializer=tf.constant_initializer(-2), kernel_initializer=tf.keras.initializers.RandomUniform(-0.9, 0.9),  name='representations')(items)
relations = Input(shape=(4,), name='relations')
concatenated = concatenate([representations, relations])
hidden = Dense(NUM_HIDDEN_UNITS, input_shape=(12,), bias_initializer=tf.constant_initializer(-2), kernel_initializer=tf.keras.initializers.RandomUniform(-0.9, 0.9),
          activation="sigmoid", name='hidden')(concatenated)
predictions = Dense(34, activation='sigmoid', name='attributes', kernel_initializer=tf.keras.initializers.RandomUniform(-0.9, 0.9),
                    bias_initializer=tf.constant_initializer(-2))(concatenated)

model = keras.Model(inputs=[items, relations], outputs=predictions)

model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=LEARNING_RATE,),
              loss='mean_squared_error', metrics=['accuracy'])

# Map the different inputs to flat arrays
flat_items = flat([[x[0]]*len(x[2]) for x in data])
flat_relations = flat([[x[1]]*len(x[2]) for x in data])
flat_attributes = flat([x[2] for x in data])

# Make sure we didn't mess this up somewhere
assert len(flat_items) == len(flat_relations) and len(
    flat_items) == len(flat_attributes)

# Encode the strings to vectors and fit the model!
input_items_1 = np.array(encode_array(flat_items, kind='items'))
input_relations_1 = np.array(encode_array(flat_relations, kind='relations'))
attributes = np.array(encode_array(flat_attributes, kind='attributes'))

# Custom callbacks for viewing the model in TensorBoard; launched with command:
# $ tensorboard --logdir logs/fit
# executed in project root dir
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# writer = tf.summary.create_file_writer(log_dir)

class MyCallback(keras.callbacks.Callback):
  def __init__(self, tensorboard_callback):
    self.tensorboard_callback = tensorboard_callback

  def on_train_begin(self, logs=None):
    writer = self.tensorboard_callback._get_writer(self.tensorboard_callback._train_run_name)
    self.writer = writer

  def on_epoch_end(self, epoch, logs=None):
    # On the last epoch, this isn't run - weirdly - so we need the second condition to log a summary
    if epoch % 500 == 0 or epoch == NUM_EPOCHS - 1:
      [weights, biases] = model.get_layer(name='representations').get_weights()
      current_representation = tf.squeeze(weights)
      # Write it
      with self.writer.as_default():
        summary = tf.summary.histogram('Representations', current_representation, step=epoch)
        self.writer.flush()

    def on_train_end(self, logs=None):
      self.writer.close()

tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1, embeddings_freq=50)

my_callback = MyCallback(tensorboard_callback)

history = model.fit([input_items_1, input_relations_1],
          attributes, epochs=NUM_EPOCHS, callbacks=[tensorboard_callback, my_callback])

# Questions:
# 1) How should I handle multi-input networks?
# 2) Terminology is often confusing - "semantic network" == "deep neural network"; what is this in ML language?
#   Classification, clustering?
# 3) Am I handling string data the right way?
# 4) How to actually visualise this? TensorBoard, manual plots of weights (like on p. 86-87 of "Semantic Cognition")?
# -) Office space; any options?
