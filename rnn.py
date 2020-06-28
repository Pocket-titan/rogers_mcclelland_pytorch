from utils import to_categorical, do_svd, plot_svd
from learning_metrics import plot_SVD
from matplotlib import pyplot as plt
from collections import OrderedDict
from torch.utils import tensorboard
from get_data import get_data
from functools import reduce
import numpy as np
import scipy as sp
import datetime
import torch

LEARNING_RATE = 1e-2
NUM_HIDDEN_UNITS = 8
LOG_INTERVAL = 500  # epochs
NUM_EPOCHS = 5000
BATCH_SIZE = 64

[items, attributes, df] = get_data()

NUM_ITEMS = len(items)
NUM_ATTRIBUTES = len(attributes)

# Create feature and target tensors
x = torch.tensor(to_categorical(range(NUM_ITEMS)), dtype=torch.float32)  # features
y = torch.tensor(df.values, dtype=torch.float32)  # targets

# Prepare the model
class RNN(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(RNN, self).__init__()
        self.layer = torch.nn.RNN(input_size, hidden_size, num_layers)

    def forward():
        pass

    def init_hidden():
        pass


# Note: we have twice the layers compared to the Keras model, since keras layers (like
# keras.layers.Dense) do _both_ the Linear & activation parts; PyTorch doesn't.
model = torch.nn.Sequential(
    OrderedDict(
        [
            ("representations", RNN(NUM_ITEMS, NUM_HIDDEN_UNITS)),
            ("attributes", torch.nn.Linear(NUM_HIDDEN_UNITS, NUM_ATTRIBUTES)),
            ("attributes_activation", torch.nn.Sigmoid()),
        ]
    )
)


def init_weights(layer):
    """Custom initializers for our model weights/biases"""
    if type(layer) == torch.nn.Linear:
        torch.nn.init.uniform_(layer.weight, 0, 0.9)  # (weight, mean, stdev)
        torch.nn.init.constant_(layer.bias, 0)


model.apply(init_weights)

# Define our loss and optimizer functions
loss_fn = torch.nn.MSELoss(reduction="sum")
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

# Custom callbacks for viewing the model in TensorBoard; which is launched with command:
# $ tensorboard --logdir logs/fit
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = tensorboard.SummaryWriter(log_dir)

# Main loop (look at how clean this looks compared to tf!)
for epoch in range(NUM_EPOCHS):
    y_pred = model(x)

    loss = loss_fn(y_pred, y)
    writer.add_scalar("Training loss", loss.item(), epoch)

    if epoch % LOG_INTERVAL == 0:
        print(f"Epoch: {epoch}, loss: {loss.item()}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
