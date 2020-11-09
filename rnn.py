from utils import to_categorical, plot_svd
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

print(items)

NUM_ITEMS = len(items)
NUM_ATTRIBUTES = len(attributes)

# Create feature and target tensors
x = torch.tensor(to_categorical(range(NUM_ITEMS)), dtype=torch.float32)  # features
y = torch.tensor(df.values, dtype=torch.float32)  # targets

# Prepare the model
class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_dim, output_size, num_layers):
        super(RNN, self).__init__()
        self.rnn = torch.nn.RNN(
            input_size,
            hidden_dim,
            num_layers,
            batch_first=True,
            nonlinearity="relu",
        )
        self.fc = torch.nn.Linear()

    def forward():
        pass

    def init_hidden():
        pass


model = torch.nn.Sequential(
    OrderedDict(
        [
            ("representations", RNN(NUM_ITEMS, NUM_HIDDEN_UNITS, NUM_HIDDEN_UNITS, 1)),
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

loss_fn = torch.nn.MSELoss(reduction="sum")
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

log_dir = "logs/rnn/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = tensorboard.SummaryWriter(log_dir)

for epoch in range(NUM_EPOCHS):
    y_pred = model(x)

    loss = loss_fn(y_pred, y)
    writer.add_scalar("Training loss", loss.item(), epoch)

    if epoch % LOG_INTERVAL == 0:
        print(f"Epoch: {epoch}, loss: {loss.item()}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
