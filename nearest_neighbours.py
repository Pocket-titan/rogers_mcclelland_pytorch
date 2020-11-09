from utils import to_categorical, plot_svd
from torch.utils import tensorboard
from collections import OrderedDict
from get_data import get_data
import numpy as np
import datetime
import torch
import math

LEARNING_RATE = 1e-2
NUM_HIDDEN_UNITS = 8
LOG_INTERVAL = 500  # epochs
NUM_EPOCHS = 3500
BATCH_SIZE = 64

[items, attributes, df] = get_data()

NUM_ITEMS = len(items)
NUM_ATTRIBUTES = len(attributes)

x = torch.tensor(to_categorical(range(NUM_ITEMS)), dtype=torch.float32)  # features
y = torch.tensor(df.values, dtype=torch.float32)  # targets

NUM_RECURRENT_GENERATIONS = 5
NUM_RUNS = 5


class Lateral(torch.nn.Module):
    def __init__(self, hidden_dimension):
        super(Lateral, self).__init__()
        self.hidden_dimension = hidden_dimension
        self.init_weights()

    def init_weights(self):
        self.weights = torch.nn.Parameter(
            torch.zeros([self.hidden_dimension - 1, 1]), requires_grad=True
        )
        self.bias = torch.nn.Parameter(
            torch.zeros([self.hidden_dimension - 1, 1]), requires_grad=True
        )
        # # Default pytorch weights/bias init
        # stdv = 1.0 / math.sqrt(self.weights.size(1))
        # self.weights.data.uniform_(-stdv, stdv)
        # self.bias.data.uniform_(-stdv, stdv)
        torch.nn.init.uniform_(self.weights, 0, 0.9)
        torch.nn.init.constant_(self.bias, 0)

    def forward(self, x):
        y_pred = torch.autograd.Variable(x.clone(), requires_grad=False)
        for _ in range(0, NUM_RECURRENT_GENERATIONS):
            for i, neuron in enumerate(y_pred):
                if i > 0:
                    y_pred[i] = x[i] + x[i - 1] * self.weights[i - 1] + self.bias[i - 1]
        return y_pred

    def reset_parameters(self):
        self.init_weights()


model = torch.nn.Sequential(
    OrderedDict(
        [
            ("representations", torch.nn.Linear(NUM_ITEMS, NUM_HIDDEN_UNITS)),
            ("representations_activation", torch.nn.Sigmoid()),
            ("lateral", Lateral(NUM_HIDDEN_UNITS)),
            ("attributes", torch.nn.Linear(NUM_HIDDEN_UNITS, NUM_ATTRIBUTES)),
            ("attributes_activation", torch.nn.Sigmoid()),
        ]
    )
)

loss_fn = torch.nn.MSELoss(reduction="sum")
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

info = {
    "rec_gen": NUM_RECURRENT_GENERATIONS,
    "avg_of_runs": NUM_RUNS,
    "hidden_units": NUM_HIDDEN_UNITS,
}

name = "nearest_neighbours"

log_dir = (
    f"logs/fit/{name}"
    + ":"
    + ",".join([f"{key}={value}" for (key, value) in info.items()])
)

# log_dir = (
#     "logs/fit/nearest_neighbours"
#     + f"_{NUM_RECURRENT_GENERATIONS}"
#     + f"_{NUM_RUNS}_"
#     # + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# )

writer = tensorboard.SummaryWriter(log_dir)

total_loss = []

for run in range(NUM_RUNS):
    print(f"Model run: {run + 1}/{NUM_RUNS}")
    run_loss = []
    for epoch in range(NUM_EPOCHS):
        loss = 0
        for i, sample in enumerate(x):
            y_pred = model(sample)
            loss += loss_fn(y_pred, y[i])

        # writer.add_scalar("Training loss", loss.item(), epoch)
        run_loss.append(loss.item())

        if epoch % LOG_INTERVAL == 0:
            print(f"Epoch: {epoch}, loss: {loss.item()}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    total_loss.append(run_loss)
    # Hackyyyy
    for layer in model.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()

total_loss = np.transpose(total_loss)
total_loss = [np.mean(x) for x in total_loss]
for epoch, loss in enumerate(total_loss):
    writer.add_scalar("Training loss", loss, epoch)
