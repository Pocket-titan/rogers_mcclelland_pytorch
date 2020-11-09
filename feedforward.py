from torch.utils import tensorboard, data
from matplotlib import pyplot as plt
from collections import OrderedDict
from get_data import get_data
from functools import reduce
from utils import (
    plot_item_attribute_matrix,
    plot_singular_dimensions,
    plot_covariance_matrix,
    plot_distance_matrix,
    plot_dendrogram,
    plot_PCA,
    plot_svd,
    get_pdfs,
    to_categorical,
    show,
)
import pandas as pd
import numpy as np
import scipy as sp
import datetime
import torch

LEARNING_RATE = 1e-2
NUM_HIDDEN_UNITS = 8
LOG_INTERVAL = 500  # epochs
NUM_EPOCHS = 3500
BATCH_SIZE = 4

[items, attributes, df] = get_data()

# show(plot_distance_matrix(df.transpose()))
# show(plot_svd(df.transpose()))
# show(plot_singular_dimensions(df.transpose()))
# show(plot_covariance_matrix(df.transpose()))

NUM_ITEMS = len(items)
NUM_ATTRIBUTES = len(attributes)

# Create feature and target tensors
features = torch.tensor(to_categorical(range(NUM_ITEMS)), dtype=torch.float32)
targets = torch.tensor(df.values, dtype=torch.float32)

NUM_RUNS = 5


class Dataset(data.Dataset):
    "Construct a PyTorch Dataset & Dataloader to automate batching & shuffling"

    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

        if len(targets) != len(features):
            raise ValueError("Length of features and targets vectors are different")

        self.n_samples = len(targets)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return (self.features[index], self.targets[index])


dataset = Dataset(features, targets)
dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Prepare the model
# Note: we have twice the layers compared to the Keras model, since keras layers (like
# keras.layers.Dense) do _both_ the Linear & activation parts; PyTorch doesn't.
model = torch.nn.Sequential(
    OrderedDict(
        [
            ("representations", torch.nn.Linear(NUM_ITEMS, NUM_HIDDEN_UNITS)),
            ("representations_activation", torch.nn.Sigmoid()),
            ("attributes", torch.nn.Linear(NUM_HIDDEN_UNITS, NUM_ATTRIBUTES)),
            ("attributes_activation", torch.nn.Sigmoid()),
        ]
    )
)


def init_weights(layer):
    """
    Custom initializers for our model weights/biases
    Note: our Sigmoid layers don't have weights/biases, so they're not initialized here!
    """
    if type(layer) == torch.nn.Linear:
        torch.nn.init.uniform_(layer.weight, 0, 0.9)  # (weight, mean, stdev)
        torch.nn.init.constant_(layer.bias, 0)


model.apply(init_weights)

# Define our loss and optimizer functions
loss_fn = torch.nn.MSELoss(reduction="sum")
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

# Custom callbacks for viewing the model in TensorBoard; which is launched with command:
# $ tensorboard --logdir logs/fit
info = {"avg_of_runs": NUM_RUNS, "hidden_units": NUM_HIDDEN_UNITS}

name = "feedforward"

log_dir = (
    f"logs/fit/{name}"
    + ":"
    + ",".join([f"{key}={value}" for (key, value) in info.items()])
)
# log_dir = "logs/fit/feedforward" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = tensorboard.SummaryWriter(log_dir)
writer.add_graph(model, features)

metrics = [
    {"name": "SVD/Singular value decomposition", "f": plot_svd,},
    {"name": "SVD/Singular dimensions", "f": plot_singular_dimensions},
    # Actually, this one below is equal to distance_matrix
    # {"name": "Metrics/Item-attribute matrix", "f": plot_item_attribute_matrix},
    {"name": "Metrics/Covariance matrix", "f": plot_covariance_matrix,},
    {"name": "Metrics/Dendrogram", "f": plot_dendrogram},
    {"name": "Metrics/PCA", "f": plot_PCA},
    {"name": "Metrics/Distance matrix", "f": plot_distance_matrix},
]

# # Main loop (look at how clean this looks compared to tf!)
# for epoch in range(NUM_EPOCHS):
#     # Enumerate over our batches of BATCH_SIZE
#     for i, (x, y) in enumerate(dataloader):
#         y_pred = model(x)
#         loss = loss_fn(y_pred, y)
#         writer.add_scalar("Training loss", loss.item(), epoch)

#     if epoch % LOG_INTERVAL == 0:
# # Predict y for the whole features vector, not just a single batch
# y_pred_all = model(features)
# df_pred = pd.DataFrame(
#     data=y_pred_all.detach().numpy(), index=df.index, columns=df.columns,
# )

# # Generate our metrics and save them to Tensorboard
# for metric in metrics:
#     figure = metric["f"](df_pred.transpose())
#     # show(figure)
#     writer.add_figure(metric["name"], figure, global_step=epoch)

#         print(f"Epoch: {epoch}, loss: {loss.item()}")

#     # In PyTorch, we have to manually zero our gradients & optimize
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

total_loss = []

for run in range(NUM_RUNS):
    print(f"Model run: {run + 1}/{NUM_RUNS}")
    run_loss = []
    for epoch in range(NUM_EPOCHS):
        loss = 0
        for i, (x, y) in enumerate(dataloader):
            y_pred = model(x)
            loss += loss_fn(y_pred, y)

        # writer.add_scalar("Training loss", loss.item(), epoch)
        run_loss.append(loss.item())

        if epoch % LOG_INTERVAL == 0:
            print(f"Epoch: {epoch}, loss: {loss.item()}")

        if run == NUM_RUNS - 1 and epoch % LOG_INTERVAL == 0:
            # Predict y for the whole features vector, not just a single batch
            y_pred_all = model(features)
            df_pred = pd.DataFrame(
                data=y_pred_all.detach().numpy(), index=df.index, columns=df.columns,
            )

            # Generate our metrics and save them to Tensorboard
            for metric in metrics:
                figure = metric["f"](df_pred.transpose())
                # show(figure)
                writer.add_figure(metric["name"], figure, global_step=epoch)

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
