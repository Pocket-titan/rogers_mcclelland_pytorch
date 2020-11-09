from utils import to_categorical, plot_svd
from torch.nn import functional as F
from torch.utils import tensorboard
from collections import OrderedDict
from get_data import get_data
import numpy as np
import datetime
import torch
import math

NUM_EPOCHS = 3500
LOG_INTERVAL = 500
LEARNING_RATE = 1e-3

[items, attributes, df] = get_data()

NUM_ITEMS = len(items)
NUM_ATTRIBUTES = len(attributes)

x = torch.tensor(to_categorical(range(NUM_ITEMS)), dtype=torch.float32)  # features
y = torch.tensor(df.values, dtype=torch.float32)  # targets

INPUT_DIM = NUM_ATTRIBUTES
HIDDEN_DIM = 2000
OUTPUT_DIM = NUM_ITEMS


class VariationalAutoEncoder(torch.nn.Module):
    def __init__(self):
        super(VariationalAutoEncoder, self).__init__()
        self.fc1 = torch.nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.fc21 = torch.nn.Linear(HIDDEN_DIM, OUTPUT_DIM)
        self.fc22 = torch.nn.Linear(HIDDEN_DIM, OUTPUT_DIM)
        self.fc3 = torch.nn.Linear(OUTPUT_DIM, HIDDEN_DIM)
        self.fc4 = torch.nn.Linear(HIDDEN_DIM, INPUT_DIM)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, INPUT_DIM))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VariationalAutoEncoder()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, INPUT_DIM), reduction="sum")

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, OUTPUT_DIM14
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def calculate_difference(x, y):
    return torch.sum(torch.sub(x, y))


log_dir = "logs/fit/vae" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = tensorboard.SummaryWriter(log_dir)


def train(epoch):
    model.train()
    loss = 0
    difference = 0

    for i, sample in enumerate(y):
        y_pred, mu, logvar = model(sample)
        loss += loss_function(y_pred, sample, mu, logvar)
        with torch.no_grad():
            difference += calculate_difference(sample, y_pred)

    writer.add_scalar("Training loss", loss.item(), epoch)
    writer.add_scalar("Difference", difference, epoch)

    if epoch % LOG_INTERVAL == 0:
        print(f"Epoch: {epoch}, loss: {loss.item()}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def test(epoch):
    model.eval()
    loss = 0
    with torch.no_grad():
        for i, sample in enumerate(y):
            y_pred, mu, logvar = model(sample)
            loss += loss_function(y_pred, sample, mu, logvar)
            print("Diff:", calculate_difference(sample, y_pred))


if __name__ == "__main__":
    for epoch in range(NUM_EPOCHS):
        train(epoch)

        # if epoch % LOG_INTERVAL == 0:
        #     test(epoch)
    with torch.no_grad():
        sample = torch.autograd.Variable(torch.randn(OUTPUT_DIM), OUTPUT_DIM)
        result = model.decode(sample)
        print(result)
