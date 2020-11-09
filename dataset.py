from torch.utils import data
from get_data import get_data
import torch

BATCH_SIZE = 4

[items, attributes, df] = get_data()

NUM_ITEMS = len(items)
NUM_ATTRIBUTES = len(attributes)

features = torch.tensor(to_categorical(range(NUM_ITEMS)), dtype=torch.float32)
targets = torch.tensor(df.values, dtype=torch.float32)


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
