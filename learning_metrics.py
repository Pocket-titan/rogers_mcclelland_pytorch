from tensorflow import keras
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use("plotstyle.mplstyle")

# Read and format the data
data = pd.read_csv("data/Rumelhart_livingthings.csv", sep=",")

items = sorted(data.Item.unique())
relations = sorted(data.Relation.unique())
attributes = sorted(data.Attribute.unique())

num_items = len(items)
num_relations = len(relations)
num_attributes = len(attributes)

data_frame = pd.pivot_table(
    data, values="TRUE", index=["Item"], columns=["Attribute"], fill_value=0
).astype(float)

# Sort our axes by related things, so our data hierarchy looks nicer when plotted!
data_frame = data_frame.reindex(
    ["Robin", "Canary", "Sunfish", "Salmon", "Daisy", "Rose", "Oak", "Pine"], axis="index"
)
data_frame = data_frame.reindex(
    [
        "Grow",
        "Living",
        "LivingThing",
        "Animal",
        "Move",
        "Skin",
        "Bird",
        "Feathers",
        "Fly",
        "Wings",
        "Fish",
        "Gills",
        "Scales",
        "Swim",
        "Yellow",
        "Red",
        "Sing",
        "Robin",
        "Canary",
        "Sunfish",
        "Salmon",
        "Daisy",
        "Rose",
        "Oak",
        "Pine",
        "Green",
        "Bark",
        "Big",
        "Tree",
        "Branches",
        "Pretty",
        "Petals",
        "Flower",
        "Leaves",
        "Roots",
        "Plant",
    ],
    axis="columns",
)


def plot_covariance_matrix(df: pd.DataFrame) -> None:
    """Plots the covariance matrix of the items (given their attributes), showing the hierarchy of correlated (positive
    values), uncorrelated (0) and negatively correlated (negative values) items"""
    cov = np.cov(df)
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    fig, ax = plt.subplots()
    cax = ax.imshow(cov, cmap=cmap, interpolation="nearest")
    plt.xticks(range(df.shape[0]), df.index, rotation=45)
    plt.yticks(range(df.shape[0]), df.index)
    fig.colorbar(cax, ticks=np.unique(np.round(cov, decimals=2)))
    plt.show()


# plot_covariance_matrix(data_frame)


def plot_item_attribute_matrix(df: pd.DataFrame) -> None:
    """Plots the item-attribute matrix, with values 1=True as black and 0=False as white"""
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    cax = ax.imshow(df, cmap="binary")
    plt.xticks(range(df.shape[1]), df.columns, fontsize=10, rotation=90)
    plt.yticks(range(df.shape[0]), df.index, fontsize=10)
    fig.colorbar(cax, ticks=[0, 1])
    plt.show()


# plot_item_attribute_matrix(data_frame)


def plot_SVD(df: pd.DataFrame) -> None:
    """Plots the SVD (singular value decomposition) of the dataframe"""
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(14, 7), dpi=90)
    u, s, vT = np.linalg.svd(df, full_matrices=False)
    color_range = 1
    im1 = ax1.imshow(df, cmap="bwr", vmin=-np.max(df.values), vmax=np.max(df.values))
    im2 = ax2.imshow(u, cmap="bwr")
    im3 = ax3.imshow(np.diag(s), cmap="bwr", vmin=-np.max(s), vmax=np.max(np.diag(s)))
    im4 = ax4.imshow(vT, cmap="bwr", vmin=-color_range, vmax=color_range)

    first_letters = list(map(lambda x: x[0], df.index))
    modes = list(map(lambda x: x + 1, range(len(df.columns))))

    ax1.set_yticks(range(len(df.index)))
    ax1.set_yticklabels(df.index)
    ax1.set_xticks(range(len(df.columns)))
    ax1.set_xticklabels(df.columns, rotation=45, fontdict={"fontsize": 8})
    ax1.set_xlabel("Items")
    ax1.set_ylabel("Attributes")
    ax1.set_title("$\Sigma^{yx}$")

    ax2.set_yticks(range(len(first_letters)))
    ax2.set_yticklabels(first_letters)
    ax2.set_xticks(range(len(modes)))
    ax2.set_xticklabels(modes)
    ax2.set_xlabel("Modes")
    ax2.set_ylabel("Attributes")
    ax2.set_title("$U$")

    ax3.set_xticks(range(len(modes)))
    ax3.set_xticklabels(modes)
    ax3.set_yticks(range(len(modes)))
    ax3.set_yticklabels(modes)
    ax3.set_xlabel("Modes")
    ax3.set_ylabel("Modes")
    ax3.set_title("$S$")

    ax4.set_xticks(range(len(df.columns)))
    ax4.set_xticklabels(df.columns, rotation=45)
    ax4.set_yticks(range(len(modes)))
    ax4.set_yticklabels(modes)
    ax4.set_xlabel("Items")
    ax4.set_ylabel("Modes")
    ax4.set_title("$V^{T}$")

    fig.colorbar(im1, ax=ax1)
    fig.colorbar(im2, ax=ax2)
    fig.colorbar(im3, ax=ax3)
    fig.colorbar(im4, ax=ax4)
    plt.show()

    sdcomp = np.zeros((len(df.index), len(df.index), len(df.columns)), dtype=np.float32,)

    for i in range(len(df.columns)):
        sdcomp[i, :, :] = s[i] * np.outer(u[:, i], vT[i, :])

    width = (len(df.columns) + 2) // 2
    fig, ax = plt.subplots(2, width, figsize=(5, 7))

    for i in range(len(df.columns)):
        # hacky!
        index1 = 0 if i < 4 else 1
        index2 = i % 4
        ax[index1, index2].imshow(
            sdcomp[i, :, :], cmap="bwr", vmin=-color_range, vmax=color_range
        )
        ax[index1, index2].set_xticks([])
        ax[index1, index2].set_yticks([])

    ax[0, width - 1].imshow(
        np.sum(sdcomp[0:4, :, :], 0), cmap="bwr", vmin=-color_range, vmax=color_range
    )
    ax[0, width - 1].set_xticks([])
    ax[0, width - 1].set_yticks([])
    ax[1, width - 1].imshow(
        np.sum(sdcomp[0:8, :, :], 0), cmap="bwr", vmin=-color_range, vmax=color_range
    )
    ax[1, width - 1].set_xticks([])
    ax[1, width - 1].set_yticks([])
    plt.show()


# plot_SVD(data_frame.transpose())
