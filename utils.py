from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from mpl_toolkits import axes_grid1
from torchvision import transforms
from collections import Counter
from get_data import get_data
from torch.utils import data
import seaborn as sns
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib
import torch
import io

plt.style.use("plotstyle.mplstyle")

PLOT_DPI = 90


[items, attributes, df] = get_data()


def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """
    Add a vertical color bar to an image plot.
    S/o to https://nbviewer.jupyter.org/github/mgeier/python-audio/blob/master/plotting/matplotlib-colorbar.ipynb
    """
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1.0 / aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


def show(fig: matplotlib.figure.Figure) -> None:
    "Show an existing matplotlib figure"
    fig.show()
    plt.show()


def plot_covariance_matrix(df: pd.DataFrame) -> matplotlib.figure.Figure:
    """
    Plots the covariance matrix of the items (given their attributes), showing the hierarchy of correlated (positive
    values), uncorrelated (0) and negatively correlated (negative values) items
    """
    cov = np.cov(df)
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    fig, ax = plt.subplots(dpi=PLOT_DPI)
    cax = ax.imshow(cov, cmap=cmap, interpolation="nearest")
    plt.xticks(range(df.shape[0]), df.index, rotation=45)
    plt.yticks(range(df.shape[0]), df.index)
    add_colorbar(cax, ticks=np.unique(np.round(cov, decimals=2)))
    return fig


# show(plot_covariance_matrix(df.transpose()))


def plot_input_input_correlation(df: pd.DataFrame) -> matplotlib.figure.Figure:
    """
    Plot the input-input correlation matrix
    Note: how does this differ from covariance_matrix & item_attribute_matrix?
    """
    corr = df.corr()
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    fig, ax = plt.subplots(dpi=PLOT_DPI)
    cax = ax.imshow(corr, cmap=cmap, interpolation="nearest")
    plt.xticks(range(df.shape[1]), df.columns, rotation=45)
    plt.yticks(range(df.shape[1]), df.columns)
    add_colorbar(cax, ticks=np.unique(np.round(corr, decimals=2)))
    return fig


# show(input_input_correlation(df.transpose()))


def plot_item_attribute_matrix(df: pd.DataFrame) -> matplotlib.figure.Figure:
    "Plots the item-attribute matrix of a dataframe, with values 1=True as black and 0=False as white"
    fig = plt.figure(figsize=(7, 5), dpi=PLOT_DPI)
    ax = fig.add_subplot(111)
    cax = ax.imshow(df, cmap="binary")
    plt.xticks(range(df.shape[1]), df.columns, fontsize=10, rotation=90)
    plt.yticks(range(df.shape[0]), df.index, fontsize=10)
    add_colorbar(cax, ticks=[0, 1])
    return fig


# show(plot_item_attribute_matrix(df.transpose()))


def plot_svd(df: pd.DataFrame) -> matplotlib.figure.Figure:
    "Calculates and plots the SVD (singular value decomposition) of a dataframe"
    u, s, vT = np.linalg.svd(df, full_matrices=False)

    color_range = 1
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 5), dpi=PLOT_DPI)
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

    add_colorbar(im1)
    add_colorbar(im2)
    add_colorbar(im3, aspect=12)
    add_colorbar(im4, aspect=12)
    fig.suptitle("Singular value decomposition", fontsize=16)

    return fig


# show(plot_svd(df.transpose()))


def plot_singular_dimensions(df: pd.DataFrame) -> matplotlib.figure.Figure:
    "Calculate the SVD of a dataframe and plot the singular dimensions and their sums"
    color_range = 1
    u, s, vT = np.linalg.svd(df, full_matrices=False)

    sdcomp = np.zeros((len(df.index), len(df.index), len(df.columns)), dtype=np.float32,)

    for i in range(len(df.columns)):
        sdcomp[i, :, :] = s[i] * np.outer(u[:, i], vT[i, :])

    width = (len(df.columns) + 2) // 2
    fig, ax = plt.subplots(2, width, figsize=(5, 7), dpi=PLOT_DPI)

    for i in range(len(df.columns)):
        # hacky!
        index1 = 0 if i < (width - 1) else 1
        index2 = i % (width - 1)
        ax[index1, index2].imshow(
            sdcomp[i, :, :], cmap="bwr", vmin=-color_range, vmax=color_range
        )
        ax[index1, index2].set_xticks([])
        ax[index1, index2].set_yticks([])

    ax[0, width - 1].imshow(
        np.sum(sdcomp[0 : (width - 1), :, :], 0),
        cmap="bwr",
        vmin=-color_range,
        vmax=color_range,
    )
    ax[0, width - 1].set_xticks([])
    ax[0, width - 1].set_yticks([])
    ax[1, width - 1].imshow(
        np.sum(sdcomp[0 : 2 * (width - 1), :, :], 0),
        cmap="bwr",
        vmin=-color_range,
        vmax=color_range,
    )
    ax[1, width - 1].set_xticks([])
    ax[1, width - 1].set_yticks([])
    fig.suptitle("Singular dimensions", fontsize=16)
    return fig


# show(plot_singular_dimensions(df.transpose()))


def plot_dendrogram(df: pd.DataFrame) -> matplotlib.figure.Figure:
    fig = plt.figure(figsize=(8, 8), dpi=PLOT_DPI)
    Z = sp.cluster.hierarchy.linkage(df.transpose(), "ward")
    sp.cluster.hierarchy.dendrogram(
        Z, leaf_rotation=90, leaf_font_size=8, labels=df.transpose().index
    )
    fig.suptitle("Dendrogram", fontsize=16)
    return fig


# show(plot_dendrogram(df.transpose()))


def plot_PCA(df: pd.DataFrame) -> matplotlib.figure.Figure:
    pca = PCA(n_components=3)
    pca_df = pca.fit_transform(df)
    reduced = pd.DataFrame(data=pca_df, index=df.index, columns=["PC1", "PC2", "PC3"])

    D1 = 0
    D2 = 1

    x = reduced.iloc[:, D1]
    y = reduced.iloc[:, D2]
    n = reduced.index

    fig = plt.figure(figsize=(8, 8), dpi=PLOT_DPI)
    plt.title("PCA")
    plt.tight_layout()
    plt.scatter(x, y)

    for i, txt in enumerate(n):
        plt.annotate(txt, (x[i], y[i]))

    plt.xlabel(f"PC{D1 + 1}")
    plt.ylabel(f"PC{D2 + 1}")

    return fig


# show(plot_PCA(df.transpose()))


def plot_distance_matrix(df: pd.DataFrame) -> matplotlib.figure.Figure:
    df_distances = pd.DataFrame(
        sp.spatial.distance_matrix(df.values, df.values),
        index=df.index,
        columns=df.index,
    )
    fig = plt.figure(figsize=(8, 8))
    plt.title("Distance matrix")
    cmap = sns.cubehelix_palette(8, reverse=False, as_cmap=True)
    sns.heatmap(df, cmap=cmap, square=True)
    plt.xticks(rotation=45)
    return fig


# show(plot_distance_matrix(df.transpose()))

# Keras doesn't support python 3.8 yet so I stole this (all credit to them)
def to_categorical(y, num_classes=None, dtype="float32"):
    """
    Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    Usage Example:
    >>> y = [0, 1, 2, 3]
    >>> tf.keras.utils.to_categorical(y, num_classes=4)
    array([[1., 0., 0., 0.],
          [0., 1., 0., 0.],
          [0., 0., 1., 0.],
          [0., 0., 0., 1.]], dtype=float32)
    Arguments:
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input. Default: `'float32'`.
    Returns:
        A binary matrix representation of the input. The classes axis is placed
        last.
    """
    y = np.array(y, dtype="int")
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def P(x):
    return 1


def P_joint(x, y):
    return 2


def I(X, Y):
    return sum(
        [P_joint(x, y) * np.log(P_joint(x, y) / (P(x) * P(y))) for x, y in zip(X, Y)]
    )


# print(I([1,1], [2,2,]))


def get_pdfs(dataset: data.Dataset):
    pdf_x, pdf_t, pdf_xt = [Counter(), Counter(), Counter()]
    n_samples = dataset.__len__()

    for i in range(n_samples):
        (x, y) = dataset.__getitem__(i)
        print(x, y)

    # for i in range(n_train_samples):
    #     pdf_x[x_train_int[i]] += 1 / float(n_train_samples)
    #     pdf_y[y_train[i, 0]] += 1 / float(n_train_samples)
    #     pdf_xt[(x_train_int[i],) + tuple(indices[i, :])] += 1 / float(n_train_samples)
    #     pdf_yt[(y_train[i, 0],) + tuple(indices[i, :])] += 1 / float(n_train_samples)
    #     pdf_t[tuple(indices[i, :])] += 1 / float(n_train_samples)


def I(a, b):
    pdf_x, pdf_t, pdf_xt = [Counter(), Counter(), Counter()]

    mutual_information = 0


# I(5, 5)


def calculate_mututal_information():
    "Calculates the mutual information"
    # discretization
    n_bins = 30
    bins = np.linspace(-1, 1, n_bins + 1)

