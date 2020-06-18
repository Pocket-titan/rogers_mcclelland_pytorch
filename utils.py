import matplotlib.pyplot as plt
import numpy as np

def plot_svd(u, s, vT) -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, dpi=90)

    s_is_diagonal_matrix = True if type(s[0]) is np.ndarray else False

    im1 = ax1.imshow(u, cmap="bwr", vmin=-1, vmax=1)
    im2 = ax2.imshow(
        s if s_is_diagonal_matrix else np.diag(s),
        cmap="bwr",
        vmin=-np.max(s),
        vmax=np.max(s),
    )
    im3 = ax3.imshow(vT, cmap="bwr", vmin=-1, vmax=1)
    fig.colorbar(im1, ax=ax1)
    fig.colorbar(im2, ax=ax2)
    fig.colorbar(im3, ax=ax3)
    plt.tight_layout()
    plt.show()

def do_svd():
  return
