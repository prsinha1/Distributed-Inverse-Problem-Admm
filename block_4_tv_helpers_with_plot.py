# block_4_tv_helpers.py
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

def isotropic_tv_on_vector(x_vec, N):
    """
    x_vec is length N*N, reshape to N x N image,
    isotropic TV: sum over pixels of sqrt( (Dx)^2 + (Dy)^2 )
    """
    X = cp.reshape(x_vec, (N, N))
    Dx = X[1:, :] - X[:-1, :]
    Dy = X[:, 1:] - X[:, :-1]
    tv = cp.sum(
        cp.norm(
            cp.hstack([cp.reshape(Dx, (-1, 1)), cp.reshape(Dy, (-1, 1))]),
            2,
            axis=1,
        )
    )
    return tv

def edge_map_from_vector(x_vec, N, normalize=True):
    """
    Compute the per pixel edge magnitude from forward differences,
    using NumPy only for visualization and diagnostics.
    Returns an array of shape (N, N).
    """
    X = np.asarray(x_vec, dtype=float).reshape(N, N)

    Dx = np.zeros_like(X)
    Dy = np.zeros_like(X)

    # forward differences inside the image
    Dx[:-1, :] = X[1:, :] - X[:-1, :]
    Dy[:, :-1] = X[:, 1:] - X[:, :-1]


    # isotropic magnitude
    mag = np.sqrt(Dx * Dx + Dy * Dy)

    if normalize:
        m = mag.max()
        if m > 0:
            mag = mag / m
    return mag

def save_edge_map(x_vec, N, out_path, show=False, cmap="gray", dpi=300):
    """
    Convenience helper to save the edge map image to disk.
    """
    mag = edge_map_from_vector(x_vec, N, normalize=True)
    plt.figure(figsize=(5, 5))
    plt.imshow(mag, cmap=cmap)
    plt.title("Discrete gradient magnitude")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close()
