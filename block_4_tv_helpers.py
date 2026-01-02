# block_4_tv_helpers.py
import numpy as np
import cvxpy as cp

def isotropic_tv_on_vector(x_vec, N):
    """
    x_vec is length N*N, reshape to N x N image,
    isotropic TV: sum over pixels of sqrt( (Dx)^2 + (Dy)^2 )
    """
    X = cp.reshape(x_vec, (N, N))
    Dx = X[1:, :] - X[:-1, :]
    Dy = X[:, 1:] - X[:, :-1]
    tv = cp.sum(cp.norm(cp.hstack([cp.reshape(Dx, (-1, 1)), cp.reshape(Dy, (-1, 1))]), 2, axis=1))
    return tv

# ===== TV subgradient helpers for inexact ADMM =====
def _grad_forward_2d_from_vec(x_vec, N):
    X = x_vec.reshape(N, N)
    gx = np.zeros_like(X)
    gy = np.zeros_like(X)
    gx[:-1, :] = X[1:, :] - X[:-1, :]
    gy[:, :-1] = X[:, 1:] - X[:, :-1]
    return gx, gy

def _div_backward_2d_to_vec(px, py, N):
    div = np.zeros((N, N), dtype=float)
    # adjoint of forward differences (negative divergence)
    div[0, :] -= px[0, :]
    div[1:-1, :] += px[1:-1, :] - px[:-2, :]
    div[-1, :] += px[-2, :]

    div[:, 0] -= py[:, 0]
    div[:, 1:-1] += py[:, 1:-1] - py[:, :-2]
    div[:, -1] += py[:, -2]
    return (-div).reshape(N * N)

def kt_subgrad_isotropic_tv_from_x(x_vec, N, eps=1e-12):
    gx, gy = _grad_forward_2d_from_vec(x_vec, N)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    mask = mag > eps
    px = np.zeros_like(gx)
    py = np.zeros_like(gy)
    px[mask] = gx[mask] / mag[mask]
    py[mask] = gy[mask] / mag[mask]
    kt_s = _div_backward_2d_to_vec(px, py, N)
    return kt_s
