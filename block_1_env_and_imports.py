# block_1_env_and_imports.py
import os
import math
import pickle
import numpy as np
import cvxpy as cp
import networkx as nx

# Small helpers
def vec(img_2d):
    return img_2d.reshape(-1)

def unvec(x_vec, N):
    return x_vec.reshape(N, N)

def diag_from_column_norms(A_dense):
    # eta_j = ||A(:, j)||_2^2
    return np.sum(A_dense * A_dense, axis=0)  # shape (n,)
