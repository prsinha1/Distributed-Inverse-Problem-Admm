# block_7_main_test.py
# Minimal script to save the discrete-gradient edge map of the TRUE phantom only.

import os
import numpy as np
import odl

from block_4_tv_helpers_with_plot import save_edge_map  # uses edge_map_from_vector under the hood
from Gen_Sino_Partitioned import ConstIm
from Gen_Sino_Partitioned import randIm

# ---- Config ----
N = 64
edge_dir = os.path.join("diagnostics", "edge_maps")
os.makedirs(edge_dir, exist_ok=True)

# ---- Build the true phantom exactly as in Block 2 style ----
#phantom_array = ConstIm(N) 
phantom_array = randIm(N)                              # returns an N x N ndarray
space = odl.uniform_discr([-1, -1], [1, 1], [N, N], dtype="float32")
phantom = space.element(phantom_array)                  # ODL element
x_true = phantom.asarray().ravel()                      # vectorize to length N*N

# ---- Save edge map for the true phantom ----
edge_path_true = os.path.join(edge_dir, "true_phantom_edge_map.png")
save_edge_map(x_true, N, edge_path_true, show=False)
print(f"[Diagnostics] Saved true phantom edge map to {edge_path_true}")

print("[Done] Edge map creation for true phantom completed.")
