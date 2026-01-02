import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import odl
from datetime import datetime
from Gen_Sino_Partitioned import randIm, ConstIm
from math import sqrt
import math
import cvxpy as cp
import networkx as nx

# Load saved forward operators
with open("saved_operators_Incmp_Span/ray_transforms.pkl", "rb") as f:
    ray_transforms = pickle.load(f)
with open("saved_operators_Incmp_Span/aggregate_op.pkl", "rb") as f:
    agg_ray_trafo = pickle.load(f)
with open("saved_operators_Incmp_Span/A_dense_list.pkl", "rb") as f:
    A_dense_list = pickle.load(f)
with open("saved_operators_Incmp_Span/A_agg.pkl", "rb") as f:
    A_agg = pickle.load(f)

# === Parameters ===
N = 64
num_nodes = 5
det_shape = N
noise_level = 0.005 # noise level for sinograms


# === Create output directory ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"recon_output_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# === Generate phantom
phantom_array = ConstIm(N)
# Create uniform discretization space
space = odl.uniform_discr([-1, -1], [1, 1], [N, N], dtype='float32')


# Diagnostics
mse_lists = [[] for _ in range(num_nodes)] # initialize image-space MSE array
mse_sino_lists = [[] for _ in range(num_nodes)] # initialize sinogram-space MSE array
mse_agg_list, mse_agg_sino_list = [], []
column_norms_all = [np.linalg.norm(A_i_dense, axis=0) for A_i_dense in A_dense_list]  # shape (N^2,)


phantom = space.element(phantom_array)

#sinograms = [op(phantom) + noise_level * op.range.element(np.random.normal(loc=0.0, scale=1.0, size=op.range.shape)) for op in ray_transforms]
#as a list
sinograms = [
    op(phantom).asarray() + noise_level * np.random.normal(0, 1, op(phantom).shape)
    for op in ray_transforms
]
# as a stacked array
#sinograms = np.stack([
#    op(phantom).asarray() + noise_level * np.random.normal(0, 1, op(phantom).shape)
#    for op in ray_transforms
#])





# ================================ Plots and Visuals =============================================================
# Save true phantom
# === Plot 1: True Phantom ===
fig1 = plt.figure(figsize=(4, 4))
plt.imshow(phantom, cmap='gray')
plt.title("True Phantom")
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "plot1_true_phantom.png"), dpi=300)
plt.show()
#plt.close(fig1)
fig3, axes = plt.subplots(1, num_nodes + 2, figsize=(4 * (num_nodes + 2), 4))
for i in range(num_nodes):
    axes[i + 1].imshow(sinograms[i], cmap='gray', aspect='auto', origin='upper')
    axes[i + 1].set_title(f"Node {i + 1} Sinogram")
    axes[i + 1].set_xlabel("Detector")
    axes[i + 1].set_ylabel("Angle")


# Plot aggregate sinogram


plt.tight_layout()
plt.savefig(os.path.join(output_dir, "plot3_clean_true_sinograms.png"), dpi=300)
plt.show()

# === Plot 2: Reconstructed Phantoms ===

# === Plot 2b: True minus reconstructed per node ===


# One shared colorbar


############################################### PLOTTING #################################################################
# Plotting the true phantom and sinograms

