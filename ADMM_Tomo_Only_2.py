import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import odl
from datetime import datetime
from Gen_Sino_Partitioned import randIm, ConstIm
from math import sqrt

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

niter = 100
#lambda_penalty = 0.005#0.005 # weightage of TV penalty for per node reconst
alpha_tv = 0#0.0005 # decay factor for lambda_tv; lambda_tv = lambda_penalty * np.exp(alpha_tv* k)
#lambda_agg = 0.009 # weightage of TV penalty for Agg reconst

lambda_penalty = 0.005 # weightage of TV penalty for per node reconst
lambda_agg = 0.005 # weightage of TV penalty for Agg reconst
gamma = 2 # try 2 #0.01 # Quadratic consensus penalty


# === Create output directory ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"recon_output_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# === Generate phantom
phantom_array = ConstIm(N)
# Create uniform discretization space
space = odl.uniform_discr([-1, -1], [1, 1], [N, N], dtype='float32')
x_vars = [space.zero() for _ in range(num_nodes)] 
y_vars = []

# Diagnostics
mse_lists = [[] for _ in range(num_nodes)] # initialize image-space MSE array
mse_sino_lists = [[] for _ in range(num_nodes)] # initialize sinogram-space MSE array
mse_agg_list, mse_agg_sino_list = [], []
#column_norms_all = []
column_norms_all = [np.linalg.norm(A_i_dense, axis=0) for A_i_dense in A_dense_list]  # shape (N^2,)
#column_norms_all.append(column_norms)

phantom = space.element(phantom_array)
# Generate sinograms
#sinograms = [rt(phantom) for rt in ray_transforms]
sinograms = [op(phantom) + noise_level * op.range.element(np.random.normal(loc=0.0, scale=1.0, size=op.range.shape)) for op in ray_transforms]




 
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


############################################### PLOTTING #################################################################
# Plotting the true phantom and sinograms

x_true_vec = phantom.asarray().reshape(-1, 1)
agg_sinogram_vec = A_agg @ x_true_vec

total_height = sum(rt.range.shape[0] for rt in ray_transforms)
agg_sinogram = agg_sinogram_vec.reshape((total_height, det_shape))
fig3, axes = plt.subplots(1, num_nodes + 2, figsize=(4 * (num_nodes + 2), 4))

# Plot true phantom
axes[0].imshow(phantom.asarray(), cmap='gray')
axes[0].set_title("True Phantom")
axes[0].axis('off')
#plt.tight_layout()
plt.savefig(os.path.join(output_dir, "true_phantom.png"), dpi=300)
#plt.show()

# Plot individual sinograms
for i in range(num_nodes):
    axes[i + 1].imshow(sinograms[i].asarray(), cmap='gray', aspect='auto', origin='upper')
    axes[i + 1].set_title(f"Node {i + 1} Sinogram")
    axes[i + 1].set_xlabel("Detector")
    axes[i + 1].set_ylabel("Angle")

# Plot aggregate sinogram
axes[-1].imshow(agg_sinogram, cmap='gray', aspect='auto', origin='upper')
axes[-1].set_title("Aggregate A*x_true")
axes[-1].set_xlabel("Detector")
axes[-1].set_ylabel("Angle")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "plot3_clean_true_sinograms.png"), dpi=300)
plt.show()

