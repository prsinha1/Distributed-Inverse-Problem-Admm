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


# Define TV and f penalty
tv = odl.Gradient(space)
tv_ops = [tv for _ in range(num_nodes)]  # redundant list of the same gradient, useful later for gradient along rays
g_funcs = [
    odl.solvers.SeparableSum(
        odl.solvers.L2NormSquared(ray_trafo.range).translated(sinogram),
        odl.solvers.GroupL1Norm(tv.range)
    )
    for ray_trafo, sinogram in zip(ray_transforms, sinograms)
]


x_agg = space.zero()
tv_agg = odl.Gradient(space)  
op_agg = odl.BroadcastOperator(*ray_transforms, tv_agg)
y_agg = op_agg.range.zero()  # Warm-start dual variable for global PDHG
f_agg = odl.solvers.ZeroFunctional(space) # needs to be defined for PDHG logic


g_agg = odl.solvers.SeparableSum(
    *[
        odl.solvers.L2NormSquared(rt.range).translated(sg)
        for rt, sg in zip(ray_transforms, sinograms)
    ],
    lambda_agg*odl.solvers.GroupL1Norm(tv_agg.range)  # Matches the last component of op_agg.range
)

for k in range(niter):
    # ===== Compute per node per iteration sinogram error ==============================
    # ============ error_i^k = ||A_i x_vars[i]^k - b_i||_2 =============================
    mse_vals = [np.linalg.norm(ray_transforms[i](x_vars[i]).asarray() - sinograms[i].asarray())
                for i in range(num_nodes)]
    # Dynamic lambda_tv decay 
    lambda_tv = lambda_penalty * np.exp(alpha_tv* k)
    #lambda_tv = lambda_penalty * (1 + np.log1p(k))
    print(f"Iteration {k+1:03d}, lambda_tv = {lambda_tv:.6f}")


    eta_pj_all = []
    for i in range(num_nodes):
        #eta_pj = column_norms_all[i] / (mse_vals[i] + 1e-8)
        recon_vector = x_vars[i].asarray().ravel()
        phantom_vector = phantom.asarray().ravel()
        abs_error = np.abs(recon_vector - phantom_vector) + 1e-8
        eta_pj = column_norms_all[i] / abs_error


        eta_pj_all.append(eta_pj)

    eta_pj_matrix = np.vstack(eta_pj_all)  # shape: (num_nodes, N^2)
    eta_sum_per_pixel = np.sum(eta_pj_matrix, axis=0)
    eta_norml_matrix = eta_pj_matrix / (eta_sum_per_pixel + 1e-8)

    # =================== Dynamic Convex Combination of Local Solutions xa_k ==========================
    x_vars_matrix = np.vstack([x_vars[i].asarray().ravel() for i in range(num_nodes)])  # shape (num_nodes, N^2)
    x_a_k_vector = np.sum(eta_norml_matrix * x_vars_matrix, axis=0)  # weighted sum across nodes
    x_a_k_image = space.element(x_a_k_vector.reshape((N, N)))

    # =================== ADMM Quadratic Consensus Term, f_penalty ==========================
    for i in range(num_nodes):
        # f_penalty(x) = (l2_norm(x-xa_k))^2
        f_penalty = gamma * odl.solvers.L2NormSquared(space).translated(x_a_k_image)
        op = odl.BroadcastOperator(ray_transforms[i], tv_ops[i])
        y_i = op.range.zero()

        # Estimate operator norm for choosing step sizes in PDHG
        op_norm = odl.power_method_opnorm(op) # Lin operator K_i = op

        # Solve a single iteration of the local optimization at node-i using PDHG     
        # minimize f_penalty(x) + g_i(K_i x), K_i x = A_i x + ∇x   
        odl.solvers.pdhg(x_vars[i], f_penalty, lambda_tv*g_funcs[i], op, niter=5,
                         tau=1.0 / op_norm, sigma=1.0 / op_norm, theta=1.0, y = y_i)
        # Compute Img/Phantom Sp. MSE at node-i per iteration
        #mse = np.sum((x_vars[i].asarray() - phantom) ** 2)
        mse = np.mean((x_vars[i].asarray() - phantom) ** 2)
        mse_lists[i].append(mse)
        # Compute Sinogram Sp. MSE at node-i per iteration
        mse_sino = np.linalg.norm(ray_transforms[i](x_vars[i]).asarray() - sinograms[i].asarray())
        mse_sino_lists[i].append(mse_sino)

    # ================================ Iterative PDHG Loop for Global Problem ========================================

    # Estimate operator norm for choosing step sizes in global PDHG
    op_norm = odl.power_method_opnorm(op_agg) # K_agg = op_agg

    # Run PDHG solver to minimize f_agg(x) + g_agg(K_agg x), K_agg x = A_agg x + ∇x
    odl.solvers.pdhg(x_agg, f_agg, g_agg, op_agg, niter=15, tau=1.0 / op_norm, sigma=1.0 / op_norm, theta=1.0, y=y_agg)

    # Compute Img/Phantom Sp. MSE per iteration from the Global problem

    #mse_agg = np.sum((x_agg.asarray() - phantom) ** 2)
    mse_agg = np.mean((x_agg.asarray() - phantom) ** 2)
    mse_agg_list.append(mse_agg)

    # Compute Sinogram Sp. MSE per iteration from the Global problem

    agg_recon = agg_ray_trafo(x_agg)
    diff = [agg_recon[i].asarray() - sinograms[i].asarray() for i in range(num_nodes)]
    mse_agg_sino = np.linalg.norm(np.concatenate([d.ravel() for d in diff]))
    mse_agg_sino_list.append(mse_agg_sino)

    if k % 20 == 0 or k == niter - 1:
        print(f"Iteration {k+1:03d}")
        print("  Image MSEs       =", ", ".join(f"{v[-1]:.4f}" for v in mse_lists))
        print("  Sinogram MSEs    =", ", ".join(f"{v[-1]:.4f}" for v in mse_sino_lists))
        print(f"  Agg Image MSE    = {mse_agg:.4f}")
        print(f"  Agg Sinogram MSE = {mse_agg_sino:.4f}")


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


# === Plot 2: Reconstructed Phantoms ===
fig2, axes = plt.subplots(1, num_nodes + 1, figsize=(4 * (num_nodes + 1), 4))
for i in range(num_nodes):
    ax = axes[i]
    ax.imshow(x_vars[i].asarray(), cmap='gray')
    ax.set_title(f"x_vars[{i}]")
    ax.axis('off')

ax = axes[-1]
ax.imshow(x_agg.asarray(), cmap='gray')
ax.set_title("x_agg")
ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "plot2_reconstructed_phantoms.png"), dpi=300)
plt.show()
# plt.close(fig2)

# === Plot 2b: True minus reconstructed per node ===
# Build residuals and use a shared color scale for fair comparison
residuals = [phantom.asarray() - x_vars[i].asarray() for i in range(num_nodes)]
v = max(np.max(np.abs(r)) for r in residuals) if residuals else 1.0

fig2b, axes = plt.subplots(1, num_nodes, figsize=(4 * num_nodes, 4))
for i in range(num_nodes):
    ax = axes[i] if num_nodes > 1 else axes
    im = ax.imshow(residuals[i], cmap='bwr', vmin=-v, vmax=v)
    ax.set_title(f"True minus x_vars[{i}]")
    ax.axis('off')

# One shared colorbar
cbar = fig2b.colorbar(im, ax=axes if num_nodes > 1 else [axes], fraction=0.046, pad=0.04)
cbar.set_label("Intensity difference")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "plot2b_true_minus_recon_nodes.png"), dpi=300)
plt.show()


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

# Path to save the results
save_path = os.path.join(output_dir, "reconstruction_data.pkl")

# Save all relevant reconstruction data and parameters
data_to_save = {
    "phantom": phantom.asarray(),
    "x_vars": [x.asarray() for x in x_vars],
    "x_agg": x_agg.asarray(),
    "sinograms": [s.asarray() for s in sinograms],
    "mse_lists": mse_lists,
    "mse_sino_lists": mse_sino_lists,
    "mse_agg_list": mse_agg_list,
    "mse_agg_sino_list": mse_agg_sino_list,
    "column_norms_all": column_norms_all,
    "params": {
        "N": N,
        "num_nodes": num_nodes,
        "niter": niter,
        "lambda_penalty": lambda_penalty,
        "alpha_tv": alpha_tv,
        "lambda_agg": lambda_agg,
        "gamma": gamma
    }
}

with open(save_path, "wb") as f:
    pickle.dump(data_to_save, f)

print(f"Saved reconstruction data to: {save_path}")

# Save parameter values to a plain text file
params_txt_path = os.path.join(output_dir, "reconstruction_params.txt")
with open(params_txt_path, "w") as f:
    for key, value in data_to_save["params"].items():
        f.write(f"{key} = {value}\n")

print(f"Saved reconstruction parameters to: {params_txt_path}")