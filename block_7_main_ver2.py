# block_7_run_graphs.py
# Run decentralized ADMM on MST, chain, and kNN graphs built by Block 3.

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from block_2_load_odl_data import load_odl_data
from block_3_graph_and_precisions import build_pixel_connected_Q_provider  # masked Q and union graph
from block_6_admm_loop_ver0 import decentralized_admm

import sys
import contextlib

def save_recons(x_list, N, out_dir, tag):
    os.makedirs(out_dir, exist_ok=True)
    for i, x_i in enumerate(x_list):
        img = x_i.reshape(N, N)
        np.save(os.path.join(out_dir, f"{tag}_node_{i}.npy"), img)
        plt.figure(figsize=(5, 5))
        plt.imshow(img, cmap="gray")
        plt.title(f"{tag}  node {i}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{tag}_node_{i}.png"), dpi=220)
        plt.close()


def run_one_strategy(strategy, k, data, N, lam_tv, rho,
                     max_iters, eps_pri, eps_dual,
                     base_dir, out_root, show_plots=False, verbose=True, snapshot_div=5):
    
    tag = f"{strategy}_k{k}" if strategy == "knn" else strategy
    out_dir = os.path.join(out_root, tag)
    os.makedirs(out_dir, exist_ok=True)



    # Build per pixel masks and masked weights, get union node graph
    G_union, Wi_list, Qij_diag_fn_masked, keep = build_pixel_connected_Q_provider(
    base_dir=base_dir,
    strategy=strategy,
    k=k,
    seed=123,
    q_mode= "harmonic", #"arithmetic"
    verbose=verbose,
    plot_union=True,
    show_plots=show_plots,
    output_dir=os.path.join(out_dir, "union_figs"),
)


    # Pull tomography data
    A_dense_list = data["A_dense_list"]
    sinograms = data["sinograms"]

    print(f"Max ADMM Iteration in Block-7 Inside run_one {max_iters}")
    print(f"\n[Run] ADMM on strategy {tag}")

    snap_dir = os.path.join(out_dir, "snapshots")
    os.makedirs(snap_dir, exist_ok=True)
    #snapshot_div = 5
    snap_every = max(1, max_iters // snapshot_div)


    x_list, hist = decentralized_admm(
        A_dense_list=A_dense_list,
        sinograms=sinograms,
        G=G_union,
        Wi_list=Wi_list,
        Qij_diag_fn=Qij_diag_fn_masked,
        N=N,
        lam_tv=lam_tv,
        rho=rho,
        max_iters=max_iters,
        eps_pri=eps_pri,
        eps_dual=eps_dual,
        verbose=True,
        snapshot_dir=snap_dir,          # NEW
        snapshot_every=snap_every,
        snapshot_div=snapshot_div,
        phantom_true=data["phantom"]
    )

    # ----- place right after x_list, hist = decentralized_admm(...) and any existing plots -----

    

    T = len(hist["primal"])
    iters = range(T)

    # 1 Objective per node versus outer iteration
    obj_per_node = np.array(hist["obj_per_node"]) if "obj_per_node" in hist else np.array(hist["obj_per_node_history"])
    plt.figure(figsize=(6, 4))
    for i in range(obj_per_node.shape[1]):
        plt.plot(iters, obj_per_node[:, i], label=f"node {i}")
    plt.xlabel("iteration")
    plt.ylabel("objective")
    plt.title(f"Objective per node, {tag}")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{tag}_obj_per_node.png"), dpi=220)
    plt.close()
    np.save(os.path.join(out_dir, f"{tag}_obj_per_node.npy"), obj_per_node)

    # 2 Total objective versus outer iteration
    obj_total = np.array(hist["obj_total"]) if "obj_total" in hist else np.array(hist["obj_total_history"])
    plt.figure(figsize=(6, 4))
    plt.plot(iters, obj_total)
    plt.xlabel("iteration")
    plt.ylabel("objective")
    plt.title(f"Total objective, {tag}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{tag}_obj_total.png"), dpi=220)
    plt.close()
    np.save(os.path.join(out_dir, f"{tag}_obj_total.npy"), obj_total)

    # 3 Primal residual per node versus outer iteration
    pri_per_node = np.array(hist["pri_per_node"]) if "pri_per_node" in hist else np.array(hist["pri_per_node_history"])
    plt.figure(figsize=(6, 4))
    for i in range(pri_per_node.shape[1]):
        plt.semilogy(iters, pri_per_node[:, i], label=f"node {i}")
    plt.xlabel("iteration")
    plt.ylabel("primal residual")
    plt.title(f"Primal residual per node, {tag}")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{tag}_pri_per_node.png"), dpi=220)
    plt.close()
    np.save(os.path.join(out_dir, f"{tag}_pri_per_node.npy"), pri_per_node)

    # 4 Dual residual per node versus outer iteration
    dual_per_node = np.array(hist["dual_per_node"]) if "dual_per_node" in hist else np.array(hist["dual_per_node_history"])
    plt.figure(figsize=(6, 4))
    for i in range(dual_per_node.shape[1]):
        plt.semilogy(iters, dual_per_node[:, i], label=f"node {i}")
    plt.xlabel("iteration")
    plt.ylabel("dual residual")
    plt.title(f"Dual residual per node, {tag}")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{tag}_dual_per_node.png"), dpi=220)
    plt.close()
    np.save(os.path.join(out_dir, f"{tag}_dual_per_node.npy"), dual_per_node)
    # ----- end of added block -----


    # Save reconstructions and simple residual plots
    save_recons(x_list, N, out_dir, tag)

    # Plot residual history
    #import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.semilogy(hist["primal"], label="primal")
    plt.semilogy(hist["dual"], label="dual")
    plt.xlabel("iteration")
    plt.ylabel("norm")
    plt.title(f"Residuals, {tag}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{tag}_residuals.png"), dpi=220)
    plt.close()

    # Also save raw histories
    np.save(os.path.join(out_dir, f"{tag}_primal_hist.npy"), np.array(hist["primal"]))
    np.save(os.path.join(out_dir, f"{tag}_dual_hist.npy"), np.array(hist["dual"]))

    # ----- NEW, sinogram MSE plots -----
    T = len(hist["primal"])
    iters = range(T)

    # Per node sinogram MSE vs outer iteration
    mse_per_node = np.array(hist["mse_sino_per_node"])
    plt.figure(figsize=(6, 4))
    for i in range(mse_per_node.shape[1]):
        plt.semilogy(iters, mse_per_node[:, i], label=f"node {i}")
    plt.xlabel("iteration")
    plt.ylabel("sinogram MSE  ||A_i x_i - b_i||_2^2")
    plt.title(f"Per node sinogram MSE, {tag}")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{tag}_sino_mse_per_node.png"), dpi=220)
    plt.close()
    np.save(os.path.join(out_dir, f"{tag}_sino_mse_per_node.npy"), mse_per_node)

    # Total sinogram MSE vs outer iteration
    mse_total = np.array(hist["mse_sino_total"])
    plt.figure(figsize=(6, 4))
    plt.semilogy(iters, mse_total)
    plt.xlabel("iteration")
    plt.ylabel("total sinogram MSE  sum_i ||A_i x_i - b_i||_2^2")
    plt.title(f"Total sinogram MSE, {tag}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{tag}_sino_mse_total.png"), dpi=220)
    plt.close()
    np.save(os.path.join(out_dir, f"{tag}_sino_mse_total.npy"), mse_total)
    # ----- end NEW -----


    print(f"[Done] Saved outputs to {out_dir}")
    return x_list, hist


def main():
    # Settings
    N = 64
    num_nodes = 5
    lam_tv = 0.02
    rho = 1.0
    max_iters = 50#200
    eps_pri = 1e-3
    eps_dual = 1e-3
    noise_level = 0.005
    base_dir = "saved_operators_Incmp_Span"
    snapshot_div = 3  # for snapshot frequency calculation

    # Load data
    data = load_odl_data(base_dir=base_dir, N=N, num_nodes=num_nodes, noise_level=noise_level)

    # Output folder
    out_root = f"Recon_Out_ADMM_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"Max ADMM Iteration in Block-7 B4 run_one {max_iters}")

    # MST
    run_one_strategy("mst", k=0, data=data, N=N, lam_tv=lam_tv, rho=rho,
                     max_iters=max_iters, eps_pri=eps_pri, eps_dual=eps_dual,
                     base_dir=base_dir, out_root=out_root, snapshot_div=snapshot_div)

    # Chain
    run_one_strategy("chain", k=0, data=data, N=N, lam_tv=lam_tv, rho=rho,
                     max_iters=max_iters, eps_pri=eps_pri, eps_dual=eps_dual,
                     base_dir=base_dir, out_root=out_root, snapshot_div=snapshot_div)

    # kNN with k neighbors per node
    run_one_strategy("knn", k=2, data=data, N=N, lam_tv=lam_tv, rho=rho,
                     max_iters=max_iters, eps_pri=eps_pri, eps_dual=eps_dual,
                     base_dir=base_dir, out_root=out_root, snapshot_div=snapshot_div)


if __name__ == "__main__":
    main()
