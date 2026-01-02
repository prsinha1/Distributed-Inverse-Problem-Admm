# block_7_run_graphs.py
# Run decentralized ADMM on MST, chain, and kNN graphs built by Block 3.

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from block_2_load_odl_data import load_odl_data
from block_3_graph_and_precisions import build_pixel_connected_Q_provider  # masked Q and union graph
from block_6_admm_loop_ver2 import decentralized_admm

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
                     max_iters, max_inner_iters, eps_pri, eps_dual,
                     base_dir, out_root, show_plots=False, verbose=True, snapshot_div=5, phantom_true=None):
    
    tag = f"{strategy}_k{k}" if strategy == "knn" else strategy
    out_dir = os.path.join(out_root, tag)
    os.makedirs(out_dir, exist_ok=True)

    # === Save run parameters to a text file ===
    param_file = os.path.join(out_dir, "run_parameters.txt")
    with open(param_file, "w") as f:
        f.write("===== Global Parameters =====\n")
        f.write(f"Strategy: {strategy}\n")
        f.write(f"Number of nodes: {N}\n")
        f.write(f"Lambda_TV: {lam_tv}\n")
        f.write(f"Rho: {rho}\n")
        f.write(f"Max ADMM iterations: {max_iters}\n")
        f.write(f"Max Inner iterations: {max_inner_iters}\n")
        f.write(f"Date-Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n===== Solver Defaults =====\n")
        #f.write("SCS max_iters = 400\n")
        f.write("Initial eps_try = 3e-3\n")
        f.write("Tightening factor = 0.3\n")
        f.write("Eps cap = 1e-2\n")
        f.write("Warm start = True\n")
        f.write("\n===== Data Info =====\n")
        f.write(f"Output directory: {out_dir}\n")
        f.write(f"A_dense_list file used: {data.get('A_dense_list_path', 'unknown')}\n")




    # Build per pixel masks and masked weights, get union node graph
    G_union, Wi_list, Qij_diag_fn_masked, keep = build_pixel_connected_Q_provider(base_dir=base_dir,
    strategy=strategy,
    k=k,
    seed=123,
    q_mode= "arithmetic",#"harmonic", #"arithmetic"
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
        max_inner_iters=max_inner_iters,
        eps_pri=eps_pri,
        eps_dual=eps_dual,
        verbose=True,
        snapshot_dir=snap_dir,          # NEW
        snapshot_every=snap_every,
        snapshot_div=snapshot_div,
        phantom_true=phantom_true
    )

    # ----- place right after x_list, hist = decentralized_admm(...) and any existing plots -----

    # === per node ||g_x,i||_2 over outer iterations ===
    #g_hist = np.asarray(hist.get("g_norm_history", []))  # shape (T, num_nodes)


    g_hist = np.asarray(hist.get("g_norm_history", []))  # shape (T, num_nodes)
    eps_used_hist = np.asarray(hist.get("eps_used_history", []))       # NEW
    eps_targ_hist = np.asarray(hist.get("eps_target_history", []))     # NEW


    if g_hist.size == 0:
        # create a tiny figure that just says "no data" so the file always exists
        plt.figure(figsize=(5, 3))
        plt.text(0.5, 0.5, "g_norm_history is empty", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{tag}_g_norm_stats.png"), dpi=220)
        plt.close()
    else:
        T = g_hist.shape[0]
        iters = np.arange(T)

        # per node curves with ratio overlay
        plt.figure(figsize=(7, 4))
        ax1 = plt.gca()
        for i in range(g_hist.shape[1]):
            ax1.semilogy(iters, g_hist[:, i], label=f"node {i}")
        ax1.set_xlabel("iteration")
        ax1.set_ylabel(r"$\|g_{x,i}\|_2$")
        ax1.set_title(f"Per node stationarity residual, {tag}")
        ax1.grid(True, which="both")

        # overlay ratio eps_target / eps_used on a second y axis if available
        if eps_used_hist.size > 0 and eps_targ_hist.size > 0:
            ratio = np.divide(eps_targ_hist, np.maximum(eps_used_hist, 1e-16))
            ax2 = ax1.twinx()
            for i in range(ratio.shape[1]):
                ax2.semilogy(iters, ratio[:, i], linestyle=":", alpha=0.6)   # light dotted overlay
            ax2.set_ylabel(r"$\varepsilon_k / \text{eps\_used}$")

        # legend only for g curves to keep it readable
        ax1.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{tag}_g_norm_per_node.png"), dpi=220)
        plt.close()

        # mean and median across nodes
        mean_g = g_hist.mean(axis=1)
        med_g  = np.median(g_hist, axis=1)

        plt.figure(figsize=(6, 4))
        plt.semilogy(iters, mean_g, label="mean")
        plt.semilogy(iters, med_g, label="median")
        plt.xlabel("iteration")
        plt.ylabel(r"$\|g_{x,i}\|_2$")
        plt.title(f"Mean and median stationarity residual, {tag}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{tag}_g_norm_stats.png"), dpi=220)
        plt.close()


    T = len(hist["primal"])
    iters = range(T)

    # 1 Objective per node versus outer iteration
    obj_per_node = np.array(hist["obj_per_node"]) if "obj_per_node" in hist else np.array(hist["obj_per_node_history"])
    plt.figure(figsize=(6, 4))
    
    for i in range(obj_per_node.shape[1]):
        #plt.plot(iters, obj_per_node[:, i], label=f"node {i}")
        plt.semilogy(iters, np.abs(obj_per_node[:, i]) + 1e-12, label=f"node {i}")

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
    #plt.plot(iters, obj_total)
    plt.semilogy(iters, np.abs(obj_total) + 1e-12)

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
    plt.ylabel("Log_10(L2 norm)")
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
    m_vec = np.array([sinograms[i].size for i in range(len(sinograms))], dtype=float)
    mse_per_node = mse_per_node / m_vec[np.newaxis, :]

    plt.figure(figsize=(6, 4))
    for i in range(mse_per_node.shape[1]):
        plt.semilogy(iters, mse_per_node[:, i], label=f"node {i}")
    plt.xlabel("iteration")
    plt.ylabel("sinogram MSE  log((1/m_i)*[||A_i x_i - b_i||_2^2])")
    plt.title(f"Per node sinogram MSE, {tag}")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{tag}_sino_mse_per_node.png"), dpi=220)
    plt.close()
    np.save(os.path.join(out_dir, f"{tag}_sino_mse_per_node.npy"), mse_per_node)

    # Total sinogram MSE vs outer iteration
    #mse_total = np.array(hist["mse_sino_total"])
    mse_total = np.array(hist["mse_sino_total"]) / float(m_vec.sum())

    plt.figure(figsize=(6, 4))
    plt.semilogy(iters, mse_total)
    plt.xlabel("iteration")
    plt.ylabel("total sinogram MSE  log_10((1/m)*[sum_i ||A_i x_i - b_i||_2^2])")
    plt.title(f"Total sinogram MSE, {tag}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{tag}_sino_mse_total.png"), dpi=220)
    plt.close()
    np.save(os.path.join(out_dir, f"{tag}_sino_mse_total.npy"), mse_total)
    # ----- end NEW -----

        # === NEW: image space MSE plots, consistent with sinogram plots ===
    img_mse_per_node = hist.get("img_mse_per_node", [])
    img_mse_total = hist.get("img_mse_total", [])

    img_mse_arr = np.asarray(hist.get("img_mse_per_node", []))
    if img_mse_arr.size > 0:
        n_pix = float(N * N)                     # total number of pixels per image
        img_mse_arr = img_mse_arr / n_pix        # normalize each node’s MSE by image length

    if img_mse_arr.size > 0:
        T = img_mse_arr.shape[0]
        iters = np.arange(T)
        N_local = img_mse_arr.shape[1]
        plt.figure(figsize=(6, 4))
        for i in range(N_local):
            plt.semilogy(iters, img_mse_arr[:, i], label=f"node {i}")
        plt.xlabel("iteration"); plt.ylabel("image MSE  log_10((1/N)*||x_i - x_true||_2^2)")
        plt.title(f"Per node image MSE, {tag}")
        plt.legend(ncol=2, fontsize=8); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{tag}_img_mse_per_node.png"), dpi=220)
        plt.close()
        np.save(os.path.join(out_dir, f"{tag}_img_mse_per_node.npy"), img_mse_arr)

    #img_mse_total = np.asarray(hist.get("img_mse_total", []))
    img_mse_total = np.asarray(hist.get("img_mse_total", [])) / float(N * N)

    if img_mse_total.size > 0:
        plt.figure(figsize=(6, 4))
        plt.semilogy(np.arange(len(img_mse_total)), img_mse_total)
        plt.xlabel("iteration"); plt.ylabel("total image MSE  log_10((1/N)*sum_i ||x_i - x_true||_2^2)")
        plt.title(f"Total image MSE, {tag}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{tag}_img_mse_total.png"), dpi=220)
        plt.close()
        np.save(os.path.join(out_dir, f"{tag}_img_mse_total.npy"), img_mse_total)
    

    print(f"[Done] Saved outputs to {out_dir}")
    return x_list, hist


def main():
    # Settings
    N = 64
    num_nodes = 5
    lam_tv = 0.02
    rho = 2.0
    max_iters = 200
    max_inner_iters = 100
    eps_pri = 1e-3
    eps_dual = 1e-3
    noise_level = 0.005
    base_dir = "saved_operators_Incmp_Span"
    snapshot_div = 2  # for snapshot frequency calculation

    # Load data
    data = load_odl_data(base_dir=base_dir, N=N, num_nodes=num_nodes, noise_level=noise_level)
    phantom_true = data.get("phantom", None)

    # Output folder
    out_root = f"Recon_Out_ADMM_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"Max ADMM Iteration in Block-7 B4 run_one {max_iters}")

    phantom_true = data.get("phantom", None)  # None if not provided


    # MST
    #run_one_strategy("mst", k=0, data=data, N=N, lam_tv=lam_tv, rho=rho,
    #                 max_iters=max_iters, eps_pri=eps_pri, eps_dual=eps_dual,
    #                 base_dir=base_dir, out_root=out_root, snapshot_div=snapshot_div, phantom_true=phantom_true)

    # Chain
    #run_one_strategy("chain", k=0, data=data, N=N, lam_tv=lam_tv, rho=rho,
    #                 max_iters=max_iters, eps_pri=eps_pri, eps_dual=eps_dual,
    #                 base_dir=base_dir, out_root=out_root, snapshot_div=snapshot_div, phantom_true=phantom_true)

    # kNN with k neighbors per node
    run_one_strategy("knn", k=2, data=data, N=N, lam_tv=lam_tv, rho=rho,
                     max_iters=max_iters, max_inner_iters=max_inner_iters, eps_pri=eps_pri, eps_dual=eps_dual,
                     base_dir=base_dir, out_root=out_root, snapshot_div=snapshot_div, phantom_true=phantom_true)


if __name__ == "__main__":
    main()
