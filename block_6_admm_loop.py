# block_6_admm_loop.py
# Adds: chunked SCS solves with tagged logs and optional image snapshots.

import os
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from block_5_node_problem import build_node_problem

# Your existing helpers should still be present or imported in this file:
# - build_node_problem(Ai, bi, rho, neighbor_vs, N, lam_tv, neighbor_Qs)
# - Any TV helpers you already use

def _scs_solve_in_chunks(prob, x_var, N, tag, out_dir,
                         total_iters, chunk_iters,
                         use_indirect=True, eps=3e-3, alpha=1.5,
                         acceleration=1, lookback=10, scale=1e-1,
                         verbose=True, save_every_chunks=1):
    """
    Run SCS in short chunks, warm start between calls, save the image after selected chunks.

    prob, x_var: the CVXPY problem and decision variable for this node subproblem
    N: image side length, x has length N*N
    tag: will appear in the console log and file names, example 'node_2_outer_7'
    out_dir: folder to save snapshots, pass None to skip saving
    total_iters: total SCS iterations to spend on this node update
    chunk_iters: iterations per chunk, must be >= 1
    save_every_chunks: save after every k chunks, default 1 means save every chunk
    """
    remaining = int(total_iters)
    chunk_id = 0
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    while remaining > 0:
        this_chunk = min(int(chunk_iters), remaining)
        if verbose:
            print(f"\n=== SCS start  {tag}, chunk {chunk_id}, iters {this_chunk} ===")
        prob.solve(
            solver=cp.SCS,
            verbose=verbose,
            warm_start=True,
            use_indirect=use_indirect,
            eps=eps, eps_abs=eps, eps_rel=eps,
            max_iters=this_chunk,
            alpha=alpha,
            acceleration=acceleration,
            acceleration_lookback=lookback,
            scale=scale
        )
        if verbose:
            val = prob.value if prob.value is not None else float("nan")
            print(f"=== SCS end    {tag}, chunk {chunk_id}, status={prob.status}, obj={val:.6e} ===")

        if out_dir is not None and (chunk_id % int(save_every_chunks) == 0):
            x_now = x_var.value
            if x_now is not None:
                img = x_now.reshape(N, N, order="F")
                np.save(os.path.join(out_dir, f"{tag}_chunk_{chunk_id}.npy"), img)
                plt.figure(figsize=(5, 5))
                plt.imshow(img, cmap="gray")
                plt.title(f"{tag} after chunk {chunk_id}")
                plt.axis("off")
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f"{tag}_chunk_{chunk_id}.png"), dpi=220)
                plt.close()

        remaining -= this_chunk
        chunk_id += 1


def decentralized_admm(A_dense_list, sinograms, G, Wi_list, Qij_diag_fn,
                       N, lam_tv=0.01, rho=1.0,
                       max_iters=200,
                       eps_pri=1e-3, eps_dual=1e-3,
                       verbose=True,
                       # new optional SCS controls, all default to your old behavior
                       scs_total_iters=100,
                       scs_chunk_iters=None,              # None means single SCS call
                       scs_snapshot_dir=None,             # set a folder to save images
                       scs_use_indirect=True,
                       scs_eps=3e-3, scs_alpha=1.5,
                       scs_acceleration=1, scs_lookback=10, scs_scale=1e-1,
                       scs_save_every_chunks=1):
    """
    Your existing ADMM loop, now with optional chunked SCS solves and snapshots.

    Only the solver call inside the node update has been wrapped. All math is unchanged.
    """

    num_nodes = len(A_dense_list)
    n = N * N

    # Initialize variables as in your original code
    x = [np.zeros(n) for _ in range(num_nodes)]
    u = {}   # edge duals if you use them
    z = {}   # edge consensus if you use them

    # Any other initializations you already had go here

    history = {
        "primal_res": [],
        "dual_res": [],
        "obj": []
    }

    for k in range(max_iters):
        if verbose:
            print(f"\n[ADMM] outer iter {k+1}/{max_iters}")

        new_x = [None] * num_nodes

        # Node updates
        for i in range(num_nodes):
            Ai = A_dense_list[i]
            bi = sinograms[i]

            # Collect neighbor terms for node i from graph G, as in your code
            neighbor_vs = []   # fill from your current implementation
            neighbor_Qs = []   # fill from your current implementation

            # Your existing builder
            xi_var, prob = build_node_problem(Ai, bi, rho, neighbor_vs, N, lam_tv, neighbor_Qs)

            tag = f"node_{i}_outer_{k}"

            if scs_chunk_iters is None:
                # single shot, exactly like before, but tagged
                print(f"\n=== SCS start  {tag}, one shot, iters {scs_total_iters} ===")
                prob.solve(solver=cp.SCS,
                           eps=scs_eps, max_iters=scs_total_iters,
                           verbose=True, warm_start=True)
                val = prob.value if prob.value is not None else float("nan")
                print(f"=== SCS end    {tag}, status={prob.status}, obj={val:.6e} ===")
            else:
                # chunked run, optional snapshots
                node_snap_dir = None
                if scs_snapshot_dir is not None:
                    node_snap_dir = os.path.join(scs_snapshot_dir, f"node_{i}")
                _scs_solve_in_chunks(
                    prob, xi_var, N, tag, node_snap_dir,
                    total_iters=scs_total_iters, chunk_iters=scs_chunk_iters,
                    use_indirect=scs_use_indirect, eps=scs_eps, alpha=scs_alpha,
                    acceleration=scs_acceleration, lookback=scs_lookback, scale=scs_scale,
                    verbose=True, save_every_chunks=scs_save_every_chunks
                )

            new_x[i] = np.array(xi_var.value).reshape(-1)

        # Consensus and dual updates here, exactly as your current code
        # z update, u update, residual bookkeeping
        # history["primal_res"].append(...)
        # history["dual_res"].append(...)
        # history["obj"].append(...)

        x = new_x

        # Your stopping tests
        # if stop: break

    return x, history
