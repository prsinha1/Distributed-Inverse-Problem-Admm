# block_6_admm_loop.py
import numpy as np
import math
import cvxpy as cp
import networkx as nx
from block_5_node_problem import build_node_problem
from block_4_tv_helpers import kt_subgrad_isotropic_tv_from_x
from datetime import datetime


import os
import matplotlib.pyplot as plt


def decentralized_admm(A_dense_list, sinograms, G, Wi_list, Qij_diag_fn,
                       N, lam_tv=0.01, rho=1.0, 
                       max_iters=10, max_inner_iters=100,
                       eps_pri=1e-1, eps_dual=1e-1, 
                       verbose=True, snapshot_dir=None,           # NEW
                       snapshot_every=None, snapshot_div=10, phantom_true=None):
    """
    Returns x_per_node as list of reconstructions, each length n
    Also returns history of residual norms
    """
    num_nodes = len(A_dense_list)
    n = A_dense_list[0].shape[1]

    # NEW: snapshot period
    if snapshot_dir is not None:
        os.makedirs(snapshot_dir, exist_ok=True)
    if snapshot_every is None:
        snapshot_every = max(1, max_iters // snapshot_div)


    # Initialize
    x = [np.zeros(n) for _ in range(num_nodes)]
    z = {}   # z[(i,j)] for each edge i<j
    y = {}   # y_{ij,i} and y_{ij,j}, store as y[(i,j,i)] and y[(i,j,j)]
    for i, j in G.edges():
        key = (min(i, j), max(i, j))
        z[key] = np.zeros(n)
        y[(key[0], key[1], i)] = np.zeros(n)
        y[(key[0], key[1], j)] = np.zeros(n)

    # For convenience precompute b_i
    b = [sinograms[i].reshape(-1) for i in range(num_nodes)]  # flatten sinograms

    pri_history = []
    dual_history = []
    obj_total_history = []                 # NEW
    obj_per_node_history = []              # NEW
    pri_per_node_history = []              # NEW
    dual_per_node_history = []             # NEW
    g_norm_history = []   # NEW
    eps_used_history = []        # NEW
    eps_target_history = []      # NEW



    # sinogram MSE histories
    mse_sino_per_node_history = []
    mse_sino_total_history = []

    # image space MSE histories
    img_mse_per_node_history = []
    img_mse_total_history = []

    print(f"Max ADMM Iteration in Block-6 B4 Loop = {max_iters}")
    for k in range(max_iters):
        # Node updates, each solves a cvxpy problem
        new_x = [None] * num_nodes
        obj_i = np.zeros(num_nodes)# NEW, objective per node at this k
        g_norm_this_k = np.zeros(num_nodes)   # NEW
        eps_used_this_k = np.zeros(num_nodes)     # NEW
        eps_target_this_k = np.zeros(num_nodes)   # NEW
        obj_i = np.zeros(num_nodes)
        g_norm_this_k = np.zeros(num_nodes)



        for i in range(num_nodes):
            Ai = A_dense_list[i]
            bi = b[i]

            neighbor_vs = []
            neighbor_Qs = []
            for j in G.neighbors(i):
                key = (min(i, j), max(i, j))
                # a_i = x_i + y_ij,i, a_j = x_j + y_ij,j
                a_i = x[i] + y[(key[0], key[1], i)]
                a_j = x[j] + y[(key[0], key[1], j)]
                # z update uses these later, for node problem we need v_ij = z_ij - y_ij,i
                v_ij = z[key] - y[(key[0], key[1], i)]
                neighbor_vs.append(v_ij)
                neighbor_Qs.append(Qij_diag_fn(i, j))

            xi_var, prob = build_node_problem(Ai, bi, rho, neighbor_vs, N, lam_tv, neighbor_Qs)
            # add start tag print here
            tag = f"node_{i}_outer_{k}"
            # target tolerance for this outer iteration
            eps0 = 2.0
            gamma = 0.005
            eps_target = eps0 / ((k + 1) ** (1.0 + gamma))

            # initial SCS tolerance to try
            eps_cap = 1e-2
            alpha = 1.0
            scs_eps_try = min(eps_cap, alpha * eps_target)

            # retry state
            accepted = False
            tighten_tries = 0
            max_tighten = 2

            while True:

                #print(f"\n=== SCS start {tag} ===")
                print(f"\n=== SCS start {tag} eps={scs_eps_try:.3e} ===")


                #prob.solve(solver=cp.SCS, eps=3e-1, max_iters=50, verbose=False, warm_start=True)
                #prob.solve(solver=cp.SCS, eps=scs_eps_try, max_iters=50, verbose=False, warm_start=True)
                prob.solve(solver=cp.SCS, eps=scs_eps_try, max_iters=200, acceleration_lookback=20, verbose=False, warm_start=True)

                try:
                    val = float(prob.value) if prob.value is not None else float("nan")
                except Exception:
                    val = float("nan")
                #print(f"=== SCS end   {tag} status={prob.status} obj={val:.6e} ===")
                iters = getattr(prob, "solver_stats", None)
                iters = getattr(iters, "num_iters", None)
                print(f"=== SCS end   {tag} status={prob.status} obj={val:.6e} iters={iters} ===")

                # ----- compute stationarity residual g_{x,i} for diagnostics -----
                x_candidate = np.array(xi_var.value).reshape(-1)

                # Build D_i and b_cons for this node from neighbor_Qs and neighbor_vs
                if len(neighbor_Qs) > 0:
                    D_i = np.sum(np.stack(neighbor_Qs, axis=0), axis=0)
                    b_cons = np.sum([q * v for q, v in zip(neighbor_Qs, neighbor_vs)], axis=0)
                else:
                    D_i = np.zeros_like(x_candidate)
                    b_cons = np.zeros_like(x_candidate)

                data_grad = Ai.T @ (Ai @ x_candidate - bi)
                cons_grad = rho * (D_i * x_candidate - b_cons)
                tv_grad = lam_tv * kt_subgrad_isotropic_tv_from_x(x_candidate, N)
                g_vec = data_grad + cons_grad + tv_grad
                g_norm = float(np.linalg.norm(g_vec))
                print(f"[node {i} outer {k}]  ||g_x,i||_2 = {g_norm:.3e}")
                print(f"[node {i} outer {k}]  target = {eps_target:.3e}")
                g_norm_this_k[i] = g_norm   # NEW


                if g_norm <= eps_target:
                    accepted = True
                    new_x[i] = x_candidate
                    obj_i[i] = val
                    eps_used_this_k[i] = scs_eps_try          # NEW
                    eps_target_this_k[i] = eps_target         # NEW
                    print(f"[node {i} outer {k}]  accepted\n")
                    break

                if tighten_tries >= max_tighten:
                    # accept current iterate to keep outer loop moving
                    new_x[i] = x_candidate
                    obj_i[i] = val
                    eps_used_this_k[i] = scs_eps_try          # NEW
                    eps_target_this_k[i] = eps_target         # NEW

                    print(f"[node {i} outer {k}]  reached tighten cap with ||g||={g_norm:.3e} > target={eps_target:.3e}\n")
                    break

                # tighten and retry
                tighten_tries += 1
                scs_eps_try = scs_eps_try / 5.0



            print(f"=== SCS end   {tag} status={prob.status} obj={val:.6e} ===")

            #new_x[i] = xi_var.value
        g_norm_history.append(g_norm_this_k.copy())   # NEW
        eps_used_history.append(eps_used_this_k.copy())        # NEW
        eps_target_history.append(eps_target_this_k.copy())    # NEW

        x = new_x

        # NEW: sinogram MSE per node and total at this outer iteration
        mse_i = np.zeros(num_nodes)
        for i in range(num_nodes):
            Ai = A_dense_list[i]
            ri = Ai @ x[i] - b[i]          # residual in sinogram domain
            mse_i[i] = float(ri @ ri)      # squared L2 norm

        mse_sino_per_node_history.append(mse_i.copy())
        mse_sino_total_history.append(float(np.sum(mse_i)))

        if phantom_true is not None:
            phantom_vec = phantom_true.ravel() if hasattr(phantom_true, "ndim") and phantom_true.ndim == 2 else phantom_true
            img_mse_node = np.zeros(num_nodes)
            for i in range(num_nodes):
                err = x[i] - phantom_vec
                img_mse_node[i] = float(err @ err)
        img_mse_per_node_history.append(img_mse_node.copy())
        img_mse_total_history.append(float(np.sum(img_mse_node)))



        # Edge updates z
        new_z = {}
        for i, j in G.edges():
            key = (min(i, j), max(i, j))
            Wi = Wi_list[i]
            Wj = Wi_list[j]

            a_i = x[i] + y[(key[0], key[1], i)]
            a_j = x[j] + y[(key[0], key[1], j)]

            # z_ij = (Wi + Wj)^{-1} (Wi a_i + Wj a_j), all diagonal so entrywise
            num =  a_i + a_j #Wi*a_i + Wj*a_j #
            den =  2.0 #Wi + Wj 
            new_z[key] = num / den

        # Dual updates
        new_y = {}
        for i, j in G.edges():
            key = (min(i, j), max(i, j))
            new_y[(key[0], key[1], i)] = y[(key[0], key[1], i)] + x[i] - new_z[key]
            new_y[(key[0], key[1], j)] = y[(key[0], key[1], j)] + x[j] - new_z[key]

        # Residuals
        # Primal residual stacks [x_i - z_ij, x_j - z_ij] across edges
        r2 = 0.0
        s2 = 0.0
        pri_node = np.zeros(num_nodes) # NEW, per node primal residual squared
        dual_node = np.zeros(num_nodes) # NEW, per node dual residual squared
        for i, j in G.edges():
            key = (min(i, j), max(i, j))
            ri = x[i] - new_z[key]
            rj = x[j] - new_z[key]
            r2 += np.sum(ri * ri) + np.sum(rj * rj)
            # dual residual s = rho (z^{k+1} - z^k)
            # NEW, attribute primal pieces to incident nodes
            pri_node[i] += np.sum(ri * ri)
            pri_node[j] += np.sum(rj * rj)

            dz = new_z[key] - z[key]
            s2 += rho * rho * np.sum(dz * dz)

            dual_piece = rho * rho * np.sum(dz * dz) # NEW per-node dual
            dual_node[i] += dual_piece
            dual_node[j] += dual_piece

        pri_norm = math.sqrt(r2)
        dual_norm = math.sqrt(s2)
        pri_history.append(pri_norm)
        dual_history.append(dual_norm)
        # NEW: store objectives
        obj_per_node_history.append(obj_i.copy())
        obj_total_history.append(np.sum(obj_i))

        pri_per_node_history.append(np.sqrt(pri_node))
        dual_per_node_history.append(np.sqrt(dual_node))

        z = new_z
        y = new_y

        # NEW: periodic snapshots every snapshot_every outer iterations
        if snapshot_dir is not None and ((k + 1) % snapshot_every == 0):
            it_tag = f"iter_{k+1:04d}"
            for i, xi in enumerate(x):
                img = xi.reshape(N, N)
                np.save(os.path.join(snapshot_dir, f"{it_tag}_node_{i}.npy"), img)
                plt.figure(figsize=(5, 5))
                plt.imshow(img, cmap="gray")
                plt.title(f"{it_tag}  node {i}")
                plt.axis("off")
                plt.tight_layout()
                plt.savefig(os.path.join(snapshot_dir, f"{it_tag}_node_{i}.png"), dpi=220)
                plt.close()

        if verbose and k % 10 == 0:
            print(f"iter {k}, primal {pri_norm:.3e}, dual {dual_norm:.3e}")

        if pri_norm < eps_pri and dual_norm < eps_dual:
            if verbose:
                print(f"stopped at iter {k}, primal {pri_norm:.3e}, dual {dual_norm:.3e}")
            break

        # === Save local parameters for debugging ===
        # === Save local parameters for debugging ===
    try:
        log_dir = snapshot_dir if snapshot_dir is not None else os.getcwd()
        os.makedirs(log_dir, exist_ok=True)
        debug_file = os.path.join(log_dir, "admm_internal_params.txt")
        with open(debug_file, "w") as f:
            f.write("===== ADMM Internal Parameters =====\n")
            f.write(f"rho = {rho}\n")
            f.write(f"lambda_tv = {lam_tv}\n")
            f.write(f"Number of nodes = {num_nodes}\n")
            f.write(f"Calib alpha = {alpha if 'alpha' in locals() else 'N/A'}\n")
            f.write(f"Eps cap = {eps_cap if 'eps_cap' in locals() else 'N/A'}\n")
            f.write(f"Date-Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    except Exception as e:
        print(f"[WARN] Could not save internal params: {e}")



    return x, {
    "primal": pri_history,
    "dual": dual_history,
    "pri_per_node": pri_per_node_history,
    "dual_per_node": dual_per_node_history,
    "obj_per_node": obj_per_node_history,
    "obj_total": obj_total_history,
    "mse_sino_per_node": mse_sino_per_node_history,   # NEW
    "mse_sino_total": mse_sino_total_history,         # NEW
    "img_mse_per_node": img_mse_per_node_history,
    "img_mse_total": img_mse_total_history,
    "g_norm_history": g_norm_history,   # NEW b bngv
    "eps_used_history": eps_used_history,           # NEW
    "eps_target_history": eps_target_history,       # NEW


}
