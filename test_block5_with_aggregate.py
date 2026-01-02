# test_block5_with_aggregate.py
# Goal: exercise Block 5 with A_agg and aggregate sinogram, using rho=0

import os
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from block_2_load_odl_data import load_odl_data
from block_4_tv_helpers import isotropic_tv_on_vector
from block_5_node_problem import build_node_problem


def pick_solver(pref=("ECOS", "SCS")):
    avail = set(cp.installed_solvers())
    for s in pref:
        if s in avail:
            return s
    return None


def stack_aggregate(A_dense_list, sinograms):
    A_agg = np.vstack(A_dense_list)
    b_list = [b.reshape(-1) for b in sinograms]
    b_agg = np.concatenate(b_list, axis=0)
    return A_agg, b_agg


def run_block5_single(A_agg, b_agg, N, lam_tv=0.0, rho=0.0, solver=None, out_dir="block5_out"):
    os.makedirs(out_dir, exist_ok=True)

    if solver is None:
        solver = pick_solver()
        if solver is None:
            # If no conic solver is available and TV is off, do closed form LS
            if lam_tv == 0.0:
                print("[Block5] No conic solver found, using numpy lstsq for pure LS")
                x_hat = np.linalg.lstsq(A_agg, b_agg, rcond=None)[0]
                img = x_hat.reshape(N, N)
                plt.figure(figsize=(5, 5))
                plt.imshow(img, cmap="gray")
                plt.title("Block5 recon, lam_tv=0, rho=0, lstsq")
                plt.axis("off")
                out_path = os.path.join(out_dir, "recon_block5_lam0_rho0_lstsq.png")
                plt.tight_layout()
                plt.savefig(out_path, dpi=250)
                plt.close()
                resid = A_agg @ x_hat - b_agg
                mse = float(np.mean(resid**2))
                print(f"[Block5] Data MSE {mse:.6e}")
                return x_hat, {"mse": mse, "tv": None, "status": "lstsq", "obj": 0.5 * np.sum(resid**2)}
            else:
                raise RuntimeError("No ECOS or SCS found. Install one solver or set lam_tv=0 for lstsq fallback.")

    # No neighbors when rho = 0
    neighbor_terms = []
    Qij_terms = []

    xi, prob = build_node_problem(
        Ai=A_agg,
        bi=b_agg,
        rho=rho,
        neighbor_terms=neighbor_terms,
        N=N,
        lam_tv=lam_tv,
        Qij_terms=Qij_terms,
    )

    print(f"[Block5] Building problem with n = {A_agg.shape[1]}, m = {A_agg.shape[0]}")
    prob.solve(solver=solver, verbose=False)
    print(f"[Block5] Status {prob.status}, objective {prob.value:.6e}")

    x_hat = xi.value
    if x_hat is None:
        raise RuntimeError("Solver failed to return a solution")

    img = x_hat.reshape(N, N)
    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap="gray")
    plt.title(f"Block5 recon, lam_tv={lam_tv}, rho={rho}, solver={solver}")
    plt.axis("off")
    out_path = os.path.join(out_dir, f"recon_block5_lam{lam_tv}_rho{rho}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()
    print(f"[Block5] Saved reconstruction to {out_path}")

    resid = A_agg @ x_hat - b_agg
    mse = float(np.mean(resid**2))
    print(f"[Block5] Data MSE {mse:.6e}")

    tv_val = None
    if lam_tv > 0:
        tv_var = cp.Variable(A_agg.shape[1])
        tv_expr = isotropic_tv_on_vector(tv_var, N)
        tv_prob = cp.Problem(cp.Minimize(tv_expr), [tv_var == x_hat])
        tv_prob.solve(solver=solver, verbose=False)
        tv_val = tv_expr.value
        print(f"[Block5] TV value {tv_val:.6e}")

    return x_hat, {"mse": mse, "tv": tv_val, "status": prob.status, "obj": prob.value}


def main():
    N = 64
    num_nodes = 5
    base_dir = "saved_operators_Incmp_Span"
    noise_level = 0.005

    data = load_odl_data(base_dir=base_dir, N=N, num_nodes=num_nodes, noise_level=noise_level)
    A_dense_list = data["A_dense_list"]
    sinograms = data["sinograms"]

    A_agg, b_agg = stack_aggregate(A_dense_list, sinograms)
    print(f"[Agg] A_agg shape {A_agg.shape}, b_agg shape {b_agg.shape}")

    solver = pick_solver()
    if solver is None:
        print("[Block5] Neither ECOS nor SCS appears installed. "
              "pip install ecos or pip install scs for the TV case. Falling back where possible.")

    print("\n[Run] Block 5 with lam_tv = 0, rho = 0")
    run_block5_single(A_agg, b_agg, N, lam_tv=0.0, rho=0.0, solver=solver, out_dir="block5_out")

    print("\n[Run] Block 5 with lam_tv = 0.02, rho = 0")
    run_block5_single(A_agg, b_agg, N, lam_tv=0.02, rho=0.0, solver=solver, out_dir="block5_out")


if __name__ == "__main__":
    main()
