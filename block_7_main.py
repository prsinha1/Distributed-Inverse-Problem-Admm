# block_7_main.py
# Runs decentralized ADMM for a chosen graph strategy and enables chunked SCS logging and snapshots.

import os
import pickle
import numpy as np

from block_3_graph_and_precisions import build_pixel_connected_Q_provider
from block_2_load_odl_data import load_odl_data
#build_graphs_and_Q_provider
from block_6_admm_loop import decentralized_admm

def main():
    # Load your prepared data the same way you already do
    # Expecting a dict with 'A_dense_list', 'sinograms', 'Wi_list', 'N', and maybe others
    with open("saved_operators_Incmp_Span/data_block2.pkl", "rb") as f:
        data = pickle.load(f)

    A_dense_list = data["A_dense_list"]
    sinograms = data["sinograms"]
    Wi_list = data["Wi_list"]
    N = data["N"]

    out_root = data.get("out_dir", "recon_out")
    os.makedirs(out_root, exist_ok=True)

    # Pick a graph strategy the way you do now
    strategy = "mst"     # or "chain" or "knn"
    k = 2                # used only for knn

    G_union, Qij_diag_fn_masked, diagW = build_graphs_and_Q_provider(
        Wi_list=Wi_list,
        strategy=strategy,
        k=k,
        ensure_connected=True,
        out_dir=os.path.join(out_root, strategy)
    )

    lam_tv = 0.02
    rho = 1.0
    max_admm = 50
    eps_pri = 1e-3
    eps_dual = 1e-3

    # Folder to save interim SCS snapshots
    scs_snap_dir = os.path.join(out_root, strategy, "scs_snapshots")
    os.makedirs(scs_snap_dir, exist_ok=True)

    x_list, hist = decentralized_admm(
        A_dense_list=A_dense_list,
        sinograms=sinograms,
        G=G_union,
        Wi_list=Wi_list,
        Qij_diag_fn=Qij_diag_fn_masked,
        N=N,
        lam_tv=lam_tv,
        rho=rho,
        max_iters=max_admm,
        eps_pri=eps_pri,
        eps_dual=eps_dual,
        verbose=True,
        # new SCS options
        scs_total_iters=100,            # budget per node update
        scs_chunk_iters=20,             # save every 20 SCS iterations
        scs_snapshot_dir=scs_snap_dir,  # set to None to skip images
        scs_use_indirect=True,
        scs_eps=3e-3,
        scs_alpha=1.5,
        scs_acceleration=1,
        scs_lookback=10,
        scs_scale=1e-1,
        scs_save_every_chunks=1
    )

    # Save final results as you prefer
    np.save(os.path.join(out_root, strategy, "final_x_list.npy"), np.stack(x_list, axis=0))
    with open(os.path.join(out_root, strategy, "history.pkl"), "wb") as f:
        pickle.dump(hist, f)

if __name__ == "__main__":
    main()
