import numpy as np

def evaluate_strategy(strategy, k, data, N_side, max_iters=200):
    from Tomo_Reconst.ADMM.block_6_admm_loop_ver0 import decentralized_admm
    from block_3_graph_and_precisions import build_pixel_connected_Q_provider

    # Build pixel masks for this strategy
    G_union, Wi_list, Qij_diag_fn, keep = build_pixel_connected_Q_provider(
        base_dir="saved_operators_Incmp_Span",
        strategy=strategy,
        k=k,
        verbose=True,
        plot_union=False,
        show_plots=False
    )

    # Pass mask to ADMM
    x_list, hist = decentralized_admm(
        A_dense_list=data["A_dense_list"],
        sinograms=data["sinograms"],
        G=G_union,
        Wi_list=Wi_list,
        Qij_diag_fn=Qij_diag_fn,
        N=N_side,
        lam_tv=0.02,
        rho=1.0,
        max_iters=max_iters,
        eps_pri=1e-3,
        eps_dual=1e-3,
        verbose=True,
        edge_mask_provider=lambda i, j: keep[i, j, :]
    )
    return x_list, hist

# Example driver
# data = load_odl_data(...), as in your Block 2
# N_side = data["N"]
# for s in ["mst", "chain", "knn"]:
#     _, hist = evaluate_strategy(s, k=2, data=data, N_side=N_side)
#     print(s, "iters", len(hist["primal"]), "final primal", hist["primal"][-1], "final dual", hist["dual"][-1])
def psnr(x_hat, x_true, data_range=1.0):
    mse = np.mean((x_hat - x_true) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(data_range) - 10.0 * np.log10(mse)

# After evaluate_strategy, if you have the true phantom array
# x_true = data["phantom_array"] or phantom.asarray()
# psnrs = [psnr(x_list[i].reshape(N_side, N_side), x_true, data_range=x_true.max() - x_true.min()) for i in range(len(x_list))]
# print("mean PSNR across nodes", np.mean(psnrs))
