# block_2_load_odl_data.py  (revised to also SHOW plots and add aggregate reconstruction)
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib
matplotlib.use('TkAgg')
from Gen_Sino_Partitioned import randIm, ConstIm

# Added: ODL only for optional FBP reconstruction
import odl


def load_odl_data(base_dir="saved_operators_Incmp_Span",
                  N=64,
                  num_nodes=5,
                  noise_level=0.005,
                  phantom_array=None,
                  ray_transforms_pickle="ray_transforms.pkl",
                  A_dense_list_pickle="A_dense_list.pkl",
                  agg_op_pickle="aggregate_op.pkl",
                  A_agg_pickle="A_agg.pkl",
                  make_plots=True,
                  show_plots=True,
                  output_dir=None):
    # Load operators and matrices
    with open(os.path.join(base_dir, ray_transforms_pickle), "rb") as f:
        ray_transforms = pickle.load(f)
    with open(os.path.join(base_dir, A_dense_list_pickle), "rb") as f:
        A_dense_list = pickle.load(f)

    agg_ray_trafo = None
    A_agg = None
    agg_op_path = os.path.join(base_dir, agg_op_pickle)
    A_agg_path = os.path.join(base_dir, A_agg_pickle)
    if os.path.exists(agg_op_path):
        with open(agg_op_path, "rb") as f:
            agg_ray_trafo = pickle.load(f)
    if os.path.exists(A_agg_path):
        with open(A_agg_path, "rb") as f:
            A_agg = pickle.load(f)

    assert len(ray_transforms) == num_nodes
    assert len(A_dense_list) == num_nodes

    space = ray_transforms[0].domain
    if phantom_array is None:
        #phantom_array = ConstIm(N)
        phantom_array = randIm(N)
    phantom = space.element(phantom_array)

    # Per node sinograms
    sinograms_odl = [
        op(phantom) + noise_level * op.range.element(
            np.random.normal(0.0, 1.0, size=op.range.shape)
        )
        for op in ray_transforms
    ]
    sinograms = [s.asarray() for s in sinograms_odl]

    column_norms_all = [np.linalg.norm(A_i, axis=0) for A_i in A_dense_list]

    # Aggregate sinogram
    agg_sinogram_stack = np.vstack(sinograms)
    agg_sinogram = agg_sinogram_stack
    if A_agg is not None:
        x_true_vec = phantom.asarray().reshape(-1, 1)
        agg_vec = A_agg @ x_true_vec
        total_rows = sum(rt.range.shape[0] for rt in ray_transforms)
        det_cols = ray_transforms[0].range.shape[1]
        agg_from_A = agg_vec.reshape(total_rows, det_cols)
        agg_from_A = agg_from_A + noise_level * np.random.normal(0.0, 1.0, size=agg_from_A.shape)
        agg_sinogram = agg_from_A

    # -------- Aggregate reconstruction test, minimal additions start here --------
    agg_fbp_recon = None
    agg_ls_recon = None


    # small ridge least squares if only A_agg is present

    n = N * N
    lam_ridge = 1e-3
    ATA = A_agg.T @ A_agg
    ATb = A_agg.T @ agg_sinogram.reshape(-1, 1)
    x_vec = np.linalg.solve(ATA + lam_ridge * np.eye(n), ATb)
    agg_ls_recon = x_vec.reshape(N, N)

    # -------- Aggregate reconstruction test, minimal additions end here --------

    # Plotting
    if make_plots:
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"Recon_Op_ADMM_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)

        # True phantom
        fig1 = plt.figure(figsize=(4, 4))
        plt.imshow(phantom.asarray(), cmap='gray')
        plt.title("True Phantom")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "true_phantom.png"), dpi=300)
        if show_plots:
            plt.show()
        plt.close(fig1) if not show_plots else None

        # Phantom and sinograms
        fig3, axes = plt.subplots(1, num_nodes + 2, figsize=(4 * (num_nodes + 2), 4))

        axes[0].imshow(phantom.asarray(), cmap='gray')
        axes[0].set_title("True Phantom")
        axes[0].axis('off')

        for i in range(num_nodes):
            axes[i + 1].imshow(sinograms_odl[i].asarray(), cmap='gray', aspect='auto', origin='upper')
            axes[i + 1].set_title(f"Node {i + 1} Sinogram")
            axes[i + 1].set_xlabel("Detector")
            axes[i + 1].set_ylabel("Angle")

        axes[-1].imshow(agg_sinogram, cmap='gray', aspect='auto', origin='upper')
        axes[-1].set_title("Aggregate A*x_true")
        axes[-1].set_xlabel("Detector")
        axes[-1].set_ylabel("Angle")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "plot3_clean_true_sinograms.png"), dpi=300)
        if show_plots:
            plt.show()
        plt.close(fig3) if not show_plots else None

        # Added: save and show reconstruction figure if available
        if agg_fbp_recon is not None or agg_ls_recon is not None:
            figR = plt.figure(figsize=(4, 4))
            if agg_fbp_recon is not None:
                plt.imshow(agg_fbp_recon, cmap='gray')
                plt.title("Aggregate FBP Reconstruction")
                out_name = "agg_fbp_recon.png"
            else:
                plt.imshow(agg_ls_recon, cmap='gray')
                plt.title("Aggregate LS Reconstruction")
                out_name = "agg_ls_recon.png"
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, out_name), dpi=300)
            #if show_plots:
            #    plt.show()
            #plt.close(figR) if not show_plots else None
            plt.show() if show_plots else plt.close(figR)


    return {
        "A_dense_list": A_dense_list,
        "sinograms": sinograms,
        "column_norms_all": column_norms_all,
        "N": N,
        "num_nodes": num_nodes,
        "agg_ray_trafo": agg_ray_trafo,
        "A_agg": A_agg,
        "agg_sinogram": agg_sinogram,
        # Added: return recon arrays for quick inspection
        "agg_fbp_recon": agg_fbp_recon,
        "agg_ls_recon": agg_ls_recon,
        "output_dir": output_dir,
    }
