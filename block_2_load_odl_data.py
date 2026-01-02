# block_2_load_odl_data.py

import os
import pickle
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import odl

# These helpers are already used elsewhere in your project
from block_1_env_and_imports import ConstIm, randIm


def _build_parallel_beam_operators(N, num_nodes, angles_total=None, det_width_factor=1.0):
    """
    Create a list of per node ray transforms and an aggregate transform.
    The construction mirrors standard ODL parallel beam geometry.
    """

    # Discrete image space
    space = odl.uniform_discr(
        min_pt=[-1.0, -1.0],
        max_pt=[1.0, 1.0],
        shape=[N, N],
        dtype="float32"
    )

    # Total number of angles, then split evenly across nodes
    if angles_total is None:
        # Heuristic choice
        angles_total = max(180, 3 * N)

    # Equal split of angles across nodes
    angles_per_node = [angles_total // num_nodes] * num_nodes
    for i in range(angles_total % num_nodes):
        angles_per_node[i] += 1

    # Detector
    # Width scaled so that the detector covers the object
    det_width = det_width_factor * 2.0
    # Number of detector pixels
    det_pixels = N

    # Build per node transforms
    ray_transforms = []
    start = 0
    for k in range(num_nodes):
        m_k = angles_per_node[k]
        part_angles = odl.uniform_partition(0.0, np.pi, m_k)
        part_det = odl.uniform_partition(-det_width / 2.0, det_width / 2.0, det_pixels)
        geom = odl.tomo.Parallel2dGeometry(part_angles, part_det)
        op_k = odl.tomo.RayTransform(space, geom)
        ray_transforms.append(op_k)
        start += m_k

    # Aggregate transform, simply stack per node transforms
    agg_geom = odl.tomo.Parallel2dGeometry(
        odl.uniform_partition(0.0, np.pi, angles_total),
        odl.uniform_partition(-det_width / 2.0, det_width / 2.0, det_pixels),
    )
    agg_ray_trafo = odl.tomo.RayTransform(space, agg_geom)

    return space, ray_transforms, agg_ray_trafo


def _to_dense_matrix(trafo):
    """
    Turn an ODL operator into a dense matrix.
    This can be large, but your project already uses dense lists elsewhere.
    """
    # Shape is (m, n) with n = N*N
    # Build dense representation by applying trafo to canonical basis blocks
    # ODL provides utilities, but we do a safe fallback here
    dom_shape = trafo.domain.shape
    n = dom_shape[0] * dom_shape[1]

    # Range array shape m x d, flatten per measurement row afterwards
    # Easier path is to use matrix_representation if available
    try:
        from odl.operator.oputils import matrix_representation
        M = matrix_representation(trafo)
        return np.array(M, dtype=np.float32)
    except Exception:
        # Fallback, slow but robust for moderate N
        e = trafo.domain.one()
        out = []
        for j in range(n):
            e.set_zero()
            # set j-th pixel to one
            e.asarray().flat[j] = 1.0
            y = trafo(e).asarray()
            out.append(y.reshape(-1))
        M = np.stack(out, axis=1)  # shape m by n
        return M.astype(np.float32)


def load_odl_data(
    N=128,
    num_nodes=5,
    noise_level=0.005,
    output_dir=None,
    make_plots=True,
    show_plots=False,
    phantom_array=None,
    save_operators_dir="saved_operators_Incmp_Span",
    build_dense=True,
):
    """
    Build per node operators and data for tomography experiments.

    Returns a dict with keys used by downstream blocks:
      - A_dense_list: list of dense A_i matrices
      - sinograms: list of per node sinograms as numpy arrays
      - column_norms_all: list of per node column norms of A_i
      - N: image side length
      - num_nodes: number of nodes
      - agg_ray_trafo: aggregate ODL ray transform
      - A_agg: dense aggregate matrix if build_dense is True, else None
      - agg_sinogram: aggregate sinogram array
      - output_dir: output directory for figures
      - phantom: legacy single phantom array for compatibility, node 0
      - phantoms: list of per node phantom arrays
    """

    # Build operators
    space, ray_transforms, agg_ray_trafo = _build_parallel_beam_operators(
        N=N, num_nodes=num_nodes
    )

    # Per node phantoms
    phantoms_array_list = []
    if phantom_array is None:
        # Different phantom per node, seeded for reproducibility
        for i in range(num_nodes):
            phantoms_array_list.append(randIm(N, seed=i))
    else:
        if isinstance(phantom_array, list):
            assert len(phantom_array) == num_nodes, "phantom_array list must have length num_nodes"
            phantoms_array_list = phantom_array
        else:
            phantoms_array_list = [phantom_array for _ in range(num_nodes)]

    phantoms_odl = [space.element(arr) for arr in phantoms_array_list]

    # Per node sinograms
    sinograms_odl = [
        op(phantoms_odl[i]) + noise_level * op.range.element(
            np.random.normal(0.0, 1.0, size=op.range.shape)
        )
        for i, op in enumerate(ray_transforms)
    ]
    sinograms = [s.asarray() for s in sinograms_odl]

    # Aggregate sinogram from aggregate operator and a reference phantom
    # The reference phantom is chosen as node 0
    A_agg = None
    agg_sinogram = np.vstack(sinograms)
    if build_dense:
        # Build dense per node matrices and the aggregate
        A_dense_list = []
        for i, op in enumerate(ray_transforms):
            Ai = _to_dense_matrix(op)
            A_dense_list.append(Ai)

        A_agg = _to_dense_matrix(agg_ray_trafo)

        # Build aggregate sinogram directly from A_agg and phantom 0
        x_true_vec = phantoms_odl[0].asarray().reshape(-1, 1)
        agg_vec = A_agg @ x_true_vec
        # Reshape to sinogram image
        total_rows = sum(rt.range.shape[0] for rt in ray_transforms)
        det_cols = ray_transforms[0].range.shape[1]
        agg_from_A = agg_vec.reshape(total_rows, det_cols)
        agg_from_A = agg_from_A + noise_level * np.random.normal(0.0, 1.0, size=agg_from_A.shape)
        agg_sinogram = agg_from_A
    else:
        A_dense_list = [None for _ in range(num_nodes)]

    # Column norms for preconditioning
    column_norms_all = []
    for i, Ai in enumerate(A_dense_list):
        if Ai is None:
            column_norms_all.append(None)
        else:
            # l2 norm per column
            cn = np.linalg.norm(Ai, axis=0)
            column_norms_all.append(cn)

    # Output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"Recon_Op_ADMM_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Save operators if requested
    if build_dense:
        os.makedirs(save_operators_dir, exist_ok=True)
        with open(os.path.join(save_operators_dir, "A_dense_list.pkl"), "wb") as f:
            pickle.dump(A_dense_list, f)

    # Plots
    if make_plots:
        # Per node phantoms
        fig1, axes1 = plt.subplots(1, num_nodes, figsize=(4 * num_nodes, 4))
        if num_nodes == 1:
            axes = [axes1]
        for i in range(num_nodes):
            axes1[i].imshow(phantoms_odl[i].asarray(), cmap="gray")
            axes1[i].set_title(f"Node {i + 1} Phantom")
            axes1[i].axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "node_phantoms.png"), dpi=300)
        if show_plots:
            plt.show()
        else:
            plt.close(fig1)

        # Sinograms per node and aggregate
        fig2, axes2 = plt.subplots(1, num_nodes + 1, figsize=(4 * (num_nodes + 1), 4))
        for i in range(num_nodes):
            axes2[i].imshow(sinograms[i], cmap="gray", aspect="auto", origin="upper")
            axes2[i].set_title(f"Sino {i + 1}")
            axes2[i].set_xlabel("Detector")
            axes2[i].set_ylabel("Angle rows")
        axes2[-1].imshow(agg_sinogram, cmap="gray", aspect="auto", origin="upper")
        axes2[-1].set_title("Agg A x")
        axes2[-1].set_xlabel("Detector")
        axes2[-1].set_ylabel("Angle rows")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "sinograms_and_agg.png"), dpi=300)
        if show_plots:
            plt.show()
        else:
            plt.close(fig2)

    # Return structure used by downstream blocks
    return {
        "A_dense_list": A_dense_list,
        "sinograms": sinograms,
        "column_norms_all": column_norms_all,
        "N": N,
        "num_nodes": num_nodes,
        "agg_ray_trafo": agg_ray_trafo,
        "A_agg": A_agg,
        "agg_sinogram": agg_sinogram,
        "output_dir": output_dir,
        # Legacy single phantom for compatibility, choose node 0
        "phantom": phantoms_odl[0].asarray(),
        # New list of per node phantoms
        "phantoms": [p.asarray() for p in phantoms_odl],
    }


# Backward compatible aliases if an older block expects a different name
load_data = load_odl_data
prepare_data = load_odl_data

if __name__ == "__main__":
    # Quick smoke test
    data = load_odl_data(N=64, num_nodes=5, noise_level=0.005, make_plots=True, show_plots=False)
    print("[Block2] Built data with keys:", list(data.keys()))
    print("[Block2] A_dense_list length:", len(data["A_dense_list"]))
    print("[Block2] Sinogram shapes:", [s.shape for s in data["sinograms"]]) 