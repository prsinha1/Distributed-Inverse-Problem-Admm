import os
import numpy as np
import networkx as nx

from block_3_graph_and_precisions import build_pixel_connected_Q_provider

#Guarantees that every pixel graph is connected.

#Confirms symmetry of the mask.

#Confirms edge counts per pixel match the strategy, tree and chain have exactly N minus 1 edges per pixel.

#Saves the union graph pictures to pixel_graphs_<strategy> for a quick sanity glance.

def assert_per_pixel_connected(keep):
    """
    keep has shape (N_nodes, N_nodes, n_pixels)
    Check that for every pixel the induced graph is connected
    """
    N, _, n = keep.shape
    for p in range(n):
        Gp = nx.Graph()
        Gp.add_nodes_from(range(N))
        for i in range(N):
            for j in range(i + 1, N):
                if keep[i, j, p]:
                    Gp.add_edge(i, j)
        if not nx.is_connected(Gp):
            raise AssertionError(f"Pixel {p} graph is not connected")

def check_symmetry(keep):
    if not np.array_equal(keep, np.transpose(keep, (1, 0, 2))):
        raise AssertionError("keep mask is not symmetric in i and j")

def check_edge_counts(keep, strategy, k):
    """
    Sanity on number of edges per pixel
    MST and chain must be exactly N-1 edges per pixel
    kNN will be at least N-1 per pixel, at most N*(min(k, N-1)) in the directed view,
    after symmetrization the undirected edge count per pixel is between N-1 and N*(N-1)/2
    """
    N, _, n = keep.shape
    m_per_pixel = []
    for p in range(n):
        m = 0
        for i in range(N):
            for j in range(i + 1, N):
                if keep[i, j, p]:
                    m += 1
        m_per_pixel.append(m)

    m_per_pixel = np.asarray(m_per_pixel)
    if strategy == "mst" or strategy == "chain":
        if not np.all(m_per_pixel == N - 1):
            bad = np.where(m_per_pixel != N - 1)[0][:10]
            raise AssertionError(f"Expected exactly N-1 edges per pixel, mismatches at pixels {bad}")
    elif strategy == "knn":
        if not np.all(m_per_pixel >= N - 1):
            bad = np.where(m_per_pixel < N - 1)[0][:10]
            raise AssertionError(f"kNN must be connected per pixel, found fewer than N-1 edges at pixels {bad}")

def run_structural_suite(strategy, k=2, base_dir="saved_operators_Incmp_Span"):
    print(f"--- Structural test, strategy={strategy}, k={k} ---")
    G_union, Wi_list, Qij_diag_fn, keep = build_pixel_connected_Q_provider(
        base_dir=base_dir,
        strategy=strategy,
        k=k,
        verbose=True,
        plot_union=True,      # saves union pictures for a quick glance
        show_plots=False,
        output_dir=f"pixel_graphs_{strategy}"
    )


    check_symmetry(keep)
    assert_per_pixel_connected(keep)
    check_edge_counts(keep, strategy, k)

    # Quick summary
    N = len(Wi_list)
    n = Wi_list[0].shape[0]
    active_ratio = keep.sum() / keep.size
    print(f"nodes {N}, pixels {n}, active ratio {active_ratio:.4f}")
    print("Structural suite passed")

if __name__ == "__main__":
    for strat in ["mst", "chain", "knn"]:
        run_structural_suite(strat, k=2)
