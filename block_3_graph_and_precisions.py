# block_3_graph_and_precisions.py
import os
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Literal, Tuple


# 1. Compute Wi and define  harmonic/arithmetic mean Qij
def make_precisions(A_dense_list, q_mode: Literal["harmonic", "arithmetic"] = "arithmetic"):
    """
    Wi[p] = ||A_i[:, p]||_2^2
    If q_mode == "harmonic":
        Qij[p] = (Wi[p] * Wj[p]) / (Wi[p] + Wj[p])   # H.M.
    If q_mode == "arithmetic":
        Qij[p] = 0.5 * (Wi[p] + Wj[p])               # A.M.
    """
    eps = 1e-12
    Wi_list = []
    for A_i in A_dense_list:
        eta = np.sum(A_i * A_i, axis=0)
        eta = np.maximum(eta, eps)
        Wi_list.append(eta)

    if q_mode == "harmonic":
        def Qij_diag(i, j):
            eta_i = Wi_list[i]
            eta_j = Wi_list[j]
            q = (eta_i * eta_j) / (eta_i + eta_j)
            q = np.maximum(q, eps)
            return q
    elif q_mode == "arithmetic":
        def Qij_diag(i, j):
            eta_i = Wi_list[i]
            eta_j = Wi_list[j]
            q = 0.5 * (eta_i + eta_j)
            q = np.maximum(q, eps)
            return q
    else:
        raise ValueError("q_mode must be 'harmonic' or 'arithmetic'")

    return Wi_list, Qij_diag



# 2. Precompute q_ij vectors for all ordered pairs
def _precompute_q_cache(num_nodes, Qij_diag):
    """
    Returns dict[(i, j)] = q_ij, shape (n,)
    """
    q_cache = {}
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            q_cache[(i, j)] = Qij_diag(i, j)
    return q_cache


# 3a. Per pixel kNN mask, then connect components minimally using maximum weight edges
def _pixel_mask_knn_then_connect(q_col, k, num_nodes):
    """
    q_col is a matrix of shape (num_nodes, num_nodes) filled with weights at a single pixel p
    Diagonal entries are zero
    Step 1 select k largest neighbors per node, symmetrize
    Step 2 if not connected, add edges from maximum spanning tree until connected
    Returns adjacency mask of shape (num_nodes, num_nodes) with ones on active edges
    """
    nV = num_nodes
    # Initial kNN selection per node
    adj = np.zeros((nV, nV), dtype=bool)
    for i in range(nV):
        cand = q_col[i, :].copy()
        cand[i] = -np.inf
        k_eff = min(k, nV - 1)
        if k_eff > 0:
            idx = np.argpartition(cand, -k_eff)[-k_eff:]
            adj[i, idx] = True
    # Symmetrize
    adj = np.logical_or(adj, adj.T)

    # Build graph and ensure connectivity using maximum spanning tree edges
    G = nx.Graph()
    G.add_nodes_from(range(nV))
    # add selected edges
    es = np.argwhere(adj)
    for i, j in es:
        if i < j:
            G.add_edge(i, j, weight=float(q_col[i, j]))

    if not nx.is_connected(G):
        # Full complete graph for this pixel with weights
        G_full = nx.Graph()
        G_full.add_nodes_from(range(nV))
        for i in range(nV):
            for j in range(i + 1, nV):
                G_full.add_edge(i, j, weight=float(q_col[i, j]))
        # Maximum spanning tree
        T = nx.maximum_spanning_tree(G_full, weight="weight")
        # Add missing tree edges
        for u, v, d in T.edges(data=True):
            G.add_edge(u, v, weight=d["weight"])

    # Convert back to mask
    out = np.zeros((nV, nV), dtype=bool)
    for u, v in G.edges():
        out[u, v] = True
        out[v, u] = True
    return out


# 3b. Per pixel maximum spanning tree mask
def _pixel_mask_mst(q_col, num_nodes):
    """
    Use maximum spanning tree on the complete graph at this pixel
    Returns adjacency mask of shape (num_nodes, num_nodes)
    """
    nV = num_nodes
    G_full = nx.Graph()
    G_full.add_nodes_from(range(nV))
    for i in range(nV):
        for j in range(i + 1, nV):
            G_full.add_edge(i, j, weight=float(q_col[i, j]))
    T = nx.maximum_spanning_tree(G_full, weight="weight")
    out = np.zeros((nV, nV), dtype=bool)
    for u, v in T.edges():
        out[u, v] = True
        out[v, u] = True
    return out


# 3c. Per pixel random chain mask
def _pixel_mask_chain(num_nodes, rng):
    """
    Random permutation chain for this pixel
    Always connected and very sparse
    """
    order = rng.permutation(num_nodes)
    out = np.zeros((num_nodes, num_nodes), dtype=bool)
    for t in range(num_nodes - 1):
        u = order[t]
        v = order[t + 1]
        out[u, v] = True
        out[v, u] = True
    return out


# 4. Build all pixel masks according to a strategy
def _build_all_pixel_masks(q_cache, num_nodes, n, strategy, k, seed):
    """
    Returns keep[i, j, p] boolean tensor
    """
    rng = np.random.default_rng(seed)
    keep = np.zeros((num_nodes, num_nodes, n), dtype=bool)

    # For fast access, assemble a 3D tensor q[i, j, p]
    # Diagonal entries remain zero
    q = np.zeros((num_nodes, num_nodes, n), dtype=float)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            q[i, j, :] = q_cache[(i, j)]

    # Process each pixel p
    for p in range(n):
        q_col = q[:, :, p]
        # Force exact symmetry on the weights for safety
        # Use average of both directions if any floating drift exists
        q_sym = 0.5 * (q_col + q_col.T)
        np.fill_diagonal(q_sym, 0.0)

        if strategy == "knn":
            adj = _pixel_mask_knn_then_connect(q_sym, k=k, num_nodes=num_nodes)
        elif strategy == "mst":
            adj = _pixel_mask_mst(q_sym, num_nodes=num_nodes)
        elif strategy == "chain":
            adj = _pixel_mask_chain(num_nodes=num_nodes, rng=rng)
        else:
            raise ValueError("strategy must be one of 'knn', 'mst', or 'chain'")

        keep[:, :, p] = adj

    # Symmetrize once more across i and j over all pixels
    keep = np.logical_or(keep, np.transpose(keep, (1, 0, 2)))
    return keep


# 5. Build a union node graph for quick visualization and print diagnostics
def _summarize_and_plot_union(keep, output_dir, show_plots, verbose, title_suffix):
    """
    Collapse per pixel masks into a single node graph by union over pixels
    Save a picture and a degree histogram
    Print simple stats
    """
    os.makedirs(output_dir, exist_ok=True)
    num_nodes, _, n = keep.shape

    # Union graph
    G_union = nx.Graph()
    G_union.add_nodes_from(range(num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.any(keep[i, j, :]):
                G_union.add_edge(i, j)

    degrees = np.array([deg for _, deg in G_union.degree()])
    active_ratio = keep.sum() / keep.size if keep.size > 0 else 0.0

    if verbose:
        print(f"[Block3] strategy {title_suffix}")
        print(f"[Block3] nodes {G_union.number_of_nodes()}, edges {G_union.number_of_edges()}")
        print(f"[Block3] connected {nx.is_connected(G_union)}")
        if degrees.size > 0:
            print(f"[Block3] degree min mean max {degrees.min()}  {degrees.mean():.2f}  {degrees.max()}")
        print(f"[Block3] active pixel ratio {active_ratio:.4f}")

    # Plot the union graph
    try:
        pos = nx.spring_layout(G_union, seed=42)
        plt.figure(figsize=(6, 6))
        nx.draw_networkx(G_union, pos=pos, with_labels=True, node_size=600, font_size=10)
        gp = os.path.join(output_dir, f"pixel_union_graph_{title_suffix}.png")
        plt.tight_layout()
        plt.savefig(gp, dpi=200)
        if show_plots:
            plt.show()
        else:
            plt.close()
        if verbose:
            print(f"[Block3] saved {gp}")
    except Exception as e:
        print("[Block3] union graph plot failed:", repr(e))

    # Plot a degree histogram
    try:
        if degrees.size == 0:
            return G_union
        plt.figure(figsize=(6, 4))
        bins = range(int(degrees.min()), int(degrees.max()) + 2)
        plt.hist(degrees, bins=bins)
        plt.xlabel("Degree")
        plt.ylabel("Count")
        plt.title(f"Node degree histogram, strategy {title_suffix}")
        dh = os.path.join(output_dir, f"pixel_union_degree_{title_suffix}.png")
        plt.tight_layout()
        plt.savefig(dh, dpi=200)
        if show_plots:
            plt.show()
        else:
            plt.close()
        if verbose:
            print(f"[Block3] saved {dh}")
    except Exception as e:
        print("[Block3] histogram plot failed:", repr(e))

    return G_union


# 6. Main entry. Self contained loader and provider builder
def build_pixel_connected_Q_provider(base_dir="saved_operators_Incmp_Span",
                                     A_dense_list_pickle="A_dense_list.pkl",
                                     strategy: Literal["knn", "mst", "chain"] = "knn",
                                     k: int = 2,
                                     seed: int = 0,
                                     q_mode: Literal["harmonic", "arithmetic"] = "arithmetic",  # NEW
                                     verbose: bool = True,
                                     plot_union: bool = True,
                                     show_plots: bool = True,
                                     output_dir: str = "pixel_graphs_out"):

    """
    Load A_dense_list, compute Wi and q_ij
    Build a connected graph for each pixel using the chosen strategy
      knn   top k per node then minimally connect components at that pixel
      mst   maximum spanning tree at that pixel
      chain random chain at that pixel
    Return a masked Q provider where non selected pixel edges are zeroed out
    Also return Wi_list and a union node graph for visualization convenience
    """
    # Load A_dense_list
    A_path = os.path.join(base_dir, A_dense_list_pickle)
    if not os.path.exists(A_path):
        raise FileNotFoundError(f"A_dense_list pickle not found at {A_path}")
    with open(A_path, "rb") as f:
        A_dense_list = pickle.load(f)
    if verbose:
        print(f"[Block3] loaded A_dense_list from {A_path}  nodes {len(A_dense_list)}")

    # Precisions and base weights
    Wi_list, Qij_diag = make_precisions(A_dense_list, q_mode=q_mode)

    num_nodes = len(Wi_list)
    n = Wi_list[0].shape[0]

    # Precompute all q_ij vectors
    q_cache = _precompute_q_cache(num_nodes, Qij_diag)

    # Build per pixel masks
    keep = _build_all_pixel_masks(q_cache, num_nodes, n, strategy=strategy, k=k, seed=seed)

    # Optional union visualization
    G_union = None
    if plot_union:
        #tag = f"{strategy}_k{k}" if strategy == "knn" else strategy
        tag = f"{strategy}_k{k}_{q_mode}" if strategy == "knn" else f"{strategy}_{q_mode}"

        G_union = _summarize_and_plot_union(keep, output_dir, show_plots, verbose, tag)

    # Masked provider
    def Qij_diag_masked(i, j):
        if i == j:
            return np.zeros(n, dtype=float)
        q = q_cache[(i, j)]
        m = keep[i, j, :]
        return np.where(m, q, 0.0)

    return G_union, Wi_list, Qij_diag_masked, keep
