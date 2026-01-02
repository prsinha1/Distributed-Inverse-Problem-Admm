# test_block3_pixel_graphs.py
# For a chosen pixel p, print Wi[p], Q_unmasked[p], Q_masked[p], keep[p],
# draw and save the pixel graph G(p) for MST, chain, and kNN.
# All console output is also saved to a text log file.

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from contextlib import redirect_stdout
from datetime import datetime

from block_3_graph_and_precisions import build_pixel_connected_Q_provider


class Tee:
    """Write to terminal and to a file-like object at the same time."""
    def __init__(self, stream_a, stream_b):
        self.a = stream_a
        self.b = stream_b
    def write(self, data):
        self.a.write(data)
        self.b.write(data)
    def flush(self):
        self.a.flush()
        self.b.flush()


def harmonic_Q_unmasked(Wi_list):
    """Return dict (i, j) -> q_ij vector, unmasked harmonic mean."""
    N = len(Wi_list)
    q_cache = {}
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            eta_i = Wi_list[i]
            eta_j = Wi_list[j]
            q = (eta_i * eta_j) / (eta_i + eta_j)
            q_cache[(i, j)] = q
    return q_cache


def masked_Q_matrix_at_pixel(Qij_provider_masked, p, N):
    """Build N by N matrix of masked Q at pixel p, diagonal set to zero."""
    M = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            v = Qij_provider_masked(i, j)
            M[i, j] = float(v[p])
    return M


def unmasked_Q_matrix_at_pixel(q_cache_unmasked, p, N):
    """Build N by N matrix of unmasked Q at pixel p, diagonal set to zero."""
    M = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            M[i, j] = float(q_cache_unmasked[(i, j)][p])
    return M


def keep_matrix_at_pixel(keep, p):
    """Return N by N integer matrix of keep at pixel p."""
    N = keep.shape[0]
    K = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(N):
            if i != j:
                K[i, j] = int(bool(keep[i, j, p]))
    return K


def build_pixel_graph_from_keep(keep, q_cache_unmasked, p):
    """Construct G(p) from keep[:,:,p], with edge attribute w equal to unmasked Q_ij[p]."""
    N = keep.shape[0]
    Gp = nx.Graph()
    Gp.add_nodes_from(range(N))
    for i in range(N):
        for j in range(i + 1, N):
            if keep[i, j, p]:
                w = float(q_cache_unmasked[(i, j)][p])
                Gp.add_edge(i, j, w=w)
    return Gp


def print_matrix(label, M, fmt="{: .6e}"):
    print(f"\n{label}")
    for r in range(M.shape[0]):
        print(" ".join(fmt.format(M[r, c]) for c in range(M.shape[1])))


def print_graph_summary(Gp, title):
    print(f"\n--- {title} ---")
    print(f"nodes {Gp.number_of_nodes()}, edges {Gp.number_of_edges()}")
    degs = dict(Gp.degree())
    print("degrees per node:", [int(degs[v]) for v in Gp.nodes()])
    print("edge list with weights Q_ij[p]:")
    for u, v, d in Gp.edges(data=True):
        print(f"({u},{v})  w={d.get('w', 0.0):.6e}")


def draw_pixel_graph(Gp, out_png, show=True):
    if Gp.number_of_nodes() == 0:
        print("empty graph, nothing to draw")
        return
    pos = nx.spring_layout(Gp, seed=42)
    plt.figure(figsize=(6, 6))
    nx.draw_networkx_nodes(Gp, pos, node_size=600)
    nx.draw_networkx_labels(Gp, pos, font_size=10)
    nx.draw_networkx_edges(Gp, pos, width=2.0)
    edge_labels = {(u, v): f"{d.get('w', 0.0):.2e}" for u, v, d in Gp.edges(data=True)}
    nx.draw_networkx_edge_labels(Gp, pos, edge_labels=edge_labels, font_size=8)
    plt.title(os.path.basename(out_png).replace("_", " "))
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    if show:
        plt.show()
    else:
        plt.close()


def run_for_strategy(strategy, k, base_dir, p_view, out_dir, show_plots, log_file_handle):
    """Build keep and Q for a strategy. Print Wi[p], Q matrices, keep, graph summary. Save figure."""
    # Route all prints to both terminal and log file during this block
    tee = Tee(sys.stdout, log_file_handle)
    with redirect_stdout(tee):
        print(f"\n========== Strategy {strategy} ==========")
        if strategy == "knn":
            print(f"k = {k}")

        G_union, Wi_list, Qij_diag_fn_masked, keep = build_pixel_connected_Q_provider(
            base_dir=base_dir,
            strategy=strategy,
            k=k,
            verbose=True,
            plot_union=False,
            show_plots=False,
            output_dir=os.path.join(out_dir, f"union_{strategy}")
        )

        N = len(Wi_list)
        # Unmasked Q cache and matrices at pixel p
        q_cache_unmasked = harmonic_Q_unmasked(Wi_list)
        Q_unmasked_p = unmasked_Q_matrix_at_pixel(q_cache_unmasked, p_view, N)
        Q_masked_p = masked_Q_matrix_at_pixel(Qij_diag_fn_masked, p_view, N)
        K_p = keep_matrix_at_pixel(keep, p_view)

        # Print Wi[p] for all i
        print(f"\n--- Pixel p = {p_view} diagnostics ---")
        for i in range(N):
            print(f"W[{i}][p] = {Wi_list[i][p_view]:.6e}")

        # Print matrices
        print_matrix("Unmasked Q_ij[p] harmonic mean:", Q_unmasked_p)
        print_matrix("Masked Q_ij[p] after pruning:", Q_masked_p)
        print_matrix("keep[i,j,p] mask:", K_p, fmt="{:d}")

        # Build and summarize G(p)
        Gp = build_pixel_graph_from_keep(keep, q_cache_unmasked, p_view)
        tag = f"{strategy}_k{k}" if strategy == "knn" else strategy
        title = f"pixel p={p_view}, strategy={tag}"
        print_graph_summary(Gp, title)

    # Save figure outside redirect so figure logs are already printed
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{strategy}_k{k}" if strategy == "knn" else strategy
    out_png = os.path.join(out_dir, f"pixel_graph_{tag}_p{p_view}.png")
    draw_pixel_graph(Gp, out_png, show=show_plots)

    return Gp


def main():
    base_dir = "saved_operators_Incmp_Span"
    out_dir = "pixel_graph_figs"
    p_view = 150
    k = 2
    show_plots = True

    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(out_dir, f"pixel_graph_log_p{p_view}_{timestamp}.txt")

    print(f"Saving console output to {log_path}")

    with open(log_path, "w", encoding="utf-8") as logf:
        # MST
        run_for_strategy("mst", k, base_dir, p_view, out_dir, show_plots, logf)
        # Chain
        run_for_strategy("chain", k, base_dir, p_view, out_dir, show_plots, logf)
        # kNN
        run_for_strategy("knn", k, base_dir, p_view, out_dir, show_plots, logf)

    print("Done.")


if __name__ == "__main__":
    main()
