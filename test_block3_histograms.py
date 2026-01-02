# test_block3_histograms.py
# Purpose: build per pixel masks using Block 3, then compute and plot
#          1 counts of active pixel edges per node pair across all pixels
#          2 sums of unmasked weights across active pixels per node pair
#          It also prints degree summaries based on both views
#          All console output is logged to a text file

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
from datetime import datetime

from block_3_graph_and_precisions import build_pixel_connected_Q_provider

class Tee:
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

def histograms_for_pair_counts_and_weights(keep, q_cache_unmasked, out_dir, title_tag, show_plots):
    """
    Build and plot
      counts per pair across pixels
      sums of unmasked weights across active pixels per pair
    Return the count matrix, the weight sum matrix, and degree summaries
    """
    os.makedirs(out_dir, exist_ok=True)
    N, _, n = keep.shape

    count_mat = np.zeros((N, N), dtype=int)
    weight_sum_mat = np.zeros((N, N), dtype=float)

    for i in range(N):
        for j in range(i + 1, N):
            m = keep[i, j, :]
            count = int(np.sum(m))
            wsum = float(np.sum(q_cache_unmasked[(i, j)][m]))
            count_mat[i, j] = count
            count_mat[j, i] = count
            weight_sum_mat[i, j] = wsum
            weight_sum_mat[j, i] = wsum

    degree_count = np.sum(count_mat, axis=1)
    degree_weight = np.sum(weight_sum_mat, axis=1)

    # Print summaries
    print("\nPair count matrix, upper triangle shown by symmetry")
    print(count_mat)
    print("\nPair weight sum matrix, upper triangle shown by symmetry")
    print(weight_sum_mat)
    print("\nDegree from counts per node")
    print(degree_count)
    print("\nDegree from weights per node")
    print(degree_weight)

    # Histograms over upper triangle
    upper_idx = np.triu_indices(N, k=1)
    counts_vec = count_mat[upper_idx]
    weights_vec = weight_sum_mat[upper_idx]

    plt.figure(figsize=(6, 4))
    plt.hist(counts_vec, bins=20)
    plt.xlabel("active pixel edges per pair")
    plt.ylabel("pair frequency")
    plt.title(f"Histogram of pair counts {title_tag}")
    plt.tight_layout()
    fp1 = os.path.join(out_dir, f"hist_pair_counts_{title_tag}.png")
    plt.savefig(fp1, dpi=200)
    if show_plots:
        plt.show()
    else:
        plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(weights_vec, bins=20)
    plt.xlabel("sum of weights per pair across active pixels")
    plt.ylabel("pair frequency")
    plt.title(f"Histogram of pair weight sums {title_tag}")
    plt.tight_layout()
    fp2 = os.path.join(out_dir, f"hist_pair_weights_{title_tag}.png")
    plt.savefig(fp2, dpi=200)
    if show_plots:
        plt.show()
    else:
        plt.close()

    # Also print a few top pairs by count and by weight
    pairs = list(zip(upper_idx[0], upper_idx[1]))
    top_by_count = sorted(pairs, key=lambda ij: count_mat[ij[0], ij[1]], reverse=True)[:10]
    top_by_weight = sorted(pairs, key=lambda ij: weight_sum_mat[ij[0], ij[1]], reverse=True)[:10]

    print("\nTop pairs by count")
    for i, j in top_by_count:
        print(f"({i},{j}) count={count_mat[i, j]}  weight_sum={weight_sum_mat[i, j]:.6e}")

    print("\nTop pairs by weight sum")
    for i, j in top_by_weight:
        print(f"({i},{j}) weight_sum={weight_sum_mat[i, j]:.6e}  count={count_mat[i, j]}")

    return count_mat, weight_sum_mat, degree_count, degree_weight

def run_histograms(strategy="mst", k=2, base_dir="saved_operators_Incmp_Span",
                   out_dir="hist_outputs", show_plots=True):
    """
    Build pixel connected graphs using Block 3 for the chosen strategy.
    Compute and plot histograms and print degree summaries.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Log all prints
    from datetime import datetime
    tag = f"{strategy}_k{k}" if strategy == "knn" else strategy
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(out_dir, f"hist_{tag}_{timestamp}.txt")
    with open(log_path, "w", encoding="utf-8") as logf:
        tee = Tee(sys.stdout, logf)
        with redirect_stdout(tee):
            print(f"Strategy {strategy}")
            if strategy == "knn":
                print(f"k = {k}")

            # Build masks and provider
            G_union, Wi_list, Qij_diag_fn_masked, keep = build_pixel_connected_Q_provider(
                base_dir=base_dir,
                strategy=strategy,
                k=k,
                verbose=True,
                plot_union=False,
                show_plots=False,
                output_dir=os.path.join(out_dir, f"union_{strategy}")
            )

            # Unmasked Q cache from harmonic mean
            q_cache_unmasked = harmonic_Q_unmasked(Wi_list)

            # Histograms and summaries
            counts, weights, degC, degW = histograms_for_pair_counts_and_weights(
                keep, q_cache_unmasked, out_dir=out_dir, title_tag=tag, show_plots=show_plots
            )

            print(f"\nSaved histogram figures and log to {out_dir}")
    print(f"Log written to {log_path}")

def main():
    out_dir = "hist_outputs"
    show_plots = True
    base_dir = "saved_operators_Incmp_Span"
    os.makedirs(out_dir, exist_ok=True)

    # Run each strategy
    run_histograms(strategy="mst", k=2, base_dir=base_dir, out_dir=out_dir, show_plots=show_plots)
    run_histograms(strategy="chain", k=2, base_dir=base_dir, out_dir=out_dir, show_plots=show_plots)
    run_histograms(strategy="knn", k=2, base_dir=base_dir, out_dir=out_dir, show_plots=show_plots)

if __name__ == "__main__":
    main()
