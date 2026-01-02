# test_block3_histogram_checks.py
# Build global pair summaries using Block 3, then verify invariants and bounds.
# Also plot per pixel undirected edge counts across all pixels.

import os
import numpy as np
import matplotlib.pyplot as plt

from block_3_graph_and_precisions import build_pixel_connected_Q_provider


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


def build_pair_summaries(keep, q_cache_unmasked):
    """
    From keep[i,j,p] and unmasked q, build:
      count_mat[i,j] = number of pixels where edge i j is active
      weight_sum_mat[i,j] = sum over active pixels of unmasked Q_ij[p]
      degree_count[i] and degree_weight[i] as row sums
    """
    N, _, n = keep.shape
    count_mat = np.zeros((N, N), dtype=int)
    weight_sum_mat = np.zeros((N, N), dtype=float)

    for i in range(N):
        for j in range(i + 1, N):
            m = keep[i, j, :]
            cnt = int(np.sum(m))
            wsum = float(np.sum(q_cache_unmasked[(i, j)][m]))
            count_mat[i, j] = cnt
            count_mat[j, i] = cnt
            weight_sum_mat[i, j] = wsum
            weight_sum_mat[j, i] = wsum

    degree_count = np.sum(count_mat, axis=1)
    degree_weight = np.sum(weight_sum_mat, axis=1)
    return count_mat, weight_sum_mat, degree_count, degree_weight


def check_chain_or_mst(count_mat, N, n, label):
    total_pairs = int(np.sum(np.triu(count_mat, k=1)))
    expected = n * (N - 1)
    ok = (total_pairs == expected)
    print(f"[{label}] total active pixel edges across all pixels = {total_pairs}, expected {expected}, pass = {ok}")
    return ok


def check_knn_bounds(count_mat, N, n, k):
    """
    Correct kNN bounds after symmetrization:
      per pixel edges are at least N-1
      per pixel edges are at most min(N*k, N*(N-1)/2)
    """
    total_pairs = int(np.sum(np.triu(count_mat, k=1)))
    lower = n * (N - 1)
    upper_per_pixel = min(N * k, N * (N - 1) // 2)
    upper = n * upper_per_pixel
    per_pixel_avg = total_pairs / n if n > 0 else 0.0
    ok_lower = total_pairs >= lower
    ok_upper = total_pairs <= upper
    print(f"[knn] total active pixel edges = {total_pairs}, lower {lower}, upper {upper} "
          f"(per pixel upper {upper_per_pixel}), per pixel average {per_pixel_avg:.3f}, "
          f"pass lower = {ok_lower}, pass upper = {ok_upper}")
    return ok_lower and ok_upper


def check_weight_upper_bounds(weight_sum_mat, Wi_list, num_pairs_to_sample=5, rng_seed=0):
    """
    For sampled pairs, verify sum_p keep*Q_ij[p] <= sum_p min(W_i[p], W_j[p]).
    """
    N = len(Wi_list)
    U = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(i + 1, N):
            Uij = float(np.sum(np.minimum(Wi_list[i], Wi_list[j])))
            U[i, j] = Uij
            U[j, i] = Uij

    rng = np.random.default_rng(rng_seed)
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    if len(pairs) == 0:
        print("no pairs to check")
        return True

    sample_idx = rng.choice(len(pairs), size=min(num_pairs_to_sample, len(pairs)), replace=False)
    ok_all = True
    for idx in sample_idx:
        i, j = pairs[idx]
        lhs = weight_sum_mat[i, j]
        rhs = U[i, j]
        ok = lhs <= rhs + 1e-12
        ok_all = ok_all and ok
        print(f"[weights] pair ({i},{j}) sum {lhs:.6e} upper bound {rhs:.6e}, pass = {ok}")
    return ok_all


def check_conservation(count_mat, weight_sum_mat):
    """
    Check that sum of degrees equals twice the sum over upper triangle, for both counts and weights.
    """
    sum_counts_upper = float(np.sum(np.triu(count_mat, k=1)))
    sum_weights_upper = float(np.sum(np.triu(weight_sum_mat, k=1)))
    degree_count = np.sum(count_mat, axis=1)
    degree_weight = np.sum(weight_sum_mat, axis=1)
    lhs_counts = float(np.sum(degree_count))
    lhs_weights = float(np.sum(degree_weight))
    ok_counts = np.isclose(lhs_counts, 2.0 * sum_counts_upper)
    ok_weights = np.isclose(lhs_weights, 2.0 * sum_weights_upper)
    print(f"[conservation] counts sum over degrees {int(lhs_counts)} equals twice upper triangle {int(2*sum_counts_upper)}, pass = {ok_counts}")
    print(f"[conservation] weights sum over degrees {lhs_weights:.6e} equals twice upper triangle {2*sum_weights_upper:.6e}, pass = {ok_weights}")
    return ok_counts and ok_weights


def per_pixel_edge_counts(keep):
    """
    For each pixel p, compute the number of undirected edges in G(p),
    which is the count of i<j where keep[i,j,p] is true.
    """
    N, _, n = keep.shape
    counts = np.zeros(n, dtype=int)
    for p in range(n):
        c = 0
        for i in range(N):
            for j in range(i + 1, N):
                if keep[i, j, p]:
                    c += 1
        counts[p] = c
    return counts


def plot_histograms(count_mat, weight_sum_mat, per_pixel_counts, out_dir, tag, show_plots=True):
    """
    Save three plots:
      1 histogram of pair counts across pixels
      2 histogram of pair weight sums across pixels
      3 histogram of per pixel undirected edge counts
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1 pair counts histogram
    upper_idx = np.triu_indices(count_mat.shape[0], k=1)
    counts_vec = count_mat[upper_idx]
    plt.figure(figsize=(6, 4))
    plt.hist(counts_vec, bins=20)
    plt.xlabel("active pixel edges per pair")
    plt.ylabel("pair frequency")
    plt.title(f"Histogram of pair counts {tag}")
    plt.tight_layout()
    f1 = os.path.join(out_dir, f"hist_pair_counts_{tag}.png")
    plt.savefig(f1, dpi=200)
    if show_plots:
        plt.show()
    else:
        plt.close()

    # 2 pair weight sums histogram
    weights_vec = weight_sum_mat[upper_idx]
    plt.figure(figsize=(6, 4))
    plt.hist(weights_vec, bins=20)
    plt.xlabel("sum of weights per pair across active pixels")
    plt.ylabel("pair frequency")
    plt.title(f"Histogram of pair weight sums {tag}")
    plt.tight_layout()
    f2 = os.path.join(out_dir, f"hist_pair_weights_{tag}.png")
    plt.savefig(f2, dpi=200)
    if show_plots:
        plt.show()
    else:
        plt.close()

    # 3 per pixel undirected edge count histogram
    plt.figure(figsize=(6, 4))
    plt.hist(per_pixel_counts, bins=range(int(per_pixel_counts.min()), int(per_pixel_counts.max()) + 2))
    plt.xlabel("undirected edges in G(p)")
    plt.ylabel("pixel frequency")
    plt.title(f"Histogram of per pixel edge counts {tag}")
    plt.tight_layout()
    f3 = os.path.join(out_dir, f"hist_per_pixel_edges_{tag}.png")
    plt.savefig(f3, dpi=200)
    if show_plots:
        plt.show()
    else:
        plt.close()

    print(f"Saved plots:\n  {f1}\n  {f2}\n  {f3}")


def run_checks(strategy="mst", k=2, base_dir="saved_operators_Incmp_Span", out_dir="hist_checks_tmp", show_plots=True, verbose=True):
    G_union, Wi_list, Qij_diag_fn_masked, keep = build_pixel_connected_Q_provider(
        base_dir=base_dir,
        strategy=strategy,
        k=k,
        verbose=verbose,
        plot_union=False,
        show_plots=False,
        output_dir=os.path.join(out_dir, f"union_{strategy}")
    )
    N = len(Wi_list)
    n = Wi_list[0].shape[0]
    q_cache_unmasked = harmonic_Q_unmasked(Wi_list)
    count_mat, weight_sum_mat, degree_count, degree_weight = build_pair_summaries(keep, q_cache_unmasked)

    total_pairs = int(np.sum(np.triu(count_mat, k=1)))
    mean_per_pair = total_pairs / (N * (N - 1) / 2.0) if N > 1 else 0.0
    print(f"\nStrategy {strategy}, N = {N}, n = {n}")
    print(f"total active pixel edges across all pixels = {total_pairs}, mean per pair = {mean_per_pair:.3f}")

    ok_list = []

    if strategy in ["mst", "chain"]:
        ok_list.append(check_chain_or_mst(count_mat, N, n, label=strategy))
    elif strategy == "knn":
        ok_list.append(check_knn_bounds(count_mat, N, n, k))
    else:
        print("unknown strategy label")

    ok_list.append(check_weight_upper_bounds(weight_sum_mat, Wi_list, num_pairs_to_sample=min(10, N*(N-1)//2)))
    ok_list.append(check_conservation(count_mat, weight_sum_mat))

    # Per pixel edge count histogram
    p_counts = per_pixel_edge_counts(keep)
    tag = f"{strategy}_k{k}" if strategy == "knn" else strategy
    plot_histograms(count_mat, weight_sum_mat, p_counts, out_dir=out_dir, tag=tag, show_plots=show_plots)

    all_ok = all(ok_list)
    print(f"\nOverall pass = {all_ok}")
    return {
        "count_mat": count_mat,
        "weight_sum_mat": weight_sum_mat,
        "degree_count": degree_count,
        "degree_weight": degree_weight,
        "per_pixel_edge_counts": p_counts,
        "all_ok": all_ok
    }


if __name__ == "__main__":
    os.makedirs("hist_checks_tmp", exist_ok=True)
    run_checks(strategy="mst", k=2, out_dir="hist_checks_tmp", show_plots=True)
    run_checks(strategy="chain", k=2, out_dir="hist_checks_tmp", show_plots=True)
    run_checks(strategy="knn", k=2, out_dir="hist_checks_tmp", show_plots=True)
