# 1. Pixel Graph and Precision Builder (`block_3_graph_and_precisions.py`)

## 1.1 Overview

This module constructs pixel-level connectivity graphs and precision (Q) providers from a list of dense forward operators stored in `A_dense_list`.  
Each node typically corresponds to a local operator, such as those appearing in distributed tomography or multi-agent reconstruction.  
The script defines per-pixel coupling weights \( Q_{ij}[p] \) and connectivity masks describing how nodes interact for each pixel.  
Several graph construction strategies are supported, along with visualization and diagnostic tools.
Q_ij used in equation (1) and W_i, W_j used in equation (2) of algorithm.

---

## 2. Functional Summary

### 2.1 `make_precisions(A_dense_list, q_mode)`
Computes per-node precision vectors  
\[
W_i[p] = \|A_i[:, p]\|_2^2
\]
and defines per-pair weights:

- **Harmonic mean:**  
  \( Q_{ij}[p] = \frac{W_i[p] W_j[p]}{W_i[p] + W_j[p]} \)
- **Arithmetic mean:**  
  \( Q_{ij}[p] = 0.5 \times (W_i[p] + W_j[p]) \)

Returns:
- `Wi_list`: list of per-node weight vectors  
- `Qij_diag(i, j)`: callable returning \( Q_{ij} \)

---

### 2.2 `_precompute_q_cache(num_nodes, Qij_diag)`
Creates a cache of all \( Q_{ij} \) vectors for ordered node pairs.  
The result is a dictionary mapping `(i, j)` to a NumPy array of per-pixel weights.

---

### 2.3 Pixel Graph Construction Methods

Each pixel builds an adjacency mask defining which nodes are connected at that pixel.

- **`_pixel_mask_knn_then_connect(q_col, k, num_nodes)`**  
  Selects the top-k neighbors per node, symmetrizes the connections, and ensures connectivity via a maximum spanning tree.

- **`_pixel_mask_mst(q_col, num_nodes)`**  
  Builds a complete weighted graph for that pixel and retains only the edges from the maximum spanning tree.

- **`_pixel_mask_chain(num_nodes, rng)`**  
  Generates a random chain linking all nodes sparsely while maintaining connectivity.

---

### 2.4 `_build_all_pixel_masks(q_cache, num_nodes, n, strategy, k, seed)`
Constructs per-pixel adjacency masks (`keep[i, j, p]`) using one of the above strategies.  
Parameters include:
- `strategy`: `"knn"`, `"mst"`, or `"chain"`
- `k`: number of neighbors (used for the kNN strategy)
- `seed`: random seed for reproducibility

---

### 2.5 `_summarize_and_plot_union(keep, output_dir, show_plots, verbose, title_suffix)`
Creates a union graph across all pixels and saves:
- a visualization of the node connectivity graph  
- a node-degree histogram  

Diagnostic statistics are printed, including connectivity, degree range, and the ratio of active pixel edges.

---

### 2.6 `build_pixel_connected_Q_provider(...)`
Main entry point for constructing the Q provider and associated graph structures.  
Performs the following operations:

1. Loads `A_dense_list` from a pickle file.  
2. Computes `Wi_list` and the corresponding base `Qij` weights.  
3. Builds per-pixel connectivity masks using the chosen strategy.  
4. Optionally generates union-graph visualizations and degree histograms.  
5. Returns:
   - `G_union`: NetworkX graph representing the pixel-union connectivity  
   - `Wi_list`: list of per-node precisions  
   - `Qij_diag_masked(i, j)`: callable returning masked \( Q_{ij} \) per pixel  
   - `keep`: boolean tensor `[num_nodes, num_nodes, n_pixels]` representing active edges

---

## 3. Directory Structure

```
project_root/
│
├── block_3_graph_and_precisions.py
├── saved_operators_Incmp_Span/
│   └── A_dense_list.pkl          # Serialized list of A_i matrices
├── pixel_graphs_out/             # Output folder for plots and summaries
└── ...
```

---

## 4. Example of Use

```python
from block_3_graph_and_precisions import build_pixel_connected_Q_provider

G_union, Wi_list, Qij_diag_masked, keep = build_pixel_connected_Q_provider(
    base_dir="saved_operators_Incmp_Span",
    A_dense_list_pickle="A_dense_list.pkl",
    strategy="knn",      # or "mst", "chain"
    k=2,
    seed=42,
    q_mode="arithmetic", # or "harmonic"
    verbose=True,
    plot_union=True,
    show_plots=False,
    output_dir="pixel_graphs_out"
)
```

---

## 5. Output Description

Console diagnostics may include:
```
[Block3] loaded A_dense_list from ...
[Block3] strategy knn_k2_arithmetic
[Block3] nodes 4, edges 5
[Block3] connected True
[Block3] degree min mean max 2  2.50  3
[Block3] active pixel ratio 0.4231
```

Generated figures:
- `pixel_union_graph_knn_k2_arithmetic.png`
- `pixel_union_degree_knn_k2_arithmetic.png`

---

## 6. Dependencies

The following Python libraries are required:
```
numpy
matplotlib
networkx
pickle
```

---

## 7. Notes

- The file `A_dense_list.pkl` must exist within the specified `base_dir`.  
  It is expected to contain a Python list of NumPy arrays, each array representing a node-specific operator.  
- The parameter `q_mode` controls whether harmonic or arithmetic averaging is used when computing \( Q_{ij} \).  
- The choice of graph strategy (`knn`, `mst`, `chain`) determines sparsity and connectivity behavior at the pixel level.
