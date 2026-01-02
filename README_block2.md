# 1. ODL Data Loader and Visualization (`block_2_load_odl_data.py`)

## 1.1 Purpose

This module loads precomputed ODL ray transform operators and their dense matrix forms, prepares a phantom image, synthesizes per-node sinograms with additive Gaussian noise, and optionally builds an aggregate sinogram. It also produces visual outputs for inspection and debugging.

This module is typically used as the second stage in a distributed tomography pipeline, following the creation of ODL operators and their dense equivalents.



## 2. Background on ODL Concepts

### 2.1 Space

In [ODL](https://odlgroup.github.io/odl/), a **space** represents the function space in which images or signals reside.  
For example, a uniformly discretized 2D grid of size \( N 	imes N \) is a space.  
Each operator in ODL defines a mapping between two spaces — a domain and a range.

In the script, the line:
```python
space = ray_transforms[0].domain
```
retrieves the **image space** on which the ray transform operates.

### 2.2 Elements

An **element** is a specific value in that space. It behaves like a NumPy array but carries metadata (such as shape and grid structure) so that ODL operators can act on it safely.  
An element is created by wrapping an array as follows:
```python
phantom = space.element(phantom_array)
```
This ensures compatibility between the phantom image and all ODL operators that share the same domain.

More details: [ODL Space Elements](https://odlgroup.github.io/odl/api/odl.space.html#module-odl.space).

### 2.3 Domain and Range

Every ODL operator \( T \) has a **domain** and a **range**:

- `T.domain` — the space of input elements (for example, images).  
- `T.range` — the space of output elements (for example, sinograms).

When an operator is called on an element, as in `op(phantom)`, the result is an element in the operator’s range.  
Calling `.asarray()` converts this object into a NumPy array for standard processing.

### 2.4 Ray Transforms and Dense Matrices

The `ray_transforms` list contains one ODL operator per node. Each operator can represent a subset of projection angles or detectors.  
Corresponding dense matrix forms of these operators are stored in `A_dense_list.pkl`, allowing direct linear-algebraic manipulations.

Dense matrices are used for diagnostics, aggregate sinogram construction, and verification against operator-based outputs.

---

## 3. Inputs Expected from Previous Stage

The function reads the following pickled files from `base_dir`:

| File | Description |
|------|--------------|
| `ray_transforms.pkl` | List of ODL ray transform operators (one per node). |
| `A_dense_list.pkl` | List of dense forward matrices corresponding to each node. |
| `aggregate_op.pkl` | (Optional) Aggregate ODL operator representing all nodes together. |
| `A_agg.pkl` | (Optional) Aggregate dense matrix corresponding to the combined operator. |

These files are created during the operator generation stage (Block 1) and must be available before this script is executed.

---

## 4. Phantom Creation and Configuration

If no custom phantom is supplied, the helper function `ConstIm(N)` from `Gen_Sino_Partitioned.py` creates a uniform phantom.  
This array is then wrapped in the ODL image space via:
```python
phantom = space.element(phantom_array)
```

This guarantees that the phantom aligns correctly with the domain of the ray transform.  
Custom phantoms can also be passed in as NumPy arrays with shape `(N, N)`.

---

## 5. Sinogram Generation and Noise Injection

For each node \( i \), a noisy sinogram is generated as
\[
s_i = T_i(	ext{phantom}) + \sigma \, arepsilon_i,
\]
where \( \sigma = 	ext{noise\_level} \) and \( arepsilon_i \sim \mathcal{N}(0, I) \).

In the code, the noise is created with:
```python
op.range.element(np.random.normal(0.0, 1.0, size=op.range.shape))
```
and added to the noiseless projection `op(phantom)`.

The resulting list `sinograms` stores all sinograms as NumPy arrays obtained through `.asarray()`.

Additionally, the column-wise norms of all dense matrices \( A_i \) are computed to facilitate later weighting or normalization.

---

## 6. Aggregate Operator and Sinogram

If the aggregate dense matrix `A_agg` is available, the combined sinogram is computed as:
\[
s_{	ext{agg}} = A_{	ext{agg}} x_{	ext{true}} + 	ext{noise},
\]
where \( x_{	ext{true}} \) is the vectorized phantom.  
Otherwise, the aggregate sinogram is constructed by vertically stacking all per-node sinograms.

This allows validation of per-node and global forward models under identical noise assumptions.

---

## 7. Plotting and Visualization

When `make_plots=True`, the module saves visualizations to a timestamped output folder.  
Generated figures include:

- **`true_phantom.png`** — the original phantom image in grayscale.  
- **`plot3_clean_true_sinograms.png`** — a composite panel showing the phantom, individual node sinograms, and the aggregate sinogram.

If `show_plots=True`, figures are displayed interactively using the `TkAgg` backend.  
For headless execution, the backend may be changed before importing Matplotlib.

---

## 8. Return Structure

The function returns a dictionary with the following entries:

| Key | Type | Description |
|-----|------|-------------|
| `"A_dense_list"` | `list[np.ndarray]` | Dense forward matrices \( A_i \) for each node, used in equation(1). |
| `"sinograms"` | `list[np.ndarray]` | Per-node noisy sinograms b_i used in equation(1). |
| `"column_norms_all"` | `list[np.ndarray]` | Column-wise norms of each dense operator A_i, names eta_i or W_i used in equation (2), and (1). |
| `"N"` | `int` | Image side length (phantom dimension). |
| `"num_nodes"` | `int` | Number of nodes or partitions. |
| `"agg_ray_trafo"` | `odl.Operator` or `None` | Aggregate ODL operator. |
| `"A_agg"` | `np.ndarray` or `None` | Aggregate dense matrix vertically stacked matrices [A_1, A_2,...A_P]. |
| `"agg_sinogram"` | `np.ndarray` | Aggregate sinogram array. |
| `"output_dir"` | `str` | Directory where output plots are saved. |
| `"phantom"` | `np.ndarray` | The ground truth phantom as a NumPy array. |

---

## 9. Example Usage

```python
from block_2_load_odl_data import load_odl_data

data = load_odl_data(
    base_dir="saved_operators_Incmp_Span",
    N=64,
    num_nodes=5,
    noise_level=0.005,
    make_plots=True,
    show_plots=False
)

phantom = data["phantom"]
sinograms = data["sinograms"]
A_dense_list = data["A_dense_list"]
agg_sino = data["agg_sinogram"]
```

---

## 10. Shape Summary

| Quantity | Symbol | Shape |
|-----------|---------|--------|
| Image (phantom) | \( x \) | \( N 	imes N \) |
| Dense operator per node | \( A_i \) | \( m_i 	imes N^2 \) |
| Aggregate operator | \( A_{	ext{agg}} \) | \( (\sum_i m_i) 	imes N^2 \) |
| Per-node sinogram | \( s_i \) | `op.range.shape` |
| Aggregate sinogram | \( s_{	ext{agg}} \) | `[sum of all node rows, detector count]` |

---

## 11. Typical Directory Structure

```
project_root/
│
├── block_2_load_odl_data.py
├── saved_operators_Incmp_Span/
│   ├── ray_transforms.pkl
│   ├── A_dense_list.pkl
│   ├── aggregate_op.pkl       # optional
│   └── A_agg.pkl              # optional
└── Gen_Sino_Partitioned.py
```

---

## 12. Dependencies

- [NumPy](https://numpy.org/doc/)  
- [Matplotlib](https://matplotlib.org/stable/users/index.html)  
- [ODL](https://odlgroup.github.io/odl/)  
- Standard Python libraries: `pickle`, `os`, `datetime`

---

## 13. Notes and Recommendations

- The number of ODL ray transforms and dense matrices must match `num_nodes`.  
- Wrapping arrays with `space.element` ensures correct type and shape consistency.  
- For automated pipelines, set `show_plots=False` to avoid GUI blocking.  
- When using remote or cluster environments without display servers, switch to a non-interactive Matplotlib backend such as `Agg`.  
- This module serves as the **data preparation and visualization** step before reconstruction (Block 3 and beyond).

