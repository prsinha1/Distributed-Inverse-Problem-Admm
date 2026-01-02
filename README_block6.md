# 1. Decentralized ADMM Loop (`block_6_admm_loop.py`)

## 1.1 Purpose

This module runs the outer Alternating Direction Method of Multipliers loop for a decentralized reconstruction. One convex subproblem is solved per node at every outer iteration using CVXPY with the SCS solver. Consensus variables and dual variables are updated on graph edges, primal and dual residuals are tracked, optional snapshots are saved, and several error histories are recorded. fileciteturn3file0

---

## 2. Function summary

### `decentralized_admm(A_dense_list, sinograms, G, Wi_list, Qij_diag_fn, N, lam_tv=0.01, rho=1.0, max_iters=10, eps_pri=1e-1, eps_dual=1e-1, verbose=True, snapshot_dir=None, snapshot_every=None, snapshot_div=10, phantom_true=None)`

Runs decentralized ADMM over a node graph. Returns per node reconstructions and a dictionary of histories. fileciteturn3file0

---

## 3. Inputs

| Name | Type | Shape | Meaning |
|------|------|-------|---------|
| `A_dense_list` | `list[np.ndarray]` | length `num_nodes`, each `(m_i, n)` | Dense forward matrices per node. Used for the data fit and residuals. fileciteturn3file0 |
| `sinograms` | `list[np.ndarray]` | length `num_nodes`, each as an array | Per node sinograms. Each is flattened to a vector `b[i]`. fileciteturn3file0 |
| `G` | `networkx.Graph` | with integer nodes `[0, …, num_nodes-1]` | Node adjacency describes communication pairs for consensus. fileciteturn3file0 |
| `Wi_list` | `list[np.ndarray]` | length `num_nodes`, each `(n,)` | Per coordinate weights used in the edge update design. The current code averages without these weights, see Section 6. fileciteturn3file0 |
| `Qij_diag_fn` | `callable` | returns `(n,)` for pair `(i, j)` | Provides diagonal weights `q_ij` used inside node problems to weight squared differences. fileciteturn3file0 |
| `N` | `int` | scalar | Image side length. Used by the TV term inside the node problems. |
| `lam_tv` | `float` | scalar | Weight of isotropic total variation in node problems. |
| `rho` | `float` | scalar | ADMM penalty parameter in the coupling terms and dual updates. |
| `max_iters` | `int` | scalar | Maximum number of outer ADMM iterations. |
| `eps_pri` | `float` | scalar | Target threshold for primal residual norm. |
| `eps_dual` | `float` | scalar | Target threshold for dual residual norm. |
| `verbose` | `bool` | scalar | Prints progress every few iterations and at solver start and end per node. fileciteturn3file0 |
| `snapshot_dir` | `str or None` | path | If set, images are saved periodically in this directory. fileciteturn3file0 |
| `snapshot_every` | `int or None` | scalar | Snapshot period in outer iterations. If None a value is derived from `max_iters` and `snapshot_div`. fileciteturn3file0 |
| `snapshot_div` | `int` | scalar | Used to derive a default snapshot period. fileciteturn3file0 |
| `phantom_true` | `np.ndarray or None` | `(N, N)` or `(n,)` | If provided, image space errors are computed with respect to this ground truth. fileciteturn3file0 |

---

## 4. Outer iteration structure

At each outer iteration `k`:

1. **Node subproblems**. For each node `i` the function `build_node_problem` is called to construct a convex problem in CVXPY for the variable `x_i`. The neighbor vectors `v_ij` and diagonal weights `q_ij` are passed in. The problem is solved with SCS and the result is stored in `x[i]`. Objective values are recorded. fileciteturn3file0  
2. **Sinogram errors**. Per node residuals `r_i = A_i x_i − b_i` are computed. Squared norms are stored per node and summed for the total. fileciteturn3file0  
3. **Image errors**. If a phantom is provided it is vectorized and per node squared errors `‖x_i − x_true‖_2^2` are recorded along with their sum. fileciteturn3file0  
4. **Consensus update**. For each edge `(i, j)` a new `z_ij` is computed from current `x_i` and `x_j` and duals. See Section 6 for the exact formula. fileciteturn3file0  
5. **Dual update**. Edge dual variables are updated using the standard ADMM formula with parameter `rho`. fileciteturn3file0  
6. **Residuals**. The primal residual stacks deviations of nodes from edge variables. The dual residual is based on the change in `z` and is scaled by `rho`. Norms are recorded globally and per node. fileciteturn3file0  
7. **Snapshots and stopping**. Images are saved every `snapshot_every` iterations if a directory is provided. The loop stops early if both norms fall below their thresholds. fileciteturn3file0

---

## 5. CVXPY and SCS usage inside the loop

Each node problem is built by `build_node_problem` and solved with SCS as follows:
```python
xi_var, prob = build_node_problem(Ai, bi, rho, neighbor_vs, N, lam_tv, neighbor_Qs)
tag = f"node_{i}_outer_{k}"
print(f"=== SCS start {tag} ===")
prob.solve(
    solver=cp.SCS,
    eps=3e-2,
    max_iters=200,
    verbose=False,
    warm_start=True
)
print(f"=== SCS end   {tag} status={prob.status} obj={prob.value} ===")
```
Inputs to `prob.solve`:

- `solver=cp.SCS` selects the Splitting Conic Solver. SCS solves cone programs by operator splitting and is effective for large problems with nonsmooth terms such as TV.
- `eps=3e-2` sets the accuracy target for primal and dual feasibility and objective gap inside SCS. Smaller values increase accuracy at the cost of time.
- `max_iters=200` caps the number of SCS iterations. Increase this value if the status reports incomplete convergence.
- `verbose=False` suppresses SCS iteration logs. Set to `True` for detailed progress.
- `warm_start=True` keeps previous values for primal and dual variables when solving a related problem again. Warm starts help when problems change slowly across outer iterations.

Additional optional SCS parameters that may be useful:

Solving Equation (1)
```python
prob.solve(
    solver=cp.SCS,
    eps=1e-3,
    max_iters=5000,
    alpha=1.5,          # relaxation parameter in ADMM splitting inside SCS
    scale=1.0,          # data scaling level
    acceleration_lookback=0,  # turn on Anderson accel with positive integer
    verbose=True
)
```
- `alpha` controls relaxation in the internal ADMM iterations of SCS. Values slightly above one can improve practical convergence.  
- `scale` applies data scaling to improve conditioning before iterations.  
- `acceleration_lookback` turns on Anderson acceleration when positive. This can speed up convergence on some problems.

A minimal example outside the loop for testing SCS on a node problem:

```python
import cvxpy as cp, numpy as np
n = 64
x = cp.Variable(n)
A = np.random.randn(200, n)
b = np.random.randn(200)
obj = 0.5*cp.sum_squares(A @ x - b)
prob = cp.Problem(cp.Minimize(obj))
prob.solve(solver=cp.SCS, eps=1e-3, max_iters=2000, verbose=True)
```

---

## 6. Edge variable and dual updates (Equation. 2)

For edge `(i, j)` with key `key = (min(i, j), max(i, j))` the following quantities are formed:
```python
a_i = x[i] + y[(key[0], key[1], i)]
a_j = x[j] + y[(key[0], key[1], j)]
num = a_i + a_j
den = 2.0
z[key] = num / den
```
This is a simple average of `a_i` and `a_j` per coordinate. A weighted version based on `Wi` and `Wj` is commented in the source. After `z` is updated, duals are advanced as
```python ## Equation (3)
y[(key[0], key[1], i)] += x[i] - z[key]
y[(key[0], key[1], j)] += x[j] - z[key]
```
which is the standard scaled dual update with parameter `rho` folded into the dual residual definition. fileciteturn3file0

---

## 7. Residuals and stopping

Primal residual stacks node deviations from edge variables across all edges
```python
ri = x[i] - z[key]
rj = x[j] - z[key]
```
and sums their squared norms. The dual residual uses
```python
dz = new_z[key] - z[key]
s2 += rho * rho * np.sum(dz * dz)
```
Global norms are recorded as `sqrt(r2)` and `sqrt(s2)`. The loop stops if both are below `eps_pri` and `eps_dual`. fileciteturn3file0

---

## 8. Histories and metrics

The function records several arrays per iteration:

- `primal`, `dual` are the global residual norms.  
- `pri_per_node`, `dual_per_node` store per node residual contributions after square root.  
- `obj_per_node`, `obj_total` store objectives from node solves and their sum.  
- `mse_sino_per_node`, `mse_sino_total` are squared residual norms in sinogram space per node and in total.  
- `img_mse_per_node`, `img_mse_total` are squared errors to the true image when `phantom_true` is provided.  

Note. The MSE values are unnormalized squared norms. Divide by the number of entries to compare directly to per entry noise variance when needed. fileciteturn3file0

---

## 9. Snapshots

When `snapshot_dir` is set, a directory is created if needed. Every `snapshot_every` iterations the current `x[i]` is reshaped to `(N, N)`, saved as `.npy`, and written as a grayscale `.png` with title annotations. The period defaults to `max(1, max_iters // snapshot_div)` when `snapshot_every` is not specified. fileciteturn3file0

---

## 10. Return structure

The function returns `(x, history)` where:

- `x` is a list of length `num_nodes`. Entry `x[i]` is the current image vector of node `i` with length `n`.  
- `history` is a dictionary with keys  
  `primal`, `dual`, `pri_per_node`, `dual_per_node`, `obj_per_node`, `obj_total`, `mse_sino_per_node`, `mse_sino_total`, `img_mse_per_node`, `img_mse_total`.  
  Each is a Python list of per iteration values. fileciteturn3file0

---

## 11. Example usage

```python
import numpy as np, networkx as nx
from block_6_admm_loop import decentralized_admm
from block_5_node_problem import build_node_problem  # dependency for node problems

# Assume A_dense_list, sinograms, Wi_list, and Qij provider already exist
num_nodes = len(A_dense_list)
n = A_dense_list[0].shape[1]

# Build a simple connected graph
G = nx.cycle_graph(num_nodes)

# Run ADMM
x, hist = decentralized_admm(
    A_dense_list=A_dense_list,
    sinograms=sinograms,
    G=G,
    Wi_list=Wi_list,
    Qij_diag_fn=lambda i, j: np.ones(n),
    N=64,
    lam_tv=0.05,
    rho=1.0,
    max_iters=50,
    eps_pri=1e-2,
    eps_dual=1e-2,
    verbose=True,
    snapshot_dir="snapshots_block6",
    snapshot_every=5
)
print("Final primal residual:", hist["primal"][-1])
print("Final dual residual:", hist["dual"][-1])
```

---

## 12. Dependencies

- NumPy for arrays and linear algebra  
- NetworkX for graph handling  
- CVXPY for convex modeling and SCS integration  
- Matplotlib for snapshot images  
- `block_5_node_problem` for the node level objective construction

---

## 13. Practical remarks

- The SCS accuracy parameter `eps` should be aligned with the outer stopping thresholds. A very loose inner tolerance can slow or destabilize outer convergence.  
- The TV weight and the ADMM penalty require tuning for a given dataset. Monitoring `obj_total` and residual norms helps with selection.  
- The edge average can be replaced with a weighted average using `Wi` and `Wj` if those vectors reflect meaningful per coordinate confidence. The code includes commented lines that show the weighted form.

