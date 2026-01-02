# 1. Node-Level Optimization Problem (`block_5_node_problem.py`)

## 1.1 Purpose

This module defines and constructs the **per-node optimization subproblem** in a distributed or consensus-based tomography reconstruction framework.  
Each node solves a local convex optimization problem involving:

1. A data fidelity term derived from its own measurement matrix \( A_i \) and data \( b_i \).  
2. A total variation (TV) regularization term promoting spatial smoothness.  
3. A quadratic penalty coupling the node with its neighbors via consensus variables and weighted diagonal metrics \( Q_{ij} \).  

The problem is formulated and solved using **[CVXPY](https://www.cvxpy.org/)**, a Python package for convex optimization.

---

## 2. Function Summary

### `build_node_problem(Ai, bi, rho, neighbor_terms, N, lam_tv, Qij_terms)`

Constructs and returns a convex optimization problem and decision variable representing the node’s local reconstruction task.

---

## 3. Mathematical Formulation

For a given node \( i \), the optimization problem is:

\[
\min_{x_i \in \mathbb{R}^n} \quad 
\frac{1}{2} \| A_i x_i - b_i \|_2^2 
+ \lambda_{\text{tv}} \, \text{TV}(x_i)
+ \frac{\rho}{2} \sum_{j \in \mathcal{N}_i} \| x_i - v_{ij} \|_{Q_{ij}}^2,
\] equation (1)
where:
- \( \|u\|_Q^2 = \sum_k Q[k] \, u[k]^2 \),
- \( v_{ij} = z_{ij} - y_{ij,i} \) are adjusted neighbor variables, equation(3)
- \( \text{TV}(x_i) \) is the isotropic total variation of the reshaped image.

This represents a **quadratic + nonsmooth convex optimization problem**, which is solved using CVXPY’s disciplined convex programming framework.

---

## 4. Explanation of Inputs

| Parameter | Type | Shape | Description |
|------------|------|--------|-------------|
| `Ai` | `np.ndarray` | `(m_i, n)` | Local dense measurement matrix for node \( i \). Each row corresponds to a projection measurement. |
| `bi` | `np.ndarray` | `(m_i,)` | Local sinogram or measurement vector. |
| `rho` | `float` | scalar | Consensus penalty parameter controlling coupling strength across nodes. |
| `neighbor_terms` | `list[np.ndarray]` | `[v_ij]` where each \( v_{ij} \in \mathbb{R}^n \) | Adjusted neighbor vectors \( v_{ij} = z_{ij} - y_{ij,i} \). One for each neighbor node \( j \). |
| `N` | `int` | scalar | Image side length; used to reshape the vector \( x_i \) into an \( N \times N \) image inside the TV regularization term. |
| `lam_tv` | `float` | scalar | Regularization weight for the total variation penalty. |
| `Qij_terms` | `list[np.ndarray]` | `[q_ij]` where each \( q_{ij} \in \mathbb{R}^n \) | Diagonal weight vectors defining the local metric \( Q_{ij} \) in the quadratic penalty. |

---

## 5. Construction of the Optimization Problem

The code defines the optimization variable and constructs the objective step-by-step using CVXPY primitives.

### 5.1 Decision Variable

```python
xi = cp.Variable(n)
```
Creates a **CVXPY variable** representing \( x_i \in \mathbb{R}^n \).  
CVXPY tracks this as a symbolic vector and automatically infers dimensions from its usage.

---

### 5.2 Data-Fidelity Term

```python
data_fit = 0.5 * cp.sum_squares(Ai @ xi - bi)
```
Represents \( \frac{1}{2} \| A_i x_i - b_i \|_2^2 \).  
- `Ai @ xi`: matrix-vector multiplication (NumPy-compatible).  
- `cp.sum_squares(expr)`: CVXPY atom returning the sum of elementwise squares, equivalent to \(\|expr\|_2^2\).  
- Scaling by `0.5` ensures smooth gradients and consistency with least-squares formulations.

---

### 5.3 Total Variation Term

```python
tv_term = lam_tv * isotropic_tv_on_vector(xi, N)
```
Adds the **isotropic total variation** penalty promoting local smoothness while preserving edges.  
`isotropic_tv_on_vector` is imported from `block_4_tv_helpers` and interprets the vector `xi` as an \( N \times N \) image grid.  
The function internally computes finite differences and sums Euclidean magnitudes per pixel.  
It is multiplied by `lam_tv` to control regularization strength.

---

### 5.4 Quadratic Consensus Penalty

```python
quad_terms = 0
for v_ij, q_diag in zip(neighbor_terms, Qij_terms):
    diff = xi - v_ij
    quad_terms += 0.5 * rho * cp.sum(cp.multiply(q_diag, cp.square(diff)))
```
Each loop iteration corresponds to one neighbor \( j \):

- `diff = xi - v_ij`: deviation from the adjusted neighbor variable.  
- `cp.square(diff)`: elementwise squaring of the difference vector.  
- `cp.multiply(q_diag, cp.square(diff))`: applies per-coordinate weights \( q_{ij}[k] \).  
- `cp.sum(...)`: computes \( \sum_k q_{ij}[k](x_i[k] - v_{ij}[k])^2 \).  
- Multiplying by `0.5 * rho` yields \( \frac{\rho}{2} \| x_i - v_{ij} \|_{Q_{ij}}^2 \).  
- The term `quad_terms` accumulates all neighbor penalties.

---

### 5.5 Full Objective Function

```python
obj = data_fit + tv_term + quad_terms
```
Mathematically:
\[
\text{Objective}(x_i) = \frac{1}{2}\|A_i x_i - b_i\|_2^2 
+ \lambda_{\text{tv}} \, \text{TV}(x_i)
+ \frac{\rho}{2}\sum_j \|x_i - v_{ij}\|_{Q_{ij}}^2
\]

---

### 5.6 Problem Definition

```python
prob = cp.Problem(cp.Minimize(obj))
```
Constructs a CVXPY `Problem` object specifying a **minimization** task.  
- `cp.Minimize(obj)` defines the objective node.  
- `cp.Problem` wraps it into a solvable problem graph.  
- The resulting `prob` can be solved using any convex solver, for example:
```python
prob.solve(solver=cp.SCS, eps=1e-3, max_iters=1000, verbose=True)
```

---

## 6. Return Values

```python
return xi, prob
```

| Variable | Type | Description |
|-----------|------|-------------|
| `xi` | `cvxpy.Variable` | Optimization variable representing the node’s local image vector. |
| `prob` | `cvxpy.Problem` | Fully defined CVXPY problem ready for solving. |

---

## 7. Example Usage

```python
from block_5_node_problem import build_node_problem
import numpy as np
import cvxpy as cp

m_i, n = 200, 64
Ai = np.random.randn(m_i, n)
bi = np.random.randn(m_i)
rho = 1.0
lam_tv = 0.1
N = 8
neighbor_terms = [np.random.randn(n), np.random.randn(n)]
Qij_terms = [np.ones(n), np.linspace(0.5, 1.5, n)]

xi, prob = build_node_problem(Ai, bi, rho, neighbor_terms, N, lam_tv, Qij_terms)

prob.solve(solver=cp.SCS, eps=1e-3, max_iters=500, verbose=True)

print("Optimal value:", prob.value)
print("First 10 entries of x_i:", xi.value[:10])
```

---

## 8. Dependencies

- [NumPy](https://numpy.org/doc/stable/) — numerical computation and array operations.  
- [CVXPY](https://www.cvxpy.org/tutorial/index.html) — convex problem modeling and solving.  
- `block_4_tv_helpers` — defines `isotropic_tv_on_vector`, required for TV regularization.

---

## 9. Notes

- The optimization problem is convex and differentiable except at points of nondifferentiability in the TV term.  
- Solvers such as **SCS** and **CVXOPT** are recommended for TV-regularized problems.  
- Each node can solve its local problem independently and share consensus variables with neighbors between iterations in a distributed ADMM framework.  
- Proper tuning of `rho` and `lam_tv` ensures stable and convergent updates across nodes.
