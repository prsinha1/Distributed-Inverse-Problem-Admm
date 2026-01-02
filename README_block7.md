# 1. Block 7 Orchestrator, run and compare graph strategies (`block_7_main_ver3.py`)

## 1.1 Purpose and outline

This script loads tomography data, builds a pixel connected union graph and masked precision providers, then runs the decentralized ADMM loop on one or more graph strategies. Plots and arrays of residuals and errors are saved, and per node reconstructions are exported as images and arrays. fileciteturn4file0

Outline of operations:
1. Load ODL based operators and dense matrices, generate a phantom and sinograms, create output folders.  
2. Build pixel level connectivity and masked diagonal weights, obtain the union node graph from Block 3.  
3. Run the decentralized ADMM loop from Block 6 on the chosen strategy, while saving snapshots.  
4. Save reconstructions and histories, generate plots for objective values, residuals, and MSE curves. fileciteturn4file0

---

## 2. Key entry points

### 2.1 `run_one_strategy(strategy, k, data, N, lam_tv, rho, max_iters, eps_pri, eps_dual, base_dir, out_root, show_plots=False, verbose=True, snapshot_div=5, phantom_true=None)`

Runs a full pipeline for a single graph strategy, then saves reconstructions and all diagnostic figures. Returns the list of reconstructed vectors and a history dictionary. fileciteturn4file0

### 2.2 `main()`

Defines defaults, calls the Block 2 loader, selects a strategy, then calls `run_one_strategy`. The example in the file enables kNN with `k=2`. Other strategies are present as commented lines and can be enabled. fileciteturn4file0

---

## 3. Inputs, what they mean, and how to set them

The script accepts inputs through the `main` defaults or through explicit arguments to `run_one_strategy`.

| Name | Where set | Type and shape | Meaning and guidance |
|------|-----------|----------------|----------------------|
| `N` | `main` | int | Image side length; the reconstruction vector has length `N*N`. Typical values are 64 or 128. Larger values increase computation. fileciteturn4file0 |
| `num_nodes` | `main` then passed to Block 2 | int | Number of local operators and sinograms. Must match the saved operators in `base_dir`. fileciteturn4file0 |
| `lam_tv` | `main`, `run_one_strategy` | float | Weight on the isotropic TV penalty inside node problems. Increase to enforce smoother images. Default is 0.02. fileciteturn4file0 |
| `rho` | `main`, `run_one_strategy` | float | ADMM penalty parameter used both in node penalties and in dual updates. Start with 1.0 and tune based on residual curves. fileciteturn4file0 |
| `max_iters` | `main`, `run_one_strategy` | int | Maximum number of outer ADMM iterations. Default is 200. More iterations improve accuracy with more time. fileciteturn4file0 |
| `eps_pri`, `eps_dual` | `main`, `run_one_strategy` | float | Stopping thresholds for primal and dual residual norms. Defaults are `1e-3`. Tight thresholds require more iterations. fileciteturn4file0 |
| `noise_level` | `main` then used in Block 2 | float | Standard deviation used when creating noisy sinograms. Default is 0.005. fileciteturn4file0 |
| `base_dir` | `main`, `run_one_strategy` | str | Folder with pickled operators and matrices created by earlier blocks, for example `saved_operators_Incmp_Span`. fileciteturn4file0 |
| `strategy` | `run_one_strategy` | str | Graph construction mode. One of `knn`, `mst`, `chain`. For kNN the neighbor count `k` is also required. fileciteturn4file0 |
| `k` | `run_one_strategy` | int | Number of neighbors for kNN. Ignored for MST and chain. Start with 2. fileciteturn4file0 |
| `show_plots` | `run_one_strategy` | bool | If `True`, union graph plots from Block 3 are displayed. When running non interactively, keep `False`. fileciteturn4file0 |
| `verbose` | `run_one_strategy` | bool | Controls printing inside Block 3 and Block 6. Keep `True` while integrating, lower once stable. fileciteturn4file0 |
| `snapshot_div` | `main`, `run_one_strategy` | int | Controls snapshot frequency as `snap_every = max(1, max_iters // snapshot_div)`. A smaller value means more frequent snapshots. Default is 3 in `main`. fileciteturn4file0 |
| `phantom_true` | `run_one_strategy` | ndarray or None | If provided, image MSE curves are computed. Pass the phantom from Block 2 as in `phantom_true = data.get("phantom", None)`. fileciteturn4file0 |
| `out_root` | `run_one_strategy` | str | Root output folder created by `main`, for example `Recon_Out_ADMM_YYYYMMDD_HHMMSS`. A subfolder is made per strategy. fileciteturn4file0 |
| `data` | `run_one_strategy` | dict | Object returned by Block 2 loader that contains `A_dense_list`, `sinograms`, and `phantom`. fileciteturn4file0 |

Notes on strategy selection and masks:  
`build_pixel_connected_Q_provider` constructs per pixel adjacency masks and masked diagonal weights, then returns the union node graph `G_union`. The ADMM loop runs on this union graph, not on the full all to all graph. fileciteturn4file0

---

## 4. What happens inside `run_one_strategy`

1. A tag is created per strategy, for example `knn_k2`. A subfolder is created inside `out_root`.  
2. Block 3 is invoked to construct the union graph and the masked precision provider. Figures showing the union graph and degree distributions are saved under `out_root/tag/union_figs`.  
3. The decentralized ADMM loop from Block 6 is called with all inputs. A snapshot folder is created and a snapshot period is chosen as described above.  
4. After ADMM returns, several plots are created and saved. Objective per node, total objective, per node primal residual, and per node dual residual are saved as `.png` and `.npy`.  
5. Reconstructions for each node are saved as `.npy` arrays and `.png` images.  
6. Sinogram MSE per node and total are computed and saved as `.png` and `.npy`. If a phantom was provided, image MSE per node and total are also saved in the same way. fileciteturn4file0

---

## 5. Outputs, file names, and where they are saved

Let `out_root = Recon_Out_ADMM_YYYYMMDD_HHMMSS`, and let `tag` be `mst`, `chain`, or `knn_kK`.

Under `out_root/tag` the following are written:

- Reconstruction images and arrays, one per node  
  `tag_node_i.png`, `tag_node_i.npy` created by `save_recons`.  
- Union graph figures from Block 3  
  `union_figs/*.png`.  
- Objective per node across iterations  
  `tag_obj_per_node.png`, `tag_obj_per_node.npy`.  
- Total objective across iterations  
  `tag_obj_total.png`, `tag_obj_total.npy`.  
- Per node primal residual  
  `tag_pri_per_node.png`, `tag_pri_per_node.npy`.  
- Per node dual residual  
  `tag_dual_per_node.png`, `tag_dual_per_node.npy`.  
- Global residual curves across iterations  
  `tag_residuals.png`, plus raw histories  
  `tag_primal_hist.npy`, `tag_dual_hist.npy`.  
- Sinogram MSE per node and total  
  `tag_sino_mse_per_node.png`, `tag_sino_mse_per_node.npy`, `tag_sino_mse_total.png`, `tag_sino_mse_total.npy`.  
- Image MSE per node and total, only when a phantom is supplied  
  `tag_img_mse_per_node.png`, `tag_img_mse_per_node.npy`, `tag_img_mse_total.png`, `tag_img_mse_total.npy`.  
- Snapshots from ADMM outer iterations  
  `snapshots/iter_t_node_i.npy`, `snapshots/iter_t_node_i.png` with `t` equal to the snapshot iteration index. fileciteturn4file0

---

## 6. Example usage

### 6.1 Run the default main entry point

```bash
python block_7_main_ver3.py
```
This creates a time stamped folder named like `Recon_Out_ADMM_20251103_153012`, then runs kNN with `k=2`, and writes all outputs to `Recon_Out_ADMM_.../knn_k2`. To enable MST or chain, uncomment the corresponding calls in `main`. fileciteturn4file0

### 6.2 Call `run_one_strategy` from another script or a notebook

```python
from block_2_load_odl_data import load_odl_data
from block_7_main_ver3 import run_one_strategy

data = load_odl_data(base_dir="saved_operators_Incmp_Span", N=64, num_nodes=5, noise_level=0.005)
phantom_true = data.get("phantom", None)
x_list, hist = run_one_strategy(
    strategy="knn", k=2, data=data, N=64, lam_tv=0.02, rho=1.0,
    max_iters=200, eps_pri=1e-3, eps_dual=1e-3,
    base_dir="saved_operators_Incmp_Span", out_root="Recon_Out_ADMM_test",
    show_plots=False, verbose=True, snapshot_div=3, phantom_true=phantom_true
)
```

---

## 7. Dependencies

- Block 2 loader, `load_odl_data`, provides `A_dense_list`, `sinograms`, and `phantom`.  
- Block 3 graph and precision builder, `build_pixel_connected_Q_provider`, provides the union graph, masked diagonal weights, and summaries.  
- Block 6 decentralized ADMM loop, `decentralized_admm`, solves the node problems with CVXPY and SCS, and produces the histories used in the plots.  
- NumPy, Matplotlib, NetworkX, and the Python standard library for folders and timestamps. fileciteturn4file0

---

## 8. Practical notes

- Ensure that the operator pickles exist in `base_dir` before running this script.  
- Snapshot frequency should balance storage and insight; a good initial choice is `snapshots` every `max_iters // 3`.  
- When memory is limited, disable `show_plots` and avoid large `N`.  
- When comparing strategies, run the same `N`, `rho`, `lam_tv`, and `max_iters` so the curves are directly comparable. fileciteturn4file0
