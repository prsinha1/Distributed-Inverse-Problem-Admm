# block_5_node_problem.py
import numpy as np
import cvxpy as cp
from block_4_tv_helpers import isotropic_tv_on_vector

def build_node_problem(Ai, bi, rho, neighbor_terms, N, lam_tv, Qij_terms):
    """
    Ai: dense matrix shape (m_i, n)
    bi: vector shape (m_i,)
    neighbor_terms: list of vectors v_ij = z_ij - y_ij,i for each neighbor j
    Qij_terms: list of q_ij diagonal vectors shape (n,) to weight the squared norms

    Decision variable xi in R^n
    Objective: 0.5*||Ai xi - bi||_2^2 + lam_tv * TV(xi)
               + (rho/2) * sum_j || xi - v_ij ||_{Qij}^2
    where ||u||_{Q}^2 = sum_k Q[k] * u[k]^2 with Q diagonal as vector.
    """
    n = Ai.shape[1]
    xi = cp.Variable(n)

    data_fit = 0.5 * cp.sum_squares(Ai @ xi - bi)
    tv_term = lam_tv * isotropic_tv_on_vector(xi, N)

    quad_terms = 0
    for v_ij, q_diag in zip(neighbor_terms, Qij_terms):
        diff = xi - v_ij
        quad_terms += 0.5 * rho * cp.sum(cp.multiply(q_diag, cp.square(diff)))

    obj = data_fit + tv_term + quad_terms
    prob = cp.Problem(cp.Minimize(obj))

    return xi, prob
