# snapvx_graph_admm.py
import numpy as np
import networkx as nx
from snapvx import TGraphVX
import cvxpy as cp

def snapvx_demo_random(Nnodes=4, Nside=64, m_per_node=None, seed=0):
    rng = np.random.default_rng(seed)
    n = Nside * Nside
    if m_per_node is None:
        m_per_node = n // 2

    # ground truth and sensing
    x_true = rng.normal(size=n)
    A_list = []
    b_list = []
    for _ in range(Nnodes):
        A = rng.normal(scale=1.0/np.sqrt(n), size=(m_per_node, n))
        b = A @ x_true + 0.01 * rng.normal(size=m_per_node)
        A_list.append(A)
        b_list.append(b)

    # node precisions W_i and edge precisions Q_ij
    W = [0.01*np.eye(n) for _ in range(Nnodes)]
    def Q_block(): return 0.5*np.eye(n)

    # graph
    G = nx.Graph()
    G.add_nodes_from(range(Nnodes))
    for i in range(Nnodes-1):
        G.add_edge(i, i+1)
    G.add_edge(0, Nnodes-1)

    # build SnapVX problem
    prob = TGraphVX()
    x_vars = {}
    for i in range(Nnodes):
        x_i = cp.Variable(n)
        A = A_list[i]
        b = b_list[i]
        Wi = W[i]
        # node objective: 0.5||A x - b||^2 + 0.5||x||_{Wi}^2
        f_node = 0.5*cp.sum_squares(A @ x_i - b) + 0.5*cp.quad_form(x_i, Wi)
        prob.AddNode(i, Objective=f_node, Variables=[x_i])
        x_vars[i] = x_i

    for (i, j) in G.edges():
        Qij = Q_block()
        x_i = x_vars[i]
        x_j = x_vars[j]
        # edge objective: 0.5 (x_i - x_j)^T Q_ij (x_i - x_j)
        f_edge = 0.5*cp.quad_form(x_i - x_j, Qij)
        prob.AddEdge(i, j, Objective=f_edge)

    # solve by ADMM inside SnapVX
    prob.Solve(UseADMM=True, MaxIters=50, Rho=1.0, Quiet=False)

    X = np.vstack([x_vars[i].value for i in range(Nnodes)])
    mse = np.mean([np.mean((X[i] - x_true)**2) for i in range(Nnodes)])
    print("SnapVX MSE", mse)
    return X
