from gurobipy import *

import control as ct
import numpy as np
import math
from waypoint_plan import *
from ellipsoid_plt import plot_ellipsoid_inv
import matplotlib.pyplot as plt
from bounding_box_plan import test_disjoint_segments
import cvxpy as cp

class SingleSwarm:
    def __init__(self, id, dynamics, total_num_agents, lqr_Q_R):
        self.id = id
        self.num_agents = total_num_agents[id]
        self.dynamics = dynamics
        self.lqr_Q_R = lqr_Q_R  #[Q,R] matrix
        self.B = cholesky_decomposition_lqr(dynamics, lqr_Q_R)
        # self.bounding_box_A = bounding_box_A
        # self.bounding_box_sigma = np.matmul(bounding_box_A, bounding_box_A.T)

def cholesky_decomposition_lqr(dynamics, lqr_Q_R = (np.eye(2), np.eye(2))):
    A = dynamics[0]
    B = dynamics[1]
    Q = lqr_Q_R[0]
    R = lqr_Q_R[1]
    _, P, _ = ct.lqr(A, B, Q, R)
    try:
        B_p = np.linalg.cholesky(P)
        return B_p
    except np.linalg.LinAlgError:
        print("Matrix P is not positive definite.")

def _sub(x1, x2):
    return [x1[i] - x2[i] for i in range(len(x1))]

def check_disjoint_segments(swarm, seg1, seg2, box1, box2, margin):
    # inter agent collision avoidance
    epsilon_r = 1e-3
    assert(len(seg1) == 2)
    assert(len(seg2) == 2)
    # assuming that bloat is the error bound in two norm for one agent
    # return 0.5 * L1Norm(model, _sub(_add(seg1[0], seg1[1]), _add(seg2[0], seg2[1]))) - 0.5 * (L1Norm(model, _sub(seg1[0], seg1[1])) + L1Norm(model, _sub(seg2[0], seg2[1]))) - 2*bloat*np.sqrt(len(seg1[0])) - EPS
    B_p = swarm.B
    p = B_p @ B_p.T
    max_eigenvalue_p = np.max(np.linalg.eigvals(p))
    gamma_5_bloat = math.sqrt(max_eigenvalue_p) * epsilon_r
    # overline_beta_1 =  L1Norm(model, (B_p.T @ _sub(seg1[1], seg1[0])))
    overline_beta_1 = np.linalg.norm(B_p.T @ (seg1[1]- seg1[0]), ord = 1)
    # overline_beta_2 =  L1Norm(model, (B_p.T @ _sub(seg2[1], seg2[0])))
    overline_beta_2 = np.linalg.norm(B_p.T @ (seg2[1] - seg2[0]), ord=1)
    eigenvalues = np.linalg.eigvals(p)
    min_eigenvalue = np.min(eigenvalues)
    # dist_bar = (1/math.sqrt(2))*L1Norm(model, _sub(seg1[1], seg2[1])) - (gamma_5_bloat + overline_beta_1)/math.sqrt(min_eigenvalue)-(gamma_5_bloat + overline_beta_2)/math.sqrt(min_eigenvalue)
    dist_bar = (1 / math.sqrt(2)) * np.linalg.norm(seg1[1] - seg2[1], ord = 1) - (
                gamma_5_bloat + overline_beta_1) / math.sqrt(min_eigenvalue) - (
                           gamma_5_bloat + overline_beta_2) / math.sqrt(min_eigenvalue)
    max_singular_value_1 = np.linalg.norm(box1, ord=2)
    max_singular_value_2 = np.linalg.norm(box2, ord=2)
    result = dist_bar - max_singular_value_1 - max_singular_value_2 - margin
    return result

def plot_figure():
    # Define points
    x11 = [0, -2]
    x12 = [0, 0]
    x21 = [12, -3]
    x22 = [12, 0]

    # Create plot
    plt.figure(figsize=(6, 6))
    plt.scatter([x11[0], x12[0], x21[0], x22[0]],
                [x11[1], x12[1], x21[1], x22[1]],
                color='red', label="Points")

    # Draw lines connecting the points
    plt.plot([x11[0], x12[0]], [x11[1], x12[1]], 'b-', label="Line 1")
    plt.plot([x21[0], x22[0]], [x21[1], x22[1]], 'g-', label="Line 2")

    # Label points
    plt.text(x11[0], x11[1], 'x11', fontsize=12, verticalalignment='bottom', horizontalalignment='right')
    plt.text(x12[0], x12[1], 'x12', fontsize=12, verticalalignment='bottom', horizontalalignment='right')
    plt.text(x21[0], x21[1], 'x21', fontsize=12, verticalalignment='bottom', horizontalalignment='left')
    plt.text(x22[0], x22[1], 'x22', fontsize=12, verticalalignment='bottom', horizontalalignment='left')

    # Formatting plot
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    # plt.title("Line Connections Between Points")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True, linestyle='--', linewidth=0.5)
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    m = Model("xref")
    x11 = m.addVars(2,lb=-GRB.INFINITY)
    x12 = m.addVars(2,lb=-GRB.INFINITY)
    x21 = m.addVars(2,lb=-GRB.INFINITY)
    x22 = m.addVars(2,lb=-GRB.INFINITY)
    m.update()
    #
    m.addConstr(x11[0] == 0)
    m.addConstr(x11[1] == -2)
    m.addConstr(x12[0] == 0)
    m.addConstr(x12[1] == 0)

    #
    m.addConstr(x21[0] == 12)
    m.addConstr(x21[1] == -3)
    m.addConstr(x22[0] == 12)
    m.addConstr(x22[1] == 0)

    num_rounds = 20
    eta = 0

    tmax = 200

    A = np.array([[0, 0], [0, 0]])
    B = np.array([[1, 0], [0, 1]])
    dynamics = [A, B]
    total_num_agents = [2, 3, 4, 5]
    print(dynamics[1])
    print(total_num_agents[0])
    Q = np.array([[1, 0], [0, 1]])
    R = np.array([[1, 0], [0, 1]])
    lqr_set = [Q, R]
    swarm = SingleSwarm(1, dynamics, total_num_agents, lqr_set)
    box1 = np.array([[0.001, 0],[0, 0.001]])
    box2 = np.array([[0.001, 0],[0, 0.001]])
    margin = m.addVar(lb=-GRB.INFINITY)
    # m.addConstr(margin == 1)
    m.update()

    x11_data = np.array([0,-2])
    x12_data = np.array([0,0])
    x21_data = np.array([12, -3])
    x22_data = np.array([12, 0])
    z_disjoint_segments = disjoint_segments(m, swarm, [x11, x12], [x21, x22], box1, box2, margin)

    # z_disjoint_segments = check_disjoint_segments(swarm, [x11_data, x12_data], [x21_data, x22_data], box1, box2, margin)

    m.addConstr(z_disjoint_segments >=0)
    obj1 = LinExpr()
    obj1 += margin
    m.update()
    m.setObjective(obj1, GRB.MAXIMIZE)
    m.optimize()
    print("----------------")
    print(margin.X)

    B_p = swarm.B
    p = B_p @ B_p.T
    max_eigenvalue_p = np.max(np.linalg.eigvals(p))
    epsilon_r = 1e-3
    x11_result = np.array([x11[i].X for i in range(2)])
    x12_result = np.array([x12[i].X for i in range(2)])
    x21_result = np.array([x21[i].X for i in range(2)])
    x22_result = np.array([x22[i].X for i in range(2)])
    #
    gamma_5_bloat = math.sqrt(max_eigenvalue_p) * epsilon_r
    temp1 = B_p.T @ (x12_result - x11_result)
    overline_beta_1 = gamma_5_bloat + np.linalg.norm(temp1, ord=1)
    temp2 = B_p.T @ (x22_result - x21_result)
    overline_beta_2 = gamma_5_bloat + np.linalg.norm(temp2, ord=1)

    plot_figure()
    plot_ellipsoid_inv(p, overline_beta_1**2, x12_result)
    plot_ellipsoid_inv(p, overline_beta_2**2, x22_result)

    plt.show()
    constraints = []
    last_margin = margin.X
    new_margin = cp.Variable()
    new_box1 = cp.Variable((2, 2))
    new_box2 = cp.Variable((2, 2))
    test_disjoint_segments(constraints,swarm, [x11_result, x12_result], [x21_result, x22_result], box1, box2 ,new_box1,new_box2, last_margin, new_margin)
    print(constraints)