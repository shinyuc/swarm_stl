import numpy as np
import control as ct
import math
from gurobipy import *
# from pwl_plan import *
# PWL includes waypoint position, time stamp, bounding box A
# PWLs = [PWL]
class SingleSwarm:
    def __init__(self, id, dynamics, total_num_agents, lqr_Q_R):
        self.id = id
        self.num_agents = total_num_agents[id]
        self.dynamics = dynamics
        self.lqr_Q_R = lqr_Q_R  #[Q,R] matrix
        self.B = cholesky_decomposition_lqr(dynamics, lqr_Q_R)
        # self.bounding_box_A = bounding_box_A
        # self.bounding_box_sigma = np.matmul(bounding_box_A, bounding_box_A.T)

def L1Norm(model, x):
    xvar = model.addVars(len(x), lb=-GRB.INFINITY)
    abs_x = model.addVars(len(x))
    model.update()
    xvar = [xvar[i] for i in range(len(xvar))]
    abs_x = [abs_x[i] for i in range(len(abs_x))]
    for i in range(len(x)):
        model.addConstr(xvar[i] == x[i])
        model.addConstr(abs_x[i] == abs_(xvar[i]))
    return sum(abs_x)

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

def positive_invariant_set(swarm, i, PWL, epsilon=1e-1):
    # segment from i to i+1
    init_pos = PWL[i][0]
    end_pos = PWL[i+1][0]
    pos_diff = (end_pos - init_pos)
    A = swarm.dynamics[0]
    B = swarm.dynamics[1]
    Q = swarm.lqr_Q_R[0]
    R = swarm.lqr_Q_R[1]
    K, P, E = ct.lqr(A, B, Q, R)
    beta = np.dot(pos_diff.T, np.dot(P, pos_diff))
    lambda_max_p = max(np.linalg.eigvals(P))
    overline_beta = (math.sqrt(lambda_max_p) * epsilon +1)**2 * beta
    return P, overline_beta

def stl_1(swarm, a,b,x0,x1,e,bloat_factor, EPS=1e-1):
    # bloat_factor denotes the margin, which is epsilon in eq(3)
    Sigma = np.array([[0.001, 0], [0, 0.001]])
    gamma_4 = b[e] - math.sqrt(a@ Sigma @a.T)
    B_p = swarm.B
    B_p_T = B_p.T
    inv_P = np.linalg.inv(B_p @ B_p.T)
    gamma_5 = math.sqrt(a@ inv_P @a.T)
    v_1 = np.array([1, 1]) @ B_p_T
    v_2 = np.array([1, -1]) @ B_p_T
    v_3 = np.array([-1, 1]) @ B_p_T
    v_4 = np.array([-1, -1]) @ B_p_T
    gamma_5_1 = gamma_5 * sum([v_1[k] * (x0[k] - x1[k]) for k in range(len(x1))])
    gamma_5_2 = gamma_5 * sum([v_2[k] * (x0[k] - x1[k]) for k in range(len(x1))])
    gamma_5_3 = gamma_5 * sum([v_3[k] * (x0[k] - x1[k]) for k in range(len(x1))])
    gamma_5_4 = gamma_5 * sum([v_4[k] * (x0[k] - x1[k]) for k in range(len(x1))])
    # result = - sum([a[k] * x[k] for k in range(len(x))]) + gamma_4 - np.linalg.norm(a) * bloat_factor - EPS
    result_1 = - sum([a[k] * x1[k] for k in range(len(x1))]) + gamma_4 - gamma_5_1 - EPS
    result_2 = - sum([a[k] * x1[k] for k in range(len(x1))]) + gamma_4 - gamma_5_2 - EPS
    result_3 = - sum([a[k] * x1[k] for k in range(len(x1))]) + gamma_4 - gamma_5_3 - EPS
    result_4 = - sum([a[k] * x1[k] for k in range(len(x1))]) + gamma_4 - gamma_5_4 - EPS
    result = [result_1, result_2, result_3, result_4]
    return result

def stl_v2(model, swarm, a,b,x0,x1,e,bloat_factor, EPS=1e-1):
    # bloat_factor denotes the margin, which is epsilon in eq(3)
    Sigma = np.array([[0.001, 0], [0, 0.001]])
    gamma_4 = b[int(e)] - math.sqrt(a.T @ Sigma @ a)
    B_p = swarm.B
    B_p_T = B_p.T
    inv_P = np.linalg.inv(B_p @ B_p.T)
    gamma_5 = math.sqrt(a.T @ inv_P @ a)
    gamma_5_1 = gamma_5 * sum([(x0[k] - x1[k]) for k in range(len(x1))])
    result = - sum([a[k] * x1[k] for k in range(len(x1))]) + gamma_4 - L1Norm(model, gamma_5_1) - EPS
    return result

def stl_last(swarm, a, b, x0, x0_last, e, bloat_factor, EPS=1e-1):
    # bloat_factor denotes the margin, which is epsilon in eq(3)
    gamma_6 = b[e] - bloat_factor * math.sqrt(a @a.T)
    gamma_7 = 0.001 # bounding box size max{Sigma_1, Sigma_2}
    result = - sum([a[k] * x0[k] for k in range(len(x0))]) + gamma_6 - gamma_7* np.linalg.norm(a)- EPS
    return result

def stl_next(swarm, a, b, x1, x1_next, e, bloat_factor, EPS=1e-1):
    # bloat_factor denotes the margin, which is epsilon in eq(3)
    gamma_6 = b[e] - bloat_factor * math.sqrt(a @ a.T)
    gamma_7 = 0.001  # bounding box size max{Sigma_1, Sigma_2}
    result = - sum([a[k] * x1[k] for k in range(len(x1))]) + gamma_6 - gamma_7 * np.linalg.norm(a) - EPS
    return result

def neg_stl_1(swarm, a,b,x0,x1,e,bloat_factor, EPS=1e-1):
    # bloat_factor denotes the margin, which is epsilon in eq(3)
    Sigma = np.array([[0.001, 0], [0, 0.001]])
    gamma_4 = -b[e] - math.sqrt(a @ Sigma @ a.T)
    B_p = swarm.B
    B_p_T = B_p.T
    inv_P = np.linalg.inv(B_p @ B_p.T)
    gamma_5 = math.sqrt(a @ inv_P @ a.T)
    v_1 = np.array([1, 1]) @ B_p_T
    v_2 = np.array([1, -1]) @ B_p_T
    v_3 = np.array([-1, 1]) @ B_p_T
    v_4 = np.array([-1, -1]) @ B_p_T
    gamma_5_1 = gamma_5 * sum([v_1[k] * (x0[k] - x1[k]) for k in range(len(x1))])
    gamma_5_2 = gamma_5 * sum([v_2[k] * (x0[k] - x1[k]) for k in range(len(x1))])
    gamma_5_3 = gamma_5 * sum([v_3[k] * (x0[k] - x1[k]) for k in range(len(x1))])
    gamma_5_4 = gamma_5 * sum([v_4[k] * (x0[k] - x1[k]) for k in range(len(x1))])
    # result = - sum([a[k] * x[k] for k in range(len(x))]) + gamma_4 - np.linalg.norm(a) * bloat_factor - EPS
    result_1 =  sum([a[k] * x1[k] for k in range(len(x1))]) + gamma_4 - gamma_5_1 - EPS
    result_2 =  sum([a[k] * x1[k] for k in range(len(x1))]) + gamma_4 - gamma_5_2 - EPS
    result_3 =  sum([a[k] * x1[k] for k in range(len(x1))]) + gamma_4 - gamma_5_3 - EPS
    result_4 =  sum([a[k] * x1[k] for k in range(len(x1))]) + gamma_4 - gamma_5_4 - EPS
    result = [result_1, result_2, result_3, result_4]
    return result

def neg_stl_last(swarm, a, b, x0, x0_last, e, bloat_factor, EPS=1e-1):
    # bloat_factor denotes the margin, which is epsilon in eq(3)
    gamma_6 = -b[e] - bloat_factor * math.sqrt(a @a.T)
    gamma_7 = 0.001 # bounding box size max{Sigma_1, Sigma_2}
    result = sum([a[k] * x0[k] for k in range(len(x0))]) + gamma_6 - gamma_7* np.linalg.norm(a)- EPS
    return result

def neg_stl_next(swarm, a, b, x1, x1_next, e, bloat_factor, EPS=1e-1):
    # bloat_factor denotes the margin, which is epsilon in eq(3)
    gamma_6 = -b[e] - bloat_factor * math.sqrt(a @ a.T)
    gamma_7 = 0.001  # bounding box size max{Sigma_1, Sigma_2}
    result = sum([a[k] * x1[k] for k in range(len(x1))]) + gamma_6 - gamma_7 * np.linalg.norm(a) - EPS
    return result
