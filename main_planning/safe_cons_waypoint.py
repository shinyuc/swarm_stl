import numpy as np
import control as ct
import math
from gurobipy import *

class SingleSwarm:
    def __init__(self, id, dynamics, total_num_agents, lqr_Q_R):
        self.id = id
        self.num_agents = total_num_agents[id]
        self.dynamics = dynamics
        self.lqr_Q_R = lqr_Q_R  #[Q,R] matrix
        self.B = cholesky_decomposition_lqr(dynamics, lqr_Q_R)
        # self.bounding_box_A = bounding_box_A
        # self.bounding_box_sigma = np.matmul(bounding_box_A, bounding_box_A.T)

def _sub(x1, x2):
    return [x1[i] - x2[i] for i in range(len(x1))]

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
    # pos_diff = (end_pos - init_pos)
    pos_diff = np.array([end_pos[0]-init_pos[0], end_pos[1]-init_pos[1]])
    A = swarm.dynamics[0]
    B = swarm.dynamics[1]
    Q = swarm.lqr_Q_R[0]
    R = swarm.lqr_Q_R[1]
    K, P, E = ct.lqr(A, B, Q, R)
    beta = np.dot(pos_diff.T, np.dot(P, pos_diff))
    lambda_max_p = max(np.linalg.eigvals(P))
    overline_beta = (math.sqrt(lambda_max_p) * epsilon +1)**2 * beta
    x0 = end_pos
    return x0, P, overline_beta

def stl_waypoints(margin, model, swarm, a,b,x0,x1, Sigma, EPS=1e-1):
    bloat_factor = Sigma @ Sigma.T
    gamma_4 = - math.sqrt(a @ bloat_factor @ a.T)
    B_p = swarm.B
    B_p_T = B_p.T
    temp = B_p @ B_p.T *1.1
    inv_P = np.linalg.inv(temp)
    gamma_5 = math.sqrt(a @ inv_P @ a.T)
    gamma_5_1 = (B_p_T @ _sub(x0, x1))
    result = - sum([a[k] * x1[k] for k in range(len(x1))]) + b + gamma_4 - gamma_5 * L1Norm(model, gamma_5_1) -margin
    return result

def stl_last(margin, swarm, a, b, x0,Sigma,Sigma_last, EPS=1e-1):
    # bloat_factor denotes the margin, which is epsilon in eq(3)
    _, S, _ = np.linalg.svd(Sigma)
    max_singular_value = np.max(S)
    gamma_6 = b - max_singular_value * math.sqrt(a @a.T)
    epsilon_r = 0.001 # bounding box size max{Sigma_1, Sigma_2}
    result1 = - sum([a[k] * x0[k] for k in range(len(x0))]) + gamma_6 - epsilon_r * math.sqrt(a @a.T) - margin
    _, S_last, _ = np.linalg.svd(Sigma_last)
    max_singular_value_last = np.max(S_last)
    gamma_6_last = b - max_singular_value_last * math.sqrt(a @ a.T)
    result2 = - sum([a[k] * x0[k] for k in range(len(x0))]) + gamma_6_last - epsilon_r * math.sqrt(a @ a.T) - margin
    result = [result1, result2]
    return result

def stl_next(margin, swarm, a, b, x1, Sigma,Sigma_next, EPS=1e-1):
    # bloat_factor denotes the margin, which is epsilon in eq(3)
    _, S, _ = np.linalg.svd(Sigma)
    max_singular_value = np.max(S)
    gamma_6 = b - max_singular_value * math.sqrt(a @ a.T)
    epsilon_r = 0.001  # bounding box size max{Sigma_1, Sigma_2}
    result1 = - sum([a[k] * x1[k] for k in range(len(x1))]) + gamma_6 - epsilon_r * np.linalg.norm(a)-margin

    _, S_next, _ = np.linalg.svd(Sigma_next)
    max_singular_value_next = np.max(S_next)
    gamma_6_next = b - max_singular_value_next * math.sqrt(a @ a.T)
    result2 = - sum([a[k] * x1[k] for k in range(len(x1))]) + gamma_6_next - epsilon_r * math.sqrt(a @ a.T)-margin
    result = [result1, result2]

    return result

def neg_stl_waypoints(margin,model, swarm, a,b_e,x0,x1,Sigma=np.array([[0.001, 0], [0, 0.001]]), EPS=1e-2):
    # Sigma = np.array([[0.001, 0], [0, 0.001]])
    bloat_factor = Sigma @ Sigma.T
    gamma_4 =  - math.sqrt(a @ bloat_factor @ a.T)
    B_p = swarm.B
    B_p_T = B_p.T
    temp = B_p @ B_p.T *1.1
    inv_P = np.linalg.inv(temp)
    gamma_5 = math.sqrt(a @ inv_P @ a.T)
    gamma_5_1 = (B_p_T @ _sub(x0, x1))
    # result =  sum([a[k] * x1[k] for k in range(len(x1))]) - b_e + gamma_4 - gamma_5 * L1Norm(model, gamma_5_1) - margin
    result =  sum([a[k] * x1[k] for k in range(len(x1))])-b_e + gamma_4 - gamma_5 * L1Norm(model, gamma_5_1) - margin
    return result

def neg_stl_last(margin, swarm, a, b_e, x0, Sigma_last, Sigma, EPS=1e-2):
    # bloat_factor denotes the margin, which is epsilon in eq(3)
    _, S, _ = np.linalg.svd(Sigma)
    max_singular_value = np.max(S)
    gamma_6 = -b_e - max_singular_value * math.sqrt(a @ a.T)
    epsilon_r = 0.001  # bounding box size max{Sigma_1, Sigma_2}
    result1 =  sum([a[k] * x0[k] for k in range(len(x0))]) + gamma_6 - epsilon_r * math.sqrt(a @ a.T) - margin
    _, S_last, _ = np.linalg.svd(Sigma_last)
    max_singular_value_last = np.max(S_last)
    gamma_6_last = -b_e - max_singular_value_last * math.sqrt(a @ a.T)
    result2 =  sum([a[k] * x0[k] for k in range(len(x0))]) + gamma_6_last - epsilon_r * math.sqrt(a @ a.T) - margin
    result = [result1, result2]
    return result

def neg_stl_next(margin, swarm, a, b_e, x1, Sigma,Sigma_next, EPS=1e-2):
    # bloat_factor denotes the margin, which is epsilon in eq(3)
    _, S, _ = np.linalg.svd(Sigma)
    max_singular_value = np.max(S)
    gamma_6 = -b_e - max_singular_value * math.sqrt(a @ a.T)
    epsilon_r = 0.001  # bounding box size max{Sigma_1, Sigma_2}
    result1 =  sum([a[k] * x1[k] for k in range(len(x1))]) + gamma_6 - epsilon_r * np.linalg.norm(a) - margin

    _, S_next, _ = np.linalg.svd(Sigma_next)
    max_singular_value_next = np.max(S_next)
    gamma_6_next = -b_e - max_singular_value_next * math.sqrt(a @ a.T)
    result2 =  sum([a[k] * x1[k] for k in range(len(x1))]) + gamma_6_next - epsilon_r * math.sqrt(
        a @ a.T) - margin
    result = [result1, result2]
    return result
