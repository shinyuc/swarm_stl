import numpy as np
import control as ct
import math

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

def safe_2_1(swarm,a_q,b_q,epsilon=1e-1):
    p_s_i = swarm.end_pos
    sigma = swarm.bounding_box_sigma
    P, overline_beta = positive_invariant_set(swarm, epsilon=1e-1)
    inv_P  = np.linalg.inv(P)
    temp_1 = a_q.T @sigma @ a_q
    temp_2 = (overline_beta*a_q.T) @ inv_P @a_q
    temp_2 = temp_2.item()
    result_LHS = a_q.T @ p_s_i + b_q - math.sqrt(temp_1)-math.sqrt(temp_2)
    return result_LHS

def safe_2_2(swarm_s_i,swarm_s_i_last, a_q,b_q,epsilon=1e-1, epsilon_r = 1):
    p_s_i_1 = swarm_s_i.init_pos
    temp_1 = math.sqrt(a_q.T @ a_q)
    lambda_max_1 = max(np.linalg.eigvals(swarm_s_i.bounding_box_sigma))
    lambda_max_2 = max(np.linalg.eigvals(swarm_s_i_last.bounding_box_sigma))
    Psi = max(math.sqrt(lambda_max_1), math.sqrt(lambda_max_2))
    result_LHS = a_q.T@p_s_i_1 + b_q - Psi*temp_1 - epsilon_r*temp_1
    return result_LHS

def safe_2_3(swarm_s_i,swarm_s_i_next, a_q,b_q,epsilon=1e-1, epsilon_r = 1):
    p_s_i = swarm_s_i.end_pos
    temp_1 = math.sqrt(a_q.T @ a_q)
    lambda_max_1 = max(np.linalg.eigvals(swarm_s_i.bounding_box_sigma))
    lambda_max_2 = max(np.linalg.eigvals(swarm_s_i_next.bounding_box_sigma))
    Psi = max(math.sqrt(lambda_max_1), math.sqrt(lambda_max_2))
    result_LHS = a_q.T@p_s_i + b_q - Psi*temp_1 - epsilon_r*temp_1
    return result_LHS

def stl_1(i, swarm, PWL, F_j, g_j, epsilon=1e-1):
    B = swarm.B
    x = PWL[i][0] # the position of i-th time step
    F_j_T = F_j.T
    bounding_box_A = PWL[i][2]
    bounding_box_sigma = np.matmul(bounding_box_A, bounding_box_A.T)
    term_1 = F_j @ x + g_j
    term_2 = math.sqrt(F_j @ bounding_box_sigma @ F_j.T)
    term_3 = math.sqrt(F_j @ np.linalg.inv(PWL[i][1]) @ F_j.T * PWL[i][3])
    result_LHS = term_1 - term_2 - term_3
    return result_LHS

def stl_1(swarm_s_i, F_q_T, g_q, epsilon=1e-1):
    # note the difference is we need a .T for F_q compared with a_q
    F_q = F_q_T.T
    p_s_i = swarm_s_i.end_pos
    sigma = swarm_s_i.bounding_box_sigma
    P, overline_beta = positive_invariant_set(swarm_s_i, epsilon=1e-1)
    inv_P  = np.linalg.inv(P)
    temp_1 = F_q @ p_s_i + g_q
    temp_2 = math.sqrt(F_q @ sigma @ F_q.T)
    temp_3 = math.sqrt(F_q @ inv_P @ F_q.T * overline_beta)
    result_LHS = temp_1 - temp_2 - temp_3
    return result_LHS

def stl_2(swarm_s_i,swarm_s_i_last, F_q_T, g_q,epsilon=1e-1, epsilon_r = 1):
    F_q = F_q_T.T
    p_s_i_1 = swarm_s_i.init_pos
    temp_1 = math.sqrt(F_q.T @ F_q)
    lambda_max_1 = max(np.linalg.eigvals(swarm_s_i.bounding_box_sigma))
    lambda_max_2 = max(np.linalg.eigvals(swarm_s_i_last.bounding_box_sigma))
    Psi = max(math.sqrt(lambda_max_1), math.sqrt(lambda_max_2))
    result_LHS = F_q @p_s_i_1 + g_q - Psi*temp_1 - epsilon_r*temp_1
    return result_LHS

def stl_3(swarm_s_i,swarm_s_i_next, F_q_T, g_q,epsilon=1e-1, epsilon_r = 1):
    F_q = F_q_T.T
    p_s_i = swarm_s_i.end_pos
    temp_1 = math.sqrt(F_q.T @ F_q)
    lambda_max_1 = max(np.linalg.eigvals(swarm_s_i.bounding_box_sigma))
    lambda_max_2 = max(np.linalg.eigvals(swarm_s_i_next.bounding_box_sigma))
    Psi = max(math.sqrt(lambda_max_1), math.sqrt(lambda_max_2))
    result_LHS = F_q.T@p_s_i + g_q - Psi*temp_1 - epsilon_r*temp_1
    return result_LHS

def stl_neg_1(swarm_s_i, F_q_T, g_q, epsilon=1e-1):
    # note the difference is we need a .T for F_q compared with a_q
    F_q = F_q_T.T
    p_s_i = swarm_s_i.end_pos
    sigma = swarm_s_i.bounding_box_sigma
    P, overline_beta = positive_invariant_set(swarm_s_i, epsilon=1e-1)
    inv_P  = np.linalg.inv(P)
    temp_1 = F_q @ p_s_i + g_q
    temp_2 = math.sqrt(F_q @ sigma @ F_q.T)
    temp_3 = math.sqrt(F_q @ inv_P @ F_q.T * overline_beta)
    result_LHS = -temp_1 - temp_2 - temp_3
    return result_LHS

def stl_neg_2(swarm_s_i,swarm_s_i_last, F_q_T, g_q,epsilon=1e-1, epsilon_r = 1):
    F_q = F_q_T.T
    p_s_i_1 = swarm_s_i.init_pos
    temp_1 = math.sqrt(F_q.T @ F_q)
    lambda_max_1 = max(np.linalg.eigvals(swarm_s_i.bounding_box_sigma))
    lambda_max_2 = max(np.linalg.eigvals(swarm_s_i_last.bounding_box_sigma))
    Psi = max(math.sqrt(lambda_max_1), math.sqrt(lambda_max_2))
    result_LHS = -(F_q @p_s_i_1 + g_q) - Psi*temp_1 - epsilon_r*temp_1
    return result_LHS

def stl_neg_3(swarm_s_i,swarm_s_i_next, F_q_T, g_q,epsilon=1e-1, epsilon_r = 1):
    F_q = F_q_T.T
    p_s_i = swarm_s_i.end_pos
    temp_1 = math.sqrt(F_q.T @ F_q)
    lambda_max_1 = max(np.linalg.eigvals(swarm_s_i.bounding_box_sigma))
    lambda_max_2 = max(np.linalg.eigvals(swarm_s_i_next.bounding_box_sigma))
    Psi = max(math.sqrt(lambda_max_1), math.sqrt(lambda_max_2))
    result_LHS = -(F_q.T@p_s_i + g_q) - Psi*temp_1 - epsilon_r*temp_1
    return result_LHS