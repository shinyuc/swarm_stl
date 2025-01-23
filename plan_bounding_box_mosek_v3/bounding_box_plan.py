import pickle
import numpy as np
import cvxpy as cp
import math
import control as ct
import time


class SingleSwarm:
    def __init__(self, id, dynamics, total_num_agents, lqr_Q_R):
        self.id = id
        self.num_agents = total_num_agents[id]
        self.dynamics = dynamics
        self.lqr_Q_R = lqr_Q_R  #[Q,R] matrix
        self.B = cholesky_decomposition_lqr(dynamics, lqr_Q_R)

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
    return x1 - x2

def _add(x1, x2):
    return x1 + x2

def L1Norm(x):
    """Compute the L1 norm using cvxpy."""
    return cp.norm(x, 1)

def test_mu(constraints, last_margin, new_margin, swarm, i, PWL, Box, last_Box, test_list):
    print(i)
    epsilon_r = 1e-3
    for test_condition in test_list:
        test_A = test_condition[0]
        test_b = test_condition[1]
        b = test_b.reshape(-1)
        num_edges = len(b)
        last_Sigma = last_Box[i+1]
        B_p = swarm.B
        p = B_p @ B_p.T
        inv_P = np.linalg.inv(p)
        for e in range(num_edges):
            a = A[e, :]
            b_e = b[int(e)]
            x0 = np.array(PWL[i][0])
            x1 = np.array(PWL[i + 1][0])
            beta_bar = (x1 - x0).T @ p @ (x1 - x0)
            gamma_4 = math.sqrt(a @ inv_P @ a.T * beta_bar)
            gamma_5 = a @ last_Sigma
            result = - a @ x1 + b_e - gamma_4 - np.linalg.norm(gamma_5, ord=1) - last_margin
            new_sigma = Box[i + 1][0]
            if result >=0:
                new_gamma_5 = a @ new_sigma
                new_result = - a @ x1 + b_e - gamma_4 - L1Norm(new_gamma_5) - new_margin
                constraints += [new_result>=0]
            if 1<=i<= len(PWL)-2:
                _, S1, _ = np.linalg.svd(last_Sigma)
                temp1 = S1[0]
                result1 = - a @ x1  + b_e - temp1 * math.sqrt(a @ a.T) - epsilon_r * math.sqrt(a @ a.T) - last_margin
                if result1 >=0:
                    new_temp1 = cp.sigma_max(new_sigma)
                    new_result1 = - a @ x1  + b_e - new_temp1 * math.sqrt(a @ a.T) - epsilon_r * math.sqrt(a @ a.T) - new_margin
                    constraints += [new_result1 >= 0]

                second_Sigma = last_Box[i]
                _, S2, _ = np.linalg.svd(second_Sigma)
                temp2 = S2[0]
                result2 = - a @ x1 + b_e - temp2 * math.sqrt(a @ a.T) - epsilon_r * math.sqrt(a @ a.T) - last_margin

                new_last_sigma = Box[i][0]
                if result2 >=0:
                    new_temp2 = cp.sigma_max(new_last_sigma)
                    new_result2 = - a @ x1 + b_e - new_temp2 * math.sqrt(a @ a.T) - epsilon_r * math.sqrt(
                        a @ a.T) - new_margin
                    constraints += [new_result2 >= 0]


def test_negmu(constraints, last_margin, new_margin, swarm, i, PWL, Box, last_Box, test_list):
    print(i)
    epsilon_r = 1e-3
    for test_condition in test_list:
        test_A = test_condition[0]
        test_b = test_condition[1]
        b = test_b.reshape(-1)
        num_edges = len(b)
        last_Sigma = last_Box[i+1]
        B_p = swarm.B
        p = B_p @ B_p.T
        inv_P = np.linalg.inv(p)
        for e in range(num_edges):
            a = A[e, :]
            b_e = b[int(e)]
            x0 = np.array(PWL[i][0])
            x1 = np.array(PWL[i + 1][0])
            beta_bar = (x1 - x0).T @ p @ (x1 - x0)
            gamma_4 = math.sqrt(a @ inv_P @ a.T * beta_bar)
            gamma_5 = a @ last_Sigma
            result =  a @ x1 - b_e - gamma_4 - np.linalg.norm(gamma_5, ord=1) - last_margin
            new_sigma = Box[i + 1][0]
            if result >=0:
                new_gamma_5 = a @ new_sigma
                new_result =  a @ x1 - b_e - gamma_4 - L1Norm(new_gamma_5) - new_margin
                constraints += [new_result>=0]
            if 1<=i<= len(PWL)-2:
                _, S1, _ = np.linalg.svd(last_Sigma)
                temp1 = S1[0]
                result1 =  a @ x1  - b_e - temp1 * math.sqrt(a @ a.T) - epsilon_r * math.sqrt(a @ a.T) - last_margin
                if result1 >=0:
                    new_temp1 = cp.sigma_max(new_sigma)
                    new_result1 =  a @ x1  - b_e - new_temp1 * math.sqrt(a @ a.T) - epsilon_r * math.sqrt(a @ a.T) - new_margin
                    constraints += [new_result1 >= 0]

                second_Sigma = last_Box[i]
                _, S2, _ = np.linalg.svd(second_Sigma)
                temp2 = S2[0]
                result2 =  a @ x1 - b_e - temp2 * math.sqrt(a @ a.T) - epsilon_r * math.sqrt(a @ a.T) - last_margin

                new_last_sigma = Box[i][0]
                if result2 >=0:
                    new_temp2 = cp.sigma_max(new_last_sigma)
                    new_result2 =  a @ x1 - b_e - new_temp2 * math.sqrt(a @ a.T) - epsilon_r * math.sqrt(
                        a @ a.T) - new_margin
                    constraints += [new_result2 >= 0]

def test_volume_constraint(constraints, Box, i):
    new_sigma = Box[i+1][0]
    result = cp.log_det(new_sigma) - 0.001
    constraints += [result >= 0]


if __name__ == "__main__":
    A = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    b2 = np.array([0, 0.9, 1, -0.5], dtype=np.float64)
    B2 = (A, b2)
    b3 = np.array([-0.2, 0.7, -0.8, 1.2], dtype=np.float64)
    B3 = (A, b3)
    C = np.array([0.4, 0.4, 0.4, 0.4], dtype=np.float64)
    C = (A, C)
    # inside the wall
    D = np.array([1.5, 1.5, 1.5, 1.5], dtype=np.float64)
    # D = np.array([2, 2, 2, 2], dtype=np.float64)
    D = (A, D)
    test_list = [B2, B3, C, D]



    with open('PWLs.pkl', 'rb') as file:
        PWLs = pickle.load(file)

    print(PWLs)
    PWL = PWLs[0]
    print(PWL)

    with open('PWLs_output_time.pkl', 'rb') as file:
        PWLs_loaded, margin_loaded, Box_loaded = pickle.load(file)

    last_margin = margin_loaded
    last_Box = Box_loaded
    # last_Box = []
    # for i in range(len(PWL)):
    #     last_Box.append(np.array([[0.001, 0],[0, 0.001]]))


    sys_A = np.array([[0, 0], [0, 0]])
    sys_B = np.array([[1, 0], [0, 1]])
    sys_dynamics = [sys_A, sys_B]
    total_num_agents = [2, 3, 4, 5]
    print(sys_dynamics[1])
    print(total_num_agents[0])
    Q = np.array([[1, 0], [0, 1]])
    R = np.array([[1, 0], [0, 1]])
    lqr_set = [Q, R]
    swarm = SingleSwarm(1, sys_dynamics, total_num_agents, lqr_set)
    # box = [a[k] * x1[k] for i in range(len(x1))])
    num_segs = 19
    Boxes = []
    constraints = []
    new_margin = cp.Variable()
    Box = []
    for i in range(num_segs + 1):
        Box.append([cp.Variable((2, 2))])
    Boxes.append(Box)


    for i in range(num_segs):
        test_mu(constraints, last_margin, new_margin, swarm, i, PWL, Box, last_Box, test_list)
        test_negmu(constraints, last_margin, new_margin, swarm, i, PWL, Box, last_Box, test_list)
        test_volume_constraint(constraints, Box, i)
        # test_negmu()



    obj = new_margin
    prob = cp.Problem(cp.Maximize(obj),
                      constraints)
    start = time.time()
    prob.solve()
    end = time.time()
    print('sovling it takes %.3f s' % (end - start))
    Boxes_output = []
    for Box in Boxes:
        Box_output = []
        for B in Box:
            Box_output.append([B[i].value for i in range(len(B))])
        Boxes_output.append(Box_output)
    margin_result = new_margin.value

print(Boxes_output)
print(margin_result)