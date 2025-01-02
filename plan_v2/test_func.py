import numpy as np
import control as ct
from math import *

EPS = 1e-2


# A = np.array([[0, 0], [0, 0]])
# B = np.array([[1, 0], [0, 1]])
# C = np.array([[9, 10], [11, 12]])
# Q = np.matrix([[1, 0], [0, 1]])
# R = np.matrix([[1, 0], [0, 1]])
# result = A @ B @ C
# print(result)
#
# K, S, E = ct.lqr(A, B, Q, R)
# print(S)
# print(np.linalg.eigvals(S))
#
# a_q = np.array([1, 0])
# print(a_q)
# a_q = a_q.T
# print(a_q)

class Disjunction(object):
    # disjunction node
    def __init__(self, deps = []):
        super(Disjunction, self).__init__()
        self.deps = deps
        self.constraints = []

def noIntersection(a, b, c, d):
    # z = 1 iff. [a, b] and [c, d] has no intersection
    # b < c or d < a
    return Disjunction([c-b-EPS, a-d-EPS])

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

if __name__ == '__main__':
    a = 2
    b = 3
    c = 4
    d = 5
    print(noIntersection(a, b, c, d))
    A = np.array([[0, 0], [0, 0]])
    B = np.array([[1, 0], [0, 1]])
    dynamics = [A, B]
    total_num_agents = [2, 3, 4, 5]
    print(dynamics[1])
    print(total_num_agents[0])
    Q = 0.5*np.array([[1, 0], [0, 1]])
    R = 0.5*np.array([[1, 0], [0, 1]])
    lqr_set = [Q, R]
    swarm = SingleSwarm(1, dynamics, total_num_agents, lqr_set)
    print(swarm.B)