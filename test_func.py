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

if __name__ == '__main__':
    a = 2
    b = 3
    c = 4
    d = 5
    print(noIntersection(a, b, c, d))