# import pickle
#
# with open('../plan_waypoints_v1/PWLs.pkl', 'rb') as file:
#     PWLs = pickle.load(file)
#
# print(PWLs)
# print(PWLs[0])
# print(PWLs[0][0])
import cvxpy as cp
import numpy as np

from gurobi_mosek_comparsion.test1 import constraints

temp = cp.Variable((2,2))
ç = [cp.log_det(temp @ temp.T)>=0.1]
a = np.array([1, 1])
test = a @ temp @ a.T
prob = cp.Problem(cp.Minimize(test),
                constraints)
prob.solve()