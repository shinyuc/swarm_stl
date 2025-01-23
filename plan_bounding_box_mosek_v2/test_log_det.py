import cvxpy as cp
import numpy as np

# Dimensions of A
n = 2

# Define the auxiliary matrix X = A @ A.T
A = cp.Variable((n, n), PSD=True)  # X is symmetric and positive semidefinite
z = cp.Variable(1, boolean=True)
# Define the constraint log_det(X) >= 3
M = 1000
constraints = [cp.log_det(A) + (1-z)*M >= 3 ]
obj = cp.Minimize(cp.norm(A))
prob = cp.Problem(obj, constraints)
prob.solve()



# Solve the problem
# problem.solve(solver=cp.SCS)

# Output the results
print("Optimal value:", prob.value)
print("Optimal A:", A.value)
