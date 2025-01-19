import cvxpy as cp
import numpy as np

# Dimensions of A
n = 4

# Define the variable A
A = cp.Variable((n, n))

# Define the auxiliary matrix X = A @ A.T
X = cp.Variable((n, n), PSD=True)  # X is symmetric and positive semidefinite

# Define the constraint log_det(X) >= 3
constraints = [cp.log_det(X) >= 3, A @ A.T == X]

# Define an example objective (e.g., minimize the Frobenius norm of A)
objective = cp.Minimize(cp.norm(A, 'fro'))

# Formulate the problem
problem = cp.Problem(objective, constraints)

# Solve the problem
problem.solve(solver=cp.SCS)

# Output the results
print("Optimal value:", problem.value)
print("Optimal A:", A.value)
