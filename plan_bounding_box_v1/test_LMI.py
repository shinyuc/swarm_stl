import cvxpy as cp
import numpy as np
import mosek

# Variables
a = cp.Variable()
b = cp.Variable()
c = cp.Variable()
d = cp.Variable()
t = cp.Variable()

# Matrix A (2x2)
A = cp.vstack([cp.hstack([a, b]), cp.hstack([c, d])])

# Construct the block matrix [t*I, A; A.T, t*I]
I = np.eye(2)
LMI_matrix = cp.vstack([
    cp.hstack([t * I, A]),
    cp.hstack([A.T, t * I])
])

# SDP constraint: LMI_matrix must be positive semidefinite
constraints = [LMI_matrix >> 0]
constraints.append(t>=1)

# Objective function (dummy example)
objective = cp.Minimize(t)  # Example: minimize t

# Problem definition
problem = cp.Problem(objective, constraints)

# Solve the problem using MOSEK
problem.solve(solver=cp.MOSEK)

# Results
print("Optimal value of t:", t.value)
print("Matrix A:\n", A.value)
