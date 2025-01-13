import cvxpy as cp
import numpy as np
import math

# Define variables
a = cp.Variable()
b = cp.Variable()
d = cp.Variable()

# Define the 2x2 matrix X
X = cp.bmat([[a, b], [b, d]])

# Define the constraints
constraints = [
    X >> 0,  # X is positive semidefinite
    cp.log_det(X) >= math.exp(1)  # Log determinant constraint
]

# Define the objective: minimize a + d
objective = cp.Minimize(a + d)

# Formulate and solve the problem
problem = cp.Problem(objective, constraints)
result = problem.solve()

# Output the results
print("Optimal value of a + d:", result)
print("Optimal values:")
print("a =", a.value)
print("b =", b.value)
print("d =", d.value)
print("Optimal X:\n", np.array([[a.value, b.value], [b.value, d.value]]))
teat_a = math.exp(1)
testa = np.array([[teat_a, 0], [0, 1]])
print(cp.log_det(testa).value)