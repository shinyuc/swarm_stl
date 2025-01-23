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
from control import margin

from gurobi_mosek_comparsion.test1 import constraints


def L1Norm(x):
    """Compute the L1 norm using cvxpy."""
    return cp.norm(x, 1)

temp = cp.Variable((2,2))
ç = [cp.log_det(temp @ temp.T)>=0.1]
a = np.array([1, 1])
test = a @ temp @ a.T
prob = cp.Problem(cp.Minimize(test),
                constraints)
prob.solve()

print(L1Norm(np.array([0, 900])).value)

x = [1, 2, 3]
print(len(x))
y = [x[i] for i in range(len(x)-1)]
print(y)

import cvxpy as cp



# Define variables and constraints


# Define variables
x = cp.Variable()

# Define constraints in the form ax + b >= 0
constraints = [
    -x + 1 <= 0,  # Equivalent to x >= 1
    x - 5 <= 0   # Equivalent to x <= 5
]

# Define the objective
objective = cp.Minimize(x)

# Solve the problem
problem = cp.Problem(objective, constraints)
problem.solve()

# Access the value of the variable
print("Optimal value of x:", x.value)

# Evaluate the constraints
for i, constraint in enumerate(constraints):
    lhs = constraint.expr.value  # Left-hand side of the constraint
    print(f"Value of constraint {i+1} (LHS): {lhs}")

lhs_1 = -x.value + 1  # For x - 1 >= 0
lhs_2 = x.value - 5  # For 5 - x >= 0
# print("Value of constraint 1 (LHS):", lhs_1)
# print("Value of constraint 2 (LHS):", lhs_2)
# for i in range(19):
#     print(i)
print("\nIn what follows, we will test the maximum singular value of a matrix")


# Define the matrix A (2x2) as a variable
A = cp.Variable((2, 2))
margin = cp.Variable(1)
# Define the objective function: minimize the Frobenius norm of A
objective = cp.Maximize(margin)

# Define the constraint: maximum singular value of A is less than or equal to 4
constraints = [10 - cp.sigma_max(A) - margin >= 4]

constraints.append(cp.log_det(A) >= 3)

# Define the problem
problem = cp.Problem(objective, constraints)

# Solve the problem
problem.solve()

# Output the results
print("Optimal value:", problem.value)
print("Optimal margin:")
print(margin.value)

import cvxpy as cp
import numpy as np

# Dimensions of A
n = 4

# Define the variable A
A = cp.Variable((n, n))

# Define the auxiliary variable X and impose the equality constraint A @ A.T = X
X = cp.Variable((n, n), PSD=True)  # X is symmetric and positive semidefinite

# Define the constraint log_det(X) >= 3
constraints = [cp.log_det(A) >= 3]

# Define an example objective (e.g., minimize the Frobenius norm of A)
objective = cp.Minimize(cp.norm(A, 'fro'))

# Formulate the problem
problem = cp.Problem(objective, constraints)

# Solve the problem
problem.solve(solver=cp.SCS)

# Output the results
print("Optimal value:", problem.value)
print("Optimal A:", A.value)

