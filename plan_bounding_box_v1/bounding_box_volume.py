import cvxpy as cp

# Define the variables a, b, c, d
a = cp.Variable()
b = cp.Variable()
c = cp.Variable()
d = cp.Variable()

# Define the 2x2 matrix A
A = cp.vstack([cp.hstack([a, b]), cp.hstack([c, d])])

# Define the constraint log(det(A)) >= 10
constraints = [cp.log_det(A) >= 10]

# Set up an optimization problem (optional, here we minimize a dummy objective)
objective = cp.Minimize(0)

# Define and solve the problem
problem = cp.Problem(objective, constraints)
problem.solve()

# Output the result
print(f"Solution: a = {a.value}, b = {b.value}, c = {c.value}, d = {d.value}")
print(f"log(det(A)) = {cp.log_det(A).value}")
