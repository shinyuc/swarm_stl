import gurobipy as gp
from gurobipy import GRB

# Create a new Gurobi model
model = gp.Model("max_singular_value")

# Define variables for matrix A (2x2 matrix)
a11 = model.addVar(lb=-GRB.INFINITY, name="a11")
a12 = model.addVar(lb=-GRB.INFINITY, name="a12")
a21 = model.addVar(lb=-GRB.INFINITY, name="a21")
a22 = model.addVar(lb=-GRB.INFINITY, name="a22")

# Symmetric matrix Q = A^T * A
q11 = model.addVar(name="q11")
q12 = model.addVar(name="q12")
q22 = model.addVar(name="q22")

# Largest eigenvalue of Q
lambda_max = model.addVar(lb=0, name="lambda_max")

# Add constraints for Q = A^T * A
model.addConstr(q11 == a11 * a11 + a21 * a21, "Q11")
model.addConstr(q12 == a11 * a12 + a21 * a22, "Q12")
model.addConstr(q22 == a12 * a12 + a22 * a22, "Q22")

# Add semidefinite constraint for Q - λ_max * I >= 0
# This ensures λ_max is the largest eigenvalue of Q
model.addConstr(q11 - lambda_max >= 0, "Eig1")
model.addConstr(q22 - lambda_max >= 0, "Eig2")
model.addConstr(q11 * q22 - q12 * q12 - lambda_max * (q11 + q22) + lambda_max * lambda_max >= 0, "Det")

# Set the objective to maximize the square root of λ_max
model.setObjective(lambda_max, GRB.MAXIMIZE)

# Optimize the model
model.optimize()

# Output results
if model.status == GRB.OPTIMAL:
    print(f"Maximum singular value: {lambda_max.X ** 0.5}")
    print(f"A = [[{a11.X}, {a12.X}], [{a21.X}, {a22.X}]]")
