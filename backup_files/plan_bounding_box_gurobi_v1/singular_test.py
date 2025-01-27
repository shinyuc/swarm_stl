import math
from gurobipy import Model, GRB, QuadExpr
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from math import *

# Create Gurobi model
def maximum_singular(elements_sum):
    model = Model("MaxSingularValueConstraint")

    # Decision variables: elements of A and the upper bound t
    a = model.addVar(name="a", lb=-GRB.INFINITY)
    b = model.addVar(name="b", lb=-GRB.INFINITY)
    c = model.addVar(name="c", lb=-GRB.INFINITY)
    d = model.addVar(name="d", lb=-GRB.INFINITY)
    t = model.addVar(name="t", lb=0)  # t >= 0 because it's a norm

    # Define the matrix A and the semidefinite constraint
    # A^T A must be positive semidefinite with maximum eigenvalue <= t^2
    # For A = [a, b; c, d], A^T A = [[a^2 + c^2, ab + cd], [ab + cd, b^2 + d^2]]
    A = np.array([[a, b],
                  [c, d]])

    ATA = [[a*a + c*c, a*b + c*d],
           [a*b + c*d, b*b + d*d]]

    # Add semidefinite constraint: t^2 * I - A^T A is PSD
    identity_matrix = [[1, 0], [0, 1]]
    model.addQConstr(t*t - ATA[0][0] >= 0, name=f"PSD_{0}_{0}")
    model.addQConstr((t*t - a*a - c*c)*(t * t - b*b - d*d) - (a*b + c*d)*(a*b + c*d) >= 0, name=f"PSD_{1}_{1}")
    model.addConstr(t >= 2, name="t_positive")

    # Objective: minimize t (or some other goal)
    model.setObjective(t, GRB.MINIMIZE)

    # Optimize the model
    model.optimize()

    # Display results
    if model.status == GRB.OPTIMAL:
        print(f"Optimal t: {t.x}")
        print(f"Matrix A: [[{a.x}, {b.x}], [{c.x}, {d.x}]]")
    else:
        print("No optimal solution found.")

def maximum_singular_2():


    # Create a model
    model = gp.Model("PSD_2x2")

    # Define variables
    a = model.addVar(name="a")
    b = model.addVar(name="b")
    c = model.addVar(name="c")
    d = model.addVar(name="d")
    a1 = model.addVar(name="a1")
    a2 = model.addVar(name="a2")
    a3 = model.addVar(name="a3")
    a4 = model.addVar(name="a4")
    model.addConstr(a1 == a*a + c*c)
    model.addConstr(a2 == a*b + c*d)
    model.addConstr(a3 == a*b + c*d)
    model.addConstr(a4 == b*b + d*d)
    # Add constraints for positive semidefiniteness
    model.addConstr(a1 >= 0, "PrincipalMinor")
    model.addConstr(a1*a3 - a2*a4 >= 0, "Determinant")
    model.addConstr(a1 >= 5)

    # (Optional) Set an objective function if needed
    # For example, minimize a + c
    model.setObjective(a + c, GRB.MINIMIZE)

    # Optimize the model
    model.optimize()

    # Print the results
    if model.status == GRB.OPTIMAL:
        print(f"Optimal values: a = {a.X}, b = {b.X}, c = {c.X}, d = {d.X}")
    else:
        print("No optimal solution found.")

def maximum_singular_3():
    # Create a model
    model = gp.Model("PSD_2x2")

    # Define variables
    a = model.addVar(name="a")
    b = model.addVar(name="b")
    c = model.addVar(name="c")
    d = model.addVar(name="d")

    matrix_a = np.array([[a, b], [c, d]])
    singulars = max(np.linalg.eigvals(matrix_a))
    # Add constraints for positive semidefiniteness
    model.addConstr(a * a >= 0, "PrincipalMinor")
    model.addConstr(a * c - b * d >= 0, "Determinant")
    model.addConstr(singulars >=3)
    model.addConstr(b >= 5)

if __name__ == "__main__":
    # model = Model("MaxSingularValueConstraint")
    # a = model.addVar(name="a", lb=-GRB.INFINITY)
    # b = model.addVar(name="b", lb=-GRB.INFINITY)
    # c = model.addVar(name="c", lb=-GRB.INFINITY)
    # d = model.addVar(name="d", lb=-GRB.INFINITY)
    # t = model.addVar(name="t", lb=0)  # t >= 0 because it's a norm
    maximum_singular_2()