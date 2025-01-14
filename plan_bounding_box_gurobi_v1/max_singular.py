import gurobipy as gp
from gurobipy import GRB
import numpy as np

def max_singular(Sigma_list):
    Sigma = np.array([[Sigma_list[0], Sigma_list[1]], [Sigma_list[2], Sigma_list[3]]])
    t = model.addVar(name="t")  # Scalar variable
    A_x = np.zeros((n, n))  # Placeholder for now; will be defined in terms of x
    for i in range(n):
        for j in range(m):
            A_x += C[i, j] * x[j]

    # Define the block matrix F
    # F = [tI, A(x); A(x)^T, tI]
    F = [[None for _ in range(2 * n)] for _ in range(2 * n)]
    I = np.eye(n)  # 2x2 identity matrix

    # Populate the entries of F
    for i in range(n):
        for j in range(n):
            F[i][j] = t * I[i, j]  # Top-left block: t * I
            F[i][j + n] = A_x[i, j]  # Top-right block: A(x)
            F[i + n][j] = A_x[j, i]  # Bottom-left block: A(x)^T
            F[i + n][j + n] = t * I[i, j]  # Bottom-right block: t * I

    # Add the semidefinite constraint
    model.addMVar(F, GRB.SEMIDEF, name="F")

    # Define an objective (e.g., minimize t)
    model.setObjective(t, GRB.MINIMIZE)

    # Optimize the model
    model.optimize()

    # Retrieve the results
    if model.status == GRB.OPTIMAL:
        print("Optimal t:", t.X)
        x_vals = [x[i].X for i in range(m)]
        print("Optimal x:", x_vals)
    else:
        print("No optimal solution found.")

    return max_singular


if __name__ == "__main__":
    # Define dimensions
    n = 2  # Size of the identity matrix
    m = 4  # Size of x

    # Define data
    C = np.random.randn(n, m)  # Random coefficient matrix

    # Create a Gurobi model
    model = gp.Model("LMI_constraint")

    # Define decision variables
    t = model.addVar(name="t")  # Scalar variable
    x = model.addVars(m, name="x")  # Vector of variables

    # Define A(x) as a linear function of x
    A_x = np.zeros((n, n))  # Placeholder for now; will be defined in terms of x
    for i in range(n):
        for j in range(m):
            A_x += C[i, j] * x[j]

    # Define the block matrix F
    # F = [tI, A(x); A(x)^T, tI]
    F = [[None for _ in range(2 * n)] for _ in range(2 * n)]
    I = np.eye(n)  # 2x2 identity matrix

    # Populate the entries of F
    for i in range(n):
        for j in range(n):
            F[i][j] = t * I[i, j]  # Top-left block: t * I
            F[i][j + n] = A_x[i, j]  # Top-right block: A(x)
            F[i + n][j] = A_x[j, i]  # Bottom-left block: A(x)^T
            F[i + n][j + n] = t * I[i, j]  # Bottom-right block: t * I

    # Add the semidefinite constraint
    model.addMVar(F, GRB.SEMIDEF, name="F")

    # Define an objective (e.g., minimize t)
    model.setObjective(t, GRB.MINIMIZE)

    # Optimize the model
    model.optimize()

    # Retrieve the results
    if model.status == GRB.OPTIMAL:
        print("Optimal t:", t.X)
        x_vals = [x[i].X for i in range(m)]
        print("Optimal x:", x_vals)
    else:
        print("No optimal solution found.")
