from gurobipy import Model, GRB

# Create a model
model = Model("example")

# Add variables
x1 = model.addVar(name="x1")
x2 = model.addVar(name="x2")

# Add the constraint x1 == x2 + 0.1
model.addConstr(x1 == x2 + 0.1, name="offset_constraint")

# Set an objective (e.g., minimize x1 + x2)
model.setObjective(x1 + x2, GRB.MINIMIZE)

# Optimize the model
model.optimize()

# Print the results
if model.status == GRB.OPTIMAL:
    print("Optimal value of x1:", x1.X)
    print("Optimal value of x2:", x2.X)
