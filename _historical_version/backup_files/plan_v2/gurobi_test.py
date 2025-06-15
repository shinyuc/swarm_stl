import gurobipy as gp
from gurobipy import GRB
import numpy as np

# Create a new model
m = gp.Model("linear program")
# Create variables
x = m.addVars(2, lb=0, name = "x")
obj_fn = 5*x[0] + 4*x[1]
m.setObjective(obj_fn, GRB.MINIMIZE)
c1 = m.addConstr(x[0] + x[1] >= 8, "c1")
c2 = m.addConstr(2*x[0] + x[1] >= 10, "c2")
c3 = m.addConstr(x[0] + 4*x[1] >= 11, "c3")
m.optimize()
# m.write('lp.lp')
print("The objective function value: ", m.ObjVal)
for v in m.getVars():
    print(f"{v.VarName} {v.X:g}")