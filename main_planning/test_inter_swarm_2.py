from gurobipy import Model, GRB

m = Model("xref")
margin = m.addVar()            # default lb = 0, ub = +∞
m.addConstr(4 - margin >= 0)   # margin <= 4
m.setObjective(margin, GRB.MAXIMIZE)
m.optimize()

print(margin.X)
# print("Model status:", m.status)    # 2 means OPTIMAL
# if m.status == GRB.OPTIMAL:
#     print("Optimal margin:", margin.X)
# else:
#     print("Model not solved to optimality.")
