from gurobipy import *
from math import *
import numpy as np
import time
import control as ct
from safe_constraint_func import *

M = 1e3
# a large M causes numerical issues and make the model infeasible to Gurobi
T_MIN_SEP = 1e-2
# see comments in GreaterThanZero
IntFeasTol  = 1e-1 * T_MIN_SEP / M

def setM(v):
    global M, IntFeasTol
    M = v
    IntFeasTol  = 1e-1 * T_MIN_SEP / M

EPS = 1e-2

def _sub(x1, x2):
    return [x1[i] - x2[i] for i in range(len(x1))]

def _add(x1, x2):
    return [x1[i] + x2[i] for i in range(len(x1))]

def L1Norm(model, x):
    xvar = model.addVars(len(x), lb=-GRB.INFINITY)
    abs_x = model.addVars(len(x))
    model.update()
    xvar = [xvar[i] for i in range(len(xvar))]
    abs_x = [abs_x[i] for i in range(len(abs_x))]
    for i in range(len(x)):
        model.addConstr(xvar[i] == x[i])
        model.addConstr(abs_x[i] == abs_(xvar[i]))
    return sum(abs_x)

class Conjunction(object):
    # conjunction node
    # and
    def __init__(self, deps = []):
        super(Conjunction, self).__init__()
        self.deps = deps
        self.constraints = []

class Disjunction(object):
    # disjunction node
    # or
    def __init__(self, deps = []):
        super(Disjunction, self).__init__()
        self.deps = deps
        self.constraints = []

def noIntersection(a, b, c, d):
    # this is used for the time overlap constraint
    # z = 1 iff. [a, b] and [c, d] has no intersection
    # b < c or d < a
    return Disjunction([c-b-EPS, a-d-EPS])

def hasIntersection(a, b, c, d):
    # this is used for the time overlap constraint
    # z = 1 iff. [a, b] and [c, d] has no intersection
    # b >= c and d >= a
    return Conjunction([b-c, d-a])

def always(i, a, b, zphis, PWL):
    # PWL here refers to SEG in our work
    # i: i-th time step, starting with zero
    t_i = PWL[i][1] # PWL[i][0] is position, PWL[i][1] is time, PWL[i][2] is A matrix for bounding box
    t_i_1 = PWL[i+1][1]
    conjunctions = []
    for j in range(len(PWL)-1): # for all time interval
        t_j = PWL[j][1]
        t_j_1 = PWL[j+1][1]
        # swarm at j segment should satisfy the phi property
        conjunctions.append(Disjunction([noIntersection(t_j, t_j_1, t_i + a, t_i_1 + b), zphis[j]]))
    return Conjunction(conjunctions)

def eventually(i, a, b, zphis, PWL):
    # i: i-th segment
    t_i = PWL[i][1]
    t_i_1 = PWL[i+1][1]
    z_intervalWidth = b-a-(t_i_1-t_i)-EPS
    disjunctions = []
    for j in range(len(PWL)-1):
        t_j = PWL[j][1]
        t_j_1 = PWL[j+1][1]
        disjunctions.append(Conjunction([hasIntersection(t_j, t_j_1, t_i_1 + a, t_i + b), zphis[j]]))
    return Conjunction([z_intervalWidth, Disjunction(disjunctions)])

def bounded_eventually(i, a, b, zphis, PWL, tmax):
    # if total time is not enough
    t_i = PWL[i][1]
    t_i_1 = PWL[i+1][1]
    z_intervalWidth = b-a-(t_i_1-t_i)-EPS
    disjunctions = []
    for j in range(len(PWL)-1):
        t_j = PWL[j][1]
        t_j_1 = PWL[j+1][1]
        disjunctions.append(Conjunction([hasIntersection(t_j, t_j_1, t_i_1 + a, t_i + b), zphis[j]]))
    return Disjunction([Conjunction([z_intervalWidth, Disjunction(disjunctions)]), t_i+b-tmax-EPS])

def until(i, a, b, zphi1s, zphi2s, PWL):
    t_i = PWL[i][1]
    t_i_1 = PWL[i+1][1]
    z_intervalWidth = b-a-(t_i_1-t_i)-EPS
    disjunctions = []
    for j in range(len(PWL)-1):
        t_j = PWL[j][1]
        t_j_1 = PWL[j+1][1]
        conjunctions = [hasIntersection(t_j, t_j_1, t_i_1 + a, t_i + b), zphi2s[j]]
        for l in range(j+1):
            t_l = PWL[l][1]
            t_l_1 = PWL[l+1][1]
            conjunctions.append(Disjunction([noIntersection(t_l, t_l_1, t_i, t_i_1 + b), zphi1s[l]]))
        disjunctions.append(Conjunction(conjunctions))
    return Conjunction([z_intervalWidth, Disjunction(disjunctions)])

def mu(i, PWL, epsilon_s, F, g):
    bloat_factor = np.max([0, epsilon_s])
    # this segment is fully contained in Ax<=b (shrinked)
    # need to change

    g = g.reshape(-1)
    num_edges = len(g)
    conjunctions = [] # and, all constraints must be satisfied
    for e in range(num_edges):
        a = F[e,:]
        x = PWL[i][0]
        conjunctions.append(stl_1())  # EPS is for strictly grater than zero
        if i >=1:
            conjunctions.append(stl_2())
        if i < len(PWL)-1:
            conjunctions.append(stl_3())
    return Conjunction(conjunctions)

def negmu(i, swarm_s_i,swarm_s_i_last,swarm_s_i_next, PWL, epsilon_s, F, g):
    # this segment is outside Ax<=b (bloated)
    # need to change
    g = g.reshape(-1)
    num_edges = len(g)
    disjunctions = []
    for e in range(num_edges):
        a = F[e,:]
        conjunctions = []
        for j in [i, i+1]:
            x = PWL[j][0]
            conjunctions.append(stl_neg_1(swarm_s_i, a, g[e], epsilon=1e-1))  # EPS is for strictly grater than zero
            conjunctions.append(stl_neg_2(swarm_s_i, swarm_s_i_last, a, g[e], epsilon=1e-1, epsilon_r=1))
            conjunctions.append(stl_neg_3(swarm_s_i, swarm_s_i_next, a, g[e], epsilon=1e-1, epsilon_r=1))
        disjunctions.append(Conjunction(conjunctions))
    return Disjunction(disjunctions)

def add_space_constraints(model, xlist, limits, bloat=0.):
    # this constraint ensures that the agents will not exceed the boundary of the environment
    # need to change
    # xlist refers to the positions of swarms
    xlim, ylim = limits
    for x in xlist:
        model.addConstr(x[0] >= (xlim[0] + bloat))
        model.addConstr(x[1] >= (ylim[0] + bloat))
        model.addConstr(x[0] <= (xlim[1] - bloat))
        model.addConstr(x[1] <= (ylim[1] - bloat))

    return None