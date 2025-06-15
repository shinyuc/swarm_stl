from fontTools.misc.bezierTools import epsilon
from gurobipy import *
from math import *
import numpy as np
import time
import safe_cons_v1 as sc

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
    # z = 1 iff. [a, b] and [c, d] has no intersection
    # b < c or d < a
    return Disjunction([c-b-EPS, a-d-EPS])

def hasIntersection(a, b, c, d):
    # z = 1 iff. [a, b] and [c, d] has no intersection
    # b >= c and d >= a
    return Conjunction([b-c, d-a])

def always(i, a, b, zphis, PWL):
    t_i = PWL[i][1]
    t_i_1 = PWL[i+1][1]
    conjunctions = []
    for j in range(len(PWL)-1):
        t_j = PWL[j][1]
        t_j_1 = PWL[j+1][1]
        conjunctions.append(Disjunction([noIntersection(t_j, t_j_1, t_i + a, t_i_1 + b), zphis[j]]))
    return Conjunction(conjunctions)

def eventually(i, a, b, zphis, PWL):
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

def release(i, a, b, zphi1s, zphi2s, PWL):
    t_i = PWL[i][1]
    t_i_1 = PWL[i+1][1]
    conjunctions = []
    for j in range(len(PWL)-1):
        t_j = PWL[j][1]
        t_j_1 = PWL[j+1][1]
        disjunctions = [noIntersection(t_j, t_j_1, t_i_1 + a, t_i + b), zphi2s[j]]
        for l in range(j):
            t_l = PWL[l][1]
            t_l_1 = PWL[l+1][1]
            disjunctions.append(Conjunction([hasIntersection(t_l, t_l_1, t_i_1, t_i_1 + b), zphi1s[l]]))
        conjunctions.append(Disjunction(disjunctions))
    return Conjunction(conjunctions)

def mu(margin, model, swarm, i, PWL, bloat_factor, A, b):
    bloat_factor = np.max([0, bloat_factor])
    # this segment is fully contained in Ax<=b (shrinked)
    b = b.reshape(-1)
    num_edges = len(b)
    conjunctions = []
    for e in range(num_edges):
        a = A[e,:]
        x0 = PWL[i][0]
        x1 = PWL[i + 1][0]
        temp = sc.stl_v2(margin, model, swarm, a, b, x0, x1, e, bloat_factor, EPS)
        conjunctions.append(temp)
        if i >= 1:
            x0_last = PWL[i-1][0]
            temp_last = sc.stl_last(margin, swarm, a, b, x0, x0_last, e, bloat_factor, EPS)
            conjunctions.append(temp_last)
        if i <= len(PWL)-2:
            x1_next = PWL[i+1][0]
            temp_next = sc.stl_next(margin, swarm, a, b, x1, x1_next, e, bloat_factor, EPS)
            conjunctions.append(temp_next)
    return Conjunction(conjunctions)

def negmu(margin, model, swarm, i, PWL, bloat_factor, A, b):
    # this segment is outside Ax<=b (bloated)
    b = b.reshape(-1)
    num_edges = len(b)
    disjunctions = []
    for e in range(num_edges):
        a = A[e,:]
        conjunctions = []
        x0 = PWL[i][0]
        x1 = PWL[i + 1][0]

        temp = sc.neg_stl_v2(margin,model, swarm, a,b,x0,x1,e,bloat_factor, EPS)
        conjunctions.append(temp)
        disjunctions.append(Conjunction(conjunctions))


        # temp1 = sum([a[k]*x0[k] for k in range(len(x0))]) - (b[e] + np.linalg.norm(a) * bloat_factor) - EPS -  margin
        # temp2 =  sum([a[k] * x1[k] for k in range(len(x1))]) - (b[e] + np.linalg.norm(a) * bloat_factor) - EPS-  margin
        # conjunctions.append(temp1)
        # conjunctions.append(temp2)
        # disjunctions.append(Conjunction(conjunctions))

    return Disjunction(disjunctions)

def add_space_constraints(model, xlist, limits, bloat=0.):
    xlim, ylim = limits
    for x in xlist:
        model.addConstr(x[0] >= (xlim[0] + bloat))
        model.addConstr(x[1] >= (ylim[0] + bloat))
        model.addConstr(x[0] <= (xlim[1] - bloat))
        model.addConstr(x[1] <= (ylim[1] - bloat))

    return None

def add_time_constraints(model, PWL, tmax=None):
    if tmax is not None:
        model.addConstr(PWL[-1][1] <= tmax - T_MIN_SEP)

    for i in range(len(PWL)-1):
        x1 = PWL[i][0]
        t1 = PWL[i][1]
        x2 = PWL[i+1][0]
        t2 = PWL[i+1][1]
        model.addConstr(t2 - t1 >= T_MIN_SEP)

def add_velocity_constraints(model, PWL, vmax=3):
    for i in range(len(PWL)-1):
        x1 = PWL[i][0]
        t1 = PWL[i][1]
        x2 = PWL[i+1][0]
        t2 = PWL[i+1][1]
        # squared_dist = sum([(x1[j]-x2[j])*(x1[j]-x2[j]) for j in range(len(x1))])
        # model.addConstr(squared_dist <= (vmax**2) * (t2 - t1) * (t2 - t1))
        L1_dist = L1Norm(model, _sub(x1,x2))
        model.addConstr(L1_dist <= vmax * (t2 - t1))

def disjoint_segments(model, seg1, seg2, bloat):
    # inter agent collision avoidance
    assert(len(seg1) == 2)
    assert(len(seg2) == 2)
    # assuming that bloat is the error bound in two norm for one agent
    return 0.5 * L1Norm(model, _sub(_add(seg1[0], seg1[1]), _add(seg2[0], seg2[1]))) - 0.5 * (L1Norm(model, _sub(seg1[0], seg1[1])) + L1Norm(model, _sub(seg2[0], seg2[1]))) - 2*bloat*np.sqrt(len(seg1[0])) - EPS

def add_mutual_clearance_constraints(model, PWLs, bloat):
    for i in range(len(PWLs)):
        for j in range(i+1, len(PWLs)):
            PWL1 = PWLs[i]
            PWL2 = PWLs[j]
            for k in range(len(PWL1)-1):
                for l in range(len(PWL2)-1):
                    x11, t11 = PWL1[k]
                    x12, t12 = PWL1[k+1]
                    x21, t21 = PWL2[l]
                    x22, t22 = PWL2[l+1]
                    z_noIntersection = noIntersection(t11, t12, t21, t22)
                    z_disjoint_segments = disjoint_segments(model, [x11, x12], [x21, x22], bloat)
                    z = Disjunction([z_noIntersection, z_disjoint_segments])
                    add_CDTree_Constraints(model, z)

class Node(object):
    """docstring for Node"""
    def __init__(self, op, deps = [], zs = [], info = []):
        super(Node, self).__init__()
        self.op = op
        self.deps = deps
        self.zs = zs
        self.info = info

def clearSpecTree(spec):
    # use to clear all deps nodes
    for dep in spec.deps:
        clearSpecTree(dep)
    spec.zs = []

def handleSpecTree(margin, model,swarm,spec, PWL, bloat_factor, size):
    for dep in spec.deps:
        handleSpecTree(margin, model,swarm, dep, PWL, bloat_factor, size)
    if len(spec.zs) == len(PWL)-1:
        return
    elif len(spec.zs) > 0:
        raise ValueError('incomplete zs')
    if spec.op == 'mu':
        spec.zs = [mu(margin, model, swarm, i, PWL, 0.1, spec.info['A'], spec.info['b']) for i in range(len(PWL)-1)]
    elif spec.op == 'negmu':
        spec.zs = [negmu(margin,model, swarm, i, PWL, bloat_factor + size, spec.info['A'], spec.info['b']) for i in range(len(PWL)-1)]
    elif spec.op == 'and':
        spec.zs = [Conjunction([dep.zs[i] for dep in spec.deps]) for i in range(len(PWL)-1)]
    elif spec.op == 'or':
        spec.zs = [Disjunction([dep.zs[i] for dep in spec.deps]) for i in range(len(PWL)-1)]
    elif spec.op == 'U':
        spec.zs = [until(i, spec.info['int'][0], spec.info['int'][1], spec.deps[0].zs, spec.deps[1].zs, PWL) for i in range(len(PWL)-1)]
    elif spec.op == 'F':
        spec.zs = [eventually(i, spec.info['int'][0], spec.info['int'][1], spec.deps[0].zs, PWL) for i in range(len(PWL)-1)]
    elif spec.op == 'BF':
        spec.zs = [bounded_eventually(i, spec.info['int'][0], spec.info['int'][1], spec.deps[0].zs, PWL, spec.info['tmax']) for i in range(len(PWL)-1)]
    elif spec.op == 'A':
        spec.zs = [always(i, spec.info['int'][0], spec.info['int'][1], spec.deps[0].zs, PWL) for i in range(len(PWL)-1)]
    else:
        raise ValueError('wrong op code')

def gen_CDTree_constraints(model, root):
    if not hasattr(root, 'deps'):
        return [root,]
    else:
        if len(root.constraints)>0:
            # TODO: more check here
            return root.constraints
        dep_constraints = []
        for dep in root.deps:
            dep_constraints.append(gen_CDTree_constraints(model, dep))
        zs = []
        for dep_con in dep_constraints:
            if isinstance(root, Disjunction):
                z = model.addVar(vtype=GRB.BINARY)
                zs.append(z)
                dep_con = [con + M * (1 - z) for con in dep_con]
            root.constraints += dep_con
        if len(zs)>0:
            root.constraints.append(sum(zs)-1)
        model.update()
        return root.constraints

def add_CDTree_Constraints(model, root):
    constrs = gen_CDTree_constraints(model, root)
    for con in constrs:
        model.addConstr(con >= 0)

def plan(swarm, x0s, specs, bloat, limits=None, num_segs=None, tasks=None, vmax=3., MIPGap=1e-4, max_segs=None, tmax=None, hard_goals=None, size=0.11*4/2):
    if num_segs is None:
        min_segs = 1
        assert max_segs is not None
    else:
        min_segs = num_segs
        max_segs = num_segs

    for num_segs in range(min_segs, max_segs+1):
        for spec in specs:
            clearSpecTree(spec)

        if tasks:
            for task in tasks:
                for t in task:
                    clearSpecTree(t)

        print('----------------------------')
        print('num_segs', num_segs)

        PWLs = []
        m = Model("xref")
        # m.setParam(GRB.Param.OutputFlag, 0)
        m.setParam(GRB.Param.IntFeasTol, IntFeasTol)
        m.setParam(GRB.Param.MIPGap, MIPGap)
        # m.setParam(GRB.Param.NonConvex, 2)
        # m.getEnv().set(GRB_IntParam_OutputFlag, 0)
        # m.addVars(1, lb=-GRB.INFINITY)
        margin = m.addVar(lb=-GRB.INFINITY)
        m.update()
        # margin = [margin[k] for k in range(len(margin))]
        # margin = 1

        for idx_a in range(len(x0s)):
            x0 = x0s[idx_a]
            x0 = np.array(x0).reshape(-1).tolist()
            spec = specs[idx_a]

            dims = len(x0)

            PWL = []
            for i in range(num_segs+1):
                # PWL.append([m.addVars(dims, lb=-GRB.INFINITY), m.addVar()])
                PWL.append([m.addVars(dims, lb=-GRB.INFINITY), m.addVar(), m.addVar()])
            PWLs.append(PWL)
            m.update()

            # the initial constriant
            m.addConstrs(PWL[0][0][i] == x0[i] for i in range(dims))
            m.addConstr(PWL[0][1] == 0)

            if hard_goals is not None:
                goal = hard_goals[idx_a]
                m.addConstrs(PWL[-1][0][i] == goal[i] for i in range(dims))

            if limits is not None:
                add_space_constraints(m, [P[0] for P in PWL], limits)

            add_velocity_constraints(m, PWL, vmax=vmax)
            # add_time_constraints(m, PWL, tmax)

            handleSpecTree(margin, m, swarm, spec, PWL, bloat, size)
            add_CDTree_Constraints(m, spec.zs[0])

        if tasks is not None:
            for idx_agent in range(len(tasks)):
                for idx_task in range(len(tasks[idx_agent])):
                    handleSpecTree(m, tasks[idx_agent][idx_task], PWLs[idx_agent], bloat, size)

            conjunctions = []
            for idx_task in range(len(tasks[0])):
                disjunctions = [tasks[idx_agent][idx_task].zs[0] for idx_agent in range(len(tasks))]
                conjunctions.append(Disjunction(disjunctions))
            z = Conjunction(conjunctions)
            add_CDTree_Constraints(m, z)

        add_mutual_clearance_constraints(m, PWLs, bloat)

        # obj = sum([L1Norm(m, _sub(PWL[i][0], PWL[i+1][0])) for PWL in PWLs for i in range(len(PWL)-1)])
        obj = sum([PWL[-1][1] for PWL in PWLs])
        temp = PWLs[0]
        test1 = temp[0][1]
        test2 = margin
        # m.setObjective(obj, GRB.MINIMIZE)
        obj1 = LinExpr()
        obj1 += margin
        m.setObjective(obj1, GRB.MAXIMIZE)
        # m.update()

        m.write("test.lp")
        print('NumBinVars: %d'%m.getAttr('NumBinVars'))

        # m.computeIIS()
        # import ipdb;ipdb.set_treace()
        try:
            start = time.time()
            m.optimize()
            end = time.time()
            print('sovling it takes %.3f s'%(end - start))
            PWLs_output = []
            for PWL in PWLs:
                PWL_output = []
                for P in PWL:
                    PWL_output.append([[P[0][i].X for i in range(len(P[0]))], P[1].X])
                PWLs_output.append(PWL_output)
            margin_result = margin.X
            m.dispose()
            return PWLs_output, margin_result
        except Exception as e:
            m.dispose()
    return [None,]
