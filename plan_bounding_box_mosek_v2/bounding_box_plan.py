import math
import cvxpy as cp
# from gurobipy import *
from math import *
import numpy as np
import time
# import safe_cons_boxes as sc

M = 1e3
# a large M causes numerical issues and make the model infeasible to Gurobi
T_MIN_SEP = 1e-2
# see comments in GreaterThanZero
IntFeasTol  = 1e-1 * T_MIN_SEP / M

def setM(v):
    global M, IntFeasTol
    M = v
    IntFeasTol  = 1e-1 * T_MIN_SEP / M

EPS = 1e-3

def _sub(x1, x2):
    return x1 - x2

def _add(x1, x2):
    return x1 + x2

def L1Norm(x):
    """Compute the L1 norm using cvxpy."""
    return cp.norm(x, 1)

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

def display_mu(swarm, i, PWL, boxes, A, b):
    b = b.reshape(-1)
    num_edges = len(b)
    conjunctions = []
    Sigma = np.array(boxes[0][i][0])
    result = []
    i = 18
    Sigma = boxes[0][18][0].value
    for e in range(num_edges):
        a = A[e, :]
        b_e = b[int(e)]
        x0 = np.array(PWL[i][0])
        x1 = np.array(PWL[i + 1][0])
        B_p = swarm.B
        p = B_p @ B_p.T
        inv_P = np.linalg.inv(p)
        beta_bar = (x1 - x0).T @ p @ (x1 - x0)
        gamma_4 = math.sqrt(a @ inv_P @ a.T * beta_bar)
        gamma_5 = a @ Sigma
        result1 = - a @ x1 + b_e - gamma_4 - np.linalg.norm(gamma_5, ord=1)
        result.append(result1)
    return result

def mu(margin, swarm, i, PWL, boxes, A, b):
    # this segment is fully contained in Ax<=b (shrinked)
    b = b.reshape(-1)
    num_edges = len(b)
    conjunctions = []
    Sigma = boxes[0][i][0]
    for e in range(num_edges):
        a = A[e,:]
        b_e = b[int(e)]
        x0 = np.array(PWL[i][0])
        x1 = np.array(PWL[i + 1][0])
        B_p = swarm.B
        p = B_p @ B_p.T
        inv_P = np.linalg.inv(p)
        beta_bar = (x1 - x0).T @ p @ (x1 - x0)
        gamma_4 = math.sqrt(a @ inv_P @ a.T * beta_bar)
        gamma_5 = a @ Sigma
        result = - a @ x1  + b_e - gamma_4 - L1Norm(gamma_5) - margin
        # print(["mu", i,  - a @ x1  + b_e - gamma_4 ])
        if i == 18:
            result = - a @ x1 + b_e - gamma_4  - L1Norm(gamma_5)  - margin
        #     print(["mu", i, a.tolist(), b_e, x1.tolist(), gamma_4, - a @ x1  + b_e - gamma_4])
        conjunctions.append(result)
    # if i == 18:
    #     print(len(conjunctions))

    return Conjunction(conjunctions)

def negmu(margin, swarm, i, PWL, boxes, A, b):
    # this segment is outside Ax<=b (bloated)
    bloat_factor = 0.01
    b = b.reshape(-1)
    num_edges = len(b)
    disjunctions = []
    print(i)
    Sigma = boxes[0][i][0]
    for e in range(num_edges):
        a = A[e, :]
        conjunctions = []
        b_e = b[int(e)]
        x0 = np.array(PWL[i][0])
        x1 = np.array(PWL[i + 1][0])

        B_p = swarm.B
        p = B_p @ B_p.T
        inv_P = np.linalg.inv(p)
        beta_bar = (x1 - x0).T @ p @ (x1 - x0)
        gamma_4 = math.sqrt(a @ inv_P @ a.T * beta_bar)
        gamma_5 = a @ Sigma
        result = a @ x1 - b_e - gamma_4 - L1Norm(gamma_5) - margin
        # print(["negmu",i, a @ x1 - b_e - gamma_4])
        if i == 18:
            result = a @ x1 - b_e - gamma_4   - L1Norm(gamma_5) - margin
            print(["negmu", i, a @ x1 - b_e - gamma_4])
        #     print(['negmu', i,a,b_e, x1, a @ x1 - b_e  - gamma_4, a @ x1 - b_e ])
            # result = margin
            # print(['negmu_test', a @ x1 - b_e - gamma_4 - L1Norm(np.array([[0, 0], [0, 0.39104226]])).value])
        # a @ x1 - b_e - gamma_4 - np.linalg.norm(a @ np.array([[0, 0], [0, 0.39104226]]), ord=1)
        conjunctions.append(result)
        disjunctions.append(Conjunction(conjunctions))
    return Disjunction(disjunctions)

def add_space_constraints(constraints, xlist, limits, bloat=0.):
    xlim, ylim = limits
    for x in xlist:
        constraints += [x[0] >= (xlim[0] + bloat)]
        constraints += [x[1] >= (ylim[0] + bloat)]
        constraints += [x[0] <= (xlim[1] - bloat)]
        constraints += [x[1] <= (ylim[1] - bloat)]

    return None

def add_time_constraints(constraints, PWL, tmax=None):
    if tmax is not None:
        constraints += [PWL[-1][1] <= tmax - T_MIN_SEP]

    for i in range(len(PWL)-1):
        x1, t1 = PWL[i]
        x2, t2 = PWL[i+1]
        constraints += [t2 - t1 >= T_MIN_SEP]

def add_velocity_constraints(constraints, PWL, vmax=3):
    for i in range(len(PWL)-1):
        x1, t1 = PWL[i]
        x2, t2 = PWL[i+1]
        # squared_dist = sum([(x1[j]-x2[j])*(x1[j]-x2[j]) for j in range(len(x1))])
        # model.addConstr(squared_dist <= (vmax**2) * (t2 - t1) * (t2 - t1))
        L1_dist = L1Norm(_sub(x1,x2))
        constraints += [L1_dist <= vmax * (t2 - t1)]

def disjoint_segments(constraints, seg1, seg2, bloat):
    # inter agent collision avoidance
    assert(len(seg1) == 2)
    assert(len(seg2) == 2)
    # assuming that bloat is the error bound in two norm for one agent
    return 0.5 * L1Norm(_sub(_add(seg1[0], seg1[1]), _add(seg2[0], seg2[1]))) - 0.5 * (L1Norm(_sub(seg1[0], seg1[1])) + L1Norm(_sub(seg2[0], seg2[1]))) - 2*bloat*np.sqrt(len(seg1[0])) - EPS

def add_mutual_clearance_constraints(constraints, PWLs, bloat):
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
                    z_disjoint_segments = disjoint_segments(constraints, [x11, x12], [x21, x22], bloat)
                    z = Disjunction([z_noIntersection, z_disjoint_segments])
                    add_CDTree_Constraints(constraints, z)

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
###########################
def handleSpecTree(boxes, margin, swarm,spec, PWL, bloat_factor, size):
    for dep in spec.deps:
        handleSpecTree(boxes,margin, swarm, dep, PWL, bloat_factor, size)
    if len(spec.zs) == len(PWL)-1:
        return
    elif len(spec.zs) > 0:
        raise ValueError('incomplete zs')
    if spec.op == 'mu':
        spec.zs = [mu(margin, swarm, i, PWL, boxes, spec.info['A'], spec.info['b']) for i in range(len(PWL)-1)]
    elif spec.op == 'negmu':
        spec.zs = [negmu(margin, swarm, i, PWL, boxes, spec.info['A'], spec.info['b']) for i in range(len(PWL)-1)]
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

def gen_CDTree_constraints(constraints, root):
    if not hasattr(root, 'deps'):
        return [root,]
    else:
        if len(root.constraints)>0:
            # TODO: more check here
            return root.constraints
        dep_constraints = []
        for dep in root.deps:
            dep_constraints.append(gen_CDTree_constraints(constraints, dep))
        zs = []
        for dep_con in dep_constraints:
            if isinstance(root, Disjunction):
                z = cp.Variable(1, boolean=True)
                zs.append(z)
                dep_con = [con + M * (1 - z) for con in dep_con]
            root.constraints += dep_con
        if len(zs)>0:
            root.constraints.append(sum(zs)-1)
        return root.constraints

def add_CDTree_Constraints(constraints, root):
    constrs = gen_CDTree_constraints(constraints, root)
    for con in constrs:
        constraints += [con >= 0]

def plan_boxes(swarm, PWLs, boxes, x0s, specs, bloat, limits=None, num_segs=None, tasks=None, vmax=3., MIPGap=1e-4, max_segs=None, tmax=None, hard_goals=None, size=0.11*4/2):
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

        Boxes = []
        constraints = []
        margin = cp.Variable(1)


        for idx_a in range(len(x0s)):
            x0 = x0s[idx_a]
            x0 = np.array(x0).reshape(-1).tolist()
            spec = specs[idx_a]

            dims = len(x0)
            PWL = PWLs[idx_a]
            # PWL = []
            # for temp in PWL_1:
            #     PWL.append(temp[0])

            Box = []
            for i in range(num_segs+1):
                # PWL.append([m.addVars(dims, lb=-GRB.INFINITY), m.addVar()])
                Box.append([cp.Variable((2,2))])
            Boxes.append(Box)

            # the initial constriant
            # m.addConstrs(PWL[0][0][i] == x0[i] for i in range(dims))
            # m.addConstr(PWL[0][1] == 0)

            # constraints += [Box[-1][0] == np.array([[0.01, 0], [0, 0.01]])]

            # if hard_goals is not None:
            #     goal = hard_goals[idx_a]
            #     m.addConstrs(PWL[-1][0][i] == goal[i] for i in range(dims))

            # if limits is not None:
            #     add_space_constraints(m, [P[0] for P in PWL], limits)

            # add_velocity_constraints(m, PWL, vmax=vmax)
            # add_time_constraints(m, PWL, tmax)
            # add volume constraint
            # temp = Boxes[0][0][0]
            # constraints += [cp.log_det(temp @ temp.T) >= 0.1]

            handleSpecTree(Boxes, margin, swarm, spec, PWL, bloat, size)
            add_CDTree_Constraints(constraints, spec.zs[0])

        if tasks is not None:
            for idx_agent in range(len(tasks)):
                for idx_task in range(len(tasks[idx_agent])):
                    handleSpecTree(Boxes,margin, swarm, tasks[idx_agent][idx_task], PWLs[idx_agent], bloat, size)

            conjunctions = []
            for idx_task in range(len(tasks[0])):
                disjunctions = [tasks[idx_agent][idx_task].zs[0] for idx_agent in range(len(tasks))]
                conjunctions.append(Disjunction(disjunctions))
            z = Conjunction(conjunctions)
            add_CDTree_Constraints(constraints, z)

        # add_mutual_clearance_constraints(m, PWLs, bloat)

        # obj = sum([L1Norm(m, _sub(PWL[i][0], PWL[i+1][0])) for PWL in PWLs for i in range(len(PWL)-1)])
        # obj = sum([PWL[-1][1] for PWL in PWLs])
        # temp = PWLs[0]
        # test1 = temp[0][1]
        # test2 = margin
        # # m.addConstr(test2 >= 0)
        # # m.setObjective(obj, GRB.MINIMIZE)
        # obj1 = LinExpr()
        obj = margin
        # m.setObjective(obj1, GRB.MAXIMIZE)
        prob = cp.Problem(cp.Maximize(obj),
                          constraints)

        # m.write("test.lp")
        # print('NumBinVars: %d'%m.getAttr('NumBinVars'))

        # m.computeIIS()
        # import ipdb;ipdb.set_treace()
        try:
            start = time.time()
            prob.solve()
            end = time.time()
            print('sovling it takes %.3f s'%(end - start))

            # for i, constraint in enumerate(constraints):
            #     lhs = constraint.expr.value  # Left-hand side of the constraint
            #     print(f"Value of constraint {i + 1} (LHS): {lhs}")


            Boxes_output = []
            for Box in Boxes:
                Box_output = []
                for B in Box:
                    Box_output.append([B[i].value for i in range(len(B))])
                Boxes_output.append(Box_output)
            margin_result = margin.value
            # A = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
            # b = np.array([1.5, 1.5, 1.5, 1.5], dtype=np.float64)
            # result_temp = display_mu(swarm, 18, PWL, Boxes, A, b)
            # print(result_temp)

            return Boxes_output, margin_result

        except Exception as e:
            print("Exception")
    return [None,]
