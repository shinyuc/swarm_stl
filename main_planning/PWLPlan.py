from gurobipy import *
from math import *
import numpy as np
import time

M = 1e3
# a large M causes numerical issues and make the model infeasible to Gurobi
T_MIN_SEP = 1e-2
# see comments in GreaterThanZero
IntFeasTol  = 1e-1 * T_MIN_SEP / M

def setM(v):
    global M, IntFeasTol
    M = v
    IntFeasTol  = 1e-1 * T_MIN_SEP / M

EPS = 1e-4

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
    def __init__(self, deps = []):
        super(Conjunction, self).__init__()
        self.deps = deps
        self.constraints = []

class Disjunction(object):
    # disjunction node
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
    # for j in range(len(PWL)-1):
    for j in range(len(zphis)):
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

def mu(epsilon, i, PWL, boxes, bloat_factor, A, b):
    bloat_factor = np.max([0, bloat_factor]) # bloat is eta
    # this segment is fully contained in Ax<=b (shrinked)
    b = b.reshape(-1)
    num_edges = len(b)
    conjunctions = []
    for e in range(num_edges):
        a = A[e,:]
        for j in [i, i+1]:
            x = PWL[j][0]
            # conjunctions.append(b[e] - np.linalg.norm(a) * bloat_factor - sum([a[k]*x[k] for k in range(len(x))]) - EPS)
            conjunctions.append(b[e] - sum([a[k] * x[k] for k in range(len(x))])
                                - np.linalg.norm(a) * (bloat_factor+epsilon)-np.sqrt(a@boxes[i]@a.T))
            # conjunctions.append(b[e] - sum([a[k] * x[k] for k in range(len(x))])
            #                     - np.linalg.norm(a) * (bloat_factor + EPS))
    # need to add joint transform constraints

    return Conjunction(conjunctions)

def negmu(epsilon, i, PWL, boxes, bloat_factor, A, b):
    # this segment is outside Ax<=b (bloated)
    b = b.reshape(-1)
    num_edges = len(b)
    disjunctions = []
    for e in range(num_edges):
        a = A[e,:]
        conjunctions = []
        for j in [i, i+1]:
            x = PWL[j][0]
            # conjunctions.append(sum([a[k]*x[k] for k in range(len(x))]) - (b[e] + np.linalg.norm(a) * bloat_factor) - EPS)
            conjunctions.append(sum([a[k]*x[k] for k in range(len(x))]) - b[e]
                                - np.linalg.norm(a) * (bloat_factor + epsilon) - np.sqrt(a @ boxes[i] @ a.T))
            # conjunctions.append(sum([a[k] * x[k] for k in range(len(x))]) - b[e]
            #                     - np.linalg.norm(a) * (bloat_factor + EPS))
        disjunctions.append(Conjunction(conjunctions))
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
        x1, t1 = PWL[i]
        x2, t2 = PWL[i+1]
        model.addConstr(t2 - t1 >= T_MIN_SEP)

def add_velocity_constraints(model, PWL, vmax=3):
    for i in range(len(PWL)-1):
        x1, t1 = PWL[i]
        x2, t2 = PWL[i+1]
        # squared_dist = sum([(x1[j]-x2[j])*(x1[j]-x2[j]) for j in range(len(x1))])
        # model.addConstr(squared_dist <= (vmax**2) * (t2 - t1) * (t2 - t1))
        L1_dist = L1Norm(model, _sub(x1,x2))
        model.addConstr(L1_dist <= vmax * (t2 - t1))

def disjoint_segments(model,  sqrt_max_eig_1, sqrt_max_eig_2, epsilon,zeta, seg1, seg2,box1,box2, bloat):
    assert(len(seg1) == 2)
    assert(len(seg2) == 2)
    # eigenvalues1, _ = np.linalg.eig(box1)
    # largest1 = np.max(np.real(eigenvalues1))
    # eigenvalues2, _ = np.linalg.eig(box2)
    # largest2 = np.max(np.real(eigenvalues2))
    # EPS = 0.001
    # if abs(np.sqrt(largest1) - sqrt_max_eig_1) >= EPS or abs(np.sqrt(largest2) - sqrt_max_eig_2) >= EPS:
    #     print("False, they are not the same")

    # return (0.5 * L1Norm(model, _sub(_add(seg1[0], seg1[1]), _add(seg2[0], seg2[1])))
    #         - 0.5 * (L1Norm(model, _sub(seg1[0], seg1[1])) + L1Norm(model, _sub(seg2[0], seg2[1])))
    #         - np.sqrt(len(seg1[0]))*(2*bloat +np.sqrt(largest1) +np.sqrt(largest2)) - (zeta+epsilon)*np.sqrt(len(seg1[0])))
    return (0.5 * L1Norm(model, _sub(_add(seg1[0], seg1[1]), _add(seg2[0], seg2[1])))
            - 0.5 * (L1Norm(model, _sub(seg1[0], seg1[1])) + L1Norm(model, _sub(seg2[0], seg2[1])))
            - np.sqrt(len(seg1[0])) * (2 * bloat + sqrt_max_eig_1 + sqrt_max_eig_2) - (zeta + epsilon) * np.sqrt(
                len(seg1[0])))

def add_mutual_clearance_constraints(model,max_eigenvalues_list, epsilon,zeta, PWLs, Boxes, bloat):
    for i in range(len(PWLs)):
        for j in range(i+1, len(PWLs)):
            PWL1 = PWLs[i]
            PWL2 = PWLs[j]
            Box1 = Boxes[i]
            Box2 = Boxes[j]
            for k in range(len(PWL1)-1):
                for l in range(len(PWL2)-1):
                    x11, t11 = PWL1[k]
                    x12, t12 = PWL1[k+1]
                    x21, t21 = PWL2[l]
                    x22, t22 = PWL2[l+1]
                    box1 = Box1[k]
                    box2 = Box2[l]
                    sqrt_max_eig_1 = max_eigenvalues_list[i][k]
                    sqrt_max_eig_2 = max_eigenvalues_list[j][l]
                    z_noIntersection = noIntersection(t11, t12, t21, t22)
                    z_disjoint_segments = disjoint_segments(model, sqrt_max_eig_1, sqrt_max_eig_2, epsilon,zeta, [x11, x12], [x21, x22],box1,box2, bloat)
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
    for dep in spec.deps:
        clearSpecTree(dep)
    spec.zs = []

def handleSpecTree(epsilon, spec, PWL, boxes, bloat_factor, size, agent_count, mu_thresholds):
    for dep in spec.deps:
        handleSpecTree(epsilon, dep, PWL, boxes, bloat_factor, size, agent_count, mu_thresholds)
    if len(spec.zs) == len(PWL)-1:
        return
    elif len(spec.zs) > 0:
        raise ValueError('incomplete zs')
    if spec.op == 'mu':
        spec.zs = [mu(epsilon, i, PWL, boxes, bloat_factor, spec.info['A'], spec.info['b']) for i in range(len(PWL)-1)]
    elif spec.op == 'negmu':
        spec.zs = [negmu(epsilon, i, PWL, boxes, bloat_factor , spec.info['A'], spec.info['b']) for i in range(len(PWL)-1)]
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

# def plan(x0s, specs, bloat, limits=None, num_segs=None, tasks=None, vmax=3., MIPGap=1e-4, max_segs=None, tmax=None, hard_goals=None, size=0.11*4/2):
def plan(x0s, Boxes, specs, bloat, zeta=0.05, limits=None, num_segs=None, tasks=None, vmax=3., MIPGap=1e-4, max_segs=None, tmax=None,
             hard_goals=None, size=0.01 * 4 / 2, agent_counts=None, mu_thresholds=None):


    max_eigenvalues_list = []
    for box in Boxes:
        sqrt_max_eigenvalues = [np.sqrt(np.max(np.linalg.eigvals(matrix).real)) for matrix in box]
        max_eigenvalues_list.append(sqrt_max_eigenvalues)



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

        m.setParam(GRB.Param.IntFeasTol, IntFeasTol)
        m.setParam(GRB.Param.MIPGap, MIPGap)

        epsilon = m.addVar(lb=-GRB.INFINITY)
        # epsilon1 = m.addVar(lb=-GRB.INFINITY)
        # m.addConstr(epsilon1 >= epsilon)
        m.update()

        for idx_a in range(len(x0s)):
            x0 = x0s[idx_a]
            x0 = np.array(x0).reshape(-1).tolist()
            spec = specs[idx_a]

            dims = len(x0)
            boxes = Boxes[idx_a]
            PWL = []
            for i in range(num_segs+1):
                PWL.append([m.addVars(dims, lb=-GRB.INFINITY), m.addVar()])
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
            add_time_constraints(m, PWL, tmax)

            # handleSpecTree(epsilon, spec, PWL, boxes, bloat, size)
            print(f"Swarm {idx_a} is being checked")
            handleSpecTree(epsilon, spec, PWL, boxes, bloat, size,
                           agent_counts[idx_a], mu_thresholds)
            add_CDTree_Constraints(m, spec.zs[0])


        if tasks is not None:
            for idx_agent in range(len(tasks)):
                for idx_task in range(len(tasks[idx_agent])):
                    handleSpecTree(epsilon,tasks[idx_agent][idx_task], PWLs[idx_agent], Boxes[idx_agent]
                                   , bloat, size, agent_counts[idx_agent], mu_thresholds)

            conjunctions = []
            if agent_counts is not None:

                for idx_task in range(len(tasks[0])):
                    # disjunctions = [tasks[idx_agent][idx_task].zs[0] for idx_agent in range(len(tasks))]
                    disjunctions = []
                    for idx_agent in range(len(tasks)):
                        N_mu = mu_thresholds[idx_task]
                        agent_count = agent_counts[idx_agent]
                        if agent_count >= N_mu:
                            disjunction = tasks[idx_agent][idx_task].zs[0]
                            print(f"We are checking agent-{idx_agent}.")
                            print(f"We are checking goal region {idx_task}.")
                            disjunctions.append(disjunction)

                    conjunctions.append(Disjunction(disjunctions))
                # z = Conjunction(conjunctions)
                # add_CDTree_Constraints(m, z)
            else:
                for idx_task in range(len(tasks[0])):
                    disjunctions = [tasks[idx_agent][idx_task].zs[0] for idx_agent in range(len(tasks))]
                    print(f"We are checking agent-{idx_agent}.")
                    print(f"We are checking goal region {idx_task}.")
                    conjunctions.append(Disjunction(disjunctions))
                z = Conjunction(conjunctions)
                add_CDTree_Constraints(m, z)

            z = Conjunction(conjunctions)
            add_CDTree_Constraints(m, z)
            # for idx_task in range(len(tasks[0])):
            #     # disjunctions = [tasks[idx_agent][idx_task].zs[0] for idx_agent in range(len(tasks))]
            #     disjunctions = []
            #     for idx_agent in range(len(tasks)):
            #         N_mu = mu_thresholds[idx_task]
            #         agent_count = agent_counts[idx_agent]
            #         if agent_count >= N_mu:
            #             disjunction = tasks[idx_agent][idx_task].zs[0]
            #             print(idx_agent)
            #             print(idx_task)
            #             disjunctions.append(disjunction)
            #
            #     conjunctions.append(Disjunction(disjunctions))
            # z = Conjunction(conjunctions)
            # add_CDTree_Constraints(m, z)

        add_mutual_clearance_constraints(m,max_eigenvalues_list, epsilon,zeta, PWLs, Boxes, bloat)


        # obj = sum([PWL[-1][1] for PWL in PWLs])
        # m.setObjective(obj, GRB.MINIMIZE)

        obj1 = LinExpr()
        obj1 += epsilon
        m.setObjective(obj1, GRB.MAXIMIZE)




        m.write("test.lp")
        print('NumBinVars: %d'%m.getAttr('NumBinVars'))

        # m.computeIIS()
        # import ipdb;ipdb.set_treace()
        try:
            start = time.time()
            m.optimize()
            end = time.time()
            print('sovling it takes %.3f s'%(end - start))
            total_time = end - start
            PWLs_output = []
            for PWL in PWLs:
                PWL_output = []
                for P in PWL:
                    PWL_output.append([[P[0][i].X for i in range(len(P[0]))], P[1].X])
                PWLs_output.append(PWL_output)
            epsilon_result = epsilon.X
            m.dispose()
            return [PWLs_output, total_time, epsilon_result]
        except Exception as e:
            m.dispose()
    return [None,]
