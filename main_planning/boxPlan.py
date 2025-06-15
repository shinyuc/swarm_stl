import pickle
import numpy as np
import cvxpy as cp
import math
import control as ct
import time
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.metrics_utils import *

# def _sub(x1, x2):
#     return x1 - x2

def _sub(x1, x2):
    x1 = np.array(x1)
    x2 = np.array(x2)
    return x1 - x2

# def _add(x1, x2):
#     return x1 + x2

def _add(x1, x2):
    x1 = np.array(x1)
    x2 = np.array(x2)
    return x1 + x2

def L1Norm(x):
    """Compute the L1 norm using cvxpy."""
    return cp.norm(x, 1)

def L2Norm(x):
    """Compute the L1 norm using cvxpy."""
    return cp.norm(x, 2)

def test_mu(constraints, last_margin, new_margin, swarm, i, PWL, Box, last_Box, test_list, bloat_factor):

    for test_condition in test_list:
        test_A = test_condition[0]
        test_b = test_condition[1]
        b = test_b.reshape(-1)
        num_edges = len(b)
        x1 = PWL[i][0]
        for e in range(num_edges):
            a = test_A[e, :]
            b_e = b[int(e)]
            result1 = - a @ x1 + b_e - np.linalg.norm(a) * (
                    bloat_factor + last_margin) - np.sqrt(a @ last_Box[i] @ a.T)
            if result1 >=0:
                new_sigma = Box[i][0]
                temp_sigma_1 = a @ new_sigma
                # new_constraint1 = - a @ x1 + b_e - np.linalg.norm(a) * (bloat_factor
                #                                                                 + new_margin) - L1Norm(temp_sigma_1)
                new_constraint1 = - a @ x1 + b_e - np.linalg.norm(a) * (bloat_factor
                                                                        + new_margin) - L2Norm(temp_sigma_1)
                constraints += [new_constraint1 >= 0]

        x2 = PWL[i + 1][0]
        for e in range(num_edges):
            a = test_A[e, :]
            b_e = b[int(e)]
            result2 = - a @ x2 + b_e - np.linalg.norm(a) * (
                    bloat_factor + last_margin) - np.sqrt(a @ last_Box[i] @ a.T)
            if result2 >=0:
                new_sigma = Box[i][0]
                temp_sigma_2 = a @ new_sigma
                # new_constraint2 = - a @ x2 + b_e - np.linalg.norm(a) * (bloat_factor
                #                                                                 + new_margin) - L1Norm(temp_sigma_2)
                new_constraint2 = - a @ x2 + b_e - np.linalg.norm(a) * (bloat_factor
                                                                        + new_margin) - L2Norm(temp_sigma_2)
                constraints += [new_constraint2 >= 0]



            # conjunctions.append(b[e] - np.linalg.norm(a) * bloat_factor - sum([a[k]*x[k] for k in range(len(x))]) - EPS)
        # result2 = - test_A @ x1 + b - np.linalg.norm(test_A) * (bloat_factor
        #                                                         + last_margin) - np.sqrt(test_A @ last_Box[i] @ test_A.T)

        # if result1 >=0 and result2 >= 0:
        #     new_sigma = Box[i]
        #     temp_sigma_1 = test_A @ new_sigma
        #     new_constraint1 = - test_A @ x1 + b - np.linalg.norm(test_A) * (bloat_factor
        #                                                                     + new_margin) - L1Norm(temp_sigma_1)
        #     constraints += [new_constraint1 >= 0]
        #     new_constraint2 = - test_A @ x2 + b - np.linalg.norm(test_A) * (bloat_factor
        #                                                                     + new_margin) - L1Norm(temp_sigma_1)
        #     constraints += [new_constraint2 >= 0]



def test_negmu(constraints, last_margin, new_margin, swarm, i, PWL, Box, last_Box, test_list, bloat_factor):
    for test_condition in test_list:
        test_A = test_condition[0]
        test_b = test_condition[1]
        b = test_b.reshape(-1)
        num_edges = len(b)
        x1 = PWL[i][0]
        for e in range(num_edges):
            a = test_A[e, :]
            b_e = b[int(e)]
            result1 =  a @ x1 - b_e - np.linalg.norm(a) * (bloat_factor + last_margin) - np.sqrt(a @ last_Box[i] @ a.T)
            if result1 >=0:
                new_sigma = Box[i][0]
                temp_sigma_1 = a @ new_sigma
                # new_constraint1 = a @ x1 - b_e - np.linalg.norm(a) * (bloat_factor + new_margin) - L1Norm(temp_sigma_1)
                new_constraint1 = a @ x1 - b_e - np.linalg.norm(a) * (bloat_factor + new_margin) - L2Norm(temp_sigma_1)
                constraints += [new_constraint1 >= 0]

        x2 = PWL[i + 1][0]
        for e in range(num_edges):
            a = test_A[e, :]
            b_e = b[int(e)]
            result2 =  a @ x2 - b_e - np.linalg.norm(a) * (bloat_factor + last_margin) - np.sqrt(a @ last_Box[i] @ a.T)
            if result2 >=0:
                new_sigma = Box[i][0]
                temp_sigma_2 = a @ new_sigma
                # new_constraint2 = a @ x2 - b_e - np.linalg.norm(a) * (bloat_factor + new_margin) - L1Norm(temp_sigma_2)
                new_constraint2 = a @ x2 - b_e - np.linalg.norm(a) * (bloat_factor + new_margin) - L2Norm(temp_sigma_2)
                constraints += [new_constraint2 >= 0]


def test_volume_constraint(constraints, swarm, idx_a, Box, i):
    # log_det(A) is to calculate log_{e}(det(A))
    # set eta = 2, b = 0.01, n = 2
    EPS_relaxation = 2
    b = 0.01
    n = 2
    N = swarm[idx_a]
    eta = 2
    value = 2 * np.log(eta * N * b**n)
    new_sigma = Box[i][0]
    # result = cp.log_det(new_sigma) + 4.5
    #
    result = 2 * cp.log_det(new_sigma)  - value - EPS_relaxation
    constraints += [result >= 0]

def test_disjoint_segments(t11,t12,t21,t22, bloat,zeta, constraints, swarm, seg1, seg2, last_box1, last_box2,new_box1,new_box2, last_margin, new_margin):
    eigenvalues1, _ = np.linalg.eig(last_box1)
    largest1 = np.max(np.real(eigenvalues1))
    eigenvalues2, _ = np.linalg.eig(last_box2)
    largest2 = np.max(np.real(eigenvalues2))
    last_result = (0.5 * np.linalg.norm(_sub(_add(seg1[0], seg1[1]), _add(seg2[0], seg2[1])), ord=1)
                   - 0.5 * (np.linalg.norm( _sub(seg1[0], seg1[1]), ord=1) + np.linalg.norm(_sub(seg2[0], seg2[1]), ord=1))
                   - np.sqrt(len(seg1[0]))*(2*bloat +largest1 +largest2) - (last_margin+zeta) *np.sqrt(len(seg1[0])))
    if last_result >= 0:
        EPS = 1e-2  # or any small positive value
        # if t12 - t21 >= -EPS and t22 - t11 >= -EPS:
        #     print("time overlap, distance >= 0")
        # else:
        #     print("time not overlap, distance >= 0")
        new_max_singular_value_1 = cp.norm(new_box1[0], 2)
        new_max_singular_value_2 = cp.norm(new_box2[0], 2)
        result = (0.5 * np.linalg.norm(_sub(_add(seg1[0], seg1[1]), _add(seg2[0], seg2[1])), ord=1)
                   - 0.5 *(np.linalg.norm( _sub(seg1[0], seg1[1]), ord=1) + np.linalg.norm(_sub(seg2[0], seg2[1]), ord=1))
                   - np.sqrt(len(seg1[0]))* (2*bloat +new_max_singular_value_1 +new_max_singular_value_2) - (zeta+new_margin) *np.sqrt(len(seg1[0])))
        constraints += [result >= 0]
    else:
        EPS = 1e-2  # or any small positive value
        if t12 - t21 >= 0 and t22 - t11 >= 0:
            print("time overlap, distance < 0")
        # else:
        #     print("time not overlap, distance < 0")

def test_disjoint_segments_new(bloat,zeta, constraints, swarm, seg1, seg2, last_box1, last_box2,new_box1,new_box2, last_margin, new_margin):
    new_max_singular_value_1 = cp.norm(new_box1[0], 2)
    new_max_singular_value_2 = cp.norm(new_box2[0], 2)
    result = (0.5 * np.linalg.norm(_sub(_add(seg1[0], seg1[1]), _add(seg2[0], seg2[1])), ord=1)
              - 0.5 * (np.linalg.norm(_sub(seg1[0], seg1[1]), ord=1) + np.linalg.norm(_sub(seg2[0], seg2[1]), ord=1))
              - np.sqrt(len(seg1[0])) * (2 * bloat + new_max_singular_value_1 + new_max_singular_value_2) - (
                          zeta + new_margin) * np.sqrt(len(seg1[0])))
    constraints += [result >= 0]

def add_mutual_clearance_constraints(bloat, zeta,constraints,swarm, PWLs,Boxes, last_Boxes, last_margin, new_margin):
    for i in range(len(PWLs)):
        for j in range(i+1, len(PWLs)):
            PWL1 = PWLs[i]
            last_Box1 = last_Boxes[i]
            new_Box1 = Boxes[i]
            PWL2 = PWLs[j]
            last_Box2 = last_Boxes[j]
            new_Box2 = Boxes[j]
            for k in range(len(PWL1)-1):
                for l in range(len(PWL2)-1):
                    x11, t11 = PWL1[k]
                    x12, t12 = PWL1[k+1]
                    last_box1 = last_Box1[k]
                    new_box1 = new_Box1[k]
                    x21, t21 = PWL2[l]
                    x22, t22 = PWL2[l+1]
                    last_box2 = last_Box2[l]
                    new_box2 = new_Box2[l]
                    # test_disjoint_segments(t11,t12,t21,t22,bloat, zeta,constraints,swarm, [x11, x12], [x21, x22], last_box1, last_box2,new_box1,new_box2, last_margin, new_margin)
                    EPS = 1e-4
                    if t12 - t21 >= EPS and t22 - t11 >= EPS:
                        test_disjoint_segments_new(bloat, zeta, constraints, swarm, [x11, x12], [x21, x22], last_box1,
                                               last_box2, new_box1, new_box2, last_margin, new_margin)


def optimize_boundingbox(PWLs, swarm, margin_loaded, Box_loaded, test_list, num_segs, bloat_factor, zeta = 0.05):

    PWL = PWLs[0]
    last_margin = margin_loaded
    last_Box = Box_loaded


    constraints = []
    new_margin = cp.Variable()

    Boxes = []

    for j in range(len(PWLs)):
        Box = []
        for i in range(num_segs):
            Box.append([cp.Variable((2, 2))])
        Boxes.append(Box)

    for idx_a in range(len(PWLs)):
        PWL = PWLs[idx_a]
        last_Box = Box_loaded[idx_a]
        Box = Boxes[idx_a]
        for i in range(num_segs):
            test_mu(constraints, last_margin, new_margin, swarm, i, PWL, Box, last_Box, test_list, bloat_factor)
            test_negmu(constraints, last_margin, new_margin, swarm, i, PWL, Box, last_Box, test_list, bloat_factor)
            test_volume_constraint(constraints, swarm, idx_a, Box, i)

    add_mutual_clearance_constraints(bloat_factor,zeta, constraints, swarm, PWLs, Boxes, Box_loaded, last_margin, new_margin)
    obj = new_margin
    prob = cp.Problem(cp.Maximize(obj),
                      constraints)
    start = time.time()
    prob.solve(solver=cp.MOSEK)
    print("Solver used:", prob.solver_stats.solver_name)
    end = time.time()
    print('sovling it takes %.3f s' % (end - start))
    running_time = end - start
    Boxes_output = []
    for Box in Boxes:
        Box_output = []
        for B in Box:
            # Box_output.append([np.array(B[i].value) for i in range(len(B))])
            # Box_output.append([B[i].value for i in range(len(B))])
            values = [B[i].value for i in range(len(B))]
            # print(values)
            array_values = np.array(values[0])
            temp = array_values @ array_values.T
            tol = 1e-6
            temp[np.abs(temp) < tol] = 0
            Box_output.append(temp)
        Boxes_output.append(Box_output)
    margin_result = new_margin.value

    # print(Boxes_output)
    # print(margin_result)
    # with open('../data/model_2-auto/PWLs_output_bounding_Boxes.pkl', 'wb') as file:
    #     pickle.dump((PWLs, Boxes_output, margin_result), file)

    return running_time, margin_result, Boxes_output