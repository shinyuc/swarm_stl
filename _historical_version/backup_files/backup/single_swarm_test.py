import numpy as np
import control as ct
import math
from safe_constraint_func import *

if __name__ == "__main__":
    A = np.array([[0, 0], [0, 0]])
    B = np.array([[1, 0], [0, 1]])
    dynamics = [A,B]
    total_num_agents = [2, 3, 4,5]
    print(dynamics[1])
    Q = np.array([[1, 0], [0, 1]])
    R = np.array([[1, 0], [0, 1]])
    lqr_set = [Q, R]
    bounding_box_A = np.array([[1, 0], [0, 1]])
    init_pos_temp = np.array([0, 0])
    init_pos = np.vstack(init_pos_temp)
    end_pos_temp = np.array([4, 3])
    end_pos = np.vstack(end_pos_temp)
    # swarm_1 = SingleSwarm(1,dynamics,total_num_agents,init_pos,end_pos,bounding_box_A,lqr_set)
    # new_init_pos = np.array([-1, -1])
    # swarm_0 = SingleSwarm(0, dynamics, total_num_agents,new_init_pos, init_pos, bounding_box_A, lqr_set)
    # new_end_pos = np.array([2, 2])
    # swarm_2 = SingleSwarm(2, dynamics, total_num_agents, end_pos, new_end_pos, bounding_box_A, lqr_set)
    # print(swarm_1.bounding_box_sigma)
    # print(positive_invariant_set(swarm_1, epsilon=1e-1))

    # a_q = np.array([1, 1]).T
    # b_q = 10
    # res_1 = safe_2_1(swarm_1, a_q, b_q, epsilon=1e-1)
    # print("the result is:")
    # print(res_1)
    #
    # res_2 = safe_2_2(swarm_1,swarm_0, a_q,b_q,epsilon=1e-1, epsilon_r = 1)
    # print("the result is:")
    # print(res_2)
    #
    # res_3 = safe_2_3(swarm_1, swarm_2, a_q, b_q, epsilon=1e-1, epsilon_r=1)
    # print("the result is:")
    # print(res_2)