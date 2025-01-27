import numpy as np
import control as ct
import math
from single_swarm_test import SingleSwarm

def dynamics_equilibrium(swarm, u_eq):
    # res_1>=0 and res_2>=0 ensure the existence of equilibrium
    p_s_i = swarm.end_pos
    A = swarm.dynamics[0]
    B = swarm.dynamics[1]
    res_1 = A @ p_s_i + B @ u_eq
    return res_1

def volume_constraint(swarm, r=1, eta = 2):
    # r: radius of the agent
    num_agents = swarm.num_agents
    sigma = swarm.bounding_box_sigma
    det_Sigma = np.linalg.det(sigma)
    vol_ellipsoid = math.sqrt(det_Sigma)
    vol_single_agent = math.pi*(r**2)
    vol_agents = num_agents * vol_single_agent
    return vol_agents - vol_agents
