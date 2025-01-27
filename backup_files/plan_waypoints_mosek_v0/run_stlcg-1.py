import sys
import numpy as np
from vis import vis

from waypoint_plan import plan_waypoints, Node
from safe_cons_waypoint import *
import matplotlib.pyplot as plt

from ellipsoid_plt import plot_ellipsoid

def test(swarm,boxes,num_segs=9):
    x0 = [-1., -1]

    wall_half_width = 0.05
    A = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    walls = []

    walls.append(np.array([-1.5, -1.5, -1.5, 1.5], dtype = np.float64))
    walls.append(np.array([1.5, 1.5, -1.5, 1.5], dtype = np.float64))
    walls.append(np.array([-1.5, 1.5, -1.5, -1.5], dtype = np.float64))
    walls.append(np.array([-1.5, 1.5, 1.5, 1.5], dtype = np.float64))

    obs = []
    for wall in walls:
        if wall[0]==wall[1]:
            wall[0] -= wall_half_width
            wall[1] += wall_half_width
        elif wall[2]==wall[3]:
            wall[2] -= wall_half_width
            wall[3] += wall_half_width
        else:
            raise ValueError('wrong shape for axis-aligned wall')
        wall *= np.array([-1,1,-1,1])
        obs.append((A, wall))

    b1 = np.array([1, -0.7, 0.25, 0.5], dtype = np.float64)
    B1 = (A, b1)
    b2 = np.array([0, 0.9, 1, -0.5], dtype = np.float64)
    B2 = (A, b2)
    b3 = np.array([-0.2, 0.7, -0.8, 1.2], dtype = np.float64)
    B3 = (A, b3)
    C = np.array([0.4, 0.4, 0.4, 0.4], dtype = np.float64)
    C = (A, C)
    # inside the wall
    D = np.array([1.5, 1.5, 1.5, 1.5], dtype=np.float64)
    D = (A, D)

    goal = [1,1]

    plots = [[[B1,], 'y'], [[B2,], 'r'], [[B3,], 'g'], [[C,], 'b'], [obs, 'k']]
    tmax = 15.
    vmax = 1.

    notC = Node('negmu', info={'A':C[0], 'b':C[1]})
    B1 = Node('mu', info={'A':B1[0], 'b':B1[1]})
    B2 = Node('mu', info={'A':B2[0], 'b':B2[1]})
    B3 = Node('mu', info={'A':B3[0], 'b':B3[1]})
    alwaysD = Node('mu', info={'A':D[0], 'b':D[1]})

    phi_1 = Node('F', deps=[Node('A', deps=[B2,], info={'int':[0,5]}),], info={'int':[0,tmax]})
    phi_2 = Node('F', deps=[Node('A', deps=[B3,], info={'int':[0,5]}),], info={'int':[0,tmax]})
    phi_3 = Node('A', deps=[notC,], info={'int':[0,tmax]})
    phi_4 = Node('A', deps=[alwaysD, ], info={'int': [0, tmax]})
    spec = Node('and', deps=[phi_1, phi_2, phi_3, phi_4])

    x0s = [x0,]
    specs = [spec,]
    goals = [goal, ]
    # PWL = plan(swarm, x0s, specs, bloat=0.05, size=0.11/2, num_segs=9, tmax=tmax, vmax=vmax, hard_goals=goals)
    PWL = plan_waypoints(swarm,boxes, x0s, specs, bloat=0.05, size=0.11/2, num_segs=9, tmax=tmax, vmax=vmax, hard_goals=goals)

    return x0s, plots, PWL, margin

if __name__ == '__main__':
    A = np.array([[0, 0], [0, 0]])
    B = np.array([[1, 0], [0, 1]])
    dynamics = [A, B]
    total_num_agents = [2, 3, 4, 5]
    print(dynamics[1])
    print(total_num_agents[0])
    Q = np.array([[1, 0], [0, 1]])
    R = np.array([[1, 0], [0, 1]])
    lqr_set = [Q, R]
    swarm = SingleSwarm(1, dynamics, total_num_agents, lqr_set)
    # box = [a[k] * x1[k] for i in range(len(x1))])
    num_segs = 9
    # Note: Sigma of bounding box = box@box.T
    initial_box = np.array([[0.001, 0], [0, 0.001]])
    boxes = []
    for i in range(num_segs+1):
        boxes.append(initial_box)
    _, plots, PWLs, margin = test(swarm,boxes, num_segs=9)

    results = vis(test, swarm,boxes)
    print(f"\nThe value of margin is {margin}.")
    PWL = PWLs[0]
    print(len(PWL))
    distance = []
    for i in range(len(PWL)-1):
        distance.append(np.array(PWL[i+1][0]) - np.array(PWL[i][0]))
    print(distance)
    test_1_norm = [np.linalg.norm(distance[i], ord=1) for i in range(len(distance))]
    test_root = [np.linalg.norm(distance[i], ord=2) for i in range(len(distance))]
    print(test_1_norm)
    print(test_root)



    # for i in range(1, num_segs-1):
    #     x0, P, overline_beta = positive_invariant_set(swarm, i, PWLs[0], epsilon=1e-1)
    #     print(P)
    #     print(overline_beta)
    #     plot_ellipsoid(P, overline_beta, x0)
    # plt.show()
    plt.show()