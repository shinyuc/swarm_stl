import sys
import numpy as np

from PWLPlan import plan, Node
from vis import vis
import pickle
from boxPlan import  optimize_boundingbox
import  datetime

def test(tmax, vmax):

    mu_thresholds = [5]
    agent_counts = [5]

    x0s = [[-1., -1]]

    wall_half_width = 0.02
    A = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    walls = []

    walls.append(np.array([-1.5, -1.5, -1.5, 1.5], dtype=np.float64))
    walls.append(np.array([1.5, 1.5, -1.5, 1.5], dtype=np.float64))
    walls.append(np.array([-1.5, 1.5, -1.5, -1.5], dtype=np.float64))
    walls.append(np.array([-1.5, 1.5, 1.5, 1.5], dtype=np.float64))

    obs = []
    for wall in walls:
        if wall[0] == wall[1]:
            wall[0] -= wall_half_width
            wall[1] += wall_half_width
        elif wall[2] == wall[3]:
            wall[2] -= wall_half_width
            wall[3] += wall_half_width
        else:
            raise ValueError('wrong shape for axis-aligned wall')
        wall *= np.array([-1, 1, -1, 1])
        obs.append((A, wall))

    b1 = np.array([1, -0.7, 0.25, 0.5], dtype=np.float64)
    B1 = (A, b1)
    b2 = np.array([0, 0.9, 1, -0.5], dtype=np.float64)
    B2 = (A, b2)
    b3 = np.array([-0.2, 0.7, -0.8, 1.2], dtype=np.float64)
    B3 = (A, b3)
    C = np.array([0.4, 0.4, 0.4, 0.4], dtype=np.float64)
    C = (A, C)
    D = np.array([1.45, 1.45, 1.45, 1.45], dtype=np.float64)
    D = (A, D)

    plots = [[[B1, ], 'y'], [[B2, ], 'r'], [[B3, ], 'g'], [[C, ], 'b'], [obs, 'k']]

    goal = [1, 1]

    test_list = [B1, B2, B3, C, D]





    # tmax = 5.
    # vmax = 3.

    notC = Node('negmu', info={'A': C[0], 'b': C[1]})
    B1 = Node('mu', info={'A': B1[0], 'b': B1[1]})
    B2 = Node('mu', info={'A': B2[0], 'b': B2[1]})
    B3 = Node('mu', info={'A': B3[0], 'b': B3[1]})
    alwaysD = Node('mu', info={'A': D[0], 'b': D[1]})

    phi_1 = Node('F', deps=[Node('A', deps=[B2, ], info={'int': [0, 5]}), ], info={'int': [0, tmax]})
    phi_2 = Node('F', deps=[Node('A', deps=[B3, ], info={'int': [0, 5]}), ], info={'int': [0, tmax]})
    phi_3 = Node('A', deps=[notC, ], info={'int': [0, tmax]})
    phi_4 = Node('A', deps=[alwaysD, ], info={'int': [0, tmax]})
    spec = [Node('and', deps=[phi_1, phi_2, phi_3, phi_4])]
    # spec = [Node('and', deps=[phi_1, phi_2, phi_3])]



    return x0s, plots, spec, test_list, agent_counts, mu_thresholds

def check_volume_constraint(swarm, Boxes):
    b = 0.01
    n = 2
    eta = 2
    final_Resutls = []
    for idx_a in range(len(Boxes)):
        final_resutls = []
        boxes = Boxes[idx_a]
        N = swarm[idx_a]
        value = 2 * np.log(eta * N * b ** n)
        for box in boxes:
            new_sigma = box
            sign, logdet = np.linalg.slogdet(new_sigma)
            if sign <= 0:
                raise ValueError("Matrix is not positive definite.")
            result = logdet - value
            final_resutls.append(result)
        final_Resutls.append(final_resutls)
    return final_Resutls




if __name__ == '__main__':
    num_rounds = 10
    zeta = 0.01
    # swarm = [2, 3, 4, 6]
    # swarm = [10, 15, 20, 30]
    goal = [1, 1]
    goals = [goal, ]
    tmax = 15.
    vmax = 1.
    num_segments = 9
    eta = 0.05
    x0s, plots, specs, test_list, agent_counts, mu_thresholds = test(tmax, vmax)
    swarm = agent_counts
    epsilon = 0.01

    # Setup of swarms
    running_time_list = []
    for num_round in range(num_rounds):
        if num_round == 0:
            Boxes = []
            initial_box = np.array([[0.01, 0], [0, 0.01]])
            for idx in range(len(x0s)):
                boxes = []
                for i in range(num_segments):
                    boxes.append(initial_box)
                Boxes.append(boxes)
        else:
            with open('../data/model_1/PWLs_output_bounding_Boxes.pkl', 'rb') as file:
                PWLs_loaded, Box_loaded, margin_loaded  = pickle.load(file)

            Boxes = Box_loaded

        # results = plan(x0s, Boxes, specs, bloat=eta, MIPGap=0.05, num_segs=num_segments, tmax=tmax, vmax=vmax)
        results = plan(x0s, Boxes, specs,
                       bloat=eta,zeta = zeta, MIPGap=0.05, num_segs=num_segments, tmax=tmax, vmax=vmax,
                       agent_counts = agent_counts,  mu_thresholds = mu_thresholds, hard_goals=goals)
        PWLs = results[0]
        running_time_1 = results[1]
        epsilon_result = results[2]
        print(PWLs)
        print(epsilon_result)
        results = vis(plots, PWLs, data_width=4, data_height = 4)

        with open('../data/model_1/PWLs_output_waypoints.pkl', 'wb') as file:
            pickle.dump((PWLs, Boxes, epsilon_result), file)



        # ======================= Bounding boxes optimization =======================
        with open('../data/model_1/PWLs_output_waypoints.pkl', 'rb') as file:
            PWLs_loaded, Box_loaded, epsilon_loaded = pickle.load(file)

        PWLs = PWLs_loaded
        epsilon_loaded = epsilon_loaded

        running_time_2, margin_result, Boxes_output = optimize_boundingbox(
            PWLs, swarm, epsilon_loaded, Box_loaded, test_list, num_segments, bloat_factor=eta, zeta = zeta)

        print(running_time_2)
        print(Boxes_output)
        print(margin_result)
        with open('../data/model_1/PWLs_output_bounding_Boxes.pkl', 'wb') as file:
            pickle.dump((PWLs, Boxes_output, margin_result), file)
        final_results = check_volume_constraint(swarm, Boxes_output)
        # Check if all elements in all sublists are positive
        all_positive = all(x > 0 for row in final_results for x in row)

        print(all_positive)
        print(margin_result)
        total_time = running_time_1 + running_time_2

        running_time_list.append([running_time_1, running_time_2, total_time])

        if margin_result >= 0:
            print("===========================================")
            print(f"In round {num_round}, the solution is feasible.")
            total = sum(total_t[2] for total_t in running_time_list)
            print(f"The total time is {total}.")
            print(f"The value of margin is {margin_result}.")
            with open('plot_low_level_env.pkl', 'wb') as file:
                pickle.dump((PWLs, Boxes_output, plots), file)
            filename = f"run_stlcg-1-auto_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            with open(filename, 'wb') as file:
                pickle.dump((total, margin_result, running_time_list), file)
            break
