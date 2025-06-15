import sys
import numpy as np
import datetime

from PWLPlan import plan, Node
from vis import vis
import pickle
from boxPlan import  optimize_boundingbox

def test(tmax, vmax):
    # agent_counts = [15, 15, 25, 40]  # number of agents in each swarm
    # agent_counts = [8, 4]
    # mu_thresholds = [3, 1, 6, 4]
    # agent_counts = [5, 2, 8, 5]
    mu_thresholds = [1, 1, 1, 1]
    agent_counts = [1, 1, 1, 1]
    # mu_thresholds = {'goal0': 10, 'goal1': 15, 'goal2': 20, 'goal3': 30}
    # mu_thresholds = {'goal0': 2, 'goal1': 3, 'goal2': 4, 'goal3': 6}
    # mu_thresholds = {'goal0': 2, 'goal1': 3, 'goal2': 6, 'goal3': 4}

    # mu_thresholds = [2, 6]

    x0s = [[2, 0.5], [4, 0.5], [6, 0.5], [8, 0.5]]

    wall_half_width = 0.05
    A = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    walls = []

    walls.append(np.array([0, 0, 0, 4], dtype=np.float64))
    walls.append(np.array([10, 10, 0, 4], dtype=np.float64))
    walls.append(np.array([0, 10, 0, 0], dtype=np.float64))
    walls.append(np.array([0, 10, 4, 4], dtype=np.float64))

    walls.append(np.array([0, 4., 2, 2], dtype=np.float64))
    walls.append(np.array([5., 10, 2, 2], dtype=np.float64))

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

    b1 = np.array([-1.5, 2.5, -3, 4], dtype=np.float64)
    b2 = np.array([-3.5, 4.5, -3, 4], dtype=np.float64)
    b3 = np.array([-5.5, 6.5, -3, 4], dtype=np.float64)
    b4 = np.array([-7.5, 8.5, -3, 4], dtype=np.float64)
    goals = [(A, b1), (A, b2), (A, b3), (A, b4)]

    tmax = 5.
    vmax = 3.

    specs = []
    for i in range(4):
        # avoids = [Node('negmu', info={'A':A, 'b':b}) for A, b in obs]
        # avoids += [Node('negmu', info={'A':goals1[j][0], 'b':goals1[j][1]}) for j in range(4) if j is not i]
        # avoids += [Node('negmu', info={'A':goals2[j][0], 'b':goals2[j][1]}) for j in range(4) if j is not i]
        # avoid_obs = Node('and', deps=avoids)

        # # notGoal1 = Node('negmu', info={'A':goals1[i][0], 'b':goals1[i][1]})
        # goal1 = Node('mu', info={'A':goals1[i][0], 'b':goals1[i][1]})
        # finally_reach_goal1 = Node('F', deps=[goal1,], info={'int':[0, tmax]})
        # # goal2 = Node('mu', info={'A':goals2[i][0], 'b':goals2[i][1]})
        # # finally_reach_goal2 = Node('F', deps=[goal2,], info={'int':[0, tmax]})
        # # trans = [Node('or', deps=[notGoal1, finally_reach_goal2]),]
        # # always = Node('A', deps=[Node('and', deps=[avoid_obs,]+trans), ], info={'int':[0, tmax]})
        # always = Node('A', deps=[Node('and', deps=[avoid_obs,]), ], info={'int':[0, tmax]})

        # specs.append(Node('and', deps=[always, finally_reach_goal1]))

        avoids = [Node('negmu', info={'A': A, 'b': b}) for A, b in obs]
        avoids += [Node('negmu', info={'A': goals[j][0], 'b': goals[j][1]}) for j in range(4) if j is not i]
        avoid_obs = Node('and', deps=avoids)
        always_avoid_obs = Node('A', deps=[avoid_obs, ], info={'int': [0, tmax]})
        reach_goal = Node('mu', info={'A': goals[i][0], 'b': goals[i][1]})
        finally_reach_goal = Node('F', deps=[reach_goal, ], info={'int': [0, tmax]})
        specs.append(Node('and', deps=[always_avoid_obs, finally_reach_goal]))

    test_list = [(A, b1), (A, b2), (A, b3), (A, b4)]
    for obs_element in obs:
        test_list.append(obs_element)

    plots = [[goals, 'g'], [obs, 'k']]
    # return x0s, plots, specs, test_list, agent_counts, mu_thresholds, finally_visit_observs
    return x0s, plots, specs, test_list, agent_counts, mu_thresholds

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
    # swarm = [2, 3, 4, 6]
    # swarm = [10, 15, 20, 30]
    zeta = 0.01

    tmax = 5.
    vmax = 3.
    num_segments = 6
    eta = 0.05
    x0s, plots, specs, test_list, agent_counts, mu_thresholds= test(tmax, vmax)
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
            with open('../data/model_2-assigned/PWLs_output_bounding_Boxes.pkl', 'rb') as file:
                PWLs_loaded, Box_loaded, margin_loaded  = pickle.load(file)

            Boxes = Box_loaded

        # results = plan(x0s, Boxes, specs, bloat=eta, MIPGap=0.05, num_segs=num_segments, tmax=tmax, vmax=vmax)
        results = plan(x0s, Boxes, specs,
                       bloat=eta, MIPGap=0.05, num_segs=num_segments, tmax=tmax, vmax=vmax,
                       agent_counts = agent_counts,  mu_thresholds = mu_thresholds)
        PWLs = results[0]
        running_time_1 = results[1]
        epsilon_result = results[2]
        print(PWLs)
        print(epsilon_result)
        results = vis(plots, PWLs)

        final_results = check_volume_constraint(swarm, Boxes)
        # Check if all elements in all sublists are positive
        all_positive = all(x > 0 for row in final_results for x in row)

        print(all_positive)
        print(epsilon_result)
        total_time = running_time_1

        running_time_list.append([running_time_1])
        if epsilon_result >= 0:
            print(f"In round {num_round}, the solution is feasible.")

            print(f"The total time is {total_time}.")
            print(f"The value of margin is {epsilon_result}.")
            filename = f"run_wall-1-assigned_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            with open(filename, 'wb') as file:
                pickle.dump((epsilon_result, running_time_list), file)
            break
