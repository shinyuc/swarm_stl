import pickle
import matplotlib as plt
from vis import vis
import numpy as np


def test(tmax, vmax):
    # agent_counts = [15, 15, 25, 40]  # number of agents in each swarm
    # agent_counts = [8, 4]
    mu_thresholds = [3, 1, 6, 4]
    agent_counts = [5, 2, 5, 8]
    # mu_thresholds = {'goal0': 10, 'goal1': 15, 'goal2': 20, 'goal3': 30}
    # mu_thresholds = {'goal0': 2, 'goal1': 3, 'goal2': 4, 'goal3': 6}
    # mu_thresholds = {'goal0': 2, 'goal1': 3, 'goal2': 6, 'goal3': 4}

    # mu_thresholds = [2, 6]

    x0s = [[2, 0.5], [4, 0.5], [6, 0.5], [8, 0.5]]
    # x0s = [[2, 0.5],  [8, 0.5]]

    wall_half_width = 0.05
    A = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    walls = []

    walls.append(np.array([0, 0, 0, 4], dtype = np.float64))
    walls.append(np.array([10, 10, 0, 4], dtype = np.float64))
    walls.append(np.array([0, 10, 0, 0], dtype = np.float64))
    walls.append(np.array([0, 10, 4, 4], dtype = np.float64))

    walls.append(np.array([0, 4., 2, 2], dtype = np.float64))
    walls.append(np.array([5., 10, 2, 2], dtype = np.float64))

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

    b1 = np.array([-1.5, 2.5, -3, 4], dtype = np.float64)
    b2 = np.array([-3.5, 4.5, -3, 4], dtype = np.float64)
    b3 = np.array([-5.5, 6.5, -3, 4], dtype = np.float64)
    b4 = np.array([-7.5, 8.5, -3, 4], dtype = np.float64)
    goals = [(A, b1), (A, b2), (A, b3), (A, b4)]
    # goals = [(A, b1), (A, b4)]

    # test_list = [B1, B2, B3, B4, D]
    test_list = [(A, b1), (A, b2), (A, b3), (A, b4)]
    for obs_element in obs:
        test_list.append(obs_element)






    plots = [[goals, 'g'], [obs, 'k']]
    return plots

if __name__ == '__main__':
    with open('../data/model_2-auto/PWLs_output_waypoints.pkl', 'rb') as file:
        PWLs_loaded, Box_loaded, epsilon_loaded = pickle.load(file)


    plots = test(1, 1)

    results = vis(plots, PWLs_loaded)