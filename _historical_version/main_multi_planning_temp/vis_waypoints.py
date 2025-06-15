from gurobipy import *
import numpy as np
from collections import namedtuple
import time
import pickle
import plot_figures as pf
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import pickle

def vis_waypoints(test_waypoint, time_stamps, swarm, boxes, limits=None, equal_aspect=True):
    _, plots, PWLs, margin, boxes, total_time = test_waypoint(time_stamps, swarm,boxes)

    print(PWLs)
    print(f"The value of margin is: {margin}.")

    with open('PWLs_output_waypoints.pkl', 'wb') as file:
        pickle.dump((PWLs, margin, boxes), file)

    plt.rcParams["figure.figsize"] = [6.4, 6.4]
    plt.rcParams['axes.titlesize'] = 20
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis('off')

    vertices = []
    for plot in plots:
        for A, b in plot[0]:
            pf.plot_polygon_ge(A, b, color = plot[1])


    # if limits is not None:
    #     plt.xlim(limits[0])
    #     plt.ylim(limits[1])
    # else:
    #     vertices = np.concatenate(vertices, axis=0)
    #     xmin, ymin = vertices.min(axis=0)
    #     xmax, ymax = vertices.max(axis=0)
    #     plt.xlim([xmin - 0.1, xmax + 0.1])
    #     plt.ylim([ymin - 0.1, ymax + 0.1])

    if equal_aspect:
        plt.gca().set_aspect('equal', adjustable='box')

    if PWLs is None or PWLs[0] is None:
        # plt.show()
        return

    if len(PWLs) <= 4:
        colors = ['k', np.array([153,0,71])/255, np.array([6,0,153])/255, np.array([0, 150, 0])/255]
    else:
        cmap = plt.get_cmap('tab10')
        colors = [cmap(i) for i in np.linspace(0, 0.85, len(PWLs))]

    for i in range(len(PWLs)):
        PWL = PWLs[i]
        ax.plot([P[0][0] for P in PWL], [P[0][1] for P in PWL], '-', color = colors[i])
        ax.plot(PWL[-1][0][0], PWL[-1][0][1], '*', color = colors[i])
        ax.plot(PWL[0][0][0], PWL[0][0][1], 'o', color = colors[i])
    plt.show()

    print(f"The total time is {total_time}.")
    return total_time
