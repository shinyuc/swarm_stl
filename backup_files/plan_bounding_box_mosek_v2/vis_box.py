from gurobipy import *
import numpy as np
from collections import namedtuple
import time
import pickle
# import pypoman as ppm
# import plot_figures as pf
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import pickle
from scipy.spatial import ConvexHull

def compute_polytope_vertices_ge(A, b):
    """
    Compute the vertices of the polygon defined by Ax + b >= 0.

    Parameters:
        A (numpy.ndarray): Coefficients of the constraints (m x n).
        b (numpy.ndarray): Constants of the constraints (m x 1).

    Returns:
        vertices (list): List of vertices as [x, y].
    """
    # Transform Ax + b >= 0 into -Ax - b <= 0
    A_transformed = A
    b_transformed = -b
    n = A_transformed.shape[1]  # Dimension of the space (2D assumed here)
    vertices = []
    # Loop through combinations of constraints
    for i in range(A_transformed.shape[0]):
        for j in range(i + 1, A_transformed.shape[0]):
            A_sub = np.array([A_transformed[i], A_transformed[j]])
            b_sub = np.array([b_transformed[i], b_transformed[j]])
            try:
                # Solve A_sub x = -b_sub for x
                vertex = np.linalg.solve(A_sub, -b_sub)
                # Check if the vertex satisfies all constraints
                if np.all(A_transformed @ vertex + b_transformed <= 1e-7):  # Tolerance for floating-point errors
                    vertices.append(vertex)
            except np.linalg.LinAlgError:
                # Skip parallel lines or singular matrices
                continue

    return np.array(vertices)


def plot_polygon_ge(A, b, color="b", alpha=0.5):
    vertices = compute_polytope_vertices_ge(A, b)
    if len(vertices) == 0:
        print("No feasible region found.")
        return
    hull = ConvexHull(vertices)
    ordered_vertices = vertices[hull.vertices]
    plt.fill(ordered_vertices[:, 0], ordered_vertices[:, 1], color, alpha)
    plt.axis("equal")
    plt.grid()
    plt.axis('off')

def vis_boxes(test_boxes, swarm, PWLs, limits=None, equal_aspect=True):
    _, plots, Boxes, margin = test_boxes(swarm,PWLs)

    print(Boxes)
    print(f"The value of margin is: {margin[0]}.")
    print(f"The length of box 1 is: {len(Boxes[0])}")

    with open('../plan_waypoints_v1/PWLs.pkl', 'wb') as file:
        pickle.dump(PWLs, file)

    plt.rcParams["figure.figsize"] = [6.4, 6.4]
    plt.rcParams['axes.titlesize'] = 20
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis('off')

    vertices = []
    for plot in plots:
        for A, b in plot[0]:
            # vs = ppm.duality.compute_polytope_vertices(A, b)
            # vertices.append(vs)
            # ppm.polygon.plot_polygon(vs, color = plot[1], alpha=1.)
            plot_polygon_ge(A, b, color = plot[1])


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
    # plt.show()
