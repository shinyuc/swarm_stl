import gurobipy as gp
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
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import numpy as np
from itertools import combinations
from scipy.linalg import sqrtm

def compute_2d_polytope_vertices(A, b, tol=1e-7):
    """
    Approximate replacement for ppm.duality.compute_polytope_vertices in 2D.
    Given Ax <= b, returns the list of vertices.
    """
    num_constraints = A.shape[0]
    vertices = []

    # Iterate over all pairs of constraints (lines)
    for (i, j) in combinations(range(num_constraints), 2):
        Ai, Aj = A[i], A[j]
        bi, bj = b[i], b[j]

        # Build the matrix and RHS
        M = np.array([Ai, Aj])
        d = np.array([bi, bj])

        # Try to solve the system (intersection point)
        try:
            x = np.linalg.solve(M, d)
        except np.linalg.LinAlgError:
            continue  # Skip if lines are parallel

        # Check if the point satisfies all constraints
        if np.all(A @ x <= b + tol):
            vertices.append(x)

    # Remove duplicates (if any)
    vertices = np.array(vertices)
    vertices = np.unique(np.round(vertices, decimals=8), axis=0)

    return vertices






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
    # Compute convex hull for ordered vertices
    hull = ConvexHull(vertices)
    ordered_vertices = vertices[hull.vertices]
    plt.fill(ordered_vertices[:, 0], ordered_vertices[:, 1], color, alpha)
    # plt.plot(ordered_vertices[:, 0], ordered_vertices[:, 1], "k-", linewidth=1)
    # plt.scatter(vertices[:, 0], vertices[:, 1], color="red", zorder=5)  # Show vertices
    # plt.xlabel("x1")
    # plt.ylabel("x2")
    # plt.title("Polygon defined by Ax + b >= 0")
    plt.axis("equal")
    plt.grid()
    plt.axis('off')


# def vis(test, limits=None, equal_aspect=True):
def vis(plots, PWLs, limits=None, equal_aspect=True, data_width=10, data_height = 4):

    # plt.rcParams["figure.figsize"] = [6.4, 6.4]
    # plt.rcParams['axes.titlesize'] = 20
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.axis('off')
    # data_width = 10
    # data_height = 4
    aspect_ratio = data_width / data_height

    base_height = 6.4  # you choose this
    fig_width = base_height * aspect_ratio
    fig = plt.figure(figsize=(fig_width, base_height))
    ax = fig.add_subplot(111)
    ax.axis('off')

    vertices = []
    for plot in plots:
        for A, b in plot[0]:
            # vs = ppm.duality.compute_polytope_vertices(A, b)
            vs = compute_2d_polytope_vertices(A, b, tol=1e-7)
            vertices.append(vs)
            # ppm.polygon.plot_polygon(vs, color = plot[1], alpha=1.)
            plot_polygon_ge(A, b, color = plot[1])

    if limits is not None:
        plt.xlim(limits[0])
        plt.ylim(limits[1])
    else:
        vertices = np.concatenate(vertices, axis=0)
        xmin, ymin = vertices.min(axis=0)
        xmax, ymax = vertices.max(axis=0)
        plt.xlim([xmin - 0.1, xmax + 0.1])
        plt.ylim([ymin - 0.1, ymax + 0.1])

    if equal_aspect:
        plt.gca().set_aspect('equal', adjustable='box')

    if PWLs is None or PWLs[0] is None:
        plt.show()
        return

    if len(PWLs) <= 4:
        colors = ['k', np.array([153,0,71])/255, np.array([6,0,153])/255, np.array([0, 150, 0])/255]
    else:
        cmap = plt.get_cmap('tab10')
        colors = [cmap(i) for i in np.linspace(0, 0.85, len(PWLs))]

    for i in range(len(PWLs)):
        PWL = PWLs[i]
        ax.plot([P[0][0] for P in PWL], [P[0][1] for P in PWL], '-', color = colors[i],linewidth=2)
        ax.plot(PWL[-1][0][0], PWL[-1][0][1], '*', color = colors[i], markersize = 10)
        ax.plot(PWL[0][0][0], PWL[0][0][1], 'o', color = colors[i], markersize = 10)
    plt.tight_layout(pad=0.1)
    plt.show()

def plot_boxes(x0, P):
    # x0 = np.array([1.0, 2.0])
    # P = np.array([[3.0, 1.0], [1.0, 2.0]])
    x0 = np.array(x0)
    # Compute the inverse of P
    P_inv = np.linalg.inv(P)

    # Generate points on a unit circle
    theta = np.linspace(0, 2 * np.pi, 100)
    circle = np.stack((np.cos(theta), np.sin(theta)))  # shape: (2, 100)

    # Transform the circle to the ellipsoid
    # Compute matrix square root of P using Cholesky (or SVD for general case)

    P_sqrt = sqrtm(P)
    ellipsoid = x0[:, np.newaxis] + P_sqrt @ circle

    # Plot
    # plt.figure()
    plt.plot(ellipsoid[0], ellipsoid[1], color='red', label='Ellipsoid Boundary')

    # plt.scatter(*x0, color='red')
    # plt.show()

def vis_boxes(plots, PWLs, Boxes, limits=None, equal_aspect=True):

    # plt.rcParams["figure.figsize"] = [6.4, 6.4]
    # plt.rcParams['axes.titlesize'] = 20
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.axis('off')
    # Example: Change size based on some logic
    data_width = 4
    data_height = 10
    aspect_ratio = data_width / data_height

    base_height = 6.4  # you choose this
    fig_width = base_height * aspect_ratio

    fig = plt.figure(figsize=(fig_width, base_height))
    ax = fig.add_subplot(111)
    ax.axis('off')

    vertices = []
    for plot in plots:
        for A, b in plot[0]:
            # vs = ppm.duality.compute_polytope_vertices(A, b)
            vs = compute_2d_polytope_vertices(A, b, tol=1e-7)
            vertices.append(vs)
            # ppm.polygon.plot_polygon(vs, color = plot[1], alpha=1.)
            plot_polygon_ge(A, b, color = plot[1])

    if limits is not None:
        plt.xlim(limits[0])
        plt.ylim(limits[1])
    else:
        vertices = np.concatenate(vertices, axis=0)
        xmin, ymin = vertices.min(axis=0)
        xmax, ymax = vertices.max(axis=0)
        plt.xlim([xmin - 0.1, xmax + 0.1])
        plt.ylim([ymin - 0.1, ymax + 0.1])

    if equal_aspect:
        plt.gca().set_aspect('equal', adjustable='box')

    if PWLs is None or PWLs[0] is None:
        plt.show()
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

    for i in range(len(PWLs)):
        PWL = PWLs[i]
        boxes = Boxes[i]
        for j in range(len(PWL)-1):
            x0 = PWL[j][0]
            box = boxes[j]
            plot_boxes(x0, box)



    plt.show()






# Example
if __name__ == "__main__":
    A =  np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    b1 = np.array([1, -0.7, 0.25, 0.5])
    b2 = np.array([0, 0.9, 1, -0.5])
    b3 = np.array([-0.2, 0.7, -0.8, 1.2])
    b4 = np.array([0.4, 0.4, 0.4, 0.4])
    plt.figure(figsize=(6, 6))
    plot_polygon_ge(A, b1,'y')
    plot_polygon_ge(A, b2, 'r')
    plot_polygon_ge(A, b3,'g')
    plot_polygon_ge(A, b4, 'b')

    wall_half_width = 0.05
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
        # obs.append((A, wall))
    for i in range(4):
        plot_polygon_ge(A, walls[i], 'k')

    plt.show()
