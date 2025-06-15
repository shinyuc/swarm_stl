import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


def compute_polytope_vertices(A, b):
    """
    Compute the vertices of the polygon defined by Ax + b <= 0.

    Parameters:
        A (numpy.ndarray): Coefficients of the constraints (m x n).
        b (numpy.ndarray): Constants of the constraints (m x 1).

    Returns:
        vertices (list): List of vertices as [x, y].
    """
    n = A.shape[1]  # Dimension of the space (2D assumed here)
    vertices = []
    # Loop through combinations of constraints
    for i in range(A.shape[0]):
        for j in range(i + 1, A.shape[0]):
            A_sub = np.array([A[i], A[j]])
            b_sub = np.array([b[i], b[j]])
            try:
                # Solve A_sub x = -b_sub for x
                vertex = np.linalg.solve(A_sub, -b_sub)
                # Check if the vertex satisfies all constraints
                if np.all(A @ vertex + b <= 1e-7):  # Tolerance for floating-point errors
                    vertices.append(vertex)
            except np.linalg.LinAlgError:
                # Skip parallel lines or singular matrices
                continue

    return np.array(vertices)


def plot_polygon(A, b, color="blue", alpha=0.5):
    vertices = compute_polytope_vertices(A, b)
    if len(vertices) == 0:
        print("No feasible region found.")
        return
    # Compute convex hull for ordered vertices
    hull = ConvexHull(vertices)
    ordered_vertices = vertices[hull.vertices]
    plt.fill(ordered_vertices[:, 0], ordered_vertices[:, 1], color='b', alpha=0.8)
    # plt.plot(ordered_vertices[:, 0], ordered_vertices[:, 1], "r", linewidth=5)
    # plt.scatter(vertices[:, 0], vertices[:, 1], color="red", zorder=5)  # Show vertices
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Polygon defined by Ax + b <= 0")
    plt.axis("equal")
    plt.grid()


# Example
A = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
b1 = np.array([-1, 0.7, -0.25, -0.5])
b2 = -np.array([0, 0.9, 1, -0.5])
b3 = -np.array([-0.2, 0.7, -0.8, 1.2])
b4 = -np.array([0.4, 0.4, 0.4, 0.4])
plt.figure(figsize=(6, 6))
plot_polygon(A, b1)
plot_polygon(A, b2)
plot_polygon(A, b3)
plot_polygon(A, b4)
plt.show()
