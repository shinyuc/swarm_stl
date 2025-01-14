import numpy as np
import matplotlib.pyplot as plt
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
