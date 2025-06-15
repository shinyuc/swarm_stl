import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.linalg import sqrtm
from itertools import combinations

from scipy.linalg import sqrtm

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
    plt.axis("equal")
    plt.grid()
    plt.axis('off')


def plot_ellipse_2d(x0, A):
    """
    Plots the 2D ellipse defined by
        (x - x0) (A A^T)^{-1} (x - x0)^T <= 1
    Parameters
    ----------
    x0 : np.ndarray
        Center of the ellipse, shape (2,).
    A : np.ndarray
        A 2x2 matrix.
    """
    # Compute B = A @ A.T
    B = A @ A.T
    # Factor B using Cholesky, giving L such that L L^T = B
    L = np.linalg.cholesky(B)
    # Create 200 points on the unit circle
    theta = np.linspace(0, 2 * np.pi, 200)
    circle = np.vstack([np.cos(theta), np.sin(theta)])  # shape (2, 200)
    # Map the circle to the ellipse
    ellipse = (L @ circle).T + x0  # shape (200, 2)
    # Plot the ellipse
    plt.plot(ellipse[:, 0], ellipse[:, 1], 'b-')

    # Make axes have equal scale
    plt.axis('equal')

def plot_ellipsoid_inv(P, beta, x0):
    """
    Plot an ellipse defined by (x - x0)^T P^{-1} (x - x0) <= beta.

    Parameters:
        P (2x2 ndarray): Positive definite matrix defining the ellipse.
        beta (float): The parameter controlling the size of the ellipse.
        x0 (1D array-like): Center of the ellipse.
    """
    # Validate inputs
    if not (P.shape == (2, 2)):
        raise ValueError("P must be a 2x2 matrix.")
    if beta <= 0:
        raise ValueError("Beta must be positive.")

    # Compute the transformation matrix
    P_inv_sqrt = sqrtm(np.linalg.inv(P))  # P^{-1/2}

    # Generate points on a unit circle
    theta = np.linspace(0, 2 * np.pi, 500)
    circle = np.array([np.cos(theta), np.sin(theta)])

    # Transform the circle into the ellipse
    scaling_factor = np.sqrt(beta)
    ellipse = scaling_factor * (P_inv_sqrt @ circle)

    # Shift the ellipse to the center x0
    ellipse[0, :] += x0[0]
    ellipse[1, :] += x0[1]

    # Plot the ellipse
    plt.plot(ellipse[0, :], ellipse[1, :], label='Ellipse')
    # plt.scatter(*x0, color='r', s=100, label='Center')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('Ellipse Plot')
    # plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')  # Ensure the plot is not distorted
    # plt.show()

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

def plot_pwls_ellipsoid(PWLs, swarm):
    B_p = swarm.B
    # temp = B_p @ B_p.T *1.1
    P = B_p @ B_p.T

    P_inv = np.linalg.inv(P)
    for PWL in PWLs:
        for SEG in PWL:
            x, t = SEG
            x1 = np.array(x[0])
            x2 =  np.array(x[1])

            center = x2
            # Create unit circle points
            theta = np.linspace(0, 2 * np.pi, 200)
            circle = np.stack([np.cos(theta), np.sin(theta)])  # shape (2, N)

            # Transformation: A such that A @ circle gives boundary of ellipsoid

            A = sqrtm(P)  # P = A^T A -> A = sqrtm(P)
            ellipsoid = A @ circle + center[:, np.newaxis]

            # Plot
            plt.figure(figsize=(6, 6))
            plt.plot(ellipsoid[0], ellipsoid[1], label='Ellipsoid Boundary')
            plt.scatter(*x1, color='blue', label='x1')
            plt.scatter(*x2, color='red', label='x2 (Center)')
            plt.axis('equal')
            plt.grid(True)
            plt.legend()
            plt.title('Ellipsoid: $(x_1 - x_2)^T P^{-1} (x_1 - x_2) \leq 1$')
            plt.xlabel('x')
            plt.ylabel('y')

def plot_invariance_sets(PWLs):
    for PWL in PWLs:
        for i in range(len(PWLs[0]) - 1):
            x1, t1 = PWL[i]
            x1 = np.array(x1)
            x2, t2 = PWL[i + 1]
            x2 = np.array(x2)
            radius = np.linalg.norm(x2 - x1)
            theta = np.linspace(0, 2 * np.pi, 100)
            circle_x = x2[0] + radius * np.cos(theta)
            circle_y = x2[1] + radius * np.sin(theta)
            # Plotting

            plt.plot(circle_x, circle_y, label="Circle")