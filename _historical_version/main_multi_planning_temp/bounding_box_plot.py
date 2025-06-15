import numpy as np
import matplotlib.pyplot as plt


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

    # (Optional) Plot the center
    # plt.scatter(x0[0], x0[1], color='red')

    # Make axes have equal scale
    plt.axis('equal')

if __name__ == '__main__':
    x0 = np.array([1.0, 2.0])
    A = np.array([[2.0, 0.5],
                  [0.5, 1.0]])

    # Call the function
    plot_ellipse_2d(x0, A)

    # Show the figure
    plt.show()