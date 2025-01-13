import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm


def plot_ellipsoid(P, beta, x0):
    """
    Plot an ellipsoid defined by (x - x0)^T P (x - x0) <= beta.

    Parameters:
        P (2x2 ndarray): Positive definite matrix defining the ellipsoid.
        beta (float): The parameter controlling the size of the ellipsoid.
        x0 (1D array-like): Center of the ellipsoid. Defaults to [0, 0].
    """
    # Validate inputs
    if not (P.shape == (2, 2)):
        raise ValueError("P must be a 2x2 matrix.")
    if beta <= 0:
        raise ValueError("Beta must be positive.")

    # Compute the transformation matrix
    P_inv_sqrt = np.linalg.inv(sqrtm(P))  # P^{-1/2}

    # Generate points on a unit circle
    theta = np.linspace(0, 2 * np.pi, 500)
    circle = np.array([np.cos(theta), np.sin(theta)])

    # Transform the circle into the ellipsoid
    scaling_factor = np.sqrt(beta)
    ellipsoid = scaling_factor * (P_inv_sqrt @ circle)

    # Shift the ellipsoid to the center x0
    ellipsoid[0, :] += x0[0]
    ellipsoid[1, :] += x0[1]

    # Plot the ellipsoid
    # plt.figure(figsize=(6, 6))
    # plt.plot(ellipsoid[0, :], ellipsoid[1, :], label=r'$(x-x_0)^T P (x-x_0) \leq \beta$')
    plt.plot(ellipsoid[0, :], ellipsoid[1, :])
    # plt.scatter(x0[0], x0[1], color='red', label='Center ($x_0$)')
    # plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    # plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.gca().set_aspect('equal', adjustable='box')  # Ensure the plot is not distorted
    # plt.xlabel('$x_1$')
    # plt.ylabel('$x_2$')
    # plt.title('Ellipsoid')
    # plt.legend()



# Example Usage
if __name__ == "__main__":
    P = np.array([[2, 0.5], [0.5, 1]])
    x0=np.array([0, 0])
    beta = 1.7423947706268075
    plot_ellipsoid(P, beta, x0)
