import numpy as np

# Example matrix P (symmetric positive definite)
P = np.array([[1, 0],
              [0,1]])

# Cholesky decomposition
def cholesky_decomposition(P):
    try:
        B = np.linalg.cholesky(P)
        return B
    except np.linalg.LinAlgError:
        print("Matrix P is not positive definite.")

B = cholesky_decomposition(P)
print(B)
print(B @ B.T)