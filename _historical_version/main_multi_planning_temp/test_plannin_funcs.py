import numpy as np
import math

A = np.array([[0.01, 0], [0, 0.01]])
# A = np.array([[math.exp(1), 0], [0, math.exp(1)]])
sign, logdet = np.linalg.slogdet(A)

print("Sign:", sign)
print("Log Determinant:", logdet)