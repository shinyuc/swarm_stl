import numpy as np

test1 = np.array([[1,0], [0,1]])
temp = []
for i in range(2):
    temp.append(test1)
print(temp)
print(temp[1])
print(len(temp))
Sigma = np.array([[2,0], [0,1]])
_, S, _ = np.linalg.svd(Sigma)
max_singular_value = np.max(S)
print(max_singular_value)
bloat_factor = max_singular_value ** 2
print(bloat_factor)
Sigma_list = [2, 0, 0, 1]
Sigma = np.array([[Sigma_list[0], Sigma_list[1]],[Sigma_list[2], Sigma_list[3]]])
a = np.array([1,2])
print(Sigma @ a.T)