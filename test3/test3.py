import numpy as np

A = np.array([
    [7.99,7.97,7.90,7.70,7.23],
    [7.49,7.80,8.00,8.02,7.69],
    [6.99,7.64,8.11,8.32,8.14],
    [6.49,7.47,8.22,8.62,8.59],
    [5.99,7.31,8.32,8.93,9.04]
])

for a in range(0,5,1):
    print(np.array([A[a]]).reshape(5,2))