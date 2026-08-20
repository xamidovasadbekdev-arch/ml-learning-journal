# explanation

import numpy as np
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(A @ B)  # Using the @ operator for matrix multiplication
print(np.dot(A, B))  # Using np.dot for matrix multiplication

print("Are the results equal? ", np.array_equal(A @ B, np.dot(A, B)))
print(B @ A)  # Using the @ operator for matrix multiplication
print("Are the results equal? ", np.array_equal(A @ B, B @ A))
