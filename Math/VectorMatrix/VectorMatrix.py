# 1
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
print("Array shape: ", arr.shape)
print("Array dimensions: ", arr.ndim)
print("Array dtype: ", arr.dtype)

# 2 
matrix = np.array([[1, 2, 3], [4, 5, 6]])
matrix_zeros = np.zeros((3, 4))
m = np.random.randint(1, 10, size=(3, 3))
print(matrix)
print(matrix_zeros)
print(m)

# 3 
def vector_to_matrix(vector):
    return vector.reshape(-1, 1)

print(vector_to_matrix(arr))

# 4 
matrix = np.random.randint(1, 26, size=(5, 5))
print("Original Matrix:\n", matrix)
print("The middle row: ", matrix[2, :])
print("Last column: ", matrix[:, -1])

# 5 
def flatten_matrix(matrix):
    return matrix.flatten()

print("Flattened Matrix: ", flatten_matrix(matrix))