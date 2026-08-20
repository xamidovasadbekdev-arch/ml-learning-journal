# Vector and Matrices


### Vector

A vector is an ordered list of numbers. You can think of it as:

- A point in space, or
- An arrow with direction and magnitude

```jsx
# Example 3 elements in an array

v = [2, 5, 1]

v.shape
# (3,)
```

In ML, a vector often represents a single data point — e.g., a vector of pixel values, or a set of features like `[age, income, height]`.
****

### Matrix

A matrix is a 2D grid of numbers, arranged in rows and columns. It's essentially a collection of vectors stacked together.

```jsx
# Example (2 rows × 3 columns, called a "2×3 matrix"):

A = [ 80  2  3 ]  -  1st home(80 kv^2, 2 rooms, 3 years old)
    [ 90  3  6 ]  -  1st home(90 kv^2, 3 rooms, 6 years old)

A.shape
# (2, 3)
```

In ML, a matrix often represents a whole dataset (rows = samples, columns = features), or the weights in a neural network layer.

### How they relate

- A vector is technically a special case of a matrix (a matrix with just 1 row or 1 column).
- You can multiply a matrix by a vector to transform it — this is the core operation behind things like linear regression, neural network layers, and rotations/scaling in graphics.

### Scalar

A **scalar** is just a single number — no direction, no rows/columns, just one value.

```jsx
# Example

5, 3, 10000, 90
```

#### Conclusion

!image.png