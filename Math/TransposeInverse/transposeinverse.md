## Transpose

**What it is**: flip a matrix — rows become columns, columns become rows.

```python
A = [[1, 2, 3],
     [4, 5, 6]]      # shape (2, 3)

A.T = [[1, 4],
       [2, 5],
       [3, 6]]        # shape (3, 2)
```

```python
import numpy as np
A = np.array([[1, 2, 3], [4, 5, 6]])
A.T
```

Simple rule: `(m, n)` becomes `(n, m)`.

### Why transpose matters in ML — mostly about fixing shapes

The #1 practical use: **making shapes compatible for matrix multiplication.**

Remember the rule — `A(m,n) @ B(n,p)` requires inner dimensions to match. Very often in ML code, you'll have two matrices/vectors that *don't* line up correctly until you transpose one of them

```jsx
X = np.array([[1500, 3, 10],
              [2000, 4, 5]])     # shape (2, 3) — 2 houses, 3 features

w = np.array([100, 5000, -200])  # shape (3,) — a plain 1D vector

X @ w   # works fine → shape (2,)
```

But if `w` were accidentally shaped `(1, 3)` instead of `(3,)`:

```python
w = np.array([[100, 5000, -200]])   # shape (1, 3)
X @ w        # ❌ Error: (2,3) @ (1,3) doesn't match
X @ w.T      # ✅ Works: (2,3) @ (3,1) → (2,1)
```

This exact kind of shape-mismatch-fixed-by-transpose happens **constantly** in real ML code. It's honestly one of the most common one-line bug fixes you'll write.

#### Other places transpose shows up

- **Covariance matrices / PCA**: computed as `X.T @ X`
- **Neural network backprop**: gradients often require transposing weight matrices to flow errors backward correctly
- **Word embeddings/attention (transformers)**: `Q @ K.T` is literally the core of the attention mechanism

### The core issue: orientation changes what operation you're doing

With **plain 1D vectors**, transpose does nothing (a 1D array has no rows/columns to flip):

```python
a = np.array([1, 2, 3])
a.T   # → [1, 2, 3]   (unchanged, transpose is a no-op on 1D)
```

But the moment you're working with **2D matrices** (like your houses data), transpose flips rows ↔ columns — and that changes *what shape meets what shape*, which changes what calculation actually happens.

### Concrete house example

```python
houses = np.array([[1500, 3, 10],
                    [2000, 4, 5]])   # shape (2, 3) — 2 houses, 3 features
```

**Without transpose:**

```python
houses @ houses.T
```

This computes: **dot product of each house's features with every other house's features.**

- Result[0][0] = house1 · house1 (dot product with itself)
- Result[0][1] = house1 · house2
- Result[1][0] = house2 · house1
- Result[1][1] = house2 · house2

Result shape: `(2, 2)` — a **house-to-house similarity matrix.**

**With transpose flipped the other way:**

```python
houses.T @ houses
```

This computes: **dot product of each feature column with every other feature column.**

- Result[0][0] = size · size (across all houses)
- Result[0][1] = size · bedrooms
- Result[1][0] = bedrooms · size

Result shape: `(3, 3)` — a **feature-to-feature relationship matrix** (this is literally how covariance matrices / PCA work).

### Why this happens — the actual reason

Matrix multiplication is defined by **rows of the left matrix dotted with columns of the right matrix**. Transposing swaps which one is "rows" and which is "columns" — so you're no longer combining the same pairs of numbers together.

```
houses @ houses.T   → combines ROW-to-ROW (house vs house)
houses.T @ houses   → combines COLUMN-to-COLUMN (feature vs feature)
```

These are genuinely two **different questions** being asked of the same data:

- "How similar are these houses to each other?" vs
- "How correlated are these features with each other?"

### The practical takeaway

Transpose isn't just a formatting fix — **it changes which vectors get dot-producted with which.** That's why swapping `A @ B` to `A.T @ B` (or `A @ B.T`) doesn't give you the same numbers "flipped" — it gives you a completely different calculation with a different meaning, often even a different shape.

**Rule of thumb**: before transposing to fix a shape error, ask yourself *what pairing of rows/columns you actually want* — don't just transpose until Python stops complaining. Getting the right transpose is about intent (what comparison you want), not just about making shapes match.

## Inverse

**What it is**: for a square matrix `A`, its inverse `A⁻¹` is the matrix such that:

```
A @ A⁻¹ = I   (identity matrix — like "1" but for matrices)
```

It's the matrix equivalent of division. Just like `5 × (1/5) = 1`, `A @ A⁻¹ = I`.

```python
A = np.array([[2, 0],
              [0, 4]])

A_inv = np.linalg.inv(A)
# → [[0.5, 0  ],
#    [0,   0.25]]

A @ A_inv
# → [[1, 0],
#    [0, 1]]   ← identity matrix
```

#### Why inverse matters in ML — mostly about solving equations directly

The classic use: **the closed-form solution to linear regression** (called the Normal Equation) uses the inverse directly:

```jsx
w = (Xᵀ X)⁻¹ Xᵀ y
```

This solves for the *exact best* weights in one shot — **no gradient descent, no training loop, no iterations.** Just one formula. In code:

```jsx
w = np.linalg.inv(X.T @ X) @ X.T @ y
```

This is genuinely how simple linear regression can be solved directly, when the dataset isn't too huge — no random init, no "guess and adjust" loop like we did with gradient descent earlier. Both methods (gradient descent vs. this normal equation) arrive at basically the same answer.
****

#### Why inverse is used *less* than transpose in practice

- Only works on **square matrices**, and only if the matrix is "invertible" (not all matrices have an inverse — this is called being "singular")
- Computationally expensive for large matrices — this is actually *why* gradient descent is preferred for big real-world datasets, even though the inverse method is more "exact." Neural networks especially never use direct inverses — way too slow/impossible at that scale.