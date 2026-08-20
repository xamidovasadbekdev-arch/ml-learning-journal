# Matrix Multiplication
Matrix multiplication = many dot products at once

```jsx
A = [1  2]      B = [5  6]
    [3  4]          [7  8]

# How it works
A @ B works like this:

Result[0][0] = row 0 of A · column 0 of B = (1×5)+(2×7) = 19
Result[0][1] = row 0 of A · column 1 of B = (1×6)+(2×8) = 22
Result[1][0] = row 1 of A · column 0 of B = (3×5)+(4×7) = 43
Result[1][1] = row 1 of A · column 1 of B = (3×6)+(4×8) = 50

# in NumPy
import numpy as np
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
A @ B
# → [[19, 22],
#    [43, 50]]

```

The shape rule

For `A @ B` to work: **A's columns must equal B's rows.**

```
A: (m, n)  @  B: (n, p)  →  result: (m, p)
```

### Where matrix multiplication actually powers ML

**1. A whole dataset through a whole model at once**

Instead of looping through houses one by one:

```jsx
for house in houses:
    price = np.dot(house, w) + b
```

You do it all in one shot:

```python
prices = X @ w + b   # all houses, all at once
```

This isn't just "cleaner code" — it's *dramatically* faster, because matrix multiplication is heavily optimized (and GPUs are basically matrix-multiplication machines).

**2. Neural network layers**

A single layer is:

```
output = activation(X @ W + b)
```

where `X` = batch of inputs, `W` = matrix of weights connecting every input neuron to every output neuron. Stack layers, and each one is just another matrix multiplication:

```
layer1 = activation(X @ W1 + b1)
layer2 = activation(layer1 @ W2 + b2)
layer3 = activation(layer2 @ W3 + b3)
```

"Deep learning" literally means: **chain a bunch of matrix multiplications (with nonlinearities in between) together.**

**3. Why GPUs matter**

CPUs do things mostly sequentially. GPUs have thousands of small cores built to do many multiplications and additions **in parallel** — which is exactly what matrix multiplication needs (each output cell is independent of the others). This is the entire reason deep learning became practical: matrix multiplication at massive scale, done in parallel, is what a GPU is *built* for.

**4. Transformers (the architecture behind Claude, GPT, etc.)**

Attention — the mechanism that lets a model figure out "which words in a sentence relate to which" — is computed almost entirely through matrix multiplications between Query, Key, and Value matrices. No new operation, just the same `A @ B` you just learned, applied cleverly.

### A tiny neural network example, concretely

```jsx
import numpy as np

X = np.array([[1500, 3, 10],   # house 1
              [2000, 4, 5]])   # house 2 — shape (2, 3)

W = np.array([[100],           # weight for size
              [5000],          # weight for bedrooms
              [-200]])         # weight for age — shape (3, 1)

b = 10000

output = X @ W + b
# shape (2,3) @ (3,1) = (2,1) → one prediction per house
print(output)
```

Notice: the inner dimensions (3 and 3) matched, so this worked. If `W` had been shape `(2,1)` instead, you'd get a shape error — this is the #1 debugging headache once you start building actual networks, so get comfortable checking `.shape` constantly.

### Bottom line

Dot product → scaled up to matrix multiplication → scaled up to GPUs doing millions of them in parallel → that's the computational backbone of basically all of modern ML and deep learning. Everything from linear regression to transformers is, mechanically, "multiply matrices, add bias, apply nonlinearity, repeat."

## CODE

```jsx
import numpy as np

X = np.array([
    [1200, 3, 10],
    [1800, 4, 5],
    [2500, 5, 2],
    [1000, 2, 15]
])

W = np.array([[500], [10000], [-2000]])

bias = 20000

# barcha bashoratlar bir vaqtning o'zida
narxlar = X @ W + bias

for i, n in enumerate(narxlar.flatten()):
    print(f"{i+1}-uy: ${n:,.0f}")

print(f"\nShape: {X.shape} @ {W.shape} = {narxlar.shape}")
```