# Dot Product

What is Dot product?

The **dot product** takes two vectors of the same length and combines them into a **single scalar number**.

#### How it works

Multiply corresponding elements, then add them all up.

```jsx
a = [1, 2, 3]
b = [4, 5, 6]

dot product = (1×4) + (2×5) + (3×6)
            = 4 + 10 + 18
            = 32
```

In NumPy:

```jsx
import numpy as np
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

np.dot(a, b)   # → 32
# or equivalently
a @ b          # → 32
```

### Why it matters (this is the big one for ML)

The dot product is *the* core operation inside neural networks. When a layer computes:

```jsx
output = weights · inputs + bias

Let's take house price calculation:
output = (kv^2*w1) + (rooms*w2) + (age*w3) + b

b - bias
w1, w2, w3 - weights
```

that `weights · inputs` part is a dot product — it's literally how a neuron combines all its input signals into one number before applying an activation function.

### Geometric intuition

The dot product also tells you something about the angle between two vectors:

- **Large positive** → vectors point in a similar direction
- **Zero** → vectors are perpendicular (90°)
- **Negative** → vectors point in roughly opposite directions

This is why dot products show up in things like **cosine similarity** (comparing how similar two pieces of text or two embeddings are) — a concept you'll definitely run into if your course touches NLP or embeddings at all.

### One shape rule to remember

Both vectors must be the **same length**, or the dot product isn't defined:

```jsx
a = np.array([1, 2, 3])
b = np.array([4, 5])
np.dot(a, b)   # ❌ Error — shapes don't match
```

Real example in code

```jsx
import numpy as np
# model weights 
W = np.array([500, 10000, -2000])
bias = 20000

# new house 
home = np.array([80, 3, 10])

# prediction = dot product + bias
prediction = np.dot(W, home) + bias
print("House price: ", prediction,"$")

# Answer:
House price:  70000 $
```

Is Dot Product biggest concept in ML?

- **Every layer of a neural network** is essentially `output = activation(W @ x + b)` — a matrix multiplication followed by a nonlinearity. Stack enough of these and you get deep learning.
- **Training a whole batch of data at once** (instead of one example at a time) is done via matrix multiplication — this is *why* GPUs matter, since they're built to do massive matrix multiplications in parallel.
- **Attention mechanisms in transformers** (the architecture behind GPT, Claude, etc.) are fundamentally built from matrix multiplications (query, key, value matrices multiplying each other).
- Linear regression, PCA, embeddings, convolutions — nearly every classical and modern ML technique reduces to matrix operations under the hood.

### Common misunderstanding about WEIGHTS and BIAS

The simple flow

1. **Before training**: weights and bias = random junk numbers (basically meaningless).
2. **Training data**: you already have real houses with known prices — `X_train` (features) and `y_train` (actual prices).
3. **The computer runs a loop**, like hundreds or thousands of times:
    - Guess a price using current (bad) weights/bias
    - Check how wrong the guess was vs the real price
    - **Use the error to update weights AND bias** (this is called **gradient descent**) and slightly adjust weights and bias to be less wrong
    
    ```jsx
    w = w - learning_rate * (gradient of error w.r.t. w)
    b = b - learning_rate * (gradient of error w.r.t. b)
    ```
    
    - Repeat
4. **After training finishes**: weights and bias are no longer random — they're now numbers that fit your training data well. This final `w` and `b` are what get saved/used.
5. **Predicting** on new data (`X_test`) just plugs in these *already-learned* numbers:

```jsx
   predicted_price = X_test @ w + b
   
   # No more guessing at this point — w and b are fixed, learned values.
```

1. Then compare with `Y_test`  to find accuracy of a model.

So what do weights and bias actually *mean*?

- **Weight** = how much each feature matters. E.g., if `w_size = 150`, it means "each extra sqft adds ~$150 to price." The training process figures out these numbers from the patterns in your data.
- **Bias** = the baseline value when all features are 0. E.g., "a hypothetical 0 sqft, 0 bedroom house" would cost `b` dollars. It shifts the whole prediction up or down — like the y-intercept in `y = mx + b` from algebra class. You've seen this exact formula before, actually — linear regression is just the multi-feature version of it.

## So the real flow is:

```jsx
random weights & bias → predict → compare to truth (error) → 
→ adjust weights & bias → repeat

# Training phase (using X_train, y_train):
X_train + y_train → training loop → learns w, b

# Testing phase
y_pred = X_test @ w + b        ← predict using learned w, b
compare y_pred vs y_test        ← how good are the predictions?
```

### Conclusion

```jsx
**# 1
model.fit(X_train, y_train)    # learns w, b internally

# 2, after fit()
predictions = model.predict(X_test)   # uses the already-learned w, b

# We can take them using: 
model.coef_        # the learned weights (w)
model.intercept_    # the learned bias (b)**
```