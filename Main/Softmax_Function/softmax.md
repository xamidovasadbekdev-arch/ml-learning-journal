## Decision Boundary

```python
It is a geometric line that seperates space into different regions corresponding to 
predicted classes.

sigmoid function:

sigma = 1/(1+e^(-z))
```

**How It Works in Binary Logistic Regression**
Your model outputs a predicted probability y_pred = sigmoid(z), where:z = w_1 x_1 + w_2 x_2 + b
By default, we set a **classification threshold at 0.5**:
• If y_pred ≥0.5   ——> Predict **Class 1**
• If y ≤ 0.5 ——→ Predict **Class 0**

**Finding the Boundary Equation**
Recall how the Sigmoid function behaves:

```python
sigmoid(z) = 0.5 ——> z=0
```

The exact line where the model switches its decision from $0$ to $1$ occurs precisely when $z = 0$

```python
z = w1x1 + w2x2 + b = 0
```

If we rearrange this linear equation into slope-intercept form ($y = mx + c$), letting $x_2$ be $y$ and $x_1$ be $x$:

```python
x2 = - w1x1/w2 - b/w2
```

This straight line in a 2D feature space is the **Decision Boundary**:
• Points on one side yield z > 0 —→ sigma(z) > 0.5 ——> **Class 1**
• Points on the other side yield z < 0 ———> sigma(z) < 0.5 ——>**Class 0**

## Softmax Function

While **Sigmoid** maps a single continuous score z to a probability between 0 and 1 for binary classification, **Softmax** generalizes this to multiclass classification (K > 2 classes).

**The Problem with standard scores (logits)**
Suppose a model outputs raw linear scores (called **logits**) z = [z1, z2, z3] for three classes (e.g., Cat, Dog, Bird):
z = [2.0, 1.0, -1.0]

These numbers are not probabilities because:

1. They don’t sum to 1
2. Probabilities cannot be negative

Softmax converts a vector of K real numbers into probability distribution.

<img width="273" height="153" alt="image" src="https://github.com/user-attachments/assets/a75ac7c1-9128-4870-8e83-8bf03914f64d" />


Why exponentiate(e(z, k))?

1. Ensures positivity e^z is always strictly positive, even if z is negative.
2. Amplifies difference: Exponentiation makes larger scores stand out significantly more than smaller scores.

**Why divide by ∑(e^z,j)?**

 Dividing by the total sum forces all resulting probabilities to **sum to 1.0 (100%)**.

**Softmax Step-by-step calculation:**

Lets compare softmax for logits ——> z = [2.0, 1.0, 0.1]

<img width="1006" height="295" alt="image" src="https://github.com/user-attachments/assets/3e952552-7965-46df-8838-a905912dd1d4" />

Python implementation in code:

```python
import numpy as np
def softmax(z):
    # Subtract np.max(z) for numerical stability (prevents e^z overflow)
    exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

logits = np.array([2.0, 1.0, 0.1])
probabilities = softmax(logits)

print("Probabilities:", np.round(probabilities, 4))
print("Sum of probabilities:", np.sum(probabilities))
```

The class with the **largest (maximum) logit value** will always have the **highest probability** after Softmax.

### Connection Summary

Decision Boundary: The line (z=0) where the probability output equals to 0.5.

Sigmoid: Used for Binary classification (K=2), outputting probability 1.

Softmax: Used for Multiclass classification(K>2), outputting K probabilities that sum to 1.
