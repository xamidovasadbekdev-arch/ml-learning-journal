### Normal Equation

**The Normal Equation is a formula that directly calculates the best possible weights for linear regression in one shot — no training loop, no guessing, no iterations.**

```jsx
w = (Xᵀ X)⁻¹ Xᵀ y
```

Where:

- `X` = your feature matrix (rows = samples, columns = features)
- `y` = actual target values (e.g., real house prices)
- `w` = the weights it solves for
- `Xᵀ` = X transposed
- `(...)⁻¹` = inverse

Where this formula comes from: 

```jsx
X @ w = y 
```

"Features times weights equals actual answers, perfectly." But this almost never solves exactly (real data is noisy — no perfect line fits every point), so instead the Normal Equation finds the `w` that makes the **total squared error as small as possible** — the same goal as gradient descent, just solved with pure algebra instead of trial-and-error.

Quick derivation logic, using what you already know:

```
X @ w = y
Xᵀ X @ w = Xᵀ y              # multiply both sides by Xᵀ (needed to make X square/invertible)
(Xᵀ X)⁻¹ Xᵀ X @ w = (Xᵀ X)⁻¹ Xᵀ y     # multiply both sides by the inverse
w = (Xᵀ X)⁻¹ Xᵀ y             # (XᵀX)⁻¹(XᵀX) cancels to I, leaving w alone
```

This is literally the exact same "cancel out to isolate the unknown" move we walked through with `5x=10 → x=10/5` and `X@w=y → w=X⁻¹@y`, just adjusted with the extra `Xᵀ` step because `X` usually isn't square (can't invert it directly unless it is).

### In code

```python
import numpy as np

X = np.array([[1500, 3, 10],
              [2000, 4, 5],
              [1200, 2, 20]])
y = np.array([270000, 310000, 190000])

w = np.linalg.inv(X.T @ X) @ X.T @ y
```

This gives you the optimal weights directly — this is literally what `sklearn.linear_model.LinearRegression()` computes internally when you call `.fit()`.

### Why it matters

1. **It's exact** — not an approximation like gradient descent settling after many iterations. One calculation, optimal answer, guaranteed (for plain linear regression).
2. **It shows gradient descent isn't the only way to "learn"** — training doesn't always mean loop → guess → adjust. Sometimes the answer can be solved directly with linear algebra. This is a good conceptual checkpoint: "training" = "finding optimal parameters," and there's more than one way to do that.
3. **It explains *why* gradient descent exists at all** — Normal Equation only works because linear regression has a clean algebraic solution. The moment your model has any nonlinearity (logistic regression, neural networks), there's no equivalent clean formula — you're forced into gradient descent. So understanding Normal Equation is really understanding **the one special case where you don't need gradient descent**, which makes you appreciate why gradient descent is necessary everywhere else.
4. **Practical limitation worth noting** — inverting `(XᵀX)` gets computationally expensive as the number of features grows large (roughly cubic time complexity), which is *why* even for linear regression, gradient descent is sometimes preferred on very large/high-dimensional datasets, despite being less "exact."

### One-line summary for your notes

**Normal Equation = a direct algebraic formula (`w = (XᵀX)⁻¹Xᵀy`) that solves for the exact optimal linear regression weights in one calculation, instead of the iterative guess-and-adjust process of gradient descent — it works because linear regression's error function has a clean mathematical solution, which is a privilege you lose the moment nonlinearity enters the picture.**

## What happens in **model.fit() - Normal Equation**

The formula for Normal Equation:

```jsx
w = (XᵀX)⁻¹Xᵀy

w = best weights
w[0] = bias/intercept  -- > In sklearn, model.intercept_
w[1:] = the actual feature weights  -- > In sklearn, model.coef_

 
y_preds = w @ bias
```

### The Prediction Equation:

!image.png

This equation calculates the predicted output for a single data point.
• **(y)**: The predicted target value (e.g., the price of a house).
• **(theta _0)** (Theta zero): The **bias term** or intercept. It is the starting value if all features are zero.
• **(theta_1, theta_2,   dots)**: The **weights** or coefficients. They show how much each feature influences the prediction.
• **(x_1, x_2,  dots)**: The **input features** (e.g., square footage, number of bedrooms).