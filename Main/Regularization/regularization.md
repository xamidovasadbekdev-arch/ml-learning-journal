## Regularization

```jsx
It is a technique used in ML and Statistics to prevent models from becoming too complex
and overfitting the data.
```

### The problem it solves: overfitting

When you train a model, you're minimizing a loss on your **training data**. A flexible enough model (many features, deep network, high-degree polynomial) can drive that loss almost to zero by memorizing the training set, including its noise and random quirks.

That model then does badly on new data, because the noise it memorized doesn't repeat.

Picture fitting points that roughly follow a line:

- **Underfit:** a flat line. It misses the trend.
- **Good fit:** a smooth line through the middle of the points.
- **Overfit:** a wiggly curve that passes exactly through every point. Training error is 0, but it's useless on new data.

You can see it in the numbers: **training accuracy is high, validation accuracy is much lower.** That gap is overfitting.

Regularization is any technique that stops the model from fitting the training data too perfectly, so it generalizes better to unseen data.

### The core idea: penalize complexity

Normally you minimize:

```
Loss = error on training data
```

With regularization you minimize:

```
Loss = error on training data  +  λ × (penalty for model complexity)
```

The model now has two competing goals:

1. Fit the data well.
2. Keep the weights small and simple.

It can't just memorize everything, because large, extreme weights cost it points in the penalty term.

**Why small weights mean a simpler model:** overfit models typically have huge weights, like `+5000·x₁ − 4998·x₂`, that cancel each other out to hit specific training points exactly. That's fragile: a tiny change in input causes a huge swing in output. Small weights give smoother functions that react gently to input changes, and smoother functions generalize better.

**λ (lambda)** controls the trade-off. In sklearn it's called `alpha`, or `C` where `C = 1/λ`.

- λ = 0: no regularization, so the model can overfit.
- λ too large: the weights get crushed toward zero, so the model underfits.
- λ just right: the best balance, which you find with cross-validation.

## Regularization types

There are classic 2 types of regularization:

```jsx
1. L2 regularization(Ridge, weight decay)
2. L1 regularization(Lasso)
```

### Ridge Regression(L2)

```jsx
penalty = λ × Σ wᵢ²

Squaring punishes big weights heavily and small ones barely.
It shrinks all weights toward zero, but almost never exactly to zero.
Use it as the default when you think most features are somewhat useful.
It handles correlated features well by spreading weight across them.
In sklearn: Ridge. LogisticRegression uses L2 by default. In neural nets it's called 
weight decay.
```

This forces the learning algorithm to not only fit the data but also keep the model
weights as small as possible.

### Lasso Regression(L1)

```jsx
penalty = λ × Σ |wᵢ|

It pushes weights to exactly zero, so useless features get switched off entirely.
That means it does automatic feature selection and produces a sparse model.
Use it when you suspect many features are irrelevant, or you want an interpretable model.
Its weakness is correlated features: it tends to pick one somewhat arbitrarily and 
zero out the rest.
In sklearn: Lasso, or LogisticRegression(penalty='l1', solver='liblinear').
```

### Elastic Net

Elastic Net combines L1 and L2: you get sparsity plus stability with correlated features. In sklearn: `ElasticNet`.

**Why L1 gives exact zeros and L2 doesn't:** L2's pull toward zero gets weaker as a weight gets smaller, because the gradient of w² is 2w, which shrinks as w shrinks. So the weight approaches zero but never quite arrives. L1's pull is **constant**, because the gradient of |w| is always ±1. It keeps pushing with the same force until the weight hits zero and stays there.