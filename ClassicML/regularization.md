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

!image.png

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

There are two ways to measure "how big are the weights", and the choice changes the model's behaviour completely. This **L1 vs L2** distinction is one of the most asked interview questions.

| **L2 (Ridge)** | **L1 (Lasso)** |  |
| --- | --- | --- |
| Penalty | sum of **squared** weights, Σw² | sum of **absolute** weights, Σ|w| |
| Effect on weights | Shrinks all weights smoothly toward 0 | Pushes some weights to **exactly 0** |
| Feature selection? | No — keeps all features (small) | **Yes** — zeroed features are dropped |
| Result | Dense (all features used a little) | Sparse (only the useful features survive) |

**ℹ️ Why does L1 zero things out but L2 doesn't?**

Geometric intuition: L1's constraint region is a **diamond** (pointy corners on the axes), so the best solution often lands exactly on a corner — where some weights are 0. L2's region is a **circle** (no corners), so it shrinks weights but rarely hits exactly 0. That's why **Lasso doubles as automatic feature selection**.

!image.png

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

## **The strength knob (alpha / C) — a classic gotcha**

One hyperparameter controls *how much* regularization you apply. But it's named differently — and points the opposite way — in different classes:

| **Class** | **Knob** | **Direction** |
| --- | --- | --- |
| Ridge, Lasso, ElasticNet | `alpha` | **Higher alpha = MORE regularization** (simpler model) |
| LogisticRegression, SVM | `C` | **Higher C = LESS regularization** (C = 1/alpha) |

**⚠️ Two gotchas**

(1) `alpha` and `C` go in *opposite* directions — mixing them up is a very common interview/practice mistake. (2) **Always scale your features before regularizing.** The penalty is based on weight size, so if features are on different scales the penalty is applied unfairly. Use `StandardScaler` first (ideally in a Pipeline).

In sklearn code:

```jsx
from sklearn.linear_model import Ridge, Lasso, ElasticNet

ridge = Ridge(alpha=1.0).fit(X_train, y_train)        # L2
lasso = Lasso(alpha=0.1).fit(X_train, y_train)        # L1 - some coefs become 0
enet  = ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X_train, y_train)  # mix (r = l1_ratio)

print((lasso.coef_ == 0).sum(), "features dropped by Lasso")

# Logistic Regression is L2-regularized by default; C is the (inverse) knob
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C=0.5, penalty="l2", max_iter=1000)
```

## **Which to use (decision guide)**

- **Start with Ridge** — Géron's advice: "almost always have at least a little regularization," and Ridge is the safe default.
- **Use Lasso** if you believe only a few features matter and want them auto-selected.
- **Use Elastic Net** when you have many features, some correlated — it's generally preferred over Lasso, which can behave erratically when features > samples or features are strongly correlated.
- **Bonus — Early stopping:** for iterative models, just stop training when validation error starts rising. Geoffrey Hinton called it a "beautiful free lunch." It's regularization without touching the cost function.

## **Interview questions**

| **Question** | **Strong answer** |
| --- | --- |
| What is regularization? | Adding a penalty for large weights to the cost, to reduce overfitting by keeping the model simple. |
| L1 vs L2? | L1 (Lasso) penalizes Σ|w| and drives some weights to exactly 0 (sparse, feature selection); L2 (Ridge) penalizes Σw² and shrinks all weights smoothly (dense). |
| Why does L1 produce sparsity? | Its diamond-shaped constraint has corners on the axes, so the optimum often lands where some weights are exactly 0. |
| What is Elastic Net? | A mix of L1 and L2 (ratio `l1_ratio`); good when there are many, possibly correlated features. |
| What does alpha control? And C? | alpha = regularization strength (higher = more). C = 1/alpha in LogisticRegression/SVM (higher C = less regularization). |
| Do you need to scale features first? | Yes — the penalty depends on weight magnitude, so features must be on the same scale. |
| Is the bias term regularized? | No — only the feature weights are penalized, not the intercept. |
| How does Lasso help interpretability? | By zeroing out useless features, it leaves a short list of the ones that actually matter. |