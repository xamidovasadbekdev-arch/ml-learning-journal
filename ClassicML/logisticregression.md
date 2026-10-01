# What is Logistic Regression?

Logistic Regression answers yes/no questions by predicting the probability that something belongs to class , then snapping that probability into 0 and 1.

| **Real question** | **Class 1 / Class 0** |
| --- | --- |
| Is this email spam? | 1 = spam, 0 = not spam |
| Does this patient have the disease? | 1 = disease, 0 = healthy |
| Will this customer churn? | 1 = leaves, 0 = stays |

Why Linear Regression fails at predicting classification problem?

!image.png

Logistic Regression borrows Linear Regression’s math and uses SIGMOID FUNCTION that squashes continuous number into (0, 1) interval.

## Sigmoid Function

Sigmoid formula:

```jsx
σ(z) = 1 / (1 + e⁻ᶻ)

where:
z = X @ W + bias
```

It's an S-shaped curve. Big positive input → near 1; big negative → near 0; exactly 0 → exactly 0.5:

| **z** | **−5** | **−2** | **0** | **2** | **5** |
| --- | --- | --- | --- | --- | --- |
| σ(z) | 0.007 | 0.12 | **0.5** | 0.88 | 0.993 |

!image.png

That output is read as a **probability**. `σ(z) = 0.88` means "88% likely to be class 1." This is the source of the "0 and 1" you asked about: it's not linear regression — it's the sigmoid producing a probability between 0 and 1.

The pipeline that goes from Linear Regression into Logistic Regression:

```jsx
1.  z = b + w₁x₁ + ... + wₙxₙ   ---   Linear Regression part
2. p = σ(z) = 1/(1+e⁻ᶻ)    ---   Sigmoid funtion(squash into a probability 0..1)
3) class = 1 if p ≥ 0.5 else 0  ---  last step for prediction
```

## Decision Boundary

**💡 The decision boundary:**

Imagine you are building a model to separate Emails into 2 folders: Spam or Ham.

The invisible line, curve or  wall that your model draws data to separate one class from another is called decision boundary. 

How it works: On one side, model predicts “spam” and on the other side it odes “not spam”.

```jsx
(y = w^T x + b). The exact moment where (w^T x + b = 0) is the Decision Boundary. 
If the result is greater than 0, the mode
class A. If it is less than 0, it chooses class B.
```

Because the cutoff is `p = 0.5`, and `σ(z) = 0.5` exactly when `z = 0`, the model's decision boundary is the line `b + w·x = 0`. So Logistic Regression draws a **straight-line boundary** between the classes — it's a *linear* classifier.

**The decision boundary is a straight line where p = 0.5**

!image.png

```jsx
# Final definition:

The Decision Boundary is simply the line where the model is 50/50 confused. 
Mathematically, it's where (w^T x + b = 0), which forces the Sigmoid function to 
output exactly 0.5.
```

## Log-odds, and what the coefficients mean

If we rearrange the equation, linear part equals to log-odds(also called logit)

```jsx
Log-odds (logit):

ln( p / (1 − p) ) = b + w⋅x
```

ln(p/(1-p)) = sigmoid(z)/sigmoid(-z)

### Mathematical Proof: Deriving Log-Odds from Sigmoid Ratios

As discovered through the symmetry of the Sigmoid function, the odds ratio $\frac{p}{1-p}$ is equivalent to dividing the positive class probability by the negative class probability:

$$
\frac{p}{1 - p} = \frac{\sigma(z)}{\sigma(-z)}
$$

By substituting the explicit algebraic formulas for both Sigmoid functions, we can expand and simplify the expression step-by-step:

$$
\frac{\sigma(z)}{\sigma(-z)} = \frac{\frac{1}{1 + e^{-z}}}{\frac{1}{1 + e^z}} = \frac{1 + e^z}{1 + e^{-z}}
$$

#### Step 1: Multiply the numerator and denominator by $e^z$

To eliminate the negative exponent, we multiply the entire fraction by $\frac{e^z}{e^z}$:

$$
= \frac{e^z(1 + e^z)}{e^z(1 + e^{-z})}
$$

#### Step 2: Distribute $e^z$ in the denominator

We apply exponent laws ($e^z \cdot e^{-z} = e^{z-z} = e^0 = 1$):

$$
= \frac{e^z(1 + e^z)}{e^z + e^{z-z}}
$$

$$
= \frac{e^z(1 + e^z)}{e^z + 1}
$$

#### Step 3: Cancel out common terms

Because $(1 + e^z)$ and $(e^z + 1)$ are identical expressions, they cancel out completely:

$$
= e^z
$$

#### Step 4: Apply the Natural Logarithm ($\ln$)

Now, we substitute this back into our original log-odds formula:

$$
\ln\left(\frac{p}{1-p}\right) = \ln(e^z)
$$

Because the natural log ($\ln$) and the exponential base ($e$) are functional inverses, they cancel each other out, leaving only the raw linear engine:

$$
z = \ln(e^z) = b + w \cdot x
$$

**Q.E.D.** (Which stands for *Quod Erat Demonstrandum* — "Which was to be demonstrated").

## Log loss(cross entropy), and why not MSE

Logistic Regression is trained to minimize **log loss** (a.k.a. binary cross-entropy), not MSE:

```jsx
loss = −[ y·ln(p) + (1 − y)·ln(1 − p) ]
```

Read it intuitively: if the true label is 1, the loss is `−ln(p)` — tiny when `p` is near 1, but **exploding toward infinity** as `p` heads to 0. In other words, it punishes *confident wrong* answers brutally, which is exactly what you want from a probability model.

!image.png

**⚠️ Why not just use MSE?**

Two reasons. (1) With the sigmoid inside, MSE becomes **non-convex** — full of local minima where gradient descent can get stuck. Log loss stays convex. (2) MSE barely penalizes confident wrong predictions. Log loss is the right tool for probabilities.

## Learning it on sklearn

There is **no closed-form / Normal Equation** for logistic regression — so it's trained by **gradient descent** (technically maximizing likelihood = minimizing log loss). This is exactly why gradient descent is the *only* road here.

```jsx
from sklearn.linear_model import LogisticRegression

# max_iter helps it converge; scale features first for best results
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred  = model.predict(X_test)            # final 0/1 labels
y_proba = model.predict_proba(X_test)[:, 1]  # probability of class 1
print(model.coef_, model.intercept_)        # the w's and b (log-odds)
```

**💡 predict vs predict_proba + threshold tuning**

`predict` uses the default 0.5 cutoff. But you can move the threshold: for disease detection you might flag class 1 at `p ≥ 0.3` to catch more cases (higher recall). That's why `predict_proba` is so useful — it gives you the probability to threshold yourself.

## Softmax Funtion

```jsx
Sigmoid handles 2 classes. For 3+ classes, there are 2 strategies, and sklearn picks one
automatically. 
```

- **Softmax (multinomial)** — generalizes the sigmoid to output a probability for each class, all summing to 1. One model, the modern default.
- **One-vs-Rest (OvR)** — train one binary classifier per class ("is it class A or not?") and pick the highest. Simple, older approach.

## **When to use it + gotchas**

| **✅ Great when** | **❌ Weak when** |
| --- | --- |
| You need a fast, interpretable baseline classifier | The true boundary is very non-linear |
| You need calibrated probabilities, not just labels | Complex feature interactions matter (trees do better) |
| Classes are roughly linearly separable | Many irrelevant / highly correlated features |
- **Complexity:** training is fast; prediction is just a dot product + sigmoid (very fast).
- **Hyperparameters that matter:** `C` (inverse regularization strength — smaller C = more regularization), `penalty` (l2/l1), `solver`, `class_weight` (for imbalance), `max_iter`.
- **Common errors (from your notebooks):** forgetting to **scale features** (it's gradient/regularization based) → slow convergence or a "failed to converge" warning; fix with scaling or higher `max_iter`. On imbalanced data, don't trust accuracy — use `class_weight="balanced"` and Precision/Recall.

Softmax definition:

```jsx
The Softmax Function is a mathematical function that takes a vector of raw, 
unscaled numerical scores (called logits) and transforms them into a probability 
distribution.

The output is a set of percentages where:
1. Every individual probability is strictly bounded between 0 and 1.
2. All the probabilities added together equal exactly 1.0 (100%).
```

In machine learning, models speak in raw numbers, but humans speak in probabilities. When building a classifier that can choose between **three or more categories** (like identifying if an image contains a 🐱 **Cat**, 🐶 **Dog**, or 🦜 **Parrot**), we need a mathematical translator.

The mathematical formula:

$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$

```jsx
z is the vector of raw scores(logits) generated by our model's liear layer(w.T@X+b)
z(i) - • is the specific raw score for the class you are currently calculating.
e - Euler's number (roughly equals to 2.71)
denominator sum - means sum up the results for every single class from 1 to K 
```

Imagine NN looks at a photo and outputs these raw continuous scores(logits) for 3 potential names.

Bob(z1): 2.0

Alice(z2): 1.0

Charlie(z3): 0.0

Let’s apply Softmax formula step-by-step:

```jsx
1. Calculate the Numerators(e^z):
Bob: e^2.0 = 7.389
Alice: e^1.0 = 2.718
Charlie: e^0.0 = 1.000 

2. Calculate Denominator(Sum):
a) total_pie = 7.389 + 2.718 + 1.000 = 11.107
b) 
Bob: 7.389/11.107 = 0.665  (66.5%)
Alice: 2.718/11.107 = 0.245  (24.5%)
Charlie: 1.000/11.107 = 0.090  (9.0%)

Total_verification = 66.5% + 24.5% + 9.0% = 100%
```

Why does it called “Soft” Max?

 A hard **"Max"** function is unyielding. It looks at the scores `[2.0, 1.0, 0.0]`, sees that Bob has the highest number, and outputs: **Bob: 100%, Alice: 0%, Charlie: 0%**.

A **"Softmax"** function is nuanced and realistic. It acknowledges that Bob is the clear frontrunner, but it leaves room for uncertainty by keeping Alice and Charlie on the board with smaller, realistic probabilities.

## Interview Questions

| **Question** | **Strong answer** |
| --- | --- |
| Is logistic regression classification or regression? | Classification — the "regression" in the name is historical. |
| How does it differ from linear regression? | Same linear core `b+w·x`, but passed through a sigmoid to produce a probability, then thresholded to a class. |
| What does the sigmoid do? | Squashes any real number into (0,1) so it can be read as a probability. |
| What's the cost function? Why not MSE? | Log loss (cross-entropy). MSE+sigmoid is non-convex and under-penalizes confident errors. |
| Does it have a closed-form solution? | No — it's trained with gradient descent (maximum likelihood). |
| What do the coefficients mean? | Each unit of a feature changes the log-odds by w, i.e. multiplies the odds by e^w. |
| What is the decision boundary? | The line where z = b+w·x = 0 (p = 0.5) — it's a linear boundary. |
| How do you handle 3+ classes? | Softmax (multinomial) or One-vs-Rest. |
| How do you handle class imbalance? | `class_weight="balanced"`, move the threshold, and judge with Precision/Recall not accuracy. |
| Why might it fail to converge? | Unscaled or highly correlated features — scale the data or raise `max_iter`. |