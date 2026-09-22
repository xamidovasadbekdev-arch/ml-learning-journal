## Vectors and Matrices 
### Vector

All data in ML is just numbers arranged in rows and columns. Two words describe them:

A vector is a list of numbers - one row or one sample

Example: 

```jsx
A single house:

[100, 4, 100_000]
kv, rooms, price of the house
```

### Matrix

Matrix is a grid of numbers - the whole dataset.

We call it X: raws/samples, columns/features

```jsx
X has shape (m, n)  →  m = number of samples,  n = number of features
y has shape (m,)    →  one label per sample

```

That's the whole vocabulary of data: a **vector** is one row, a **matrix** is the whole table.

#######################################################################

## Dot Product
**Dot Product — how a prediction is actually made**

The **dot product** is the single most important operation in all of ML. *Every* prediction — Linear Regression, Logistic Regression, even one neuron inside a neural network — is a dot product at its core. If you understand this one thing, `model.predict()` stops being a black box.

The rule is tiny: **multiply matching elements, then add them all up**. Two vectors go in, one number comes out.

```jsx
The rule:

a = [1, 2, 3]
b = [4, 5, 6]

a · b = (1×4) + (2×5) + (3×6) = 4 + 10 + 18 = 32
        └── multiply pairs ──┘   └── then sum ──┘
```

In ML the two vectors are your **features** and the model's **weights**, plus a bias added at the end:

```jsx
Dot product (a prediction):

prediction = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```

This is literally Linear Regression:

```jsx
That formula — features times weights, summed, plus a bias b — is a Linear Regression 
prediction. Most of "learning" is just finding good values for the w's.
```

### **A worked example: predicting a house price**

The formula above is abstract, so let's make it real. Say we describe a house with three features — **size, rooms, age** — and the model has already learned three weights and a bias:

```jsx
One house, one prediction:

features = [ 80 m² ,  3 rooms ,  10 yrs ]
weights  = [ $500  ,  $10,000 , −$2,000 ]      bias = $20,000

price = (80 × 500) + (3 × 10,000) + (10 × −2,000) + 20,000
      =   40,000   +    30,000    +   (−20,000)   + 20,000
      = $70,000
      
 
 
 That single number — np.dot(features, weights) + bias — is what model.predict() 
 returns. No magic: multiply each feature by its weight, add them up, add the bias. 
 The whole prediction is one dot product.
```

**One row was one house — what about a thousand houses?**

A dot product gives **one** prediction for **one** row. For a whole dataset you don't loop — you **matrix-multiply**. Stack all houses in `X` and multiply by the weight column `W`; every row gets dotted with `W` at once

```jsx
All predictions at once:

X (4×3)        @  W (3×1)   =  predictions (4×1)
[[ 80, 3, 10],     [[  500],      [[ 70000],
 [120, 4,  5],  @   [10000],   =   [110000],
 [ 60, 2, 20],      [−2000]]       [ 30000],
 [200, 5,  1]]                     [168000]]

Shape rule:  (m, n) @ (n, p) = (m, p)   ← the inner n's must match

```

#######################################################################

## Notation Cheat Sheet

| Symbol | Name | Plain meaning |
| --- | --- | --- |
| μ (mu) | mean | the average of a list of numbers |
| σ (sigma) | standard deviation | how spread out the numbers are |
| Σ (capital sigma) | sum | "add all of these up" |
| θ (theta) / w | parameters / weights | the numbers the model learns |
| α (alpha) | learning rate | how big a step we take when learning |
| ŷ (y-hat) | prediction | the model's guess (vs the true y) |

Greek letters scare students more than the ideas do. Here's the whole vocabulary you'll meet:

```jsx
When a formula appears in a lecture, read the Greek letters out loud as their plain 
meaning: "Σ(y − ŷ)²" becomes "add up all the squared mistakes." Suddenly it's not scary.
```

#######################################################################

## Cost/Loss Funtions

To improve our model, first we need to know how wrong it currently is. The wrongness is called cost(loss) funtion. Training makes cost function as small as possible.

For regression, the classic cost funtion is MSE: the residual between prediction and reality, square it(to hurt big errors more) and average it.

```jsx
Mean Squared Error(MSE)

MSE = (1/m) Σ (yᵢ − ŷᵢ)²
```

- **Regression losses:** MSE (punishes big errors hard), MAE (mean absolute error, robust to outliers).
- **Classification loss:** log loss / cross-entropy (punishes confident wrong answers).

```jsx
## Analogy
The loss is like a golf score: lower is better, and the whole game is about getting 
it down. The model keeps adjusting its swing (its weights) to shave points off.
```

#######################################################################

## Gradient Descent

So after we calculare an error, we need to minimize it as much as possible untill we reach to minimum. **GRADIENT DESCENT** is the algorithm that does it - and it powers Linear/Logistic Regressio, nueral nets and most of deep learning.

```jsx
🧩 Analogy (Géron's)
Imagine you're lost on a mountain in thick fog and want to reach the valley. 
You can't see far, but you can feel the slope under your feet. So you take a step in 
the steepest downhill direction, then feel again, then step again. Keep going and you 
reach the bottom. That bottom is the minimum cost — and "feeling the slope" is the 
gradient.

### The procedure

1. Starting with random weights
2. Compute the cost 
3. Compute the gradient 
4. Take a small step that way. Step size is learning rate(alpha)
5. Repeat the process

The update step:

new_weight = old_weight − α × gradient
```

!{03621F82-FE88-40DD-8497-1690AECC81F5}.png

Learning rate is everything:

Too small - tiny steps, training takes forever to converge.

Too big - big steps, jumps over the valley, may diverge(cost explodes)

Moderate - steady, decrease to the minimum 💯.

!{36BF3367-714D-4436-8D4A-AE0F733E1169}.png

```jsx
### Debugging tip (Brownlee)

Plot the cost after each iteration. A healthy run shows the cost going down every step.
If it bounces around or grows, your learning rate is too high — lower it. Also: 
scaling your features (Chapter 2) makes the cost "bowl" rounder, so gradient descent 
reaches the bottom much faster.

### Convex = one valley
For Linear Regression, the MSE cost is convex — shaped like a single smooth bowl with 
exactly one bottom. So gradient descent is guaranteed to find the best answer 
(the global minimum). Other models can have multiple valleys (local minima), where it 
might get stuck in a not-quite-best spot.
```

## Gradient Descent types

Gradient descent is divident into 3 groups:

1. Batch GD - the whole training set - stable but slow on big data
2. Stochastic GD - One sample at a time - fast&scalable, but noisy
3. Mini-batch GD - a small batch(32, 64, …) - the practical middle ground

```jsx
ℹ️ In scikit-learn
You rarely write gradient descent yourself — .fit() runs it for you. But you'll see it
directly in SGDClassifier / SGDRegressor, and the learning_rate hyperparameter appears 
all over gradient-boosting and neural networks.
```

#######################################################################


## Interview Questions

The questions interviewers actually ask about Linear Regression, with short, correct answers:

| Question | Strong Answer |
| --- | --- |
| **Difference between Linear Regression and Gradient Descent?** | One is a **model** (predicts a number as a linear combination of features); the other is an **optimization algorithm** that finds the model's parameters. Different kinds of things — and LR can be trained without GD (Normal Equation). |
| **Is Linear Regression classification or regression?** | **Regression** — it predicts continuous numbers, not categories. |
| **Does it output 0 and 1?** | **No.** It outputs any real number. 0/1 are class labels in classification; the 0–1 probability comes from Logistic Regression's sigmoid. |
| **What's its cost function?** | **Mean Squared Error (MSE)** via Ordinary Least Squares — the average squared residual. |
| **Why square the errors?** | To **remove the sign**, **punish big errors more**, and make the cost function smooth and convex for optimization. |
| **How can it be trained?** | Two ways: the **Normal Equation** (closed-form, exact) or **Gradient Descent** (iterative, scales to big data). |
| **What do the coefficients mean?** | Each weight is the **change in the prediction per one-unit increase of that feature** (holding all other features fixed). |
| **What are its assumptions?** | **Linear relationship**, **little multicollinearity**, **homoscedasticity** (constant error variance), **no big outliers**, and **comparable feature scales**. |
| **How do you fight overfitting in LR?** | **Regularization** — Ridge ($L_2$) and Lasso ($L_1$), which shrink the weights to prevent complexity. |
| **$R^2$ of 0.0 means?** | The model is **no better than always predicting the mean of $y$**. (Negative $R^2$ = worse than the mean.) |
