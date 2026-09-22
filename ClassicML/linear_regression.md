## What is Linear regression
Linear regression is a regression algorithm that predicts number by fitting the best line through the data. 

```jsx
🧩 Analogy
Imagine plotting house size (x) against price (y) as dots on a graph. 
Linear Regression draws the single straight line that passes as close as possible to 
all the dots. To predict the price of a new house, you just read the line at that size. 
That's the whole idea.

"Best line" has a precise meaning: the line where the total squared distance from the 
dots to the line is as small as possible. Those vertical gaps are the model's errors 
(called residuals), and we square them so big misses count more and positive/negative 
gaps don't cancel out. Minimizing them is called Ordinary Least Squares (OLS).
```

!image.png

What for we use Linear Regression in real life:

Salary calculation: Features - years of experience, education; Prediction - salary($3,000)

Student grade: Features - attendance, exam results, study hours ; Prediction - student grade(89.4)

#######################################################################

## Linear Regression vs Gradient Descent
This is one of the most common interview questions, it is like a trip to fall in. 

The question: “What is the difference between Gradient Descent and Linear regression”: it is like what is the difference between driving and a car.

|  | Linear Regression | Gradient Descent |
| --- | --- | --- |
| **What it is** | A *model* — a hypothesis that y is a straight-line combination of the features | An *optimization algorithm* — a general method to minimize a cost function |
| **It answers** | "WHAT is the shape of the relationship?" | "HOW do we find the best parameters?" |
| **Produces** | A prediction (a number) | The best weights `w` and bias `b` |
| **Category** | A regression model | A training / optimization technique |

```jsx
🧩 Analogy
Linear Regression is the destination (the best-fit line you want to reach). 
Gradient Descent is one route to get there. You can reach the exact same destination 
by a different route — the Normal Equation. The destination and the route are not 
"two options of the same kind."

### Two facts that prove they are fundamentally different things:

You can train Linear Regression without any gradient descent — scikit-learn's 
LinearRegression uses the Normal Equation (a direct one-shot formula). No iterations, 
no learning rate.
Gradient Descent trains many other models — Logistic Regression, neural networks, 
gradient boosting. It is not tied to Linear Regression at all.
```

```jsx
ℹ️ A crisp answer to give in an interview
"Linear Regression is a model: it assumes the target is a linear combination of the 
features and predicts a continuous number. Gradient Descent is an optimization algorithm: 
it finds the model's parameters by iteratively stepping downhill on a cost function. 
One is the 'what', the other is the 'how'. In fact Linear Regression can be trained 
without gradient descent, using the Normal Equation, and gradient descent is used to train 
many models beyond linear regression."
```

#######################################################################

## The equations and coefficients

## **The equation & coefficients**

The model is just an equation of a line.

```jsx
# simple one feature linear regression
ŷ = b + w⋅x

####################################################

# with many features
ŷ = b + w₁x₁ + w₂x₂ + ... + wₙxₙ

w = coefficients/weights - how much ŷ changes per unit of that feature
b = bias/intercept - the prediction when all features are 0
ŷ = prediction - the model's output

####################################################################
ℹ️ The weights are interpretable
This is a big selling point. If w for "rooms" is 50,000, 
the model says "each extra room adds ~$50,000 to the predicted price." 
Few algorithms are this easy to explain to a non-technical audience.

```

#######################################################################

## The cost funtion MSE

## The cost function(MSE) - where the gradient descent came from

"Best line" needs a number to measure. That number is the **cost function**. For each training point we take the residual (prediction − truth), and the cost is the average of the squared residuals — **Mean Squared Error (MSE)**:

```jsx
# MSE cost

J(w, b) = (1/m) Σ (ŷᵢ − yᵢ)²   where  ŷᵢ = w⋅xᵢ + b

Why squared (three real reasons, all worth telling students):

1. Removes the sign — a +10 error and a −10 error are both "10 wrong"; 
without squaring they'd cancel to 0 and hide the error.
2. Punishes big mistakes harder — an error of 10 contributes 100, an error of 2 
contributes 4. The model is pushed hardest to fix its worst predictions.
3. It's smooth — squaring makes the cost differentiable and convex (the single bowl), 
which is exactly what gradient descent needs.

```

!image.png

#######################################################################

## How it learns(OSL/Gradient Descent)

## How it learns(OLS&GradientDescent)

‘Learning’ means finding the best ‘w’ and ‘b’ so that line fit best - minimizes MSE as small as possible. There are 2 ways to get there: 

**Normal Equation**: direct formula in one step - fast for few features - used by LinearRegression

**Gradient Descent**: iteratively walk on downhill - better for many features/huge data - used by SGDRegressor, nueral nets

Normal Equation is the closed-form solution, computes the best weights directly, no iterations:

```jsx
θ = (XᵀX)⁻¹ Xᵀy

Where does it come from? The cost is a bowl; its lowest point is exactly where its slope
(derivative) is zero. If you set the derivative of the MSE to zero and solve for the 
parameters with linear algebra, you get this single formula. It jumps straight to the 
bottom of the bowl — no learning rate, no steps. (Downside: inverting XᵀX gets slow 
when you have very many features, which is when gradient descent wins.)

ℹ️ Good news
Because the MSE cost for linear regression is convex (the single bowl), both methods 
reach the same best answer — there are no local minima to get stuck in. 
The Normal Equation and Gradient Descent literally find the same line.
```

#######################################################################

## From Scratch

```jsx
import numpy as np

X = np.array([1, 2, 3, 4, 5], dtype=float)   # feature
y = np.array([2, 4, 6, 8, 10], dtype=float)  # target (here y = 2x)

w, b = 0.0, 0.0          # start with zero weights
lr = 0.01                # learning rate (alpha)
m = len(X)

for epoch in range(1000):
    y_pred = w * X + b               # 1. predict
    error  = y_pred - y             # how wrong
    # 2. gradients of MSE w.r.t. w and b
    dw = (2/m) * np.sum(error * X)
    db = (2/m) * np.sum(error)
    # 3. step downhill
    w -= lr * dw
    b -= lr * db

print(w, b)   # ~2.0 and ~0.0 -> it learned y = 2x

## Anology
That's it — predict, measure error, nudge the weights downhill, repeat. 
Every gradient-based model in this course is a fancier version of this loop.

```

```jsx
ℹ️ Where does every number come from? (so you can answer students)

w, b = 0.0, 0.0 — the starting point. Gradient descent must start somewhere. 
Because the cost is a convex bowl, any start reaches the same bottom, so 0 is the 
simplest honest choice. (For neural networks the start matters and we use small random values 
instead — but for linear regression, 0 is fine.) This is the "where did the 0 come from" 
answer — it's the initial guess, not a property of the model.

lr = 0.01 — the learning rate (α). Chosen by trial: too big and the cost diverges, 
too small and it crawls. 0.01 is a common safe starting value.

range(1000) — the number of epochs (full passes over the data). We just repeat until 
the cost stops dropping.

dw, db — these are the gradient formulas we derived in "The cost function" above: 
(2/m)Σ(error⋅x) and (2/m)Σ(error). Not magic — they're the slope of MSE.

The minus sign in w -= lr*dw — the gradient points uphill, so to go downhill 
(reduce cost) we step in the opposite direction.
```

### In sklearn, how is the code:

```jsx
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)         # learns w's and b
y_pred = model.predict(X_test)

print(model.coef_)        # the weights (one per feature)
print(model.intercept_)   # the bias b
print(model.score(X_test, y_test))  # R squared

💡 Teaching tip
Have students inspect model.coef_ and connect each number back to a feature. 
Seeing "income → +0.7, age → −0.1" makes the abstract equation feel real, and previews 
feature importance.
```

#######################################################################

## Evaluation Regression model(R^2, RMSE, MAE, MSE)

Accuracy is for classification. For regression we measure *how far off* the predictions are:

```jsx
R^2 - Fraction of variance explained (1 = perfect, 0 = no better than the mean) - .score()
RMSE - 	Typical error, in the same units as y - punishes big errors
MAE - Average absolute error - more robust to outliers

## In Sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)
```

#######################################################################

### Does Linear Regression output 0 and 1? (No here’s the real answer)

This is a confusion worth killing completely, because students ask it and it's easy to answer wrong. **Linear Regression does NOT output 0 and 1.** Its output is any real number: −3.2, 0, 47, 250000 — whatever the line gives. It predicts *quantities* (price, salary, score), so "0 or 1" simply isn't what it does.

So what is it for? It is classification problem - Logistic Regression solves using it.

Here is the real chain:

1. **The labels.** In binary classification we encode the two classes as `0` and `1` — e.g. 0 = "no disease", 1 = "disease"; 0 = "not spam", 1 = "spam". These are just numeric names for two categories. That's the first place "0 and 1" appears.
2. **The sigmoid.** Logistic Regression starts with the *same* linear part as linear regression, `z = b + w⋅x` (which can be any number). Then it passes `z` through the **sigmoid function**, which squashes any number into the range (0, 1):

```jsx
# Sigmoid — squashes any number into 0..1

σ(z) = 1 / (1 + e⁻ᶻ)

The result is a probability between 0 and 1 (e.g. 0.83 = "83% likely to be class 1"). 
A threshold (usually 0.5) then turns that probability into a final 0 or 1.

## Why the confusion is so common
Both are called "regression" and both compute the same b + w⋅x inside. 
The difference is one extra step: Linear Regression stops at the number; 
Logistic Regression wraps that number in a sigmoid to get a 0–1 probability.
```

#######################################################################

## Assumptions

Linear Regression makes assumptions (Brownlee's rules of thumb). Break them badly and it predicts poorly:

- **Linearity** — it assumes a straight-line relationship. If the trend is curved, transform a feature (e.g. `log`) or use Polynomial Features.
- **Remove collinearity** — highly correlated features make the weights unstable. Drop or combine them.
- **Watch outliers** — because errors are squared, one extreme point can drag the whole line.
- **Scale your features** — helps gradient descent and is required before Ridge/Lasso.

**⚠️ Gotcha: overfitting with many features**

With lots of features (especially correlated ones), plain Linear Regression can overfit. The fix is **regularization** — Ridge (L2) and Lasso (L1), which shrink the weights.


#######################################################################


## Interview Questions about LinearRegression

| Question | Strong answer |
| --- | --- |
| **Difference between Linear Regression and Gradient Descent?** | One is a model (predicts a number as a linear combination of features); the other is an optimization algorithm that finds the model's parameters. Different kinds of things — and LR can be trained without GD (Normal Equation). |
| **Is Linear Regression classification or regression?** | Regression — it predicts continuous numbers, not categories. |
| **Does it output 0 and 1?** | No. It outputs any real number. 0/1 are class labels in classification; the 0–1 probability comes from Logistic Regression's sigmoid. |
| **What's its cost function?** | Mean Squared Error (Ordinary Least Squares) — the average squared residual. |
| **Why square the errors?** | To remove the sign, punish big errors more, and make the cost smooth/convex for optimization. |
| **How can it be trained?** | Two ways: the Normal Equation (closed-form, exact) or Gradient Descent (iterative, scales to big data). |
| **What do the coefficients mean?** | Each weight is the change in the prediction per one-unit increase of that feature (holding others fixed). |
| **What are its assumptions?** | Linear relationship, little multicollinearity, no big outliers, (ideally) roughly Gaussian, comparable feature scales. |
| **How do you fight overfitting in LR?** | Regularization — Ridge (L2) and Lasso (L1), which shrink the weights (Chapter 6). |
| **R² of 0.0 means?** | The model is no better than always predicting the mean of y. (Negative R² = worse than the mean.) |

