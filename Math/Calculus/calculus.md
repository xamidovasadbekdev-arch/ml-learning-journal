# Calculus in Machine Learning

## Why does an ML engineer need calculus?

Not to solve calculus problems by hand — but because **calculus is the actual
mechanism that powers gradient descent**. Every time a model "learns" by
adjusting its weights, calculus is running under the hood, telling it which
direction to move.

## What is a derivative?

A derivative tells you the **slope of a function at a specific point** — in
plain terms, "if I nudge the input slightly, does the output go up or down,
and how fast?"

Take a simple parabola:

```
f(x) = x²
f'(x) = 2x        (this is the derivative)
```

Plug in a few points:

```
At x = 2:   f'(2) = 4    → positive slope → function is going UP here
At x = -2:  f'(-2) = -4  → negative slope → function is going DOWN here
At x = 0:   f'(0) = 0    → flat slope → this is the minimum (bottom of the curve)
```

The derivative doesn't hand you the minimum directly — it tells you the
slope. You find the minimum by figuring out **where the slope is zero**,
since that's the flat point at the bottom (or top) of the curve.

## Why this matters: gradient descent IS calculus

A cost function (like Mean Squared Error) has the exact same parabola shape
— low in the middle, rising on both sides. Gradient descent's whole job is
to find the bottom of that curve, i.e. the weights that produce the lowest
possible cost.

Here's the connection, spelled out:

```
cost(w) = (y_pred - y)²      ← shaped like a parabola, same as f(x) = x²

gradient descent asks: "what's the slope of cost, with respect to w, right now?"

if slope is positive → w is too high → decrease w
if slope is negative → w is too low  → increase w
if slope is ~0       → we're near the minimum → stop adjusting
```

In code, this single line **is** calculus in action:

```python
w = w - learning_rate * derivative_of_cost_wrt_w
```

That's the entire "guess → check → adjust" loop from earlier — the
"adjust" step only works because a derivative told it which direction to
move.

## Concrete example: gradient descent on f(x) = x²

```python
def f(x):
    return x ** 2

def f_derivative(x):
    return 2 * x

x = 10                # start far from the minimum
learning_rate = 0.1

for step in range(20):
    grad = f_derivative(x)
    x = x - learning_rate * grad
    print(f"step {step}: x = {x:.4f}, f(x) = {f(x):.4f}")
```

Running this shows `x` sliding from `10` down toward `0` — the derivative
gets smaller as `x` approaches the minimum, so the steps naturally get
smaller too, and the process settles instead of overshooting.

## Normal Equation vs. gradient descent — the calculus difference

Both methods are solving the exact same question: "where is the cost
function at its minimum?" They just get there differently:

- **Normal Equation**: set the derivative to zero and solve directly with
algebra, in one shot. Only works when there's a clean algebraic solution
— true for plain linear regression, not true once nonlinearity (like
logistic regression or neural networks) is involved.
- **Gradient descent**: use the derivative repeatedly, taking small steps
downhill, over and over, until the slope is close enough to zero. Works
for any model, linear or not — which is why it's the dominant method
across almost all of ML and deep learning.

## What confused me at first, and what fixed it

Initially it felt like the derivative "shows you" the minimum directly.
What actually clicked was realizing the derivative only tells you the
**slope at your current position** — the minimum is found by *using* that
slope repeatedly (gradient descent) or *solving* for where it equals zero
(Normal Equation). The derivative is a direction-finder, not a destination-finder.

## One-line summary

**A derivative tells you the slope at a point; gradient descent uses that
slope, over and over, to walk downhill toward the lowest possible cost —
which means every weight update during training is calculus happening in
real time.**