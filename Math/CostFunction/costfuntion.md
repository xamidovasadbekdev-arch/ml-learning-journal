## Cost function

```jsx
A cost function measures how wrong your model's predictions are — as a single number. 
Lower = better model. That's the whole idea.
```

### Where it comes from — using what you already know

You already know:

```
y        = real/actual values
y_pred   = X @ w + b   (your model's guesses)
```

The **error** for one prediction:

```
error = y_pred - y
```

But you have *many* errors (one per house/data point) — you need to turn all of them into **one single score** that says "overall, how bad is this model?" That single score is the **cost function**.

### The most common one: MSE (Mean Squared Error)

```
cost = average of (y_pred - y)²
```

In code:

```python
errors = y_pred - y
cost = np.mean(errors ** 2)
```

Walking through a tiny example:

```jsx
y = [3, 5, 7]      # real
y_pred = [4, 5, 6]      # model's guesses

errors = [1, 0, -1]         # y_pred - y
squared = [1, 0, 1]         # square each error
cost = (1 + 0 + 1) / 3 = 0.667
```

### Why **square** the errors? (this trips people up, worth explaining)

1. **Gets rid of negative signs** — an error of `5` and `+5` are both "5 wrong," but without squaring they'd cancel out when averaged. Squaring makes everything positive.
2. **Punishes big mistakes more** — an error of `10` becomes `100` when squared, but an error of `2` only becomes `4`. This pushes the model to avoid being *very* wrong on any single point, not just "roughly okay on average."

### How this connects to everything you've already learned

- **Normal Equation** — literally *is* the formula that finds the `w` (and bias) that makes this cost as small as mathematically possible, in one shot.
- **Gradient descent** — is the "guess → check error → adjust" loop, where the "check error" step is exactly calculating this cost function, and "adjust" means nudging `w`/`b` in the direction that makes cost go down.

!image.png

**Cost function** = the general concept: "a number that measures how wrong the model is"

**MSE** = one specific formula for calculating that number, used for regression problems

```jsx
Regression (predicting numbers, e.g. house price) --->	MSE (Mean Squared Error)
Classification (predicting categories, e.g. spam/not spam) --->	Cross-entropy / Log loss
Some robust regression cases --->	MAE (Mean Absolute Error)
```