Log-loss (logarithmic or cross-entropy loss) **measures how closely a logistic regression model’s predicted probabilities match the actual binary class labels**

1. Why Linear Regression’s MSE fails here:

In Linear Regression, Mean Squared Error (MSE) measures distance: $\text{MSE} = \frac{1}{N} \sum (y - \hat{y})^2$.

If you plug the non-linear **Sigmoid function** ($\hat{y} = \sigma(z)$) into MSE, the loss curve becomes wavy, filled with multiple local minima (**non-convex**). Gradient Descent would easily get trapped in a local minimum instead of finding the global minimum.

```python
Non-Convex MSE Loss                    Convex Log Loss
   (Gradient Descent Gets Stuck)          (Guaranteed Global Minimum)

      \     /\                              \                /
       \  _/  \   /_                         \              /
        \/     \_/                            \____________/
```

Log Loss produces a smooth, bowl-shaped (**convex**) curve, guaranteeing that Gradient Descent will find the optimal weights.

1. The Core Intuition: Penalizing Confidence

Log Loss measures **how wrong and how confident** the model is:
• If the true label is 1, but the model predicts y_pred = 0.99  **tiny penalty**.
• If the true label is 1, but the model predicts y_pred = 0.01  **huge (near-infinite) penalty**.

```python
###### Mathematical Formulas ######

### For a single instance, the loss for each possible class is:
If y = 1 ---> Loss = -ln(y_pred)
If y = 0 ---> Loss = -ln(1 - y_pred)

### To combine both cases into a single binary cross-entropy formula:
Loss(y, y_pred) = -(yln(y_pred) + (1-y)ln(1-y_pred))

### General formula
J(theta) = - (1/N)∑(yln(y_pred) + (1-y)ln(1-y_pred))

N = samples count
y = real y
y_pred = prediction
```

!image.png

!image.png