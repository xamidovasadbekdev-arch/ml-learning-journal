## Polynomial Regression

```jsx
What if our data is more complex than a straight line? Surprisingly, we can use a linear
model for nonlinear data. A simple way to do this is to add powers to each features and 
then training a linear model on this.
This technique is called Polynomial Regression.
```

Let’s see in code:

```jsx
import seaborn as sns

m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)
sns.scatterplot(X, y)
```

!image.png

It is clear that linear line never fits this graph properly. We have **PolynomialFeatures in sklearn**

to transform our training data, adding square (2nd degree polynomial) of each feature set as new feature:

```jsx
>>> from sklearn.preprocessing import PolynomialFeatures
>>> poly_features = PolynomialFeatures(degree=2, include_bias=False)
>>> X_poly = poly_features.fit_transform(X)
>>> X[0]
array([-0.75275929])
>>> X_poly[0]
array([-0.75275929, 0.56664654])
```

X_poly now contains the original feature of X plus the square of this feature. Now you
can fit a LinearRegression model to this extended training data

```jsx
>>> lin_reg = LinearRegression()
>>> lin_reg.fit(X_poly, y)
>>> lin_reg.intercept_, lin_reg.coef_
(array([1.78134581]), array([[0.93366893, 0.56456263]]))
```

!image.png

Not bad : model estimates **y = 0.56x^2 + 0.93x + 1.78** when in fact the original function was

**y = 0.5x^2 + 1x + 2 + Gaussian noise.**

!!!

*PolynomialFeatures*(degree=d) transforms an array containing n features into an array containing (n + d)! / d!n! features, where n! is the factorial of n, equal to 1 × 2 × 3 × ⋯ × n. Beware of the combi‐
natorial explosion of the number of features!