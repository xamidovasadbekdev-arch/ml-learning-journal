### Gradient Descent

Gradient descent is an optimization algorithm, that throws away the formula and searches. It starts with a guess and repeatedly steps downhill on the cost. 

```jsx
The steps of Gradient Descent:
1. Starting with random weights(w) - You are in a random place 
2. Prediction: y_pred = X @ w
3. Error calculation: error = y_pred-y_real
4. Gradient calculation: (2/m) @ X.T @ error
5. Renewing weights: w = w - learning_rate * gradient
6. Repeating 2-5 steps untill optimal weights come
```

Pros and Cons 

```jsx
✅ Pros	
Scales to huge numbers of features	
Scales to huge / streaming data (SGD), even if it doesn't fit in memory
Works for models with no closed-form (Logistic Regression, neural nets)
Low memory (mini-batch / SGD)

❌ Cons
You must choose a learning rate (tuning)
Needs many iterations to converge
Needs feature scaling to converge well
Only approximate — stops near the minimum
```

Comparison with Normal Equation

!image.png

Gradient Descent types:

**Batch / Stochastic / Mini-batch**

The only difference between these is *how much data you look at before each step*:

!image.png

```jsx
In scikit-learn:

You rarely write gradient descent yourself — .fit() runs it for you. 
But you'll see it directly in SGDClassifier / SGDRegressor, and 
the learning_rate hyperparameter appears all over gradient-boosting and neural networks.
```