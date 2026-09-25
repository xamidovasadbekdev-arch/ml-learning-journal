## Linear Regression

Linear Regression is an algorithm that predicts a number, by fitting the best straight line through your data. 

The formula for LinearRegression is:
```
y = θ0 + θ1X1 + θ2X2 + ⋯ + θnXn

Here:
y = predicted value
n = the number of features
X = i(th) feature value
θ = i(th) model parameter(including bias theta_0)
```

There are 2 ways to calculate the best weights:

1. Normal Equation(closed-form) - solves in one shot, answers directly
2. Gradient Descent(iterative form) - search for the answer step by step, walking downhill

🧩 Analogy
You want the lowest point of a valley. Normal Equation = you have the full map and a formula, so you instantly compute the exact coordinates of the bottom. Gradient Descent = you're standing in fog and walk downhill, feeling the slope, one step at a time, until you arrive. Same bottom — two ways of getting there.


## Normal Equation
Formula for Normal Equation:
```
θ = (XᵀX)⁻¹ Xᵀy

What it actually does, step by step:

Add a column of 1s to X (so the bias b gets a weight too).
Compute XᵀX (a small (n+1)×(n+1) matrix).
Invert it, multiply by Xᵀy — out come all the weights at once.
```

✅ Pros
Exact answer in one step
No hyperparameters (no learning rate to tune)
No iterations, fully deterministic
No feature scaling needed

❌ Cons
Must invert XᵀX → roughly O(n³) in the number of features
Very slow when features are many (e.g. 100,000)
Needs all data in memory at once
Breaks if XᵀX isn't invertible (redundant features)


## Gradient Descent
Formula for Gradient Descent
```
The steps of Gradient Descent:
1. Starting with random weights(w) - You are in a random place 
2. Prediction: y_pred = X @ w
3. Error calculation: error = y_pred-y_real
4. Gradient calculation: (2/m) @ X.T @ error
5. Renewing weights: w = w - learning_rate * gradient
6. Repeating 2-5 steps untill optimal weights come
```

Gradient Descent throws away the formula and instead searches. It starts with a guess and repeatedly steps downhill on the cost 


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

