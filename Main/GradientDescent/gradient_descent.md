
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



## Gradient Descent types
1. Batch GD
- In every step, it uses all the data
- Exact gradient calculation as all data used
- Slow, if 1 mln data available, in every step it uses all the data
- Good for small datasets

2. Stochastic GD(SGD):
- Every step uses one random sample
- Noisy gradient - one sample cannot make decision on the whole dataset
- Too fast - every step is fast
- Noisy but can be run away from blocking on local minimums
- Stochastic - means random

3. Mini-batch GD:
- In every step, one group samples are used(f.e - 32 or 64 samples)
- Deal between speed and precision
- Noisy than Batch GD, but more precise than SGD
- the most common used one in real life

```
!!! EPOCH is one complete pass of training dataset
```





