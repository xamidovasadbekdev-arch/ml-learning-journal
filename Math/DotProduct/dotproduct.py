# Explanation
import numpy as np
homes = np.array([
    [80, 3, 10], 
    [120, 4, 5],
    [60, 2, 20],
    [200, 5, 1]
])
# info about houses - area, rooms, age

bias = 20_000
weights = np.array([150, 10_000, -2_000])

y_preds = homes @ weights + bias
y_preds2 = np.dot(homes, weights) + bias
print("Predictions using @ operator: ", y_preds)
print("Predictions using np.dot: ", y_preds2)


