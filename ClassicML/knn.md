# KNN(K-Nearest Neighbours)

KNN is used for both classification and regression, in which its work based on the principle that similar data points sit close to each other in a feature space.

**🧩 Analogy**

"Tell me who your friends are, and I'll tell you who you are." To guess something about a new person, look at the handful of people most similar to them and assume they're alike. To predict a neighborhood's house price, look at the nearest few houses.

KNN has three famous nicknames, each describing a real property — all common interview terms:

| **Name** | **Because...** |
| --- | --- |
| **Lazy learner** | It does *no* work at training time — all the work happens at prediction. |
| **Instance-based** | Its "model" is literally the stored training instances — there are no learned parameters. |
| **Nonparametric** | It makes no assumption about the shape of the data. |

```python
"Training" is instant (just store the data, O(1)). But prediction is slow — to classify 
one point it must measure distance to every training point (O(n)). This is the opposite 
of most models (slow to train, fast to predict). A great interview contrast.
```

### How it predicts:

1. Measure the **distance** from the new point to every training point.
2. Pick the **K closest** ones (the "neighbors").
3. **Classification:** majority vote of the neighbors' classes. **Regression:** the mean (or median) of the neighbors' values.

It even gives probabilities: with K=5 and 4 neighbors of class 1, `P(class 1) = 4/5 = 0.8`.

<img width="880" height="575" alt="image" src="https://github.com/user-attachments/assets/8e6351e8-f35e-455a-8422-4479164f7a7a" />
