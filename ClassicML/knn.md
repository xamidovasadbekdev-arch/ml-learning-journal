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

<<<<<<< HEAD
<img width="880" height="575" alt="image" src="https://github.com/user-attachments/assets/8e6351e8-f35e-455a-8422-4479164f7a7a" />
=======
!image.png

!image.png

## Distance metrics

“Closest” needs a definition of distance. There are 3 common methods - and they are all special cases of one general formula(Minkowski).

Minkowski formula:

!image.png

```python
X and Y are 2 data sample
p = Minkowski parameter(norm parameter, p>=1)

# explanation with example
X = House A (A new house you want to evaluate): (3 bedrooms, 1,500 sq ft)
Y = House B (An existing house in your database): (2 bedrooms, 1,200 sq ft) 

Here, features become:
x1= 3, x2 = 1500
y1 = 2, y2 = 1200

Let's take p=1(Manhattan distance)

|3-2|+|1500-1200| = 301 
```

There are 3 common methods to calculate distance:

| **Metric** | **Idea** | **Use when** |
| --- | --- | --- |
| **Euclidean** (L2) | Straight-line distance | Features are similar in type (default) |
| **Manhattan** (L1) | Sum of absolute diffs ("city blocks") | Features differ in type (age, height, ...) |
| **Minkowski** | Generalization of both (p parameter) | sklearn's default family (p=2 = Euclidean) |

### L1 norm (Manhattan distance / City block distance):

```python
p = 1 (sum of absolute diffs ("city blocks"))====>

L1 = d = Σᵢ |aᵢ − bᵢ|
Formula comes from Minkowski if we put p as 1 this formula comes
```

### L2 norm (Euclidean distance / Straight-line distance):

```python
p = 2 (straight-line distance)====>

L2 = d = √( Σᵢ (aᵢ − bᵢ)²)

```

### Chebyshev Distance (𝑝=∞)

if we equal p=inf, it becomes Chebyshev’s distance.

## Bias-Variance Tradeoff

Choosing K is a bias-variance tradeoff. K is hyperparameter, it directly controls bias-variance.

| **K** | **Behaviour** | **Risk** |
| --- | --- | --- |
| **Small (K=1)** | Boundary hugs every point — very flexible | Low bias, **high variance** → overfitting (one noisy point changes the answer) |
| **Large** | Smooth, averaged boundary | High bias, low variance → underfitting (very large K just predicts the majority class) |

```python
## practical K tips
Tune K by trying many values (e.g. 1–21) and picking the best on validation — exactly 
the "find optimal K" loop in your notebook. Use an odd K for binary classification to 
avoid tie votes. A common rule of thumb is K ≈ √n as a starting point.
```

## From scratch

```python
import numpy as np

def knn_predict(X_train, y_train, x_new, k=3):
    dists = np.sqrt(((X_train - x_new) ** 2).sum(axis=1))  # Euclidean to every point
    nearest = dists.argsort()[:k]                          # indices of k closest
    return np.bincount(y_train[nearest]).argmax()          # majority vote
```

## In scikit-learn

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Scaling is NOT optional for KNN - it's distance-based
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)   # K = 5
knn.fit(X_train_s, y_train)
print(knn.score(X_test_s, y_test))
```

**⚠️ The #1 KNN mistake: not scaling**

KNN is built on distance, so a feature with a big range (salary 0–100,000) completely drowns out a small one (age 0–100) — the model effectively ignores age. **Always `StandardScaler` (or MinMax) before KNN.** Your Part 3 notebook correctly fits on `X_train_scaled` — that's the right habit.

## When to use it +curse of dimensionality

| **✅ Good when** | **❌ Weak when** |
| --- | --- |
| Small/medium dataset, few features | Large dataset (slow predictions) |
| You want a simple, intuitive baseline | Many features (curse of dimensionality) |
| The decision boundary is irregular | Features unscaled or full of missing values |

**⚠️ The curse of dimensionality**

KNN shines with few features but breaks down with many. As dimensions grow, the space becomes so vast that *every* point is far from every other — "nearest" stops being meaningful, and distance-based methods fail. Fixes: feature selection or dimensionality reduction (PCA). This is a classic interview concept.

## **Interview questions**

| **Question** | **Strong answer** |
| --- | --- |
| Why is KNN called a "lazy" learner? | It does no training — it just stores the data and does all the work at prediction time. |
| Is KNN parametric or nonparametric? | Nonparametric — no learned parameters, no assumption about data shape. |
| How does it predict? | Find the K nearest points by distance; classification = majority vote, regression = mean. |
| Why must you scale features for KNN? | It's distance-based; unscaled large-range features dominate the distance. |
| Effect of small vs large K?  | Small K = low bias/high variance (overfit); large K = high bias/low variance (underfit). |
| Why use an odd K? | To avoid tie votes in binary classification. |
| What's the curse of dimensionality? | In high dimensions all points become roughly equidistant, so "nearest" loses meaning and KNN degrades. |
| Training vs prediction cost? | Training O(1) (just store); prediction O(n) per query (compare to all points) — slow at scale. |
| Default distance metric? | Euclidean (Minkowski with p=2). |
>>>>>>> 003af61 (KNN: distance metrics(Minkowski formula), coding and interview questions)
