# Decision Tree

```python
CART = Classification And Regression Trees. The idea: repeatedly ask yes/no questions 
about your features, each one chosen to split the data into purer groups, until you can 
make a prediction.
```

# 1. The structure

- **Root node**: all the training data
- **Internal nodes**: a question like `age <= 6?`
- **Leaves**: the final prediction
- **Depth**: the number of questions
- **Branches**: yes/no seperated paths

CART always makes **binary splits** (exactly two children). Every split is of the form `feature <= threshold` for numeric features, or `feature in {some categories}` for categorical ones.

<img width="867" height="614" alt="image" src="https://github.com/user-attachments/assets/0834f01a-39a1-4a9b-aa05-8d681430406e" />


# 2. How does it measure “impurity”?

### Gini Impurity vs Entropy

<img width="222" height="78" alt="image" src="https://github.com/user-attachments/assets/0ca7eae5-0c8f-483e-bed0-97dbe0498622" />


where p_k is the fraction of samples in the node belonging to class k.

- Node with 100% one class: Gini = 0 (pure)
- Node with 50/50 split (2 classes): Gini = 0.5 (worst)
<img width="196" height="30" alt="image" src="https://github.com/user-attachments/assets/54a02483-6470-42ce-a0d3-bb5faf90089a" />


where p_k is the fraction of samples in the node belonging to class k.

- Node with 100% one class: Gini = 0 (pure)
- Node with 50/50 split (2 classes): Gini = 0.5 (worst)

<img width="891" height="613" alt="image" src="https://github.com/user-attachments/assets/5cbc56d0-86df-4024-b975-d68092541e59" />

|  | **Gini** | **Entropy**  |
| --- | --- | --- |
| Measures | Probability of mislabeling a random pick | Information / "surprise" |
| Speed | Faster (no logarithm) | Slightly slower (log) |
| Result | Almost always nearly identical trees | Almost always nearly identical trees |

In scikit-learn, `criterion="gini"` (default) or `"entropy"`. **For regression** trees there's no Gini — splits minimize squared error (`Σ(yᵢ − ŷ)²`) and a leaf predicts the *mean* of its samples.

**Which to pick?**

It rarely matters — they give very similar trees. Gini is the default because it's a bit faster. Don't lose sleep choosing; spend your time on `max_depth` instead.

# **3. How CART picks a split (greedy search)**

At each node, CART tries **every feature and every possible threshold**, and picks the one that reduces impurity the most:

<img width="393" height="72" alt="image" src="https://github.com/user-attachments/assets/b247b8f9-21c7-4818-9c0c-21e91aceb5a3" />


Lowest weighted impurity wins. For numeric features, candidate thresholds are the **midpoints between consecutive sorted unique values**.

# 4. Overfitting and Pruning

**⚠️ Trees overfit by nature**

Left unconstrained, a tree keeps splitting until every leaf is one training point — memorizing the data perfectly (and noise with it). A fully-grown tree is the textbook example of **high variance**: change a few training rows and you get a very different tree.

Two cures:

- **Pre-pruning** (stop early): `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_leaf_nodes`.
- **Post-pruning**: grow fully, then cut weak branches — **cost-complexity pruning** via `ccp_alpha` (bigger alpha = more pruning).

# 5. In sklearn(no scaling needed !!!)

```python
from sklearn.tree import DecisionTreeClassifier

# No StandardScaler! Trees split on thresholds, so scale is irrelevant.
tree = DecisionTreeClassifier(max_depth=4, random_state=42)
tree.fit(X_train, y_train)
print(tree.score(X_test, y_test))
print(tree.feature_importances_)   # which features the splits relied on
```

**ℹ️ The big contrast: no scaling**

Unlike KNN, SVM, and Logistic Regression, decision trees **do not need feature scaling** — a split like "age > 30" works the same whatever the units. This is a favourite interview distinction (tree-based vs distance/gradient-based models).
<img width="887" height="533" alt="image" src="https://github.com/user-attachments/assets/a767dabe-82f6-45b0-a2ab-9d3b92f74367" />


# 6. When to use it

| **✅ Strengths** | **❌ Weaknesses** |
| --- | --- |
| Fully interpretable ("white box" — you can read the rules) | Overfits easily; high variance |
| No scaling; handles mixed numeric + categorical, non-linear | Boxy, axis-aligned boundaries (can't draw a diagonal) |
| Fast; gives feature importances | Unstable — small data changes → different tree |

# 7. Interview Questions

| **Question** | **Strong answer** |
| --- | --- |
| What does CART stand for? | Classification And Regression Trees. |
| How does a tree decide where to split? | Greedy recursive binary splitting — try all features/thresholds, pick the split that most reduces impurity. |
| Gini vs Entropy? | Both measure node impurity (0 = pure). Gini is faster (no log); entropy relates to information gain. Results are nearly identical. |
| How do regression trees split? | Minimize squared error; a leaf predicts the mean of its samples. |
| Why do trees overfit, and how to prevent it? | They split until leaves are pure (memorize). Prevent with max_depth / min_samples_leaf (pre-prune) or ccp_alpha (post-prune). |
| Do decision trees need feature scaling? | No — they split on thresholds, so scale doesn't matter. |
| Are they high or low variance? | High variance (unstable) — which is why we ensemble them. |
| What kind of decision boundary do they make? | Axis-aligned, boxy rectangles. |
