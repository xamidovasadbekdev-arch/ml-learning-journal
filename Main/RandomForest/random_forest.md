# Random Forest

```python
Random Forest is an ensemble algorithm that handes what single decision tree cannot
handle. Specifically, The decision tree is prone to high-variance, leads to overfitting,
but an ensemble combines many models into one stronger predictor.

## Anology - wisdom of crowds
If you ask from one expert, you might get a biased or unlucky answer. This is decision
tree.
If you ask from a thousands of experts and average their guesses, the answer will be
remarkably accurate as you are doing calculations considering the whole crowd, no one
expert. Many weak, diverse opinions combine into one strong one — that's the 
"wisdom of crowds."
```

There are 2 great ensemble families:

**Bagging** - train many models *in parallel* on different data, then average. **Reduces variance.**

**Boosting** - train models in sequence, each fixing last one’s mistakes. **Reduces bias.**

## Bagging (Bootstrap Aggregating)

- Bootstrap - make many random sub-samples of the data *with replacement*(the same row can appear more than once)
- Aggregate - train one model on each sub-sample, the combine: *vote* for Classification, *average* for Regression

**Why it reduces variance**

Each tree overfits its own sample *differently*, so their errors are partly random and independent. Averaging cancels out that random error — the wobble of any one tree is smoothed away. That's why bagging is the perfect partner for high-variance models like deep decision trees.

## **OOB score & feature importance — two free gifts**

- **Out-of-Bag (OOB) score** — each bootstrap leaves out ~37% of rows; each tree is tested on the rows it didn't see, giving a **free validation score** with no separate split needed (`oob_score=True`).
- **Feature importance** — averaging how much each feature reduced impurity across all trees gives a ranking of which features matter (`feature_importances_`).

## Random Forest(one clever tweak)

Plain bagged trees have a weakness: they're *greedy*, so they often pick the same strong feature to split on first — making the trees **correlated**, and averaging correlated models helps less.

**Random Forest = bagging of trees + one tweak**:

### Feature Bagging

```python
At *each split*, the tree may only consider a **random subset of features** (not all of them).
This forces the trees to be different from each other — **decorrelated** — so averaging 
them helps much more. Common default: m = √p features per split for classification 
(p/3 for regression), where p = total features. This is called **FEATURE BAGGING**.

## Anology
If every voter reads only the same one newspaper, they'll all vote the same way 
(correlated). Force each to read a different random mix of sources, and the crowd's 
combined judgment becomes much wiser. The random feature subset is that "different 
sources."
```

## Sklearn Coding

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,     # number of trees (more = better, then plateaus)
    max_features="sqrt",  # m = √p features per split (the RF tweak)
    max_depth=None,       # trees grown deep on purpose
    oob_score=True,       # free validation score
    n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)   # no scaling needed (tree-based)
print(rf.oob_score_, rf.feature_importances_)
```

- **`n_estimators=200`**: **Spot on.** 200 trees will be grown. Each one gets its own bootstrap sample. At the end, they will all vote (since it's a `Classifier`) to give you the final average/majority prediction.
- **`max_features="sqrt"(Feature Bagging)`**: **Perfect math.** If you have 100 features, sqrt(100) = 10. Each time a tree needs to split, it will randomly pick 10 features out of the 100 and choose the best one.
- **`max_depth=None`**: The trees will grow as deep as possible until they cleanly separate the data. In Random Forest, we *want* individual deep trees because even though a single deep tree overfits, the averaging process cancels that overfitting out.
- **`oob_score=True`**: This tells sklearn to calculate that free validation score using the ~37% of left-out data for each tree. It saves you from needing a separate validation set.
- **`n_jobs=-1`**: **Brilliant.** Because every tree is completely independent of the others (Tree 2 doesn't care what Tree 1 is doing), they can be trained at the exact same time. `n_jobs=-1` tells your laptop to use *all* available CPU cores to train multiple trees in parallel, heavily speeding up the process.
- **`rf.oob_score_`**: A single accuracy score (e.g., `0.88`), showing how well the forest predicts unseen data based on the OOB samples.
- **`rf.feature_importances_`**: An array of 100 numbers (one for each feature). It ranks which features were chosen most often and provided the best splits across all 200 trees, letting you know exactly which columns are your heaviest hitters.

```python
!!! **Difference**
In Decision Tree, more *max_depth* leads to overfitting(high-variance). But
In Random Forest, more *n_estimators* does not mean overfitting. We can add any more 
trees as much as we want.
```

## **Bagging vs Boosting (the interview pair)**

|  | **Bagging (Random Forest)** | **Boosting (Ch.13)** |
| --- | --- | --- |
| Training | Parallel & independent | Sequential (each fixes the last) |
| Mainly reduces | **Variance** | **Bias** |
| Base models | Deep (low-bias, high-variance) trees | Shallow (high-bias) trees / stumps |
| Overfitting | Resistant (more trees is safe) | Can overfit (needs care) |
| Speed | Parallelizable (fast) | Sequential (slower) |

![Uploading image.png…]()


## **When to use it + gotchas**

| **✅ Strengths** | **❌ Weaknesses** |
| --- | --- |
| Strong, reliable accuracy with little tuning | Less interpretable than one tree ("forest" of rules) |
| Resists overfitting; handles non-linear, mixed data | Larger/slower model; more memory |
| No scaling; gives importances & OOB score | Often beaten by boosting on tabular competitions |

## **Interview questions**

| **Question** | **Strong answer** |
| --- | --- |
| What is bagging? | Train many models on bootstrap samples (random, with replacement) and aggregate (vote/mean) to reduce variance. |
| What is a bootstrap sample? | A random sample of the data taken with replacement (same row can repeat). |
| How is Random Forest different from plain bagging? | It also picks a random subset of features at each split, which decorrelates the trees so averaging helps more. |
| Bagging vs Boosting? | Bagging = parallel, reduces variance; Boosting = sequential, reduces bias. |
| What is the OOB score? | Validation accuracy from the ~37% of rows left out of each bootstrap — a free estimate of test error. |
| Does more trees overfit a Random Forest? | No — accuracy plateaus; more trees is safe (just slower). |
| Default max_features? | √p for classification, p/3 for regression. |
| Does RF need feature scaling? | No — it's tree-based. |

## **Summary**

| **Topic** | **The one thing to remember** |
| --- | --- |
| **Ensemble** | Combine many models → wisdom of crowds. Bagging (variance) vs Boosting (bias). |
| **Bagging** | Train on bootstrap samples + aggregate → reduces variance of high-variance models. |
| **Random Forest** | Bagged trees + random feature subset per split → decorrelated, stronger. |
| **Free gifts** | OOB score (free validation) + feature_importances_. |
| **Practical** | Little tuning, resists overfitting, no scaling; raise n_estimators freely. |
