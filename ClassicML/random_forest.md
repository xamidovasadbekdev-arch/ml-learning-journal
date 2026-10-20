# RandomForest

```jsx
A Random Forest is an ensemble machine learning algorithm that combines the predictions 
of hundreds of individual decision trees to create a single, highly accurate model. 
Instead of relying on one tree—which is prone to errors and overfitting—the algorithm 
operates on a democratic principle: the collective vote of a diverse crowd outperforms 
any single expert. That's the "wisdom of crowds."
```

To ensure the trees do not make identical mistakes, the algorithm introduces **randomness** in two distinct ways during training.

There are two great ensemble families (and this is a key interview distinction):

- **Bagging** — train many models *in parallel* on different data, then average. Reduces **variance**.
- **Boosting** — train models *in sequence*, each fixing the last one's mistakes. Reduces **bias**

## Row Randomization(Bootstrap Agregating = Bagging)

Two words: **Bootstrap** + **Aggregating**.

- **Bootstrap** = make many random sub-samples of the data *with replacement* (the same row can appear more than once).
- **Aggregate** = train one model on each sub-sample, then combine: *vote* for classification, *average* for regression.

The algorithm does not feed the exact same dataset to every tree. Instead, it creates a unique training set for each tree using **Bootstrap Aggregating (Bagging)**. The model randomly samples rows from the original dataset **with replacement**, meaning some data points are repeated and others are left out entirely for each tree.

```jsx
Bagging in 3 steps
1) Draw N bootstrap samples (random, with replacement)
2) Train one model (usually a deep tree) on each sample
3) Predict = majority vote (classification) or mean (regression)
```

**ℹ️ Why it reduces variance**

```jsx
Each tree overfits its own sample *differently*, so their errors are partly random and 
independent. Averaging cancels out that random error — the wobble of any one tree is 
smoothed away. That's why bagging is the perfect partner for high-variance models like 
deep decision trees.
```

## **Random Forest (one clever tweak)**

Plain bagged trees have a weakness: they're *greedy*, so they often pick the same strong feature to split on first — making the trees **correlated**, and averaging correlated models helps less. **Random Forest = bagging of trees + one tweak**:

**The tweak that makes a Forest**

At *each split*, the tree may only consider a **random subset of features** (not all of them). This forces the trees to be different from each other — **decorrelated** — so averaging them helps much more. Common default: `m = √p` features per split for classification (`p/3` for regression), where p = total features.

**Analogy**

If every voter reads only the same one newspaper, they'll all vote the same way (correlated). Force each to read a different random mix of sources, and the crowd's combined judgment becomes much wiser. The random feature subset is that "different sources."

## **OOB score & feature importance — two free gifts**

• **Feature importance** — averaging how much each feature reduced impurity across all trees gives a ranking of which features matter (`feature_importances_`).

• **Out-of-Bag (OOB) score** — each bootstrap leaves out ~37% of rows; each tree is tested on the rows it didn't see, giving a **free validation score** with no separate split needed (`oob_score=True`).

## In scikit-learn

```jsx
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

**💡 More trees never overfit**

Unlike a single deep tree, adding more trees to a Random Forest **does not overfit** — accuracy just plateaus. So set `n_estimators` as high as your compute budget allows. Also: `ExtraTreesClassifier` adds even more randomness (random split thresholds) for faster training and sometimes better results.

## **Bagging vs Boosting (the interview pair)**

|  | **Bagging (Random Forest)** | **Boosting (Ch.13)** |
| --- | --- | --- |
| Training | Parallel & independent | Sequential (each fixes the last) |
| Mainly reduces | **Variance** | **Bias** |
| Base models | Deep (low-bias, high-variance) trees | Shallow (high-bias) trees / stumps |
| Overfitting | Resistant (more trees is safe) | Can overfit (needs care) |
| Speed | Parallelizable (fast) | Sequential (slower) |

<img width="857" height="327" alt="image" src="https://github.com/user-attachments/assets/80ca86e9-502d-498a-9740-08d925a3c4fc" />

<img width="872" height="299" alt="image" src="https://github.com/user-attachments/assets/5935ea59-3c89-41ba-8268-e0b46eaab9c5" />


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

---

## **Summary**

| **Topic** | **The one thing to remember** |
| --- | --- |
| **Ensemble** | Combine many models → wisdom of crowds. Bagging (variance) vs Boosting (bias). |
| **Bagging** | Train on bootstrap samples + aggregate → reduces variance of high-variance models. |
| **Random Forest** | Bagged trees + random feature subset per split → decorrelated, stronger. |
| **Free gifts** | OOB score (free validation) + feature_importances_. |
| **Practical** | Little tuning, resists overfitting, no scaling; raise n_estimators freely. |
