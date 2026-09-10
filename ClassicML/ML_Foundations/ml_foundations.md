## What is Machine Learning(ML)?

Machine Learning is the field of study that gives computers the ability to learn from data instead of hand-written rules.

#### Traditional Programming vs Machine Learning

```jsx
+-----------------------+--------------------------+--------------------------+--------------------------+
| Approach              | You give it...           | Who writes the logic     | You get...               |
+-----------------------+--------------------------+--------------------------+--------------------------+
| Traditional           | Data + Rules             | The programmer (by hand) | Answers                  |
| programming           |                          |                          |                          |
+-----------------------+--------------------------+--------------------------+--------------------------+
| Machine Learning      | Data + Answers           | The model (learns them)  | Rules (a trained model)  |
+-----------------------+--------------------------+--------------------------+--------------------------+
```

```jsx
**Anology**

Imagine you are teaching a child what a cat is. You do not need to give instructions, 
like fury body, 2 ears and eyes, … You just show him many cats and after enough examples
he can recognize if you show him a new cat they have never seen. 

ML works in the same way: if you give model enough labelled examples, 
it learns pattern itself.
```

### The one idea behind every algorithms:

Every algorithm try to do the same one thing  - find a function that maps input X to an output Y.

```jsx
Y = f(X) + e

X = inputs(features) - age, blood pressure, cholestrol
Y = outputs - heart disease or not, spam or ham
f = true pattern we wanna predict, that connecting X to Y
e = irreducible error, random noise that we can never remove

Different algorithms(Linear Regression, Random Forest, Nueral Nets, ...) are just 
different strategies for guessing <f>
```

### When we should use ML?

- Problems that need long list of hand-written rules (like spam)
- Problems with no good traditional solution (like speech recognition)
- Changing environments

#######################################################################

## Types of Machine Learning

ML systems are grouped by what kinda data they learn from - specifically, whether data comes with the right answers(labels) or not.

```jsx
+---------------+------------------------------+-----------------------+----------------------------------+
| Type          | Data it learns from          | Goal                  | Example                          |
+---------------+------------------------------+-----------------------+----------------------------------+
| Supervised    | Inputs and labels (X and Y)  | Predict Y for new X   | Predict house price, detect spam |
+---------------+------------------------------+-----------------------+----------------------------------+
| Unsupervised  | Only inputs (X), no labels   | Find hidden structure | Customer segmentation, PCA       |
+---------------+------------------------------+-----------------------+----------------------------------+
| Reinforcement | Rewards from an environment  | Learn the best strategy| Game-playing AI, robotics        |
+---------------+------------------------------+-----------------------+----------------------------------+

## Anology
# Supervised learning is like studying with a teacher who has the answer key: 
the model makes a guess, the teacher corrects it, and it improves. 
# Unsupervised learning has no teacher and no answer key — 
the model is handed a pile of data and asked to find interesting groupings on its own.
```

#######################################################################

### Supervised learning types

Supervised Learning divides into 2 groups by what the output Y looks like:

Regression and Classification

```jsx
+--------------------+------------------------------------------+------------------------------------+
|                    | Classification                           | Regression                         |
+--------------------+------------------------------------------+------------------------------------+
| Output (Y)         | A category / class                       | A continuous number                |
+--------------------+------------------------------------------+------------------------------------+
| The question       | "Which group does it belong to?"         | "How much / how many?"             |
+--------------------+------------------------------------------+------------------------------------+
| Examples           | Spam / not spam · cat / dog              | House price · temperature          |
+--------------------+------------------------------------------+------------------------------------+
| Typical metrics    | Accuracy, Precision, Recall, F1, ROC-AUC | MSE, RMSE, MAE, R²                 |
+--------------------+------------------------------------------+------------------------------------+
| Example algorithms | Logistic Regression, SVM, Random Forest  | Linear Regression, Ridge, RF Regressor |
+--------------------+------------------------------------------+------------------------------------+
```

```jsx
## Anology
The very first question to ask about any supervised problem is: 
"Is Y a category or a number?" Many algorithms come in both flavours — 
RandomForestClassifier and RandomForestRegressor — so once students identify the 
problem type, choosing the right tool is half done.
```

#######################################################################

# Parametric vs Non-Parametric Machine Learning Algorithms

Understanding the difference between **parametric** and **non-parametric** algorithms is not just about memorizing which algorithms belong to which category. The real difference is about **how the model represents what it learns from the data**.

---

## 1. Parametric Algorithms

A **parametric algorithm** assumes a specific mathematical structure for the relationship between inputs and outputs.

The model has a **fixed number of parameters** for a given model structure.

For example, suppose we want to predict house prices using:

- Size
- Number of bedrooms
- Age of the house

A Linear Regression model might learn:

```
Price = w₁ × Size + w₂ × Bedrooms + w₃ × Age + b
```

The model learns:

```
w₁
w₂
w₃
b
```

So there are **4 parameters**.

Even if we increase our dataset from:

```
100 houses
```

to:

```
100,000 houses
```

the model still has:

```
3 weights + 1 bias = 4 parameters
```

The values of the parameters change during training, but their **number remains fixed**.

### Important idea

"Fixed number of parameters" does **not** mean the parameter values are fixed.

For example:

```
Before training:
w₁ = 0
w₂ = 0
w₃ = 0
b  = 0

After training:
w₁ = 2.31
w₂ = 15.72
w₃ = -4.18
b  = 1200
```

The values changed, but there are still only 4 parameters.

---

## 2. What About Feature Engineering?

Feature engineering does not make a parametric model non-parametric.

Suppose we start with:

```
Age
Blood Pressure
Cholesterol
```

We have 3 features.

Then we create a new feature:

```
BMI
```

Now Linear Regression can learn:

```
w₁ × Age
+ w₂ × Blood Pressure
+ w₃ × Cholesterol
+ w₄ × BMI
+ b
```

Now there are:

```
4 weights + 1 bias = 5 parameters
```

But this is still a **parametric model**.

Why?

Because once the model specification is decided, its number of parameters is fixed.

The model itself is not dynamically creating new weights as more training examples arrive.

---

# 3. Non-Parametric Algorithms

A **non-parametric algorithm** does not assume a fixed mathematical form with a fixed number of parameters.

Its model complexity can **grow or adapt according to the training data**.

This is where the word "flexible" becomes important.

It does NOT mean:

> "The number of features changes."
> 

Your features can remain exactly the same.

For example:

```
Age
Salary
Experience
```

You still have 3 features.

What changes is the **amount and complexity of information the model needs to represent the training data**.

---

# 4. K-Nearest Neighbors (KNN)

KNN is a very intuitive example.

Suppose we have:

```
Person 1 → [25, 3000, 2]
Person 2 → [30, 4000, 5]
Person 3 → [40, 7000, 10]
Person 4 → [22, 2500, 1]
```

The features are:

```
Age
Salary
Experience
```

KNN does not learn:

```
Price = w₁ × Age + w₂ × Salary + w₃ × Experience + b
```

Instead, it essentially keeps the training examples and uses their distances when making predictions.

If we have:

```
100 training examples
```

KNN stores those examples.

If we have:

```
10,000 training examples
```

KNN stores far more information.

If we have:

```
1,000,000 training examples
```

the model's stored representation becomes much larger.

The number of features is still:

```
3
```

But the amount of information used by the model grows with the dataset.

This is one of the important ideas behind **non-parametric learning**.

---

# 5. Decision Trees

Decision Trees give us another great example.

Suppose we have the same 3 features:

```
Age
Salary
Experience
```

A small dataset might produce a simple tree:

```
             Salary > 5000?
              /           \
            No             Yes
            |               |
    Experience > 3?       Age > 35?
```

With more complicated data, the tree can grow:

```
             Salary > 5000?
             /             \
           ...              ...
          /  \             /   \
        ...  ...         ...   ...
              ...
```

The tree can contain:

```
10 nodes
50 nodes
500 nodes
1000 nodes
```

depending on the data and the model's hyperparameters.

So again:

```
Number of features → stays the same
Model complexity   → can grow
```

This is the flexibility we mean.

---

# 6. Parametric vs Non-Parametric

|  | Parametric | Non-Parametric |
| --- | --- | --- |
| Assumes fixed mathematical form? | Usually yes | Usually no |
| Number of learned parameters/model components | Fixed for a given specification | Can grow/adapt with data |
| Number of features | Does not inherently change | Does not inherently change |
| Model complexity | More constrained | More flexible |
| Example | Linear Regression | KNN |
| Example | Logistic Regression | Decision Tree |
| Example | Naive Bayes | Random Forest |
| Example | Neural Network | Kernel SVM |

---

# 7. The Most Important Mental Model

Think of Linear Regression like this:

```
DATA
  ↓
Fixed mathematical structure
  ↓
w₁, w₂, w₃, ..., b
```

The model compresses the information into a relatively fixed set of learned parameters.

For example:

```
1,000,000 rows
        ↓
   4 parameters
```

The parameters summarize the relationship the model assumes.

---

For a non-parametric model, think:

```
DATA
  ↓
Flexible representation
  ↓
Model structure can grow/adapt
```

For example, a Decision Tree might go from:

```
20 nodes
```

to:

```
200 nodes
```

as the data becomes more complex.

KNN is even more direct:

```
More training examples
        ↓
More examples stored/used
        ↓
Larger model representation
```

---

# 8. One Important Correction: "Weights"

It is better not to say:

> "Non-parametric algorithms have a flexible number of weights."
> 

That can be misleading.

A better statement is:

> **Non-parametric algorithms have a flexible model complexity or representation that can grow with the data.**
> 

Why?

Because different algorithms represent what they learn differently.

For example:

### Linear Regression

Uses explicit weights:

```
w₁, w₂, w₃, b
```

### KNN

Doesn't learn traditional weights like Linear Regression. It mainly stores training observations and uses distances/neighbors.

### Decision Tree

Learns a structure consisting of:

```
splits
nodes
branches
leaves
```

So "parameters" and "weights" are not always interchangeable.

---

# 9. Why Is This Useful?

The difference gives us a trade-off.

### Parametric models

Because they have a constrained structure, they can be:

- Faster
- More memory efficient
- Easier to interpret
- Less data-hungry in some situations

But they can struggle when the real relationship is very complicated and doesn't match their assumptions.

### Non-parametric models

Because they are more flexible, they can model complicated relationships.

But that flexibility can require:

- More data
- More computation
- More memory
- More careful control of overfitting

---

# 10. Final Intuition

The simplest way to remember everything is:

> **Parametric = "I decide the shape of the model first, then training learns the values."**
> 

> **Non-parametric = "I don't force the model into one fixed shape; the learned representation can adapt to the data."**
> 

And remember:

```
Features ≠ Parameters ≠ Hyperparameters
```

For example:

```
Features:
Age, Salary, Experience

Parameters:
w₁, w₂, w₃, b

Hyperparameters:
learning_rate
max_depth
n_neighbors
regularization strength
```

A model can have a **fixed number of features** while still having a flexible model complexity.

That is the core idea behind **parametric vs non-parametric machine learning**.

#######################################################################

## ML workflow with real example

Every ML project follows the same 6 steps:

1. Problem definition - What are we predicting? Classification or regression? How do we measure success?
2. **Data** — Collect, load, and clean the data.
3. **Evaluation** — Decide the metric and what score is "good enough".
4. **Features** — Prepare and engineer the inputs (encoding, scaling, selection).
5. **Modelling** — Choose a model, train it (fit), predict, tune it.
6. **Experimentation** — Analyse results, improve, and repeat.

Let’s do an end-to-end project with real world example:

Classification problem:

predict class `1` (has disease) or `0` (no disease).

The dataset's columns (the **features** X, and the **label** y = `target`)

The entire workflow — load → split → choose model → fit → predict → evaluate — in a dozen lines:

```jsx
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Get data ready: separate features (X) from the label (y)
df = pd.read_csv("heart-disease.csv")
X = df.drop("target", axis=1)   # all columns except target
y = df["target"]                # the thing we predict (0 or 1)

# 2. Split: learn on train, judge on test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Choose a model + hyperparameters
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 4. Fit (train) and 5. predict
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 6. Evaluate on unseen data
print("Accuracy:", model.score(X_test, y_test))   # e.g. ~0.85

### Anology
Look back at it: features vs label, train/test split with stratify, choosing a model, 
fit → predict → score, and accuracy as the metric. Every algorithm chapter that follows
just swaps the one model line — the surrounding workflow never changes. 
That is the whole point of scikit-learn's consistent API
```

#######################################################################

## **Train / Validation / Test split**

The rule: Never judge a model on the same data it learned from. A model simply can memorize and perform perfectly - importantly, it tells nothing about new real-world data. 

The data split should be like this:

```jsx
+----------------+--------------+-------------------------------------------------------------+
| Set            | Share        | What it's for                                               |
+----------------+--------------+-------------------------------------------------------------+
| Training set   | ~60–80%      | The model learns its parameters here (fit).                 |
+----------------+--------------+-------------------------------------------------------------+
| Validation set | ~10–20%      | Choosing hyperparameters and comparing models.              |
+----------------+--------------+-------------------------------------------------------------+
| Test set       | ~10–20%      | The final, honest score — used only once, at the very end.  |
+----------------+--------------+-------------------------------------------------------------+
```

```jsx
⚠️ Warning: Data Leakage
The most common and most dangerous mistake. Do not do any preprocessing before the split. 
If you fit a StandardScaler or SimpleImputer on the whole dataset, 
information about the test set leaks into training, and your score looks better 
than it really is. 
Correct order: split first → fit preprocessing on the training set only → apply it to the test set. 
A scikit-learn Pipeline handles this automatically.
```

#######################################################################

## Overfitting and Underfitting

These are 2 errors that model can fail. The real goal is generalization: doing well on new, unseen data, not on training data.

```jsx
+----------------+--------------------------------------+----------------------------------------------+
|                | Underfitting                         | Overfitting                                  |
+----------------+--------------------------------------+----------------------------------------------+
| Meaning        | Model too simple to capture pattern  | Model memorised the training data and noise  |
+----------------+--------------------------------------+----------------------------------------------+
| Training score | Low                                  | Very high                                    |
+----------------+--------------------------------------+----------------------------------------------+
| Test score     | Low                                  | Low                                          |
+----------------+--------------------------------------+----------------------------------------------+
| Fix            | More complex model, better features  | More data, regularization, simpler model     |
+----------------+--------------------------------------+----------------------------------------------+

## Anology
A student who underfits barely studied — bad on both practice and the real exam. 
A student who overfits memorised the practice answers — perfect on practice, 
lost on the real exam. We want the student who actually understood the material.

## Diagnosis
Compare training vs test score: both low → underfitting; 
training high but test much lower → overfitting; 
both high and close → a good model ✅

```

#######################################################################

## Key Terms(Glossary)

```jsx
+---------------------+--------------------------------------------------------------+
| Term                | Meaning                                                      |
+---------------------+--------------------------------------------------------------+
| Feature (X)         | An input variable (a column).                                |
+---------------------+--------------------------------------------------------------+
| Label / Target (y)  | The output variable we want to predict.                      |
+---------------------+--------------------------------------------------------------+
| Instance / Sample   | One row — a single example.                                  |
+---------------------+--------------------------------------------------------------+
| Model               | The function learned from data that maps X to y.             |
+---------------------+--------------------------------------------------------------+
| Parameter           | A value the model learns during fit() (e.g. regression       |
|                     | coefficients).                                               |
+---------------------+--------------------------------------------------------------+
| Hyperparameter      | A setting you choose before training (e.g. n_estimators, k). |
+---------------------+--------------------------------------------------------------+
| Loss / Cost function| A measure of how wrong the model is — training minimises it. |
+---------------------+--------------------------------------------------------------+
| Generalization      | How well the model performs on new, unseen data — the real   |
|                     | goal.                                                        |
+---------------------+--------------------------------------------------------------+

**# N1-Confusion**:
Parameter ≠ Hyperparameter. 
The model finds parameters on its own during training. 
You set hyperparameters beforehand, and tune them. Beginners mix these up constantly.
```

## Common mistakes(Gotchas)

1. **Data Leakage** - preprocessing before splitting. 

How to fix: doing it inside <Pipeline> after splitting.

1. **Tuning on the test set** - test set should be touched once, at the very end. 
2. **Forgetting to scale** - distance-based models(KNN, SVM, Logistic Reg) need <StandardScaler>. Tree-based models do not.
3. **Trusting accuracy on imbalanced data** — if 95% of patients are healthy, "everyone is healthy" scores 95% and is useless. Use Precision / Recall / F1.
4. **No `random_state`** — results won't reproduce. Set `random_state=42` everywhere.
5. **Forgetting `stratify=y`** — without it, a classification split can distort the class balance.

#######################################################################

## Interview Questions

These foundation questions open almost every ML interview. Crisp answers:

```jsx
+-----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------+
| Question                          | Strong answer                                                                                                                     |
+-----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------+
| What is Machine Learning?         | Getting a computer to learn patterns from data instead of being given explicit hand-written rules.                                 |
+-----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------+
| AI vs ML vs Deep Learning?        | AI is the broad goal (machines acting intelligently); ML is a subset (learning from data); Deep Learning is a subset of ML using |
|                                   | multi-layer neural networks.                                                                                                      |
+-----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------+
| Supervised vs Unsupervised?       | Supervised has labelled data (X and y) and predicts y; unsupervised has only X and finds structure (clusters, components).        |
+-----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------+
| Classification vs Regression?     | Classification predicts a category; regression predicts a continuous number.                                                     |
+-----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------+
| What is overfitting? How do you   | Model memorises training data and fails on new data. Detect: high train score but low test score. Fix: more data,                 |
| detect & fix it?                  | regularization, simpler model, cross-validation.                                                                                 |
+-----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------+
| Explain the bias-variance         | Bias = error from too-simple assumptions (underfitting); variance = sensitivity to the training data (overfitting). Lowering    |
| tradeoff.                         | one tends to raise the other; you seek the balance.                                                                               |
+-----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------+
| Parameter vs hyperparameter?      | Parameters are learned during fit() (e.g. weights); hyperparameters are set by you beforehand (e.g. n_estimators) and chosen    |
|                                   | by tuning.                                                                                                                        |
+-----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------+
| Why split into train/validation/  | Train to fit, validation to tune/compare, test for a single honest final score on unseen data.                                    |
| test?                             |                                                                                                                                   |
+-----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------+
| What is data leakage?             | When information from the test set (or the future) sneaks into training — e.g. scaling before the split — giving falsely high   |
|                                   | scores.                                                                                                                           |
+-----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------+
| Why not always use the most       | Complex models overfit, need more data, are slower and harder to interpret. Start simple (a baseline) and add complexity only    |
| complex model?                    | if it helps.                                                                                                                      |
+-----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------+
```