## Why data preparation matters

## Garbage in, Garbage out

A model is only as good as the data u feed it. Almost every algorithm needs its inputs to be:

Numerical - most algorithms cannot handle text categories

Complete - most algorithms crash on missing data

Scaled - distance and  gradient based models are distorted with very different ranges

```jsx
**Géron's advice**: 
Never prepare data by hand in one-off steps. 
Write it as reusable functions / transformers, so you can (1) reproduce it on any fresh dataset, 
(2) reuse it in your live system, and (3) easily try different combinations.
```

```jsx
**Unfortunate truth:

Tell students the uncomfortable truth early: 
Real ML is roughly 80% data cleaning and preparation, 20% modelling. 
The fancy algorithm is the easy part.**
```

#######################################################################

## Encoding

```jsx
Encoding is turning categorical columns into numerical values

### But why?
Models speak numbers, not words. A column like city = ["Tashkent", "Samarkand", ...] 
must become numbers. How you convert it depends on whether the categories have a 
natural order.
```

There are different types of encoding:

OrdinalEncoding, OneHotEncoding, LabelEncoding, TargetEncoding and so on…

### OrdinalEncoder

OrdinalEncoder works with ordinal data. If categories have an order like → (low, medium, high) and we can map into 0, 1, 2.

```jsx
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder(categories=[["low", "medium", "high"]])
X_enc = enc.fit_transform(X[["size"]])   # low->0, medium->1, high->2
```

Another type for ordinal encoding is with “.map()”

```jsx
# Binary: Yes/No -> 1/0
df["smoker"] = df["smoker"].map({"yes": 1, "no": 0})

# Ordinal by hand: set the order yourself
df["size"] = df["size"].map({"low": 0, "medium": 1, "high": 2})
```

### OneHotEncoder

OneHotEncoder works with nominal data. If categories have *no* order (city, color), giving them `0,1,2` is wrong — the model would think "city 2" is somehow between "city 1" and "city 3". Instead, create one binary (0/1) column per category. This is **one-hot encoding**.

```jsx
from sklearn.preprocessing import OneHotEncoder
# handle_unknown='ignore' = don't crash on a category unseen during training
ohe = OneHotEncoder(handle_unknown="ignore")
X_ohe = ohe.fit_transform(X[["city"]])   # returns a sparse matrix

# Quick pandas alternative for exploration:
import pandas as pd
pd.get_dummies(X, columns=["city"])      # one 0/1 column per city
```

### LabelEncoder

```jsx
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 1. Create sample categorical data
cities = ["paris", "tokyo", "amsterdam", "paris", "tokyo"]

# 2. Initialize the LabelEncoder
le = LabelEncoder()

# 3. Fit and transform the data
# fit() learns the unique categories; transform() converts them to integers
encoded_cities = le.fit_transform(cities)

print("Encoded Data:", encoded_cities)
# Output: [1 2 0 1 2] 
# (amsterdam -> 0, paris -> 1, tokyo -> 2 based on alphabetical sorting)

```

```jsx
⚠️ Gotcha: LabelEncoder is for the target y, not features

LabelEncoder looks tempting for encoding a feature column, 
but it assigns arbitrary order (Tashkent=2, Samarkand=1...) — the exact ordinal trap above. 
Use it only on the label y. For feature columns use OneHotEncoder (nominal) or 
OrdinalEncoder (ordinal). (This exact mistake appears in many tutorials.)

⚠️ Gotcha: high-cardinality columns
One-hot encoding a column with thousands of values (user_id, zip code) creates 
thousands of columns and slows everything down. Better: replace with a meaningful 
number (e.g. region → population), use target encoding, or an embedding.

```

#######################################################################

Handling missing values

Most algorithms refuse to train if any value is missing (`NaN`). Géron lists three options:

```jsx
1. Drop the rows - very few rows affected - df.dropna()
2. Drop the column - the column is mostly empty - df.drop(col, axis=1)
3. Fill it in (impute) - Usually the best choice - SimpleImputer
```

When you choose to fill (impute), what you fill with matters. The standard strategies:

| Strategy | Best for | Why |
| --- | --- | --- |
| **Mean** | Numeric, roughly symmetric | The average; pulled around by outliers |
| **Median** | Numeric with outliers | The middle value; robust to extremes — usually the safe default |
| **Mode (most_frequent)** | Categorical | The most common category |
| **Forward / backward fill** | Time series | Carry the previous/next value |

```jsx
from sklearn.impute import SimpleImputer

# median is robust to outliers; use 'most_frequent' for categorical columns
imputer = SimpleImputer(strategy="median")
imputer.fit(X_train)               # learns each column's median FROM TRAIN ONLY
X_train = imputer.transform(X_train)
X_test  = imputer.transform(X_test)   # fills test with the TRAIN medians
```

```jsx
⚠️ Gotcha: compute the fill value on the training set only
If you compute the median over the whole dataset (train + test), the test set's values 
leak into training — data leakage. Always fit the imputer on train, then transform both.
The Pipeline below does this for you automatically.
```

#######################################################################

## Feature Scaling

Imagine two features: `age` (0–100) and `salary` (0–100,000). To a distance-based model, salary completely dominates age simply because its numbers are bigger. **Scaling** puts every feature on a comparable range. The two standard methods:

|  | StandardScaler (standardization) | MinMaxScaler (normalization) |
| --- | --- | --- |
| **Result** | mean = 0, std = 1 | squeezed into [0, 1] |
| **Outliers** | Fairly robust | Sensitive (one outlier squashes the rest) |
| **Bounded range?** | No | Yes |
| **Good default for** | Most models | Neural networks, image pixels |

```jsx
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # learn mean/std from train
X_test  = scaler.transform(X_test)        # apply the same mean/std
```

```jsx
💡 Who needs scaling?
Needs it: KNN, SVM, Logistic/Linear Regression, neural networks 
(anything using distances or gradients). Doesn't need it: tree-based models 
(Decision Tree, Random Forest, XGBoost) — they split on thresholds, so scale doesn't 
matter. Note: usually you do not scale the target y.
```

#######################################################################

## **Pipelines & ColumnTransformer (the leakage-safe way)**

So far every step was “fit train, transform both”. Doing this always by hand is tedious and easy to get wrong. A **PIPELINE** chains the steps into one object that behaves like a single model — and when you call `.fit()`, it correctly fits every step on the training data only.

```jsx
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])
```

But numerical and categorical columns need *different* treatment. **ColumnTransformer** applies the right pipeline to the right columns and glues the results together:

```jsx
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

num_cols = ["age", "salary"]
cat_cols = ["city"]

preprocess = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])
```

Now combine preprocessing and the model into one final pipeline. This is the clean, professional, leakage-proof pattern:

```jsx
from sklearn.ensemble import RandomForestClassifier

model = Pipeline([
    ("prep",  preprocess),
    ("clf",   RandomForestClassifier(random_state=42)),
])

model.fit(X_train, y_train)        # prep is fit on TRAIN only - no leakage
model.score(X_test, y_test)        # test is transformed with train's parameters
```

**💡 Why this matters so much**

Once preprocessing lives inside a Pipeline, cross-validation and hyperparameter tuning (Chapter 14) automatically re-fit the preprocessing on each fold's training portion — so leakage becomes almost impossible. This single habit prevents the most common silent bug in ML.

#######################################################################

## Interview questions

| Question | Strong answer |
| --- | --- |
| **One-Hot vs Label vs Ordinal encoding — when each?** | One-Hot for nominal features (no order); Ordinal/.map() for ordered categories; LabelEncoder only for the target y. |
| **Why is LabelEncoder wrong for a feature?** | It assigns arbitrary integers, inventing a fake order/distance the model will wrongly believe. |
| **Standardization vs Normalization?** | Standardization (StandardScaler): mean 0, std 1, unbounded, robust to outliers. Normalization (MinMaxScaler): squeezed to [0,1], outlier-sensitive. |
| **Which models need feature scaling?** | Distance/gradient based — KNN, SVM, Logistic/Linear Regression, neural nets. Tree-based models don't. |
| **Mean vs median for imputation?** | Median is robust to outliers, so it's the safer default; mean is fine for roughly symmetric data; mode for categoricals. |
| **What is data leakage and how do you prevent it?** | Test/future info entering training (e.g. scaling before the split). Prevent it by fitting preprocessing on train only — easiest via a Pipeline. |
| **fit vs transform vs fit_transform?** | fit learns the parameters (e.g. mean/std); transform applies them; fit_transform does both — use it on train, but only transform on test. |
| **Should you fit the scaler on all data or just train?** | Train only — otherwise test information leaks in. |
| **Problem with one-hot encoding a high-cardinality column?** | It explodes into thousands of columns (slow, sparse). Use target encoding, embeddings, or a meaningful numeric replacement. |
| **What does ColumnTransformer do?** | Applies different transformers to different columns (e.g. scale numerics, one-hot categoricals) and joins the results. |