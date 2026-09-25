# Medical Insurance Cost Prediction

## Project Overview
This project predicts individual annual medical charges using patient demographic and behavioral features. It implements polynomial feature expansion, feature scaling, and regularized linear models ($L_1$, $L_2$, and ElasticNet) to evaluate model performance and diagnose the bias-variance tradeoff.

---

## Dataset Analysis
* **Sample Count:** 1,338 raw records (1 duplicate identified and removed, resulting in 1,337 unique samples).
* **Target Variable ($y$):** `charges` (Continuous numerical values $\rightarrow$ Regression task).
* **Predictor Features ($X$):** `age`, `sex`, `bmi`, `children`, `smoker`, and `region`.
* **Class Distribution Insights:**
  * **Gender (`sex`):** Balanced distribution across samples (675 male, 662 female).
  * **Smoking Status (`smoker`):** Imbalanced distribution, with non-smokers outnumbering smokers by roughly 4:1 (1,063 non-smokers vs. 274 smokers).

---

## Data Preprocessing & Encoding
1. **Binary Encoding:**
   * `sex`: Mapped `male` to $1$ and `female` to $0$.
   * `smoker`: Mapped `yes` to $1$ and `no` to $0$.
2. **One-Hot Encoding:**
   * `region`: Applied `pd.get_dummies(..., drop_first=True)` to convert geographical regions into dummy variables while avoiding the dummy variable trap.
3. **Data Splitting:**
   * Partitioned the dataset into an **80% Training set** and a **20% Test set** (`random_state=42`).
4. **Polynomial Features & Scaling:**
   * Generated degree-2 polynomial features (`PolynomialFeatures(degree=2, include_bias=False)`) to capture cross-feature interaction terms (e.g., $bmi \times smoker$).
   * Standardized features using `StandardScaler()` following polynomial expansion so regularization penalties apply uniformly.

---

## Model Evaluation & Comparison

All models were evaluated on the 20% test split using $R^2$ score, Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).

### Initial Model Comparison

| Model | $R^2$ Score | RMSE | MAE |
| :--- | :---: | :---: | :---: |
| **Linear Regression (Baseline)** | **0.8825** | **4,646.06** | **2,867.32** |
| **Ridge Regression** ($\alpha=10.0$) | 0.8800 | 4,696.26 | 2,933.53 |
| **Lasso Regression** ($\alpha=100.0$) | 0.8782 | 4,731.73 | 2,948.97 |
| **ElasticNet** ($\alpha=1.0, \text{l1\_ratio}=0.5$) | 0.8084 | 5,933.41 | 4,121.96 |

### Hyperparameter Tuning Results
Adjusting the regularization strength ($\alpha$) revealed that over-regularization initially constrained model accuracy:

* **Tuned Ridge ($\alpha=0.1$):** $R^2 = 0.8826$
* **Tuned Lasso ($\alpha=1.0$):** $R^2 = 0.8828$

---

## Key Takeaways
1. **Interaction Effects:** Degree-2 polynomial expansion captures critical non-linear interactions (specifically `bmi` combined with `smoker`), driving overall predictive variance ($R^2$) from standard baselines up to $\approx 88.28\%$.
2. **Regularization Behavior:** Because sample size ($N = 1,069$) significantly exceeds feature count ($p = 45$), standard OLS performs strongly without severe overfitting. High initial penalty parameters ($\alpha=100.0$) underfit the data, whereas light $L_1$ regularization ($\alpha=1.0$) eliminates weak interaction terms and achieves the highest test $R^2$ score.
