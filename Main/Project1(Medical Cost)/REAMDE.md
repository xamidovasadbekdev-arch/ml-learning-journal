# 🩺 Medical Cost Prediction — Exploratory Data Analysis & Modeling Strategy

## 📌 Project Overview
This project predicts individual annual medical charges using personal demographic and behavioral attributes from the Medical Cost dataset. The core objective is to analyze key risk factors driving medical expenses and demonstrate why linear models fail without feature interactions.

---

## 📊 Dataset Overview
* **Observations:** 1,337 clean samples (1 duplicate entry identified and removed).
* **Missing Values:** 0 missing or null entries across all fields.
* **Target Variable:** `charges` (Continuous numerical, annual USD).
* **Feature Set (6 Predictors):**
  * Numerical: `age`, `bmi`, `children`
  * Categorical: `sex`, `smoker`, `region`

---

## 📈 Target Distribution & Characteristics
* **Right-Skewed Distribution:** The majority of charges lie between **$0 and $15,000**.
* **Central Tendency:**
  * **Mean:** $13,279
  * **Median:** $9,386
* **Multimodal Clustering:** Charges split into three distinct charge bands:
  1. **Bottom Band ($0–$15k):** General population and non-smokers.
  2. **Middle Band ($15k–$32k):** Non-obese smokers and high-cost non-smokers.
  3. **Top Band ($30k–$50k):** Obese smokers ($\text{BMI} \ge 30$).

---

## 🔍 Key Exploratory Findings

### 1. Age (The Universal Baseline Slope)
* Charges increase steadily with age across all cost tiers due to increased health risks.
* Age forms **three parallel lines** stacked on top of each other.
* **Takeaway:** Age explains the **slope within a band**, but does *not* determine which charge band a patient belongs to.

### 2. Smoking (The Primary Stratifier)
* Non-smokers overwhelmingly cluster in the lowest tier ($0–$15k).
* Smoking acts as the single primary decision point separating lower-cost patients from higher-cost tiers.

### 3. BMI & The Step Threshold ($\text{BMI} = 30$)
* For non-smokers, an increased BMI slightly increases costs.
* For smokers, there is a **sharp step-function jump at $\text{BMI} = 30$**:
  * $\text{BMI} < 30$: Charges cluster between $15k and $30k.
  * $\text{BMI} \ge 30$: Charges instantly jump to $35k–$50k.

### 4. Non-Predictive Features
* `sex` and `region` exhibit mixed distributions across all bands and offer negligible explanatory power.
* `children` does not split higher-cost smoking bands.

---

## 🧮 Interaction Discovery: Smoking $\times$ Obesity

Group median analysis reveals a strong synergistic interaction between smoking and obesity ($\text{BMI} \ge 30$):

| Smoker | Obese ($\text{BMI} \ge 30$) | Median Medical Charges |
| :--- | :--- | :--- |
| **No** | **No** ($0$) | **$6,762** *(Base Group)* |
| **No** | **Yes** ($1$) | **$8,084** |
| **Yes** | **No** ($0$) | **$20,167** |
| **Yes** | **Yes** ($1$) | **$40,904** |

### Individual Feature Effects:
* **Obesity effect for non-smokers:** $\$8,084 - \$6,762 = \mathbf{+\$1,322}$
* **Smoking effect for non-obese individuals:** $\$20,167 - \$6,762 = \mathbf{+\$13,405}$
* **Obesity effect for smokers:** $\$40,904 - \$20,167 = \mathbf{+\$20,737}$

---

## 💡 The Linear Model Failure (The Interaction Gap)

A purely additive linear regression model assumes independent feature contributions:

$$\text{Predicted Charge} = \text{Base} + \text{Smoking Effect} + \text{Obesity Effect}$$

$$\text{Additive Prediction} = \$6,762 + \$13,405 + \$1,322 = \mathbf{\$21,489}$$

### The Gap:
$$\text{Actual Obese Smoker Median} - \text{Additive Guess} = \$40,904 - \$21,489 = \mathbf{+\$19,415}$$

### Conclusion:
* Being obese increases costs by **$1,322 for non-smokers**, but by **$20,737 for smokers** ($\approx 16\times$ higher effect).
* An additive model underpredicts every obese smoker by approximately **$19,400**.
* **Engineering Solution:** Including an explicit interaction term ($\text{smoker} \times \text{obese}$) or creating polynomial features ($\text{degree}=2$) is mandatory to resolve this systematic error.

---

## 🛠 Target Model Formulation

$$\text{charges} = \theta_0 + \theta_1(\text{age}) + \theta_2(\text{sex}) + \theta_3(\text{bmi}) + \theta_4(\text{children}) + \theta_5(\text{smoker}) + \sum_{j=1}^3 \theta_{5+j}(\text{region}_j) + \theta_9(\text{smoker} \times \text{obese}) + \epsilon$$
