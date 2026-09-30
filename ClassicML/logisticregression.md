# What is Logistic Regression?

Logistic Regression answers yes/no questions by predicting the probability that something belongs to class , then snapping that probability into 0 and 1.

| **Real question** | **Class 1 / Class 0** |
| --- | --- |
| Is this email spam? | 1 = spam, 0 = not spam |
| Does this patient have the disease? | 1 = disease, 0 = healthy |
| Will this customer churn? | 1 = leaves, 0 = stays |

Why Linear Regression fails at predicting classification problem?

!image.png

Logistic Regression borrows Linear Regression’s math and uses SIGMOID FUNCTION that squashes continuous number into (0, 1) interval.

## Sigmoid Function

Sigmoid formula:

```jsx
σ(z) = 1 / (1 + e⁻ᶻ)

where:
z = X @ W + bias
```

It's an S-shaped curve. Big positive input → near 1; big negative → near 0; exactly 0 → exactly 0.5:

| **z** | **−5** | **−2** | **0** | **2** | **5** |
| --- | --- | --- | --- | --- | --- |
| σ(z) | 0.007 | 0.12 | **0.5** | 0.88 | 0.993 |

!image.png

That output is read as a **probability**. `σ(z) = 0.88` means "88% likely to be class 1." This is the source of the "0 and 1" you asked about: it's not linear regression — it's the sigmoid producing a probability between 0 and 1.

The pipeline that goes from Linear Regression into Logistic Regression:

```jsx
1.  z = b + w₁x₁ + ... + wₙxₙ   ---   Linear Regression part
2. p = σ(z) = 1/(1+e⁻ᶻ)    ---   Sigmoid funtion(squash into a probability 0..1)
3) class = 1 if p ≥ 0.5 else 0  ---  last step for prediction
```