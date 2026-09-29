# LogisticRegression

Logistic Regression answers **yes/no questions** by predicting the *probability* that something belongs to a class, then snapping that probability to a 0 or 1.

| **Real question** | **Class 1 / Class 0** |
| --- | --- |
| Is this email spam? | 1 = spam, 0 = not spam |
| Does this patient have the disease? | 1 = disease, 0 = healthy |
| Will this customer churn? | 1 = leaves, 0 = stays |

"Logistic *Regression*" is a **classification** algorithm. The word "regression" is historical (it borrows linear regression's math). If an interviewer asks "is logistic regression classification or regression?" — the answer is **classification**.

<img width="883" height="584" alt="image" src="https://github.com/user-attachments/assets/6fa09ee4-4c30-48ca-a070-938784d5a8c2" />


```jsx
# Why linear regression does not work for classification?
```

```
Linear regression does not work well for classification because it predicts unbounded
continuous numbers (-∞ to +∞) instead of probabilities or discrete class labels

Why It Fails
Outputs outside 0 and 1: Classification probabilities must stay between 0 and 1.
Linear regression lines can easily output values like -0.5 or 1.8, which do not make
sense as probabilities

## Sigmoid Function
And there is a sigmoid function that squezes numbers into 0 and 1 so that probability will not
go out from (0, 1)

Formula:
σ(z) = 1 / (1 + e⁻ᶻ)
```

It's an S-shaped curve. Big positive input → near 1; big negative → near 0; exactly 0 → exactly 0.5:

| **z** | **−5** | **−2** | **0** | **2** | **5** |
| --- | --- | --- | --- | --- | --- |
| σ(z) | 0.007 | 0.12 | **0.5** | 0.88 | 0.993 |

<img width="886" height="601" alt="image" src="https://github.com/user-attachments/assets/1cf70661-8410-465e-8da5-93f82b7b2f37" />

That output is read as a **probability**. `σ(z) = 0.88` means "88% likely to be class 1." This is the source of the "0 and 1" you asked about: it's not linear regression — it's the sigmoid producing a probability between 0 and 1.
