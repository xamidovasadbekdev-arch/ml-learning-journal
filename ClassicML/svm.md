# SVM(Support Vector Machines)

## **What is an SVM? (the widest street)**

Like logistic regression, an SVM separates two classes with a straight boundary (a **hyperplane**). But it picks a *special* one: the boundary that sits in the **middle of the widest possible gap** between the classes.

**🧩 Analogy (the widest street)**

Imagine paving the widest possible street between the two groups of points without any point landing on the road. The SVM's decision boundary is the centre line of that street. A wider street = more room for error = better generalization. (This is why SVM is also called a "large margin classifier.")

## Margin & support vectors

Two definitions that are the heart of SVM:

- **Margin** — the width of that street: the distance from the boundary to the nearest points of each class. SVM maximizes it.
- **Support vectors** — the few points sitting right on the edge of the street. **They alone define the boundary** — move or delete any other point and nothing changes; move a support vector and the boundary shifts.

The hyperplane (2 features)

```python
b + w₁x₁ + w₂x₂ = 0     (above the line → one class, below → the other)

ℹ️ Why "support vectors" matter
Because only the support vectors define the model, SVM is memory-efficient 
(it can forget all other points) and robust to far-away points. That's a common 
interview point: "what are support vectors and why do they matter?"
```

!image.png

Of all lines that separate the classes, SVM chooses the one with the **widest street** (margin) between them. Only the **support vectors** — the points touching the street's edges (ringed) — define that boundary; every other point could move freely and the line wouldn't budge.