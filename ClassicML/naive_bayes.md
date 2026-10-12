# Naive Bayes

```python
The Naive Bayes algorithm is a supervised learning algorithm, that works for 
classification problems based on Bayes' Theorem.
```

Naive Bayes do not use gradient descent to find best weights. Instead, it is a **counting and lookup algorithm**. It calculates its "weights" (which are actually just probabilities) in a single pass over the training data using simple arithmetic.

### Bayes Theorem

<img width="255" height="74" alt="image" src="https://github.com/user-attachments/assets/cf958e71-6d9b-4ae0-9737-c878b4cd0703" />


```python
Bayes' theorem flips a conditional probability around. For classification, it answers: 
"given these features, how probable is each class?”

P(class | data) = [ P(data | class) × P(class) ] / P(data)
```

| **Term** | **Name** | **Meaning** |
| --- | --- | --- |
| P(class | data) | **Posterior** | What we want: probability of the class given the features |
| P(data | class) | **Likelihood** | How likely these features are *if* the class were true |
| P(class) | **Prior** | How common the class is overall (before seeing features) |
| P(data) | **Evidence** | How common the features are (a normalizing constant) |

To classify, we compute the posterior for each class and pick the highest — this is the **MAP** (maximum a posteriori) decision. Since `P(data)` is the same for every class, we can drop it and just compare `P(data | class) × P(class)`.

## The “naive” assumption

Computing `P(data | class)` for all features together is hard. So Naive Bayes makes one bold shortcut: it assumes **all features are independent given the class**. That lets it multiply each feature's probability separately:

```python
P(x₁, x₂, ..., xₙ | class) ≈ P(x₁|class) × P(x₂|class) × ... × P(xₙ|class)

ℹ️ Why "naive"?
The independence assumption is almost always false in real data (in an email, the word 
"free" and the word "money" are clearly related). It's a naive belief — hence the name. 
The shocking part: the algorithm still works very well in practice, because for 
classification we only need the right class to win, not perfectly accurate probabilities.
```

## How it classifies

**Training is just counting** — no optimization. It computes two things from the data:

- **Class priors** — e.g. P(go-out) = (# go-out days) / (total days).
- **Conditional probabilities** — e.g. P(weather=sunny | go-out) = (# sunny go-out days) / (# go-out days).

**Example** (Brownlee's): predict go-out vs stay-home for a *sunny* day with a *working* car. For each class, multiply:

```python
# Score per class (then pick the max)
go-out   = P(sunny|go-out) × P(working|go-out) × P(go-out)
stay-home = P(sunny|stay) × P(working|stay) × P(stay)
```

Whichever score is larger is the prediction; normalize the two to turn them into probabilities. That's the whole algorithm — counts multiplied together.

<img width="892" height="531" alt="image" src="https://github.com/user-attachments/assets/77fe003f-1ff8-4091-bc39-c85453e5bfad" />
