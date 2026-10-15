# Decision Tree

```python
CART - Classification and Regression Trees. A Decision Tree is the most human readable
model in ML - a flowchart with yes/no questions.

## Anology
It's the game "20 Questions": "Is height > 180cm? → no. Is weight > 80kg? → no → predict
Female." Each question narrows things down until the answer is obvious. A doctor's 
diagnosis flowchart or a loan-approval checklist is literally a decision tree.
```

```
If Height > 180 cm            -> Male
Else if Weight > 80 kg       -> Male
Else                         -> Female
```

<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/80e77b8b-7400-4133-9548-9385e86ccdef" />

```python
Root node --->  the very first decision question asked
Node ---> Question (f.e: age>20)
Leaf node ---> final result 
Branch ---> the answer route yes/no
Depth ---> the number of questions asked
```

<img width="640" height="482" alt="image" src="https://github.com/user-attachments/assets/2f752c8f-e7e0-4e18-a9de-7842f6710399" />


Here:

```python
branches - yes/no
root node - age>20?
depth - 2
leaf nodes = low risk | high risk | underage 
internal nodes = income>$30k?
```

## How it is going to split(Greedy Recursive Splitting)

Training builds the tree by **recursive binary splitting**: at each node it tries every feature and every possible split point, scores each by how "pure" the resulting groups are, and greedily picks the best one. Then it repeats on each child — top-down, one best split at a time.

"Pure" means each group contains mostly one class. The split that makes the children purest wins.

## Gini Impurity

```python
What is Gini Impurity? Gini is the impurity of features(mix of features)
```

<img width="304" height="152" alt="image" src="https://github.com/user-attachments/assets/f9ffc4be-6e44-4b63-ac55-b9f45ab8dba6" />


p(i) - i class proportion

```python
pure node - 50 apples, 0 pears ----> gini=0 perfect, no impurity
mixed node - 25 apples, 25 pears ----> gini=0.5 worst case
average node - 40 apples, 10 pears ----> gini = 0.32 little mixed
```

Manual calculation:

```python
total_fruites = 50
apples = 40
pears = 10

p_apples = 40/50 = 0.8
p_pears = 10/50 = 0.2

gini = 1 - (0.8^2+0.2^2) = 1-0.64-0.04 = 0.32
```

!!! Anology

The less gini impurity, the more accuracy
