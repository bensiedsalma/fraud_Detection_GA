# Final Report — Genetic Algorithm for Decision Tree Construction

## Objective

This project evolves limited-depth binary decision trees with a genetic
algorithm (GA) for credit-card fraud detection. It compares the evolved tree
with CART, Random Forest, and Gradient Boosting, considering both predictive
performance and interpretability.

## Methodology

The original dataset contains 284,807 transactions and 31 columns. After
removing 1,081 duplicate rows, the data are split stratifiably into training
and independent test sets. `Time` and `Amount` are standardized from the
training data only.

The GA uses:

- tree-based encoding through internal split nodes and binary prediction leaves;
- subtree crossover;
- node mutation;
- a repair operator that restores binary structure and maximum depth;
- the required fitness function: **Accuracy − Complexity**, where complexity
  combines normalized depth and number of leaves.

The GA receives a training/validation split drawn only from the training set.
The independent test set is reserved for one final evaluation of all four
models.

## Fresh experiment results

The following values come from the fresh end-to-end execution and are read
directly from `results/tables/comparaison_modeles.csv`. They have not been
modified or selectively removed.

| Model | Accuracy | Precision | Recall | F1-score | Training time |
|---|---:|---:|---:|---:|---:|
| CART | 0.999013 | 0.734940 | 0.642105 | 0.685393 | 19.175862 s |
| Random Forest | 0.999471 | 0.971014 | 0.705263 | 0.817073 | 104.386446 s |
| Gradient Boosting | 0.999207 | 0.867647 | 0.621053 | 0.723926 | 171.560699 s |
| GA | 0.973743 | 0.047372 | 0.768421 | 0.089242 | 4.699825 s |

The fresh GA tree has depth 5 and 8 leaves, compared with CART at depth 33 and
222 leaves. It is therefore substantially simpler and easier to explain. Its
recall is higher than the other models (0.768421), but its precision (0.047372)
and F1-score (0.089242) are much lower. In this imbalanced fraud setting, that
means the GA identifies many frauds but also produces many false positives.
This performance-versus-simplicity trade-off must be reported honestly; the
tree's interpretability alone does not make it the strongest predictive model.

## Implementation verification

The source includes regression tests for each required GA operator and for the
comparison path that previously raised `KeyError: 20`. The test command is:

```powershell
C:\Users\po\miniconda3\python.exe -B -m unittest discover -s tests -v
```

The full experiment was executed after the source repair. The generated tables,
figures, and serialized GA tree in `results/` are the current deliverables.

## Conclusion

The project demonstrates the requested GA operators and makes the required
comparison with CART and ensemble baselines. The fresh run shows a clear
performance-versus-simplicity trade-off: the GA tree is much smaller than CART
and has the highest recall, whereas Random Forest has the strongest F1-score
(0.817073). Further parameter studies may improve GA performance, but they
must retain the required Accuracy − Complexity objective and must never use the
independent test set in the evolution process.
