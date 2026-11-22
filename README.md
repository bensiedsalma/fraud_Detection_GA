# Fraud Detection — Genetic Algorithm Decision Trees

This academic project evolves limited-depth binary decision trees with a
genetic algorithm (GA) and compares them with CART, Random Forest, and
Gradient Boosting on credit-card fraud detection.

## Teacher specification implemented

- Tree-based binary encoding: `src/genetic_algorithm/representation.py`
- Subtree crossover: `src/genetic_algorithm/crossover.py`
- Node mutation: `src/genetic_algorithm/mutation.py`
- Fitness: **Accuracy − Complexity**: `src/genetic_algorithm/fitness.py`
- Repair operators and maximum-depth enforcement:
  `src/genetic_algorithm/repair.py`
- Predictive and interpretability comparison in `notebooks/evaluation.ipynb`

The independent `X_test`/`y_test` set is for final evaluation only. The GA
receives a split of `X_train`/`y_train` as its training and validation data.

## Reproducible run order

1. Create an environment with `pip install -r requirements.txt`.
2. Put `creditcard.csv` in `dataset/`.
3. Run `notebooks/analyse_dataset.ipynb`.
4. Run `notebooks/preprocessing.ipynb` to create `dataset/processed/`.
5. Run `notebooks/evaluation.ipynb` from top to bottom.
6. Run component tests with:

   ```powershell
   C:\Users\po\miniconda3\python.exe -B -m unittest discover -s tests -v
   ```

## Interpreting results

The GA is intentionally optimized for accuracy minus a normalized complexity
penalty, as required. Precision, recall, F1-score, and confusion matrices are
reported as additional fraud-detection diagnostics. If the GA is less accurate
than ensemble models but has fewer leaves or lower depth, that performance /
interpretability trade-off must be reported rather than hidden.

Existing files under `results/` are historical outputs. Rerun the notebooks to
produce a fresh, internally consistent result set after source changes.
