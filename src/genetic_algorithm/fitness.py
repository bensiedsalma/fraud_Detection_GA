
"""Fitness evaluation for evolved decision trees.

The project specification requires the objective ``Accuracy - Complexity``.
Additional fraud metrics belong in the final model evaluation, not here.
"""

import numpy as np
from sklearn.metrics import accuracy_score

try:
    from .representation import profondeur, nombre_feuilles
except ImportError:  # pragma: no cover - supports notebook imports
    from representation import profondeur, nombre_feuilles


def predire_plus_vite(arbre, X):
    """Predict a 2-D numeric array while preserving row order."""
    values = np.asarray(X)
    if values.ndim != 2:
        raise ValueError("X must be a two-dimensional feature array.")

    predictions = np.empty(len(values), dtype=int)

    def parcourir(node, indices):
        if len(indices) == 0:
            return
        if node.is_leaf():
            predictions[indices] = node.prediction
            return

        mask = values[indices, node.feature] <= node.threshold
        parcourir(node.left, indices[mask])
        parcourir(node.right, indices[~mask])

    parcourir(arbre, np.arange(len(values)))
    return predictions


def calculer_fitness(
    arbre, X_val, y_val,
    max_depth=5, max_leaves=20,
    alpha=0.1
):
    if max_depth <= 0 or max_leaves <= 0:
        raise ValueError("max_depth and max_leaves must be positive.")

    X = np.asarray(X_val)

    y_pred = predire_plus_vite(arbre, X)

    accuracy = accuracy_score(y_val, y_pred)

    depth = profondeur(arbre)
    leaves = nombre_feuilles(arbre)

    complexite = (
        0.5 * (depth / max_depth)
        + 0.5 * (leaves / max_leaves)
    )

    # Required objective: accuracy minus a normalized complexity penalty.
    fitness = accuracy - alpha * complexite

    return fitness
