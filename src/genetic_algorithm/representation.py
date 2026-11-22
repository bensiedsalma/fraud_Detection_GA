"""Tree-based representation used by the genetic algorithm.

Every internal node has exactly two children.  Every leaf stores a binary class
prediction (0 or 1).  The repair operator is responsible for restoring these
invariants after crossover or mutation.
"""

import numpy as np


class Node:
    """A node in a binary decision tree."""

    def __init__(self, feature=None, threshold=None, left=None, right=None,
                 prediction=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.prediction = prediction

    def is_leaf(self):
        """Return ``True`` when this node represents a class prediction."""
        return self.prediction is not None

    def predict(self, row):
        """Predict the class of one 1-D observation.

        A complete DataFrame is deliberately rejected: integer feature indices
        are positional, whereas ``DataFrame[20]`` means a column label and was
        the source of the recorded ``KeyError: 20``.  Batch callers must pass
        rows (or use the batch helper in :mod:`fitness`).
        """
        values = np.asarray(row)
        if values.ndim != 1:
            raise ValueError(
                "Node.predict expects one 1-D observation; pass DataFrame rows "
                "or a NumPy row, not a complete table."
            )

        current = self
        while not current.is_leaf():
            if current.feature is None or current.threshold is None:
                raise ValueError("Invalid internal node: missing feature or threshold.")
            if current.left is None or current.right is None:
                raise ValueError("Invalid internal node: both children are required.")
            if values[current.feature] <= current.threshold:
                current = current.left
            else:
                current = current.right
        return int(current.prediction)

# CALCULER LA PROFONDEUR

def profondeur(node):
    """Return the number of splits on the longest root-to-leaf path."""
    if node is None or node.is_leaf():
        return 0
    return 1 + max(
        profondeur(node.left),
        profondeur(node.right)
    )

# COMPTER LE NOMBRE DE FEUILLES

def nombre_feuilles(node):
    """Return the number of leaves in ``node``."""
    if node is None:
        return 0
    if node.is_leaf():
        return 1

    return (
        nombre_feuilles(node.left)
        + nombre_feuilles(node.right)
    )
