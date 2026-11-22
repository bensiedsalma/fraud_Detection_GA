
"""Repair invalid trees produced by genetic operators."""

try:  # Supports both package imports and the existing notebook imports.
    from .representation import Node
except ImportError:  # pragma: no cover - exercised by notebook execution
    from representation import Node


def repair(arbre, max_depth=5, n_features=None, default_prediction=0):
    """Return a valid binary tree whose depth does not exceed ``max_depth``.

    The operator turns malformed or over-deep subtrees into binary-class leaves,
    removes child pointers from leaves, and ensures that internal nodes retain
    both children.  It modifies and returns ``arbre`` for efficiency.
    """
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative.")

    def as_leaf(node):
        if node is None:
            node = Node()
        prediction = node.prediction
        if prediction not in (0, 1):
            prediction = default_prediction
        node.feature = None
        node.threshold = None
        node.left = None
        node.right = None
        node.prediction = int(prediction)
        return node

    def corriger(node, niveau):
        if node is None:
            return Node(prediction=default_prediction)
        if node.is_leaf() or niveau >= max_depth:
            return as_leaf(node)
        if node.feature is None or node.threshold is None:
            return as_leaf(node)
        if not isinstance(node.feature, int) or node.feature < 0:
            return as_leaf(node)
        if n_features is not None and node.feature >= n_features:
            return as_leaf(node)

        node.prediction = None
        node.left = corriger(node.left, niveau + 1)
        node.right = corriger(node.right, niveau + 1)
        return node

    return corriger(arbre, 0)
