
"""Node-mutation operator for binary trees."""

import copy
import random


def mutation(arbre, n_features, proba=0.8, feature_values=None, rng=None):
    """Mutate one node and return a deep-copied child.

    If feature values are supplied, thresholds are sampled from values actually
    present in the corresponding feature.  This keeps mutations meaningful for
    all features, including scaled ``Time`` and ``Amount``.
    """
    if n_features <= 0:
        raise ValueError("n_features must be positive.")
    rng = rng or random

    # Copier l'arbre original
    nouvel_arbre = copy.deepcopy(arbre)

    # Recuperer tous les noeuds
    noeuds = []
    def parcourir(node):
        if node is None:
            return
        noeuds.append(node)
        if not node.is_leaf():
            parcourir(node.left)
            parcourir(node.right)
    parcourir(nouvel_arbre)
    # Choisir un noeud aleatoire
    node = rng.choice(noeuds)
    # Mutation du noeud
    if rng.random() < proba:
        if node.is_leaf():
            # Changer la prediction
            node.prediction = 1 - node.prediction
        else:
            # Choisir entre feature et threshold
            if rng.choice([True, False]):
                node.feature = rng.randrange(n_features)
                if feature_values is not None and len(feature_values[node.feature]):
                    node.threshold = rng.choice(feature_values[node.feature])
            else:
                if feature_values is not None and len(feature_values[node.feature]):
                    node.threshold = rng.choice(feature_values[node.feature])
                else:
                    node.threshold += rng.uniform(-1.0, 1.0)
    return nouvel_arbre
