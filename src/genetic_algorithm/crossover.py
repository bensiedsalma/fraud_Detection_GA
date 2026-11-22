
import random
import copy

try:
    from .repair import repair
except ImportError:  # pragma: no cover - supports notebook imports
    from repair import repair
# Recuperer les noeuds de l'arbre
def obtenir_noeuds(arbre, parent=None, cote=None):

    noeuds = []

    if arbre is None:
        return noeuds

    noeuds.append((arbre, parent, cote))

    if not arbre.is_leaf():
        noeuds += obtenir_noeuds(arbre.left, arbre, "left")
        noeuds += obtenir_noeuds(arbre.right, arbre, "right")

    return noeuds


# Croisement de sous-arbres
def crossover(parent1, parent2, max_depth=5, n_features=None, rng=None):
    """Exchange one subtree from each parent and repair both children."""
    rng = rng or random

    enfant1 = copy.deepcopy(parent1)
    enfant2 = copy.deepcopy(parent2)

    noeuds1 = obtenir_noeuds(enfant1)
    noeuds2 = obtenir_noeuds(enfant2)

    noeud1, pere1, cote1 = rng.choice(noeuds1)
    noeud2, pere2, cote2 = rng.choice(noeuds2)

    if pere1 is None:
        enfant1 = copy.deepcopy(noeud2)
    elif cote1 == "left":
        pere1.left = copy.deepcopy(noeud2)
    else:
        pere1.right = copy.deepcopy(noeud2)

    if pere2 is None:
        enfant2 = copy.deepcopy(noeud1)
    elif cote2 == "left":
        pere2.left = copy.deepcopy(noeud1)
    else:
        pere2.right = copy.deepcopy(noeud1)

    return (
        repair(enfant1, max_depth=max_depth, n_features=n_features),
        repair(enfant2, max_depth=max_depth, n_features=n_features),
    )
