"""Genetic algorithm for constructing limited-depth binary decision trees."""

import copy
import random

import numpy as np

try:
    from .representation import Node, profondeur
    from .fitness import calculer_fitness
    from .selection import selection_tournoi, selection_roulette
    from .crossover import crossover
    from .mutation import mutation
    from .repair import repair
except ImportError:  # pragma: no cover - supports the existing notebook imports
    from representation import Node, profondeur
    from fitness import calculer_fitness
    from selection import selection_tournoi, selection_roulette
    from crossover import crossover
    from mutation import mutation
    from repair import repair


def creer_arbre_aleatoire(X_train, y_train, max_depth, niveau=0, indices=None,
                          rng=None):
    """Create a valid random binary tree from observed training values."""
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative.")
    rng = rng or random
    X = np.asarray(X_train)
    y = np.asarray(y_train).reshape(-1)
    if X.ndim != 2 or len(X) != len(y) or len(X) == 0:
        raise ValueError("X_train and y_train must be non-empty aligned arrays.")

    indices = np.arange(len(X)) if indices is None else np.asarray(indices, dtype=int)
    classes, counts = np.unique(y[indices], return_counts=True)
    classe_majoritaire = int(classes[np.argmax(counts)])

    if niveau >= max_depth or (niveau > 0 and rng.random() < 0.3):
        return Node(prediction=classe_majoritaire)

    feature = rng.randrange(X.shape[1])
    valeurs = X[indices, feature]
    valeurs = valeurs[~np.isnan(valeurs)]
    if len(valeurs) == 0:
        return Node(prediction=classe_majoritaire)

    threshold = float(rng.choice(valeurs.tolist()))
    left_indices = indices[X[indices, feature] <= threshold]
    right_indices = indices[X[indices, feature] > threshold]
    if len(left_indices) == 0 or len(right_indices) == 0:
        return Node(prediction=classe_majoritaire)

    return Node(
        feature=feature,
        threshold=threshold,
        left=creer_arbre_aleatoire(X, y, max_depth, niveau + 1, left_indices, rng),
        right=creer_arbre_aleatoire(X, y, max_depth, niveau + 1, right_indices, rng),
    )


def genetic_algorithm(
    X_train, y_train, X_val, y_val, population_size=20, generations=10,
    max_depth=5, taux_mutation=0.2, taux_croisement=0.8,
    methode_selection="tournoi", random_state=42,
):
    """Evolve binary trees using the required GA operators.

    The fitness remains ``Accuracy - Complexity``.  Only validation data are
    used by fitness; the independent test set belongs exclusively to final
    evaluation outside this function.
    """
    if population_size < 2:
        raise ValueError("population_size must be at least 2.")
    if generations < 1:
        raise ValueError("generations must be at least 1.")
    if not 0 <= taux_mutation <= 1 or not 0 <= taux_croisement <= 1:
        raise ValueError("Mutation and crossover rates must be between 0 and 1.")
    if methode_selection not in {"tournoi", "roulette"}:
        raise ValueError("Methode de selection inconnue")

    rng = random.Random(random_state)
    X_train_array = np.asarray(X_train, dtype=float)
    y_train_array = np.asarray(y_train).reshape(-1)
    X_val_array = np.asarray(X_val, dtype=float)
    y_val_array = np.asarray(y_val).reshape(-1)
    if X_train_array.ndim != 2 or X_val_array.ndim != 2:
        raise ValueError("Training and validation features must be two-dimensional.")
    if len(X_train_array) != len(y_train_array) or len(X_val_array) != len(y_val_array):
        raise ValueError("Feature and target lengths must match.")
    if X_train_array.shape[1] != X_val_array.shape[1]:
        raise ValueError("Training and validation sets must have the same features.")

    n_features = X_train_array.shape[1]
    feature_values = [
        X_train_array[~np.isnan(X_train_array[:, feature]), feature].tolist()
        for feature in range(n_features)
    ]
    population = [
        repair(
            creer_arbre_aleatoire(X_train_array, y_train_array, max_depth, rng=rng),
            max_depth=max_depth, n_features=n_features,
        )
        for _ in range(population_size)
    ]
    historique = []

    for generation in range(generations):
        fitness_values = [
            calculer_fitness(
                arbre, X_val_array, y_val_array,
                max_depth=max_depth, max_leaves=2 ** max_depth,
            )
            for arbre in population
        ]
        meilleur_index = int(np.argmax(fitness_values))
        meilleur_arbre = copy.deepcopy(population[meilleur_index])
        meilleure_fitness = fitness_values[meilleur_index]
        historique.append(meilleure_fitness)

        print("Generation", generation + 1,
              "| Meilleure fitness :", round(meilleure_fitness, 5),
              "| Profondeur :", profondeur(meilleur_arbre))

        nouvelle_population = [meilleur_arbre]  # elitism
        while len(nouvelle_population) < population_size:
            selector = selection_tournoi if methode_selection == "tournoi" else selection_roulette
            parent1 = selector(population, fitness_values, rng=rng)
            parent2 = selector(population, fitness_values, rng=rng)

            if rng.random() < taux_croisement:
                enfant1, enfant2 = crossover(
                    parent1, parent2, max_depth=max_depth,
                    n_features=n_features, rng=rng,
                )
            else:
                enfant1, enfant2 = copy.deepcopy(parent1), copy.deepcopy(parent2)

            if rng.random() < taux_mutation:
                enfant1 = mutation(enfant1, n_features, proba=1.0,
                                   feature_values=feature_values, rng=rng)
            if rng.random() < taux_mutation:
                enfant2 = mutation(enfant2, n_features, proba=1.0,
                                   feature_values=feature_values, rng=rng)

            nouvelle_population.append(
                repair(enfant1, max_depth=max_depth, n_features=n_features)
            )
            if len(nouvelle_population) < population_size:
                nouvelle_population.append(
                    repair(enfant2, max_depth=max_depth, n_features=n_features)
                )
        population = nouvelle_population

    fitness_finale = [
        calculer_fitness(arbre, X_val_array, y_val_array,
                          max_depth=max_depth, max_leaves=2 ** max_depth)
        for arbre in population
    ]
    return population[int(np.argmax(fitness_finale))], historique
