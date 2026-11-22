"""Parent-selection operators for the genetic algorithm."""

import random


def _validate(population, fitness_values):
    if not population:
        raise ValueError("The population cannot be empty.")
    if len(population) != len(fitness_values):
        raise ValueError("population and fitness_values must have the same length.")


def selection_tournoi(population, fitness_values, taille_tournoi=3, rng=None):
    """Select one parent with tournament selection."""
    _validate(population, fitness_values)
    rng = rng or random
    if taille_tournoi <= 0:
        raise ValueError("taille_tournoi must be positive.")
    # Choisir des indices aleatoires
    indices = rng.sample(
        range(len(population)),
        min(taille_tournoi, len(population))
    )
    # Trouver le meilleur individu
    meilleur = max(
        indices,
        key=lambda i: fitness_values[i]
    )
    return population[meilleur]
    
# SELECTION PAR ROULETTE

def selection_roulette(population, fitness_values, rng=None):
    """Select one parent with shifted non-negative roulette weights."""
    _validate(population, fitness_values)
    rng = rng or random
    # Transformer les fitness pour avoir des valeurs positives
    minimum = min(fitness_values)
    poids = [
        f - minimum + 0.0001
        for f in fitness_values
    ]
    # Choisir un individu selon sa probabilite
    individu = rng.choices(
        population,
        weights=poids,
        k=1
    )[0]

    return individu
