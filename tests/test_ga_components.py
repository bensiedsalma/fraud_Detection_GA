"""Regression tests for the GA operators required by the project brief."""

import random
import unittest

import numpy as np
import pandas as pd

from src.genetic_algorithm.crossover import crossover
from src.genetic_algorithm.fitness import calculer_fitness
from src.genetic_algorithm.genetic_algorithm import (
    creer_arbre_aleatoire,
    genetic_algorithm,
)
from src.genetic_algorithm.mutation import mutation
from src.genetic_algorithm.repair import repair
from src.genetic_algorithm.representation import Node, nombre_feuilles, profondeur
from src.models_comparison.comparison import evaluer_modele


def assert_valid_binary_tree(test_case, node, max_depth, depth=0):
    test_case.assertIsNotNone(node)
    test_case.assertLessEqual(depth, max_depth)
    if node.is_leaf():
        test_case.assertIn(node.prediction, (0, 1))
        test_case.assertIsNone(node.left)
        test_case.assertIsNone(node.right)
        return
    test_case.assertIsNone(node.prediction)
    test_case.assertIsInstance(node.feature, int)
    test_case.assertIsNotNone(node.threshold)
    assert_valid_binary_tree(test_case, node.left, max_depth, depth + 1)
    assert_valid_binary_tree(test_case, node.right, max_depth, depth + 1)


class GeneticAlgorithmComponentTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([
            [-2.0, 0.0], [-1.0, 1.0], [-0.5, 0.2],
            [0.5, 0.1], [1.0, 1.1], [2.0, 0.4],
        ])
        self.y = np.array([0, 0, 0, 1, 1, 1])
        self.parent_a = Node(0, 0.0, Node(prediction=0), Node(prediction=1))
        self.parent_b = Node(1, 0.5, Node(prediction=0), Node(prediction=1))

    def test_random_encoding_is_limited_and_binary(self):
        tree = creer_arbre_aleatoire(self.X, self.y, max_depth=3,
                                     rng=random.Random(12))
        assert_valid_binary_tree(self, tree, max_depth=3)

    def test_subtree_crossover_and_node_mutation_remain_valid(self):
        child_a, child_b = crossover(
            self.parent_a, self.parent_b, max_depth=2, n_features=2,
            rng=random.Random(2),
        )
        values = [self.X[:, i].tolist() for i in range(self.X.shape[1])]
        mutated = mutation(child_a, 2, proba=1.0,
                           feature_values=values, rng=random.Random(3))
        assert_valid_binary_tree(self, child_a, max_depth=2)
        assert_valid_binary_tree(self, child_b, max_depth=2)
        assert_valid_binary_tree(self, repair(mutated, 2, n_features=2), max_depth=2)

    def test_repair_converts_malformed_and_too_deep_nodes_to_leaves(self):
        malformed = Node(
            feature=7, threshold=1.0,
            left=Node(feature=0, threshold=0.0,
                      left=Node(prediction=0), right=None),
            right=Node(prediction=5),
        )
        repaired = repair(malformed, max_depth=1, n_features=2)
        assert_valid_binary_tree(self, repaired, max_depth=1)
        self.assertLessEqual(profondeur(repaired), 1)

    def test_fitness_is_accuracy_minus_complexity(self):
        tree = Node(0, 0.0, Node(prediction=0), Node(prediction=1))
        score = calculer_fitness(tree, self.X, self.y,
                                 max_depth=2, max_leaves=4, alpha=0.1)
        # Perfect accuracy - 0.1 * (0.5*(1/2) + 0.5*(2/4)).
        self.assertAlmostEqual(score, 0.95)

    def test_ga_uses_only_passed_validation_data_and_respects_depth(self):
        tree, history = genetic_algorithm(
            self.X, self.y, self.X, self.y,
            population_size=4, generations=3, max_depth=2,
            taux_mutation=0.5, taux_croisement=0.8, random_state=8,
        )
        assert_valid_binary_tree(self, tree, max_depth=2)
        self.assertEqual(len(history), 3)
        self.assertLessEqual(nombre_feuilles(tree), 4)

    def test_comparison_handles_ga_rows_and_reports_interpretability(self):
        X_frame = pd.DataFrame(self.X, columns=["a", "b"])
        result = evaluer_modele("AG", self.parent_a, X_frame, self.y, 0.0)
        self.assertEqual(result["Profondeur"], 1)
        self.assertEqual(result["Nombre de feuilles"], 2)
        with self.assertRaises(ValueError):
            self.parent_a.predict(X_frame)
if __name__ == "__main__":
    unittest.main()