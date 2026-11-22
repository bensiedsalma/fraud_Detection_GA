from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import time


def entrainer_cart(X_train, y_train):
    """
    Entraîner un modèle CART.
    """

    modele = DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    )

    debut = time.time()

    modele.fit(X_train, y_train)

    temps = time.time() - debut

    return modele, temps


def evaluer_cart(modele, X_test, y_test):
    """
    Évaluer le modèle CART.
    """

    predictions = modele.predict(X_test)

    resultats = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1-score": f1_score(
            y_test, predictions, zero_division=0
        )
    }

    return predictions, resultats


def complexite_cart(modele):
    """
    Mesurer la complexité de l'arbre.
    """

    return {
        "Profondeur": modele.get_depth(),
        "Nombre de feuilles": modele.get_n_leaves()
    }
