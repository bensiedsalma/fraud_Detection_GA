
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

try:
    from src.genetic_algorithm.representation import profondeur, nombre_feuilles
except ImportError:  # pragma: no cover - supports notebook path imports
    try:
        from genetic_algorithm.representation import profondeur, nombre_feuilles
    except ImportError:
        try:
            from representation import profondeur, nombre_feuilles
        except ImportError:
            profondeur = nombre_feuilles = None



def evaluer_modele(nom, modele, X_test, y_test, temps):
    """Evaluate one model on the independent test set.

    The custom GA tree predicts one numeric row at a time; sklearn estimators
    predict a complete feature table.  This distinction prevents the former
    ``KeyError: 20`` caused by passing a DataFrame to ``Node.predict``.
    """

    # Prediction
    if nom == "AG":
        predictions = [
            modele.predict(row)
            for row in X_test.to_numpy()
        ]
    else:
        predictions = modele.predict(X_test)

    # Calcul des métriques
    resultats = {
        "Modele": nom,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1-score": f1_score(y_test, predictions, zero_division=0),
        "Temps (s)": temps
    }

    if hasattr(modele, "get_depth"):
        resultats["Profondeur"] = modele.get_depth()
        resultats["Nombre de feuilles"] = modele.get_n_leaves()
    elif nom == "AG" and profondeur is not None:
        resultats["Profondeur"] = profondeur(modele)
        resultats["Nombre de feuilles"] = nombre_feuilles(modele)
    else:
        resultats["Profondeur"] = "-"
        resultats["Nombre de feuilles"] = "-"

    return resultats


def comparer_modeles(modeles, X_test, y_test):

    resultats = []

    for nom, modele, temps in modeles:

        resultat = evaluer_modele(
            nom,
            modele,
            X_test,
            y_test,
            temps
        )

        resultats.append(resultat)

    tableau = pd.DataFrame(resultats)

    return tableau


def sauvegarder_comparaison(tableau, chemin):

    tableau.to_csv(chemin, index=False)

    print("Tableau de comparaison sauvegardé avec succès !")
