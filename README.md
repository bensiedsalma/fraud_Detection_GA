# Fraud Detection Using Genetic Algorithm-Based Decision Trees

A Machine Learning project that explores the use of **Genetic Algorithms (GA)** to evolve interpretable binary decision trees for credit card fraud detection.

The project compares the proposed GA-based approach with three classical machine learning models: **CART, Random Forest, and Gradient Boosting**.

## Project Overview

Credit card fraud detection is a challenging classification problem because fraudulent transactions represent only a small proportion of all transactions.

This project investigates whether Genetic Algorithms can construct simple and interpretable decision trees while maintaining good predictive performance.

The main objectives are to:
- Implement a Genetic Algorithm for decision tree construction.
- Optimize tree structures using genetic operators.
- Compare the GA-based model with traditional machine learning algorithms.
- Evaluate both predictive performance and model complexity.

## Methodology

The Genetic Algorithm evolves a population of binary decision trees through several operations:

| Component | Description |
|---|---|
| Representation | Binary decision tree encoding |
| Selection | Selection of individuals based on fitness |
| Crossover | Subtree crossover between parent trees |
| Mutation | Modification of tree nodes |
| Repair | Validation of tree structure and depth constraints |
| Fitness | Accuracy − Complexity |

The fitness function combines predictive accuracy with a complexity penalty to encourage simpler trees.

## Models Compared

Four approaches are evaluated:

1. **Genetic Algorithm Decision Tree** — an evolved decision tree optimized through genetic operations.
2. **CART** — a classical Classification and Regression Tree.
3. **Random Forest** — an ensemble of decision trees.
4. **Gradient Boosting** — an ensemble learning method that builds trees sequentially.

## Dataset

The project uses the Credit Card Fraud Detection dataset.

It contains anonymized transaction features, including:
- `Time`: time elapsed between transactions.
- `Amount`: transaction amount.
- `V1` to `V28`: anonymized numerical features.
- `Class`: target variable (`0` = legitimate transaction, `1` = fraudulent transaction).

**Dataset source:** [Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

The dataset is not included in this repository. Download it and place `creditcard.csv` inside the `dataset/` directory before running the notebooks.

## Project Structure

```text
fraud_Detection_GA/
│
├── dataset/
│   └── processed/
│
├── notebooks/
│   ├── analyse_dataset.ipynb
│   ├── preprocessing.ipynb
│   └── evaluation.ipynb
│
├── src/
│   ├── decision_tree/
│   │   └── cart.py
│   │
│   ├── genetic_algorithm/
│   │   ├── representation.py
│   │   ├── selection.py
│   │   ├── crossover.py
│   │   ├── mutation.py
│   │   ├── repair.py
│   │   ├── fitness.py
│   │   └── genetic_algorithm.py
│   │
│   └── models_comparison/
│       └── comparison.py
│
├── tests/
│   └── test_ga_components.py
│
├── results/
│   ├── best_tree/
│   ├── graphs/
│   └── tables/
│
├── requirements.txt
├── README.md
└── REPORT.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/bensiedsalma/fraud_Detection_GA.git
cd fraud_Detection_GA
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Place the downloaded `creditcard.csv` file inside:

```text
dataset/creditcard.csv
```

## Running the Project

Execute the notebooks in the following order:

1. `notebooks/analyse_dataset.ipynb` — exploratory data analysis.
2. `notebooks/preprocessing.ipynb` — data cleaning and preprocessing.
3. `notebooks/evaluation.ipynb` — model training, comparison, and evaluation.

Run the unit tests using:

```bash
python -m unittest discover -s tests -v
```

## Evaluation Metrics

The models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Training Time

Tree interpretability is also assessed using:
- Maximum tree depth
- Number of leaves

The GA fitness function uses accuracy and a normalized complexity penalty. Additional classification metrics are reported to provide a more complete assessment of fraud detection performance.

All models are evaluated on the same independent test set to ensure a fair comparison.

## Results and Visualization

The `results/` directory contains generated outputs, including:

- Model performance comparisons.
- Decision tree complexity comparisons.
- Genetic Algorithm convergence history.
- Confusion matrices.
- Best evolved decision tree.

These results help analyze the balance between predictive performance and model interpretability.

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

## Authors

**Developed as an academic Data Science project.**

---

*This project explores the application of evolutionary computation to interpretable machine learning for credit card fraud detection.*
