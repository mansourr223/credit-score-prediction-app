"""
Model comparison numbers, copied as-is from the 5-fold cross_validate() output
in Credit_Score.ipynb (macro-averaged, in %). These are training results, not
something the deployed app recomputes live — re-running CV for 4 models inside
a web app would be far too slow, exactly like a real MLOps dashboard would
show a frozen "experiment tracking" snapshot.
"""

MODEL_COMPARISON = [
    {"model": "Gaussian NB",   "train_precision": 60.40, "test_precision": 60.34,
     "train_recall": 65.73, "test_recall": 65.71, "train_f1": 56.03, "test_f1": 55.97,
     "test_accuracy": 55.84},
    {"model": "Decision Tree", "train_precision": 100.0, "test_precision": 61.62,
     "train_recall": 100.0, "test_recall": 62.38, "train_f1": 100.0, "test_f1": 61.98,
     "test_accuracy": 64.35},
    {"model": "Random Forest", "train_precision": 100.0, "test_precision": 72.36,
     "train_recall": 100.0, "test_recall": 75.19, "train_f1": 100.0, "test_f1": 73.45,
     "test_accuracy": 74.87},
    {"model": "XGBoost",       "train_precision": 80.26, "test_precision": 72.71,
     "train_recall": 80.66, "test_recall": 72.75, "train_f1": 80.45, "test_f1": 72.71,
     "test_accuracy": None},
]

FINAL_MODEL_NAME = "Random Forest"

FINAL_MODEL = {
    "algorithm": "Random Forest Classifier",
    "imbalance_handling": "SMOTE (oversampling on the training folds)",
    "cv_folds": 5,
    "test_accuracy": 74.87,
    "test_precision_macro": 72.36,
    "test_recall_macro": 75.19,
    "test_f1_macro": 73.45,
}

# RandomizedSearchCV experiment: tuning was tried but the tuned model
# under-performed the untuned default Random Forest on the held-out test set,
# so the default model was kept for production.
TUNING_EXPERIMENT = {
    "search_space": "Model__max_depth: [5, 6, 7], Model__n_estimators: [50, 100, 200]",
    "best_params": {"n_estimators": 100, "max_depth": 7},
    "best_cv_f1_macro": 65.82,
    "test_f1_macro": 65.76,
    "conclusion": (
        "The constrained max_depth (5-7) underfit compared to the default, "
        "fully-grown Random Forest — the untuned model scored a higher "
        "macro F1 on the test set (73.45% vs 65.76%), so it was kept as the "
        "production model."
    ),
}

CLASS_DISTRIBUTION = {"Standard": 53.17, "Poor": 29.00, "Good": 17.83}
