"""Model comparison, metrics, plots and interpretation helpers."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.model_selection import cross_validate

SCORING = {"accuracy": "accuracy", "macro_f1": "f1_macro"}


def compare_models(models: dict, X, y, cv, groups=None) -> pd.DataFrame:
    """Cross-validate each model and return mean/std of accuracy, macro-F1 and fit time."""
    rows = []
    for name, model in models.items():
        scores = cross_validate(model, X, y, groups=groups, cv=cv, scoring=SCORING, n_jobs=-1)
        rows.append({
            "model": name,
            "accuracy_mean": scores["test_accuracy"].mean(),
            "accuracy_std": scores["test_accuracy"].std(),
            "macro_f1_mean": scores["test_macro_f1"].mean(),
            "macro_f1_std": scores["test_macro_f1"].std(),
            "fit_time_s": scores["fit_time"].mean(),
        })
    return pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False).reset_index(drop=True)


def plot_confusion_matrix(y_true, y_pred, labels, path=None):
    """Row-normalised confusion matrix: each row shows where one true class goes."""
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=labels, normalize="true", values_format=".2f",
        cmap="Blues", ax=ax, colorbar=False,
    )
    ax.set_title("Confusion matrix (rows = true activity, normalised)")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def per_subject_accuracy(y_true, y_pred, subjects) -> pd.Series:
    """Accuracy for each test subject: shows how well the model generalises to people."""
    frame = pd.DataFrame({"correct": np.asarray(y_true) == np.asarray(y_pred),
                          "subject": np.asarray(subjects)})
    return frame.groupby("subject")["correct"].mean().sort_values()


def group_permutation_importance(fitted_model, X: pd.DataFrame, y, feature_groups: dict,
                                 n_repeats: int = 3, random_state: int = 0) -> pd.Series:
    """Drop in macro-F1 when all features of one sensor group are shuffled together.

    Shuffling a whole group (instead of single features) avoids a common problem:
    with many correlated features, shuffling one of them changes little, because
    the others still carry the same information.
    """
    rng = np.random.default_rng(random_state)
    baseline = f1_score(y, fitted_model.predict(X), average="macro")
    importance = {}
    for group, columns in feature_groups.items():
        drops = []
        for _ in range(n_repeats):
            X_shuffled = X.copy()
            order = rng.permutation(len(X))
            X_shuffled[columns] = X[columns].to_numpy()[order]
            drops.append(baseline - f1_score(y, fitted_model.predict(X_shuffled), average="macro"))
        importance[group] = np.mean(drops)
    return pd.Series(importance).sort_values(ascending=False)


def evaluate(model, X_train, y_train, X_test, y_test) -> dict:
    """Fit a fresh copy of the model and return test metrics and predictions."""
    fitted = clone(model).fit(X_train, y_train)
    y_pred = fitted.predict(X_test)
    return {
        "model": fitted,
        "y_pred": y_pred,
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
    }
