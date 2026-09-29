"""Candidate models. Every model is a Pipeline, so scaling is fitted inside each CV fold."""

from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src import config


def build_models() -> dict:
    """Six classic models, from simple to complex.

    Scaling matters for distance- and margin-based models (kNN, SVM, logistic
    regression). Tree models do not need it, but it does not hurt them either.
    """
    rs = config.RANDOM_STATE
    return {
        "knn": Pipeline([
            ("scale", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=10)),
        ]),
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=1.0, max_iter=3000)),
        ]),
        "linear_svm": Pipeline([
            ("scale", StandardScaler()),
            ("model", SVC(kernel="linear", C=0.1)),
        ]),
        "rbf_svm": Pipeline([
            ("scale", StandardScaler()),
            ("model", SVC(kernel="rbf", C=10, gamma="scale")),
        ]),
        "random_forest": Pipeline([
            ("model", RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=rs)),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("model", HistGradientBoostingClassifier(
                learning_rate=0.1, max_iter=200, early_stopping=False, random_state=rs,
            )),
        ]),
    }


def with_pca(model: Pipeline, n_components) -> Pipeline:
    """Insert PCA after scaling (or at the start if the model has no scaler)."""
    steps = list(model.steps)
    position = 1 if steps[0][0] == "scale" else 0
    steps.insert(position, ("pca", PCA(n_components=n_components, random_state=config.RANDOM_STATE)))
    return Pipeline(steps)
