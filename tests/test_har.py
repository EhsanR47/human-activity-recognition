"""Unit tests for data loading, models, evaluation helpers and dimensionality demo."""
######################################################################################
import numpy as np
import pandas as pd
import pytest

from src.data import feature_group, load_dataset, load_feature_names
from src.dimensionality import distance_contrast
from src.evaluation import group_permutation_importance, per_subject_accuracy
from src.models import build_models, with_pca

ACTIVITIES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]
FEATURES = ["tBodyAcc-mean()-X", "tBodyAcc-std()-X", "fBodyGyro-bandsEnergy()-1,8",
            "fBodyGyro-bandsEnergy()-1,8", "angle(X,gravityMean)"]


@pytest.fixture
def raw_dir(tmp_path):
    """Write a tiny fake dataset with the same file layout as UCI HAR."""
    rng = np.random.default_rng(0)
    pd.DataFrame({0: range(1, len(FEATURES) + 1), 1: FEATURES}).to_csv(
        tmp_path / "features.txt", sep=" ", header=False, index=False)
    pd.DataFrame({0: range(1, 7), 1: ACTIVITIES}).to_csv(
        tmp_path / "activity_labels.txt", sep=" ", header=False, index=False)
    for split, n_rows, subjects in [("train", 60, [1, 3, 5]), ("test", 30, [2, 4])]:
        folder = tmp_path / split
        folder.mkdir()
        np.savetxt(folder / f"X_{split}.txt", rng.normal(size=(n_rows, len(FEATURES))))
        np.savetxt(folder / f"y_{split}.txt", np.tile(np.arange(1, 7), n_rows // 6), fmt="%d")
        np.savetxt(folder / f"subject_{split}.txt", np.resize(subjects, n_rows), fmt="%d")
    return tmp_path


def test_feature_names_are_unique(raw_dir):
    names = load_feature_names(raw_dir)
    assert len(names) == len(set(names)) == len(FEATURES)
    assert names[3] == "fBodyGyro-bandsEnergy()-1,8__2"


def test_load_dataset(raw_dir):
    (X_train, y_train, s_train), (X_test, y_test, s_test) = load_dataset(raw_dir)
    assert X_train.shape == (60, len(FEATURES))
    assert set(y_train) == set(ACTIVITIES)
    assert set(s_train).isdisjoint(set(s_test))  # official split is by subject


def test_feature_group():
    assert feature_group("tBodyAcc-mean()-X") == "tBodyAcc"
    assert feature_group("fBodyGyroMag-std()") == "fBodyGyroMag"
    assert feature_group("angle(X,gravityMean)") == "angle"


@pytest.mark.parametrize("name", list(build_models()))
def test_models_fit_and_predict(raw_dir, name):
    (X_train, y_train, _), (X_test, _, _) = load_dataset(raw_dir)
    model = build_models()[name].fit(X_train, y_train)
    assert set(model.predict(X_test)) <= set(ACTIVITIES)


def test_with_pca_inserts_step_after_scaler():
    model = with_pca(build_models()["knn"], n_components=3)
    assert [step for step, _ in model.steps] == ["scale", "pca", "model"]
    model = with_pca(build_models()["random_forest"], n_components=3)
    assert model.steps[0][0] == "pca"


def test_per_subject_accuracy():
    result = per_subject_accuracy(["a", "b", "a", "b"], ["a", "b", "b", "b"], [1, 1, 2, 2])
    assert result.loc[1] == 1.0 and result.loc[2] == 0.5


def test_group_permutation_importance_finds_useful_group():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"useful": rng.normal(size=400), "noise": rng.normal(size=400)})
    y = np.where(X["useful"] > 0, "up", "down")
    model = build_models()["logistic_regression"].fit(X, y)
    importance = group_permutation_importance(model, X, y, {"useful": ["useful"], "noise": ["noise"]})
    assert importance["useful"] > 0.3 > abs(importance["noise"])


def test_distance_contrast_shrinks_with_dimensions():
    result = distance_contrast([2, 10, 100, 1000], n_points=200)
    assert result["contrast"].is_monotonic_decreasing
