"""Download and load the UCI Human Activity Recognition (HAR) dataset.

30 volunteers wore a smartphone on the waist. Accelerometer and gyroscope signals
(50 Hz) were cut into windows of 2.56 s, and 561 features were computed per window.
Each window has one of 6 activity labels.
"""

import urllib.request
from pathlib import Path

import pandas as pd

from src import config


def download_data(force: bool = False) -> None:
    """Download all dataset files, trying each mirror in order."""
    missing = [f for f in config.DATA_FILES if force or not (config.RAW_DIR / f).exists()]
    if not missing:
        print(f"Data already exists: {config.RAW_DIR.relative_to(config.ROOT_DIR)}")
        return

    for base_url in config.DATA_BASE_URLS:
        try:
            for name in missing:
                target = config.RAW_DIR / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with urllib.request.urlopen(f"{base_url}/{name}", timeout=120) as response:
                    target.write_bytes(response.read())
            print(f"Saved {len(missing)} files to {config.RAW_DIR.relative_to(config.ROOT_DIR)}")
            return
        except Exception as error:  # try the next mirror
            print(f"Download failed from {base_url}: {error}")
    raise RuntimeError("All download sources failed. Download the UCI HAR dataset manually.")


def load_feature_names(raw_dir: Path = config.RAW_DIR) -> list:
    """Read the 561 feature names and make them unique.

    Some names appear several times in features.txt (e.g. bandsEnergy features),
    so a counter is added to repeated names.
    """
    names = pd.read_csv(raw_dir / "features.txt", sep=r"\s+", header=None)[1].tolist()
    seen, unique = {}, []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        unique.append(name if seen[name] == 1 else f"{name}__{seen[name]}")
    return unique


def load_activity_labels(raw_dir: Path = config.RAW_DIR) -> dict:
    """Map activity ids (1-6) to names, e.g. 1 -> 'WALKING'."""
    labels = pd.read_csv(raw_dir / "activity_labels.txt", sep=r"\s+", header=None)
    return dict(zip(labels[0], labels[1]))


def load_split(split: str, raw_dir: Path = config.RAW_DIR):
    """Load one official split ('train' or 'test').

    Returns X (DataFrame, 561 features), y (activity names) and subject ids.
    """
    folder = raw_dir / split
    X = pd.read_csv(folder / f"X_{split}.txt", sep=r"\s+", header=None)
    X.columns = load_feature_names(raw_dir)

    activity_names = load_activity_labels(raw_dir)
    y_ids = pd.read_csv(folder / f"y_{split}.txt", header=None)[0]
    y = y_ids.map(activity_names).rename("activity")
    subjects = pd.read_csv(folder / f"subject_{split}.txt", header=None)[0].rename("subject")
    return X, y, subjects


def load_dataset(raw_dir: Path = config.RAW_DIR):
    """Return the official train and test splits.

    The official split is by subject: 21 people in train, 9 different people in test.
    """
    return load_split("train", raw_dir), load_split("test", raw_dir)


def feature_group(name: str) -> str:
    """Sensor-signal group of a feature, e.g. 'tBodyAcc-mean()-X' -> 'tBodyAcc'.

    The prefix 't' means time domain and 'f' means frequency domain (FFT).
    """
    if name.startswith("angle"):
        return "angle"
    return name.split("-")[0]


if __name__ == "__main__":
    download_data()
    (X_train, y_train, s_train), (X_test, y_test, s_test) = load_dataset()
    print(f"Train: {X_train.shape}, {s_train.nunique()} subjects")
    print(f"Test:  {X_test.shape}, {s_test.nunique()} subjects")
