"""Central configuration: paths, data sources and experiment settings."""

from pathlib import Path

# --- Paths ---
ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# --- Data sources ---
# Public copies of the UCI "Human Activity Recognition Using Smartphones" dataset
# (identical content). They are tried in order.
DATA_BASE_URLS = [
    "https://raw.githubusercontent.com/greenglobal/uci-har-dataset/master",
    "https://raw.githubusercontent.com/schakraborty369/UCI-HAR-Dataset/master/data",
]
DATA_FILES = [
    "features.txt",
    "activity_labels.txt",
    "train/X_train.txt", "train/y_train.txt", "train/subject_train.txt",
    "test/X_test.txt", "test/y_test.txt", "test/subject_test.txt",
]

# --- Experiment settings ---
RANDOM_STATE = 42
CV_FOLDS = 5
