"""Central configuration: paths, dataset constants, and column groupings.

Everything that the pipeline needs to know about the *shape* of the
Communities and Crime dataset lives here, so the data/train/serve layers
never hard-code column names of their own.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

for _d in (DATA_DIR, MODEL_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RAW_CACHE = DATA_DIR / "communities_raw.parquet"
VARIABLES_CACHE = DATA_DIR / "variables.csv"
METADATA_CACHE = DATA_DIR / "metadata.json"

MODEL_BUNDLE = MODEL_DIR / "model_bundle.joblib"
METRICS_JSON = MODEL_DIR / "metrics.json"

# --------------------------------------------------------------------------
# Dataset constants
# --------------------------------------------------------------------------
UCI_DATASET_ID = 183
TARGET = "ViolentCrimesPerPop"

# The UCI documentation marks these as non-predictive. `fold` is the
# dataset's own 10-fold CV assignment -- we strip it from the feature
# matrix but keep it aside to build an honest train/test split.
NON_PREDICTIVE = ["state", "county", "community", "communityname", "fold"]
FOLD_COL = "fold"
NAME_COL = "communityname"
STATE_COL = "state"

# Folds reserved as the held-out test set. Using the dataset's own fold
# assignment (rather than a fresh random split) keeps results comparable
# with the literature and removes any chance of us picking a lucky seed.
TEST_FOLDS = (9, 10)

# Missing marker used throughout the raw UCI file.
MISSING_MARKER = "?"

# Columns missing more than this fraction are dropped rather than imputed.
# At 0.5 this removes the 22 LEMAS police-survey columns (84% missing),
# where imputation would be inventing the majority of the data.
MAX_MISSING_FRACTION = 0.5

# Race/ethnicity-derived predictors. The pipeline can be trained with or
# without these; see README for why that switch exists.
RACE_FEATURES = [
    "racepctblack",
    "racePctWhite",
    "racePctAsian",
    "racePctHisp",
    "whitePerCap",
    "blackPerCap",
    "indianPerCap",
    "AsianPerCap",
    "OtherPerCap",
    "HispPerCap",
    "PctSpeakEnglOnly",
    "PctNotSpeakEnglWell",
    "PctForeignBorn",
    "PctImmigRecent",
    "PctImmigRec5",
    "PctImmigRec8",
    "PctImmigRec10",
    "PctRecentImmig",
    "PctRecImmig5",
    "PctRecImmig8",
    "PctRecImmig10",
    "NumImmig",
]

# Target is a normalised per-population rate in [0, 1]; predictions are
# clipped to this range at serving time.
TARGET_RANGE = (0.0, 1.0)

RANDOM_STATE = 42

# FIPS state code -> postal abbreviation, for readable EDA in the app.
STATE_CODES = {
    1: "AL", 2: "AK", 4: "AZ", 5: "AR", 6: "CA", 8: "CO", 9: "CT", 10: "DE",
    11: "DC", 12: "FL", 13: "GA", 15: "HI", 16: "ID", 17: "IL", 18: "IN",
    19: "IA", 20: "KS", 21: "KY", 22: "LA", 23: "ME", 24: "MD", 25: "MA",
    26: "MI", 27: "MN", 28: "MS", 29: "MO", 30: "MT", 31: "NE", 32: "NV",
    33: "NH", 34: "NJ", 35: "NM", 36: "NY", 37: "NC", 38: "ND", 39: "OH",
    40: "OK", 41: "OR", 42: "PA", 44: "RI", 45: "SC", 46: "SD", 47: "TN",
    48: "TX", 49: "UT", 50: "VT", 51: "VA", 53: "WA", 54: "WV", 55: "WI",
    56: "WY",
}
