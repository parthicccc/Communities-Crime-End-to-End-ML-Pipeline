"""Preprocessing: column selection and the sklearn transformer stack.

All surviving predictors are continuous and already min-max normalised to
[0, 1] by the dataset authors, so preprocessing is deliberately small:
median imputation for the handful of residual gaps, plus standardisation
for the models that need it (the linear family). Fitting happens inside a
Pipeline so the imputer/scaler statistics are learned on training folds
only -- no leakage into validation.
"""

from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config


def select_features(
    X: pd.DataFrame, exclude_race: bool = False
) -> tuple[pd.DataFrame, list[str]]:
    """Return the feature matrix and the ordered list of columns used.

    With `exclude_race=True`, race/ethnicity/immigration-derived predictors
    listed in `config.RACE_FEATURES` are dropped. Training both variants
    lets the app show what those columns are actually buying in accuracy.
    """
    cols = list(X.columns)
    if exclude_race:
        drop = set(config.RACE_FEATURES)
        cols = [c for c in cols if c not in drop]
    return X[cols].copy(), cols


def build_preprocessor(scale: bool = True) -> Pipeline:
    """Imputation (+ optional standardisation) as a fittable Pipeline."""
    steps: list[tuple[str, object]] = [
        ("impute", SimpleImputer(strategy="median")),
    ]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


def align_to_schema(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Coerce an arbitrary input frame to the schema the model expects.

    Reorders columns, adds any that are absent as NaN (the imputer fills
    them), and drops extras. This is what makes batch CSV upload in the app
    forgiving about column order and stray identifier columns.
    """
    out = df.reindex(columns=feature_names)
    return out.astype("float64")


def schema_diff(df: pd.DataFrame, feature_names: list[str]) -> dict:
    """Describe how an uploaded frame differs from the training schema."""
    incoming = set(df.columns)
    expected = set(feature_names)
    return {
        "missing": sorted(expected - incoming),
        "extra": sorted(incoming - expected),
        "matched": sorted(expected & incoming),
    }
