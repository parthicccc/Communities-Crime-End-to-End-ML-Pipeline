"""Ingestion layer: fetch the UCI dataset, cache it, and clean it.

`load_raw()` is the only function that touches the network. Everything
downstream works off the cached parquet, so training and the Streamlit app
are reproducible offline.
"""

from __future__ import annotations

import json
import logging
import re

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)


def _enable_system_trust_store() -> None:
    """Verify TLS against the OS trust store instead of certifi.

    Networks that terminate TLS (university proxies, some antivirus
    products) present a root CA that lives in the Windows/macOS trust
    store but not in certifi's bundle, which makes the UCI download fail
    with CERTIFICATE_VERIFY_FAILED. Delegating to the OS verifier fixes
    that. It is a no-op on a normal connection, hence best-effort.
    """
    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception:  # pragma: no cover - purely environmental
        pass


def download_raw(force: bool = False) -> pd.DataFrame:
    """Fetch dataset 183 from UCI and cache features+target as one frame."""
    if config.RAW_CACHE.exists() and not force:
        return pd.read_parquet(config.RAW_CACHE)

    _enable_system_trust_store()
    from ucimlrepo import fetch_ucirepo

    log.info("Downloading UCI dataset id=%s ...", config.UCI_DATASET_ID)
    ds = fetch_ucirepo(id=config.UCI_DATASET_ID)

    raw = pd.concat([ds.data.features, ds.data.targets], axis=1)
    # Parquet needs a single dtype per column; the raw frame mixes numeric
    # and '?'-bearing string columns, so store everything as text and let
    # `clean()` do the typing. Round-trips losslessly.
    raw.astype("string").to_parquet(config.RAW_CACHE, index=False)

    ds.variables.to_csv(config.VARIABLES_CACHE, index=False)
    config.METADATA_CACHE.write_text(
        json.dumps(ds.metadata, indent=2, default=str), encoding="utf-8"
    )
    log.info("Cached %d rows x %d cols -> %s", *raw.shape, config.RAW_CACHE)
    return pd.read_parquet(config.RAW_CACHE)


def load_raw(force_download: bool = False) -> pd.DataFrame:
    """Return the raw frame, downloading on first use."""
    return download_raw(force=force_download)


def to_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce every column to float, turning the '?' marker into NaN.

    Used by both training and serving so an uploaded CSV is typed exactly
    the way the training data was.
    """
    cols = {}
    for col in df.columns:
        s = df[col]
        if s.dtype == object or isinstance(s.dtype, pd.StringDtype):
            s = s.replace(config.MISSING_MARKER, np.nan)
        cols[col] = pd.to_numeric(s, errors="coerce").astype("float64")
    # Build in one shot; assigning column-by-column fragments the frame.
    return pd.DataFrame(cols, index=df.index)


def clean(raw: pd.DataFrame) -> dict:
    """Turn the raw frame into a modelling-ready feature matrix.

    Returns a dict with the feature matrix `X`, target `y`, the `fold`
    assignment, human-readable `meta` columns, and a `report` describing
    every column that was dropped and why.
    """
    df = raw.copy()

    # Identifier/meta columns kept for display but never used as features.
    meta = pd.DataFrame(index=df.index)
    if config.NAME_COL in df.columns:
        meta[config.NAME_COL] = df[config.NAME_COL].astype("string")
    if config.STATE_COL in df.columns:
        state = pd.to_numeric(df[config.STATE_COL], errors="coerce")
        meta["state_code"] = state
        meta["state"] = state.map(config.STATE_CODES).astype("string")

    fold = pd.to_numeric(df[config.FOLD_COL], errors="coerce").astype("Int64")
    y = pd.to_numeric(df[config.TARGET], errors="coerce").astype("float64")

    feature_cols = [
        c
        for c in df.columns
        if c not in config.NON_PREDICTIVE and c != config.TARGET
    ]
    X = to_numeric_features(df[feature_cols])

    # Drop columns that are mostly missing -- imputing an 84%-missing
    # column fabricates most of its values.
    missing_frac = X.isna().mean()
    high_missing = sorted(
        missing_frac[missing_frac > config.MAX_MISSING_FRACTION].index
    )
    X = X.drop(columns=high_missing)

    # Drop constant columns; they carry no signal and break scaling.
    constant = sorted(X.columns[X.nunique(dropna=True) <= 1])
    X = X.drop(columns=constant)

    # Rows with no target cannot train or score.
    keep = y.notna()
    X, y, fold, meta = X[keep], y[keep], fold[keep], meta[keep]

    report = {
        "n_rows_raw": int(len(raw)),
        "n_rows_kept": int(keep.sum()),
        "n_features_raw": len(feature_cols),
        "n_features_kept": X.shape[1],
        "dropped_non_predictive": list(config.NON_PREDICTIVE),
        "dropped_high_missing": high_missing,
        "dropped_constant": constant,
        "max_missing_fraction": config.MAX_MISSING_FRACTION,
        "residual_missing_cols": {
            c: round(float(v), 5)
            for c, v in X.isna().mean().items()
            if v > 0
        },
    }

    return {
        "X": X.reset_index(drop=True),
        "y": y.reset_index(drop=True),
        "fold": fold.reset_index(drop=True),
        "meta": meta.reset_index(drop=True),
        "report": report,
    }


def load_clean(force_download: bool = False) -> dict:
    """Convenience: download (or read cache) then clean."""
    return clean(load_raw(force_download=force_download))


def train_test_split_by_fold(data: dict, test_folds=config.TEST_FOLDS):
    """Split using the dataset's own fold column.

    Returns (X_train, X_test, y_train, y_test, fold_train, meta_train,
    meta_test). Using the published folds instead of a random split keeps
    the held-out set fixed across runs and comparable with prior work.
    """
    is_test = data["fold"].isin(list(test_folds)).to_numpy()
    X, y, fold, meta = data["X"], data["y"], data["fold"], data["meta"]
    return (
        X[~is_test].reset_index(drop=True),
        X[is_test].reset_index(drop=True),
        y[~is_test].reset_index(drop=True),
        y[is_test].reset_index(drop=True),
        fold[~is_test].reset_index(drop=True),
        meta[~is_test].reset_index(drop=True),
        meta[is_test].reset_index(drop=True),
    )


_VAR_LINE = re.compile(r"^\s*--\s*([A-Za-z0-9_]+)\s*:\s*(.+?)\s*$")


def variable_descriptions() -> dict[str, str]:
    """Map column name -> plain-English description, for tooltips in the app.

    The UCI API returns an empty `description` column for this dataset, but
    the same information is present as a free-text block in the metadata
    under `additional_info.variable_info`, formatted as
    ``-- colname: description (numeric - decimal)``. We parse that and fall
    back to the variables CSV if it ever gets populated upstream.
    """
    out: dict[str, str] = {}

    if config.METADATA_CACHE.exists():
        try:
            meta = json.loads(config.METADATA_CACHE.read_text(encoding="utf-8"))
            info = (meta.get("additional_info") or {}).get("variable_info") or ""
            for line in str(info).splitlines():
                m = _VAR_LINE.match(line)
                if not m:
                    continue
                name, desc = m.group(1), m.group(2)
                # Strip the "(numeric - decimal)" style type annotations.
                desc = re.sub(r"\s*\((?:numeric|string|nominal)[^)]*\)", "", desc)
                out[name] = desc.strip().rstrip(" .:")
        except Exception:
            pass

    if config.VARIABLES_CACHE.exists():
        v = pd.read_csv(config.VARIABLES_CACHE)
        if {"name", "description"}.issubset(v.columns):
            for _, r in v.iterrows():
                if pd.notna(r.get("description")) and str(r["name"]) not in out:
                    out[str(r["name"])] = str(r["description"])

    return out


def dataset_metadata() -> dict:
    """Raw UCI metadata dict (abstract, citation, etc.) for the app."""
    if not config.METADATA_CACHE.exists():
        return {}
    try:
        return json.loads(config.METADATA_CACHE.read_text(encoding="utf-8"))
    except Exception:
        return {}
