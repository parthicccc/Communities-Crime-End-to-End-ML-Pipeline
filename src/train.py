"""Model selection, evaluation and artifact persistence.

Run as a script:

    python -m src.train                 # train both variants, save bundle
    python -m src.train --quick         # smaller grids, for a fast smoke run
    python -m src.train --force-download

Design notes
------------
* Candidates are compared by cross-validated RMSE on the *training* folds
  only. The held-out folds (9 and 10, per `config.TEST_FOLDS`) are scored
  exactly once, at the end, for the winner and every runner-up -- they are
  never used to choose anything.
* Cross-validation uses the dataset's own `fold` column via
  `PredefinedSplit` rather than a random KFold. Those folds are the
  non-random split published with the data, so results are stable across
  runs and comparable with prior work on this dataset.
* Every candidate is a full Pipeline (impute -> scale -> estimator), so
  preprocessing statistics are refit inside each CV fold.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline

from . import config, data as data_mod, features as feat_mod

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Candidate definitions
# --------------------------------------------------------------------------
@dataclass
class Candidate:
    name: str
    estimator: object
    param_grid: dict = field(default_factory=dict)
    scale: bool = True
    note: str = ""


def _sqrt_target(estimator):
    """Wrap an estimator so it learns on sqrt(y).

    The target is a right-skewed rate (skew ~1.5) bounded at 0, so a square
    root stabilises the variance for models that assume homoscedastic
    errors. The wrapper inverts the transform on predict, so the reported
    metrics stay on the original scale.
    """
    return TransformedTargetRegressor(
        regressor=estimator, func=np.sqrt, inverse_func=np.square
    )


def build_candidates(quick: bool = False) -> list[Candidate]:
    rs = config.RANDOM_STATE
    alphas = np.logspace(-3, 2, 8 if quick else 20)

    cands = [
        Candidate(
            "Baseline (mean)",
            DummyRegressor(strategy="mean"),
            scale=False,
            note="Predicts the training mean. Any useful model must beat this.",
        ),
        Candidate(
            "Ridge",
            Ridge(random_state=rs),
            {"model__alpha": alphas},
            note="L2-penalised linear model; handles the correlated census columns.",
        ),
        Candidate(
            "Lasso",
            Lasso(random_state=rs, max_iter=20000),
            {"model__alpha": np.logspace(-4, 0, 8 if quick else 20)},
            note="L1 penalty; performs feature selection by zeroing coefficients.",
        ),
        Candidate(
            "ElasticNet",
            ElasticNet(random_state=rs, max_iter=20000),
            {
                "model__alpha": np.logspace(-4, 0, 6 if quick else 12),
                "model__l1_ratio": [0.1, 0.5, 0.9],
            },
            note="Blends L1 and L2; a middle ground between Ridge and Lasso.",
        ),
        Candidate(
            "Ridge (sqrt target)",
            _sqrt_target(Ridge(random_state=rs)),
            {"model__regressor__alpha": alphas},
            note="Ridge trained on sqrt(y) to counter the target's right skew.",
        ),
        Candidate(
            "Random Forest",
            RandomForestRegressor(
                n_estimators=200 if quick else 400,
                random_state=rs,
                n_jobs=-1,
            ),
            {}
            if quick
            else {
                "model__max_features": ["sqrt", 0.3],
                "model__min_samples_leaf": [1, 2, 5],
            },
            scale=False,
            note="Bagged trees; captures non-linearity and interactions.",
        ),
        Candidate(
            "Gradient Boosting",
            HistGradientBoostingRegressor(random_state=rs),
            {}
            if quick
            else {
                "model__learning_rate": [0.03, 0.06, 0.1],
                "model__max_leaf_nodes": [15, 31],
                "model__min_samples_leaf": [10, 20],
            },
            scale=False,
            note="Histogram-based boosted trees; usually the strongest tabular baseline.",
        ),
    ]
    return cands


def make_pipeline(cand: Candidate) -> Pipeline:
    pre = feat_mod.build_preprocessor(scale=cand.scale)
    return Pipeline([*pre.steps, ("model", cand.estimator)])


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    # Spearman is undefined when either side is constant (e.g. the mean
    # baseline, which predicts one value for every row).
    if len(y_true) > 2 and y_true.std() > 0 and y_pred.std() > 0:
        rho = spearmanr(y_true, y_pred).statistic
    else:
        rho = np.nan
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": float(rho) if rho == rho else None,
    }


def clip_to_range(pred: np.ndarray) -> np.ndarray:
    """Constrain predictions to the target's valid [0, 1] range."""
    lo, hi = config.TARGET_RANGE
    return np.clip(np.asarray(pred, dtype=float), lo, hi)


def _predefined_split(fold_train: pd.Series) -> PredefinedSplit:
    """Build a CV splitter from the dataset's published fold assignment."""
    codes = pd.Categorical(fold_train.astype("int64")).codes
    return PredefinedSplit(test_fold=codes)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train_variant(
    dataset: dict, exclude_race: bool = False, quick: bool = False
) -> dict:
    """Run the full selection + evaluation loop for one feature variant."""
    X_tr, X_te, y_tr, y_te, fold_tr, meta_tr, meta_te = (
        data_mod.train_test_split_by_fold(dataset)
    )
    X_tr, feature_names = feat_mod.select_features(X_tr, exclude_race)
    X_te, _ = feat_mod.select_features(X_te, exclude_race)

    cv = _predefined_split(fold_tr)
    n_splits = cv.get_n_splits()
    label = "without race features" if exclude_race else "all features"
    log.info(
        "Training variant [%s]: %d train / %d test rows, %d features, %d CV folds",
        label, len(X_tr), len(X_te), len(feature_names), n_splits,
    )

    results, fitted = [], {}
    for cand in build_candidates(quick=quick):
        t0 = time.perf_counter()
        pipe = make_pipeline(cand)

        if cand.param_grid:
            search = GridSearchCV(
                pipe,
                cand.param_grid,
                scoring="neg_root_mean_squared_error",
                cv=cv,
                n_jobs=-1,
                refit=True,
            )
            search.fit(X_tr, y_tr)
            best, cv_rmse, best_params = (
                search.best_estimator_,
                -search.best_score_,
                search.best_params_,
            )
            idx = int(search.best_index_)
            cv_std = float(search.cv_results_["std_test_score"][idx])
        else:
            from sklearn.model_selection import cross_val_score

            scores = cross_val_score(
                pipe, X_tr, y_tr,
                scoring="neg_root_mean_squared_error", cv=cv, n_jobs=-1,
            )
            cv_rmse, cv_std, best_params = -scores.mean(), float(scores.std()), {}
            best = pipe.fit(X_tr, y_tr)

        test_metrics = regression_metrics(y_te, clip_to_range(best.predict(X_te)))
        elapsed = time.perf_counter() - t0

        results.append(
            {
                "model": cand.name,
                "note": cand.note,
                "cv_rmse": float(cv_rmse),
                "cv_rmse_std": cv_std,
                "test_rmse": test_metrics["rmse"],
                "test_mae": test_metrics["mae"],
                "test_r2": test_metrics["r2"],
                "test_spearman": test_metrics["spearman"],
                "best_params": {k: _jsonable(v) for k, v in best_params.items()},
                "fit_seconds": round(elapsed, 2),
            }
        )
        fitted[cand.name] = best
        log.info(
            "  %-22s CV RMSE %.4f (+/-%.4f)  test RMSE %.4f  R2 %.3f  [%.1fs]",
            cand.name, cv_rmse, cv_std, test_metrics["rmse"], test_metrics["r2"], elapsed,
        )

    leaderboard = sorted(results, key=lambda r: r["cv_rmse"])
    best_name = leaderboard[0]["model"]
    best_model = fitted[best_name]
    log.info("  -> selected %s (lowest CV RMSE)", best_name)

    y_pred_te = clip_to_range(best_model.predict(X_te))
    y_pred_tr = clip_to_range(best_model.predict(X_tr))

    return {
        "exclude_race": exclude_race,
        "label": label,
        "feature_names": feature_names,
        "best_model_name": best_name,
        "model": best_model,
        "leaderboard": leaderboard,
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "n_cv_folds": int(n_splits),
        "train_metrics": regression_metrics(y_tr, y_pred_tr),
        "test_metrics": regression_metrics(y_te, y_pred_te),
        # Kept for the diagnostics page: actual vs predicted, residuals.
        "test_predictions": pd.DataFrame(
            {
                "community": meta_te.get("communityname", pd.Series(dtype="string")),
                "state": meta_te.get("state", pd.Series(dtype="string")),
                "actual": y_te.to_numpy(),
                "predicted": y_pred_te,
                "residual": y_te.to_numpy() - y_pred_te,
            }
        ),
        # Median of each training feature: the default when the app's
        # single-community form starts up, and the neutral fill for any
        # slider the user does not touch.
        "feature_medians": X_tr.median(numeric_only=True).to_dict(),
        "feature_quantiles": {
            c: {
                "min": float(X_tr[c].min()),
                "q1": float(X_tr[c].quantile(0.25)),
                "median": float(X_tr[c].median()),
                "q3": float(X_tr[c].quantile(0.75)),
                "max": float(X_tr[c].max()),
            }
            for c in feature_names
        },
    }


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def run(quick: bool = False, force_download: bool = False) -> dict:
    """Train both feature variants and persist a single model bundle."""
    dataset = data_mod.load_clean(force_download=force_download)
    log.info(
        "Cleaned data: %d rows, %d features (dropped %d high-missing columns)",
        len(dataset["X"]),
        dataset["X"].shape[1],
        len(dataset["report"]["dropped_high_missing"]),
    )

    variants = {
        "all_features": train_variant(dataset, exclude_race=False, quick=quick),
        "no_race_features": train_variant(dataset, exclude_race=True, quick=quick),
    }

    from . import explain as explain_mod

    for key, v in variants.items():
        X_tr, X_te, y_tr, y_te, *_ = data_mod.train_test_split_by_fold(dataset)
        X_te, _ = feat_mod.select_features(X_te, v["exclude_race"])
        v["importances"] = explain_mod.permutation_importances(
            v["model"], X_te, y_te, n_repeats=10
        )
        log.info("Permutation importance computed for %s", key)

    bundle = {
        "created_utc": pd.Timestamp.now("UTC").isoformat(),
        "dataset": {
            "uci_id": config.UCI_DATASET_ID,
            "target": config.TARGET,
            "test_folds": list(config.TEST_FOLDS),
            **dataset["report"],
        },
        "target_stats": {
            "mean": float(dataset["y"].mean()),
            "std": float(dataset["y"].std()),
            "min": float(dataset["y"].min()),
            "max": float(dataset["y"].max()),
            "skew": float(dataset["y"].skew()),
        },
        "variants": variants,
    }
    joblib.dump(bundle, config.MODEL_BUNDLE, compress=3)
    log.info("Saved model bundle -> %s", config.MODEL_BUNDLE)

    # A JSON-only summary, handy for reading metrics without loading joblib.
    summary = {
        "created_utc": bundle["created_utc"],
        "dataset": bundle["dataset"],
        "target_stats": bundle["target_stats"],
        "variants": {
            k: {
                "label": v["label"],
                "best_model_name": v["best_model_name"],
                "n_features": len(v["feature_names"]),
                "n_train": v["n_train"],
                "n_test": v["n_test"],
                "train_metrics": v["train_metrics"],
                "test_metrics": v["test_metrics"],
                "leaderboard": v["leaderboard"],
                "top_features": v["importances"].head(15).to_dict("records"),
            }
            for k, v in variants.items()
        },
    }
    config.METRICS_JSON.write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    log.info("Saved metrics summary -> %s", config.METRICS_JSON)
    return bundle


def main() -> None:
    p = argparse.ArgumentParser(description="Train the Communities & Crime models.")
    p.add_argument("--quick", action="store_true", help="smaller grids, faster run")
    p.add_argument(
        "--force-download", action="store_true", help="re-fetch from UCI, ignore cache"
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S"
    )
    run(quick=args.quick, force_download=args.force_download)


if __name__ == "__main__":
    main()
