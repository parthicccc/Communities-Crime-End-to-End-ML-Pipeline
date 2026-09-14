"""Model explanation helpers.

Permutation importance is used as the primary global explanation because
it is model-agnostic (the winner may be linear or tree-based) and measured
on held-out data, so it reflects what the model actually relies on to
generalise rather than what it memorised.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from . import config


def permutation_importances(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 10,
) -> pd.DataFrame:
    """Held-out permutation importance, sorted most-important first.

    Values are the increase in RMSE when a column is shuffled, so larger
    means the model depends on that column more. Near-zero or negative
    values mean the column is doing no work.
    """
    r = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=config.RANDOM_STATE,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    return (
        pd.DataFrame(
            {
                "feature": list(X.columns),
                "importance": r.importances_mean,
                "std": r.importances_std,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def linear_coefficients(model: Pipeline, feature_names: list[str]) -> pd.DataFrame | None:
    """Standardised coefficients, when the winning model is linear.

    Returns None for tree ensembles, which have no single coefficient
    vector. Because the pipeline standardises first, these coefficients are
    directly comparable across features.
    """
    est = model.named_steps.get("model")
    if est is None:
        return None
    # Unwrap a TransformedTargetRegressor if the target was transformed.
    inner = getattr(est, "regressor_", est)
    coef = getattr(inner, "coef_", None)
    if coef is None:
        return None
    coef = np.ravel(coef)
    if len(coef) != len(feature_names):
        return None
    return (
        pd.DataFrame({"feature": feature_names, "coefficient": coef})
        .assign(abs_coefficient=lambda d: d["coefficient"].abs())
        .sort_values("abs_coefficient", ascending=False)
        .reset_index(drop=True)
    )


def local_contributions(
    model: Pipeline,
    row: pd.DataFrame,
    baseline: pd.DataFrame,
    top_n: int = 12,
) -> pd.DataFrame:
    """Explain one prediction by one-at-a-time feature substitution.

    Starting from a `baseline` community (the training median), each
    feature is set to the value from `row` in isolation and the change in
    the prediction is recorded. This is an additive-in-isolation attribution
    -- cheap, model-agnostic, and easy to read -- but it ignores feature
    interactions, so contributions will not sum exactly to the final
    prediction. It is a directional explanation, not an exact decomposition.
    """
    base_pred = float(model.predict(baseline)[0])
    rows = []
    for col in row.columns:
        if pd.isna(row.iloc[0][col]):
            continue
        probe = baseline.copy()
        probe.iloc[0, probe.columns.get_loc(col)] = row.iloc[0][col]
        delta = float(model.predict(probe)[0]) - base_pred
        rows.append(
            {
                "feature": col,
                "value": float(row.iloc[0][col]),
                "baseline_value": float(baseline.iloc[0][col]),
                "contribution": delta,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return (
        out.assign(abs_contribution=lambda d: d["contribution"].abs())
        .sort_values("abs_contribution", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
