"""End-to-end checks for the pipeline and the serving paths the app uses.

Run with:  python -m tests.test_pipeline
(Also works under pytest if it is installed.)
"""

from __future__ import annotations

import io
import sys

import joblib
import numpy as np
import pandas as pd

from src import config, data as data_mod, explain as explain_mod, features as feat_mod
from src.train import clip_to_range, regression_metrics

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


# --------------------------------------------------------------------------
def test_cleaning():
    print("\nCleaning")
    d = data_mod.load_clean()
    X, y, fold = d["X"], d["y"], d["fold"]

    check("all predictors are float", (X.dtypes == "float64").all())
    check("no '?' survives", not X.isin(["?"]).any().any())
    check("target has no NaN", int(y.isna().sum()) == 0)
    check(
        "target within [0, 1]",
        float(y.min()) >= 0.0 and float(y.max()) <= 1.0,
        f"[{y.min():.3f}, {y.max():.3f}]",
    )
    check(
        "non-predictive columns removed",
        all(c not in X.columns for c in config.NON_PREDICTIVE),
    )
    check(
        "no column exceeds the missingness threshold",
        bool((X.isna().mean() <= config.MAX_MISSING_FRACTION).all()),
    )
    check("LEMAS columns dropped", "PolicOperBudg" not in X.columns)
    check("fold covers 1..10", sorted(fold.unique().tolist()) == list(range(1, 11)))
    return d


def test_split(d):
    print("\nTrain/test split")
    Xtr, Xte, ytr, yte, ftr, mtr, mte = data_mod.train_test_split_by_fold(d)
    check("train + test = all rows", len(Xtr) + len(Xte) == len(d["X"]))
    check(
        "test folds are exactly the reserved ones",
        set(d["fold"][d["fold"].isin(config.TEST_FOLDS)].unique()) == set(config.TEST_FOLDS),
    )
    check("no fold appears in both sides", not set(ftr.unique()) & set(config.TEST_FOLDS))
    check("column order identical across split", list(Xtr.columns) == list(Xte.columns))
    return Xtr, Xte, ytr, yte


def test_bundle():
    print("\nModel bundle")
    check("bundle file exists", config.MODEL_BUNDLE.exists())
    if not config.MODEL_BUNDLE.exists():
        return None
    b = joblib.load(config.MODEL_BUNDLE)
    check("both variants present", set(b["variants"]) == {"all_features", "no_race_features"})
    for k, v in b["variants"].items():
        check(f"{k}: beats the mean baseline", v["test_metrics"]["r2"] > 0.3,
              f"R²={v['test_metrics']['r2']:.3f}")
        check(f"{k}: leaderboard sorted by CV RMSE",
              all(a["cv_rmse"] <= z["cv_rmse"] + 1e-12
                  for a, z in zip(v["leaderboard"], v["leaderboard"][1:])))
        check(f"{k}: winner is the CV leader",
              v["best_model_name"] == v["leaderboard"][0]["model"])
    nr = b["variants"]["no_race_features"]
    check(
        "race features truly absent from the no-race variant",
        not (set(nr["feature_names"]) & set(config.RACE_FEATURES)),
    )
    return b


def test_predictions(b, Xte, yte):
    print("\nPrediction behaviour")
    v = b["variants"]["all_features"]
    model, fnames = v["model"], v["feature_names"]
    Xte_v, _ = feat_mod.select_features(Xte, v["exclude_race"])

    p = clip_to_range(model.predict(Xte_v))
    check("one prediction per row", len(p) == len(Xte_v))
    check("predictions inside [0, 1]", bool((p >= 0).all() and (p <= 1).all()),
          f"[{p.min():.3f}, {p.max():.3f}]")
    check("predictions are not constant", float(np.std(p)) > 0.01)

    m = regression_metrics(yte, p)
    check("held-out metrics reproduce the bundle",
          abs(m["rmse"] - v["test_metrics"]["rmse"]) < 1e-9,
          f"RMSE={m['rmse']:.4f}")

    # Determinism: the same input must give the same answer twice.
    check("deterministic", np.allclose(p, clip_to_range(model.predict(Xte_v))))
    return model, fnames


def test_schema_robustness(model, fnames, Xte):
    """The app's batch path must survive shuffled, partial and dirty CSVs."""
    print("\nSchema robustness (batch CSV path)")
    base = Xte[fnames].head(50).copy()
    ref = clip_to_range(model.predict(feat_mod.align_to_schema(base, fnames)))

    shuffled = base[list(np.random.default_rng(0).permutation(fnames))]
    got = clip_to_range(model.predict(feat_mod.align_to_schema(shuffled, fnames)))
    check("column order does not matter", np.allclose(ref, got))

    extra = base.copy()
    extra["communityname"] = "Somewhere"
    extra["totally_unrelated"] = 123.4
    got = clip_to_range(
        model.predict(feat_mod.align_to_schema(
            extra.drop(columns=["communityname"]), fnames))
    )
    check("extra columns are ignored", np.allclose(ref, got))

    partial = base.drop(columns=fnames[:5])
    aligned = feat_mod.align_to_schema(partial, fnames)
    check("missing columns become NaN for the imputer",
          aligned[fnames[:5]].isna().all().all())
    got = clip_to_range(model.predict(aligned))
    check("still predicts with columns missing", len(got) == len(partial)
          and bool(np.isfinite(got).all()))

    diff = feat_mod.schema_diff(partial, fnames)
    check("schema_diff reports the gap", set(diff["missing"]) == set(fnames[:5]))

    # A CSV round-trip carrying the '?' marker, as the raw UCI file does.
    dirty = base.head(5).astype(object).copy()
    dirty.iloc[0, 0] = "?"
    dirty.iloc[1, 1] = ""
    buf = io.StringIO()
    dirty.to_csv(buf, index=False)
    reread = pd.read_csv(io.StringIO(buf.getvalue()))
    numeric = data_mod.to_numeric_features(reread)
    check("'?' parsed as missing, not as a string", bool(np.isnan(numeric.iloc[0, 0])))
    got = clip_to_range(model.predict(feat_mod.align_to_schema(numeric, fnames)))
    check("dirty CSV still scores", bool(np.isfinite(got).all()))


def test_explanations(model, fnames, Xte, yte, b):
    print("\nExplanations")
    v = b["variants"]["all_features"]
    imp = v["importances"]
    check("importance covers every feature", len(imp) == len(fnames))
    check("importances are sorted descending",
          bool((imp["importance"].diff().dropna() <= 1e-12).all()))
    check("top feature is a plausible driver", imp.iloc[0]["importance"] > 0,
          f"{imp.iloc[0]['feature']} ({imp.iloc[0]['importance']:.4f})")

    medians = pd.Series(v["feature_medians"]).reindex(fnames).to_frame().T.astype("float64")
    row = Xte[fnames].head(1).reset_index(drop=True)
    contrib = explain_mod.local_contributions(model, row, medians, top_n=10)
    check("local contributions returned", len(contrib) == 10)
    check("contributions are finite", bool(np.isfinite(contrib["contribution"]).all()))

    coefs = explain_mod.linear_coefficients(model, fnames)
    if coefs is not None:
        check("coefficient per feature", len(coefs) == len(fnames))


def test_descriptions(d):
    print("\nMetadata")
    desc = data_mod.variable_descriptions()
    missing = [c for c in d["X"].columns if c not in desc]
    check("every feature has a description", not missing, f"missing: {missing[:5]}")
    check("descriptions are non-trivial",
          all(len(desc[c]) > 5 for c in d["X"].columns))


def main() -> int:
    print("=" * 68)
    print("Communities & Crime — pipeline tests")
    print("=" * 68)

    d = test_cleaning()
    Xtr, Xte, ytr, yte = test_split(d)
    b = test_bundle()
    if b is not None:
        model, fnames = test_predictions(b, Xte, yte)
        test_schema_robustness(model, fnames, Xte)
        test_explanations(model, fnames, Xte, yte, b)
    test_descriptions(d)

    print("\n" + "=" * 68)
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
