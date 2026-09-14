"""Streamlit frontend for the Communities & Crime prediction pipeline.

Run with:  streamlit run app.py

The app is read-only over the artifacts produced by `python -m src.train`;
it never fits a model itself, so what you see here is exactly the model
that was evaluated on the held-out folds.
"""

from __future__ import annotations

import io

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config, data as data_mod, explain as explain_mod, features as feat_mod

st.set_page_config(
    page_title="Communities & Crime — ML Pipeline",
    page_icon="🏘️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT = "#4C78A8"
SEQ = px.colors.sequential.Blues


# ==========================================================================
# Cached loaders
# ==========================================================================
@st.cache_resource(show_spinner="Loading trained model bundle…")
def load_bundle(mtime: float):
    """Load the joblib bundle. `mtime` busts the cache after retraining."""
    return joblib.load(config.MODEL_BUNDLE)


@st.cache_data(show_spinner="Loading dataset…")
def load_dataset():
    d = data_mod.load_clean()
    return d["X"], d["y"], d["fold"], d["meta"], d["report"]


@st.cache_data(show_spinner=False)
def load_descriptions():
    return data_mod.variable_descriptions()


@st.cache_data(show_spinner=False)
def load_uci_metadata():
    return data_mod.dataset_metadata()


def bundle_or_stop():
    if not config.MODEL_BUNDLE.exists():
        st.title("🏘️ Communities & Crime — ML Pipeline")
        st.error("No trained model found yet.")
        st.markdown(
            "Train the pipeline first, then reload this page:\n\n"
            "```bash\npython -m src.train\n```\n\n"
            "Use `python -m src.train --quick` for a faster run with smaller "
            "hyper-parameter grids."
        )
        st.stop()
    return load_bundle(config.MODEL_BUNDLE.stat().st_mtime)


# ==========================================================================
# Small presentation helpers
# ==========================================================================
def pct_rank(value: float, series: pd.Series) -> float:
    """Percentile of `value` within `series`, 0-100."""
    return float((series < value).mean() * 100)


def risk_band(percentile: float) -> tuple[str, str]:
    if percentile < 25:
        return "Lower than most communities", "#2E7D32"
    if percentile < 50:
        return "Below the national median", "#7CB342"
    if percentile < 75:
        return "Above the national median", "#F9A825"
    if percentile < 90:
        return "High relative to most communities", "#EF6C00"
    return "Among the highest in the dataset", "#C62828"


def metric_row(metrics: dict, prefix: str = "") -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{prefix}RMSE", f"{metrics['rmse']:.4f}")
    c2.metric(f"{prefix}MAE", f"{metrics['mae']:.4f}")
    c3.metric(f"{prefix}R²", f"{metrics['r2']:.3f}")
    sp = metrics.get("spearman")
    c4.metric(f"{prefix}Spearman ρ", f"{sp:.3f}" if sp is not None else "n/a")


def describe(col: str, descriptions: dict) -> str:
    return descriptions.get(col, "—")


def binned_trend(x: pd.Series, y: pd.Series, bins: int = 10) -> dict | None:
    """Mean target per quantile bin of `x` — a dependency-free trend line.

    Plotly's built-in `trendline="lowess"` needs statsmodels; binning by
    quantile gives the same read on shape (including non-linearity) using
    only numpy, and each point is directly interpretable as "the average
    outcome for communities in this decile of the feature".
    """
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(d) < bins * 5 or d["x"].nunique() < bins:
        return None
    try:
        d["bin"] = pd.qcut(d["x"], bins, duplicates="drop")
    except ValueError:
        return None
    g = d.groupby("bin", observed=True).agg(x=("x", "mean"), y=("y", "mean"))
    return {"x": g["x"].to_numpy(), "y": g["y"].to_numpy()}


def ethics_note(expanded: bool = False) -> None:
    with st.expander("⚠️ Read this before interpreting any prediction", expanded=expanded):
        st.markdown(
            """
This dataset pairs 1990 US Census socio-economic data with 1995 FBI Uniform
Crime Report figures, and it is one of the most-studied examples in the
algorithmic-fairness literature. A few things follow from that:

- **The target is reported crime, not crime.** `ViolentCrimesPerPop` measures
  what was *recorded* by police agencies. Reporting rates and enforcement
  intensity vary between communities, so the label carries the biases of the
  reporting process as well as the underlying phenomenon.
- **Correlation is not a mechanism.** The model finds statistical association
  in 1990s aggregates. It says nothing about what *causes* crime, and moving a
  slider does not simulate a policy intervention.
- **The data is ~30 years old** and describes communities of 1990. It should
  not be used to make decisions about any community today.
- **Race and ethnicity columns are present in the raw data.** Use the variant
  selector in the sidebar to compare a model trained with them against one
  trained without — the performance gap is small, which is itself the point
  worth discussing.
- **This is a coursework artifact.** It is not fit for policing, resource
  allocation, lending, insurance, or any decision about real people or places.
"""
        )


# ==========================================================================
# Sidebar
# ==========================================================================
def sidebar(bundle) -> tuple[str, dict]:
    st.sidebar.title("🏘️ Communities & Crime")
    st.sidebar.caption("UCI dataset 183 · end-to-end regression pipeline")

    labels = {k: v["label"] for k, v in bundle["variants"].items()}
    keys = list(labels)
    choice = st.sidebar.radio(
        "Feature set",
        keys,
        format_func=lambda k: labels[k].capitalize(),
        help=(
            "Two independently trained models. 'Without race features' drops "
            "the race, ethnicity and immigration columns listed in "
            "src/config.py so you can see what they contribute."
        ),
    )
    variant = bundle["variants"][choice]

    st.sidebar.divider()
    st.sidebar.markdown(
        f"**Selected model**  \n{variant['best_model_name']}  \n"
        f"<span style='color:#888'>{len(variant['feature_names'])} features · "
        f"test R² {variant['test_metrics']['r2']:.3f}</span>",
        unsafe_allow_html=True,
    )
    st.sidebar.caption(
        f"Trained {pd.Timestamp(bundle['created_utc']).strftime('%Y-%m-%d %H:%M UTC')}"
    )
    return choice, variant


# ==========================================================================
# Page: Overview
# ==========================================================================
def page_overview(bundle, variant):
    st.title("Predicting violent-crime rates from community characteristics")
    st.markdown(
        "An end-to-end regression pipeline over **UCI dataset 183 "
        "(Communities and Crime)** — ingestion, cleaning, model selection, "
        "held-out evaluation and serving."
    )

    ds, tgt = bundle["dataset"], bundle["target_stats"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Communities", f"{ds['n_rows_kept']:,}")
    c2.metric("Predictors used", ds["n_features_kept"])
    c3.metric("Held-out test rows", variant["n_test"])
    c4.metric("Test R²", f"{variant['test_metrics']['r2']:.3f}")

    ethics_note()

    st.subheader("What the pipeline does")
    st.markdown(
        f"""
| Stage | What happens |
|---|---|
| **1 · Ingest** | `fetch_ucirepo(id=183)` → cached to `data/communities_raw.parquet`. Network is touched once. |
| **2 · Clean** | `?` → `NaN`; every predictor coerced to float; 5 non-predictive ID columns dropped. |
| **3 · Prune** | {len(ds['dropped_high_missing'])} columns missing >{int(ds['max_missing_fraction'] * 100)}% of values dropped rather than imputed. |
| **4 · Split** | The dataset's own `fold` column: folds {ds['test_folds']} held out, folds 1–8 used for cross-validation. |
| **5 · Preprocess** | Median imputation, then standardisation for the linear models — fit *inside* each CV fold, so no leakage. |
| **6 · Select** | 7 candidate models, hyper-parameters grid-searched on 8-fold CV, ranked by CV RMSE. |
| **7 · Evaluate** | The winner is scored **once** on the held-out folds. |
| **8 · Serve** | This app loads the saved bundle; predictions are clipped to the target's valid [0, 1] range. |
"""
    )

    st.subheader("Why these columns were dropped")
    cA, cB = st.columns([2, 3])
    with cA:
        st.markdown(
            f"**Non-predictive identifiers ({len(ds['dropped_non_predictive'])})**  \n"
            + ", ".join(f"`{c}`" for c in ds["dropped_non_predictive"])
        )
        st.caption(
            "Flagged non-predictive by the dataset authors. `fold` is kept "
            "aside to build the train/test split."
        )
    with cB:
        st.markdown(f"**Mostly-missing columns ({len(ds['dropped_high_missing'])})**")
        st.caption(
            "These are the LEMAS police-survey fields — 84% of communities "
            "did not respond. Imputing them would fabricate the majority of "
            "their values, so they are removed instead."
        )
        with st.expander("Show the dropped columns"):
            st.code("\n".join(ds["dropped_high_missing"]))

    st.subheader("The target: ViolentCrimesPerPop")
    _, y, _, _, _ = load_dataset()
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = px.histogram(
            y, nbins=50, labels={"value": "ViolentCrimesPerPop (normalised)"},
            color_discrete_sequence=[ACCENT],
        )
        fig.update_layout(
            showlegend=False, height=340, margin=dict(t=30, b=10),
            yaxis_title="Communities", title="Right-skewed, bounded at 0 and 1",
        )
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.markdown(
            f"""
The target is the number of violent crimes per 100 000 people, **min–max
normalised to [0, 1]** by the dataset authors. The original per-100K scale
is not recoverable from the published file, so this app reports the
normalised value alongside a **percentile** against all
{ds['n_rows_kept']:,} communities, which is the interpretable part.

- Mean **{tgt['mean']:.3f}**, median **{float(y.median()):.3f}**
- Std **{tgt['std']:.3f}**, skew **{tgt['skew']:.2f}**
- {int((y >= 1.0).sum())} communities sit at the 1.0 ceiling

That right skew is why a √y-transformed model is among the candidates.
"""
        )

    meta = load_uci_metadata()
    if meta.get("abstract"):
        with st.expander("UCI dataset abstract & citation"):
            st.write(meta["abstract"])
            cite = (meta.get("additional_info") or {}).get("citation")
            if cite:
                st.caption(cite)
            if meta.get("repository_url"):
                st.markdown(f"[View on UCI ML Repository]({meta['repository_url']})")


# ==========================================================================
# Page: Data explorer
# ==========================================================================
def page_data(bundle, variant):
    st.title("Data explorer")
    X, y, fold, meta, report = load_dataset()
    descriptions = load_descriptions()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Feature browser", "Relationship to target", "Correlation", "Data quality"]
    )

    with tab1:
        st.caption(
            "All predictors were min–max normalised to [0, 1] by the dataset "
            "authors, so values are comparable across columns but are not in "
            "their original units."
        )
        table = pd.DataFrame(
            {
                "feature": X.columns,
                "description": [describe(c, descriptions) for c in X.columns],
                "mean": X.mean().to_numpy(),
                "std": X.std().to_numpy(),
                "min": X.min().to_numpy(),
                "median": X.median().to_numpy(),
                "max": X.max().to_numpy(),
                "missing %": (X.isna().mean() * 100).to_numpy(),
                "corr with target": X.corrwith(y).to_numpy(),
            }
        )
        q = st.text_input("Filter features", placeholder="e.g. income, Pct, Parent")
        if q:
            mask = table["feature"].str.contains(q, case=False) | table[
                "description"
            ].str.contains(q, case=False)
            table = table[mask]
        st.dataframe(
            table.style.format(
                {
                    "mean": "{:.3f}", "std": "{:.3f}", "min": "{:.3f}",
                    "median": "{:.3f}", "max": "{:.3f}", "missing %": "{:.2f}",
                    "corr with target": "{:+.3f}",
                }
            ).background_gradient(subset=["corr with target"], cmap="RdBu_r", vmin=-1, vmax=1),
            width="stretch",
            height=460,
        )

    with tab2:
        corr = X.corrwith(y).sort_values()
        c1, c2 = st.columns([2, 3])
        with c1:
            top = pd.concat([corr.head(12), corr.tail(12)]).sort_values()
            fig = px.bar(
                x=top.to_numpy(), y=top.index, orientation="h",
                labels={"x": "Pearson correlation with target", "y": ""},
                color=top.to_numpy(), color_continuous_scale="RdBu_r",
                range_color=[-0.8, 0.8],
            )
            fig.update_layout(
                height=560, coloraxis_showscale=False,
                margin=dict(l=10, t=40, b=10), title="Strongest linear associations",
            )
            st.plotly_chart(fig, width="stretch")
        with c2:
            options = list(X.columns)
            default = corr.abs().idxmax()
            col = st.selectbox(
                "Inspect a feature against the target",
                options, index=options.index(default),
            )
            st.caption(describe(col, descriptions))
            plot_df = pd.DataFrame(
                {col: X[col], "ViolentCrimesPerPop": y, "community": meta["communityname"]}
            ).dropna()
            fig = px.scatter(
                plot_df, x=col, y="ViolentCrimesPerPop", hover_name="community",
                opacity=0.45, color_discrete_sequence=[ACCENT],
            )
            trend = binned_trend(plot_df[col], plot_df["ViolentCrimesPerPop"])
            if trend is not None:
                fig.add_trace(
                    go.Scatter(
                        x=trend["x"], y=trend["y"], mode="lines+markers",
                        name="mean target per decile", line=dict(color="#C62828", width=3),
                    )
                )
            fig.update_layout(
                height=490, margin=dict(t=40, b=10),
                legend=dict(orientation="h", y=1.08, x=0),
            )
            st.plotly_chart(fig, width="stretch")
            st.metric("Pearson r", f"{float(corr[col]):+.3f}")

    with tab3:
        st.caption(
            "Census columns are heavily collinear — several income, poverty "
            "and family-structure blocks measure near-identical things. That "
            "is why L2 regularisation performs well and why individual "
            "coefficients should not be read as independent effects."
        )
        n = st.slider("Show top N features by |correlation with target|", 10, 50, 25, 5)
        cols = X.corrwith(y).abs().sort_values(ascending=False).head(n).index.tolist()
        cmat = X[cols].corr()
        fig = px.imshow(
            cmat, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto"
        )
        fig.update_layout(height=760, margin=dict(t=30))
        st.plotly_chart(fig, width="stretch")

    with tab4:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Missing values after cleaning")
            resid = report["residual_missing_cols"]
            if resid:
                st.dataframe(
                    pd.DataFrame(
                        {"feature": list(resid), "missing %": [v * 100 for v in resid.values()]}
                    ).style.format({"missing %": "{:.3f}"}),
                    width="stretch", hide_index=True,
                )
                st.caption("Filled by median imputation inside the pipeline.")
            else:
                st.success("No missing values remain.")
            st.subheader("Dropped for excessive missingness")
            st.caption(
                f"{len(report['dropped_high_missing'])} columns exceeded the "
                f"{int(report['max_missing_fraction'] * 100)}% threshold."
            )
            st.code("\n".join(report["dropped_high_missing"]) or "none")
        with c2:
            st.subheader("Communities per state")
            counts = (
                meta["state"].value_counts().rename_axis("state").reset_index(name="communities")
            )
            fig = px.bar(
                counts.head(25), x="state", y="communities",
                color_discrete_sequence=[ACCENT],
            )
            fig.update_layout(height=330, margin=dict(t=30, b=10))
            st.plotly_chart(fig, width="stretch")

            st.subheader("Mean target by state")
            by_state = (
                pd.DataFrame({"state": meta["state"], "y": y})
                .dropna().groupby("state")["y"].agg(["mean", "count"])
                .query("count >= 10").sort_values("mean", ascending=False)
            )
            fig = px.bar(
                by_state.reset_index(), x="state", y="mean", color="mean",
                color_continuous_scale=SEQ, labels={"mean": "mean target"},
            )
            fig.update_layout(height=330, coloraxis_showscale=False, margin=dict(t=30, b=10))
            st.plotly_chart(fig, width="stretch")
            st.caption("States with at least 10 communities in the sample.")


# ==========================================================================
# Page: Model performance
# ==========================================================================
def page_performance(bundle, variant):
    st.title("Model performance")
    st.caption(
        f"Model selection used 8-fold cross-validation on folds 1–8. "
        f"The held-out folds {bundle['dataset']['test_folds']} "
        f"({variant['n_test']} communities) were scored once, after selection."
    )

    st.subheader(f"Held-out test performance — {variant['best_model_name']}")
    metric_row(variant["test_metrics"])
    base = next(
        (r for r in variant["leaderboard"] if r["model"].startswith("Baseline")), None
    )
    if base:
        lift = (1 - variant["test_metrics"]["rmse"] / base["test_rmse"]) * 100
        st.caption(
            f"RMSE is {lift:.1f}% lower than the mean-only baseline "
            f"({base['test_rmse']:.4f}). An RMSE of "
            f"{variant['test_metrics']['rmse']:.3f} means a typical prediction "
            f"lands within ±{variant['test_metrics']['rmse']:.3f} of the true "
            "normalised rate."
        )

    st.subheader("Candidate leaderboard")
    lb = pd.DataFrame(variant["leaderboard"])
    # The baseline has no defined Spearman (constant predictions); keep the
    # column numeric so Styler renders it via na_rep rather than as "None".
    # Streamlit's grid renders NaN as "None" regardless of Styler na_rep, so
    # pre-format this column as text.
    lb["test_spearman"] = pd.to_numeric(lb["test_spearman"], errors="coerce").map(
        lambda v: "—" if pd.isna(v) else f"{v:.3f}"
    )
    show = lb[
        ["model", "cv_rmse", "cv_rmse_std", "test_rmse", "test_mae", "test_r2",
         "test_spearman", "fit_seconds", "note"]
    ].rename(
        columns={
            "cv_rmse": "CV RMSE", "cv_rmse_std": "CV ±", "test_rmse": "Test RMSE",
            "test_mae": "Test MAE", "test_r2": "Test R²",
            "test_spearman": "Test ρ", "fit_seconds": "Fit (s)",
        }
    )
    st.dataframe(
        show.style.format(
            {
                "CV RMSE": "{:.4f}", "CV ±": "{:.4f}", "Test RMSE": "{:.4f}",
                "Test MAE": "{:.4f}", "Test R²": "{:.3f}", "Fit (s)": "{:.1f}",
            },
        ).background_gradient(subset=["CV RMSE"], cmap="Blues_r"),
        width="stretch", hide_index=True,
    )
    st.caption(
        "Ranked by CV RMSE — the only criterion used for selection. Test "
        "columns are shown for every candidate for transparency, but they "
        "played no part in choosing the winner."
    )

    fig = go.Figure()
    fig.add_bar(
        x=lb["model"], y=lb["cv_rmse"],
        error_y=dict(type="data", array=lb["cv_rmse_std"]),
        name="CV RMSE (8-fold)", marker_color=ACCENT,
    )
    fig.add_bar(x=lb["model"], y=lb["test_rmse"], name="Held-out test RMSE",
                marker_color="#F58518")
    fig.update_layout(
        barmode="group", height=440, yaxis_title="RMSE (lower is better)",
        margin=dict(t=95), legend=dict(orientation="h", y=1.10, x=0),
        title=dict(text="Cross-validated vs held-out error", y=0.97, yanchor="top"),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "The candidates cluster tightly. With error bars of this size the "
        "top few are statistically indistinguishable — worth remembering "
        "before declaring a winner."
    )

    st.subheader("Prediction diagnostics")
    preds = variant["test_predictions"].copy()
    t1, t2, t3 = st.tabs(["Actual vs predicted", "Residuals", "Largest errors"])

    with t1:
        fig = px.scatter(
            preds, x="actual", y="predicted", hover_name="community",
            hover_data=["state"], opacity=0.6, color_discrete_sequence=[ACCENT],
        )
        fig.add_shape(
            type="line", x0=0, y0=0, x1=1, y1=1,
            line=dict(dash="dash", color="#C62828"),
        )
        fig.update_layout(
            height=540, xaxis_title="Actual", yaxis_title="Predicted",
            title="Points on the dashed line are perfect predictions",
            margin=dict(t=50),
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Predictions compress toward the middle: the model under-predicts "
            "the most violent communities and over-predicts the safest. That "
            "is the expected behaviour of a squared-error model on a skewed, "
            "bounded target."
        )

    with t2:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(
                preds, x="predicted", y="residual", hover_name="community",
                opacity=0.6, color_discrete_sequence=[ACCENT],
            )
            fig.add_hline(y=0, line_dash="dash", line_color="#C62828")
            fig.update_layout(
                height=420, title="Residual vs predicted",
                yaxis_title="actual − predicted", margin=dict(t=50),
            )
            st.plotly_chart(fig, width="stretch")
        with c2:
            fig = px.histogram(
                preds, x="residual", nbins=40, color_discrete_sequence=[ACCENT]
            )
            fig.add_vline(x=0, line_dash="dash", line_color="#C62828")
            fig.update_layout(height=420, title="Residual distribution", margin=dict(t=50))
            st.plotly_chart(fig, width="stretch")
        st.caption(
            f"Residual mean {preds['residual'].mean():+.4f}, "
            f"std {preds['residual'].std():.4f}. A fan shape widening to the "
            "right indicates heteroscedasticity — the model is less certain "
            "about high-crime communities."
        )

    with t3:
        worst = preds.reindex(
            preds["residual"].abs().sort_values(ascending=False).index
        ).head(20)
        st.dataframe(
            worst.style.format(
                {"actual": "{:.3f}", "predicted": "{:.3f}", "residual": "{:+.3f}"}
            ).background_gradient(subset=["residual"], cmap="RdBu_r"),
            width="stretch", hide_index=True,
        )
        st.caption(
            "Where the model fails hardest. Positive residuals are "
            "under-predictions."
        )


# ==========================================================================
# Page: Explainability
# ==========================================================================
def page_explain(bundle, variant):
    st.title("What drives the predictions")
    descriptions = load_descriptions()

    imp = variant["importances"].copy()
    st.subheader("Permutation importance (held-out folds)")
    st.caption(
        "Each feature is shuffled in the test set and the increase in RMSE is "
        "recorded. Larger means the model relies on it more. Measured on "
        "held-out data, so this reflects what generalises — not what was "
        "memorised."
    )
    n = st.slider("Features to show", 5, 40, 20, 5)
    top = imp.head(n).iloc[::-1]
    fig = px.bar(
        top, x="importance", y="feature", orientation="h", error_x="std",
        color="importance", color_continuous_scale=SEQ,
    )
    fig.update_layout(
        height=28 * n + 140, coloraxis_showscale=False,
        xaxis_title="Increase in RMSE when shuffled", yaxis_title="",
        margin=dict(l=10, t=30),
    )
    st.plotly_chart(fig, width="stretch")

    st.dataframe(
        imp.head(n).assign(description=lambda d: d["feature"].map(lambda c: describe(c, descriptions)))
        .style.format({"importance": "{:.5f}", "std": "{:.5f}"}),
        width="stretch", hide_index=True,
    )

    coefs = explain_mod.linear_coefficients(variant["model"], variant["feature_names"])
    if coefs is not None:
        st.subheader("Standardised coefficients")
        st.caption(
            "The winning model is linear, so it has a readable coefficient per "
            "feature. Because the pipeline standardises inputs first, these are "
            "comparable across features: each is the change in the (transformed) "
            "target per one standard deviation of that predictor. **Collinearity "
            "makes individual coefficients unstable — read them as a group, not "
            "as isolated effects.**"
        )
        show = coefs.head(n).iloc[::-1]
        fig = px.bar(
            show, x="coefficient", y="feature", orientation="h",
            color="coefficient", color_continuous_scale="RdBu_r",
        )
        fig.update_layout(
            height=28 * n + 140, coloraxis_showscale=False, yaxis_title="",
            margin=dict(l=10, t=30),
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.info(
            "The winning model is a tree ensemble, which has no single "
            "coefficient vector — permutation importance above is the "
            "model-agnostic equivalent."
        )

    st.subheader("Does dropping race features cost accuracy?")
    rows = []
    for key, v in bundle["variants"].items():
        rows.append(
            {
                "Variant": v["label"].capitalize(),
                "Model": v["best_model_name"],
                "Features": len(v["feature_names"]),
                "CV RMSE": v["leaderboard"][0]["cv_rmse"],
                "Test RMSE": v["test_metrics"]["rmse"],
                "Test R²": v["test_metrics"]["r2"],
            }
        )
    comp = pd.DataFrame(rows)
    st.dataframe(
        comp.style.format(
            {"CV RMSE": "{:.4f}", "Test RMSE": "{:.4f}", "Test R²": "{:.3f}"}
        ),
        width="stretch", hide_index=True,
    )
    d_r2 = comp["Test R²"].iloc[0] - comp["Test R²"].iloc[1]
    st.caption(
        f"Removing {len(config.RACE_FEATURES)} race, ethnicity and immigration "
        f"columns changes held-out R² by {d_r2:+.3f}. The remaining census "
        "columns are correlated with the removed ones, so a model without "
        "explicit race features is **not** thereby a race-blind model — a "
        "central point in the fairness literature on this dataset."
    )


# ==========================================================================
# Page: Predict
# ==========================================================================
def page_predict(bundle, variant):
    st.title("Make a prediction")
    ethics_note()

    X, y, fold, meta, _ = load_dataset()
    descriptions = load_descriptions()
    model = variant["model"]
    fnames = variant["feature_names"]
    medians = pd.Series(variant["feature_medians"]).reindex(fnames)

    tab_single, tab_batch = st.tabs(["Single community", "Batch scoring (CSV)"])

    # ---------------- single ----------------
    with tab_single:
        c1, c2 = st.columns([1, 1])
        with c1:
            mode = st.radio(
                "Starting point",
                ["A real community from the dataset", "A typical (median) community"],
                help="Pick a baseline, then adjust individual features below.",
            )
        with c2:
            k = st.slider(
                "Adjustable features", 4, 20, 8,
                help="The most important features are exposed as sliders; the "
                     "rest stay at the starting community's values.",
            )

        if mode.startswith("A real"):
            names = meta["communityname"].fillna("(unnamed)") + ", " + meta["state"].fillna("??")
            pick = st.selectbox("Community", options=list(names.index), format_func=lambda i: names.iloc[i])
            base_row = X.iloc[[pick]][fnames].copy().reset_index(drop=True)
            true_val = float(y.iloc[pick])
            in_test = bool(fold.iloc[pick] in config.TEST_FOLDS)
        else:
            base_row = medians.to_frame().T.astype("float64").reset_index(drop=True)
            true_val = None
            in_test = False

        st.markdown("##### Adjust features")
        st.caption(
            "Every predictor is normalised to [0, 1], where 0 is the lowest "
            "and 1 the highest value observed across all communities."
        )
        top_feats = variant["importances"]["feature"].head(k).tolist()
        row = base_row.copy()
        cols = st.columns(2)
        for i, f in enumerate(top_feats):
            start = float(row.iloc[0][f]) if pd.notna(row.iloc[0][f]) else float(medians[f])
            with cols[i % 2]:
                row.iloc[0, row.columns.get_loc(f)] = st.slider(
                    f, 0.0, 1.0, min(max(start, 0.0), 1.0), 0.01,
                    help=describe(f, descriptions),
                )

        pred = float(np.clip(model.predict(row[fnames])[0], *config.TARGET_RANGE))
        percentile = pct_rank(pred, y)
        band, colour = risk_band(percentile)

        st.divider()
        m1, m2, m3 = st.columns([1, 1, 2])
        m1.metric(
            "Predicted rate", f"{pred:.3f}",
            help="Normalised to [0, 1] across all communities in the dataset.",
        )
        m2.metric("Percentile", f"{percentile:.0f}th")
        with m3:
            st.markdown(
                f"<div style='padding:.6rem 1rem;border-left:5px solid {colour};"
                f"background:rgba(128,128,128,.08)'><b style='color:{colour}'>{band}</b><br>"
                f"<span style='font-size:.85em;color:#888'>Higher than {percentile:.0f}% "
                f"of the {len(y):,} communities in the dataset.</span></div>",
                unsafe_allow_html=True,
            )

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=pred,
                number={"valueformat": ".3f"},
                gauge={
                    "axis": {"range": [0, 1]},
                    "bar": {"color": colour},
                    "steps": [
                        {"range": [0, float(y.quantile(0.25))], "color": "#E8F5E9"},
                        {"range": [float(y.quantile(0.25)), float(y.quantile(0.5))], "color": "#F1F8E9"},
                        {"range": [float(y.quantile(0.5)), float(y.quantile(0.75))], "color": "#FFF8E1"},
                        {"range": [float(y.quantile(0.75)), 1], "color": "#FFEBEE"},
                    ],
                    "threshold": {
                        "line": {"color": "#333", "width": 3},
                        "value": float(y.median()),
                    },
                },
                title={"text": "Predicted rate vs dataset quartiles<br>"
                               "<span style='font-size:.7em'>black line = dataset median</span>"},
            )
        )
        fig.update_layout(height=300, margin=dict(t=70, b=10))
        st.plotly_chart(fig, width="stretch")

        if true_val is not None:
            e1, e2, e3 = st.columns(3)
            e1.metric("Actual (recorded)", f"{true_val:.3f}")
            e2.metric("Error", f"{pred - true_val:+.3f}")
            e3.metric(
                "Source rows",
                "Test fold" if in_test else "Train folds",
                help="Predictions on training rows are optimistic — the model "
                     "saw them during fitting.",
            )
            if not in_test:
                st.info(
                    "This community was in the training data, so the error "
                    "shown flatters the model. The honest error estimate is "
                    "the held-out RMSE on the Model performance page.",
                    icon="ℹ️",
                )
            st.caption(
                "Sliders you moved change the prediction but not the recorded "
                "actual, so a large error after editing is expected."
            )

        st.markdown("##### Why this prediction?")
        st.caption(
            "Each feature is moved from the typical community's value to this "
            "one's, in isolation, and the change in prediction is recorded. "
            "Interactions are ignored, so contributions are directional rather "
            "than an exact decomposition."
        )
        with st.spinner("Computing contributions…"):
            contrib = explain_mod.local_contributions(
                model, row[fnames], medians.to_frame().T.astype("float64"), top_n=12
            )
        if contrib.empty:
            st.info("No contributions to show.")
        else:
            contrib = contrib.iloc[::-1]
            fig = px.bar(
                contrib, x="contribution", y="feature", orientation="h",
                color="contribution", color_continuous_scale="RdBu_r",
                color_continuous_midpoint=0,
                hover_data=["value", "baseline_value"],
            )
            fig.update_layout(
                height=440, coloraxis_showscale=False, yaxis_title="",
                xaxis_title="Change in predicted rate vs a typical community",
                margin=dict(l=10, t=30),
            )
            st.plotly_chart(fig, width="stretch")
            st.dataframe(
                contrib.iloc[::-1]
                .assign(description=lambda d: d["feature"].map(lambda c: describe(c, descriptions)))
                [["feature", "value", "baseline_value", "contribution", "description"]]
                .style.format(
                    {"value": "{:.3f}", "baseline_value": "{:.3f}", "contribution": "{:+.4f}"}
                ),
                width="stretch", hide_index=True,
            )

    # ---------------- batch ----------------
    with tab_batch:
        st.markdown(
            "Upload a CSV of communities to score. Columns are matched by "
            "name — order does not matter, extra columns are ignored, and any "
            "missing predictor is median-imputed by the pipeline."
        )

        template = X[fnames].head(20).copy()
        template.insert(0, "communityname", meta["communityname"].head(20).to_numpy())
        st.download_button(
            "⬇️ Download a 20-row template CSV",
            template.to_csv(index=False).encode(),
            file_name="communities_template.csv",
            mime="text/csv",
            help="Real rows from the dataset, in the exact expected schema.",
        )

        up = st.file_uploader("CSV file", type=["csv"])
        if up is None:
            st.info(
                "Upload a file, or download the template above to see the "
                "schema. There is also a ready-made demo file at "
                "`data/sample_communities.csv` — 60 **held-out** communities "
                "spanning the full range of the target, with the true values "
                "included so the app can score itself against them.",
                icon="📄",
            )
            return

        try:
            raw = pd.read_csv(up)
        except Exception as e:
            st.error(f"Could not read that CSV: {e}")
            return

        st.write(f"Read **{len(raw):,} rows × {raw.shape[1]} columns**.")
        numeric = data_mod.to_numeric_features(
            raw.drop(columns=[c for c in ("communityname", "state") if c in raw.columns])
        )
        diff = feat_mod.schema_diff(numeric, fnames)

        c1, c2, c3 = st.columns(3)
        c1.metric("Matched predictors", f"{len(diff['matched'])}/{len(fnames)}")
        c2.metric("Missing", len(diff["missing"]))
        c3.metric("Ignored extras", len(diff["extra"]))

        if diff["missing"]:
            with st.expander(f"⚠️ {len(diff['missing'])} expected columns are absent"):
                st.caption(
                    "These will be filled with the training median, which "
                    "weakens the prediction. The more that are missing, the "
                    "less the output means."
                )
                st.code("\n".join(diff["missing"]))
        if len(diff["matched"]) == 0:
            st.error(
                "No predictor columns matched the training schema — every "
                "prediction would just be the median community. Check the "
                "template for expected column names."
            )
            return

        aligned = feat_mod.align_to_schema(numeric, fnames)
        preds = np.clip(model.predict(aligned), *config.TARGET_RANGE)

        out = pd.DataFrame()
        for c in ("communityname", "state"):
            if c in raw.columns:
                out[c] = raw[c]
        out["predicted"] = preds
        out["percentile"] = [pct_rank(p, y) for p in preds]
        out["band"] = [risk_band(p)[0] for p in out["percentile"]]
        if config.TARGET in raw.columns:
            actual = pd.to_numeric(raw[config.TARGET], errors="coerce")
            out["actual"] = actual
            out["residual"] = actual - preds

        st.success(f"Scored {len(out):,} rows.")
        st.dataframe(
            out.style.format(
                {
                    "predicted": "{:.3f}", "percentile": "{:.0f}",
                    "actual": "{:.3f}", "residual": "{:+.3f}",
                },
                na_rep="—",
            ),
            width="stretch", height=380,
        )

        if "actual" in out.columns and out["actual"].notna().any():
            valid = out.dropna(subset=["actual"])
            from src.train import regression_metrics

            st.subheader("Scored against the actuals in your file")
            metric_row(regression_metrics(valid["actual"], valid["predicted"]))
            st.caption(
                "If these rows were part of training, these numbers are "
                "optimistic — they are not a substitute for the held-out "
                "evaluation."
            )

        buf = io.StringIO()
        out.to_csv(buf, index=False)
        st.download_button(
            "⬇️ Download predictions",
            buf.getvalue().encode(),
            file_name="crime_predictions.csv",
            mime="text/csv",
        )

        fig = px.histogram(out, x="predicted", nbins=40, color_discrete_sequence=[ACCENT])
        fig.update_layout(height=320, title="Distribution of predictions", margin=dict(t=50))
        st.plotly_chart(fig, width="stretch")


# ==========================================================================
# Main
# ==========================================================================
def main():
    bundle = bundle_or_stop()
    _, variant = sidebar(bundle)

    pages = {
        "Overview": page_overview,
        "Data explorer": page_data,
        "Model performance": page_performance,
        "Explainability": page_explain,
        "Predict": page_predict,
    }
    st.sidebar.divider()
    # `?page=predict` style URLs open a page directly, so links can be shared.
    slugs = {name: name.lower().replace(" ", "-") for name in pages}
    requested = st.query_params.get("page", "")
    names = list(pages)
    start = next((i for i, n in enumerate(names) if slugs[n] == requested), 0)
    choice = st.sidebar.radio("Page", names, index=start, label_visibility="collapsed")
    st.query_params["page"] = slugs[choice]
    st.sidebar.divider()
    st.sidebar.caption(
        "Coursework artifact. Not suitable for any real-world decision about "
        "people or places — see the caveats on the Overview page."
    )
    pages[choice](bundle, variant)


if __name__ == "__main__":
    main()
