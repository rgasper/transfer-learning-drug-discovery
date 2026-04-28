import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md("""
    # 11 — RLM Base Model Comparison

    Validates that all architectures (XGBoost, Chemprop, CheMeleon) learn
    the RLM source task comparably. If the pre-trained models we transfer
    from have different RLM performance, that would confound our interpretation
    of downstream transfer effects.

    ```bash
    uv run python scripts/run-rlm-base-eval.py
    ```
    """)
    return (mo,)


@app.cell
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import polars as pl
    import seaborn as sns
    from loguru import logger
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    DATA_DIR = Path("data")
    FIGURES_DIR = Path("docs/figures")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR, FIGURES_DIR, logger, pairwise_tukeyhsd, pl, plt, sns


@app.cell
def _(DATA_DIR, logger, pl):
    rlm_base_df = pl.read_parquet(DATA_DIR / "rlm_base_results.parquet")
    logger.info(f"Loaded {rlm_base_df.height} RLM base results")
    return (rlm_base_df,)


@app.cell
def _(mo, pl, rlm_base_df):
    _summary = (
        rlm_base_df.group_by("model")
        .agg(
            pl.col("avg_precision").mean().alias("auc_pr_mean"),
            pl.col("avg_precision").std().alias("auc_pr_std"),
            pl.col("auc_roc").mean().alias("auc_roc_mean"),
            pl.col("auc_roc").std().alias("auc_roc_std"),
            pl.col("auc_roc").count().alias("n"),
        )
        .sort("auc_pr_mean", descending=True)
    )

    mo.vstack(
        [
            mo.md("## RLM Source Task: Performance by Architecture"),
            mo.ui.table(_summary),
            mo.md("""
    All three architectures should achieve comparable performance on RLM.
    If they do, any differences after transfer to HLM/PAMPA are attributable
    to the transfer mechanism, not a head start on the source task.
        """),
        ]
    )
    return


@app.cell
def _(FIGURES_DIR, logger, mo, pairwise_tukeyhsd, pl, plt, rlm_base_df, sns):
    # Combined figure: boxplot + Tukey HSD side by side
    _fig, (_ax_box, _ax_tukey) = plt.subplots(1, 2, figsize=(14, 5))

    _model_order = ["XGBoost scratch", "Chemprop scratch", "CheMeleon single-finetune"]
    _palette = {
        "XGBoost scratch": "#FF5722",
        "Chemprop scratch": "#2196F3",
        "CheMeleon single-finetune": "#4CAF50",
    }

    # Boxplot
    _df_pd = rlm_base_df.to_pandas()
    sns.boxplot(
        data=_df_pd,
        x="model",
        y="avg_precision",
        hue="model",
        ax=_ax_box,
        order=_model_order,
        palette=_palette,
        legend=False,
    )
    _ax_box.set_title("AUC-PR Distribution (25 folds)")
    _ax_box.set_xlabel("")
    _ax_box.set_ylabel("AUC-PR")
    _ax_box.tick_params(axis="x", rotation=20)

    # Tukey HSD
    _values = rlm_base_df.get_column("avg_precision").to_numpy()
    _groups = rlm_base_df.get_column("model").to_list()
    _tukey = pairwise_tukeyhsd(_values, _groups, alpha=0.05)

    _means = (
        rlm_base_df.group_by("model")
        .agg(pl.col("avg_precision").mean())
        .sort("avg_precision", descending=True)
    )
    _best_model = _means.get_column("model")[0]

    _tukey.plot_simultaneous(comparison_name=_best_model, ax=_ax_tukey, xlabel="AUC-PR")
    _ax_tukey.set_title(f"Tukey HSD (FWER = 0.05)\n(reference: {_best_model})")

    _fig.suptitle(
        "RLM Source Task: XGBoost Underperforms D-MPNN Architectures",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    _path = FIGURES_DIR / "rlm-base-comparison.png"
    _fig.savefig(_path, dpi=150, bbox_inches="tight", facecolor="white")
    logger.info(f"Saved {_path}")

    mo.vstack(
        [
            mo.md("## RLM Base Model Comparison"),
            mo.as_html(_fig),
            mo.md("""
    **Left:** AUC-PR distributions for each architecture on RLM (5x5 CV,
    25 folds). **Right:** Tukey HSD simultaneous confidence intervals.

    XGBoost on Morgan fingerprints is significantly worse than both D-MPNN
    architectures on the RLM source task and shows substantially higher
    variance across folds. Chemprop and CheMeleon are statistically
    indistinguishable. This asymmetry strengthens the transfer learning
    conclusions: XGBoost transfer helps on HLM despite a weaker and
    noisier source model, and XGBoost transfer catastrophically fails on
    PAMPA despite learning RLM *less* well -- ruling out "too-strong
    source model" as an explanation for the negative transfer. The higher
    fold-to-fold instability also suggests that fixed fingerprints capture
    less generalizable structure than learned D-MPNN representations.
        """),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
