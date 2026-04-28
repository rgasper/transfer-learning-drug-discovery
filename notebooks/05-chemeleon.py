import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md("""
    # 05 — CheMeleon Foundation Model Results

    Visualize CheMeleon results: single-finetune (foundation -> target) vs
    double-finetune (foundation -> RLM -> target). Training is run via
    `scripts/run-chemeleon-training.py`.

    ```bash
    uv run python scripts/run-chemeleon-training.py
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
    return DATA_DIR, logger, pairwise_tukeyhsd, pl, plt, sns


@app.cell
def _(DATA_DIR, logger, pl):
    chemeleon_results_df = pl.read_parquet(DATA_DIR / "chemeleon_results.parquet")
    logger.info(f"Loaded {chemeleon_results_df.height} CheMeleon results")

    # Also load Chemprop and XGBoost results for combined comparison
    chemprop_results_df = pl.read_parquet(DATA_DIR / "chemprop_results.parquet")
    xgb_results_df = pl.read_parquet(DATA_DIR / "xgb_results.parquet")
    logger.info(
        f"Loaded {chemprop_results_df.height} Chemprop, {xgb_results_df.height} XGBoost results"
    )

    all_results_df = pl.concat(
        [xgb_results_df, chemprop_results_df, chemeleon_results_df]
    )
    logger.info(f"Combined: {all_results_df.height} total results")
    return all_results_df, chemeleon_results_df, chemprop_results_df, xgb_results_df


@app.cell
def _(all_results_df, mo, pl):
    _summary = (
        all_results_df.group_by("target", "model")
        .agg(
            pl.col("auc_roc").mean().alias("auc_roc_mean"),
            pl.col("auc_roc").std().alias("auc_roc_std"),
            pl.col("avg_precision").mean().alias("avg_precision_mean"),
            pl.col("avg_precision").std().alias("avg_precision_std"),
            pl.col("auc_roc").count().alias("n_folds"),
        )
        .sort("target", "auc_roc_mean", descending=[False, True])
    )

    mo.vstack(
        [
            mo.md("## Combined Results: All Models (25 folds)"),
            mo.ui.table(_summary),
        ]
    )
    return


@app.cell
def _(all_results_df, mo, pairwise_tukeyhsd, pl, plt):
    _fig_tukey, _axes_tukey = plt.subplots(1, 2, figsize=(16, 6))

    for _i, _target in enumerate(["HLM Stability", "PAMPA pH 7.4"]):
        _subset = all_results_df.filter(pl.col("target") == _target)
        _values = _subset.get_column("auc_roc").to_numpy()
        _groups = _subset.get_column("model").to_list()

        _tukey = pairwise_tukeyhsd(_values, _groups, alpha=0.05)
        _tukey.plot_simultaneous(ax=_axes_tukey[_i])
        _axes_tukey[_i].set_title(_target)
        _axes_tukey[_i].set_xlabel("AUC-ROC")

    _fig_tukey.suptitle(
        "All Models: Tukey HSD Simultaneous Confidence Intervals", fontsize=14
    )
    plt.tight_layout()

    mo.vstack(
        [
            mo.md("## Tukey HSD: All Models"),
            mo.as_html(_fig_tukey),
        ]
    )
    return


@app.cell
def _(all_results_df, mo, pl, plt, sns):
    _model_order = [
        "XGBoost scratch",
        "XGBoost RLM-transfer",
        "Chemprop scratch",
        "Chemprop RLM-transfer",
        "CheMeleon single-finetune",
        "CheMeleon double-finetune",
    ]
    _palette = {
        "XGBoost scratch": "#FF5722",
        "XGBoost RLM-transfer": "#FF9800",
        "Chemprop scratch": "#2196F3",
        "Chemprop RLM-transfer": "#03A9F4",
        "CheMeleon single-finetune": "#4CAF50",
        "CheMeleon double-finetune": "#8BC34A",
    }

    _fig_box, _axes_box = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

    for _i, _target in enumerate(["HLM Stability", "PAMPA pH 7.4"]):
        _subset = all_results_df.filter(pl.col("target") == _target).to_pandas()
        sns.boxplot(
            data=_subset,
            x="model",
            y="auc_roc",
            hue="model",
            ax=_axes_box[_i],
            order=_model_order,
            palette=_palette,
            legend=False,
        )
        _axes_box[_i].set_title(_target)
        _axes_box[_i].set_xlabel("")
        _axes_box[_i].set_ylabel("AUC-ROC" if _i == 0 else "")
        _axes_box[_i].tick_params(axis="x", rotation=45)

    _fig_box.suptitle("All Models: AUC-ROC Distributions (25 folds)", fontsize=14)
    plt.tight_layout()

    mo.vstack(
        [
            mo.md("## AUC-ROC Distributions: All Models"),
            mo.as_html(_fig_box),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
