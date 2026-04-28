import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md("""
    # 07 — CheMeleon Frozen Encoder Comparison

    Compare CheMeleon with frozen encoder (only FFN trained) vs full
    finetuning (all weights trained). Tests the hypothesis that
    CheMeleon's underperformance is due to overfitting the 9.3M-param
    encoder on small datasets.

    ```bash
    uv run python scripts/run-chemeleon-frozen-training.py
    ```
    """)
    return (mo,)


@app.cell
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import seaborn as sns
    from loguru import logger
    from scipy import stats
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    DATA_DIR = Path("data")
    return DATA_DIR, logger, pairwise_tukeyhsd, pl, plt, sns, stats


@app.cell
def _(DATA_DIR, logger, pl):
    # Load frozen results
    frozen_df = pl.read_parquet(DATA_DIR / "chemeleon_frozen_results.parquet")
    logger.info(f"Frozen: {frozen_df.height} results")

    # Load original (unfrozen) CheMeleon results
    unfrozen_df = pl.read_parquet(DATA_DIR / "chemeleon_results.parquet")
    logger.info(f"Unfrozen: {unfrozen_df.height} results")

    # Combine for comparison
    chemeleon_compare_df = pl.concat([frozen_df, unfrozen_df])
    logger.info(f"Combined: {chemeleon_compare_df.height} results")
    return (chemeleon_compare_df,)


@app.cell
def _(chemeleon_compare_df, mo, pl):
    _summary = (
        chemeleon_compare_df.group_by("target", "model")
        .agg(
            pl.col("auc_roc").mean().alias("auc_roc_mean"),
            pl.col("auc_roc").std().alias("auc_roc_std"),
            pl.col("avg_precision").mean().alias("avg_prec_mean"),
            pl.col("avg_precision").std().alias("avg_prec_std"),
            pl.col("auc_roc").count().alias("n"),
        )
        .sort("target", "auc_roc_mean", descending=[False, True])
    )

    mo.vstack(
        [
            mo.md("## CheMeleon: Frozen vs Unfrozen Encoder"),
            mo.ui.table(_summary),
            mo.md("""
    **Frozen**: only the FFN head is trained (~615K params). The 8.7M-param
    message-passing encoder is frozen at its pre-trained weights.

    **Unfrozen** (original): all 9.3M params are trainable during finetuning.

    If overfitting is the issue, frozen should outperform unfrozen.
        """),
        ]
    )
    return


@app.cell
def _(chemeleon_compare_df, mo, pairwise_tukeyhsd, pl, plt):
    from pathlib import Path

    FIGURES_DIR = Path("docs/figures")

    _fig_tukey, _axes_tukey = plt.subplots(1, 2, figsize=(16, 7))

    for _i, _target in enumerate(["HLM Stability", "PAMPA pH 7.4"]):
        _subset = chemeleon_compare_df.filter(pl.col("target") == _target)
        _values = _subset.get_column("avg_precision").to_numpy()
        _groups = _subset.get_column("model").to_list()
        _tukey = pairwise_tukeyhsd(_values, _groups, alpha=0.05)

        _means = (
            _subset.group_by("model")
            .agg(pl.col("avg_precision").mean())
            .sort("avg_precision", descending=True)
        )
        _best_model = _means.get_column("model")[0]

        _tukey.plot_simultaneous(
            comparison_name=_best_model, ax=_axes_tukey[_i], xlabel="AUC-PR"
        )
        _axes_tukey[_i].set_title(f"{_target}\n(reference: {_best_model})", fontsize=12)

    _fig_tukey.suptitle(
        "CheMeleon Frozen vs Unfrozen: Tukey HSD (FWER = 0.05, AUC-PR)", fontsize=14
    )
    plt.tight_layout()
    _fig_tukey.savefig(
        FIGURES_DIR / "chemeleon-frozen-tukey-hsd.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )

    mo.vstack(
        [
            mo.md("## Tukey HSD: Frozen vs Unfrozen (AUC-PR)"),
            mo.as_html(_fig_tukey),
        ]
    )
    return


@app.cell
def _(chemeleon_compare_df, mo, pl, plt, sns):
    from pathlib import Path

    FIGURES_DIR = Path("docs/figures")

    _model_order = [
        "CheMeleon single-finetune",
        "CheMeleon double-finetune",
        "CheMeleon frozen single",
        "CheMeleon frozen double",
    ]
    _palette = {
        "CheMeleon single-finetune": "#4CAF50",
        "CheMeleon double-finetune": "#8BC34A",
        "CheMeleon frozen single": "#7E57C2",
        "CheMeleon frozen double": "#B39DDB",
    }

    _fig_box, _axes_box = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    for _i, _target in enumerate(["HLM Stability", "PAMPA pH 7.4"]):
        _subset = chemeleon_compare_df.filter(pl.col("target") == _target).to_pandas()
        sns.boxplot(
            data=_subset,
            x="model",
            y="avg_precision",
            hue="model",
            ax=_axes_box[_i],
            order=_model_order,
            palette=_palette,
            legend=False,
        )
        _axes_box[_i].set_title(_target)
        _axes_box[_i].set_xlabel("")
        _axes_box[_i].set_ylabel("AUC-PR" if _i == 0 else "")
        _axes_box[_i].tick_params(axis="x", rotation=45)

    _fig_box.suptitle(
        "CheMeleon: Frozen vs Unfrozen Encoder (25 folds, AUC-PR)", fontsize=14
    )
    plt.tight_layout()
    _fig_box.savefig(
        FIGURES_DIR / "chemeleon-frozen-boxplots.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )

    mo.vstack(
        [
            mo.md("## AUC-PR Distributions"),
            mo.as_html(_fig_box),
        ]
    )
    return


@app.cell
def _(chemeleon_compare_df, mo, pl, stats):
    # Paired comparison: frozen vs unfrozen for each variant
    _comparisons = []
    _pairs = [
        (
            "CheMeleon single-finetune",
            "CheMeleon frozen single",
            "Single: unfrozen vs frozen",
        ),
        (
            "CheMeleon double-finetune",
            "CheMeleon frozen double",
            "Double: unfrozen vs frozen",
        ),
    ]

    for _target in ["HLM Stability", "PAMPA pH 7.4"]:
        for _unfrozen_name, _frozen_name, _label in _pairs:
            _unfrozen_vals = (
                chemeleon_compare_df.filter(
                    (pl.col("target") == _target) & (pl.col("model") == _unfrozen_name)
                )
                .sort("replicate", "fold")
                .get_column("auc_roc")
                .to_numpy()
            )
            _frozen_vals = (
                chemeleon_compare_df.filter(
                    (pl.col("target") == _target) & (pl.col("model") == _frozen_name)
                )
                .sort("replicate", "fold")
                .get_column("auc_roc")
                .to_numpy()
            )
            _diff = _frozen_vals - _unfrozen_vals
            _t_stat, _p_val = stats.ttest_rel(_frozen_vals, _unfrozen_vals)
            _comparisons.append(
                {
                    "target": _target,
                    "comparison": _label,
                    "unfrozen_mean": round(_unfrozen_vals.mean(), 4),
                    "frozen_mean": round(_frozen_vals.mean(), 4),
                    "delta_frozen_minus_unfrozen": round(_diff.mean(), 4),
                    "paired_t_p": round(_p_val, 6),
                    "frozen_wins": int((_diff > 0).sum()),
                    "n_folds": len(_diff),
                }
            )

    _comp_df = pl.DataFrame(_comparisons)

    mo.vstack(
        [
            mo.md("## Paired Comparison: Frozen vs Unfrozen"),
            mo.ui.table(_comp_df),
            mo.md("""
    Positive delta means freezing the encoder *improved* performance
    (supporting the overfitting hypothesis). Negative delta means freezing
    *hurt* (foundation representations alone are insufficient, task-specific
    encoder adaptation is needed).
        """),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Interpretation

    **If frozen > unfrozen**: the CheMeleon encoder's pre-trained
    representations are good enough for these tasks, and the full-finetune
    models were indeed overfitting the encoder weights to small training sets.
    Freezing acts as a strong regularizer.

    **If frozen < unfrozen**: the pre-trained representations need adaptation
    for these specific ADME endpoints. The overfitting hypothesis is wrong (or
    at least insufficient) — the encoder genuinely needs to learn task-specific
    features, and the capacity issue lies elsewhere.

    **If frozen ≈ unfrozen**: the encoder adaptation during finetuning is
    neither helping nor hurting much — the FFN head capacity is the binding
    constraint regardless of whether the encoder is frozen.
    """)
    return


if __name__ == "__main__":
    app.run()
