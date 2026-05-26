import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md("""
    # 06 — Combined Analysis and Discussion

    Final comparison across all nine model variants with statistical testing.
    Generates separate Tukey HSD plots for HLM and PAMPA using AUC-PR as
    the primary metric, and saves them to docs/figures/.
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
    FIGURES_DIR = Path("docs/figures")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return (
        DATA_DIR,
        FIGURES_DIR,
        logger,
        pairwise_tukeyhsd,
        pl,
        plt,
        sns,
        stats,
    )


@app.cell
def _(DATA_DIR, logger, pl):
    xgb_df = pl.read_parquet(DATA_DIR / "xgb_results.parquet")
    chemprop_df = pl.read_parquet(DATA_DIR / "chemprop_results.parquet")
    chemeleon_df = pl.read_parquet(DATA_DIR / "chemeleon_results.parquet")
    frozen_df = pl.read_parquet(DATA_DIR / "chemeleon_frozen_results.parquet")
    all_df = pl.concat([xgb_df, chemprop_df, chemeleon_df, frozen_df])
    logger.info(
        f"Combined: {all_df.height} results "
        f"({xgb_df.height} XGB + {chemprop_df.height} Chemprop + "
        f"{chemeleon_df.height} CheMeleon + {frozen_df.height} frozen)"
    )
    return (all_df,)


@app.cell
def _(all_df, mo, pl):
    _summary = (
        all_df.group_by("target", "model")
        .agg(
            pl.col("avg_precision").mean().alias("auc_pr_mean"),
            pl.col("avg_precision").std().alias("auc_pr_std"),
            pl.col("auc_roc").mean().alias("auc_roc_mean"),
            pl.col("auc_roc").std().alias("auc_roc_std"),
            pl.col("auc_roc").count().alias("n"),
        )
        .sort("target", "auc_pr_mean", descending=[False, True])
    )

    mo.vstack(
        [
            mo.md("## Combined Results: All Models"),
            mo.ui.table(_summary),
        ]
    )
    return


@app.cell
def _(FIGURES_DIR, all_df, logger, mo, pairwise_tukeyhsd, pl, plt, sns):
    # --- Combined boxplot + Tukey HSD per endpoint (HLM, PAMPA) ---
    # Replaces the previous separate boxplots-{slug}-auc-pr.png and
    # tukey-hsd-{slug}-auc-pr.png files with a single side-by-side figure
    # that pairs the AUC-PR distributions with the statistical significance
    # test relative to the best model. Random baseline preserved on the
    # boxplot side. Saved to BOTH legacy filenames so README references stay
    # valid.

    _model_order_combined = [
        "XGBoost scratch",
        "XGBoost RLM-transfer",
        "XGBoost RLM-feature-transfer",
        "Chemprop scratch",
        "Chemprop RLM-transfer",
        "CheMeleon single-finetune",
        "CheMeleon double-finetune",
        "CheMeleon frozen single",
        "CheMeleon frozen double",
    ]
    _palette_combined = {
        "XGBoost scratch": "#FF5722",
        "XGBoost RLM-transfer": "#FF9800",
        "XGBoost RLM-feature-transfer": "#4CAF50",
        "Chemprop scratch": "#2196F3",
        "Chemprop RLM-transfer": "#03A9F4",
        "CheMeleon single-finetune": "#66BB6A",
        "CheMeleon double-finetune": "#8BC34A",
        "CheMeleon frozen single": "#7E57C2",
        "CheMeleon frozen double": "#B39DDB",
    }
    _baselines_combined = {"HLM Stability": 0.602, "PAMPA pH 7.4": 0.855}

    for _target in ["HLM Stability", "PAMPA pH 7.4"]:
        _subset = all_df.filter(pl.col("target") == _target)
        _values = _subset.get_column("avg_precision").to_numpy()
        _groups = _subset.get_column("model").to_list()
        _tukey = pairwise_tukeyhsd(_values, _groups, alpha=0.05)

        # Find best model (highest mean AUC-PR) -> reference for Tukey HSD.
        _means = (
            _subset.group_by("model")
            .agg(pl.col("avg_precision").mean())
            .sort("avg_precision", descending=True)
        )
        _best_model = _means.get_column("model")[0]

        # Combined figure: horizontal boxplot (left) + Tukey HSD (right).
        # Horizontal orientation aligns model names with the Tukey HSD plot
        # and avoids cramped 45-degree x-axis labels.
        _fig, (_ax_box, _ax_tukey) = plt.subplots(
            1, 2, figsize=(16, 7), sharey=False
        )

        # Boxplot panel (horizontal: model on Y, AUC-PR on X). Reverse the
        # order so XGBoost scratch sits on top, matching the natural reading
        # order of the model_order list.
        _df_pd = _subset.to_pandas()
        sns.boxplot(
            data=_df_pd,
            y="model",
            x="avg_precision",
            hue="model",
            ax=_ax_box,
            order=list(reversed(_model_order_combined)),
            palette=_palette_combined,
            legend=False,
        )
        _ax_box.axvline(
            _baselines_combined[_target],
            color="black",
            linestyle="--",
            linewidth=1.2,
            alpha=0.6,
            label=f"Random baseline ({_baselines_combined[_target]:.3f})",
        )
        _ax_box.legend(fontsize=9, loc="lower right")
        _ax_box.set_title("AUC-PR Distribution (25 folds)")
        _ax_box.set_xlabel("AUC-PR")
        _ax_box.set_ylabel("")

        # Tukey HSD panel.
        _tukey.plot_simultaneous(
            comparison_name=_best_model, ax=_ax_tukey, xlabel="AUC-PR"
        )
        _ax_tukey.set_title(
            f"Tukey HSD (FWER = 0.05)\n(reference: {_best_model})"
        )

        _fig.suptitle(
            f"{_target}: Model Comparison",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        # Save under both legacy filenames so existing README references
        # continue to resolve.
        _slug = "hlm" if "HLM" in _target else "pampa"
        for _name in (
            f"tukey-hsd-{_slug}-auc-pr.png",
            f"boxplots-{_slug}-auc-pr.png",
        ):
            _path = FIGURES_DIR / _name
            _fig.savefig(_path, dpi=150, bbox_inches="tight", facecolor="white")
            logger.info(f"Saved {_path}")
        plt.close(_fig)

    mo.md(
        "Saved combined boxplot + Tukey HSD plots (AUC-PR) for HLM and PAMPA. "
        "Each endpoint's figure is written to both "
        "`docs/figures/tukey-hsd-{slug}-auc-pr.png` and "
        "`docs/figures/boxplots-{slug}-auc-pr.png` so the existing README "
        "image references continue to work."
    )
    return


@app.cell
def _(FIGURES_DIR, all_df, logger, mo, pl, plt, sns):
    _model_order = [
        "XGBoost scratch",
        "XGBoost RLM-transfer",
        "XGBoost RLM-feature-transfer",
        "Chemprop scratch",
        "Chemprop RLM-transfer",
        "CheMeleon single-finetune",
        "CheMeleon double-finetune",
        "CheMeleon frozen single",
        "CheMeleon frozen double",
    ]
    _palette = {
        "XGBoost scratch": "#FF5722",
        "XGBoost RLM-transfer": "#FF9800",
        "XGBoost RLM-feature-transfer": "#4CAF50",
        "Chemprop scratch": "#2196F3",
        "Chemprop RLM-transfer": "#03A9F4",
        "CheMeleon single-finetune": "#66BB6A",
        "CheMeleon double-finetune": "#8BC34A",
        "CheMeleon frozen single": "#7E57C2",
        "CheMeleon frozen double": "#B39DDB",
    }

    # Random classifier baseline = positive class prevalence
    _baselines = {"HLM Stability": 0.602, "PAMPA pH 7.4": 0.855}

    # --- Combined figure (kept for supplementary) ---
    _fig_box, _axes_box = plt.subplots(1, 2, figsize=(20, 7), sharey=False)
    for _i, _target in enumerate(["HLM Stability", "PAMPA pH 7.4"]):
        _subset = all_df.filter(pl.col("target") == _target).to_pandas()
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
        _axes_box[_i].axhline(
            _baselines[_target],
            color="black",
            linestyle="--",
            linewidth=1.2,
            alpha=0.6,
            label=f"Random baseline ({_baselines[_target]:.3f})",
        )
        _axes_box[_i].legend(fontsize=9, loc="lower right")
        _axes_box[_i].set_title(_target)
        _axes_box[_i].set_xlabel("")
        _axes_box[_i].set_ylabel("AUC-PR" if _i == 0 else "")
        _axes_box[_i].tick_params(axis="x", rotation=45)

    _fig_box.suptitle("All Models: AUC-PR Distributions (25 folds)", fontsize=14)
    plt.tight_layout()
    _fig_box.savefig(
        FIGURES_DIR / "all-models-boxplots.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    logger.info(f"Saved {FIGURES_DIR / 'all-models-boxplots.png'}")

    # Per-endpoint boxplots are now produced by the combined boxplot+Tukey
    # cell above (saved to boxplots-{slug}-auc-pr.png AND
    # tukey-hsd-{slug}-auc-pr.png).

    mo.vstack(
        [
            mo.md("## AUC-PR Distributions"),
            mo.as_html(_fig_box),
        ]
    )
    return


@app.cell
def _(all_df, mo, pl, stats):
    # Transfer learning delta: for each fold, compute (transfer - scratch)
    _deltas = []
    _pairs = [
        ("XGBoost scratch", "XGBoost RLM-transfer", "XGBoost (decision-boundary)"),
        (
            "XGBoost scratch",
            "XGBoost RLM-feature-transfer",
            "XGBoost (feature-importance)",
        ),
        ("Chemprop scratch", "Chemprop RLM-transfer", "Chemprop"),
        (
            "CheMeleon single-finetune",
            "CheMeleon double-finetune",
            "CheMeleon (single->double)",
        ),
    ]

    for _target in ["HLM Stability", "PAMPA pH 7.4"]:
        for _base, _transfer, _label in _pairs:
            _base_vals = (
                all_df.filter(
                    (pl.col("target") == _target) & (pl.col("model") == _base)
                )
                .sort("replicate", "fold")
                .get_column("avg_precision")
                .to_numpy()
            )
            _transfer_vals = (
                all_df.filter(
                    (pl.col("target") == _target) & (pl.col("model") == _transfer)
                )
                .sort("replicate", "fold")
                .get_column("avg_precision")
                .to_numpy()
            )
            _diff = _transfer_vals - _base_vals
            _t_stat, _p_val = stats.ttest_rel(_transfer_vals, _base_vals)
            _deltas.append(
                {
                    "target": _target,
                    "comparison": _label,
                    "mean_delta": round(float(_diff.mean()), 4),
                    "std_delta": round(float(_diff.std()), 4),
                    "paired_t_p": round(float(_p_val), 6),
                    "significant": _p_val < 0.05,
                    "n_transfer_wins": int((_diff > 0).sum()),
                    "n_folds": len(_diff),
                }
            )

    _delta_df = pl.DataFrame(_deltas)

    mo.vstack(
        [
            mo.md("## Transfer Learning Effect (Paired Deltas, AUC-PR)"),
            mo.ui.table(_delta_df),
            mo.md("""
    Each row shows the mean AUC-PR improvement from the transfer variant over
    its baseline, paired by fold. Positive = transfer helps. The paired t-test
    p-value tests whether the mean difference is significantly different from zero.
        """),
        ]
    )
    return


@app.cell
def _(FIGURES_DIR, all_df, logger, mo, pairwise_tukeyhsd, pl, plt):
    # Also generate the combined side-by-side Tukey (keeping for backward compat)
    _fig_tukey, _axes_tukey = plt.subplots(1, 2, figsize=(18, 7))

    for _i, _target in enumerate(["HLM Stability", "PAMPA pH 7.4"]):
        _subset = all_df.filter(pl.col("target") == _target)
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

    _fig_tukey.suptitle("Tukey HSD: All Models (FWER = 0.05, AUC-PR)", fontsize=14)
    plt.tight_layout()

    _path = FIGURES_DIR / "all-models-tukey-hsd.png"
    _fig_tukey.savefig(_path, dpi=150, bbox_inches="tight", facecolor="white")
    logger.info(f"Saved {_path}")

    mo.vstack(
        [
            mo.md("## Tukey HSD: Simultaneous Confidence Intervals (AUC-PR)"),
            mo.as_html(_fig_tukey),
            mo.md("""
    The reference model (best mean AUC-PR) is highlighted. Groups colored
    **red** are significantly different from the reference at alpha = 0.05.
    Groups colored **gray** are not significantly different. Overlapping
    intervals between any two groups indicate no significant difference.
        """),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
