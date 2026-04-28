import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md("""
    # 06 — Combined Analysis and Discussion

    Final comparison across all six model variants with statistical testing
    and interpretation of results.
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
    xgb_df = pl.read_parquet(DATA_DIR / "xgb_results.parquet")
    chemprop_df = pl.read_parquet(DATA_DIR / "chemprop_results.parquet")
    chemeleon_df = pl.read_parquet(DATA_DIR / "chemeleon_results.parquet")
    all_df = pl.concat([xgb_df, chemprop_df, chemeleon_df])
    logger.info(
        f"Combined: {all_df.height} results ({xgb_df.height} XGB + {chemprop_df.height} Chemprop + {chemeleon_df.height} CheMeleon)"
    )
    return (all_df,)


@app.cell
def _(all_df, mo, pl):
    _summary = (
        all_df.group_by("target", "model")
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
            mo.md("## Combined Results: All Models"),
            mo.ui.table(_summary),
        ]
    )
    return


@app.cell
def _(all_df, mo, pairwise_tukeyhsd, pl, plt):
    _fig_tukey, _axes_tukey = plt.subplots(1, 2, figsize=(16, 7))

    for _i, _target in enumerate(["HLM Stability", "PAMPA pH 7.4"]):
        _subset = all_df.filter(pl.col("target") == _target)
        _values = _subset.get_column("auc_roc").to_numpy()
        _groups = _subset.get_column("model").to_list()
        _tukey = pairwise_tukeyhsd(_values, _groups, alpha=0.05)

        # Find the best model (highest mean AUC)
        _means = _subset.group_by("model").agg(pl.col("auc_roc").mean()).sort("auc_roc", descending=True)
        _best_model = _means.get_column("model")[0]

        _tukey.plot_simultaneous(comparison_name=_best_model, ax=_axes_tukey[_i], xlabel="AUC-ROC")
        _axes_tukey[_i].set_title(f"{_target}" + "\n" + f"(reference: {_best_model})", fontsize=12)

    _fig_tukey.suptitle("Tukey HSD: All Models (FWER = 0.05)", fontsize=14)
    plt.tight_layout()

    mo.vstack([
        mo.md("## Tukey HSD: Simultaneous Confidence Intervals"),
        mo.as_html(_fig_tukey),
        mo.md("""
    The reference model (best mean AUC-ROC) is highlighted. Groups colored
    **red** are significantly different from the reference at alpha = 0.05.
    Groups colored **gray** are not significantly different. Overlapping
    intervals between any two groups indicate no significant difference.
        """),
    ])

    return


@app.cell
def _(all_df, mo, pl, plt, sns):
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
        _subset = all_df.filter(pl.col("target") == _target).to_pandas()
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
        _axes_box[_i].set_title(_target, fontsize=13)
        _axes_box[_i].set_xlabel("")
        _axes_box[_i].set_ylabel("AUC-ROC" if _i == 0 else "")
        _axes_box[_i].tick_params(axis="x", rotation=45)

    _fig_box.suptitle("AUC-ROC Distributions (25 folds)", fontsize=14)
    plt.tight_layout()

    mo.vstack(
        [
            mo.md("## AUC-ROC Distributions"),
            mo.as_html(_fig_box),
        ]
    )
    return


@app.cell
def _(all_df, mo, pl, stats):
    # Transfer learning delta: for each fold, compute (transfer - scratch)
    _deltas = []
    _pairs = [
        ("XGBoost scratch", "XGBoost RLM-transfer", "XGBoost"),
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
                .get_column("auc_roc")
                .to_numpy()
            )
            _transfer_vals = (
                all_df.filter(
                    (pl.col("target") == _target) & (pl.col("model") == _transfer)
                )
                .sort("replicate", "fold")
                .get_column("auc_roc")
                .to_numpy()
            )
            _diff = _transfer_vals - _base_vals
            _t_stat, _p_val = stats.ttest_rel(_transfer_vals, _base_vals)
            _deltas.append(
                {
                    "target": _target,
                    "comparison": _label,
                    "mean_delta": round(_diff.mean(), 4),
                    "std_delta": round(_diff.std(), 4),
                    "paired_t_p": round(_p_val, 6),
                    "significant": _p_val < 0.05,
                    "n_transfer_wins": int((_diff > 0).sum()),
                    "n_folds": len(_diff),
                }
            )

    _delta_df = pl.DataFrame(_deltas)

    mo.vstack(
        [
            mo.md("## Transfer Learning Effect (Paired Deltas)"),
            mo.ui.table(_delta_df),
            mo.md("""
    Each row shows the mean AUC-ROC improvement from the transfer variant over
    its baseline, paired by fold. Positive = transfer helps. The paired t-test
    p-value tests whether the mean difference is significantly different from zero.
        """),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Discussion

    ### 1. Why does Chemprop RLM-transfer outperform everything?

    **Chemprop RLM-transfer** is the best model for both HLM (0.768) and PAMPA (0.716).
    This is notable because it beats both the foundation model (CheMeleon) and the
    larger-capacity CheMeleon double-finetune.

    The likely explanation is a combination of two factors:

    **Right-sized model for the data.** The base Chemprop D-MPNN has 318K parameters.
    CheMeleon has 9.3M — a 29x difference. With only ~720 training samples for HLM
    and ~1,626 for PAMPA, CheMeleon has a params-to-sample ratio of 12,957:1 (HLM)
    and 5,738:1 (PAMPA). This is severely overparameterized. Even with foundation
    pre-training providing a good initialization, the model has enough capacity to
    overfit during finetuning. The base Chemprop model, at 442:1 (HLM) and 196:1
    (PAMPA), is better matched to the data scale.

    **Domain-specific pre-training > generic pre-training for related tasks.** The
    RLM pre-training exposes the model to 2,529 compounds with microsomal stability
    labels — the same property family as HLM. This domain-specific signal is more
    directly useful than CheMeleon's generic Mordred descriptor pre-training on 1M
    PubChem compounds. For HLM, the RLM-pretrained encoder has already learned
    "what makes a compound metabolically stable," and fine-tuning only needs to
    adapt from rat to human metabolism.

    ### 2. Why does transfer learning work for D-MPNN but not XGBoost?

    **XGBoost transfer on PAMPA: -0.150 AUC (catastrophic).** XGBoost transfer on
    HLM: +0.066 (helpful). The key difference is *where* transfer happens in each
    architecture.

    **XGBoost transfers at the decision boundary.** When we continue boosting from
    an RLM-pretrained XGBoost, the new trees build on top of the existing RLM
    decision boundaries. If the target task has similar decision boundaries (HLM:
    also microsomal stability), this is helpful. If the target task has completely
    different decision boundaries (PAMPA: membrane permeability), the existing trees
    actively mislead — the model starts from a wrong baseline and the new trees must
    first undo the RLM predictions before learning PAMPA patterns. With early
    stopping, there may not be enough rounds to recover.

    **Chemprop transfers at the representation level.** When we load RLM-pretrained
    Chemprop weights and replace the FFN head, the message-passing encoder retains
    learned molecular features while the FFN head is re-initialized from scratch.
    The encoder features (atom environments, functional group patterns, ring systems)
    are general enough to be useful for any molecular property, even if the specific
    property is unrelated. The new FFN head learns the correct mapping from these
    features to the target, without being constrained by old decision boundaries.

    This is the fundamental advantage of representation-level transfer: the features
    generalize even when the task does not.

    ### 3. Why does CheMeleon underperform random-init Chemprop on PAMPA?

    CheMeleon single-finetune (0.676) is *worse* than Chemprop scratch (0.701) on
    PAMPA. Several hypotheses:

    **Overfitting due to model capacity.** With 9.3M parameters and ~1,626 PAMPA
    training samples, CheMeleon is extremely overparameterized. The foundation
    pre-training provides a reasonable initialization, but 30 epochs of finetuning
    may be enough to overfit. The smaller Chemprop model (318K params) has less
    capacity to memorize and generalizes better.

    **Mordred descriptor pre-training may not help for PAMPA.** CheMeleon was
    pre-trained to predict Mordred molecular descriptors from structure. These
    descriptors capture physicochemical properties (molecular weight, logP, polar
    surface area, etc.) but may not include features specifically relevant to
    membrane permeability. A random-init model that learns task-specific features
    from scratch may discover better representations for this particular endpoint.

    **The CheMeleon double-finetune partially recovers** (0.686 vs 0.676 for single),
    suggesting that the intermediate RLM finetuning step helps even the large model
    by providing some additional domain-relevant signal, though it's still not enough
    to overcome the capacity mismatch.

    ### 4. Key takeaway

    For small ADME datasets (~1,000-2,000 compounds), a right-sized D-MPNN (318K
    params) with domain-specific transfer learning outperforms both larger foundation
    models and traditional ML with fixed fingerprints. The benefit of learned
    representations over fixed fingerprints is most visible in the unrelated transfer
    case: Chemprop handles it gracefully while XGBoost fails catastrophically.

    Foundation models like CheMeleon may become more valuable as dataset sizes grow
    or when no related pre-training data is available, but for the dataset sizes
    tested here, they are outcompeted by smaller, domain-tuned models.
    """)
    return


if __name__ == "__main__":
    app.run()
