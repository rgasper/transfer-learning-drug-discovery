import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md("""
    # 14 — Reverse Transfer Experiment: PAMPA → RLM

    Tests the reverse direction: pre-train on PAMPA, finetune on RLM.
    If our architectural thesis is correct, XGBoost should again suffer
    from negative transfer (PAMPA features are irrelevant to metabolic
    stability), while D-MPNN should be robust.

    Results are summarized in docs/reverse-transfer.md.
    """)
    return (mo,)


@app.cell
def _():
    import hashlib
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import seaborn as sns
    import torch
    import xgboost as xgb
    from lightning import pytorch as lightning_pl
    from loguru import logger
    from sklearn.metrics import average_precision_score, roc_auc_score
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    from chemprop import data as chemprop_data
    from chemprop import featurizers, models, nn

    DATA_DIR = Path("data")
    FIGURES_DIR = Path("docs/figures")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return (
        DATA_DIR,
        FIGURES_DIR,
        chemprop_data,
        featurizers,
        lightning_pl,
        logger,
        models,
        nn,
        np,
        pairwise_tukeyhsd,
        pl,
        plt,
        roc_auc_score,
        average_precision_score,
        sns,
        torch,
        xgb,
    )


@app.cell
def _(DATA_DIR, logger, np):
    """Load RLM and PAMPA data."""
    import json

    with open(DATA_DIR / "split_config.json") as _f:
        split_config = json.load(_f)

    _fp_data = np.load(DATA_DIR / "morgan_fps_2048_r3.npz", allow_pickle=True)
    global_fps = _fp_data["fp_matrix"]

    _rlm_split = np.load(DATA_DIR / "rlm_splits.npz", allow_pickle=True)
    rlm_smiles = list(_rlm_split["smiles"])
    rlm_labels = _rlm_split["labels"]
    rlm_folds = _rlm_split["folds"]
    rlm_fp_indices = _rlm_split["fp_indices"]
    rlm_X = global_fps[rlm_fp_indices]

    _pampa_split = np.load(DATA_DIR / "pampa_splits.npz", allow_pickle=True)
    pampa_smiles = list(_pampa_split["smiles"])
    pampa_labels = _pampa_split["labels"]
    pampa_fp_indices = _pampa_split["fp_indices"]
    pampa_X = global_fps[pampa_fp_indices]

    logger.info(f"RLM: {len(rlm_smiles)}, PAMPA: {len(pampa_smiles)}")
    return (
        global_fps,
        pampa_X,
        pampa_labels,
        pampa_smiles,
        rlm_X,
        rlm_folds,
        rlm_labels,
        rlm_smiles,
        split_config,
    )


@app.cell
def _(
    average_precision_score,
    chemprop_data,
    featurizers,
    lightning_pl,
    logger,
    models,
    nn,
    np,
    pampa_X,
    pampa_labels,
    pampa_smiles,
    pl,
    rlm_X,
    rlm_folds,
    rlm_labels,
    rlm_smiles,
    roc_auc_score,
    split_config,
    torch,
    xgb,
):
    """Run 5x5 CV: XGBoost scratch, XGBoost PAMPA-transfer, Chemprop scratch, Chemprop PAMPA-transfer on RLM."""
    _featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    _n_reps = split_config["n_replicates"]
    _n_folds = split_config["n_folds"]

    _all_results = []

    for _rep in range(_n_reps):
        _fold_assign = rlm_folds[_rep]
        for _fold in range(_n_folds):
            _test_mask = _fold_assign == _fold
            _train_mask = ~_test_mask
            _y_test = rlm_labels[_test_mask]

            logger.info(f"rep={_rep} fold={_fold}")

            # --- XGBoost scratch on RLM ---
            np.random.seed(42 + _rep * 100 + _fold)
            _dtrain = xgb.DMatrix(rlm_X[_train_mask], label=rlm_labels[_train_mask])
            _dval = xgb.DMatrix(rlm_X[_test_mask], label=_y_test)
            _xgb_scratch = xgb.train(
                {
                    "objective": "binary:logistic",
                    "eval_metric": "logloss",
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "random_state": 42,
                    "nthread": 1,
                    "verbosity": 0,
                },
                _dtrain,
                200,
                evals=[(_dval, "val")],
                early_stopping_rounds=20,
                verbose_eval=False,
            )
            _y_prob = _xgb_scratch.predict(_dval)
            _all_results.append(
                {
                    "model": "XGBoost scratch",
                    "replicate": _rep,
                    "fold": _fold,
                    "auc_roc": roc_auc_score(_y_test, _y_prob),
                    "avg_precision": average_precision_score(_y_test, _y_prob),
                }
            )

            # --- XGBoost PAMPA-transfer on RLM ---
            _dtrain_pampa = xgb.DMatrix(pampa_X, label=pampa_labels)
            _pampa_model = xgb.train(
                {
                    "objective": "binary:logistic",
                    "eval_metric": "logloss",
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "random_state": 42,
                    "nthread": 1,
                    "verbosity": 0,
                },
                _dtrain_pampa,
                200,
                verbose_eval=False,
            )
            _xgb_transfer = xgb.train(
                {
                    "objective": "binary:logistic",
                    "eval_metric": "logloss",
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "random_state": 42,
                    "nthread": 1,
                    "verbosity": 0,
                },
                _dtrain,
                200,
                evals=[(_dval, "val")],
                early_stopping_rounds=20,
                verbose_eval=False,
                xgb_model=_pampa_model,
            )
            _y_prob_t = _xgb_transfer.predict(_dval)
            _all_results.append(
                {
                    "model": "XGBoost PAMPA-transfer",
                    "replicate": _rep,
                    "fold": _fold,
                    "auc_roc": roc_auc_score(_y_test, _y_prob_t),
                    "avg_precision": average_precision_score(_y_test, _y_prob_t),
                }
            )

            # --- Chemprop scratch on RLM ---
            lightning_pl.seed_everything(42 + _rep * 100 + _fold, workers=True)
            _train_smi = [
                rlm_smiles[i] for i in range(len(rlm_smiles)) if _train_mask[i]
            ]
            _train_y = rlm_labels[_train_mask].reshape(-1, 1).astype(float)
            _test_smi = [rlm_smiles[i] for i in range(len(rlm_smiles)) if _test_mask[i]]
            _test_y_arr = _y_test.reshape(-1, 1).astype(float)

            _n = len(_train_smi)
            _n_val = max(1, int(_n * 0.1))
            _perm = np.random.default_rng(42).permutation(_n)
            _train_data = [
                chemprop_data.MoleculeDatapoint.from_smi(_train_smi[i], _train_y[i])
                for i in _perm[_n_val:]
            ]
            _val_data = [
                chemprop_data.MoleculeDatapoint.from_smi(_train_smi[i], _train_y[i])
                for i in _perm[:_n_val]
            ]
            _test_data = [
                chemprop_data.MoleculeDatapoint.from_smi(s, y)
                for s, y in zip(_test_smi, _test_y_arr)
            ]
            _train_ds = chemprop_data.MoleculeDataset(_train_data, _featurizer)
            _val_ds = chemprop_data.MoleculeDataset(_val_data, _featurizer)
            _test_ds = chemprop_data.MoleculeDataset(_test_data, _featurizer)
            _train_loader = chemprop_data.build_dataloader(
                _train_ds, num_workers=0, batch_size=64
            )
            _val_loader = chemprop_data.build_dataloader(
                _val_ds, num_workers=0, shuffle=False, batch_size=64
            )
            _test_loader = chemprop_data.build_dataloader(
                _test_ds, num_workers=0, shuffle=False, batch_size=64
            )

            _mp = nn.BondMessagePassing()
            _agg = nn.MeanAggregation()
            _ffn = nn.BinaryClassificationFFN(input_dim=_mp.output_dim)
            _cp_scratch = models.MPNN(_mp, _agg, _ffn, batch_norm=False)
            _trainer = lightning_pl.Trainer(
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                deterministic=True,
                accelerator="gpu",
                devices=1,
                max_epochs=30,
            )
            _trainer.fit(_cp_scratch, _train_loader, _val_loader)
            _preds = _trainer.predict(_cp_scratch, _test_loader)
            _y_prob_cp = torch.cat(_preds).cpu().numpy().flatten()
            _all_results.append(
                {
                    "model": "Chemprop scratch",
                    "replicate": _rep,
                    "fold": _fold,
                    "auc_roc": roc_auc_score(_y_test, _y_prob_cp),
                    "avg_precision": average_precision_score(_y_test, _y_prob_cp),
                }
            )

            # --- Chemprop PAMPA-transfer on RLM ---
            lightning_pl.seed_everything(42 + _rep * 100 + _fold, workers=True)
            # Pre-train on PAMPA
            _pampa_y = pampa_labels.reshape(-1, 1).astype(float)
            _n_p = len(pampa_smiles)
            _n_p_val = max(1, int(_n_p * 0.1))
            _perm_p = np.random.default_rng(42).permutation(_n_p)
            _p_train = [
                chemprop_data.MoleculeDatapoint.from_smi(pampa_smiles[i], _pampa_y[i])
                for i in _perm_p[_n_p_val:]
            ]
            _p_val = [
                chemprop_data.MoleculeDatapoint.from_smi(pampa_smiles[i], _pampa_y[i])
                for i in _perm_p[:_n_p_val]
            ]
            _p_train_ds = chemprop_data.MoleculeDataset(_p_train, _featurizer)
            _p_val_ds = chemprop_data.MoleculeDataset(_p_val, _featurizer)
            _p_train_loader = chemprop_data.build_dataloader(
                _p_train_ds, num_workers=0, batch_size=64
            )
            _p_val_loader = chemprop_data.build_dataloader(
                _p_val_ds, num_workers=0, shuffle=False, batch_size=64
            )

            _mp2 = nn.BondMessagePassing()
            _agg2 = nn.MeanAggregation()
            _ffn2 = nn.BinaryClassificationFFN(input_dim=_mp2.output_dim)
            _cp_pampa = models.MPNN(_mp2, _agg2, _ffn2, batch_norm=False)
            _trainer2 = lightning_pl.Trainer(
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                deterministic=True,
                accelerator="gpu",
                devices=1,
                max_epochs=30,
            )
            _trainer2.fit(_cp_pampa, _p_train_loader, _p_val_loader)

            # Finetune on RLM with new FFN head
            _new_ffn = nn.BinaryClassificationFFN(
                input_dim=_cp_pampa.message_passing.output_dim
            )
            _cp_transfer = models.MPNN(
                _cp_pampa.message_passing, _cp_pampa.agg, _new_ffn, batch_norm=False
            )
            lightning_pl.seed_everything(42 + _rep * 100 + _fold, workers=True)
            _trainer3 = lightning_pl.Trainer(
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                deterministic=True,
                accelerator="gpu",
                devices=1,
                max_epochs=30,
            )
            _trainer3.fit(_cp_transfer, _train_loader, _val_loader)
            _preds_t = _trainer3.predict(_cp_transfer, _test_loader)
            _y_prob_cp_t = torch.cat(_preds_t).cpu().numpy().flatten()
            _all_results.append(
                {
                    "model": "Chemprop PAMPA-transfer",
                    "replicate": _rep,
                    "fold": _fold,
                    "auc_roc": roc_auc_score(_y_test, _y_prob_cp_t),
                    "avg_precision": average_precision_score(_y_test, _y_prob_cp_t),
                }
            )

    reverse_results_df = pl.DataFrame(_all_results)
    reverse_results_df.write_parquet(DATA_DIR / "reverse_transfer_results.parquet")
    logger.info(f"Saved {reverse_results_df.height} results")
    return (reverse_results_df,)


@app.cell
def _(FIGURES_DIR, logger, mo, np, pairwise_tukeyhsd, pl, plt, reverse_results_df, sns):
    """Visualize reverse transfer results."""
    _model_order = [
        "XGBoost scratch",
        "XGBoost PAMPA-transfer",
        "Chemprop scratch",
        "Chemprop PAMPA-transfer",
    ]
    _palette = {
        "XGBoost scratch": "#FF5722",
        "XGBoost PAMPA-transfer": "#FF9800",
        "Chemprop scratch": "#2196F3",
        "Chemprop PAMPA-transfer": "#03A9F4",
    }

    _fig, (_ax_box, _ax_tukey) = plt.subplots(1, 2, figsize=(16, 6))

    # Boxplot
    _df_pd = reverse_results_df.to_pandas()
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
    _rlm_baseline = 0.298
    _ax_box.axhline(
        _rlm_baseline,
        color="black",
        linestyle="--",
        linewidth=1.2,
        alpha=0.6,
        label=f"Random baseline ({_rlm_baseline:.3f})",
    )
    _ax_box.legend(fontsize=9, loc="lower right")
    _ax_box.set_title("RLM Stability: AUC-PR (25 folds)")
    _ax_box.set_xlabel("")
    _ax_box.set_ylabel("AUC-PR")
    _ax_box.tick_params(axis="x", rotation=25)

    # Tukey HSD
    _values = reverse_results_df.get_column("avg_precision").to_numpy()
    _groups = reverse_results_df.get_column("model").to_list()
    _tukey = pairwise_tukeyhsd(_values, _groups, alpha=0.05)
    _means = (
        reverse_results_df.group_by("model")
        .agg(pl.col("avg_precision").mean())
        .sort("avg_precision", descending=True)
    )
    _best = _means.get_column("model")[0]
    _tukey.plot_simultaneous(comparison_name=_best, ax=_ax_tukey, xlabel="AUC-PR")
    _ax_tukey.set_title(f"Tukey HSD (FWER = 0.05)\n(reference: {_best})")

    _fig.suptitle("Reverse Transfer: PAMPA → RLM", fontsize=14, fontweight="bold")
    plt.tight_layout()

    _path = FIGURES_DIR / "reverse-transfer-pampa-to-rlm.png"
    _fig.savefig(_path, dpi=150, bbox_inches="tight", facecolor="white")
    logger.info(f"Saved {_path}")

    # Summary stats
    _summary = (
        reverse_results_df.group_by("model")
        .agg(
            pl.col("avg_precision").mean().alias("auc_pr_mean"),
            pl.col("avg_precision").std().alias("auc_pr_std"),
        )
        .sort("auc_pr_mean", descending=True)
    )

    mo.vstack(
        [
            mo.md("## Reverse Transfer: PAMPA → RLM"),
            mo.as_html(_fig),
            mo.ui.table(_summary),
            mo.md("""
If our thesis is correct:
- **XGBoost PAMPA-transfer** should hurt (negative transfer, like RLM→PAMPA did)
- **Chemprop PAMPA-transfer** should be harmless or slightly helpful (encoder features generalize)

This mirrors the RLM→PAMPA experiment but in reverse, testing whether the
architectural robustness property holds symmetrically.
        """),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
