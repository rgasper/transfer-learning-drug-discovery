import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md("""
    # 03 — XGBoost Baseline Training

    Train XGBoost models on Morgan fingerprints (2048-bit, radius 3) for
    HLM stability and PAMPA permeability. Compare from-scratch training
    vs transfer learning (pre-trained on RLM, continued boosting on target).

    Splits are loaded from notebook 02 (PaCMAP-based clustering, 5x5 CV).
    """)
    return (mo,)


@app.cell(hide_code=True)
def _():
    import json
    from pathlib import Path

    import numpy as np
    import polars as pl
    import xgboost as xgb
    from loguru import logger
    from sklearn.metrics import average_precision_score, roc_auc_score

    DATA_DIR = Path("data")

    ENDPOINT_NAMES = {
        "rlm": "RLM Stability",
        "hlm": "HLM Stability",
        "pampa": "PAMPA pH 7.4",
    }

    TARGET_ENDPOINTS = ["hlm", "pampa"]
    PRETRAIN_ENDPOINT = "rlm"

    return (
        DATA_DIR,
        ENDPOINT_NAMES,
        PRETRAIN_ENDPOINT,
        TARGET_ENDPOINTS,
        average_precision_score,
        json,
        logger,
        np,
        pl,
        roc_auc_score,
        xgb,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Step 1: Load Splits and Fingerprints

    Fingerprints and fold assignments were computed in notebook 02
    (PaCMAP-based KMeans clustering + GroupKFoldShuffle). Load them here.
    """)
    return


@app.cell(hide_code=True)
def _(DATA_DIR, ENDPOINT_NAMES, json, logger, np):
    # Load global fingerprint matrix
    _fp_data = np.load(DATA_DIR / "morgan_fps_2048_r3.npz", allow_pickle=True)
    _global_fp_matrix = _fp_data["fp_matrix"]
    _global_smiles = list(_fp_data["smiles"])
    logger.info(f"Global FP matrix: {_global_fp_matrix.shape}")

    # Load split config
    with open(DATA_DIR / "split_config.json") as _f:
        split_config = json.load(_f)
    N_REPLICATES = split_config["n_replicates"]
    N_FOLDS = split_config["n_folds"]
    logger.info(f"Split config: {N_REPLICATES} replicates x {N_FOLDS} folds")

    # Load per-endpoint splits and build fp_data dict
    fp_data: dict[str, dict] = {}
    for _key in ENDPOINT_NAMES:
        _split = np.load(DATA_DIR / f"{_key}_splits.npz", allow_pickle=True)
        _smiles = list(_split["smiles"])
        _labels = _split["labels"]
        _fp_indices = _split["fp_indices"]
        _X = _global_fp_matrix[_fp_indices]
        _folds = _split["folds"]
        fp_data[_key] = {
            "X": _X,
            "y": _labels,
            "smiles": _smiles,
            "folds": _folds,
        }
        logger.info(
            f"{_key}: X={_X.shape}, y={_labels.shape}, "
            f"folds={_folds.shape}, active={(_labels == 1).sum()}, inactive={(_labels == 0).sum()}"
        )

    return N_FOLDS, N_REPLICATES, fp_data


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Step 2: XGBoost Training

    For each target endpoint (HLM, PAMPA), train two XGBoost variants across
    25 folds (5 replicates x 5 folds, loaded from notebook 02):

    1. **From scratch**: train only on the target endpoint's training fold
    2. **Transfer (RLM pretrain)**: train on full RLM dataset first, then
       continue boosting on the target endpoint's training fold
    """)
    return


@app.cell(hide_code=True)
def _(average_precision_score, np, roc_auc_score, xgb):
    XGB_PARAMS = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbosity": 0,
    }
    N_BOOST_ROUNDS = 200
    EARLY_STOPPING_ROUNDS = 20

    def train_xgb_from_scratch(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> xgb.Booster:
        """Train XGBoost from scratch on a single fold."""
        _dtrain = xgb.DMatrix(X_train, label=y_train)
        _dval = xgb.DMatrix(X_val, label=y_val)
        _model = xgb.train(
            XGB_PARAMS,
            _dtrain,
            num_boost_round=N_BOOST_ROUNDS,
            evals=[(_dval, "val")],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
        )
        return _model

    def train_xgb_transfer(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        pretrained_model: xgb.Booster,
    ) -> xgb.Booster:
        """Continue boosting from a pretrained XGBoost model."""
        _dtrain = xgb.DMatrix(X_train, label=y_train)
        _dval = xgb.DMatrix(X_val, label=y_val)
        _model = xgb.train(
            XGB_PARAMS,
            _dtrain,
            num_boost_round=N_BOOST_ROUNDS,
            evals=[(_dval, "val")],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
            xgb_model=pretrained_model,
        )
        return _model

    def evaluate_model(model: xgb.Booster, X: np.ndarray, y: np.ndarray) -> dict:
        """Compute classification metrics for an XGBoost model."""
        _dtest = xgb.DMatrix(X)
        _y_prob = model.predict(_dtest)
        _metrics = {}
        try:
            _metrics["auc_roc"] = roc_auc_score(y, _y_prob)
        except ValueError:
            _metrics["auc_roc"] = float("nan")
        try:
            _metrics["avg_precision"] = average_precision_score(y, _y_prob)
        except ValueError:
            _metrics["avg_precision"] = float("nan")
        return _metrics

    return (
        N_BOOST_ROUNDS,
        XGB_PARAMS,
        evaluate_model,
        train_xgb_from_scratch,
        train_xgb_transfer,
    )


@app.cell(hide_code=True)
def _(
    N_BOOST_ROUNDS,
    PRETRAIN_ENDPOINT,
    XGB_PARAMS,
    fp_data: dict[str, dict],
    logger,
    xgb,
):
    # Pre-train XGBoost on full RLM dataset
    _X_rlm = fp_data[PRETRAIN_ENDPOINT]["X"]
    _y_rlm = fp_data[PRETRAIN_ENDPOINT]["y"]
    _dtrain_rlm = xgb.DMatrix(_X_rlm, label=_y_rlm)
    rlm_pretrained_model = xgb.train(
        XGB_PARAMS,
        _dtrain_rlm,
        num_boost_round=N_BOOST_ROUNDS,
        verbose_eval=False,
    )
    logger.info(
        f"RLM pretrained model: {rlm_pretrained_model.num_boosted_rounds()} rounds"
    )
    return (rlm_pretrained_model,)


@app.cell(hide_code=True)
def _(
    ENDPOINT_NAMES,
    N_FOLDS,
    N_REPLICATES,
    TARGET_ENDPOINTS,
    evaluate_model,
    fp_data: dict[str, dict],
    logger,
    pl,
    rlm_pretrained_model,
    train_xgb_from_scratch,
    train_xgb_transfer,
):
    # Run 5x5-fold CV for each target endpoint
    all_results: list[dict] = []

    for _target_key in TARGET_ENDPOINTS:
        _X = fp_data[_target_key]["X"]
        _y = fp_data[_target_key]["y"]
        _folds_matrix = fp_data[_target_key]["folds"]
        _target_name = ENDPOINT_NAMES[_target_key]

        logger.info(f"Training on {_target_name} ({_X.shape[0]} samples)")

        for _rep in range(N_REPLICATES):
            _fold_assignments = _folds_matrix[_rep]

            for _fold in range(N_FOLDS):
                _test_mask = _fold_assignments == _fold
                _train_mask = ~_test_mask
                _X_train, _X_test = _X[_train_mask], _X[_test_mask]
                _y_train, _y_test = _y[_train_mask], _y[_test_mask]

                # From scratch
                _model_scratch = train_xgb_from_scratch(
                    _X_train, _y_train, _X_test, _y_test
                )
                _metrics_scratch = evaluate_model(_model_scratch, _X_test, _y_test)
                all_results.append(
                    {
                        "target": _target_name,
                        "model": "XGBoost scratch",
                        "replicate": _rep,
                        "fold": _fold,
                        **_metrics_scratch,
                    }
                )

                # Transfer from RLM
                _model_transfer = train_xgb_transfer(
                    _X_train,
                    _y_train,
                    _X_test,
                    _y_test,
                    rlm_pretrained_model,
                )
                _metrics_transfer = evaluate_model(_model_transfer, _X_test, _y_test)
                all_results.append(
                    {
                        "target": _target_name,
                        "model": "XGBoost RLM-transfer",
                        "replicate": _rep,
                        "fold": _fold,
                        **_metrics_transfer,
                    }
                )

        logger.info(f"  Completed {N_REPLICATES * N_FOLDS} folds for {_target_name}")

    results_df = pl.DataFrame(all_results)
    logger.info(f"Total results: {results_df.height} rows")

    return (results_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Step 3: Results Summary
    """)
    return


@app.cell(hide_code=True)
def _(mo, pl, results_df):
    # Summary statistics
    _summary = (
        results_df.group_by("target", "model")
        .agg(
            pl.col("auc_roc").mean().alias("auc_roc_mean"),
            pl.col("auc_roc").std().alias("auc_roc_std"),
            pl.col("avg_precision").mean().alias("avg_precision_mean"),
            pl.col("avg_precision").std().alias("avg_precision_std"),
            pl.col("auc_roc").count().alias("n_folds"),
        )
        .sort("target", "model")
    )

    mo.vstack(
        [
            mo.md("### Mean Metrics (25 folds)"),
            mo.ui.table(_summary),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo, pl, results_df):
    import matplotlib.pyplot as plt
    import seaborn as sns

    _FIGURES_DIR = __import__("pathlib").Path("docs/figures")
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Boxplots of AUC-PR by model and target
    _fig_box, _axes_box = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for _i, _target in enumerate(["HLM Stability", "PAMPA pH 7.4"]):
        _subset = results_df.filter(pl.col("target") == _target).to_pandas()
        sns.boxplot(
            data=_subset,
            x="model",
            y="avg_precision",
            ax=_axes_box[_i],
            palette={"XGBoost scratch": "#FF5722", "XGBoost RLM-transfer": "#2196F3"},
        )
        _axes_box[_i].set_title(_target)
        _axes_box[_i].set_xlabel("")
        _axes_box[_i].set_ylabel("AUC-PR" if _i == 0 else "")
        _axes_box[_i].tick_params(axis="x", rotation=15)

    _fig_box.suptitle("XGBoost: From Scratch vs RLM Transfer (25 folds)", fontsize=14)
    plt.tight_layout()
    _fig_box.savefig(
        _FIGURES_DIR / "xgb-boxplots.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )

    mo.vstack(
        [
            mo.md("### AUC-PR Distributions"),
            mo.as_html(_fig_box),
        ]
    )
    return (plt,)


@app.cell(hide_code=True)
def _(mo, np, pl, plt, results_df):
    from scipy import stats

    # Paired comparison plots (Walters-style)
    _fig_paired, _axes_paired = plt.subplots(1, 2, figsize=(14, 5))

    for _i, _target in enumerate(["HLM Stability", "PAMPA pH 7.4"]):
        _scratch = (
            results_df.filter(
                (pl.col("target") == _target) & (pl.col("model") == "XGBoost scratch")
            )
            .sort("replicate", "fold")
            .get_column("avg_precision")
            .to_numpy()
        )
        _transfer = (
            results_df.filter(
                (pl.col("target") == _target)
                & (pl.col("model") == "XGBoost RLM-transfer")
            )
            .sort("replicate", "fold")
            .get_column("avg_precision")
            .to_numpy()
        )

        _ax = _axes_paired[_i]
        # Lines connecting paired folds
        for _j in range(len(_scratch)):
            _color = "#2196F3" if _transfer[_j] > _scratch[_j] else "#FF5722"
            _ax.plot(
                [0, 1],
                [_scratch[_j], _transfer[_j]],
                color=_color,
                alpha=0.3,
                linewidth=0.8,
            )

        _ax.scatter(
            np.zeros(len(_scratch)),
            _scratch,
            color="#FF5722",
            s=15,
            zorder=5,
            label="Scratch",
        )
        _ax.scatter(
            np.ones(len(_transfer)),
            _transfer,
            color="#2196F3",
            s=15,
            zorder=5,
            label="Transfer",
        )

        # Paired t-test
        _t_stat, _p_value = stats.ttest_rel(_transfer, _scratch)
        _mean_diff = (_transfer - _scratch).mean()
        _title_color = (
            "#2196F3"
            if _p_value < 0.05 and _mean_diff > 0
            else ("#FF5722" if _p_value < 0.05 and _mean_diff < 0 else "black")
        )
        _title = f"{_target}" + "\n" + f"p={_p_value:.4f}, mean diff={_mean_diff:+.4f}"
        _ax.set_title(_title, color=_title_color, fontsize=11)
        _ax.set_xticks([0, 1])
        _ax.set_xticklabels(["Scratch", "RLM Transfer"])
        _ax.set_ylabel("AUC-PR")
        _ax.legend(fontsize=9)

    _fig_paired.suptitle(
        "Paired Comparison: XGBoost Scratch vs RLM Transfer", fontsize=14
    )
    plt.tight_layout()

    _fig_paired.savefig(
        __import__("pathlib").Path("docs/figures") / "xgb-paired-comparison.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )

    mo.vstack(
        [
            mo.md("### Paired Fold Comparison (AUC-PR)"),
            mo.as_html(_fig_paired),
            mo.md("""
    Lines connect the same CV fold across the two models. Blue lines = transfer wins,
    red lines = scratch wins. Title color indicates paired t-test significance
    (blue = transfer significantly better, red = scratch significantly better,
    black = no significant difference at p < 0.05).
        """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo, pl, plt, results_df):
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    _FIGURES_DIR = __import__("pathlib").Path("docs/figures")

    _fig_tukey, _axes_tukey = plt.subplots(1, 2, figsize=(14, 5))

    for _i, _target in enumerate(["HLM Stability", "PAMPA pH 7.4"]):
        _subset = results_df.filter(pl.col("target") == _target)
        _values = _subset.get_column("avg_precision").to_numpy()
        _groups = _subset.get_column("model").to_list()

        _tukey = pairwise_tukeyhsd(_values, _groups, alpha=0.05)
        _tukey.plot_simultaneous(ax=_axes_tukey[_i])
        _axes_tukey[_i].set_title(_target)
        _axes_tukey[_i].set_xlabel("AUC-PR")

    _fig_tukey.suptitle("Tukey HSD Simultaneous Confidence Intervals", fontsize=14)
    plt.tight_layout()
    _fig_tukey.savefig(
        _FIGURES_DIR / "xgb-tukey-hsd.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )

    mo.vstack(
        [
            mo.md("### Tukey HSD Comparison"),
            mo.as_html(_fig_tukey),
            mo.md("""
    Overlapping intervals indicate no statistically significant difference
    between models. Non-overlapping intervals indicate a significant
    difference (FWER-controlled at alpha = 0.05).
        """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(DATA_DIR, logger, mo, results_df):
    # Save results for downstream analysis
    results_df.write_parquet(DATA_DIR / "xgb_results.parquet")
    logger.info(f"Saved XGBoost results to {DATA_DIR / 'xgb_results.parquet'}")

    # Quick peek at results
    mo.vstack(
        [
            mo.md("### Saved Results"),
            mo.md(
                f"Results saved to `{DATA_DIR / 'xgb_results.parquet'}` ({results_df.height} rows)"
            ),
            mo.ui.table(results_df.head(10)),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
