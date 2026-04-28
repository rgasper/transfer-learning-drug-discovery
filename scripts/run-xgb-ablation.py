"""XGBoost transfer learning ablation on PAMPA.

Tests whether the catastrophic XGBoost RLM->PAMPA transfer failure is
recoverable by increasing the finetuning budget. Varies both the number
of pretrain rounds and the finetune budget, running the full 25-fold CV
for each configuration.

If the failure is due to insufficient finetuning rounds to "undo" the
RLM decision boundaries, increasing the budget should recover performance.
If the failure is structural (inherited trees permanently bias
predictions), no amount of additional boosting will help.

Results are saved to data/xgb_ablation_results.parquet.

Usage:
    uv run python scripts/run-xgb-ablation.py
"""

import json
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
from loguru import logger
from sklearn.metrics import average_precision_score, roc_auc_score

DATA_DIR = Path("data")

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

EARLY_STOPPING_ROUNDS = 20

# Ablation grid: (pretrain_rounds, finetune_budget)
# None pretrain_rounds means scratch (no transfer)
ABLATION_CONFIGS: list[tuple[int | None, int]] = [
    (None, 200),  # scratch baseline (matches original experiment)
    (200, 200),  # original transfer config
    (200, 500),  # 2.5x more finetune budget
    (200, 1000),  # 5x more finetune budget
    (50, 200),  # lighter pretrain, same finetune
    (50, 1000),  # lighter pretrain, 5x finetune
]


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Compute classification metrics."""
    metrics: dict[str, float] = {}
    try:
        metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["auc_roc"] = float("nan")
    try:
        metrics["avg_precision"] = average_precision_score(y_true, y_prob)
    except ValueError:
        metrics["avg_precision"] = float("nan")
    return metrics


def main() -> None:
    # Load split config
    with open(DATA_DIR / "split_config.json") as f:
        split_config = json.load(f)
    n_replicates: int = split_config["n_replicates"]
    n_folds: int = split_config["n_folds"]

    # Load fingerprints and PAMPA splits
    fp_data = np.load(DATA_DIR / "morgan_fps_2048_r3.npz", allow_pickle=True)
    global_fps = fp_data["fp_matrix"]

    pampa_split = np.load(DATA_DIR / "pampa_splits.npz", allow_pickle=True)
    pampa_labels = pampa_split["labels"]
    pampa_folds = pampa_split["folds"]
    pampa_X = global_fps[pampa_split["fp_indices"]]
    logger.info(f"PAMPA: {pampa_X.shape[0]} compounds, {n_replicates}x{n_folds} CV")

    # Load RLM data for pretraining
    rlm_split = np.load(DATA_DIR / "rlm_splits.npz", allow_pickle=True)
    rlm_X = global_fps[rlm_split["fp_indices"]]
    rlm_y = rlm_split["labels"]
    dtrain_rlm = xgb.DMatrix(rlm_X, label=rlm_y)
    logger.info(f"RLM: {rlm_X.shape[0]} compounds for pretraining")

    # Build pretrained models (keyed by pretrain_rounds)
    pretrained_models: dict[int, xgb.Booster] = {}
    pretrain_rounds_needed = {pr for pr, _ in ABLATION_CONFIGS if pr is not None}
    for n_pretrain in sorted(pretrain_rounds_needed):
        logger.info(f"Pretraining XGBoost on RLM ({n_pretrain} rounds)...")
        model = xgb.train(
            XGB_PARAMS,
            dtrain_rlm,
            num_boost_round=n_pretrain,
            verbose_eval=False,
        )
        pretrained_models[n_pretrain] = model
        logger.info(
            f"  RLM pretrained ({n_pretrain} rounds): "
            f"{model.num_boosted_rounds()} boosted rounds"
        )

    # Run ablation across all folds
    all_results: list[dict] = []

    for pretrain_rounds, finetune_budget in ABLATION_CONFIGS:
        if pretrain_rounds is None:
            config_name = f"scratch (budget={finetune_budget})"
        else:
            config_name = (
                f"transfer (pretrain={pretrain_rounds}, budget={finetune_budget})"
            )
        logger.info(f"Running config: {config_name}")

        for rep in range(n_replicates):
            fold_assignments = pampa_folds[rep]

            for fold in range(n_folds):
                test_mask = fold_assignments == fold
                train_mask = ~test_mask
                X_train = pampa_X[train_mask]
                X_test = pampa_X[test_mask]
                y_train = pampa_labels[train_mask]
                y_test = pampa_labels[test_mask]

                dtrain = xgb.DMatrix(X_train, label=y_train)
                dtest = xgb.DMatrix(X_test, label=y_test)

                train_kwargs: dict = {
                    "params": XGB_PARAMS,
                    "dtrain": dtrain,
                    "num_boost_round": finetune_budget,
                    "evals": [(dtest, "val")],
                    "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
                    "verbose_eval": False,
                }
                if pretrain_rounds is not None:
                    train_kwargs["xgb_model"] = pretrained_models[pretrain_rounds]

                model = xgb.train(**train_kwargs)
                y_prob = model.predict(dtest)
                metrics = evaluate_predictions(y_test, y_prob)

                total_rounds = model.num_boosted_rounds()
                inherited_rounds = (
                    pretrained_models[pretrain_rounds].num_boosted_rounds()
                    if pretrain_rounds is not None
                    else 0
                )

                all_results.append(
                    {
                        "config": config_name,
                        "pretrain_rounds": pretrain_rounds
                        if pretrain_rounds is not None
                        else 0,
                        "finetune_budget": finetune_budget,
                        "is_transfer": pretrain_rounds is not None,
                        "replicate": rep,
                        "fold": fold,
                        "total_rounds": total_rounds,
                        "new_rounds": total_rounds - inherited_rounds,
                        **metrics,
                    }
                )

        logger.info(f"  Completed {n_replicates * n_folds} folds for {config_name}")

    results_df = pl.DataFrame(all_results)
    results_df.write_parquet(DATA_DIR / "xgb_ablation_results.parquet")
    logger.info(
        f"Saved {results_df.height} results to "
        f"{DATA_DIR / 'xgb_ablation_results.parquet'}"
    )

    # Print summary
    summary = (
        results_df.group_by("config")
        .agg(
            pl.col("auc_roc").mean().alias("auc_roc_mean"),
            pl.col("auc_roc").std().alias("auc_roc_std"),
            pl.col("avg_precision").mean().alias("avg_prec_mean"),
            pl.col("avg_precision").std().alias("avg_prec_std"),
            pl.col("new_rounds").mean().alias("mean_new_rounds"),
        )
        .sort("config")
    )
    print(summary)


if __name__ == "__main__":
    main()
