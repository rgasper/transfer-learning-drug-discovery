"""XGBoost random-label pre-training control on PAMPA.

Tests whether the catastrophic XGBoost RLM->PAMPA transfer failure is
specific to inheriting *wrong* decision boundaries (from a real but
unrelated task), or whether *any* pre-training -- even on noise --
degrades the model equally.

Three conditions are tested across the full 25-fold PAMPA CV:

1. Scratch (no pre-training) -- baseline
2. RLM-pretrained transfer -- inherits real RLM decision boundaries
3. Random-label-pretrained transfer -- inherits decision boundaries
   trained on shuffled RLM labels (same molecules, same class balance,
   but no real signal)

If random-label transfer also collapses to PAMPA random-baseline
performance, the failure is about *any* extra trees degrading the model
(a structural problem with continue-boosting). If random-label transfer
performs *differently* from real-RLM transfer, the content of the
inherited decisions matters -- wrong signal is worse than no signal.

Multiple random seeds for the label shuffle provide robustness.

Results are saved to data/xgb_random_pretrain_results.parquet.

Usage:
    uv run python scripts/run-xgb-random-pretrain.py
"""

import json
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
from loguru import logger
from sklearn.metrics import average_precision_score, roc_auc_score

DATA_DIR = Path("data")

XGB_PARAMS: dict = {
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

# Random seeds for label permutation (5 seeds to average over shuffle randomness)
RANDOM_SEEDS: list[int] = [0, 1, 2, 3, 4]


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Compute classification metrics.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities.

    Returns:
        Dict with auc_roc and avg_precision.

    Example:
        >>> evaluate_predictions(np.array([0, 1, 1]), np.array([0.1, 0.9, 0.8]))
        {'auc_roc': 1.0, 'avg_precision': 1.0}
    """
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


def pretrain_on_random_labels(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> xgb.Booster:
    """Pre-train XGBoost on shuffled labels (preserving class balance).

    Args:
        X: Feature matrix (RLM molecules).
        y: Real labels (will be shuffled).
        seed: Random seed for the permutation.

    Returns:
        Pre-trained XGBoost Booster on random labels.
    """
    rng = np.random.default_rng(seed)
    y_shuffled = rng.permutation(y)
    dtrain = xgb.DMatrix(X, label=y_shuffled)
    model = xgb.train(
        XGB_PARAMS,
        dtrain,
        num_boost_round=N_BOOST_ROUNDS,
        verbose_eval=False,
    )
    logger.info(
        f"  Random-label pretrain (seed={seed}): "
        f"{model.num_boosted_rounds()} rounds, "
        f"class balance preserved = {y_shuffled.mean():.3f} vs original {y.mean():.3f}"
    )
    return model


def main() -> None:
    # Load split config
    with open(DATA_DIR / "split_config.json") as f:
        split_config = json.load(f)
    n_replicates: int = split_config["n_replicates"]
    n_folds: int = split_config["n_folds"]

    # Load fingerprints
    fp_data = np.load(DATA_DIR / "morgan_fps_2048_r3.npz", allow_pickle=True)
    global_fps = fp_data["fp_matrix"]

    # Load PAMPA splits
    pampa_split = np.load(DATA_DIR / "pampa_splits.npz", allow_pickle=True)
    pampa_labels = pampa_split["labels"]
    pampa_folds = pampa_split["folds"]
    pampa_X = global_fps[pampa_split["fp_indices"]]
    logger.info(f"PAMPA: {pampa_X.shape[0]} compounds, {n_replicates}x{n_folds} CV")

    # Load RLM data for pre-training
    rlm_split = np.load(DATA_DIR / "rlm_splits.npz", allow_pickle=True)
    rlm_X = global_fps[rlm_split["fp_indices"]]
    rlm_y = rlm_split["labels"]
    dtrain_rlm = xgb.DMatrix(rlm_X, label=rlm_y)
    logger.info(f"RLM: {rlm_X.shape[0]} compounds for pre-training")

    # Build pre-trained models
    logger.info("Pre-training on real RLM labels...")
    rlm_model = xgb.train(
        XGB_PARAMS,
        dtrain_rlm,
        num_boost_round=N_BOOST_ROUNDS,
        verbose_eval=False,
    )
    logger.info(f"  RLM pretrained: {rlm_model.num_boosted_rounds()} rounds")

    logger.info("Pre-training on random labels...")
    random_models: dict[int, xgb.Booster] = {}
    for seed in RANDOM_SEEDS:
        random_models[seed] = pretrain_on_random_labels(rlm_X, rlm_y, seed)

    # Run all conditions across 25 folds
    all_results: list[dict] = []

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

            # 1. Scratch
            model_scratch = xgb.train(
                XGB_PARAMS,
                dtrain,
                num_boost_round=N_BOOST_ROUNDS,
                evals=[(dtest, "val")],
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                verbose_eval=False,
            )
            y_prob = model_scratch.predict(dtest)
            metrics = evaluate_predictions(y_test, y_prob)
            all_results.append(
                {
                    "condition": "scratch",
                    "seed": -1,
                    "replicate": rep,
                    "fold": fold,
                    "total_rounds": model_scratch.num_boosted_rounds(),
                    "inherited_rounds": 0,
                    **metrics,
                }
            )

            # 2. Real RLM transfer
            model_rlm = xgb.train(
                XGB_PARAMS,
                dtrain,
                num_boost_round=N_BOOST_ROUNDS,
                evals=[(dtest, "val")],
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                verbose_eval=False,
                xgb_model=rlm_model,
            )
            y_prob = model_rlm.predict(dtest)
            metrics = evaluate_predictions(y_test, y_prob)
            all_results.append(
                {
                    "condition": "rlm_transfer",
                    "seed": -1,
                    "replicate": rep,
                    "fold": fold,
                    "total_rounds": model_rlm.num_boosted_rounds(),
                    "inherited_rounds": rlm_model.num_boosted_rounds(),
                    **metrics,
                }
            )

            # 3. Random-label transfer (one per seed)
            for seed in RANDOM_SEEDS:
                model_rand = xgb.train(
                    XGB_PARAMS,
                    dtrain,
                    num_boost_round=N_BOOST_ROUNDS,
                    evals=[(dtest, "val")],
                    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                    verbose_eval=False,
                    xgb_model=random_models[seed],
                )
                y_prob = model_rand.predict(dtest)
                metrics = evaluate_predictions(y_test, y_prob)
                all_results.append(
                    {
                        "condition": "random_label_transfer",
                        "seed": seed,
                        "replicate": rep,
                        "fold": fold,
                        "total_rounds": model_rand.num_boosted_rounds(),
                        "inherited_rounds": random_models[seed].num_boosted_rounds(),
                        **metrics,
                    }
                )

        logger.info(f"  Completed rep {rep + 1}/{n_replicates}")

    results_df = pl.DataFrame(all_results)
    results_df.write_parquet(DATA_DIR / "xgb_random_pretrain_results.parquet")
    logger.info(
        f"Saved {results_df.height} results to "
        f"{DATA_DIR / 'xgb_random_pretrain_results.parquet'}"
    )

    # Print summary
    summary = (
        results_df.group_by("condition")
        .agg(
            pl.col("auc_roc").mean().alias("auc_roc_mean"),
            pl.col("auc_roc").std().alias("auc_roc_std"),
            pl.col("avg_precision").mean().alias("avg_prec_mean"),
            pl.col("avg_precision").std().alias("avg_prec_std"),
            pl.col("total_rounds").mean().alias("mean_total_rounds"),
            pl.len().alias("n_observations"),
        )
        .sort("condition")
    )
    logger.info("Summary:")
    print(summary)

    # Also print per-seed breakdown for random-label condition
    random_summary = (
        results_df.filter(pl.col("condition") == "random_label_transfer")
        .group_by("seed")
        .agg(
            pl.col("auc_roc").mean().alias("auc_roc_mean"),
            pl.col("auc_roc").std().alias("auc_roc_std"),
            pl.col("avg_precision").mean().alias("avg_prec_mean"),
            pl.col("avg_precision").std().alias("avg_prec_std"),
        )
        .sort("seed")
    )
    logger.info("Random-label per-seed breakdown:")
    print(random_summary)


if __name__ == "__main__":
    main()
