"""Evaluate XGBoost on RLM to establish a comparable starting point.

Part 1 of the RLM base model evaluation. Run this before
run-rlm-base-eval-nn.py to avoid libomp conflicts between XGBoost and
PyTorch on macOS.

Usage:
    uv run python scripts/run-rlm-base-eval-xgb.py
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
from loguru import logger
from sklearn.metrics import average_precision_score, roc_auc_score

DATA_DIR = Path("data")
CACHE_DIR = DATA_DIR / "rlm_base_cache"
CACHE_DIR.mkdir(exist_ok=True)

N_BOOST_ROUNDS = 200
EARLY_STOPPING_ROUNDS = 20

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


def make_cache_key(model_type: str, rep: int, fold: int) -> str:
    """Generate a deterministic cache key."""
    h = hashlib.sha256()
    h.update(f"rlm_base_{model_type}_{rep}_{fold}".encode())
    return h.hexdigest()[:16]


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute classification metrics."""
    metrics = {}
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
    with open(DATA_DIR / "split_config.json") as f:
        split_config = json.load(f)
    n_replicates = split_config["n_replicates"]
    n_folds = split_config["n_folds"]

    fp_data = np.load(DATA_DIR / "morgan_fps_2048_r3.npz", allow_pickle=True)
    global_fps = fp_data["fp_matrix"]

    rlm_split = np.load(DATA_DIR / "rlm_splits.npz", allow_pickle=True)
    rlm_labels = rlm_split["labels"]
    rlm_fp_indices = rlm_split["fp_indices"]
    rlm_X = global_fps[rlm_fp_indices]
    rlm_folds = rlm_split["folds"]

    logger.info(f"RLM: {rlm_X.shape[0]} molecules, {n_replicates}x{n_folds} CV")

    all_results: list[dict] = []

    for rep in range(n_replicates):
        fold_assignments = rlm_folds[rep]

        for fold in range(n_folds):
            test_mask = fold_assignments == fold
            train_mask = ~test_mask
            y_test = rlm_labels[test_mask]

            cache_path = CACHE_DIR / f"{make_cache_key('xgb', rep, fold)}.npz"
            if cache_path.exists():
                y_prob = np.load(cache_path)["y_prob"]
                logger.info(f"  rep={rep} fold={fold}: XGBoost cached")
            else:
                X_train = rlm_X[train_mask]
                X_test = rlm_X[test_mask]
                y_train = rlm_labels[train_mask]

                dtrain = xgb.DMatrix(X_train, label=y_train)
                dval = xgb.DMatrix(X_test, label=y_test)
                model = xgb.train(
                    XGB_PARAMS,
                    dtrain,
                    num_boost_round=N_BOOST_ROUNDS,
                    evals=[(dval, "val")],
                    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                    verbose_eval=False,
                )
                y_prob = model.predict(xgb.DMatrix(X_test))
                np.savez_compressed(cache_path, y_prob=y_prob)
                logger.info(f"  rep={rep} fold={fold}: XGBoost trained")

            metrics = evaluate_predictions(y_test, y_prob)
            all_results.append(
                {
                    "target": "RLM Stability",
                    "model": "XGBoost scratch",
                    "replicate": rep,
                    "fold": fold,
                    **metrics,
                }
            )

    results_df = pl.DataFrame(all_results)
    out_path = DATA_DIR / "rlm_base_xgb_results.parquet"
    results_df.write_parquet(out_path)
    logger.info(f"Saved {results_df.height} results to {out_path}")

    summary = (
        results_df.group_by("model")
        .agg(
            pl.col("auc_roc").mean().alias("auc_roc_mean"),
            pl.col("auc_roc").std().alias("auc_roc_std"),
            pl.col("avg_precision").mean().alias("avg_prec_mean"),
            pl.col("avg_precision").std().alias("avg_prec_std"),
        )
        .sort("model")
    )
    print(summary)


if __name__ == "__main__":
    main()
