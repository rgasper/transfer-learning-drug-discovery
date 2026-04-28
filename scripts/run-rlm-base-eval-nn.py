"""Evaluate Chemprop and CheMeleon on RLM to establish comparable starting points.

Part 2 of the RLM base model evaluation. Run run-rlm-base-eval-xgb.py
first (separate process to avoid libomp conflicts between XGBoost and
PyTorch on macOS), then run this script. At the end it merges both
parquets into the final rlm_base_results.parquet.

Usage:
    uv run python scripts/run-rlm-base-eval-xgb.py
    uv run python scripts/run-rlm-base-eval-nn.py
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
from lightning import pytorch as lightning_pl
from loguru import logger
from sklearn.metrics import average_precision_score, roc_auc_score

from chemprop import data as chemprop_data
from chemprop import featurizers, models, nn

DATA_DIR = Path("data")
CHECKPOINTS_DIR = Path("checkpoints")
CACHE_DIR = DATA_DIR / "rlm_base_cache"
CACHE_DIR.mkdir(exist_ok=True)

CHEMELEON_PATH = CHECKPOINTS_DIR / "chemeleon_mp.pt"

MAX_EPOCHS_CHEMPROP = 30
MAX_EPOCHS_CHEMELEON = 30

featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()


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


def load_chemeleon_mp() -> nn.BondMessagePassing:
    """Load the CheMeleon BondMessagePassing encoder."""
    chemeleon_data = torch.load(CHEMELEON_PATH, weights_only=True)
    mp = nn.BondMessagePassing(**chemeleon_data["hyper_parameters"])
    mp.load_state_dict(chemeleon_data["state_dict"])
    return mp


def main() -> None:
    with open(DATA_DIR / "split_config.json") as f:
        split_config = json.load(f)
    n_replicates = split_config["n_replicates"]
    n_folds = split_config["n_folds"]

    rlm_split = np.load(DATA_DIR / "rlm_splits.npz", allow_pickle=True)
    rlm_smiles = list(rlm_split["smiles"])
    rlm_labels = rlm_split["labels"]
    rlm_folds = rlm_split["folds"]

    logger.info(f"RLM: {len(rlm_smiles)} molecules, {n_replicates}x{n_folds} CV")

    all_results: list[dict] = []

    for rep in range(n_replicates):
        fold_assignments = rlm_folds[rep]

        for fold in range(n_folds):
            test_mask = fold_assignments == fold
            train_mask = ~test_mask
            y_test = rlm_labels[test_mask]

            logger.info(
                f"rep={rep} fold={fold}: "
                f"train={train_mask.sum()}, test={test_mask.sum()}"
            )

            # --- Build dataloaders (shared by Chemprop + CheMeleon) ---
            cp_cache = CACHE_DIR / f"{make_cache_key('chemprop', rep, fold)}.npz"
            cm_cache = CACHE_DIR / f"{make_cache_key('chemeleon', rep, fold)}.npz"
            need_loaders = not cp_cache.exists() or not cm_cache.exists()

            train_loader = None
            val_loader = None
            test_loader = None

            if need_loaders:
                train_smi = [
                    rlm_smiles[i] for i in range(len(rlm_smiles)) if train_mask[i]
                ]
                train_y = rlm_labels[train_mask].reshape(-1, 1).astype(float)
                test_smi = [
                    rlm_smiles[i] for i in range(len(rlm_smiles)) if test_mask[i]
                ]
                test_y = rlm_labels[test_mask].reshape(-1, 1).astype(float)

                n = len(train_smi)
                n_val = max(1, int(n * 0.1))
                rng = np.random.default_rng(42)
                perm = rng.permutation(n)

                train_data = [
                    chemprop_data.MoleculeDatapoint.from_smi(train_smi[i], train_y[i])
                    for i in perm[n_val:]
                ]
                val_data = [
                    chemprop_data.MoleculeDatapoint.from_smi(train_smi[i], train_y[i])
                    for i in perm[:n_val]
                ]
                test_data = [
                    chemprop_data.MoleculeDatapoint.from_smi(s, y)
                    for s, y in zip(test_smi, test_y)
                ]

                train_dset = chemprop_data.MoleculeDataset(train_data, featurizer)
                val_dset = chemprop_data.MoleculeDataset(val_data, featurizer)
                test_dset = chemprop_data.MoleculeDataset(test_data, featurizer)

                train_loader = chemprop_data.build_dataloader(
                    train_dset, num_workers=0, batch_size=64
                )
                val_loader = chemprop_data.build_dataloader(
                    val_dset, num_workers=0, shuffle=False, batch_size=64
                )
                test_loader = chemprop_data.build_dataloader(
                    test_dset, num_workers=0, shuffle=False, batch_size=64
                )

            # --- Chemprop scratch ---
            if cp_cache.exists():
                y_prob_cp = np.load(cp_cache)["y_prob"]
                logger.info("  Chemprop: cached")
            else:
                mp = nn.BondMessagePassing()
                agg = nn.MeanAggregation()
                ffn = nn.BinaryClassificationFFN(input_dim=mp.output_dim)
                model_cp = models.MPNN(mp, agg, ffn, batch_norm=False)

                trainer = lightning_pl.Trainer(
                    logger=False,
                    enable_checkpointing=False,
                    enable_progress_bar=False,
                    accelerator="gpu",
                    devices=1,
                    max_epochs=MAX_EPOCHS_CHEMPROP,
                )
                trainer.fit(model_cp, train_loader, val_loader)
                preds = trainer.predict(model_cp, test_loader)
                y_prob_cp = torch.cat(preds).cpu().numpy().flatten()
                np.savez_compressed(cp_cache, y_prob=y_prob_cp)
                logger.info("  Chemprop: trained")

            metrics_cp = evaluate_predictions(y_test, y_prob_cp)
            all_results.append(
                {
                    "target": "RLM Stability",
                    "model": "Chemprop scratch",
                    "replicate": rep,
                    "fold": fold,
                    **metrics_cp,
                }
            )

            # --- CheMeleon single-finetune ---
            if cm_cache.exists():
                y_prob_cm = np.load(cm_cache)["y_prob"]
                logger.info("  CheMeleon: cached")
            else:
                cm_mp = load_chemeleon_mp()
                cm_agg = nn.MeanAggregation()
                cm_ffn = nn.BinaryClassificationFFN(input_dim=cm_mp.output_dim)
                model_cm = models.MPNN(cm_mp, cm_agg, cm_ffn, batch_norm=False)

                trainer = lightning_pl.Trainer(
                    logger=False,
                    enable_checkpointing=False,
                    enable_progress_bar=False,
                    accelerator="gpu",
                    devices=1,
                    max_epochs=MAX_EPOCHS_CHEMELEON,
                )
                trainer.fit(model_cm, train_loader, val_loader)
                preds = trainer.predict(model_cm, test_loader)
                y_prob_cm = torch.cat(preds).cpu().numpy().flatten()
                np.savez_compressed(cm_cache, y_prob=y_prob_cm)
                logger.info("  CheMeleon: trained")

            metrics_cm = evaluate_predictions(y_test, y_prob_cm)
            all_results.append(
                {
                    "target": "RLM Stability",
                    "model": "CheMeleon single-finetune",
                    "replicate": rep,
                    "fold": fold,
                    **metrics_cm,
                }
            )

            logger.info(
                f"  Chemprop={metrics_cp['auc_roc']:.3f} "
                f"CheMeleon={metrics_cm['auc_roc']:.3f}"
            )

    nn_df = pl.DataFrame(all_results)
    nn_path = DATA_DIR / "rlm_base_nn_results.parquet"
    nn_df.write_parquet(nn_path)
    logger.info(f"Saved {nn_df.height} NN results to {nn_path}")

    # --- Merge with XGBoost results into final parquet ---
    xgb_path = DATA_DIR / "rlm_base_xgb_results.parquet"
    if not xgb_path.exists():
        logger.warning(
            f"{xgb_path} not found -- run run-rlm-base-eval-xgb.py first. "
            "Saving NN-only results for now."
        )
        merged = nn_df
    else:
        xgb_df = pl.read_parquet(xgb_path)
        merged = pl.concat([xgb_df, nn_df])
        logger.info(
            f"Merged {xgb_df.height} XGB + {nn_df.height} NN = {merged.height} total"
        )

    out_path = DATA_DIR / "rlm_base_results.parquet"
    merged.write_parquet(out_path)
    logger.info(f"Saved merged results to {out_path}")

    summary = (
        merged.group_by("model")
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
