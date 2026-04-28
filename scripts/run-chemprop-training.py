"""Run Chemprop CV training with disk caching.

Trains Chemprop D-MPNN models (scratch + RLM transfer) across all folds
and saves predictions to disk. The marimo notebook loads results from
the saved parquet file.

Usage:
    uv run python scripts/run-chemprop-training.py
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
CHECKPOINTS_DIR.mkdir(exist_ok=True)
CACHE_DIR = DATA_DIR / "chemprop_cache"
CACHE_DIR.mkdir(exist_ok=True)

ENDPOINT_NAMES = {
    "rlm": "RLM Stability",
    "hlm": "HLM Stability",
    "pampa": "PAMPA pH 7.4",
}
TARGET_ENDPOINTS = ["hlm", "pampa"]
PRETRAIN_ENDPOINT = "rlm"

MAX_EPOCHS = 30
PRETRAIN_MAX_EPOCHS = 50

featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()


def make_cache_key(
    endpoint: str,
    model_type: str,
    rep: int,
    fold: int,
    smiles: list[str],
    labels: np.ndarray,
    fold_assignments: np.ndarray,
) -> str:
    """Generate a deterministic cache key from input data."""
    h = hashlib.sha256()
    h.update(endpoint.encode())
    h.update(model_type.encode())
    h.update(f"{rep}_{fold}".encode())
    h.update("".join(smiles).encode())
    h.update(labels.tobytes())
    h.update(fold_assignments.tobytes())
    h.update(f"epochs={MAX_EPOCHS}".encode())
    return h.hexdigest()[:16]


def make_chemprop_dataloaders(
    smiles_list: list[str],
    labels: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> tuple:
    """Build Chemprop train/val/test dataloaders."""
    train_smiles = [smiles_list[i] for i in range(len(smiles_list)) if train_mask[i]]
    train_labels = labels[train_mask].reshape(-1, 1).astype(float)
    test_smiles = [smiles_list[i] for i in range(len(smiles_list)) if test_mask[i]]
    test_labels = labels[test_mask].reshape(-1, 1).astype(float)

    n_train = len(train_smiles)
    n_val = max(1, int(n_train * 0.1))
    rng = np.random.default_rng(42)
    perm = rng.permutation(n_train)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_data = [
        chemprop_data.MoleculeDatapoint.from_smi(train_smiles[i], train_labels[i])
        for i in train_idx
    ]
    val_data = [
        chemprop_data.MoleculeDatapoint.from_smi(train_smiles[i], train_labels[i])
        for i in val_idx
    ]
    test_data = [
        chemprop_data.MoleculeDatapoint.from_smi(smi, y)
        for smi, y in zip(test_smiles, test_labels)
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

    return train_loader, val_loader, test_loader


def build_chemprop_model() -> models.MPNN:
    """Build a fresh Chemprop MPNN for binary classification."""
    mp = nn.BondMessagePassing()
    agg = nn.MeanAggregation()
    ffn = nn.BinaryClassificationFFN(input_dim=mp.output_dim)
    return models.MPNN(mp, agg, ffn, batch_norm=False)


def train_and_predict(
    mpnn: models.MPNN,
    train_loader,
    val_loader,
    test_loader,
    max_epochs: int = MAX_EPOCHS,
) -> np.ndarray:
    """Train a Chemprop model and return test predictions."""
    trainer = lightning_pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        accelerator="gpu",
        devices=1,
        max_epochs=max_epochs,
    )
    trainer.fit(mpnn, train_loader, val_loader)
    preds = trainer.predict(mpnn, test_loader)
    return torch.cat(preds, dim=0).cpu().numpy().flatten()


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
    # Load split config
    with open(DATA_DIR / "split_config.json") as f:
        split_config = json.load(f)
    n_replicates = split_config["n_replicates"]
    n_folds = split_config["n_folds"]

    # Load per-endpoint splits
    split_data: dict[str, dict] = {}
    for key in ENDPOINT_NAMES:
        split = np.load(DATA_DIR / f"{key}_splits.npz", allow_pickle=True)
        split_data[key] = {
            "smiles": list(split["smiles"]),
            "labels": split["labels"],
            "folds": split["folds"],
        }
        logger.info(f"{key}: {len(split_data[key]['smiles'])} molecules")

    # Pre-train on RLM
    rlm_checkpoint_path = CHECKPOINTS_DIR / "rlm_pretrained.ckpt"
    if not rlm_checkpoint_path.exists():
        logger.info("Pre-training on RLM...")
        rlm_smiles = split_data[PRETRAIN_ENDPOINT]["smiles"]
        rlm_labels = (
            split_data[PRETRAIN_ENDPOINT]["labels"].reshape(-1, 1).astype(float)
        )
        rlm_data = [
            chemprop_data.MoleculeDatapoint.from_smi(smi, y)
            for smi, y in zip(rlm_smiles, rlm_labels)
        ]

        n = len(rlm_data)
        n_val = max(1, int(n * 0.1))
        rng = np.random.default_rng(42)
        perm = rng.permutation(n)

        train_dset = chemprop_data.MoleculeDataset(
            [rlm_data[i] for i in perm[n_val:]], featurizer
        )
        val_dset = chemprop_data.MoleculeDataset(
            [rlm_data[i] for i in perm[:n_val]], featurizer
        )
        train_loader = chemprop_data.build_dataloader(
            train_dset, num_workers=0, batch_size=64
        )
        val_loader = chemprop_data.build_dataloader(
            val_dset, num_workers=0, shuffle=False, batch_size=64
        )

        rlm_model = build_chemprop_model()
        trainer = lightning_pl.Trainer(
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            accelerator="gpu",
            devices=1,
            max_epochs=PRETRAIN_MAX_EPOCHS,
        )
        trainer.fit(rlm_model, train_loader, val_loader)
        trainer.save_checkpoint(rlm_checkpoint_path)
        logger.info(f"Saved RLM checkpoint to {rlm_checkpoint_path}")
    else:
        logger.info(f"Using existing RLM checkpoint at {rlm_checkpoint_path}")

    # CV training loop
    all_results: list[dict] = []
    n_cached = 0
    n_trained = 0

    for target_key in TARGET_ENDPOINTS:
        smiles = split_data[target_key]["smiles"]
        labels_arr = split_data[target_key]["labels"]
        folds_matrix = split_data[target_key]["folds"]
        target_name = ENDPOINT_NAMES[target_key]

        logger.info(f"Training on {target_name} ({len(smiles)} samples)")

        for rep in range(n_replicates):
            fold_assignments = folds_matrix[rep]

            for fold in range(n_folds):
                test_mask = fold_assignments == fold
                train_mask = ~test_mask
                y_test = labels_arr[test_mask]

                cache_key_scratch = make_cache_key(
                    target_key,
                    "scratch",
                    rep,
                    fold,
                    smiles,
                    labels_arr,
                    fold_assignments,
                )
                cache_key_transfer = make_cache_key(
                    target_key,
                    "rlm_transfer",
                    rep,
                    fold,
                    smiles,
                    labels_arr,
                    fold_assignments,
                )
                cache_path_scratch = CACHE_DIR / f"{cache_key_scratch}.npz"
                cache_path_transfer = CACHE_DIR / f"{cache_key_transfer}.npz"

                both_cached = (
                    cache_path_scratch.exists() and cache_path_transfer.exists()
                )

                if both_cached:
                    y_prob_scratch = np.load(cache_path_scratch)["y_prob"]
                    y_prob_transfer = np.load(cache_path_transfer)["y_prob"]
                    n_cached += 2
                else:
                    train_loader, val_loader, test_loader = make_chemprop_dataloaders(
                        smiles,
                        labels_arr,
                        train_mask,
                        test_mask,
                    )

                    if cache_path_scratch.exists():
                        y_prob_scratch = np.load(cache_path_scratch)["y_prob"]
                        n_cached += 1
                    else:
                        model_scratch = build_chemprop_model()
                        y_prob_scratch = train_and_predict(
                            model_scratch,
                            train_loader,
                            val_loader,
                            test_loader,
                            MAX_EPOCHS,
                        )
                        np.savez_compressed(cache_path_scratch, y_prob=y_prob_scratch)
                        n_trained += 1

                    if cache_path_transfer.exists():
                        y_prob_transfer = np.load(cache_path_transfer)["y_prob"]
                        n_cached += 1
                    else:
                        model_transfer = models.MPNN.load_from_file(rlm_checkpoint_path)
                        new_ffn = nn.BinaryClassificationFFN(
                            input_dim=model_transfer.message_passing.output_dim
                        )
                        model_transfer.predictor = new_ffn
                        model_transfer.metrics = torch.nn.ModuleList(
                            [nn.metrics.BinaryAUROC()]
                        )
                        y_prob_transfer = train_and_predict(
                            model_transfer,
                            train_loader,
                            val_loader,
                            test_loader,
                            MAX_EPOCHS,
                        )
                        np.savez_compressed(cache_path_transfer, y_prob=y_prob_transfer)
                        n_trained += 1

                metrics_scratch = evaluate_predictions(y_test, y_prob_scratch)
                all_results.append(
                    {
                        "target": target_name,
                        "model": "Chemprop scratch",
                        "replicate": rep,
                        "fold": fold,
                        **metrics_scratch,
                    }
                )

                metrics_transfer = evaluate_predictions(y_test, y_prob_transfer)
                all_results.append(
                    {
                        "target": target_name,
                        "model": "Chemprop RLM-transfer",
                        "replicate": rep,
                        "fold": fold,
                        **metrics_transfer,
                    }
                )

                logger.info(
                    f"  {target_name} rep={rep} fold={fold}: "
                    f"scratch AUC={metrics_scratch['auc_roc']:.3f}, "
                    f"transfer AUC={metrics_transfer['auc_roc']:.3f}"
                    f" {'(cached)' if both_cached else ''}"
                )

        logger.info(f"Completed {n_replicates * n_folds} folds for {target_name}")

    logger.info(f"Total: {n_trained} trained, {n_cached} loaded from cache")
    results_df = pl.DataFrame(all_results)
    results_df.write_parquet(DATA_DIR / "chemprop_results.parquet")
    logger.info(
        f"Saved {results_df.height} results to {DATA_DIR / 'chemprop_results.parquet'}"
    )

    # Print summary
    summary = (
        results_df.group_by("target", "model")
        .agg(
            pl.col("auc_roc").mean().alias("auc_mean"),
            pl.col("auc_roc").std().alias("auc_std"),
        )
        .sort("target", "model")
    )
    print(summary)


if __name__ == "__main__":
    main()
