"""Run MIST frozen single-finetune across the data-efficiency sweep.

Mirrors the (endpoint, fraction, replicate, fold) loop in
``notebooks/15-data-efficiency.py`` but only for the MIST frozen arm.
Loads cached MIST [CLS] embeddings (run ``scripts/run-mist-embed.py`` first)
and trains a Chemprop-shaped FFN head on top.

Supports both MIST-28M and MIST-1.8B via the ``--size`` CLI argument.

Output: ``data/mist_efficiency_results.parquet`` (28M) or
``data/mist_1.8b_efficiency_results.parquet`` (1.8B) with the same schema
as the notebook's ``data_efficiency_results.parquet``, which lets you
concat them:

    full = pl.concat([
        pl.read_parquet("data/data_efficiency_results.parquet"),
        pl.read_parquet("data/mist_efficiency_results.parquet"),
        pl.read_parquet("data/mist_1.8b_efficiency_results.parquet"),
    ])

Usage:
    uv run python scripts/run-mist-data-efficiency.py             # 28M (default)
    uv run python scripts/run-mist-data-efficiency.py --size 1.8B # 1.8B
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
from loguru import logger
from sklearn.metrics import average_precision_score, roc_auc_score
from typeguard import typechecked

DATA_DIR = Path("data")
SPLIT_CONFIG_PATH = DATA_DIR / "split_config.json"
ENDPOINT_BASELINES = {"rlm": 0.298, "hlm": 0.602, "pampa": 0.855}
FRACTIONS = (0.01, 0.10, 0.25, 0.50, 0.75, 1.00)

VARIANTS: dict[str, dict[str, str]] = {
    "28M": {
        "cache_filename": "mist_embeddings.npz",
        "output_filename": "mist_efficiency_results.parquet",
        "model_label": "MIST frozen",
    },
    "1.8B": {
        "cache_filename": "mist_1.8b_embeddings.npz",
        "output_filename": "mist_1.8b_efficiency_results.parquet",
        "model_label": "MIST-1.8B frozen",
    },
}


@dataclass(frozen=True)
class EndpointArrays:
    """In-memory data for one ADME endpoint, aligned to MIST embeddings.

    Attributes:
        labels: Binary labels of shape (N,).
        folds: Fold-assignment matrix of shape (n_replicates, N); each row is
            an integer-valued fold id in [0, n_folds).
        x_mist: MIST [CLS] embeddings of shape (N, hidden_size).
    """

    labels: np.ndarray
    folds: np.ndarray
    x_mist: np.ndarray


@typechecked
def load_endpoint(name: str, mist_smiles_to_idx: dict, mist_embeddings: np.ndarray) -> EndpointArrays:
    """Load split arrays for one endpoint and align to MIST embedding rows."""
    split = np.load(DATA_DIR / f"{name}_splits.npz", allow_pickle=True)
    smiles = [str(s) for s in split["smiles"]]
    mist_idx = np.array([mist_smiles_to_idx[s] for s in smiles], dtype=np.int64)
    return EndpointArrays(
        labels=np.asarray(split["labels"]),
        folds=np.asarray(split["folds"]),
        x_mist=mist_embeddings[mist_idx],
    )


@typechecked
def train_mist_head(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    max_epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 5,
) -> np.ndarray:
    """Train a Chemprop-shaped FFN head on MIST embeddings; return test probs.

    Architecture mirrors ``chemprop.nn.BinaryClassificationFFN`` defaults:
    Linear(d -> 300) -> ReLU -> Linear(300 -> 1) with BCE-with-logits loss
    and a sigmoid at predict-time. Adam optimiser, early stopping on
    validation BCE.

    Returns:
        Predicted probabilities for ``x_test``, shape (N_test,).
    """
    torch.manual_seed(seed)
    device = torch.device("cpu")  # head is tiny; CPU avoids MPS roundtrip overhead
    d_in = int(x_train.shape[1])

    head = torch.nn.Sequential(
        torch.nn.Linear(d_in, 300),
        torch.nn.ReLU(),
        torch.nn.Linear(300, 1),
    ).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    x_train_t = torch.from_numpy(x_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).float().to(device)
    x_val_t = torch.from_numpy(x_val).float().to(device)
    y_val_t = torch.from_numpy(y_val).float().to(device)
    x_test_t = torch.from_numpy(x_test).float().to(device)

    n_train = len(x_train_t)
    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        head.train()
        perm = torch.randperm(
            n_train, generator=torch.Generator().manual_seed(seed + epoch)
        )
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            logits = head(x_train_t[idx]).squeeze(-1)
            loss = loss_fn(logits, y_train_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

        head.eval()
        with torch.no_grad():
            val_logits = head(x_val_t).squeeze(-1)
            val_loss = loss_fn(val_logits, y_val_t).item()
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        test_probs = torch.sigmoid(head(x_test_t).squeeze(-1)).cpu().numpy()
    return test_probs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MIST frozen data-efficiency sweep."
    )
    parser.add_argument(
        "--size",
        choices=list(VARIANTS.keys()),
        default="28M",
        help="MIST model size (default: 28M). Must have run "
        "run-mist-embed.py --size <SIZE> first.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variant = VARIANTS[args.size]
    cache_filename = variant["cache_filename"]
    output_filename = variant["output_filename"]
    model_label = variant["model_label"]

    cache_path = DATA_DIR / cache_filename
    output_path = DATA_DIR / output_filename

    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing {cache_path}. Run scripts/run-mist-embed.py --size {args.size} first."
        )

    with open(SPLIT_CONFIG_PATH) as f:
        split_config = json.load(f)
    n_reps = int(split_config["n_replicates"])
    n_folds = int(split_config["n_folds"])

    mist = np.load(cache_path, allow_pickle=True)
    mist_smiles_to_idx = {str(s): i for i, s in enumerate(mist["smiles"])}
    mist_embeddings = np.asarray(mist["embeddings"])
    logger.info(
        f"Loaded MIST-{args.size} embeddings: {mist_embeddings.shape}, "
        f"{len(mist_smiles_to_idx)} unique SMILES"
    )

    endpoints = {
        name: load_endpoint(name, mist_smiles_to_idx, mist_embeddings)
        for name in ENDPOINT_BASELINES
    }
    for name, ep in endpoints.items():
        logger.info(
            f"{name.upper()}: N={len(ep.labels)} positives={int(ep.labels.sum())} "
            f"X_mist={ep.x_mist.shape}"
        )

    rows = []
    total_units = len(endpoints) * len(FRACTIONS) * n_reps * n_folds
    done = 0

    for endpoint_name, ep in endpoints.items():
        labels = ep.labels
        x_mist = ep.x_mist
        folds_arr = ep.folds

        for frac in FRACTIONS:
            for rep in range(n_reps):
                fold_assign = folds_arr[rep]
                for fold in range(n_folds):
                    test_mask = fold_assign == fold
                    train_mask = ~test_mask
                    y_test = labels[test_mask]

                    train_indices = np.where(train_mask)[0]
                    rng = np.random.default_rng(42 + rep * 100 + fold)
                    n_sub = max(10, int(len(train_indices) * frac))
                    sub_indices = rng.choice(train_indices, size=n_sub, replace=False)

                    # Match the notebook's train/val partition: the same
                    # 0.1 holdout slice, drawn from a fixed permutation seed.
                    n = len(sub_indices)
                    n_val = max(2, int(n * 0.1))
                    perm = np.random.default_rng(42).permutation(n)
                    train_idx_in_sub = perm[n_val:]
                    val_idx_in_sub = perm[:n_val]

                    x_train = x_mist[sub_indices[train_idx_in_sub]]
                    y_train = labels[sub_indices[train_idx_in_sub]].astype(np.float32)
                    x_val = x_mist[sub_indices[val_idx_in_sub]]
                    y_val = labels[sub_indices[val_idx_in_sub]].astype(np.float32)
                    x_test = x_mist[test_mask]

                    probs = train_mist_head(
                        x_train=x_train,
                        y_train=y_train,
                        x_val=x_val,
                        y_val=y_val,
                        x_test=x_test,
                        seed=42 + rep * 100 + fold,
                    )

                    rows.append(
                        {
                            "endpoint": endpoint_name,
                            "fraction": float(frac),
                            "pct_label": f"{int(frac * 100)}%",
                            "model": model_label,
                            "replicate": int(rep),
                            "fold": int(fold),
                            "n_train": int(n_sub),
                            "auc_roc": float(roc_auc_score(y_test, probs)),
                            "avg_precision": float(average_precision_score(y_test, probs)),
                        }
                    )
                    done += 1
                    if done % 25 == 0 or done == total_units:
                        logger.info(
                            f"[{done}/{total_units}] {endpoint_name.upper()} "
                            f"frac={int(frac * 100)}% rep={rep} fold={fold} "
                            f"n_train={n_sub} AUC-PR={rows[-1]['avg_precision']:.3f}"
                        )

    df = pl.DataFrame(rows)
    df.write_parquet(output_path)
    logger.info(f"Saved {df.height} rows to {output_path}")

    # Brief summary
    summary = (
        df.group_by("endpoint", "fraction")
        .agg(
            pl.col("avg_precision").mean().round(3).alias("mean_aucpr"),
            pl.col("avg_precision").std().round(3).alias("std_aucpr"),
        )
        .sort("endpoint", "fraction")
    )
    logger.info(f"\n{summary}")


if __name__ == "__main__":
    main()
