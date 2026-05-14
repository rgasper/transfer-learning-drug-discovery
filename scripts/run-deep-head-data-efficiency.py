"""Run frozen MIST and frozen CheMeleon with 1-layer and 2-layer FFN heads.

Companion to ``scripts/run-mist-data-efficiency.py``. Sweeps the same
(endpoint, fraction, replicate, fold) grid for both encoders, with both
shallow (Linear -> ReLU -> Linear) and deep (Linear -> ReLU -> Linear ->
ReLU -> Linear) heads, on top of pre-cached pooled embeddings.

Pre-requisites:
    uv run python scripts/run-mist-embed.py
    uv run python scripts/run-chemeleon-embed.py

Output: ``data/deep_head_efficiency_results.parquet`` with the same schema
as the existing data-efficiency parquets, so the four new arms can be
plotted alongside the original XGBoost/Chemprop baselines:

    pl.concat([
        pl.read_parquet("data/data_efficiency_results.parquet"),
        pl.read_parquet("data/deep_head_efficiency_results.parquet"),
    ])

Model labels written to the parquet:
    - "MIST frozen 1-layer"
    - "MIST frozen 2-layer"
    - "CheMeleon frozen 1-layer"
    - "CheMeleon frozen 2-layer"

The 1-layer arm replicates the head used in
``run-mist-data-efficiency.py``; the 2-layer arm adds an extra hidden
block of width 300 between the input and output layer.

Usage:
    uv run python scripts/run-deep-head-data-efficiency.py
"""

from __future__ import annotations

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
OUTPUT_PATH = DATA_DIR / "deep_head_efficiency_results.parquet"
SPLIT_CONFIG_PATH = DATA_DIR / "split_config.json"
ENDPOINTS = ("rlm", "hlm", "pampa")
FRACTIONS = (0.01, 0.10, 0.25, 0.50, 0.75, 1.00)

# (label, cache filename) for each frozen encoder we want to evaluate.
ENCODERS = (
    ("MIST frozen", "mist_embeddings.npz"),
    ("CheMeleon frozen", "chemeleon_embeddings.npz"),
)

# Head depths to compare. ``hidden_layers`` counts the number of ReLU-Linear
# blocks before the final classification linear layer; 1 = the existing
# Chemprop default head, 2 = one additional hidden block.
HEAD_CONFIGS = (
    ("1-layer", 1),
    ("2-layer", 2),
)


@dataclass(frozen=True)
class EncoderArrays:
    """Pre-computed pooled embeddings, keyed by SMILES."""

    smiles_to_idx: dict
    embeddings: np.ndarray  # (N_unique, hidden_size)


@dataclass(frozen=True)
class EndpointArrays:
    """Per-endpoint label / fold arrays, with rows aligned to the encoder cache."""

    labels: np.ndarray
    folds: np.ndarray
    encoder_x: dict  # encoder_label -> (N, hidden_size) array


@typechecked
def load_encoder(label: str, cache_filename: str) -> EncoderArrays:
    path = DATA_DIR / cache_filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {label} embedding cache at {path}. "
            "Run scripts/run-mist-embed.py and scripts/run-chemeleon-embed.py first."
        )
    data = np.load(path, allow_pickle=True)
    smiles_to_idx = {str(s): i for i, s in enumerate(data["smiles"])}
    embeddings = np.asarray(data["embeddings"])
    return EncoderArrays(smiles_to_idx=smiles_to_idx, embeddings=embeddings)


@typechecked
def load_endpoint(name: str, encoders: dict) -> EndpointArrays:
    """Load split arrays and align to each encoder's row order."""
    split = np.load(DATA_DIR / f"{name}_splits.npz", allow_pickle=True)
    smiles = [str(s) for s in split["smiles"]]
    encoder_x: dict = {}
    for label, enc in encoders.items():
        idx = np.array([enc.smiles_to_idx[s] for s in smiles], dtype=np.int64)
        encoder_x[label] = enc.embeddings[idx]
    return EndpointArrays(
        labels=np.asarray(split["labels"]),
        folds=np.asarray(split["folds"]),
        encoder_x=encoder_x,
    )


def build_head(d_in: int, hidden_layers: int, hidden_dim: int = 300) -> torch.nn.Module:
    """Construct a feed-forward head.

    ``hidden_layers=1`` mirrors ``chemprop.nn.BinaryClassificationFFN`` defaults:
    ``Linear(d_in, 300) -> ReLU -> Linear(300, 1)``.

    ``hidden_layers=2`` adds one extra hidden block:
    ``Linear(d_in, 300) -> ReLU -> Linear(300, 300) -> ReLU -> Linear(300, 1)``.
    """
    layers: list[torch.nn.Module] = [torch.nn.Linear(d_in, hidden_dim), torch.nn.ReLU()]
    for _ in range(hidden_layers - 1):
        layers.extend([torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU()])
    layers.append(torch.nn.Linear(hidden_dim, 1))
    return torch.nn.Sequential(*layers)


@typechecked
def train_head(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    hidden_layers: int,
    max_epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 5,
) -> np.ndarray:
    """Train an FFN head on cached pooled embeddings; return test probabilities.

    Same training recipe as ``run-mist-data-efficiency.train_mist_head``: Adam
    optimiser, BCE-with-logits loss, early stopping on validation BCE.
    """
    torch.manual_seed(seed)
    device = torch.device("cpu")
    d_in = int(x_train.shape[1])
    head = build_head(d_in, hidden_layers=hidden_layers).to(device)
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


def main() -> None:
    with open(SPLIT_CONFIG_PATH) as f:
        split_config = json.load(f)
    n_reps = int(split_config["n_replicates"])
    n_folds = int(split_config["n_folds"])

    encoders = {label: load_encoder(label, fname) for label, fname in ENCODERS}
    for label, enc in encoders.items():
        logger.info(
            f"{label}: embeddings={enc.embeddings.shape}, "
            f"unique_smiles={len(enc.smiles_to_idx)}"
        )

    endpoints = {name: load_endpoint(name, encoders) for name in ENDPOINTS}
    for name, ep in endpoints.items():
        logger.info(
            f"{name.upper()}: N={len(ep.labels)} positives={int(ep.labels.sum())}"
        )

    rows = []
    total_units = (
        len(endpoints) * len(FRACTIONS) * n_reps * n_folds * len(ENCODERS) * len(HEAD_CONFIGS)
    )
    done = 0

    for endpoint_name, ep in endpoints.items():
        labels = ep.labels
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

                    n = len(sub_indices)
                    n_val = max(2, int(n * 0.1))
                    perm = np.random.default_rng(42).permutation(n)
                    train_idx_in_sub = perm[n_val:]
                    val_idx_in_sub = perm[:n_val]

                    y_train = labels[sub_indices[train_idx_in_sub]].astype(np.float32)
                    y_val = labels[sub_indices[val_idx_in_sub]].astype(np.float32)

                    for encoder_label, _ in ENCODERS:
                        x_full = ep.encoder_x[encoder_label]
                        x_train = x_full[sub_indices[train_idx_in_sub]]
                        x_val = x_full[sub_indices[val_idx_in_sub]]
                        x_test = x_full[test_mask]

                        for head_label, hidden_layers in HEAD_CONFIGS:
                            probs = train_head(
                                x_train=x_train,
                                y_train=y_train,
                                x_val=x_val,
                                y_val=y_val,
                                x_test=x_test,
                                seed=42 + rep * 100 + fold,
                                hidden_layers=hidden_layers,
                            )
                            model_name = f"{encoder_label} {head_label}"
                            rows.append(
                                {
                                    "endpoint": endpoint_name,
                                    "fraction": float(frac),
                                    "pct_label": f"{int(frac * 100)}%",
                                    "model": model_name,
                                    "replicate": int(rep),
                                    "fold": int(fold),
                                    "n_train": int(n_sub),
                                    "auc_roc": float(roc_auc_score(y_test, probs)),
                                    "avg_precision": float(
                                        average_precision_score(y_test, probs)
                                    ),
                                }
                            )
                            done += 1

                    if done % 50 == 0 or done == total_units:
                        logger.info(
                            f"[{done}/{total_units}] {endpoint_name.upper()} "
                            f"frac={int(frac * 100)}% rep={rep} fold={fold} "
                            f"n_train={n_sub} last={rows[-1]['model']} "
                            f"AUC-PR={rows[-1]['avg_precision']:.3f}"
                        )

    df = pl.DataFrame(rows)
    df.write_parquet(OUTPUT_PATH)
    logger.info(f"Saved {df.height} rows to {OUTPUT_PATH}")

    summary = (
        df.group_by("endpoint", "fraction", "model")
        .agg(
            pl.col("avg_precision").mean().round(3).alias("mean_aucpr"),
            pl.col("avg_precision").std().round(3).alias("std_aucpr"),
        )
        .sort("endpoint", "fraction", "model")
    )
    logger.info(f"\n{summary}")


if __name__ == "__main__":
    main()
