"""Pre-compute CheMeleon pooled-graph embeddings for the analysis SMILES.

The CheMeleon encoder is held frozen for the data-efficiency sweep, so its
pooled output (mean-aggregated D-MPNN message-passing vectors, dim=2048)
is a deterministic function of SMILES. Caching once eliminates redundant
forward passes through the 9.3M-parameter encoder during the ``deep
head`` comparison loop.

Outputs ``data/chemeleon_embeddings.npz`` with two arrays:

    smiles      : object array of canonical SMILES (the cache key)
    embeddings  : float32 array of shape (N, 2048)

Re-running is idempotent: if the cache already covers every SMILES from
the endpoint splits, the script exits early.

Usage:
    uv run python scripts/run-chemeleon-embed.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from loguru import logger

from chemprop import data as chemprop_data
from chemprop import featurizers, nn

DATA_DIR = Path("data")
CHECKPOINTS_DIR = Path("checkpoints")
CACHE_PATH = DATA_DIR / "chemeleon_embeddings.npz"
WEIGHTS_PATH = CHECKPOINTS_DIR / "chemeleon_mp.pt"
BATCH_SIZE = 64


def collect_unique_smiles() -> list[str]:
    """Read every SMILES list from the endpoint split files and dedupe."""
    seen: dict[str, None] = {}
    for endpoint in ("rlm", "hlm", "pampa"):
        split = np.load(DATA_DIR / f"{endpoint}_splits.npz", allow_pickle=True)
        for s in split["smiles"]:
            seen.setdefault(str(s), None)
    return list(seen.keys())


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def embed_smiles(
    smiles: list[str],
    mp_module: nn.BondMessagePassing,
    agg_module: nn.MeanAggregation,
    featurizer: featurizers.SimpleMoleculeMolGraphFeaturizer,
    device: torch.device,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """Run the CheMeleon encoder + mean aggregation; return pooled embeddings.

    Returns a (len(smiles), mp.output_dim) float32 array, indexed in the
    same order as ``smiles``.
    """
    mp_module.eval()
    agg_module.eval()
    out_chunks: list[np.ndarray] = []

    # Build a single dataset/loader for all SMILES; predictions are y=0.0 placeholder.
    datapoints = [
        chemprop_data.MoleculeDatapoint.from_smi(s, [0.0]) for s in smiles
    ]
    dataset = chemprop_data.MoleculeDataset(datapoints, featurizer)
    loader = chemprop_data.build_dataloader(
        dataset, num_workers=0, batch_size=batch_size, shuffle=False
    )

    n_total = len(smiles)
    n_done = 0
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            bmg = batch[0]
            # Move graph tensors to device
            bmg.V = bmg.V.to(device)
            bmg.E = bmg.E.to(device)
            bmg.edge_index = bmg.edge_index.to(device)
            bmg.rev_edge_index = bmg.rev_edge_index.to(device)
            bmg.batch = bmg.batch.to(device)
            h = mp_module(bmg)
            z = agg_module(h, bmg.batch)
            out_chunks.append(z.detach().to("cpu").float().numpy())
            n_done += z.shape[0]
            if (bi + 1) % 25 == 0 or n_done == n_total:
                logger.info(f"Embedded {n_done}/{n_total}")
    return np.concatenate(out_chunks, axis=0)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target_smiles = collect_unique_smiles()
    logger.info(f"Found {len(target_smiles)} unique SMILES across endpoints")

    if CACHE_PATH.exists():
        cached = np.load(CACHE_PATH, allow_pickle=True)
        cached_smiles = set(map(str, cached["smiles"]))
        if cached_smiles >= set(target_smiles):
            logger.info(
                f"Cache already covers all SMILES, exiting early ({CACHE_PATH})"
            )
            return
        missing = sorted(set(target_smiles) - cached_smiles)
        logger.info(
            f"Cache exists but missing {len(missing)} SMILES; re-computing all"
        )

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing CheMeleon weights at {WEIGHTS_PATH}. Pull the checkpoints/ "
            "directory from the GPU machine."
        )

    device = pick_device()
    logger.info(f"Loading CheMeleon weights on {device}")
    ck = torch.load(WEIGHTS_PATH, weights_only=True)
    mp_module = nn.BondMessagePassing(**ck["hyper_parameters"])
    mp_module.load_state_dict(ck["state_dict"])
    for p in mp_module.parameters():
        p.requires_grad = False
    mp_module = mp_module.to(device)
    agg_module = nn.MeanAggregation().to(device)
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    embeddings = embed_smiles(
        smiles=target_smiles,
        mp_module=mp_module,
        agg_module=agg_module,
        featurizer=featurizer,
        device=device,
    )
    logger.info(f"Embeddings shape: {embeddings.shape}")

    np.savez(
        CACHE_PATH,
        smiles=np.asarray(target_smiles, dtype=object),
        embeddings=embeddings.astype(np.float32),
    )
    logger.info(f"Saved {CACHE_PATH}")


if __name__ == "__main__":
    main()
