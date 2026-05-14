"""Pre-compute MIST [CLS] embeddings for all SMILES used in the analysis.

The MIST encoder weights are frozen for this analysis, so embeddings are
deterministic functions of SMILES. Caching them once eliminates ~25 * 6 *
3 redundant forward passes through a 28M-parameter transformer during the
data-efficiency loop.

Outputs ``data/mist_embeddings.npz`` with two arrays:

    smiles      : object array of canonical kekulized SMILES (the cache key)
    embeddings  : float32 array of shape (N, hidden_size)

Re-running the script is idempotent: if ``data/mist_embeddings.npz``
already exists and contains every SMILES from the endpoint splits, the
script exits early.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import smirk  # noqa: F401  -- registers SmirkTokenizerFast with transformers
import torch
from loguru import logger
from rdkit import Chem
from transformers import AutoModel, AutoTokenizer

DATA_DIR = Path("data")
MODEL_NAME = "mist-models/mist-28M-ti624ev1"
CACHE_PATH = DATA_DIR / "mist_embeddings.npz"
BATCH_SIZE = 16
MAX_LEN = 512


def kekulize(smiles: str) -> str:
    """Return the kekulized canonical SMILES, falling back to input on failure.

    MIST was pre-trained on kekulized SMILES, so we match that
    pre-processing.

    Example:
        >>> kekulize("c1ccccc1")
        'C1=CC=CC=C1'
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    Chem.Kekulize(mol)
    return Chem.MolToSmiles(mol, kekuleSmiles=True)


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
    tokenizer,
    model,
    device: torch.device,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """Run the MIST encoder over kekulized SMILES; return [CLS] embeddings.

    Returns a (len(smiles), hidden_size) float32 array.
    """
    model.eval()
    out_chunks: list[np.ndarray] = []
    kek = [kekulize(s) for s in smiles]
    n_batches = (len(kek) + batch_size - 1) // batch_size
    with torch.no_grad():
        for bi, start in enumerate(range(0, len(kek), batch_size)):
            chunk = kek[start : start + batch_size]
            inputs = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            cls = outputs.last_hidden_state[:, 0, :].detach().to("cpu").float().numpy()
            out_chunks.append(cls)
            if (bi + 1) % 25 == 0 or bi == n_batches - 1:
                logger.info(f"Embedded {start + len(chunk)}/{len(kek)}")
    return np.concatenate(out_chunks, axis=0)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target_smiles = collect_unique_smiles()
    logger.info(f"Found {len(target_smiles)} unique SMILES across endpoints")

    if CACHE_PATH.exists():
        cached = np.load(CACHE_PATH, allow_pickle=True)
        cached_smiles = set(map(str, cached["smiles"]))
        if cached_smiles >= set(target_smiles):
            logger.info(f"Cache already covers all SMILES, exiting early ({CACHE_PATH})")
            return
        missing = sorted(set(target_smiles) - cached_smiles)
        logger.info(f"Cache exists but missing {len(missing)} SMILES; re-computing all")

    device = pick_device()
    logger.info(f"Loading {MODEL_NAME} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)

    embeddings = embed_smiles(target_smiles, tokenizer, model, device)
    logger.info(f"Embeddings shape: {embeddings.shape}")

    np.savez(
        CACHE_PATH,
        smiles=np.asarray(target_smiles, dtype=object),
        embeddings=embeddings.astype(np.float32),
    )
    logger.info(f"Saved {CACHE_PATH}")


if __name__ == "__main__":
    main()
