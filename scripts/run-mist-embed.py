"""Pre-compute MIST [CLS] embeddings for all SMILES used in the analysis.

The MIST encoder weights are frozen for this analysis, so embeddings are
deterministic functions of SMILES. Caching them once eliminates ~25 * 6 *
3 redundant forward passes through the transformer during the
data-efficiency loop.

Supports both MIST-28M and MIST-1.8B variants. The model size is selected
via the ``--size`` CLI argument (default: ``28M``).

Outputs ``data/mist_embeddings.npz`` (28M) or
``data/mist_1.8b_embeddings.npz`` (1.8B) with two arrays:

    smiles      : object array of canonical kekulized SMILES (the cache key)
    embeddings  : float32 array of shape (N, hidden_size)

Re-running the script is idempotent: if the cache already exists and
contains every SMILES from the endpoint splits, the script exits early.

Usage:
    uv run python scripts/run-mist-embed.py            # 28M (default)
    uv run python scripts/run-mist-embed.py --size 1.8B  # 1.8B (needs ~4 GB VRAM in fp16)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import smirk  # noqa: F401  -- registers SmirkTokenizerFast with transformers
import torch
from loguru import logger
from rdkit import Chem
from transformers import AutoModel, AutoTokenizer

DATA_DIR = Path("data")

MODELS: dict[str, dict[str, object]] = {
    "28M": {
        "hf_name": "mist-models/mist-28M-ti624ev1",
        "cache_filename": "mist_embeddings.npz",
        "batch_size": 16,
        "use_fp16": False,
    },
    "1.8B": {
        "hf_name": "mist-models/mist-1.8B-dh61satt",
        "cache_filename": "mist_1.8b_embeddings.npz",
        "batch_size": 4,
        "use_fp16": True,
    },
}

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
    batch_size: int,
    use_fp16: bool,
) -> np.ndarray:
    """Run the MIST encoder over kekulized SMILES; return [CLS] embeddings.

    Returns a (len(smiles), hidden_size) float32 array.
    """
    model.eval()
    out_chunks: list[np.ndarray] = []
    kek = [kekulize(s) for s in smiles]
    n_batches = (len(kek) + batch_size - 1) // batch_size
    autocast_ctx = (
        torch.amp.autocast(device_type=device.type, dtype=torch.float16)
        if use_fp16 and device.type in ("cuda", "cpu")
        else torch.amp.autocast(device_type=device.type, enabled=False)
    )
    with torch.no_grad(), autocast_ctx:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache MIST [CLS] embeddings.")
    parser.add_argument(
        "--size",
        choices=list(MODELS.keys()),
        default="28M",
        help="MIST model size to embed with (default: 28M)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = MODELS[args.size]
    hf_name: str = cfg["hf_name"]  # type: ignore[assignment]
    cache_filename: str = cfg["cache_filename"]  # type: ignore[assignment]
    batch_size: int = cfg["batch_size"]  # type: ignore[assignment]
    use_fp16: bool = cfg["use_fp16"]  # type: ignore[assignment]
    cache_path = DATA_DIR / cache_filename

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target_smiles = collect_unique_smiles()
    logger.info(
        f"Found {len(target_smiles)} unique SMILES across endpoints "
        f"(model: MIST-{args.size})"
    )

    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        cached_smiles = set(map(str, cached["smiles"]))
        if cached_smiles >= set(target_smiles):
            logger.info(f"Cache already covers all SMILES, exiting early ({cache_path})")
            return
        missing = sorted(set(target_smiles) - cached_smiles)
        logger.info(f"Cache exists but missing {len(missing)} SMILES; re-computing all")

    device = pick_device()
    logger.info(f"Loading {hf_name} on {device}")

    load_kwargs: dict[str, object] = {"trust_remote_code": True}
    if use_fp16 and device.type == "cuda":
        load_kwargs["torch_dtype"] = torch.float16
        logger.info("Loading model weights in float16 to conserve VRAM")

    tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(hf_name, **load_kwargs).to(device)

    embeddings = embed_smiles(
        target_smiles, tokenizer, model, device, batch_size=batch_size, use_fp16=use_fp16
    )
    logger.info(f"Embeddings shape: {embeddings.shape}")

    np.savez(
        cache_path,
        smiles=np.asarray(target_smiles, dtype=object),
        embeddings=embeddings.astype(np.float32),
    )
    logger.info(f"Saved {cache_path}")


if __name__ == "__main__":
    main()
