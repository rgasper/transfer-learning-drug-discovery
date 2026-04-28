import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md("""
    # 09 — Chemprop Atom Saliency Analysis

    Per-atom gradient saliency for Chemprop scratch vs RLM-transfer on PAMPA.
    Retrains both models for one fold and computes gradients of predictions
    with respect to atom features to identify which atoms drive each model's
    decision.
    """)
    return (mo,)


@app.cell
def _():
    import io
    import json
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import torch
    from lightning import pytorch as lightning_pl
    from loguru import logger
    from PIL import Image
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem.Draw import rdMolDraw2D
    from sklearn.metrics import roc_auc_score

    from chemprop import data as chemprop_data
    from chemprop import featurizers, models, nn

    DATA_DIR = Path("data")
    CHECKPOINTS_DIR = Path("checkpoints")
    return (
        CHECKPOINTS_DIR,
        Chem,
        DATA_DIR,
        Image,
        chemprop_data,
        featurizers,
        io,
        json,
        lightning_pl,
        logger,
        models,
        nn,
        np,
        pl,
        plt,
        rdMolDraw2D,
        roc_auc_score,
        torch,
    )


@app.cell
def _(DATA_DIR, json, logger, np):
    with open(DATA_DIR / "split_config.json") as _f:
        split_config = json.load(_f)

    _pampa = np.load(DATA_DIR / "pampa_splits.npz", allow_pickle=True)
    pampa_smiles = list(_pampa["smiles"])
    pampa_labels = _pampa["labels"]
    pampa_folds = _pampa["folds"]

    _rlm = np.load(DATA_DIR / "rlm_splits.npz", allow_pickle=True)
    rlm_smiles = list(_rlm["smiles"])
    rlm_labels = _rlm["labels"]

    logger.info(f"PAMPA: {len(pampa_smiles)}, RLM: {len(rlm_smiles)}")
    return pampa_folds, pampa_labels, pampa_smiles


@app.cell
def _(
    CHECKPOINTS_DIR,
    chemprop_data,
    featurizers,
    lightning_pl,
    logger,
    models,
    nn,
    np,
    pampa_folds,
    pampa_labels,
    pampa_smiles,
    roc_auc_score,
    torch,
):
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    REP, FOLD = 0, 0
    MAX_EPOCHS = 30

    _fold_assign = pampa_folds[REP]
    test_mask = _fold_assign == FOLD
    train_mask = ~test_mask
    y_test = pampa_labels[test_mask]

    # Build dataloaders
    _train_smi = [pampa_smiles[i] for i in range(len(pampa_smiles)) if train_mask[i]]
    _train_y = pampa_labels[train_mask].reshape(-1, 1).astype(float)
    _test_smi = [pampa_smiles[i] for i in range(len(pampa_smiles)) if test_mask[i]]
    _test_y = pampa_labels[test_mask].reshape(-1, 1).astype(float)

    _n = len(_train_smi)
    _n_val = max(1, int(_n * 0.1))
    _rng = np.random.default_rng(42)
    _perm = _rng.permutation(_n)

    _train_data = [
        chemprop_data.MoleculeDatapoint.from_smi(_train_smi[i], _train_y[i])
        for i in _perm[_n_val:]
    ]
    _val_data = [
        chemprop_data.MoleculeDatapoint.from_smi(_train_smi[i], _train_y[i])
        for i in _perm[:_n_val]
    ]
    _test_data = [
        chemprop_data.MoleculeDatapoint.from_smi(s, y)
        for s, y in zip(_test_smi, _test_y)
    ]

    _train_dset = chemprop_data.MoleculeDataset(_train_data, featurizer)
    _val_dset = chemprop_data.MoleculeDataset(_val_data, featurizer)
    _test_dset = chemprop_data.MoleculeDataset(_test_data, featurizer)

    train_loader = chemprop_data.build_dataloader(
        _train_dset, num_workers=0, batch_size=64
    )
    val_loader = chemprop_data.build_dataloader(
        _val_dset, num_workers=0, shuffle=False, batch_size=64
    )
    test_loader = chemprop_data.build_dataloader(
        _test_dset, num_workers=0, shuffle=False, batch_size=64
    )

    def _build_and_train(name, pretrained_path=None):
        if pretrained_path and pretrained_path.exists():
            _mpnn = models.MPNN.load_from_file(pretrained_path)
            _new_ffn = nn.BinaryClassificationFFN(
                input_dim=_mpnn.message_passing.output_dim
            )
            _mpnn.predictor = _new_ffn
            _mpnn.metrics = torch.nn.ModuleList([nn.metrics.BinaryAUROC()])
        else:
            _mp = nn.BondMessagePassing()
            _agg = nn.MeanAggregation()
            _ffn = nn.BinaryClassificationFFN(input_dim=_mp.output_dim)
            _mpnn = models.MPNN(_mp, _agg, _ffn, batch_norm=False)

        _trainer = lightning_pl.Trainer(
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            accelerator="cpu",
            devices=1,
            max_epochs=MAX_EPOCHS,
        )
        _trainer.fit(_mpnn, train_loader, val_loader)

        _preds = _trainer.predict(_mpnn, test_loader)
        _y_prob = torch.cat(_preds).cpu().numpy().flatten()
        _auc = roc_auc_score(y_test, _y_prob)
        logger.info(f"{name}: AUC={_auc:.3f}")
        return _mpnn, _y_prob

    model_scratch, y_prob_scratch = _build_and_train("Scratch")
    model_transfer, y_prob_transfer = _build_and_train(
        "Transfer", CHECKPOINTS_DIR / "rlm_pretrained.ckpt"
    )
    return (
        featurizer,
        model_scratch,
        model_transfer,
        test_mask,
        y_prob_scratch,
        y_prob_transfer,
        y_test,
    )


@app.cell
def _(
    chemprop_data,
    featurizer,
    logger,
    model_scratch,
    model_transfer,
    np,
    pampa_smiles,
    pl,
    test_mask,
    y_prob_scratch,
    y_prob_transfer,
    y_test,
):
    # Find failure molecules (same as notebook 08)
    _test_smi = [pampa_smiles[i] for i in range(len(pampa_smiles)) if test_mask[i]]
    _err_s = np.abs(y_prob_scratch - y_test)
    _err_t = np.abs(y_prob_transfer - y_test)
    _delta = _err_t - _err_s

    _failure_df = pl.DataFrame(
        {
            "smiles": _test_smi,
            "true_label": y_test.tolist(),
            "prob_scratch": y_prob_scratch.tolist(),
            "prob_transfer": y_prob_transfer.tolist(),
            "delta_err": _delta.tolist(),
        }
    ).sort("delta_err", descending=True)

    failure_smiles = _failure_df.head(5).get_column("smiles").to_list()
    failure_info = _failure_df.head(5)

    def compute_atom_saliency(mpnn, smiles):
        """Compute per-atom gradient saliency for a single molecule.

        Args:
            mpnn: Trained Chemprop MPNN model.
            smiles: SMILES string.

        Returns:
            Tuple of (atom_saliency_array, prediction_value).
        """
        mpnn.eval()
        _dp = chemprop_data.MoleculeDatapoint.from_smi(smiles, [0.0])
        _dset = chemprop_data.MoleculeDataset([_dp], featurizer)
        _loader = chemprop_data.build_dataloader(
            _dset, num_workers=0, batch_size=1, shuffle=False
        )
        _batch = next(iter(_loader))
        _bmg = _batch[0]

        _bmg.V = _bmg.V.clone().requires_grad_(True)
        _pred = mpnn(_bmg)
        _pred.sum().backward()

        _grad = _bmg.V.grad.abs().sum(dim=1).detach().numpy()
        _pred_val = _pred.item()
        return _grad, _pred_val

    # Compute saliency for each failure molecule with both models
    saliency_data = []
    for _smi in failure_smiles:
        _sal_s, _pred_s = compute_atom_saliency(model_scratch, _smi)
        _sal_t, _pred_t = compute_atom_saliency(model_transfer, _smi)
        saliency_data.append(
            {
                "smiles": _smi,
                "saliency_scratch": _sal_s,
                "saliency_transfer": _sal_t,
                "pred_scratch": _pred_s,
                "pred_transfer": _pred_t,
            }
        )
        logger.info(f"  {_smi[:40]}... scratch={_pred_s:.3f} transfer={_pred_t:.3f}")
    return failure_info, saliency_data


@app.cell
def _(Chem, Image, failure_info, io, mo, np, plt, rdMolDraw2D, saliency_data):
    def draw_saliency_diff_on_mol(mol, sal_scratch, sal_transfer, size=(450, 350)):
        """Draw molecule colored by saliency difference (transfer - scratch).

        Green = transfer pays more attention than scratch.
        Red = scratch pays more attention than transfer.
        Uses power-law mapping (gamma=2) on the absolute difference
        to emphasize the largest disagreements.
        """
        _sal_s = np.array(sal_scratch)
        _sal_t = np.array(sal_transfer)
        # Normalize each to [0,1] before taking difference
        _max_s = _sal_s.max() if _sal_s.max() > 0 else 1.0
        _max_t = _sal_t.max() if _sal_t.max() > 0 else 1.0
        _diff = (_sal_t / _max_t) - (_sal_s / _max_s)  # range [-1, 1]

        _max_abs = max(np.abs(_diff).max(), 1e-8)
        _norm_diff = _diff / _max_abs  # normalize to [-1, 1]

        _atoms = list(range(mol.GetNumAtoms()))
        _atom_colors = {}
        for _a in _atoms:
            _v = float(_norm_diff[_a])
            # Power-law on absolute value to emphasize extremes
            _intensity = float(abs(_v) ** 2 * 0.8 + 0.1)
            if _v > 0:
                # Green: transfer > scratch
                _atom_colors[_a] = (0.2, 0.8, 0.3, _intensity)
            else:
                # Red: scratch > transfer
                _atom_colors[_a] = (1.0, 0.3, 0.2, _intensity)

        _bonds = list(range(mol.GetNumBonds()))
        _bond_colors = {}
        for _b in _bonds:
            _bond = mol.GetBondWithIdx(_b)
            _a1 = _bond.GetBeginAtomIdx()
            _a2 = _bond.GetEndAtomIdx()
            _avg_v = float((_norm_diff[_a1] + _norm_diff[_a2]) / 2)
            _intensity = float(abs(_avg_v) ** 2 * 0.8 + 0.1)
            if _avg_v > 0:
                _bond_colors[_b] = (0.2, 0.8, 0.3, _intensity)
            else:
                _bond_colors[_b] = (1.0, 0.3, 0.2, _intensity)

        _d = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        _d.DrawMolecule(
            mol,
            highlightAtoms=_atoms,
            highlightAtomColors=_atom_colors,
            highlightBonds=_bonds,
            highlightBondColors=_bond_colors,
        )
        _d.FinishDrawing()
        return Image.open(io.BytesIO(_d.GetDrawingText()))

    _outputs = []
    for _rank, _data in enumerate(saliency_data):
        _smi = _data["smiles"]
        _mol = Chem.MolFromSmiles(_smi)
        if _mol is None:
            continue
        _row = failure_info.row(_rank, named=True)
        _label = "Permeable" if _row["true_label"] == 1 else "Impermeable"

        _img_diff = draw_saliency_diff_on_mol(
            _mol, _data["saliency_scratch"], _data["saliency_transfer"]
        )

        _fig, _ax = plt.subplots(1, 1, figsize=(8, 6))
        _ax.imshow(_img_diff)
        _ax.set_title(
            f"P(scratch)={_data['pred_scratch']:.3f}  |  P(transfer)={_data['pred_transfer']:.3f}",
            fontsize=11,
        )
        _ax.axis("off")

        _fig.suptitle(
            f"Molecule #{_rank + 1}: True={_label} | {_smi[:60]}{'...' if len(_smi) > 60 else ''}",
            fontsize=12,
        )
        plt.tight_layout()
        _outputs.append(mo.as_html(_fig))
        plt.close(_fig)

    mo.vstack(
        [
            mo.md("## Chemprop Saliency Difference: Transfer - Scratch"),
            mo.md("""
    Per-atom saliency difference (transfer minus scratch, after normalizing
    each model's saliency to [0,1]). **Green** = transfer model pays more
    attention to this atom than scratch. **Red** = scratch pays more attention.
    Intensity scales with the squared magnitude of the difference, so only
    the largest disagreements stand out.
        """),
            *_outputs,
        ]
    )
    return


@app.cell
def _(
    Chem,
    chemprop_data,
    featurizer,
    logger,
    mo,
    model_scratch,
    model_transfer,
    np,
    pampa_labels,
    pampa_smiles,
    plt,
    test_mask,
):
    # Aggregated atom-type saliency across all test molecules
    _test_smi = [pampa_smiles[i] for i in range(len(pampa_smiles)) if test_mask[i]]
    _test_y = pampa_labels[test_mask]

    # Collect per-atom-type saliency across dataset
    from collections import defaultdict

    _atom_type_saliency_scratch = defaultdict(list)
    _atom_type_saliency_transfer = defaultdict(list)

    def _get_saliency(mpnn, smiles):
        mpnn.eval()
        _dp = chemprop_data.MoleculeDatapoint.from_smi(smiles, [0.0])
        _dset = chemprop_data.MoleculeDataset([_dp], featurizer)
        _loader = chemprop_data.build_dataloader(
            _dset, num_workers=0, batch_size=1, shuffle=False
        )
        _batch = next(iter(_loader))
        _bmg = _batch[0]
        _bmg.V = _bmg.V.clone().requires_grad_(True)
        _pred = mpnn(_bmg)
        _pred.sum().backward()
        return _bmg.V.grad.abs().sum(dim=1).detach().numpy()

    # Sample 100 molecules for speed
    _sample_idx = np.random.default_rng(42).choice(
        len(_test_smi), size=min(100, len(_test_smi)), replace=False
    )

    for _i in _sample_idx:
        _smi = _test_smi[_i]
        _mol = Chem.MolFromSmiles(_smi)
        if _mol is None:
            continue
        try:
            _sal_s = _get_saliency(model_scratch, _smi)
            _sal_t = _get_saliency(model_transfer, _smi)
        except Exception:
            continue

        _n_atoms = _mol.GetNumAtoms()
        if len(_sal_s) != _n_atoms or len(_sal_t) != _n_atoms:
            continue

        # Normalize per-molecule so we compare relative importance
        _max_s = _sal_s.max() if _sal_s.max() > 0 else 1.0
        _max_t = _sal_t.max() if _sal_t.max() > 0 else 1.0

        for _a in range(_n_atoms):
            _atom = _mol.GetAtomWithIdx(_a)
            _symbol = _atom.GetSymbol()
            _is_aromatic = _atom.GetIsAromatic()
            _degree = _atom.GetDegree()
            _key = f"{_symbol}{'(arom)' if _is_aromatic else ''} deg{_degree}"
            _atom_type_saliency_scratch[_key].append(float(_sal_s[_a] / _max_s))
            _atom_type_saliency_transfer[_key].append(float(_sal_t[_a] / _max_t))

    logger.info(f"Collected saliency for {len(_atom_type_saliency_scratch)} atom types")

    # Build comparison: mean saliency per atom type, scratch vs transfer
    _atom_types = sorted(
        _atom_type_saliency_scratch.keys(),
        key=lambda k: np.mean(_atom_type_saliency_scratch[k]),
        reverse=True,
    )[:15]

    _fig_agg, _ax = plt.subplots(figsize=(12, 6))
    _x = np.arange(len(_atom_types))
    _width = 0.35

    _means_s = [np.mean(_atom_type_saliency_scratch[k]) for k in _atom_types]
    _means_t = [np.mean(_atom_type_saliency_transfer[k]) for k in _atom_types]
    _stds_s = [
        np.std(_atom_type_saliency_scratch[k])
        / np.sqrt(len(_atom_type_saliency_scratch[k]))
        for k in _atom_types
    ]
    _stds_t = [
        np.std(_atom_type_saliency_transfer[k])
        / np.sqrt(len(_atom_type_saliency_transfer[k]))
        for k in _atom_types
    ]

    _ax.barh(
        _x - _width / 2,
        _means_s,
        _width,
        xerr=_stds_s,
        label="Scratch",
        color="#2196F3",
        alpha=0.7,
        capsize=3,
    )
    _ax.barh(
        _x + _width / 2,
        _means_t,
        _width,
        xerr=_stds_t,
        label="RLM-Transfer",
        color="#FF9800",
        alpha=0.7,
        capsize=3,
    )
    _ax.set_yticks(_x)
    _ax.set_yticklabels(_atom_types, fontsize=9)
    _ax.set_xlabel("Mean Normalized Saliency")
    _ax.set_title(
        "Aggregated Atom-Type Saliency: Scratch vs Transfer (100 PAMPA molecules)"
    )
    _ax.legend()
    _ax.invert_yaxis()
    plt.tight_layout()

    # Categorize atom types: agree vs disagree in direction of importance
    # "agree" = both models assign similar relative importance
    # "disagree" = one model assigns much more importance than the other
    _all_atom_types = list(_atom_type_saliency_scratch.keys())
    agree_atom_types = []
    disagree_atom_types = []

    for _k in _all_atom_types:
        _ms = np.mean(_atom_type_saliency_scratch[_k])
        _mt = np.mean(_atom_type_saliency_transfer[_k])
        _diff = _mt - _ms
        if abs(_diff) < 0.02:
            agree_atom_types.append((_k, _ms, _mt, _diff))
        elif abs(_diff) >= 0.04:
            disagree_atom_types.append((_k, _ms, _mt, _diff))

    agree_atom_types.sort(key=lambda x: (x[1] + x[2]) / 2, reverse=True)
    disagree_atom_types.sort(key=lambda x: abs(x[3]), reverse=True)

    # Store the per-molecule saliency data for the substructure cell
    sampled_saliency = []
    for _i in _sample_idx:
        _smi = _test_smi[_i]
        _mol = Chem.MolFromSmiles(_smi)
        if _mol is None:
            continue
        try:
            _sal_s = _get_saliency(model_scratch, _smi)
            _sal_t = _get_saliency(model_transfer, _smi)
        except Exception:
            continue
        _n_atoms = _mol.GetNumAtoms()
        if len(_sal_s) != _n_atoms or len(_sal_t) != _n_atoms:
            continue
        sampled_saliency.append({"smiles": _smi, "sal_s": _sal_s, "sal_t": _sal_t})

    logger.info(f"Agree: {len(agree_atom_types)}, Disagree: {len(disagree_atom_types)}")

    # Top disagreements for display
    _delta = [(k, d) for k, _ms, _mt, d in disagree_atom_types[:5]]

    mo.vstack(
        [
            mo.md("## Aggregated Atom-Type Saliency"),
            mo.as_html(_fig_agg),
            mo.md(f"""
Mean normalized saliency per atom type across 100 sampled PAMPA test
molecules. Atom types are labeled by element, aromaticity, and degree.
Error bars show standard error of the mean.

**Largest disagreements** (transfer - scratch):
{"".join(f"- **{k}**: delta = {d:+.3f}" + chr(10) for k, d in _delta[:5])}

Atom types where the transfer model assigns substantially more or less
importance than the scratch model indicate where the RLM pre-training
shifted the model's attention.
        """),
        ]
    )
    return agree_atom_types, disagree_atom_types, sampled_saliency


@app.cell
def _(
    Chem,
    Image,
    agree_atom_types,
    disagree_atom_types,
    io,
    mo,
    np,
    plt,
    rdMolDraw2D,
    sampled_saliency,
):
    # Find example molecules for top agree and disagree atom types, and
    # highlight those atoms green (agree) or red (disagree)

    def _atom_type_key(atom):
        _sym = atom.GetSymbol()
        _arom = "(arom)" if atom.GetIsAromatic() else ""
        _deg = atom.GetDegree()
        return f"{_sym}{_arom} deg{_deg}"

    def _draw_highlighted_atoms(mol, atom_indices, color, size=(300, 250)):
        _acols = {a: color for a in atom_indices}
        _bonds = []
        _bcols = {}
        for _b in range(mol.GetNumBonds()):
            _bond = mol.GetBondWithIdx(_b)
            if (
                _bond.GetBeginAtomIdx() in atom_indices
                and _bond.GetEndAtomIdx() in atom_indices
            ):
                _bonds.append(_b)
                _bcols[_b] = color
        _d = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        _d.DrawMolecule(
            mol,
            highlightAtoms=atom_indices,
            highlightAtomColors=_acols,
            highlightBonds=_bonds,
            highlightBondColors=_bcols,
        )
        _d.FinishDrawing()
        return Image.open(io.BytesIO(_d.GetDrawingText()))

    def _find_example_mol(target_atom_type, saliency_list):
        """Find a molecule that contains the target atom type and has high saliency there."""
        _best = None
        _best_score = -1
        for _entry in saliency_list:
            _mol = Chem.MolFromSmiles(_entry["smiles"])
            if _mol is None:
                continue
            _sal_s = np.array(_entry["sal_s"])
            _sal_t = np.array(_entry["sal_t"])
            _max_s = _sal_s.max() if _sal_s.max() > 0 else 1.0
            _max_t = _sal_t.max() if _sal_t.max() > 0 else 1.0
            _matching_atoms = []
            for _a in range(_mol.GetNumAtoms()):
                if _atom_type_key(_mol.GetAtomWithIdx(_a)) == target_atom_type:
                    _matching_atoms.append(_a)
            if not _matching_atoms:
                continue
            # Score by how salient these atoms are (average across models)
            _score = sum(
                (_sal_s[a] / _max_s + _sal_t[a] / _max_t) / 2 for a in _matching_atoms
            ) / len(_matching_atoms)
            if _score > _best_score:
                _best_score = _score
                _best = (_mol, _matching_atoms, _entry["smiles"])
        return _best

    _n_agree = min(6, len(agree_atom_types))
    _n_disagree = min(6, len(disagree_atom_types))
    _n_cols = max(_n_agree, _n_disagree, 1)

    _fig, _axes = plt.subplots(2, _n_cols, figsize=(_n_cols * 4, 9))
    if _n_cols == 1:
        _axes = _axes.reshape(2, 1)

    # Row 0: agree (green)
    for _j in range(_n_agree):
        _atype, _ms, _mt, _diff = agree_atom_types[_j]
        _result = _find_example_mol(_atype, sampled_saliency)
        if _result:
            _mol, _atoms, _smi = _result
            _img = _draw_highlighted_atoms(_mol, _atoms, (0.2, 0.8, 0.3, 0.5))
            _axes[0][_j].imshow(_img)
        _axes[0][_j].set_title(
            f"AGREE: {_atype}\nS={_ms:.3f} T={_mt:.3f}", fontsize=9, color="#4CAF50"
        )
        _axes[0][_j].axis("off")

    # Row 1: disagree (red)
    for _j in range(_n_disagree):
        _atype, _ms, _mt, _diff = disagree_atom_types[_j]
        _result = _find_example_mol(_atype, sampled_saliency)
        if _result:
            _mol, _atoms, _smi = _result
            _img = _draw_highlighted_atoms(_mol, _atoms, (1.0, 0.3, 0.2, 0.5))
            _axes[1][_j].imshow(_img)
        _dir_s = "more" if _diff < 0 else "less"
        _axes[1][_j].set_title(
            f"DISAGREE: {_atype}\nS={_ms:.3f} T={_mt:.3f} (d={_diff:+.3f})",
            fontsize=9,
            color="#FF5722",
        )
        _axes[1][_j].axis("off")

    for _row in range(2):
        _n = _n_agree if _row == 0 else _n_disagree
        for _j in range(_n, _n_cols):
            _axes[_row][_j].axis("off")

    _fig.suptitle(
        "Chemprop: Atom Types Models Agree (top, green) vs Disagree (bottom, red)",
        fontsize=13,
    )
    plt.tight_layout()

    mo.vstack(
        [
            mo.md("## Substructure Agreement and Disagreement"),
            mo.as_html(_fig),
            mo.md(f"""
**Agree** ({len(agree_atom_types)} atom types, green): Both models
assign similar relative importance to these atom types (difference < 0.02).
Highlighted atoms show the atom type in context on an example molecule.

**Disagree** ({len(disagree_atom_types)} atom types, red): The transfer
model assigns substantially different importance than scratch
(difference >= 0.04). These are atom environments where RLM pre-training
shifted the D-MPNN's learned representation.

S = scratch mean normalized saliency, T = transfer mean normalized saliency.
        """),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
