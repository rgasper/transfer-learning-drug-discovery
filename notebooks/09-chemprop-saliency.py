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
    def draw_saliency_on_mol(mol, atom_saliency, size=(450, 350)):
        """Draw molecule with atoms colored by saliency (blue = high importance)."""
        _sal = np.array(atom_saliency)
        _max_sal = _sal.max() if _sal.max() > 0 else 1.0
        _norm = _sal / _max_sal

        _atoms = list(range(mol.GetNumAtoms()))
        _atom_colors = {}
        for _a in _atoms:
            _intensity = float(_norm[_a])
            _atom_colors[_a] = (0.13, 0.59, 0.95, _intensity * 0.7 + 0.1)

        _bonds = list(range(mol.GetNumBonds()))
        _bond_colors = {}
        for _b in _bonds:
            _bond = mol.GetBondWithIdx(_b)
            _a1 = _bond.GetBeginAtomIdx()
            _a2 = _bond.GetEndAtomIdx()
            _avg = float((_norm[_a1] + _norm[_a2]) / 2)
            _bond_colors[_b] = (0.13, 0.59, 0.95, _avg * 0.7 + 0.1)

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

        _img_s = draw_saliency_on_mol(_mol, _data["saliency_scratch"])
        _img_t = draw_saliency_on_mol(_mol, _data["saliency_transfer"])

        _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(14, 5))
        _ax1.imshow(_img_s)
        _ax1.set_title(
            f"Chemprop Scratch\nP(active)={_data['pred_scratch']:.3f}", fontsize=11
        )
        _ax1.axis("off")
        _ax2.imshow(_img_t)
        _ax2.set_title(
            f"Chemprop RLM-Transfer\nP(active)={_data['pred_transfer']:.3f}",
            fontsize=11,
        )
        _ax2.axis("off")

        _fig.suptitle(
            f"Molecule #{_rank + 1}: True={_label} | {_smi[:60]}{'...' if len(_smi) > 60 else ''}",
            fontsize=12,
        )
        plt.tight_layout()
        _outputs.append(mo.as_html(_fig))
        plt.close(_fig)

    mo.vstack(
        [
            mo.md("## Chemprop Atom Saliency: Failure Molecules"),
            mo.md("""
    Per-atom gradient saliency for the same 5 PAMPA failure molecules.
    Darker blue = higher gradient magnitude = more important for the
    model's prediction. The scratch model (left) and transfer model (right)
    may focus on different parts of the molecule.
        """),
            *_outputs,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
