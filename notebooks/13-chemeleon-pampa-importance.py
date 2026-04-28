import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md("""
    # 13 — CheMeleon Feature Importance on PAMPA

    Compares gradient saliency between CheMeleon frozen single-finetune
    (Foundation → PAMPA) and CheMeleon frozen double-finetune
    (Foundation → RLM → PAMPA). Since the encoder is frozen in both, any
    difference in attention patterns must come from the FFN head learning
    different mappings from the same features.
    """)
    return (mo,)


@app.cell
def _():
    import io
    from collections import defaultdict
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from lightning import pytorch as lightning_pl
    from loguru import logger
    from PIL import Image
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
    from sklearn.metrics import roc_auc_score

    from chemprop import data as chemprop_data
    from chemprop import featurizers, models, nn

    DATA_DIR = Path("data")
    CHECKPOINTS_DIR = Path("checkpoints")
    FIGURES_DIR = Path("docs/figures")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return (
        CHECKPOINTS_DIR,
        Chem,
        DATA_DIR,
        FIGURES_DIR,
        Image,
        chemprop_data,
        defaultdict,
        featurizers,
        io,
        lightning_pl,
        logger,
        models,
        nn,
        np,
        plt,
        rdMolDraw2D,
        roc_auc_score,
        torch,
    )


@app.cell
def _(CHECKPOINTS_DIR, DATA_DIR, logger, nn, np, torch):
    """Load PAMPA and RLM data, plus CheMeleon encoder."""

    _pampa_split = np.load(DATA_DIR / "pampa_splits.npz", allow_pickle=True)
    pampa_smiles = list(_pampa_split["smiles"])
    pampa_labels = _pampa_split["labels"]
    pampa_folds = _pampa_split["folds"]

    _rlm_split = np.load(DATA_DIR / "rlm_splits.npz", allow_pickle=True)
    rlm_smiles = list(_rlm_split["smiles"])
    rlm_labels = _rlm_split["labels"]

    # Load CheMeleon encoder
    _cm_data = torch.load(CHECKPOINTS_DIR / "chemeleon_mp.pt", weights_only=True)
    chemeleon_mp = nn.BondMessagePassing(**_cm_data["hyper_parameters"])
    chemeleon_mp.load_state_dict(_cm_data["state_dict"])

    logger.info(
        f"PAMPA: {len(pampa_smiles)} molecules, RLM: {len(rlm_smiles)} molecules, "
        f"CheMeleon encoder: {sum(p.numel() for p in chemeleon_mp.parameters())} params"
    )
    return chemeleon_mp, pampa_folds, pampa_labels, pampa_smiles, rlm_labels, rlm_smiles


@app.cell
def _(
    Chem,
    chemeleon_mp,
    chemprop_data,
    defaultdict,
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
    """Train CheMeleon frozen single-finetune on PAMPA and compute saliency."""
    _featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    lightning_pl.seed_everything(42, workers=True)

    _fold_assign = pampa_folds[0]
    cm_single_test_mask = _fold_assign == 0
    _train_mask = ~cm_single_test_mask

    _train_smi = [pampa_smiles[i] for i in range(len(pampa_smiles)) if _train_mask[i]]
    _train_y = pampa_labels[_train_mask].reshape(-1, 1).astype(float)
    _test_smi = [
        pampa_smiles[i] for i in range(len(pampa_smiles)) if cm_single_test_mask[i]
    ]
    _test_y = pampa_labels[cm_single_test_mask].reshape(-1, 1).astype(float)

    _n = len(_train_smi)
    _n_val = max(1, int(_n * 0.1))
    _perm = np.random.default_rng(42).permutation(_n)

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

    _train_ds = chemprop_data.MoleculeDataset(_train_data, _featurizer)
    _val_ds = chemprop_data.MoleculeDataset(_val_data, _featurizer)
    _test_ds = chemprop_data.MoleculeDataset(_test_data, _featurizer)
    _train_loader = chemprop_data.build_dataloader(
        _train_ds, num_workers=0, batch_size=64
    )
    _val_loader = chemprop_data.build_dataloader(
        _val_ds, num_workers=0, shuffle=False, batch_size=64
    )
    _test_loader = chemprop_data.build_dataloader(
        _test_ds, num_workers=0, shuffle=False, batch_size=64
    )

    # Frozen encoder: copy weights, freeze
    import copy as _copy

    _mp_frozen = _copy.deepcopy(chemeleon_mp)
    for _p in _mp_frozen.parameters():
        _p.requires_grad = False

    _agg = nn.MeanAggregation()
    _ffn = nn.BinaryClassificationFFN(input_dim=_mp_frozen.output_dim)
    _model = models.MPNN(_mp_frozen, _agg, _ffn, batch_norm=False)

    _trainer = lightning_pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        deterministic=True,
        accelerator="gpu",
        devices=1,
        max_epochs=30,
    )
    _trainer.fit(_model, _train_loader, _val_loader)
    _preds = _trainer.predict(_model, _test_loader)
    _y_prob = torch.cat(_preds).cpu().numpy().flatten()
    logger.info(
        f"CheMeleon frozen single PAMPA AUC: {roc_auc_score(pampa_labels[cm_single_test_mask], _y_prob):.3f}"
    )

    # Saliency
    cm_single_saliency = defaultdict(list)
    cm_single_best = {}
    _n_success = 0
    for _i in range(len(_test_smi)):
        _smi = _test_smi[_i]
        _mol = Chem.MolFromSmiles(_smi)
        if _mol is None:
            continue
        _model.eval()
        _dp = chemprop_data.MoleculeDatapoint.from_smi(_smi, [0.0])
        _dset = chemprop_data.MoleculeDataset([_dp], _featurizer)
        _loader = chemprop_data.build_dataloader(
            _dset, num_workers=0, batch_size=1, shuffle=False
        )
        _batch = next(iter(_loader))
        _bmg = _batch[0]
        _bmg.V = _bmg.V.clone().requires_grad_(True)
        try:
            _pred = _model(_bmg)
            _pred.sum().backward()
            _sal = _bmg.V.grad.abs().sum(dim=1).detach().numpy()
        except Exception:
            continue
        _n_atoms = _mol.GetNumAtoms()
        if len(_sal) != _n_atoms:
            continue
        _max_sal = _sal.max() if _sal.max() > 0 else 1.0
        for _a in range(_n_atoms):
            _atom = _mol.GetAtomWithIdx(_a)
            _key = f"{_atom.GetSymbol()}{'(arom)' if _atom.GetIsAromatic() else ''} deg{_atom.GetDegree()}"
            _norm_sal = float(_sal[_a] / _max_sal)
            cm_single_saliency[_key].append(_norm_sal)
            if _key not in cm_single_best or _norm_sal > cm_single_best[_key][2]:
                cm_single_best[_key] = (_mol, _a, _norm_sal)
        _n_success += 1

    logger.info(f"CheMeleon frozen single saliency: {_n_success} molecules")
    cm_single_means = {k: float(np.mean(v)) for k, v in cm_single_saliency.items()}
    cm_single_stderrs = {
        k: float(np.std(v) / np.sqrt(len(v))) for k, v in cm_single_saliency.items()
    }
    cm_single_top = sorted(
        cm_single_means.keys(), key=lambda k: cm_single_means[k], reverse=True
    )[:6]
    return (
        cm_single_best,
        cm_single_means,
        cm_single_stderrs,
        cm_single_test_mask,
        cm_single_top,
    )


@app.cell
def _(
    Chem,
    chemeleon_mp,
    chemprop_data,
    defaultdict,
    featurizers,
    lightning_pl,
    logger,
    models,
    nn,
    np,
    pampa_folds,
    pampa_labels,
    pampa_smiles,
    rlm_labels,
    rlm_smiles,
    roc_auc_score,
    torch,
):
    """Train CheMeleon frozen double-finetune (Foundation→RLM→PAMPA) and compute saliency."""
    _featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    lightning_pl.seed_everything(42, workers=True)

    import copy as _copy

    # Step 1: Finetune FFN on RLM (encoder frozen)
    _rlm_y = rlm_labels.reshape(-1, 1).astype(float)
    _n_rlm = len(rlm_smiles)
    _n_rlm_val = max(1, int(_n_rlm * 0.1))
    _perm_rlm = np.random.default_rng(42).permutation(_n_rlm)

    _rlm_train = [
        chemprop_data.MoleculeDatapoint.from_smi(rlm_smiles[i], _rlm_y[i])
        for i in _perm_rlm[_n_rlm_val:]
    ]
    _rlm_val = [
        chemprop_data.MoleculeDatapoint.from_smi(rlm_smiles[i], _rlm_y[i])
        for i in _perm_rlm[:_n_rlm_val]
    ]
    _rlm_train_ds = chemprop_data.MoleculeDataset(_rlm_train, _featurizer)
    _rlm_val_ds = chemprop_data.MoleculeDataset(_rlm_val, _featurizer)
    _rlm_train_loader = chemprop_data.build_dataloader(
        _rlm_train_ds, num_workers=0, batch_size=64
    )
    _rlm_val_loader = chemprop_data.build_dataloader(
        _rlm_val_ds, num_workers=0, shuffle=False, batch_size=64
    )

    _mp_frozen = _copy.deepcopy(chemeleon_mp)
    for _p in _mp_frozen.parameters():
        _p.requires_grad = False
    _agg = nn.MeanAggregation()
    _ffn_rlm = nn.BinaryClassificationFFN(input_dim=_mp_frozen.output_dim)
    _rlm_model = models.MPNN(_mp_frozen, _agg, _ffn_rlm, batch_norm=False)

    _trainer = lightning_pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        deterministic=True,
        accelerator="gpu",
        devices=1,
        max_epochs=30,
    )
    _trainer.fit(_rlm_model, _rlm_train_loader, _rlm_val_loader)
    logger.info("CheMeleon frozen RLM intermediate done")

    # Step 2: New FFN head, finetune on PAMPA (encoder still frozen)
    _fold_assign = pampa_folds[0]
    cm_double_test_mask = _fold_assign == 0
    _train_mask = ~cm_double_test_mask

    _train_smi = [pampa_smiles[i] for i in range(len(pampa_smiles)) if _train_mask[i]]
    _train_y = pampa_labels[_train_mask].reshape(-1, 1).astype(float)
    _test_smi = [
        pampa_smiles[i] for i in range(len(pampa_smiles)) if cm_double_test_mask[i]
    ]
    _test_y = pampa_labels[cm_double_test_mask].reshape(-1, 1).astype(float)

    _n = len(_train_smi)
    _n_val = max(1, int(_n * 0.1))
    _perm = np.random.default_rng(42).permutation(_n)

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
    _train_ds = chemprop_data.MoleculeDataset(_train_data, _featurizer)
    _val_ds = chemprop_data.MoleculeDataset(_val_data, _featurizer)
    _test_ds = chemprop_data.MoleculeDataset(_test_data, _featurizer)
    _train_loader = chemprop_data.build_dataloader(
        _train_ds, num_workers=0, batch_size=64
    )
    _val_loader = chemprop_data.build_dataloader(
        _val_ds, num_workers=0, shuffle=False, batch_size=64
    )
    _test_loader = chemprop_data.build_dataloader(
        _test_ds, num_workers=0, shuffle=False, batch_size=64
    )

    # New FFN head for PAMPA (encoder is the same frozen one from RLM step)
    _ffn_pampa = nn.BinaryClassificationFFN(input_dim=_mp_frozen.output_dim)
    _pampa_model = models.MPNN(_mp_frozen, _agg, _ffn_pampa, batch_norm=False)

    lightning_pl.seed_everything(42, workers=True)
    _trainer2 = lightning_pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        deterministic=True,
        accelerator="gpu",
        devices=1,
        max_epochs=30,
    )
    _trainer2.fit(_pampa_model, _train_loader, _val_loader)
    _preds = _trainer2.predict(_pampa_model, _test_loader)
    _y_prob = torch.cat(_preds).cpu().numpy().flatten()
    logger.info(
        f"CheMeleon frozen double PAMPA AUC: {roc_auc_score(pampa_labels[cm_double_test_mask], _y_prob):.3f}"
    )

    # Saliency
    cm_double_saliency = defaultdict(list)
    cm_double_best = {}
    _n_success = 0
    for _i in range(len(_test_smi)):
        _smi = _test_smi[_i]
        _mol = Chem.MolFromSmiles(_smi)
        if _mol is None:
            continue
        _pampa_model.eval()
        _dp = chemprop_data.MoleculeDatapoint.from_smi(_smi, [0.0])
        _dset = chemprop_data.MoleculeDataset([_dp], _featurizer)
        _loader = chemprop_data.build_dataloader(
            _dset, num_workers=0, batch_size=1, shuffle=False
        )
        _batch = next(iter(_loader))
        _bmg = _batch[0]
        _bmg.V = _bmg.V.clone().requires_grad_(True)
        try:
            _pred = _pampa_model(_bmg)
            _pred.sum().backward()
            _sal = _bmg.V.grad.abs().sum(dim=1).detach().numpy()
        except Exception:
            continue
        _n_atoms = _mol.GetNumAtoms()
        if len(_sal) != _n_atoms:
            continue
        _max_sal = _sal.max() if _sal.max() > 0 else 1.0
        for _a in range(_n_atoms):
            _atom = _mol.GetAtomWithIdx(_a)
            _key = f"{_atom.GetSymbol()}{'(arom)' if _atom.GetIsAromatic() else ''} deg{_atom.GetDegree()}"
            _norm_sal = float(_sal[_a] / _max_sal)
            cm_double_saliency[_key].append(_norm_sal)
            if _key not in cm_double_best or _norm_sal > cm_double_best[_key][2]:
                cm_double_best[_key] = (_mol, _a, _norm_sal)
        _n_success += 1

    logger.info(f"CheMeleon frozen double saliency: {_n_success} molecules")
    cm_double_means = {k: float(np.mean(v)) for k, v in cm_double_saliency.items()}
    cm_double_stderrs = {
        k: float(np.std(v) / np.sqrt(len(v))) for k, v in cm_double_saliency.items()
    }
    cm_double_top = sorted(
        cm_double_means.keys(), key=lambda k: cm_double_means[k], reverse=True
    )[:6]
    return cm_double_best, cm_double_means, cm_double_stderrs, cm_double_top


@app.cell
def _(
    Chem,
    FIGURES_DIR,
    Image,
    cm_single_best,
    cm_single_means,
    cm_single_stderrs,
    cm_single_top,
    cm_double_best,
    cm_double_means,
    cm_double_stderrs,
    cm_double_top,
    io,
    logger,
    mo,
    np,
    plt,
    rdMolDraw2D,
):
    """Plot CheMeleon frozen single vs double saliency comparison."""

    def _draw_neighborhood(mol, center_atom, radius, size=(220, 160)):
        _effective_radius = max(radius, 1)
        _env = Chem.FindAtomEnvironmentOfRadiusN(mol, _effective_radius, center_atom)
        if not _env:
            _atom = mol.GetAtomWithIdx(center_atom)
            _frag = Chem.MolFromSmiles(f"[{_atom.GetSymbol()}]")
            if _frag is None:
                return None
            _d = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
            _d.drawOptions().bondLineWidth = 2.0
            _d.drawOptions().minFontSize = 16
            _d.DrawMolecule(_frag)
            _d.FinishDrawing()
            return Image.open(io.BytesIO(_d.GetDrawingText()))
        _amap = {}
        _submol = Chem.PathToSubmol(mol, _env, atomMap=_amap)
        if _submol is None or _submol.GetNumAtoms() == 0:
            return None
        _center_in_sub = _amap.get(center_atom, -1)
        _highlight_atoms = list(range(_submol.GetNumAtoms()))
        _atom_colors = {a: (0.92, 0.92, 0.92, 1.0) for a in _highlight_atoms}
        if _center_in_sub >= 0:
            _atom_colors[_center_in_sub] = (0.4, 0.65, 0.9, 0.5)
        _atom_radii = {a: 0.3 for a in _highlight_atoms}
        if _center_in_sub >= 0:
            _atom_radii[_center_in_sub] = 0.4
        _d = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        _d.drawOptions().bondLineWidth = 2.0
        _d.drawOptions().minFontSize = 14
        _d.drawOptions().padding = 0.15
        _d.DrawMolecule(
            _submol,
            highlightAtoms=_highlight_atoms,
            highlightAtomColors=_atom_colors,
            highlightAtomRadii=_atom_radii,
        )
        _d.FinishDrawing()
        return Image.open(io.BytesIO(_d.GetDrawingText()))

    _n_left = min(6, len(cm_single_top))
    _n_right = min(6, len(cm_double_top))
    _types_left = cm_single_top[:_n_left]
    _types_right = cm_double_top[:_n_right]

    _fig = plt.figure(figsize=(22, 8))
    _gs = _fig.add_gridspec(1, 4, width_ratios=[0.3, 0.7, 0.7, 0.3], wspace=0.08)

    _ax_img_l = _fig.add_subplot(_gs[0])
    _ax_img_l.set_xlim(0, 1)
    _ax_img_l.set_ylim(-0.5, _n_left - 0.5)
    _ax_img_l.invert_yaxis()
    _ax_img_l.axis("off")
    for _j, _atype in enumerate(_types_left):
        if _atype in cm_single_best:
            _mol, _idx, _ = cm_single_best[_atype]
            try:
                _img = _draw_neighborhood(_mol, _idx, 1)
                if _img:
                    _inset = _ax_img_l.inset_axes(
                        [0.0, (_n_left - 1 - _j) / _n_left, 1.0, 0.9 / _n_left],
                        transform=_ax_img_l.transAxes,
                    )
                    _inset.imshow(_img)
                    _inset.axis("off")
            except Exception:
                pass

    _ax_l = _fig.add_subplot(_gs[1])
    _means_l = [cm_single_means[k] for k in _types_left]
    _errs_l = [cm_single_stderrs[k] for k in _types_left]
    _ax_l.barh(
        np.arange(_n_left),
        _means_l,
        xerr=_errs_l,
        color="#7E57C2",
        alpha=0.8,
        height=0.7,
        capsize=3,
    )
    _ax_l.set_yticks(np.arange(_n_left))
    _ax_l.set_yticklabels(_types_left, fontsize=9)
    _ax_l.set_xlabel("Mean Normalized Saliency", fontsize=11)
    _ax_l.set_title(
        "CheMeleon Frozen Single\n(Foundation → PAMPA)", fontsize=12, fontweight="bold"
    )
    _ax_l.invert_yaxis()

    _ax_r = _fig.add_subplot(_gs[2])
    _means_r = [cm_double_means[k] for k in _types_right]
    _errs_r = [cm_double_stderrs[k] for k in _types_right]
    _ax_r.barh(
        np.arange(_n_right),
        _means_r,
        xerr=_errs_r,
        color="#00897B",
        alpha=0.8,
        height=0.7,
        capsize=3,
    )
    _ax_r.set_yticks(np.arange(_n_right))
    _ax_r.set_yticklabels(_types_right, fontsize=9)
    _ax_r.yaxis.tick_right()
    _ax_r.yaxis.set_label_position("right")
    _ax_r.set_xlabel("Mean Normalized Saliency", fontsize=11)
    _ax_r.set_title(
        "CheMeleon Frozen Double\n(Foundation → RLM → PAMPA)",
        fontsize=12,
        fontweight="bold",
    )
    _ax_r.invert_yaxis()

    _ax_img_r = _fig.add_subplot(_gs[3])
    _ax_img_r.set_xlim(0, 1)
    _ax_img_r.set_ylim(-0.5, _n_right - 0.5)
    _ax_img_r.invert_yaxis()
    _ax_img_r.axis("off")
    for _j, _atype in enumerate(_types_right):
        if _atype in cm_double_best:
            _mol, _idx, _ = cm_double_best[_atype]
            try:
                _img = _draw_neighborhood(_mol, _idx, 1)
                if _img:
                    _inset = _ax_img_r.inset_axes(
                        [0.15, (_n_right - 1 - _j) / _n_right, 0.85, 0.9 / _n_right],
                        transform=_ax_img_r.transAxes,
                    )
                    _inset.imshow(_img)
                    _inset.axis("off")
            except Exception:
                pass

    _fig.suptitle(
        "PAMPA pH 7.4: CheMeleon Frozen Single vs Double-Finetune (Saliency)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    _save_path = FIGURES_DIR / "pampa-chemeleon-single-vs-double.png"
    _fig.savefig(_save_path, dpi=150, bbox_inches="tight", facecolor="white")
    logger.info(f"Saved {_save_path}")

    mo.vstack(
        [
            mo.md("## PAMPA: CheMeleon Frozen Single vs Double-Finetune"),
            mo.as_html(_fig),
            mo.md("""
*Top 6 atom types by gradient saliency. Since the encoder is frozen in
both variants, any difference reflects the FFN head learning different
feature-to-target mappings. Near-identical attention patterns would
confirm the frozen encoder provides stable, general-purpose features
regardless of the intermediate RLM training step. Single fold.*
        """),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
