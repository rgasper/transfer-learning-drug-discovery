import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md("""
    # 12 — PAMPA Feature Importance: XGBoost SHAP + Chemprop Saliency

    Computes feature importance for PAMPA pH 7.4 predictions using both
    XGBoost (SHAP on Morgan fingerprint bits) and Chemprop (per-atom
    gradient saliency). Produces a combined figure showing what structural
    features each architecture attends to for membrane permeability.
    """)
    return (mo,)


@app.cell
def _():
    import io
    from collections import defaultdict
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import shap
    import torch
    import xgboost as xgb
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
    FIGURES_DIR = Path("docs/figures")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return (
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
        rdFingerprintGenerator,
        rdMolDraw2D,
        roc_auc_score,
        shap,
        torch,
        xgb,
    )


@app.cell
def _(DATA_DIR, logger, np):
    """Load PAMPA split data."""
    import json

    with open(DATA_DIR / "split_config.json") as _f:
        _split_config = json.load(_f)

    _fp_data = np.load(DATA_DIR / "morgan_fps_2048_r3.npz", allow_pickle=True)
    global_fps = _fp_data["fp_matrix"]

    _pampa_split = np.load(DATA_DIR / "pampa_splits.npz", allow_pickle=True)
    pampa_smiles = list(_pampa_split["smiles"])
    pampa_labels = _pampa_split["labels"]
    pampa_folds = _pampa_split["folds"]
    pampa_fp_indices = _pampa_split["fp_indices"]
    pampa_X = global_fps[pampa_fp_indices]

    logger.info(f"PAMPA: {pampa_X.shape[0]} molecules")
    return pampa_X, pampa_folds, pampa_labels, pampa_smiles


@app.cell
def _(
    logger,
    np,
    pampa_X,
    pampa_folds,
    pampa_labels,
    roc_auc_score,
    shap,
    xgb,
):
    """Train XGBoost on PAMPA (rep 0, fold 0) and compute SHAP values."""
    np.random.seed(42)
    _fold_assign = pampa_folds[0]
    xgb_test_mask = _fold_assign == 0
    _train_mask = ~xgb_test_mask

    _dtrain = xgb.DMatrix(pampa_X[_train_mask], label=pampa_labels[_train_mask])
    _dval = xgb.DMatrix(pampa_X[xgb_test_mask], label=pampa_labels[xgb_test_mask])

    xgb_model = xgb.train(
        {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "nthread": 1,
            "verbosity": 0,
        },
        _dtrain,
        200,
        evals=[(_dval, "val")],
        early_stopping_rounds=20,
        verbose_eval=False,
    )

    _y_prob = xgb_model.predict(_dval)
    logger.info(
        f"XGBoost PAMPA AUC: {roc_auc_score(pampa_labels[xgb_test_mask], _y_prob):.3f}"
    )

    xgb_explainer = shap.TreeExplainer(xgb_model)
    xgb_shap_values = xgb_explainer.shap_values(pampa_X[xgb_test_mask])

    xgb_mean_abs_shap = np.abs(xgb_shap_values).mean(axis=0)
    xgb_mean_signed_shap = xgb_shap_values.mean(axis=0)
    xgb_top_bits = np.argsort(xgb_mean_abs_shap)[-6:][::-1]

    logger.info(f"Top 6 bits: {xgb_top_bits.tolist()}")
    return (
        xgb_mean_abs_shap,
        xgb_mean_signed_shap,
        xgb_shap_values,
        xgb_test_mask,
        xgb_top_bits,
    )


@app.cell
def _(
    Chem,
    logger,
    pampa_smiles,
    rdFingerprintGenerator,
    xgb_shap_values,
    xgb_test_mask,
    xgb_top_bits,
):
    """For each top SHAP bit, find the molecule where it contributes most."""
    _gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
    _test_smiles = [
        pampa_smiles[i] for i in range(len(pampa_smiles)) if xgb_test_mask[i]
    ]

    xgb_bit_to_mol_info = {}
    _bit_best_shap = {}

    for _mol_idx, _smi in enumerate(_test_smiles):
        _mol = Chem.MolFromSmiles(_smi)
        if _mol is None:
            continue
        _ao = rdFingerprintGenerator.AdditionalOutput()
        _ao.AllocateBitInfoMap()
        _gen.GetFingerprint(_mol, additionalOutput=_ao)
        _bit_info = _ao.GetBitInfoMap()

        for _bit, _envs in _bit_info.items():
            _shap_val = abs(float(xgb_shap_values[_mol_idx, _bit]))
            if _bit not in _bit_best_shap or _shap_val > _bit_best_shap[_bit]:
                _center, _rad = _envs[0]
                _bit_best_shap[_bit] = _shap_val
                xgb_bit_to_mol_info[_bit] = (_mol, _center, _rad)

    _coverage = sum(1 for b in xgb_top_bits if int(b) in xgb_bit_to_mol_info)
    logger.info(
        f"XGB bit coverage: {_coverage}/{len(xgb_top_bits)} top bits have mol info"
    )
    return (xgb_bit_to_mol_info,)


@app.cell
def _(
    Chem,
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
    """Train Chemprop on PAMPA and compute per-atom saliency."""
    _featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    _fold_assign = pampa_folds[0]
    chemprop_test_mask = _fold_assign == 0
    _train_mask = ~chemprop_test_mask

    _train_smi = [pampa_smiles[i] for i in range(len(pampa_smiles)) if _train_mask[i]]
    _train_y = pampa_labels[_train_mask].reshape(-1, 1).astype(float)
    _test_smi = [
        pampa_smiles[i] for i in range(len(pampa_smiles)) if chemprop_test_mask[i]
    ]
    _test_y = pampa_labels[chemprop_test_mask].reshape(-1, 1).astype(float)

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

    _train_dset = chemprop_data.MoleculeDataset(_train_data, _featurizer)
    _val_dset = chemprop_data.MoleculeDataset(_val_data, _featurizer)
    _test_dset = chemprop_data.MoleculeDataset(_test_data, _featurizer)

    _train_loader = chemprop_data.build_dataloader(
        _train_dset, num_workers=0, batch_size=64
    )
    _val_loader = chemprop_data.build_dataloader(
        _val_dset, num_workers=0, shuffle=False, batch_size=64
    )
    _test_loader = chemprop_data.build_dataloader(
        _test_dset, num_workers=0, shuffle=False, batch_size=64
    )

    _mp = nn.BondMessagePassing()
    _agg = nn.MeanAggregation()
    _ffn = nn.BinaryClassificationFFN(input_dim=_mp.output_dim)
    _model = models.MPNN(_mp, _agg, _ffn, batch_norm=False)

    lightning_pl.seed_everything(42, workers=True)
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
        f"Chemprop PAMPA AUC: {roc_auc_score(pampa_labels[chemprop_test_mask], _y_prob):.3f}"
    )

    # Compute atom saliency with best-instance tracking
    chemprop_atom_type_saliency = defaultdict(list)
    chemprop_atom_type_best = {}

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
            chemprop_atom_type_saliency[_key].append(_norm_sal)
            if (
                _key not in chemprop_atom_type_best
                or _norm_sal > chemprop_atom_type_best[_key][2]
            ):
                chemprop_atom_type_best[_key] = (_mol, _a, _norm_sal)
        _n_success += 1

    logger.info(
        f"Chemprop saliency: {_n_success} molecules, {len(chemprop_atom_type_saliency)} atom types"
    )

    chemprop_atom_means = {
        k: float(np.mean(v)) for k, v in chemprop_atom_type_saliency.items()
    }
    chemprop_atom_stderrs = {
        k: float(np.std(v) / np.sqrt(len(v)))
        for k, v in chemprop_atom_type_saliency.items()
    }
    chemprop_top_atom_types = sorted(
        chemprop_atom_means.keys(), key=lambda k: chemprop_atom_means[k], reverse=True
    )[:6]
    return (
        chemprop_atom_means,
        chemprop_atom_stderrs,
        chemprop_atom_type_best,
        chemprop_top_atom_types,
    )


@app.cell
def _(
    Chem,
    FIGURES_DIR,
    Image,
    chemprop_atom_means,
    chemprop_atom_stderrs,
    chemprop_atom_type_best,
    chemprop_top_atom_types,
    io,
    logger,
    mo,
    np,
    plt,
    rdMolDraw2D,
    xgb_bit_to_mol_info,
    xgb_mean_abs_shap,
    xgb_mean_signed_shap,
    xgb_top_bits,
):
    """Generate the combined PAMPA feature importance figure."""

    def _draw_neighborhood(mol, center_atom, radius, size=(220, 160)):
        """Extract and draw only the atom neighborhood as a submolecule.

        Args:
            mol: RDKit Mol object (the full parent molecule).
            center_atom: Index of the central atom.
            radius: Bond radius of the environment to extract.
            size: Tuple of (width, height) in pixels.

        Returns:
            PIL Image of the extracted fragment, or None on failure.
        """
        if radius == 0:
            _atom = mol.GetAtomWithIdx(center_atom)
            _smi = f"[{_atom.GetSymbol()}]"
            _frag = Chem.MolFromSmiles(_smi)
            if _frag is None:
                return None
            _d = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
            _d.drawOptions().bondLineWidth = 2.0
            _d.drawOptions().minFontSize = 16
            _d.DrawMolecule(_frag)
            _d.FinishDrawing()
            return Image.open(io.BytesIO(_d.GetDrawingText()))

        _env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center_atom)
        if not _env:
            return None

        _amap = {}
        _submol = Chem.PathToSubmol(mol, _env, atomMap=_amap)
        if _submol is None or _submol.GetNumAtoms() == 0:
            return None

        _center_in_sub = _amap.get(center_atom, -1)
        _highlight_atoms = [_center_in_sub] if _center_in_sub >= 0 else []
        _atom_colors = (
            {_center_in_sub: (0.2, 0.6, 1.0, 1.0)} if _center_in_sub >= 0 else {}
        )

        _d = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        _opts = _d.drawOptions()
        _opts.bondLineWidth = 2.0
        _opts.minFontSize = 14
        _opts.padding = 0.15
        _d.DrawMolecule(
            _submol,
            highlightAtoms=_highlight_atoms,
            highlightAtomColors=_atom_colors,
        )
        _d.FinishDrawing()
        return Image.open(io.BytesIO(_d.GetDrawingText()))

    _n_xgb = min(15, len(xgb_top_bits))
    _n_chemprop = min(15, len(chemprop_top_atom_types))

    _fig = plt.figure(figsize=(22, 12))
    _gs = _fig.add_gridspec(1, 4, width_ratios=[0.3, 0.7, 0.7, 0.3], wspace=0.05)

    # --- XGBoost substructure images ---
    _ax_xgb_img = _fig.add_subplot(_gs[0])
    _ax_xgb_img.set_xlim(0, 1)
    _ax_xgb_img.set_ylim(-0.5, _n_xgb - 0.5)
    _ax_xgb_img.invert_yaxis()
    _ax_xgb_img.axis("off")

    _xgb_bits = xgb_top_bits[:_n_xgb]

    for _j, _bit in enumerate(_xgb_bits):
        if int(_bit) in xgb_bit_to_mol_info:
            _mol, _center, _rad = xgb_bit_to_mol_info[int(_bit)]
            try:
                _img = _draw_neighborhood(_mol, _center, _rad)
                if _img is not None:
                    _inset = _ax_xgb_img.inset_axes(
                        [0.0, (_n_xgb - 1 - _j) / _n_xgb, 1.0, 0.9 / _n_xgb],
                        transform=_ax_xgb_img.transAxes,
                    )
                    _inset.imshow(_img)
                    _inset.axis("off")
            except Exception:
                pass

    # --- XGBoost SHAP bars ---
    _ax_xgb = _fig.add_subplot(_gs[1])
    _xgb_vals = []
    _xgb_colors = []
    _xgb_labels = []

    for _bit in _xgb_bits:
        _xgb_vals.append(float(xgb_mean_abs_shap[_bit]))
        _xgb_labels.append(f"Bit {_bit}")
        _xgb_colors.append("#2196F3" if xgb_mean_signed_shap[_bit] > 0 else "#FF5722")

    _y_pos = np.arange(_n_xgb)
    _ax_xgb.barh(
        _y_pos, _xgb_vals, color=_xgb_colors, alpha=0.8, height=0.7, edgecolor="none"
    )
    _ax_xgb.set_yticks(_y_pos)
    _ax_xgb.set_yticklabels(_xgb_labels, fontsize=9)
    _ax_xgb.set_xlabel("Mean |SHAP value|", fontsize=11)
    _ax_xgb.set_title(
        "XGBoost: Top Morgan FP Bits\n(blue = permeable, red = impermeable)",
        fontsize=12,
        fontweight="bold",
    )
    _ax_xgb.invert_yaxis()

    # --- Chemprop saliency bars ---
    _ax_cp = _fig.add_subplot(_gs[2])
    _cp_types = chemprop_top_atom_types[:_n_chemprop]
    _cp_means = [chemprop_atom_means[k] for k in _cp_types]
    _cp_errs = [chemprop_atom_stderrs[k] for k in _cp_types]

    _y_pos_cp = np.arange(_n_chemprop)
    _ax_cp.barh(
        _y_pos_cp,
        _cp_means,
        xerr=_cp_errs,
        color="#7E57C2",
        alpha=0.8,
        height=0.7,
        edgecolor="none",
        capsize=3,
    )
    _ax_cp.set_yticks(_y_pos_cp)
    _ax_cp.set_yticklabels(_cp_types, fontsize=9)
    _ax_cp.yaxis.tick_right()
    _ax_cp.yaxis.set_label_position("right")
    _ax_cp.set_xlabel("Mean Normalized Saliency", fontsize=11)
    _ax_cp.set_title(
        "Chemprop D-MPNN: Atom-Type Saliency\n(element + aromaticity + degree)",
        fontsize=12,
        fontweight="bold",
    )
    _ax_cp.invert_yaxis()

    # --- Chemprop fragment images ---
    _ax_cp_img = _fig.add_subplot(_gs[3])
    _ax_cp_img.set_xlim(0, 1)
    _ax_cp_img.set_ylim(-0.5, _n_chemprop - 0.5)
    _ax_cp_img.invert_yaxis()
    _ax_cp_img.axis("off")

    for _j, _atype in enumerate(_cp_types):
        if _atype in chemprop_atom_type_best:
            _mol, _atom_idx, _sal_val = chemprop_atom_type_best[_atype]
            try:
                _img = _draw_neighborhood(_mol, _atom_idx, 2)
                if _img is not None:
                    _inset = _ax_cp_img.inset_axes(
                        [
                            0.0,
                            (_n_chemprop - 1 - _j) / _n_chemprop,
                            1.0,
                            0.9 / _n_chemprop,
                        ],
                        transform=_ax_cp_img.transAxes,
                    )
                    _inset.imshow(_img)
                    _inset.axis("off")
            except Exception:
                pass

    _fig.suptitle(
        "PAMPA pH 7.4: Feature Importance by Architecture",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()

    _save_path = FIGURES_DIR / "pampa-feature-importance.png"
    _fig.savefig(_save_path, dpi=150, bbox_inches="tight", facecolor="white")
    logger.info(f"Saved combined figure to {_save_path}")

    mo.vstack(
        [
            mo.md("## Combined PAMPA Feature Importance"),
            mo.as_html(_fig),
            mo.md(f"""
**Left (XGBoost):** Top 6 Morgan fingerprint bits ranked by mean |SHAP|
on PAMPA test molecules. Each fragment shows the Morgan radius-3
neighborhood from the test molecule where that bit had the highest SHAP
contribution. Blue = pushes toward "permeable"; red = "impermeable."

**Right (Chemprop):** Top 6 atom types ranked by mean normalized
gradient saliency across all PAMPA test molecules. Each fragment
shows the 2-bond neighborhood around the highest-saliency instance of
    that atom type (center atom highlighted in blue). Error bars = SEM.

    For PAMPA (passive membrane permeability), we expect important features
    to relate to lipophilicity, molecular size, and hydrogen bond donors --
    distinct from the CYP-metabolism features that dominate the HLM figure.
            """),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
