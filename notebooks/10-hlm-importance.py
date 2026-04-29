import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md("""
    # 10 — HLM Feature Importance: XGBoost SHAP + Chemprop Saliency

    **Note:** This notebook computes single-fold exploratory analysis
    (rep 0, fold 0 only). For the cross-fold aggregate figures used in the
    README, see **notebook 16** (`16-aggregate-importance.py`).

    Computes feature importance for HLM Stability predictions using both
    XGBoost (SHAP on Morgan fingerprint bits) and Chemprop (per-atom
    gradient saliency). Produces a single combined figure showing what
    structural features both architectures agree drive metabolic stability
    predictions -- evidence that the models learn shared structural rules.
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
    """Load HLM and RLM split data."""
    import json

    with open(DATA_DIR / "split_config.json") as _f:
        _split_config = json.load(_f)

    _fp_data = np.load(DATA_DIR / "morgan_fps_2048_r3.npz", allow_pickle=True)
    global_fps = _fp_data["fp_matrix"]

    _hlm_split = np.load(DATA_DIR / "hlm_splits.npz", allow_pickle=True)
    hlm_smiles = list(_hlm_split["smiles"])
    hlm_labels = _hlm_split["labels"]
    hlm_folds = _hlm_split["folds"]
    hlm_fp_indices = _hlm_split["fp_indices"]
    hlm_X = global_fps[hlm_fp_indices]

    _rlm_split = np.load(DATA_DIR / "rlm_splits.npz", allow_pickle=True)
    rlm_smiles = list(_rlm_split["smiles"])
    rlm_labels = _rlm_split["labels"]
    rlm_fp_indices = _rlm_split["fp_indices"]
    rlm_X = global_fps[rlm_fp_indices]

    logger.info(f"HLM: {hlm_X.shape[0]} molecules, RLM: {rlm_X.shape[0]} molecules")
    return (
        hlm_X,
        hlm_folds,
        hlm_labels,
        hlm_smiles,
        rlm_X,
        rlm_labels,
        rlm_smiles,
    )


@app.cell
def _(hlm_X, hlm_folds, hlm_labels, logger, np, roc_auc_score, shap, xgb):
    """Train XGBoost on HLM (rep 0, fold 0) and compute SHAP values."""
    np.random.seed(42)
    _fold_assign = hlm_folds[0]
    xgb_test_mask = _fold_assign == 0
    _train_mask = ~xgb_test_mask

    _dtrain = xgb.DMatrix(hlm_X[_train_mask], label=hlm_labels[_train_mask])
    _dval = xgb.DMatrix(hlm_X[xgb_test_mask], label=hlm_labels[xgb_test_mask])

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
        f"XGBoost HLM AUC: {roc_auc_score(hlm_labels[xgb_test_mask], _y_prob):.3f}"
    )

    xgb_explainer = shap.TreeExplainer(xgb_model)
    xgb_shap_values = xgb_explainer.shap_values(hlm_X[xgb_test_mask])

    # Mean absolute and signed SHAP per bit
    xgb_mean_abs_shap = np.abs(xgb_shap_values).mean(axis=0)
    xgb_mean_signed_shap = xgb_shap_values.mean(axis=0)
    xgb_top_bits = np.argsort(xgb_mean_abs_shap)[-6:][::-1]

    logger.info(f"Top 15 bits: {xgb_top_bits.tolist()}")
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
    hlm_smiles,
    logger,
    rdFingerprintGenerator,
    xgb_shap_values,
    xgb_test_mask,
    xgb_top_bits,
):
    """For each top SHAP bit, find the molecule where it contributes most and store mol+atom info."""
    _gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
    _test_smiles = [hlm_smiles[i] for i in range(len(hlm_smiles)) if xgb_test_mask[i]]

    # For each bit: (mol, center_atom_idx, radius) from the highest-|SHAP| molecule
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
    hlm_folds,
    hlm_labels,
    hlm_smiles,
    lightning_pl,
    logger,
    models,
    nn,
    np,
    roc_auc_score,
    torch,
):
    """Train Chemprop on HLM and compute per-atom saliency with best-instance tracking."""
    _featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    _fold_assign = hlm_folds[0]
    chemprop_test_mask = _fold_assign == 0
    _train_mask = ~chemprop_test_mask

    _train_smi = [hlm_smiles[i] for i in range(len(hlm_smiles)) if _train_mask[i]]
    _train_y = hlm_labels[_train_mask].reshape(-1, 1).astype(float)
    _test_smi = [hlm_smiles[i] for i in range(len(hlm_smiles)) if chemprop_test_mask[i]]
    _test_y = hlm_labels[chemprop_test_mask].reshape(-1, 1).astype(float)

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
        f"Chemprop HLM AUC: {roc_auc_score(hlm_labels[chemprop_test_mask], _y_prob):.3f}"
    )

    # Compute atom saliency with best-instance tracking per atom type
    chemprop_atom_type_saliency = defaultdict(list)
    chemprop_atom_type_best = {}  # key -> (mol, atom_idx, norm_saliency)

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
    """Generate the combined HLM feature importance figure.

    Draws substructure neighborhoods directly from parent molecules
    (no SMILES round-trip) to ensure structural accuracy.
    """

    def _draw_neighborhood(mol, center_atom, radius, size=(220, 160)):
        """Extract and draw only the atom neighborhood (not the full molecule).

        Uses PathToSubmol to extract the substructure, then highlights the
        center atom. Ensures at least a 2-bond neighborhood for readability.

        Args:
            mol: RDKit Mol object (the full parent molecule).
            center_atom: Index of the central atom.
            radius: Bond radius of the environment to extract (min 2 enforced).
            size: Tuple of (width, height) in pixels.

        Returns:
            PIL Image of the extracted fragment, or None on failure.
        """
        # Enforce minimum radius of 2 for readability
        _effective_radius = max(radius, 2)

        _env = Chem.FindAtomEnvironmentOfRadiusN(mol, _effective_radius, center_atom)
        if not _env:
            # Fallback: try radius 1
            _env = Chem.FindAtomEnvironmentOfRadiusN(mol, 1, center_atom)
        if not _env:
            # Last resort: single atom
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

        _amap = {}
        _submol = Chem.PathToSubmol(mol, _env, atomMap=_amap)
        if _submol is None or _submol.GetNumAtoms() == 0:
            return None

        _center_in_sub = _amap.get(center_atom, -1)
        _highlight_atoms = list(range(_submol.GetNumAtoms()))
        # Color all atoms light gray, center atom medium blue (reduced alpha)
        _atom_colors = {a: (0.92, 0.92, 0.92, 1.0) for a in _highlight_atoms}
        if _center_in_sub >= 0:
            _atom_colors[_center_in_sub] = (0.4, 0.65, 0.9, 0.5)
        _atom_radii = {a: 0.3 for a in _highlight_atoms}
        if _center_in_sub >= 0:
            _atom_radii[_center_in_sub] = 0.4

        _d = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        _opts = _d.drawOptions()
        _opts.bondLineWidth = 2.0
        _opts.minFontSize = 14
        _opts.padding = 0.15
        _d.DrawMolecule(
            _submol,
            highlightAtoms=_highlight_atoms,
            highlightAtomColors=_atom_colors,
            highlightAtomRadii=_atom_radii,
        )
        _d.FinishDrawing()
        return Image.open(io.BytesIO(_d.GetDrawingText()))

    _n_xgb = min(15, len(xgb_top_bits))
    _n_chemprop = min(15, len(chemprop_top_atom_types))

    # --- Build figure: 4 columns ---
    _fig = plt.figure(figsize=(22, 12))
    _gs = _fig.add_gridspec(1, 4, width_ratios=[0.3, 0.7, 0.7, 0.3], wspace=0.08)

    # --- XGBoost substructure images (left of bars) ---
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
        "XGBoost: Top Morgan FP Bits\n(blue = stable, red = unstable)",
        fontsize=12,
        fontweight="bold",
    )
    _ax_xgb.invert_yaxis()

    # Draw connecting lines between XGB bars and images
    for _j in range(_n_xgb):
        _ax_xgb.annotate(
            "",
            xy=(0, _j),
            xytext=(-0.02, _j),
            xycoords="data",
            textcoords="data",
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.5),
        )

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

    # --- Chemprop fragment images (right of bars) ---
    _ax_cp_img = _fig.add_subplot(_gs[3])
    _ax_cp_img.set_xlim(0, 1)
    _ax_cp_img.set_ylim(-0.5, _n_chemprop - 0.5)
    _ax_cp_img.invert_yaxis()
    _ax_cp_img.axis("off")

    for _j, _atype in enumerate(_cp_types):
        if _atype in chemprop_atom_type_best:
            _mol, _atom_idx, _sal_val = chemprop_atom_type_best[_atype]
            try:
                # Draw 1-bond neighborhood -- keeps fragments small and focused
                _img = _draw_neighborhood(_mol, _atom_idx, 1)
                if _img is not None:
                    _inset = _ax_cp_img.inset_axes(
                        [
                            0.15,
                            (_n_chemprop - 1 - _j) / _n_chemprop,
                            0.85,
                            0.9 / _n_chemprop,
                        ],
                        transform=_ax_cp_img.transAxes,
                    )
                    _inset.imshow(_img)
                    _inset.axis("off")
            except Exception:
                pass

    _fig.suptitle(
        "HLM Stability: Feature Importance by Architecture",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()

    _save_path = FIGURES_DIR / "hlm-feature-importance.png"
    _fig.savefig(_save_path, dpi=150, bbox_inches="tight", facecolor="white")
    logger.info(f"Saved combined figure to {_save_path}")

    mo.vstack(
        [
            mo.md("## Combined HLM Feature Importance"),
            mo.as_html(_fig),
            mo.md(f"""
    **Left (XGBoost):** Top 6 Morgan fingerprint bits ranked by mean |SHAP|
    on HLM test molecules. Each fragment image shows the Morgan radius-3
    neighborhood that activates that bit, drawn from the test molecule where
    that bit had the highest SHAP contribution (i.e., the molecule where the
    bit mattered most). Blue = pushes toward "stable"; red = "unstable."

    **Right (Chemprop):** Top 6 atom types ranked by mean normalized
    gradient saliency (|dPred/dAtomFeatures|) across all HLM test
    molecules. Each fragment shows the 1-bond neighborhood around the
    highest-saliency instance of that atom type (center atom highlighted in
    blue). Error bars = SEM.

    Both architectures attend to nitrogen environments and heteroatom
    functional groups -- substructures governing CYP-mediated metabolic
    stability.
            """),
        ]
    )
    return


@app.cell
def _(
    hlm_X,
    hlm_folds,
    hlm_labels,
    logger,
    np,
    rlm_X,
    rlm_labels,
    roc_auc_score,
    shap,
    xgb,
):
    """Train XGBoost RLM-transfer on HLM and compute SHAP values."""
    np.random.seed(42)
    _fold_assign = hlm_folds[0]
    _test_mask = _fold_assign == 0
    _train_mask = ~_test_mask

    # Pre-train on RLM
    _dtrain_rlm = xgb.DMatrix(rlm_X, label=rlm_labels)
    _rlm_model = xgb.train(
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
        _dtrain_rlm,
        200,
        verbose_eval=False,
    )

    # Continue boosting on HLM
    _dtrain_hlm = xgb.DMatrix(hlm_X[_train_mask], label=hlm_labels[_train_mask])
    _dval_hlm = xgb.DMatrix(hlm_X[_test_mask], label=hlm_labels[_test_mask])
    xgb_transfer_model = xgb.train(
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
        _dtrain_hlm,
        200,
        evals=[(_dval_hlm, "val")],
        early_stopping_rounds=20,
        verbose_eval=False,
        xgb_model=_rlm_model,
    )

    _y_prob = xgb_transfer_model.predict(_dval_hlm)
    logger.info(
        f"XGBoost RLM-transfer HLM AUC: {roc_auc_score(hlm_labels[_test_mask], _y_prob):.3f}"
    )

    _explainer = shap.TreeExplainer(xgb_transfer_model)
    xgb_transfer_shap_values = _explainer.shap_values(hlm_X[_test_mask])

    xgb_transfer_mean_abs_shap = np.abs(xgb_transfer_shap_values).mean(axis=0)
    xgb_transfer_mean_signed_shap = xgb_transfer_shap_values.mean(axis=0)
    xgb_transfer_top_bits = np.argsort(xgb_transfer_mean_abs_shap)[-6:][::-1]
    return (
        xgb_transfer_mean_abs_shap,
        xgb_transfer_mean_signed_shap,
        xgb_transfer_shap_values,
        xgb_transfer_top_bits,
    )


@app.cell
def _(
    Chem,
    hlm_smiles,
    logger,
    rdFingerprintGenerator,
    xgb_test_mask,
    xgb_transfer_shap_values,
    xgb_transfer_top_bits,
):
    """Map XGBoost transfer top bits to mol info."""
    _gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
    _test_smiles = [hlm_smiles[i] for i in range(len(hlm_smiles)) if xgb_test_mask[i]]

    xgb_transfer_bit_to_mol_info = {}
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
            _shap_val = abs(float(xgb_transfer_shap_values[_mol_idx, _bit]))
            if _bit not in _bit_best_shap or _shap_val > _bit_best_shap[_bit]:
                _center, _rad = _envs[0]
                _bit_best_shap[_bit] = _shap_val
                xgb_transfer_bit_to_mol_info[_bit] = (_mol, _center, _rad)

    _coverage = sum(
        1 for b in xgb_transfer_top_bits if int(b) in xgb_transfer_bit_to_mol_info
    )
    logger.info(f"XGB transfer bit coverage: {_coverage}/{len(xgb_transfer_top_bits)}")
    return (xgb_transfer_bit_to_mol_info,)


@app.cell
def _(
    Chem,
    FIGURES_DIR,
    Image,
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
    xgb_transfer_bit_to_mol_info,
    xgb_transfer_mean_abs_shap,
    xgb_transfer_mean_signed_shap,
    xgb_transfer_top_bits,
):
    """Plot XGBoost scratch vs transfer SHAP comparison for HLM."""

    def _draw_neighborhood_xgb(mol, center_atom, radius, size=(220, 160)):
        _effective_radius = max(radius, 2)
        _env = Chem.FindAtomEnvironmentOfRadiusN(mol, _effective_radius, center_atom)
        if not _env:
            _env = Chem.FindAtomEnvironmentOfRadiusN(mol, 1, center_atom)
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

    _n_left = 6
    _n_right = 6
    _bits_left = xgb_top_bits[:_n_left]
    _bits_right = xgb_transfer_top_bits[:_n_right]

    _fig = plt.figure(figsize=(22, 8))
    _gs = _fig.add_gridspec(1, 4, width_ratios=[0.3, 0.7, 0.7, 0.3], wspace=0.08)

    # Left images (scratch)
    _ax_img_l = _fig.add_subplot(_gs[0])
    _ax_img_l.set_xlim(0, 1)
    _ax_img_l.set_ylim(-0.5, _n_left - 0.5)
    _ax_img_l.invert_yaxis()
    _ax_img_l.axis("off")
    for _j, _bit in enumerate(_bits_left):
        if int(_bit) in xgb_bit_to_mol_info:
            _mol, _center, _rad = xgb_bit_to_mol_info[int(_bit)]
            try:
                _img = _draw_neighborhood_xgb(_mol, _center, _rad)
                if _img:
                    _inset = _ax_img_l.inset_axes(
                        [0.0, (_n_left - 1 - _j) / _n_left, 1.0, 0.9 / _n_left],
                        transform=_ax_img_l.transAxes,
                    )
                    _inset.imshow(_img)
                    _inset.axis("off")
            except Exception:
                pass

    # Left bars (scratch)
    _ax_l = _fig.add_subplot(_gs[1])
    _vals_l = [float(xgb_mean_abs_shap[b]) for b in _bits_left]
    _colors_l = [
        "#2196F3" if xgb_mean_signed_shap[b] > 0 else "#FF5722" for b in _bits_left
    ]
    _y_pos_l = np.arange(_n_left)
    _ax_l.barh(_y_pos_l, _vals_l, color=_colors_l, alpha=0.8, height=0.7)
    _ax_l.set_yticks(_y_pos_l)
    _ax_l.set_yticklabels([f"Bit {b}" for b in _bits_left], fontsize=9)
    _ax_l.set_xlabel("Mean |SHAP value|", fontsize=11)
    _ax_l.set_title(
        "XGBoost Scratch\n(blue = stable, red = unstable)",
        fontsize=12,
        fontweight="bold",
    )
    _ax_l.invert_yaxis()

    # Right bars (transfer)
    _ax_r = _fig.add_subplot(_gs[2])
    _vals_r = [float(xgb_transfer_mean_abs_shap[b]) for b in _bits_right]
    _colors_r = [
        "#2196F3" if xgb_transfer_mean_signed_shap[b] > 0 else "#FF5722"
        for b in _bits_right
    ]
    _y_pos_r = np.arange(_n_right)
    _ax_r.barh(_y_pos_r, _vals_r, color=_colors_r, alpha=0.8, height=0.7)
    _ax_r.set_yticks(_y_pos_r)
    _ax_r.set_yticklabels([f"Bit {b}" for b in _bits_right], fontsize=9)
    _ax_r.yaxis.tick_right()
    _ax_r.yaxis.set_label_position("right")
    _ax_r.set_xlabel("Mean |SHAP value|", fontsize=11)
    _ax_r.set_title(
        "XGBoost RLM-Transfer\n(blue = stable, red = unstable)",
        fontsize=12,
        fontweight="bold",
    )
    _ax_r.invert_yaxis()

    # Right images (transfer)
    _ax_img_r = _fig.add_subplot(_gs[3])
    _ax_img_r.set_xlim(0, 1)
    _ax_img_r.set_ylim(-0.5, _n_right - 0.5)
    _ax_img_r.invert_yaxis()
    _ax_img_r.axis("off")
    for _j, _bit in enumerate(_bits_right):
        if int(_bit) in xgb_transfer_bit_to_mol_info:
            _mol, _center, _rad = xgb_transfer_bit_to_mol_info[int(_bit)]
            try:
                _img = _draw_neighborhood_xgb(_mol, _center, _rad)
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
        "HLM Stability: XGBoost Scratch vs RLM-Transfer (SHAP)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    _save_path = FIGURES_DIR / "hlm-xgb-scratch-vs-transfer.png"
    _fig.savefig(_save_path, dpi=150, bbox_inches="tight", facecolor="white")
    logger.info(f"Saved {_save_path}")

    mo.vstack(
        [
            mo.md("## HLM: XGBoost Scratch vs RLM-Transfer"),
            mo.as_html(_fig),
            mo.md(f"""
    *Top 6 SHAP features for each model. Shared bits between the two panels
    indicate features important to both; unique bits show what transfer adds
    or changes. Single fold; rankings may vary with different test sets.*
        """),
        ]
    )
    return


@app.cell
def _(
    Chem,
    chemprop_data,
    defaultdict,
    featurizers,
    hlm_folds,
    hlm_labels,
    hlm_smiles,
    lightning_pl,
    logger,
    models,
    nn,
    np,
    rlm_labels,
    rlm_smiles,
    roc_auc_score,
    torch,
):
    """Train Chemprop RLM-transfer on HLM and compute saliency."""
    _featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    lightning_pl.seed_everything(42, workers=True)

    # Pre-train on RLM
    _rlm_y = rlm_labels.reshape(-1, 1).astype(float)
    _n_rlm = len(rlm_smiles)
    _n_rlm_val = max(1, int(_n_rlm * 0.1))
    _rng = np.random.default_rng(42)
    _perm_rlm = _rng.permutation(_n_rlm)

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

    _mp = nn.BondMessagePassing()
    _agg = nn.MeanAggregation()
    _ffn = nn.BinaryClassificationFFN(input_dim=_mp.output_dim)
    _rlm_model = models.MPNN(_mp, _agg, _ffn, batch_norm=False)

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
    logger.info("Chemprop RLM pre-training done")

    # Finetune on HLM with new FFN head
    _fold_assign = hlm_folds[0]
    chemprop_transfer_test_mask = _fold_assign == 0
    _train_mask = ~chemprop_transfer_test_mask

    _train_smi = [hlm_smiles[i] for i in range(len(hlm_smiles)) if _train_mask[i]]
    _train_y = hlm_labels[_train_mask].reshape(-1, 1).astype(float)
    _test_smi = [
        hlm_smiles[i] for i in range(len(hlm_smiles)) if chemprop_transfer_test_mask[i]
    ]
    _test_y = hlm_labels[chemprop_transfer_test_mask].reshape(-1, 1).astype(float)

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

    # New FFN head, keep pre-trained encoder
    _new_ffn = nn.BinaryClassificationFFN(
        input_dim=_rlm_model.message_passing.output_dim
    )
    _transfer_model = models.MPNN(
        _rlm_model.message_passing, _rlm_model.agg, _new_ffn, batch_norm=False
    )

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
    _trainer2.fit(_transfer_model, _train_loader, _val_loader)
    _preds = _trainer2.predict(_transfer_model, _test_loader)
    _y_prob = torch.cat(_preds).cpu().numpy().flatten()
    logger.info(
        f"Chemprop RLM-transfer HLM AUC: {roc_auc_score(hlm_labels[chemprop_transfer_test_mask], _y_prob):.3f}"
    )

    # Compute saliency
    chemprop_transfer_saliency = defaultdict(list)
    chemprop_transfer_best = {}

    _n_success = 0
    for _i in range(len(_test_smi)):
        _smi = _test_smi[_i]
        _mol = Chem.MolFromSmiles(_smi)
        if _mol is None:
            continue
        _transfer_model.eval()
        _dp = chemprop_data.MoleculeDatapoint.from_smi(_smi, [0.0])
        _dset = chemprop_data.MoleculeDataset([_dp], _featurizer)
        _loader = chemprop_data.build_dataloader(
            _dset, num_workers=0, batch_size=1, shuffle=False
        )
        _batch = next(iter(_loader))
        _bmg = _batch[0]
        _bmg.V = _bmg.V.clone().requires_grad_(True)
        try:
            _pred = _transfer_model(_bmg)
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
            chemprop_transfer_saliency[_key].append(_norm_sal)
            if (
                _key not in chemprop_transfer_best
                or _norm_sal > chemprop_transfer_best[_key][2]
            ):
                chemprop_transfer_best[_key] = (_mol, _a, _norm_sal)
        _n_success += 1

    logger.info(f"Chemprop transfer saliency: {_n_success} molecules")

    chemprop_transfer_means = {
        k: float(np.mean(v)) for k, v in chemprop_transfer_saliency.items()
    }
    chemprop_transfer_stderrs = {
        k: float(np.std(v) / np.sqrt(len(v)))
        for k, v in chemprop_transfer_saliency.items()
    }
    chemprop_transfer_top = sorted(
        chemprop_transfer_means.keys(),
        key=lambda k: chemprop_transfer_means[k],
        reverse=True,
    )[:6]
    return (
        chemprop_transfer_best,
        chemprop_transfer_means,
        chemprop_transfer_stderrs,
        chemprop_transfer_top,
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
    chemprop_transfer_best,
    chemprop_transfer_means,
    chemprop_transfer_stderrs,
    chemprop_transfer_top,
    io,
    logger,
    mo,
    np,
    plt,
    rdMolDraw2D,
):
    """Plot Chemprop scratch vs transfer saliency comparison for HLM."""

    def _draw_neighborhood_cp(mol, center_atom, radius, size=(220, 160)):
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

    _n_left = min(6, len(chemprop_top_atom_types))
    _n_right = min(6, len(chemprop_transfer_top))
    _types_left = chemprop_top_atom_types[:_n_left]
    _types_right = chemprop_transfer_top[:_n_right]

    _fig = plt.figure(figsize=(22, 8))
    _gs = _fig.add_gridspec(1, 4, width_ratios=[0.3, 0.7, 0.7, 0.3], wspace=0.08)

    # Left images (scratch)
    _ax_img_l = _fig.add_subplot(_gs[0])
    _ax_img_l.set_xlim(0, 1)
    _ax_img_l.set_ylim(-0.5, _n_left - 0.5)
    _ax_img_l.invert_yaxis()
    _ax_img_l.axis("off")
    for _j, _atype in enumerate(_types_left):
        if _atype in chemprop_atom_type_best:
            _mol, _idx, _ = chemprop_atom_type_best[_atype]
            try:
                _img = _draw_neighborhood_cp(_mol, _idx, 1)
                if _img:
                    _inset = _ax_img_l.inset_axes(
                        [0.0, (_n_left - 1 - _j) / _n_left, 1.0, 0.9 / _n_left],
                        transform=_ax_img_l.transAxes,
                    )
                    _inset.imshow(_img)
                    _inset.axis("off")
            except Exception:
                pass

    # Left bars (scratch)
    _ax_l = _fig.add_subplot(_gs[1])
    _means_l = [chemprop_atom_means[k] for k in _types_left]
    _errs_l = [chemprop_atom_stderrs[k] for k in _types_left]
    _y_pos_l = np.arange(_n_left)
    _ax_l.barh(
        _y_pos_l,
        _means_l,
        xerr=_errs_l,
        color="#7E57C2",
        alpha=0.8,
        height=0.7,
        capsize=3,
    )
    _ax_l.set_yticks(_y_pos_l)
    _ax_l.set_yticklabels(_types_left, fontsize=9)
    _ax_l.set_xlabel("Mean Normalized Saliency", fontsize=11)
    _ax_l.set_title("Chemprop Scratch", fontsize=12, fontweight="bold")
    _ax_l.invert_yaxis()

    # Right bars (transfer)
    _ax_r = _fig.add_subplot(_gs[2])
    _means_r = [chemprop_transfer_means[k] for k in _types_right]
    _errs_r = [chemprop_transfer_stderrs[k] for k in _types_right]
    _y_pos_r = np.arange(_n_right)
    _ax_r.barh(
        _y_pos_r,
        _means_r,
        xerr=_errs_r,
        color="#00897B",
        alpha=0.8,
        height=0.7,
        capsize=3,
    )
    _ax_r.set_yticks(_y_pos_r)
    _ax_r.set_yticklabels(_types_right, fontsize=9)
    _ax_r.yaxis.tick_right()
    _ax_r.yaxis.set_label_position("right")
    _ax_r.set_xlabel("Mean Normalized Saliency", fontsize=11)
    _ax_r.set_title("Chemprop RLM-Transfer", fontsize=12, fontweight="bold")
    _ax_r.invert_yaxis()

    # Right images (transfer)
    _ax_img_r = _fig.add_subplot(_gs[3])
    _ax_img_r.set_xlim(0, 1)
    _ax_img_r.set_ylim(-0.5, _n_right - 0.5)
    _ax_img_r.invert_yaxis()
    _ax_img_r.axis("off")
    for _j, _atype in enumerate(_types_right):
        if _atype in chemprop_transfer_best:
            _mol, _idx, _ = chemprop_transfer_best[_atype]
            try:
                _img = _draw_neighborhood_cp(_mol, _idx, 1)
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
        "HLM Stability: Chemprop Scratch vs RLM-Transfer (Saliency)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    _save_path = FIGURES_DIR / "hlm-chemprop-scratch-vs-transfer.png"
    _fig.savefig(_save_path, dpi=150, bbox_inches="tight", facecolor="white")
    logger.info(f"Saved {_save_path}")

    mo.vstack(
        [
            mo.md("## HLM: Chemprop Scratch vs RLM-Transfer"),
            mo.as_html(_fig),
            mo.md("""
    *Top 6 atom types by gradient saliency for each model. Similar rankings
    indicate the transfer model preserved useful attention patterns from
    pre-training. Single fold; rankings may vary with different test sets.*
        """),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
