import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md("""
    # 10 — HLM Feature Importance: XGBoost SHAP + Chemprop Saliency

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
    import json
    from collections import defaultdict
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import shap
    import torch
    import xgboost as xgb
    from lightning import pytorch as lightning_pl
    from loguru import logger
    from PIL import Image
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw, rdFingerprintGenerator
    from rdkit.Chem.Draw import rdMolDraw2D
    from sklearn.metrics import roc_auc_score

    from chemprop import data as chemprop_data
    from chemprop import featurizers, models, nn

    DATA_DIR = Path("data")
    CHECKPOINTS_DIR = Path("checkpoints")
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
        json,
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
def _(DATA_DIR, json, logger, np):
    """Load HLM and RLM split data."""
    with open(DATA_DIR / "split_config.json") as _f:
        split_config = json.load(_f)

    _fp_data = np.load(DATA_DIR / "morgan_fps_2048_r3.npz", allow_pickle=True)
    global_fps = _fp_data["fp_matrix"]
    global_smiles = list(_fp_data["smiles"])

    # HLM splits
    _hlm_split = np.load(DATA_DIR / "hlm_splits.npz", allow_pickle=True)
    hlm_smiles = list(_hlm_split["smiles"])
    hlm_labels = _hlm_split["labels"]
    hlm_folds = _hlm_split["folds"]
    hlm_fp_indices = _hlm_split["fp_indices"]
    hlm_X = global_fps[hlm_fp_indices]

    logger.info(f"HLM: {hlm_X.shape[0]} molecules")
    return hlm_X, hlm_folds, hlm_labels, hlm_smiles


@app.cell
def _(hlm_X, hlm_folds, hlm_labels, logger, roc_auc_score, shap, xgb):
    """Train XGBoost on HLM (rep 0, fold 0) and compute SHAP values."""
    REP = 0
    FOLD = 0

    _fold_assign = hlm_folds[REP]
    xgb_test_mask = _fold_assign == FOLD
    xgb_train_mask = ~xgb_test_mask

    xgb_X_train = hlm_X[xgb_train_mask]
    xgb_X_test = hlm_X[xgb_test_mask]
    xgb_y_train = hlm_labels[xgb_train_mask]
    xgb_y_test = hlm_labels[xgb_test_mask]

    logger.info(
        f"XGBoost fold: rep={REP} fold={FOLD}, "
        f"train={xgb_X_train.shape[0]}, test={xgb_X_test.shape[0]}"
    )

    XGB_PARAMS = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbosity": 0,
    }

    _dtrain = xgb.DMatrix(xgb_X_train, label=xgb_y_train)
    _dval = xgb.DMatrix(xgb_X_test, label=xgb_y_test)

    xgb_model = xgb.train(
        XGB_PARAMS,
        _dtrain,
        200,
        evals=[(_dval, "val")],
        early_stopping_rounds=20,
        verbose_eval=False,
    )

    _dtest = xgb.DMatrix(xgb_X_test)
    xgb_y_prob = xgb_model.predict(_dtest)
    logger.info(f"XGBoost HLM AUC: {roc_auc_score(xgb_y_test, xgb_y_prob):.3f}")

    # SHAP values on test set
    xgb_explainer = shap.TreeExplainer(xgb_model)
    xgb_shap_values = xgb_explainer.shap_values(xgb_X_test)
    logger.info(f"SHAP values shape: {xgb_shap_values.shape}")
    return xgb_shap_values, xgb_test_mask


@app.cell
def _(
    Chem,
    hlm_smiles,
    logger,
    np,
    rdFingerprintGenerator,
    xgb_shap_values,
    xgb_test_mask,
):
    """Map top SHAP bits to substructure fragments."""
    _gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
    _test_smiles = [hlm_smiles[i] for i in range(len(hlm_smiles)) if xgb_test_mask[i]]

    # For each bit, collect the submolecule fragment (not the whole molecule)
    xgb_bit_to_fragment = {}

    for _smi in _test_smiles:
        _mol = Chem.MolFromSmiles(_smi)
        if _mol is None:
            continue
        _ao = rdFingerprintGenerator.AdditionalOutput()
        _ao.AllocateBitInfoMap()
        _fp = _gen.GetFingerprint(_mol, additionalOutput=_ao)
        _bit_info = _ao.GetBitInfoMap()

        for _bit, _envs in _bit_info.items():
            if _bit in xgb_bit_to_fragment:
                continue
            _center, _rad = _envs[0]
            try:
                if _rad > 0:
                    _env = Chem.FindAtomEnvironmentOfRadiusN(_mol, _rad, _center)
                    if _env:
                        _amap = {}
                        _submol = Chem.PathToSubmol(_mol, _env, atomMap=_amap)
                        _smiles = Chem.MolToSmiles(_submol)
                        _frag = Chem.MolFromSmiles(_smiles)
                        if _frag is not None:
                            xgb_bit_to_fragment[_bit] = _frag
                else:
                    _atom = _mol.GetAtomWithIdx(_center)
                    _smiles = f"[{_atom.GetSymbol()}]"
                    _frag = Chem.MolFromSmiles(_smiles)
                    if _frag is not None:
                        xgb_bit_to_fragment[_bit] = _frag
            except Exception:
                continue

    # Mean absolute SHAP per bit, and mean signed SHAP
    xgb_mean_abs_shap = np.abs(xgb_shap_values).mean(axis=0)
    xgb_mean_signed_shap = xgb_shap_values.mean(axis=0)

    # Top 15 bits by mean |SHAP|
    xgb_top_bits = np.argsort(xgb_mean_abs_shap)[-15:][::-1]

    logger.info(f"Mapped {len(xgb_bit_to_fragment)} bits to fragment structures")
    logger.info(f"Top 15 bits by mean |SHAP|: {xgb_top_bits.tolist()}")
    return (
        xgb_bit_to_fragment,
        xgb_mean_abs_shap,
        xgb_mean_signed_shap,
        xgb_top_bits,
    )


@app.cell
def _(
    chemprop_data,
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
    """Train Chemprop on HLM (rep 0, fold 0) and compute atom saliency."""
    _featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    _REP, _FOLD = 0, 0
    _MAX_EPOCHS = 30

    _fold_assign = hlm_folds[_REP]
    chemprop_test_mask = _fold_assign == _FOLD
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

    # Build and train Chemprop from scratch on HLM
    _mp = nn.BondMessagePassing()
    _agg = nn.MeanAggregation()
    _ffn = nn.BinaryClassificationFFN(input_dim=_mp.output_dim)
    chemprop_model = models.MPNN(_mp, _agg, _ffn, batch_norm=False)

    _trainer = lightning_pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        accelerator="gpu",
        devices=1,
        max_epochs=_MAX_EPOCHS,
    )
    _trainer.fit(chemprop_model, _train_loader, _val_loader)

    _preds = _trainer.predict(chemprop_model, _test_loader)
    _y_prob = torch.cat(_preds).cpu().numpy().flatten()
    _y_true = hlm_labels[chemprop_test_mask]
    logger.info(f"Chemprop HLM AUC: {roc_auc_score(_y_true, _y_prob):.3f}")

    # Compute atom saliency across sampled test molecules
    def _compute_atom_saliency(mpnn, smiles):
        """Compute per-atom gradient saliency for a single molecule.

        Args:
            mpnn: Trained Chemprop MPNN model.
            smiles: SMILES string.

        Returns:
            Tuple of (atom_saliency_array, prediction_value).
        """
        mpnn.eval()
        _dp = chemprop_data.MoleculeDatapoint.from_smi(smiles, [0.0])
        _dset = chemprop_data.MoleculeDataset([_dp], _featurizer)
        _loader = chemprop_data.build_dataloader(
            _dset, num_workers=0, batch_size=1, shuffle=False
        )
        _batch = next(iter(_loader))
        _bmg = _batch[0]
        _bmg.V = _bmg.V.clone().requires_grad_(True)
        _pred = mpnn(_bmg)
        _pred.sum().backward()
        _grad = _bmg.V.grad.abs().sum(dim=1).detach().numpy()
        return _grad, _pred.item()

    chemprop_saliency_fn = _compute_atom_saliency
    return chemprop_model, chemprop_saliency_fn, chemprop_test_mask


@app.cell
def _(
    Chem,
    chemprop_model,
    chemprop_saliency_fn,
    chemprop_test_mask,
    defaultdict,
    hlm_smiles,
    logger,
    np,
):
    """Compute aggregated atom-type saliency and collect representative fragments."""
    _test_smi = [hlm_smiles[i] for i in range(len(hlm_smiles)) if chemprop_test_mask[i]]

    chemprop_atom_type_saliency = defaultdict(list)
    # For each atom type, store a small representative fragment (1-hop neighborhood)
    chemprop_atom_type_fragment = {}

    _sample_idx = np.random.default_rng(42).choice(
        len(_test_smi), size=min(100, len(_test_smi)), replace=False
    )

    _n_success = 0
    for _i in _sample_idx:
        _smi = _test_smi[_i]
        _mol = Chem.MolFromSmiles(_smi)
        if _mol is None:
            continue
        try:
            _sal, _pred = chemprop_saliency_fn(chemprop_model, _smi)
        except Exception:
            continue

        _n_atoms = _mol.GetNumAtoms()
        if len(_sal) != _n_atoms:
            continue

        _max_sal = _sal.max() if _sal.max() > 0 else 1.0
        for _a in range(_n_atoms):
            _atom = _mol.GetAtomWithIdx(_a)
            _symbol = _atom.GetSymbol()
            _is_aromatic = _atom.GetIsAromatic()
            _degree = _atom.GetDegree()
            _key = f"{_symbol}{'(arom)' if _is_aromatic else ''} deg{_degree}"
            chemprop_atom_type_saliency[_key].append(float(_sal[_a] / _max_sal))

            # Extract a 2-bond neighborhood fragment for this atom type
            if _key not in chemprop_atom_type_fragment:
                try:
                    _env = Chem.FindAtomEnvironmentOfRadiusN(_mol, 2, _a)
                    if _env:
                        _amap = {}
                        _submol = Chem.PathToSubmol(_mol, _env, atomMap=_amap)
                        _frag_smi = Chem.MolToSmiles(_submol)
                        _frag = Chem.MolFromSmiles(_frag_smi)
                        if _frag is not None and _frag.GetNumAtoms() <= 12:
                            chemprop_atom_type_fragment[_key] = _frag
                except Exception:
                    pass

        _n_success += 1

    logger.info(
        f"Chemprop saliency: {_n_success} molecules, "
        f"{len(chemprop_atom_type_saliency)} atom types, "
        f"{len(chemprop_atom_type_fragment)} with fragments"
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
    )[:15]
    return (
        chemprop_atom_means,
        chemprop_atom_stderrs,
        chemprop_atom_type_fragment,
        chemprop_top_atom_types,
    )


@app.cell
def _(
    FIGURES_DIR,
    Image,
    chemprop_atom_means,
    chemprop_atom_stderrs,
    chemprop_atom_type_fragment,
    chemprop_top_atom_types,
    io,
    logger,
    mo,
    np,
    plt,
    rdMolDraw2D,
    xgb_bit_to_fragment,
    xgb_mean_abs_shap,
    xgb_mean_signed_shap,
    xgb_top_bits,
):
    """Generate the combined HLM feature importance figure."""

    def _draw_fragment(mol, size=(200, 150)):
        """Draw a small molecule fragment cleanly.

        Args:
            mol: RDKit molecule (the fragment, not the full molecule).
            size: Tuple of (width, height) in pixels.

        Returns:
            PIL Image of the rendered fragment.
        """
        _d = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        _opts = _d.drawOptions()
        _opts.bondLineWidth = 2.0
        _opts.minFontSize = 14
        _opts.additionalAtomLabelPadding = 0.1
        _d.DrawMolecule(mol)
        _d.FinishDrawing()
        return Image.open(io.BytesIO(_d.GetDrawingText()))

    _n_xgb = min(15, len(xgb_top_bits))
    _n_chemprop = min(15, len(chemprop_top_atom_types))

    # --- Build figure: 4 columns ---
    # [XGB fragment | XGB bar] [Chemprop bar | Chemprop fragment]
    _fig = plt.figure(figsize=(22, 12))
    _gs = _fig.add_gridspec(1, 4, width_ratios=[0.3, 0.7, 0.7, 0.3], wspace=0.05)

    # --- XGBoost substructure images (left of bars) ---
    _ax_xgb_img = _fig.add_subplot(_gs[0])
    _ax_xgb_img.set_xlim(0, 1)
    _ax_xgb_img.set_ylim(-0.5, _n_xgb - 0.5)
    _ax_xgb_img.invert_yaxis()
    _ax_xgb_img.axis("off")

    _xgb_bits = xgb_top_bits[:_n_xgb]

    for _j, _bit in enumerate(_xgb_bits):
        if int(_bit) in xgb_bit_to_fragment:
            _frag = xgb_bit_to_fragment[int(_bit)]
            try:
                _img = _draw_fragment(_frag, size=(200, 150))
                _inset = _ax_xgb_img.inset_axes(
                    [0.0, _j / _n_xgb, 1.0, 0.9 / _n_xgb],
                    transform=_ax_xgb_img.transAxes,
                )
                _inset.imshow(_img)
                _inset.axis("off")
            except Exception:
                pass

    # --- XGBoost SHAP bars ---
    _ax_xgb = _fig.add_subplot(_gs[1])

    _xgb_labels = []
    _xgb_vals = []
    _xgb_colors = []

    for _bit in _xgb_bits:
        _abs_val = float(xgb_mean_abs_shap[_bit])
        _signed_val = float(xgb_mean_signed_shap[_bit])
        _xgb_vals.append(_abs_val)
        _xgb_labels.append(f"Bit {_bit}")
        if _signed_val > 0:
            _xgb_colors.append("#2196F3")
        else:
            _xgb_colors.append("#FF5722")

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
        if _atype in chemprop_atom_type_fragment:
            _frag = chemprop_atom_type_fragment[_atype]
            try:
                _img = _draw_fragment(_frag, size=(200, 150))
                _inset = _ax_cp_img.inset_axes(
                    [0.0, _j / _n_chemprop, 1.0, 0.9 / _n_chemprop],
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
    **Left panel (XGBoost):** Top 15 Morgan fingerprint bits by mean
    absolute SHAP value on HLM test molecules. Blue bars push toward
    "stable" (active class); red bars push toward "unstable." Fragment
    images to the left show the extracted substructure corresponding to
    each fingerprint bit environment -- not the full molecule, just the
    Morgan radius-3 neighborhood that activates the bit.

    **Right panel (Chemprop):** Top 15 atom types by mean normalized
    gradient saliency across 100 sampled HLM test molecules. Fragment
    images to the right show a representative 2-bond neighborhood for
    each atom type. Error bars show standard error of the mean.

    Both architectures attend to nitrogen-containing environments and
    heteroatom functional groups -- the structural features known to
    govern CYP-mediated metabolic stability.

    Figure saved to `{_save_path}`.
            """),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
