import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md("""
    # 08 — Failure Case Analysis (XGBoost SHAP)

    Identify specific molecules where the XGBoost RLM-transfer model on
    PAMPA was most wrong, and use SHAP to explain which fingerprint bits
    drove the incorrect predictions. Compare against XGBoost scratch and
    the D-MPNN models to understand why transfer helped or hurt.
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
    import seaborn as sns
    import shap
    import xgboost as xgb
    from loguru import logger
    from PIL import Image
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem.Draw import rdMolDraw2D
    from sklearn.metrics import roc_auc_score

    DATA_DIR = Path("data")
    return (
        Chem,
        DATA_DIR,
        Image,
        defaultdict,
        io,
        json,
        logger,
        np,
        pl,
        plt,
        rdFingerprintGenerator,
        rdMolDraw2D,
        roc_auc_score,
        shap,
        xgb,
    )


@app.cell
def _(DATA_DIR, json, logger, np):
    # Load splits and fingerprints
    with open(DATA_DIR / "split_config.json") as _f:
        split_config = json.load(_f)

    _fp_data = np.load(DATA_DIR / "morgan_fps_2048_r3.npz", allow_pickle=True)
    global_fps = _fp_data["fp_matrix"]
    global_smiles = list(_fp_data["smiles"])

    # Load PAMPA splits
    _pampa_split = np.load(DATA_DIR / "pampa_splits.npz", allow_pickle=True)
    pampa_smiles = list(_pampa_split["smiles"])
    pampa_labels = _pampa_split["labels"]
    pampa_folds = _pampa_split["folds"]
    pampa_fp_indices = _pampa_split["fp_indices"]
    pampa_X = global_fps[pampa_fp_indices]

    # Load RLM splits for pretrained model
    _rlm_split = np.load(DATA_DIR / "rlm_splits.npz", allow_pickle=True)
    rlm_labels = _rlm_split["labels"]
    rlm_fp_indices = _rlm_split["fp_indices"]
    rlm_X = global_fps[rlm_fp_indices]

    logger.info(f"PAMPA: {pampa_X.shape[0]} molecules, RLM: {rlm_X.shape[0]} molecules")
    return pampa_X, pampa_folds, pampa_labels, pampa_smiles, rlm_X, rlm_labels


@app.cell
def _(
    logger,
    pampa_X,
    pampa_folds,
    pampa_labels,
    rlm_X,
    rlm_labels,
    roc_auc_score,
    xgb,
):
    # Use replicate 0, fold 0
    REP = 0
    FOLD = 0

    _fold_assign = pampa_folds[REP]
    test_mask = _fold_assign == FOLD
    train_mask = ~test_mask

    X_train = pampa_X[train_mask]
    X_test = pampa_X[test_mask]
    y_train = pampa_labels[train_mask]
    y_test = pampa_labels[test_mask]

    logger.info(
        f"Fold: rep={REP} fold={FOLD}, train={X_train.shape[0]}, test={X_test.shape[0]}"
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

    # Train XGBoost scratch on PAMPA
    _dtrain = xgb.DMatrix(X_train, label=y_train)
    _dval = xgb.DMatrix(X_test, label=y_test)
    model_scratch = xgb.train(
        XGB_PARAMS,
        _dtrain,
        200,
        evals=[(_dval, "val")],
        early_stopping_rounds=20,
        verbose_eval=False,
    )

    # Pre-train on RLM, then transfer to PAMPA
    _dtrain_rlm = xgb.DMatrix(rlm_X, label=rlm_labels)
    model_rlm = xgb.train(XGB_PARAMS, _dtrain_rlm, 200, verbose_eval=False)
    model_transfer = xgb.train(
        XGB_PARAMS,
        _dtrain,
        200,
        evals=[(_dval, "val")],
        early_stopping_rounds=20,
        verbose_eval=False,
        xgb_model=model_rlm,
    )

    # Predictions
    _dtest = xgb.DMatrix(X_test)
    y_prob_scratch = model_scratch.predict(_dtest)
    y_prob_transfer = model_transfer.predict(_dtest)

    logger.info(f"Scratch AUC: {roc_auc_score(y_test, y_prob_scratch):.3f}")
    logger.info(f"Transfer AUC: {roc_auc_score(y_test, y_prob_transfer):.3f}")
    return (
        X_test,
        model_scratch,
        model_transfer,
        test_mask,
        y_prob_scratch,
        y_prob_transfer,
        y_test,
    )


@app.cell
def _(
    logger,
    mo,
    np,
    pampa_smiles,
    pl,
    test_mask,
    y_prob_scratch,
    y_prob_transfer,
    y_test,
):
    # Find molecules where transfer was most wrong but scratch was right
    _test_smiles = [pampa_smiles[i] for i in range(len(pampa_smiles)) if test_mask[i]]

    # "Wrong" = large absolute error in predicted probability vs true label
    _err_scratch = np.abs(y_prob_scratch - y_test)
    _err_transfer = np.abs(y_prob_transfer - y_test)
    _delta_err = _err_transfer - _err_scratch  # positive = transfer worse

    _failure_df = pl.DataFrame(
        {
            "smiles": _test_smiles,
            "true_label": y_test.tolist(),
            "prob_scratch": y_prob_scratch.tolist(),
            "prob_transfer": y_prob_transfer.tolist(),
            "err_scratch": _err_scratch.tolist(),
            "err_transfer": _err_transfer.tolist(),
            "delta_err": _delta_err.tolist(),
        }
    ).sort("delta_err", descending=True)

    # Top 5 where transfer was most catastrophically wrong vs scratch
    failure_molecules = _failure_df.head(5)
    failure_indices = []
    for _smi in failure_molecules.get_column("smiles").to_list():
        _idx = _test_smiles.index(_smi)
        failure_indices.append(_idx)

    logger.info(f"Top 5 failure molecules (transfer worst vs scratch):")

    mo.vstack(
        [
            mo.md("## Top 5 Failure Molecules (Transfer Most Wrong vs Scratch)"),
            mo.ui.table(failure_molecules),
            mo.md("""
    These are molecules where the XGBoost RLM-transfer model had the largest
    error relative to the scratch model. `delta_err` = transfer error minus
    scratch error. Positive means transfer was worse.
        """),
        ]
    )
    return failure_indices, failure_molecules


@app.cell
def _(X_test, logger, model_scratch, model_transfer, shap):
    # SHAP analysis
    explainer_scratch = shap.TreeExplainer(model_scratch)
    explainer_transfer = shap.TreeExplainer(model_transfer)

    shap_scratch = explainer_scratch.shap_values(X_test)
    shap_transfer = explainer_transfer.shap_values(X_test)

    logger.info(f"SHAP values computed: {shap_scratch.shape}")
    return shap_scratch, shap_transfer


@app.cell
def _(
    Chem,
    Image,
    failure_indices,
    failure_molecules,
    io,
    mo,
    np,
    plt,
    rdFingerprintGenerator,
    rdMolDraw2D,
    shap_scratch,
    shap_transfer,
):

    def get_shap_highlighted_image(
        mol,
        bit_info,
        shap_values,
        top_n=5,
        size=(450, 350),
    ):
        """Draw molecule with top SHAP bits highlighted on the structure.

        Blue = pushes toward active (positive SHAP), Red = pushes toward inactive.
        Opacity scales with |SHAP| magnitude.
        """
        top_bits = np.argsort(np.abs(shap_values))[-top_n:][::-1]
        max_shap = max(np.abs(shap_values[top_bits]).max(), 1e-8)

        all_atoms = []
        all_bonds = []
        atom_colors = {}
        bond_colors = {}

        for bit in top_bits:
            if bit not in bit_info:
                continue
            sv = shap_values[bit]
            intensity = float(min(float(abs(sv)) / float(max_shap), 1.0) * 0.6 + 0.2)
            if sv > 0:
                color = (0.13, 0.59, 0.95, intensity)  # blue
            else:
                color = (1.0, 0.34, 0.13, intensity)  # red

            for center_atom, radius in bit_info[bit]:
                if radius > 0:
                    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center_atom)
                    if env:
                        atom_map = {}
                        Chem.PathToSubmol(mol, env, atomMap=atom_map)
                        for a in atom_map.keys():
                            if a not in atom_colors:
                                all_atoms.append(a)
                                atom_colors[a] = color
                        for b in env:
                            if b not in bond_colors:
                                all_bonds.append(b)
                                bond_colors[b] = color
                else:
                    if center_atom not in atom_colors:
                        all_atoms.append(center_atom)
                        atom_colors[center_atom] = color

        drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        opts = drawer.drawOptions()
        opts.clearBackground = True
        drawer.DrawMolecule(
            mol,
            highlightAtoms=list(set(all_atoms)),
            highlightAtomColors=atom_colors,
            highlightBonds=list(set(all_bonds)),
            highlightBondColors=bond_colors,
        )
        drawer.FinishDrawing()
        png_bytes = drawer.GetDrawingText()
        return Image.open(io.BytesIO(png_bytes))

    # For each failure molecule, show highlighted structures for both models
    _gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
    _outputs = []

    for _rank, _idx in enumerate(failure_indices):
        _row = failure_molecules.row(_rank, named=True)
        _smi = _row["smiles"]
        _mol = Chem.MolFromSmiles(_smi)
        if _mol is None:
            continue
        _true = _row["true_label"]
        _label_name = "Permeable" if _true == 1 else "Impermeable"

        # Get bit info for this molecule
        _ao = rdFingerprintGenerator.AdditionalOutput()
        _ao.AllocateBitInfoMap()
        _fp = _gen.GetFingerprint(_mol, additionalOutput=_ao)
        _bit_info = _ao.GetBitInfoMap()

        _shap_s = shap_scratch[_idx]
        _shap_t = shap_transfer[_idx]

        # Generate highlighted images
        _img_scratch = get_shap_highlighted_image(_mol, _bit_info, _shap_s, top_n=5)
        _img_transfer = get_shap_highlighted_image(_mol, _bit_info, _shap_t, top_n=5)

        # Build figure: two highlighted molecules side by side
        _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Scratch highlighted molecule
        _ax1.imshow(_img_scratch)
        _ax1.set_title(
            f"XGBoost Scratch\nP(active)={_row['prob_scratch']:.3f}", fontsize=11
        )
        _ax1.axis("off")

        # Transfer highlighted molecule
        _ax2.imshow(_img_transfer)
        _ax2.set_title(
            f"XGBoost RLM-Transfer\nP(active)={_row['prob_transfer']:.3f}", fontsize=11
        )
        _ax2.axis("off")

        _fig.suptitle(
            f"Molecule #{_rank + 1}: True={_label_name} | SMILES: {_smi[:60]}{'...' if len(_smi) > 60 else ''}",
            fontsize=12,
        )
        plt.tight_layout()
        _outputs.append(mo.as_html(_fig))
        plt.close(_fig)

    mo.vstack(
        [
            mo.md("## SHAP Analysis: Per-Molecule Feature Attribution"),
            mo.md("""
    For each failure molecule: highlighted structures show the top 5 SHAP
    features mapped onto the molecular structure. **Blue** regions push the
    prediction toward active (permeable), **red** regions push toward inactive.
    Opacity scales with SHAP magnitude. The left structure shows what the
    scratch model focuses on; the right shows the transfer model's focus.
            """),
            *_outputs,
        ]
    )
    return


@app.cell
def _(
    Chem,
    Image,
    defaultdict,
    io,
    mo,
    np,
    pampa_smiles,
    plt,
    rdFingerprintGenerator,
    rdMolDraw2D,
    shap_scratch,
    shap_transfer,
    test_mask,
):
    _gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
    _test_smiles = [pampa_smiles[i] for i in range(len(pampa_smiles)) if test_mask[i]]

    # Map bits to SMARTS and collect example molecules
    _bit_to_smarts = defaultdict(set)
    _bit_to_example = {}  # bit -> (mol, center_atom, radius)

    for _smi in _test_smiles:
        _mol = Chem.MolFromSmiles(_smi)
        if _mol is None:
            continue
        _ao = rdFingerprintGenerator.AdditionalOutput()
        _ao.AllocateBitInfoMap()
        _fp = _gen.GetFingerprint(_mol, additionalOutput=_ao)
        _bit_info = _ao.GetBitInfoMap()

        for _bit, _envs in _bit_info.items():
            for _center, _rad in _envs:
                try:
                    if _rad > 0:
                        _env = Chem.FindAtomEnvironmentOfRadiusN(_mol, _rad, _center)
                        if _env:
                            _amap = {}
                            _submol = Chem.PathToSubmol(_mol, _env, atomMap=_amap)
                            _smarts = Chem.MolToSmarts(_submol)
                            _bit_to_smarts[_bit].add(_smarts)
                    else:
                        _atom = _mol.GetAtomWithIdx(_center)
                        _bit_to_smarts[_bit].add(f"[#{_atom.GetAtomicNum()}]")
                    if _bit not in _bit_to_example:
                        _bit_to_example[_bit] = (_mol, _center, _rad)
                except Exception:
                    continue

    # Mean SHAP per bit
    _mean_s = shap_scratch.mean(axis=0)
    _mean_t = shap_transfer.mean(axis=0)

    # Find important bits (top 50 by |SHAP| in either model)
    _top50_s = set(np.argsort(np.abs(_mean_s))[-50:])
    _top50_t = set(np.argsort(np.abs(_mean_t))[-50:])
    _important = _top50_s | _top50_t

    _agree = []
    _disagree = []
    for _bit in _important:
        _sv_s = float(_mean_s[_bit])
        _sv_t = float(_mean_t[_bit])
        if abs(_sv_s) < 0.001 or abs(_sv_t) < 0.001:
            continue
        if _sv_s * _sv_t > 0:
            _agree.append((_bit, _sv_s, _sv_t))
        else:
            _disagree.append((_bit, _sv_s, _sv_t))

    _agree.sort(key=lambda x: abs(x[1]) + abs(x[2]), reverse=True)
    _disagree.sort(key=lambda x: abs(x[1]) + abs(x[2]), reverse=True)

    def _draw_bit_on_example(bit, mol, center_atom, radius, size=(250, 200)):
        """Draw an example molecule with a specific bit's environment highlighted."""
        _atoms = []
        _bonds = []
        if radius > 0:
            _env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center_atom)
            if _env:
                _amap = {}
                Chem.PathToSubmol(mol, _env, atomMap=_amap)
                _atoms = list(_amap.keys())
                _bonds = list(_env)
        else:
            _atoms = [center_atom]

        _color = (0.2, 0.6, 1.0, 0.5)
        _acols = {a: _color for a in _atoms}
        _bcols = {b: _color for b in _bonds}

        _d = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        _d.DrawMolecule(
            mol,
            highlightAtoms=_atoms,
            highlightAtomColors=_acols,
            highlightBonds=_bonds,
            highlightBondColors=_bcols,
        )
        _d.FinishDrawing()
        return Image.open(io.BytesIO(_d.GetDrawingText()))

    # Draw top agree and disagree substructures
    _n_show = min(6, max(len(_agree), len(_disagree)))

    _fig, _axes = plt.subplots(2, _n_show, figsize=(_n_show * 4, 8))
    if _n_show == 1:
        _axes = _axes.reshape(2, 1)

    # Row 0: agree
    for _j in range(min(_n_show, len(_agree))):
        _bit, _sv_s, _sv_t = _agree[_j]
        _direction = "permeable" if _sv_s > 0 else "impermeable"
        if _bit in _bit_to_example:
            _mol, _center, _rad = _bit_to_example[_bit]
            _img = _draw_bit_on_example(_bit, _mol, _center, _rad)
            _axes[0][_j].imshow(_img)
        _axes[0][_j].set_title(
            f"AGREE: -> {_direction}\nS={_sv_s:+.3f} T={_sv_t:+.3f}",
            fontsize=9,
            color="#4CAF50",
        )
        _axes[0][_j].axis("off")

    # Row 1: disagree
    for _j in range(min(_n_show, len(_disagree))):
        _bit, _sv_s, _sv_t = _disagree[_j]
        _s_dir = "perm" if _sv_s > 0 else "imperm"
        _t_dir = "perm" if _sv_t > 0 else "imperm"
        if _bit in _bit_to_example:
            _mol, _center, _rad = _bit_to_example[_bit]
            _img = _draw_bit_on_example(_bit, _mol, _center, _rad)
            _axes[1][_j].imshow(_img)
        _axes[1][_j].set_title(
            f"DISAGREE\nS={_sv_s:+.3f}({_s_dir}) T={_sv_t:+.3f}({_t_dir})",
            fontsize=9,
            color="#FF5722",
        )
        _axes[1][_j].axis("off")

    # Hide unused axes
    for _row in range(2):
        _n_items = len(_agree) if _row == 0 else len(_disagree)
        for _j in range(min(_n_show, _n_items), _n_show):
            _axes[_row][_j].axis("off")

    _fig.suptitle(
        "Substructures: Models Agree (top) vs Disagree (bottom)",
        fontsize=13,
    )
    plt.tight_layout()

    mo.vstack(
        [
            mo.md("## Substructure Agreement and Disagreement"),
            mo.as_html(_fig),
            mo.md(f"""
    **Agree** ({len(_agree)} substructures, green): Both models assign the
    same direction (permeable or impermeable) to these chemical environments.
    These are structural features with a consistent effect on PAMPA
    permeability regardless of whether the model was pre-trained on RLM.

    **Disagree** ({len(_disagree)} substructures, red): The scratch model
    says one direction, the transfer model says the opposite. These are
    substructures where the RLM pre-training introduced a bias -- the
    transfer model learned an association from metabolic stability that
    conflicts with what the scratch model learned from permeability data.
    S = scratch mean SHAP, T = transfer mean SHAP. "perm" = pushes toward
    permeable, "imperm" = pushes toward impermeable.
            """),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
