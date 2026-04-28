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
    import json
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import seaborn as sns
    import shap
    import xgboost as xgb
    from loguru import logger
    from rdkit import Chem
    from rdkit.Chem import Draw, rdFingerprintGenerator
    from sklearn.metrics import roc_auc_score

    DATA_DIR = Path("data")
    return (
        Chem,
        DATA_DIR,
        Draw,
        json,
        logger,
        np,
        pl,
        plt,
        rdFingerprintGenerator,
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
    Draw,
    failure_indices,
    failure_molecules,
    mo,
    np,
    plt,
    rdFingerprintGenerator,
    shap_scratch,
    shap_transfer,
):
    # For each failure molecule, show top SHAP features and map to substructures
    _gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
    _outputs = []

    for _rank, _idx in enumerate(failure_indices):
        _row = failure_molecules.row(_rank, named=True)
        _smi = _row["smiles"]
        _mol = Chem.MolFromSmiles(_smi)
        _true = _row["true_label"]
        _label_name = "Permeable" if _true == 1 else "Impermeable"

        # Get bit info for this molecule
        _ao = rdFingerprintGenerator.AdditionalOutput()
        _ao.AllocateBitInfoMap()
        _fp = _gen.GetFingerprint(_mol, additionalOutput=_ao)
        _bit_info = _ao.GetBitInfoMap()

        # Top 10 SHAP bits by absolute value for each model
        _shap_s = shap_scratch[_idx]
        _shap_t = shap_transfer[_idx]
        _top_scratch = np.argsort(np.abs(_shap_s))[-10:][::-1]
        _top_transfer = np.argsort(np.abs(_shap_t))[-10:][::-1]

        # Build comparison figure
        _fig, (_ax1, _ax2, _ax3) = plt.subplots(1, 3, figsize=(18, 4))

        # Molecule image
        _img = Draw.MolToImage(_mol, size=(300, 300))
        _ax1.imshow(_img)
        _ax1.set_title(
            f"#{_rank + 1}: {_label_name}\nP(scratch)={_row['prob_scratch']:.3f}, P(transfer)={_row['prob_transfer']:.3f}"
        )
        _ax1.axis("off")

        # SHAP waterfall-style bar for scratch
        _top_bits_s = _top_scratch[:8]
        _bar_vals_s = [_shap_s[b] for b in _top_bits_s]
        _bar_labels_s = []
        for _b in _top_bits_s:
            if _b in _bit_info:
                _atoms = [
                    f"{_mol.GetAtomWithIdx(a).GetSymbol()}{a}r{r}"
                    for a, r in _bit_info[_b]
                ]
                _bar_labels_s.append(f"bit{_b}\n{','.join(_atoms[:2])}")
            else:
                _bar_labels_s.append(f"bit{_b}\n(off)")
        _colors_s = ["#2196F3" if v > 0 else "#FF5722" for v in _bar_vals_s]
        _ax2.barh(range(len(_top_bits_s)), _bar_vals_s, color=_colors_s)
        _ax2.set_yticks(range(len(_top_bits_s)))
        _ax2.set_yticklabels(_bar_labels_s, fontsize=8)
        _ax2.set_xlabel("SHAP value")
        _ax2.set_title("XGBoost Scratch")
        _ax2.invert_yaxis()

        # SHAP waterfall-style bar for transfer
        _top_bits_t = _top_transfer[:8]
        _bar_vals_t = [_shap_t[b] for b in _top_bits_t]
        _bar_labels_t = []
        for _b in _top_bits_t:
            if _b in _bit_info:
                _atoms = [
                    f"{_mol.GetAtomWithIdx(a).GetSymbol()}{a}r{r}"
                    for a, r in _bit_info[_b]
                ]
                _bar_labels_t.append(f"bit{_b}\n{','.join(_atoms[:2])}")
            else:
                _bar_labels_t.append(f"bit{_b}\n(off)")
        _colors_t = ["#2196F3" if v > 0 else "#FF5722" for v in _bar_vals_t]
        _ax3.barh(range(len(_top_bits_t)), _bar_vals_t, color=_colors_t)
        _ax3.set_yticks(range(len(_top_bits_t)))
        _ax3.set_yticklabels(_bar_labels_t, fontsize=8)
        _ax3.set_xlabel("SHAP value")
        _ax3.set_title("XGBoost RLM-Transfer")
        _ax3.invert_yaxis()

        plt.tight_layout()
        _outputs.append(mo.as_html(_fig))
        plt.close(_fig)

    mo.vstack(
        [
            mo.md("## SHAP Analysis: Per-Molecule Feature Attribution"),
            mo.md("""
    For each failure molecule: the structure (left), top 8 SHAP features
    for XGBoost scratch (center), and top 8 SHAP features for XGBoost
    RLM-transfer (right). Blue bars push prediction toward active (permeable),
    red bars push toward inactive. Bit labels show the fingerprint bit index
    and the atom center + radius of the Morgan environment.
        """),
            *_outputs,
        ]
    )
    return


@app.cell
def _(mo, np, plt, shap_scratch, shap_transfer):
    # Global SHAP comparison: which bits matter most for each model?
    _mean_abs_scratch = np.abs(shap_scratch).mean(axis=0)
    _mean_abs_transfer = np.abs(shap_transfer).mean(axis=0)

    # Top 20 bits by importance in each model
    _top20_scratch = np.argsort(_mean_abs_scratch)[-20:][::-1]
    _top20_transfer = np.argsort(_mean_abs_transfer)[-20:][::-1]

    _fig_global, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(16, 6))

    _ax1.barh(
        range(20), [_mean_abs_scratch[b] for b in _top20_scratch], color="#2196F3"
    )
    _ax1.set_yticks(range(20))
    _ax1.set_yticklabels([f"bit {b}" for b in _top20_scratch], fontsize=9)
    _ax1.set_xlabel("Mean |SHAP|")
    _ax1.set_title("XGBoost Scratch: Top 20 Features")
    _ax1.invert_yaxis()

    _ax2.barh(
        range(20), [_mean_abs_transfer[b] for b in _top20_transfer], color="#FF9800"
    )
    _ax2.set_yticks(range(20))
    _ax2.set_yticklabels([f"bit {b}" for b in _top20_transfer], fontsize=9)
    _ax2.set_xlabel("Mean |SHAP|")
    _ax2.set_title("XGBoost RLM-Transfer: Top 20 Features")
    _ax2.invert_yaxis()

    plt.tight_layout()

    # Overlap in top features
    _overlap = len(set(_top20_scratch) & set(_top20_transfer))

    mo.vstack(
        [
            mo.md("## Global Feature Importance Comparison"),
            mo.as_html(_fig_global),
            mo.md(f"""
    Top 20 features overlap: **{_overlap}/20** bits shared between scratch
    and transfer models. The transfer model inherits feature importances from
    the RLM pre-training, which may prioritize bits relevant to metabolic
    stability rather than membrane permeability.
        """),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
