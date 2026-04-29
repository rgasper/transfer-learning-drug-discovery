import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md("""
    # 16 — Cross-Fold Aggregate Feature Importance

    Computes SHAP (XGBoost) and gradient saliency (Chemprop) across all
    25 CV folds (5 reps x 5 folds) for both HLM and PAMPA, scratch and
    RLM-transfer variants. Produces aggregate figures with cross-fold error
    bars and rank-stability annotations, replacing the single-fold snapshots
    in notebooks 10 and 12.

    **8 model variants x 25 folds = 200 training runs.**
    XGBoost is fast (~seconds each). Chemprop is the bottleneck (~2-3 min
    each on GPU, ~50-75 min per condition, ~3-5 hours total).

    Results are cached to disk per fold so interrupted runs can resume.
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
    import shap
    import torch
    import xgboost as xgb
    from lightning import pytorch as lightning_pl
    from loguru import logger
    from PIL import Image
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem.Draw import rdMolDraw2D
    from sklearn.metrics import average_precision_score

    from chemprop import data as chemprop_data
    from chemprop import featurizers, models, nn

    DATA_DIR = Path("data")
    FIGURES_DIR = Path("docs/figures")
    CACHE_DIR = Path("data/aggregate_importance_cache")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    N_REPS = 5
    N_FOLDS = 5
    TOP_K = 10
    return (
        CACHE_DIR,
        Chem,
        DATA_DIR,
        FIGURES_DIR,
        Image,
        N_FOLDS,
        N_REPS,
        TOP_K,
        average_precision_score,
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
        shap,
        torch,
        xgb,
    )


@app.cell
def _(DATA_DIR, logger, np):
    """Load all split data and fingerprints."""

    _fp_data = np.load(DATA_DIR / "morgan_fps_2048_r3.npz", allow_pickle=True)
    global_fps = _fp_data["fp_matrix"]

    _hlm_split = np.load(DATA_DIR / "hlm_splits.npz", allow_pickle=True)
    hlm_smiles = list(_hlm_split["smiles"])
    hlm_labels = _hlm_split["labels"]
    hlm_folds = _hlm_split["folds"]
    hlm_fp_indices = _hlm_split["fp_indices"]
    hlm_X = global_fps[hlm_fp_indices]

    _pampa_split = np.load(DATA_DIR / "pampa_splits.npz", allow_pickle=True)
    pampa_smiles = list(_pampa_split["smiles"])
    pampa_labels = _pampa_split["labels"]
    pampa_folds = _pampa_split["folds"]
    pampa_fp_indices = _pampa_split["fp_indices"]
    pampa_X = global_fps[pampa_fp_indices]

    _rlm_split = np.load(DATA_DIR / "rlm_splits.npz", allow_pickle=True)
    rlm_smiles = list(_rlm_split["smiles"])
    rlm_labels = _rlm_split["labels"]
    rlm_fp_indices = _rlm_split["fp_indices"]
    rlm_X = global_fps[rlm_fp_indices]

    logger.info(
        f"HLM: {hlm_X.shape[0]}, PAMPA: {pampa_X.shape[0]}, RLM: {rlm_X.shape[0]}"
    )
    return (
        hlm_X,
        hlm_folds,
        hlm_labels,
        hlm_smiles,
        pampa_X,
        pampa_folds,
        pampa_labels,
        pampa_smiles,
        rlm_X,
        rlm_labels,
        rlm_smiles,
    )


@app.cell
def _(
    CACHE_DIR,
    N_FOLDS,
    N_REPS,
    average_precision_score,
    hlm_X,
    hlm_folds,
    hlm_labels,
    logger,
    np,
    pampa_X,
    pampa_folds,
    pampa_labels,
    rlm_X,
    rlm_labels,
    shap,
    xgb,
):
    """Compute per-fold XGBoost SHAP for all 4 conditions (cached)."""

    XGB_PARAMS = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "nthread": 1,
        "verbosity": 0,
    }

    def _train_xgb_scratch(X, labels, train_mask, test_mask, seed):
        """Train XGBoost from scratch and return SHAP values on test set."""
        _dtrain = xgb.DMatrix(X[train_mask], label=labels[train_mask])
        _dval = xgb.DMatrix(X[test_mask], label=labels[test_mask])
        _params = {**XGB_PARAMS, "random_state": seed}
        _model = xgb.train(
            _params,
            _dtrain,
            200,
            evals=[(_dval, "val")],
            early_stopping_rounds=20,
            verbose_eval=False,
        )
        _explainer = shap.TreeExplainer(_model)
        _shap_vals = _explainer.shap_values(X[test_mask])
        _y_prob = _model.predict(_dval)
        _auc_pr = average_precision_score(labels[test_mask], _y_prob)
        return _shap_vals, _auc_pr

    def _train_xgb_transfer(X, labels, train_mask, test_mask, seed):
        """Pre-train on RLM, continue-boost on target, return SHAP on test."""
        _dtrain_rlm = xgb.DMatrix(rlm_X, label=rlm_labels)
        _params = {**XGB_PARAMS, "random_state": seed}
        _rlm_model = xgb.train(
            _params,
            _dtrain_rlm,
            200,
            verbose_eval=False,
        )
        _dtrain = xgb.DMatrix(X[train_mask], label=labels[train_mask])
        _dval = xgb.DMatrix(X[test_mask], label=labels[test_mask])
        _model = xgb.train(
            _params,
            _dtrain,
            200,
            evals=[(_dval, "val")],
            early_stopping_rounds=20,
            verbose_eval=False,
            xgb_model=_rlm_model,
        )
        _explainer = shap.TreeExplainer(_model)
        _shap_vals = _explainer.shap_values(X[test_mask])
        _y_prob = _model.predict(_dval)
        _auc_pr = average_precision_score(labels[test_mask], _y_prob)
        return _shap_vals, _auc_pr

    def _run_xgb_condition(name, X, labels, folds, train_fn):
        """Run one XGBoost condition across all folds, caching results."""
        _cache_path = CACHE_DIR / f"xgb_{name}.npz"
        if _cache_path.exists():
            _cached = np.load(_cache_path, allow_pickle=True)
            logger.info(f"XGB {name}: loaded from cache ({_cache_path})")
            return _cached["mean_abs_shap_per_fold"], _cached[
                "mean_signed_shap_per_fold"
            ]

        _n_features = X.shape[1]
        _mean_abs_per_fold = np.zeros((N_REPS * N_FOLDS, _n_features))
        _mean_signed_per_fold = np.zeros((N_REPS * N_FOLDS, _n_features))

        for _rep in range(N_REPS):
            _fold_assign = folds[_rep]
            for _fold in range(N_FOLDS):
                _idx = _rep * N_FOLDS + _fold
                _test_mask = _fold_assign == _fold
                _train_mask = ~_test_mask
                _seed = _rep * 1000 + _fold

                _shap_vals, _auc_pr = train_fn(
                    X, labels, _train_mask, _test_mask, _seed
                )
                _mean_abs_per_fold[_idx] = np.abs(_shap_vals).mean(axis=0)
                _mean_signed_per_fold[_idx] = _shap_vals.mean(axis=0)

                logger.info(f"XGB {name} rep={_rep} fold={_fold} AUC-PR={_auc_pr:.3f}")

        np.savez(
            _cache_path,
            mean_abs_shap_per_fold=_mean_abs_per_fold,
            mean_signed_shap_per_fold=_mean_signed_per_fold,
        )
        logger.info(f"XGB {name}: saved to {_cache_path}")
        return _mean_abs_per_fold, _mean_signed_per_fold

    # Run all 4 XGBoost conditions
    xgb_hlm_scratch_abs, xgb_hlm_scratch_signed = _run_xgb_condition(
        "hlm_scratch",
        hlm_X,
        hlm_labels,
        hlm_folds,
        _train_xgb_scratch,
    )
    xgb_hlm_transfer_abs, xgb_hlm_transfer_signed = _run_xgb_condition(
        "hlm_transfer",
        hlm_X,
        hlm_labels,
        hlm_folds,
        _train_xgb_transfer,
    )
    xgb_pampa_scratch_abs, xgb_pampa_scratch_signed = _run_xgb_condition(
        "pampa_scratch",
        pampa_X,
        pampa_labels,
        pampa_folds,
        _train_xgb_scratch,
    )
    xgb_pampa_transfer_abs, xgb_pampa_transfer_signed = _run_xgb_condition(
        "pampa_transfer",
        pampa_X,
        pampa_labels,
        pampa_folds,
        _train_xgb_transfer,
    )

    logger.info("All XGBoost SHAP conditions complete.")
    return (
        xgb_hlm_scratch_abs,
        xgb_hlm_scratch_signed,
        xgb_hlm_transfer_abs,
        xgb_hlm_transfer_signed,
        xgb_pampa_scratch_abs,
        xgb_pampa_scratch_signed,
        xgb_pampa_transfer_abs,
        xgb_pampa_transfer_signed,
    )


@app.cell
def _(
    CACHE_DIR,
    Chem,
    N_FOLDS,
    N_REPS,
    average_precision_score,
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
    pampa_folds,
    pampa_labels,
    pampa_smiles,
    rlm_labels,
    rlm_smiles,
    torch,
):
    """Compute per-fold Chemprop saliency for all 4 conditions (cached)."""

    _featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    def _build_loaders(smiles_list, labels_arr, train_mask, test_mask, seed):
        """Build train/val/test dataloaders from masks."""
        _train_smi = [smiles_list[i] for i in range(len(smiles_list)) if train_mask[i]]
        _train_y = labels_arr[train_mask].reshape(-1, 1).astype(float)
        _test_smi = [smiles_list[i] for i in range(len(smiles_list)) if test_mask[i]]
        _test_y = labels_arr[test_mask].reshape(-1, 1).astype(float)

        _n = len(_train_smi)
        _n_val = max(1, int(_n * 0.1))
        _rng = np.random.default_rng(seed)
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
        return _train_loader, _val_loader, _test_loader, _test_smi, _test_y

    def _compute_atom_saliency(model, test_smiles):
        """Compute per-atom-type saliency across test molecules.

        Args:
            model: Trained Chemprop MPNN.
            test_smiles: List of SMILES strings.

        Returns:
            Dict mapping atom-type keys to list of normalized saliency values.
        """
        _saliency = defaultdict(list)
        model.eval()

        for _smi in test_smiles:
            _mol = Chem.MolFromSmiles(_smi)
            if _mol is None:
                continue
            _dp = chemprop_data.MoleculeDatapoint.from_smi(_smi, [0.0])
            _dset = chemprop_data.MoleculeDataset([_dp], _featurizer)
            _loader = chemprop_data.build_dataloader(
                _dset, num_workers=0, batch_size=1, shuffle=False
            )
            _batch = next(iter(_loader))
            _bmg = _batch[0]
            _bmg.V = _bmg.V.clone().requires_grad_(True)
            try:
                _pred = model(_bmg)
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
                _key = (
                    f"{_atom.GetSymbol()}"
                    f"{'(arom)' if _atom.GetIsAromatic() else ''}"
                    f" deg{_atom.GetDegree()}"
                )
                _saliency[_key].append(float(_sal[_a] / _max_sal))

        return _saliency

    def _saliency_to_means(saliency_dict):
        """Convert per-instance saliency lists to mean per atom type.

        Args:
            saliency_dict: Dict mapping atom-type keys to lists of floats.

        Returns:
            Dict mapping atom-type keys to mean saliency floats.
        """
        return {k: float(np.mean(v)) for k, v in saliency_dict.items()}

    def _train_chemprop_scratch(smiles_list, labels_arr, train_mask, test_mask, seed):
        """Train Chemprop from scratch, return atom-type mean saliency dict."""
        lightning_pl.seed_everything(seed, workers=True)
        _train_loader, _val_loader, _test_loader, _test_smi, _test_y = _build_loaders(
            smiles_list, labels_arr, train_mask, test_mask, seed
        )
        _mp = nn.BondMessagePassing()
        _agg = nn.MeanAggregation()
        _ffn = nn.BinaryClassificationFFN(input_dim=_mp.output_dim)
        _model = models.MPNN(_mp, _agg, _ffn, batch_norm=False)

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
        _auc_pr = average_precision_score(labels_arr[test_mask], _y_prob)

        _saliency = _compute_atom_saliency(_model, _test_smi)
        return _saliency_to_means(_saliency), _auc_pr

    def _train_chemprop_transfer(smiles_list, labels_arr, train_mask, test_mask, seed):
        """Pre-train on RLM, transfer to target, return atom-type saliency."""
        lightning_pl.seed_everything(seed, workers=True)

        # Pre-train on RLM
        _rlm_y = rlm_labels.reshape(-1, 1).astype(float)
        _n_rlm = len(rlm_smiles)
        _n_rlm_val = max(1, int(_n_rlm * 0.1))
        _rng = np.random.default_rng(seed)
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

        # Transfer: keep encoder, new FFN head
        _train_loader, _val_loader, _test_loader, _test_smi, _test_y = _build_loaders(
            smiles_list, labels_arr, train_mask, test_mask, seed
        )
        _new_ffn = nn.BinaryClassificationFFN(
            input_dim=_rlm_model.message_passing.output_dim
        )
        _transfer_model = models.MPNN(
            _rlm_model.message_passing, _rlm_model.agg, _new_ffn, batch_norm=False
        )

        lightning_pl.seed_everything(seed, workers=True)
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
        _auc_pr = average_precision_score(labels_arr[test_mask], _y_prob)

        _saliency = _compute_atom_saliency(_transfer_model, _test_smi)
        return _saliency_to_means(_saliency), _auc_pr

    def _run_chemprop_condition(name, smiles_list, labels_arr, folds, train_fn):
        """Run one Chemprop condition across all 25 folds, caching results.

        Args:
            name: Condition name for caching (e.g. 'hlm_scratch').
            smiles_list: List of SMILES strings for the target endpoint.
            labels_arr: Binary labels array for the target endpoint.
            folds: Fold assignment array, shape (n_reps, n_molecules).
            train_fn: Training function returning (saliency_means_dict, auc_pr).

        Returns:
            List of 25 dicts, each mapping atom-type keys to mean saliency.
        """
        _cache_path = CACHE_DIR / f"chemprop_{name}.npz"
        if _cache_path.exists():
            _cached = np.load(_cache_path, allow_pickle=True)
            _results = list(_cached["fold_saliency_means"])
            logger.info(f"Chemprop {name}: loaded {len(_results)} folds from cache")
            return _results

        _results = []
        for _rep in range(N_REPS):
            _fold_assign = folds[_rep]
            for _fold in range(N_FOLDS):
                _test_mask = _fold_assign == _fold
                _train_mask = ~_test_mask
                _seed = _rep * 1000 + _fold

                _means, _auc_pr = train_fn(
                    smiles_list,
                    labels_arr,
                    _train_mask,
                    _test_mask,
                    _seed,
                )
                _results.append(_means)
                logger.info(
                    f"Chemprop {name} rep={_rep} fold={_fold} "
                    f"AUC-PR={_auc_pr:.3f} n_atom_types={len(_means)}"
                )

        np.savez(_cache_path, fold_saliency_means=np.array(_results, dtype=object))
        logger.info(f"Chemprop {name}: saved to {_cache_path}")
        return _results

    # Run all 4 Chemprop conditions
    cp_hlm_scratch_folds = _run_chemprop_condition(
        "hlm_scratch",
        hlm_smiles,
        hlm_labels,
        hlm_folds,
        _train_chemprop_scratch,
    )
    cp_hlm_transfer_folds = _run_chemprop_condition(
        "hlm_transfer",
        hlm_smiles,
        hlm_labels,
        hlm_folds,
        _train_chemprop_transfer,
    )
    cp_pampa_scratch_folds = _run_chemprop_condition(
        "pampa_scratch",
        pampa_smiles,
        pampa_labels,
        pampa_folds,
        _train_chemprop_scratch,
    )
    cp_pampa_transfer_folds = _run_chemprop_condition(
        "pampa_transfer",
        pampa_smiles,
        pampa_labels,
        pampa_folds,
        _train_chemprop_transfer,
    )

    logger.info("All Chemprop saliency conditions complete.")
    return (
        cp_hlm_scratch_folds,
        cp_hlm_transfer_folds,
        cp_pampa_scratch_folds,
        cp_pampa_transfer_folds,
    )


# ---------------------------------------------------------------------------
# Exemplar molecules for substructure images
# ---------------------------------------------------------------------------


@app.cell
def _(Chem, hlm_smiles, logger, pampa_smiles, rdFingerprintGenerator):
    """Build bit-to-mol-info dicts and atom-type-to-mol dicts for drawing.

    Scans all molecules in each dataset to find a representative molecule
    for each Morgan fingerprint bit (XGBoost images) and each atom type
    (Chemprop images). These exemplars are used purely for illustration --
    the aggregate importance values come from the cross-fold computation.
    """
    _gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)

    def _build_bit_to_mol_info(smiles_list):
        """For each Morgan bit, store (mol, center_atom, radius) from the
        first molecule that activates it.

        Args:
            smiles_list: List of SMILES strings.

        Returns:
            Dict mapping bit index to (RDKit Mol, center_atom_idx, radius).
        """
        _bit_to_mol = {}
        for _smi in smiles_list:
            _mol = Chem.MolFromSmiles(_smi)
            if _mol is None:
                continue
            _ao = rdFingerprintGenerator.AdditionalOutput()
            _ao.AllocateBitInfoMap()
            _gen.GetFingerprint(_mol, additionalOutput=_ao)
            _bit_info = _ao.GetBitInfoMap()
            for _bit, _envs in _bit_info.items():
                if _bit not in _bit_to_mol:
                    _center, _rad = _envs[0]
                    _bit_to_mol[_bit] = (_mol, _center, _rad)
        return _bit_to_mol

    def _build_atom_type_to_mol(smiles_list):
        """For each atom type key, store (mol, atom_idx) from the first
        molecule that contains it.

        Args:
            smiles_list: List of SMILES strings.

        Returns:
            Dict mapping atom-type key string to (RDKit Mol, atom_idx).
        """
        _atype_to_mol = {}
        for _smi in smiles_list:
            _mol = Chem.MolFromSmiles(_smi)
            if _mol is None:
                continue
            for _a in range(_mol.GetNumAtoms()):
                _atom = _mol.GetAtomWithIdx(_a)
                _key = (
                    f"{_atom.GetSymbol()}"
                    f"{'(arom)' if _atom.GetIsAromatic() else ''}"
                    f" deg{_atom.GetDegree()}"
                )
                if _key not in _atype_to_mol:
                    _atype_to_mol[_key] = (_mol, _a)
        return _atype_to_mol

    hlm_bit_to_mol = _build_bit_to_mol_info(hlm_smiles)
    pampa_bit_to_mol = _build_bit_to_mol_info(pampa_smiles)
    hlm_atype_to_mol = _build_atom_type_to_mol(hlm_smiles)
    pampa_atype_to_mol = _build_atom_type_to_mol(pampa_smiles)

    logger.info(
        f"Exemplars: HLM bits={len(hlm_bit_to_mol)}, "
        f"PAMPA bits={len(pampa_bit_to_mol)}, "
        f"HLM atypes={len(hlm_atype_to_mol)}, "
        f"PAMPA atypes={len(pampa_atype_to_mol)}"
    )
    return (
        hlm_atype_to_mol,
        hlm_bit_to_mol,
        pampa_atype_to_mol,
        pampa_bit_to_mol,
    )


@app.cell
def _(N_FOLDS, N_REPS, TOP_K, np):
    """Define aggregation utilities used by the plotting cells."""

    def aggregate_xgb_shap(mean_abs_per_fold, mean_signed_per_fold, top_k=TOP_K):
        """Aggregate XGBoost SHAP across folds.

        Args:
            mean_abs_per_fold: Array shape (25, 2048) of mean |SHAP| per fold.
            mean_signed_per_fold: Array shape (25, 2048) of mean signed SHAP.
            top_k: Number of top features to return.

        Returns:
            Dict with keys:
                - top_bits: array of top-k bit indices by grand mean |SHAP|.
                - grand_mean_abs: mean |SHAP| across folds for each bit.
                - grand_std_abs: std of mean |SHAP| across folds for each bit.
                - grand_mean_signed: mean signed SHAP across folds for each bit.
                - rank_stability: for each top bit, count of folds where it
                  appeared in the top-k.

            Example::

                | bit  | grand_mean_abs | grand_std_abs | signed | top_k_count |
                |------|----------------|---------------|--------|-------------|
                | 807  | 0.042          | 0.011         | +0.042 | 24/25       |
                | 168  | 0.031          | 0.009         | -0.031 | 21/25       |
        """
        _n_folds = mean_abs_per_fold.shape[0]
        _grand_mean_abs = mean_abs_per_fold.mean(axis=0)
        _grand_std_abs = mean_abs_per_fold.std(axis=0)
        _grand_mean_signed = mean_signed_per_fold.mean(axis=0)

        _top_bits = np.argsort(_grand_mean_abs)[-top_k:][::-1]

        # Rank stability: in how many folds does each bit appear in top-k?
        _rank_stability = {}
        for _bit in _top_bits:
            _count = 0
            for _f in range(_n_folds):
                _fold_top = set(np.argsort(mean_abs_per_fold[_f])[-top_k:][::-1])
                if _bit in _fold_top:
                    _count += 1
            _rank_stability[int(_bit)] = _count

        return {
            "top_bits": _top_bits,
            "grand_mean_abs": _grand_mean_abs,
            "grand_std_abs": _grand_std_abs,
            "grand_mean_signed": _grand_mean_signed,
            "rank_stability": _rank_stability,
        }

    def aggregate_chemprop_saliency(fold_results, top_k=TOP_K):
        """Aggregate Chemprop atom-type saliency across folds.

        Args:
            fold_results: List of 25 dicts, each mapping atom-type -> mean saliency.
            top_k: Number of top atom types to return.

        Returns:
            Dict with keys:
                - top_types: list of top-k atom-type strings by grand mean.
                - grand_means: dict of atom-type -> mean across folds.
                - grand_stds: dict of atom-type -> std across folds.
                - rank_stability: dict of atom-type -> count of folds in top-k.

            Example::

                | atom_type   | grand_mean | grand_std | top_k_count |
                |-------------|------------|-----------|-------------|
                | S deg2      | 0.38       | 0.05      | 25/25       |
                | N deg3      | 0.31       | 0.07      | 22/25       |
        """
        _n_folds = len(fold_results)

        # Collect all atom types seen across any fold
        _all_types = set()
        for _fold_dict in fold_results:
            _all_types.update(_fold_dict.keys())

        # For each atom type, gather values across folds (0.0 if absent)
        _type_values = {}
        for _atype in _all_types:
            _vals = []
            for _fold_dict in fold_results:
                _vals.append(_fold_dict.get(_atype, 0.0))
            _type_values[_atype] = np.array(_vals)

        _grand_means = {k: float(v.mean()) for k, v in _type_values.items()}
        _grand_stds = {k: float(v.std()) for k, v in _type_values.items()}

        _top_types = sorted(
            _grand_means.keys(),
            key=lambda k: _grand_means[k],
            reverse=True,
        )[:top_k]

        # Rank stability
        _rank_stability = {}
        for _atype in _top_types:
            _count = 0
            for _fold_dict in fold_results:
                _fold_top = sorted(
                    _fold_dict.keys(),
                    key=lambda k: _fold_dict[k],
                    reverse=True,
                )[:top_k]
                if _atype in _fold_top:
                    _count += 1
            _rank_stability[_atype] = _count

        return {
            "top_types": _top_types,
            "grand_means": _grand_means,
            "grand_stds": _grand_stds,
            "rank_stability": _rank_stability,
        }

    n_total_folds = N_REPS * N_FOLDS
    return aggregate_chemprop_saliency, aggregate_xgb_shap, n_total_folds


@app.cell
def _(
    Chem,
    FIGURES_DIR,
    Image,
    aggregate_xgb_shap,
    hlm_bit_to_mol,
    io,
    logger,
    mo,
    n_total_folds,
    np,
    pampa_bit_to_mol,
    plt,
    rdMolDraw2D,
    xgb_hlm_scratch_abs,
    xgb_hlm_scratch_signed,
    xgb_hlm_transfer_abs,
    xgb_hlm_transfer_signed,
    xgb_pampa_scratch_abs,
    xgb_pampa_scratch_signed,
    xgb_pampa_transfer_abs,
    xgb_pampa_transfer_signed,
):
    """Plot XGBoost SHAP: scratch vs transfer for HLM and PAMPA."""

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

    def _plot_xgb_comparison(
        agg_scratch,
        agg_transfer,
        bit_to_mol,
        endpoint,
        label_pos,
        label_neg,
        save_name,
    ):
        """Draw 4-column XGBoost SHAP figure: images | bars | bars | images.

        Args:
            agg_scratch: Aggregated scratch results from aggregate_xgb_shap.
            agg_transfer: Aggregated transfer results from aggregate_xgb_shap.
            bit_to_mol: Dict mapping bit index to (mol, center_atom, radius).
            endpoint: Name string (e.g. 'HLM Stability').
            label_pos: Label for positive SHAP direction (e.g. 'stable').
            label_neg: Label for negative SHAP direction (e.g. 'unstable').
            save_name: Filename stem for saving.
        """
        _top_s = agg_scratch["top_bits"]
        _top_t = agg_transfer["top_bits"]
        _n_left = len(_top_s)
        _n_right = len(_top_t)

        _fig = plt.figure(figsize=(22, 8))
        _gs = _fig.add_gridspec(
            1,
            4,
            width_ratios=[0.3, 0.7, 0.7, 0.3],
            wspace=0.08,
        )

        # --- Left images (scratch) ---
        _ax_img_l = _fig.add_subplot(_gs[0])
        _ax_img_l.set_xlim(0, 1)
        _ax_img_l.set_ylim(-0.5, _n_left - 0.5)
        _ax_img_l.invert_yaxis()
        _ax_img_l.axis("off")
        for _j, _bit in enumerate(_top_s):
            if int(_bit) in bit_to_mol:
                _mol, _center, _rad = bit_to_mol[int(_bit)]
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

        # --- Left bars (scratch) ---
        _ax_l = _fig.add_subplot(_gs[1])
        _vals_l = [float(agg_scratch["grand_mean_abs"][b]) for b in _top_s]
        _errs_l = [float(agg_scratch["grand_std_abs"][b]) for b in _top_s]
        _colors_l = [
            "#2196F3" if agg_scratch["grand_mean_signed"][b] > 0 else "#FF5722"
            for b in _top_s
        ]
        _y_pos_l = np.arange(_n_left)
        _ax_l.barh(
            _y_pos_l,
            _vals_l,
            xerr=_errs_l,
            color=_colors_l,
            alpha=0.8,
            height=0.7,
            capsize=3,
            edgecolor="none",
        )
        _ax_l.set_yticks(_y_pos_l)
        _ax_l.set_yticklabels([f"Bit {b}" for b in _top_s], fontsize=9)
        _ax_l.set_xlabel("Mean |SHAP| across 25 folds", fontsize=11)
        _ax_l.set_title(
            f"XGBoost Scratch\n(blue = {label_pos}, red = {label_neg})",
            fontsize=12,
            fontweight="bold",
        )
        _ax_l.invert_yaxis()

        # Rank stability annotations
        for _j, _bit in enumerate(_top_s):
            _count = agg_scratch["rank_stability"][int(_bit)]
            _ax_l.annotate(
                f"{_count}/{n_total_folds}",
                xy=(_vals_l[_j] + _errs_l[_j], _j),
                xytext=(4, 0),
                textcoords="offset points",
                fontsize=8,
                color="#555555",
                va="center",
                fontstyle="italic",
            )

        # --- Right bars (transfer) ---
        _ax_r = _fig.add_subplot(_gs[2])
        _vals_r = [float(agg_transfer["grand_mean_abs"][b]) for b in _top_t]
        _errs_r = [float(agg_transfer["grand_std_abs"][b]) for b in _top_t]
        _colors_r = [
            "#2196F3" if agg_transfer["grand_mean_signed"][b] > 0 else "#FF5722"
            for b in _top_t
        ]
        _y_pos_r = np.arange(_n_right)
        _ax_r.barh(
            _y_pos_r,
            _vals_r,
            xerr=_errs_r,
            color=_colors_r,
            alpha=0.8,
            height=0.7,
            capsize=3,
            edgecolor="none",
        )
        _ax_r.set_yticks(_y_pos_r)
        _ax_r.set_yticklabels([f"Bit {b}" for b in _top_t], fontsize=9)
        _ax_r.yaxis.tick_right()
        _ax_r.yaxis.set_label_position("right")
        _ax_r.set_xlabel("Mean |SHAP| across 25 folds", fontsize=11)
        _ax_r.set_title(
            f"XGBoost RLM-Transfer\n(blue = {label_pos}, red = {label_neg})",
            fontsize=12,
            fontweight="bold",
        )
        _ax_r.invert_yaxis()

        for _j, _bit in enumerate(_top_t):
            _count = agg_transfer["rank_stability"][int(_bit)]
            _ax_r.annotate(
                f"{_count}/{n_total_folds}",
                xy=(_vals_r[_j] + _errs_r[_j], _j),
                xytext=(4, 0),
                textcoords="offset points",
                fontsize=8,
                color="#555555",
                va="center",
                fontstyle="italic",
            )

        # --- Right images (transfer) ---
        _ax_img_r = _fig.add_subplot(_gs[3])
        _ax_img_r.set_xlim(0, 1)
        _ax_img_r.set_ylim(-0.5, _n_right - 0.5)
        _ax_img_r.invert_yaxis()
        _ax_img_r.axis("off")
        for _j, _bit in enumerate(_top_t):
            if int(_bit) in bit_to_mol:
                _mol, _center, _rad = bit_to_mol[int(_bit)]
                try:
                    _img = _draw_neighborhood_xgb(_mol, _center, _rad)
                    if _img:
                        _inset = _ax_img_r.inset_axes(
                            [
                                0.15,
                                (_n_right - 1 - _j) / _n_right,
                                0.85,
                                0.9 / _n_right,
                            ],
                            transform=_ax_img_r.transAxes,
                        )
                        _inset.imshow(_img)
                        _inset.axis("off")
                except Exception:
                    pass

        _fig.suptitle(
            f"{endpoint}: XGBoost Scratch vs RLM-Transfer (SHAP, 25-fold aggregate)",
            fontsize=14,
            fontweight="bold",
            y=1.01,
        )
        plt.tight_layout()
        _save_path = FIGURES_DIR / f"{save_name}.png"
        _fig.savefig(_save_path, dpi=150, bbox_inches="tight", facecolor="white")
        logger.info(f"Saved {_save_path}")
        return _fig

    # Aggregate
    _agg_hlm_s = aggregate_xgb_shap(xgb_hlm_scratch_abs, xgb_hlm_scratch_signed)
    _agg_hlm_t = aggregate_xgb_shap(xgb_hlm_transfer_abs, xgb_hlm_transfer_signed)
    _agg_pampa_s = aggregate_xgb_shap(xgb_pampa_scratch_abs, xgb_pampa_scratch_signed)
    _agg_pampa_t = aggregate_xgb_shap(
        xgb_pampa_transfer_abs,
        xgb_pampa_transfer_signed,
    )

    _fig_hlm = _plot_xgb_comparison(
        _agg_hlm_s,
        _agg_hlm_t,
        hlm_bit_to_mol,
        "HLM Stability",
        "stable",
        "unstable",
        "hlm-xgb-scratch-vs-transfer-aggregate",
    )
    _fig_pampa = _plot_xgb_comparison(
        _agg_pampa_s,
        _agg_pampa_t,
        pampa_bit_to_mol,
        "PAMPA pH 7.4",
        "permeable",
        "impermeable",
        "pampa-xgb-scratch-vs-transfer-aggregate",
    )

    # Log shared bits
    _shared_hlm = set(_agg_hlm_s["top_bits"]) & set(_agg_hlm_t["top_bits"])
    _shared_pampa = set(_agg_pampa_s["top_bits"]) & set(_agg_pampa_t["top_bits"])
    logger.info(f"HLM shared top-10 bits: {len(_shared_hlm)} ({sorted(_shared_hlm)})")
    logger.info(
        f"PAMPA shared top-10 bits: {len(_shared_pampa)} ({sorted(_shared_pampa)})"
    )

    mo.vstack(
        [
            mo.md("## XGBoost SHAP: 25-Fold Aggregate"),
            mo.md("""
    Mean |SHAP| across all 25 CV folds, with cross-fold standard deviation
    as error bars. Italic annotations show rank stability: how many of the
    25 folds each bit appeared in that model's top-10. Substructure images
    show the Morgan radius-3 neighborhood that activates each bit.
            """),
            mo.md("### HLM Stability"),
            mo.as_html(_fig_hlm),
            mo.md("### PAMPA pH 7.4"),
            mo.as_html(_fig_pampa),
        ]
    )
    return


@app.cell
def _(
    Chem,
    FIGURES_DIR,
    Image,
    aggregate_chemprop_saliency,
    cp_hlm_scratch_folds,
    cp_hlm_transfer_folds,
    cp_pampa_scratch_folds,
    cp_pampa_transfer_folds,
    hlm_atype_to_mol,
    io,
    logger,
    mo,
    n_total_folds,
    np,
    pampa_atype_to_mol,
    plt,
    rdMolDraw2D,
):
    """Plot Chemprop saliency: scratch vs transfer for HLM and PAMPA."""

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

    def _plot_chemprop_comparison(
        agg_scratch,
        agg_transfer,
        atype_to_mol,
        endpoint,
        save_name,
    ):
        """Draw 4-column Chemprop saliency figure: images | bars | bars | images.

        Args:
            agg_scratch: Aggregated scratch results from aggregate_chemprop_saliency.
            agg_transfer: Aggregated transfer results.
            atype_to_mol: Dict mapping atom-type key to (mol, atom_idx).
            endpoint: Name string (e.g. 'HLM Stability').
            save_name: Filename stem for saving.
        """
        _top_s = agg_scratch["top_types"]
        _top_t = agg_transfer["top_types"]
        _n_left = len(_top_s)
        _n_right = len(_top_t)

        _fig = plt.figure(figsize=(22, 8))
        _gs = _fig.add_gridspec(
            1,
            4,
            width_ratios=[0.3, 0.7, 0.7, 0.3],
            wspace=0.08,
        )

        # --- Left images (scratch) ---
        _ax_img_l = _fig.add_subplot(_gs[0])
        _ax_img_l.set_xlim(0, 1)
        _ax_img_l.set_ylim(-0.5, _n_left - 0.5)
        _ax_img_l.invert_yaxis()
        _ax_img_l.axis("off")
        for _j, _atype in enumerate(_top_s):
            if _atype in atype_to_mol:
                _mol, _idx = atype_to_mol[_atype]
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

        # --- Left bars (scratch) ---
        _ax_l = _fig.add_subplot(_gs[1])
        _means_l = [agg_scratch["grand_means"][k] for k in _top_s]
        _stds_l = [agg_scratch["grand_stds"][k] for k in _top_s]
        _y_pos_l = np.arange(_n_left)
        _ax_l.barh(
            _y_pos_l,
            _means_l,
            xerr=_stds_l,
            color="#7E57C2",
            alpha=0.8,
            height=0.7,
            capsize=3,
            edgecolor="none",
        )
        _ax_l.set_yticks(_y_pos_l)
        _ax_l.set_yticklabels(_top_s, fontsize=9)
        _ax_l.set_xlabel("Mean normalized saliency across 25 folds", fontsize=11)
        _ax_l.set_title("Chemprop Scratch", fontsize=12, fontweight="bold")
        _ax_l.invert_yaxis()

        for _j, _atype in enumerate(_top_s):
            _count = agg_scratch["rank_stability"][_atype]
            _ax_l.annotate(
                f"{_count}/{n_total_folds}",
                xy=(_means_l[_j] + _stds_l[_j], _j),
                xytext=(4, 0),
                textcoords="offset points",
                fontsize=8,
                color="#555555",
                va="center",
                fontstyle="italic",
            )

        # --- Right bars (transfer) ---
        _ax_r = _fig.add_subplot(_gs[2])
        _means_r = [agg_transfer["grand_means"][k] for k in _top_t]
        _stds_r = [agg_transfer["grand_stds"][k] for k in _top_t]
        _y_pos_r = np.arange(_n_right)
        _ax_r.barh(
            _y_pos_r,
            _means_r,
            xerr=_stds_r,
            color="#00897B",
            alpha=0.8,
            height=0.7,
            capsize=3,
            edgecolor="none",
        )
        _ax_r.set_yticks(_y_pos_r)
        _ax_r.set_yticklabels(_top_t, fontsize=9)
        _ax_r.yaxis.tick_right()
        _ax_r.yaxis.set_label_position("right")
        _ax_r.set_xlabel("Mean normalized saliency across 25 folds", fontsize=11)
        _ax_r.set_title("Chemprop RLM-Transfer", fontsize=12, fontweight="bold")
        _ax_r.invert_yaxis()

        for _j, _atype in enumerate(_top_t):
            _count = agg_transfer["rank_stability"][_atype]
            _ax_r.annotate(
                f"{_count}/{n_total_folds}",
                xy=(_means_r[_j] + _stds_r[_j], _j),
                xytext=(4, 0),
                textcoords="offset points",
                fontsize=8,
                color="#555555",
                va="center",
                fontstyle="italic",
            )

        # --- Right images (transfer) ---
        _ax_img_r = _fig.add_subplot(_gs[3])
        _ax_img_r.set_xlim(0, 1)
        _ax_img_r.set_ylim(-0.5, _n_right - 0.5)
        _ax_img_r.invert_yaxis()
        _ax_img_r.axis("off")
        for _j, _atype in enumerate(_top_t):
            if _atype in atype_to_mol:
                _mol, _idx = atype_to_mol[_atype]
                try:
                    _img = _draw_neighborhood_cp(_mol, _idx, 1)
                    if _img:
                        _inset = _ax_img_r.inset_axes(
                            [
                                0.15,
                                (_n_right - 1 - _j) / _n_right,
                                0.85,
                                0.9 / _n_right,
                            ],
                            transform=_ax_img_r.transAxes,
                        )
                        _inset.imshow(_img)
                        _inset.axis("off")
                except Exception:
                    pass

        _fig.suptitle(
            f"{endpoint}: Chemprop Scratch vs RLM-Transfer "
            f"(Saliency, 25-fold aggregate)",
            fontsize=14,
            fontweight="bold",
            y=1.01,
        )
        plt.tight_layout()
        _save_path = FIGURES_DIR / f"{save_name}.png"
        _fig.savefig(_save_path, dpi=150, bbox_inches="tight", facecolor="white")
        logger.info(f"Saved {_save_path}")
        return _fig

    # Aggregate
    _agg_hlm_s = aggregate_chemprop_saliency(cp_hlm_scratch_folds)
    _agg_hlm_t = aggregate_chemprop_saliency(cp_hlm_transfer_folds)
    _agg_pampa_s = aggregate_chemprop_saliency(cp_pampa_scratch_folds)
    _agg_pampa_t = aggregate_chemprop_saliency(cp_pampa_transfer_folds)

    _fig_hlm = _plot_chemprop_comparison(
        _agg_hlm_s,
        _agg_hlm_t,
        hlm_atype_to_mol,
        "HLM Stability",
        "hlm-chemprop-scratch-vs-transfer-aggregate",
    )
    _fig_pampa = _plot_chemprop_comparison(
        _agg_pampa_s,
        _agg_pampa_t,
        pampa_atype_to_mol,
        "PAMPA pH 7.4",
        "pampa-chemprop-scratch-vs-transfer-aggregate",
    )

    # Log shared types
    _shared_hlm = set(_agg_hlm_s["top_types"]) & set(_agg_hlm_t["top_types"])
    _shared_pampa = set(_agg_pampa_s["top_types"]) & set(_agg_pampa_t["top_types"])
    logger.info(
        f"HLM shared top-10 atom types: {len(_shared_hlm)} ({sorted(_shared_hlm)})"
    )
    logger.info(
        f"PAMPA shared top-10 atom types: {len(_shared_pampa)} "
        f"({sorted(_shared_pampa)})"
    )

    mo.vstack(
        [
            mo.md("## Chemprop Saliency: 25-Fold Aggregate"),
            mo.md("""
    Mean normalized atom-type saliency across all 25 CV folds, with
    cross-fold standard deviation as error bars. Italic annotations show
    rank stability: how many of the 25 folds each atom type appeared in
    that model's top-10. Fragment images show the 1-bond neighborhood
    around a representative instance of each atom type (center highlighted).
            """),
            mo.md("### HLM Stability"),
            mo.as_html(_fig_hlm),
            mo.md("### PAMPA pH 7.4"),
            mo.as_html(_fig_pampa),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
