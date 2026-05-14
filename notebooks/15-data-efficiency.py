import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md("""
    # 15 — Data Efficiency: How Much Data Do You Need?

    Trains XGBoost scratch, Chemprop scratch, CheMeleon frozen
    single-finetune, and MIST frozen single-finetune on 1%, 10%, 25%, 50%,
    75%, and 100% of the training data for each endpoint (RLM, HLM,
    PAMPA). Tests whether the advantage of learned representations grows
    with less data across mechanistically diverse endpoints.

    MIST is a 28M-parameter SMILES transformer (RoBERTa-PreLayerNorm,
    MLM-pretrained on Enamine REAL Space). Because its encoder is frozen
    here, [CLS] embeddings are deterministic and pre-computed once per
    SMILES via ``scripts/run-mist-embed.py``. The data-efficiency loop
    then trains only a Chemprop-matched FFN head on the cached
    embeddings.
    """)
    return (mo,)


@app.cell
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import torch
    import xgboost as xgb
    from lightning import pytorch as lightning_pl
    from loguru import logger
    from sklearn.metrics import average_precision_score, roc_auc_score
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    from chemprop import data as chemprop_data
    from chemprop import featurizers, models, nn

    DATA_DIR = Path("data")
    CHECKPOINTS_DIR = Path("checkpoints")
    FIGURES_DIR = Path("docs/figures")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    def _pick_accelerator() -> tuple[str, int]:
        """Return (accelerator, devices) appropriate for Lightning Trainer."""
        if torch.cuda.is_available():
            return "gpu", 1
        if torch.backends.mps.is_available():
            return "mps", 1
        return "cpu", 1

    LIGHTNING_ACCELERATOR, LIGHTNING_DEVICES = _pick_accelerator()
    logger.info(
        f"Using Lightning accelerator={LIGHTNING_ACCELERATOR} "
        f"devices={LIGHTNING_DEVICES}"
    )
    return (
        CHECKPOINTS_DIR,
        DATA_DIR,
        FIGURES_DIR,
        LIGHTNING_ACCELERATOR,
        LIGHTNING_DEVICES,
        average_precision_score,
        chemprop_data,
        featurizers,
        lightning_pl,
        logger,
        models,
        nn,
        np,
        pairwise_tukeyhsd,
        pl,
        plt,
        roc_auc_score,
        torch,
        xgb,
    )


@app.cell
def _(CHECKPOINTS_DIR, DATA_DIR, logger, np, torch):
    """Load data for all three endpoints, CheMeleon encoder, and MIST embeddings."""
    import json

    with open(DATA_DIR / "split_config.json") as _f:
        split_config = json.load(_f)

    _fp_data = np.load(DATA_DIR / "morgan_fps_2048_r3.npz", allow_pickle=True)
    global_fps = _fp_data["fp_matrix"]

    # Load MIST embeddings (run scripts/run-mist-embed.py first if missing)
    _mist_path = DATA_DIR / "mist_embeddings.npz"
    if not _mist_path.exists():
        raise FileNotFoundError(
            f"MIST embedding cache not found at {_mist_path}. "
            "Run `uv run python scripts/run-mist-embed.py` first."
        )
    _mist_data = np.load(_mist_path, allow_pickle=True)
    mist_smiles_to_idx = {str(s): i for i, s in enumerate(_mist_data["smiles"])}
    mist_embeddings = _mist_data["embeddings"]  # (N_unique, hidden_size)
    logger.info(
        f"MIST embeddings: {mist_embeddings.shape} "
        f"covering {len(mist_smiles_to_idx)} unique SMILES"
    )

    # Load all three endpoints
    endpoint_data = {}
    for _endpoint, _baseline in [("rlm", 0.298), ("hlm", 0.602), ("pampa", 0.855)]:
        _split = np.load(DATA_DIR / f"{_endpoint}_splits.npz", allow_pickle=True)
        _smi_list = list(_split["smiles"])
        # Per-endpoint MIST embedding matrix, aligned to the endpoint's SMILES order.
        _mist_idx = np.array(
            [mist_smiles_to_idx[str(s)] for s in _smi_list], dtype=np.int64
        )
        endpoint_data[_endpoint] = {
            "smiles": _smi_list,
            "labels": _split["labels"],
            "folds": _split["folds"],
            "fp_indices": _split["fp_indices"],
            "X": global_fps[_split["fp_indices"]],
            "X_mist": mist_embeddings[_mist_idx],
            "baseline": _baseline,
        }
        logger.info(
            f"{_endpoint.upper()}: {len(endpoint_data[_endpoint]['smiles'])} molecules, "
            f"positive rate={_split['labels'].mean():.3f}"
        )

    _cm_data = torch.load(CHECKPOINTS_DIR / "chemeleon_mp.pt", weights_only=True)
    chemeleon_mp_params = _cm_data["hyper_parameters"]
    chemeleon_mp_state = _cm_data["state_dict"]

    logger.info(f"CV: {split_config['n_replicates']}x{split_config['n_folds']}")
    return chemeleon_mp_params, chemeleon_mp_state, endpoint_data, split_config


@app.cell
def _(
    DATA_DIR,
    LIGHTNING_ACCELERATOR,
    LIGHTNING_DEVICES,
    average_precision_score,
    chemeleon_mp_params,
    chemeleon_mp_state,
    chemprop_data,
    endpoint_data,
    featurizers,
    lightning_pl,
    logger,
    models,
    nn,
    np,
    pl,
    roc_auc_score,
    split_config,
    torch,
    xgb,
):
    """Run data efficiency experiment across all endpoints, fractions, and models."""
    import copy

    def _train_mist_head(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        x_test: np.ndarray,
        seed: int,
        max_epochs: int = 30,
        batch_size: int = 64,
        lr: float = 1e-3,
        patience: int = 5,
    ) -> np.ndarray:
        """Train a Chemprop-shaped FFN head on cached MIST [CLS] embeddings.

        Mirrors ``chemprop.nn.BinaryClassificationFFN`` defaults so the
        result is comparable to Chemprop / CheMeleon frozen runs:
        Linear(d -> 300) -> ReLU -> Linear(300 -> 1) -> Sigmoid, BCE loss.

        The MIST encoder is not part of this module: we operate on the
        pre-computed [CLS] embeddings, so each (rep, fold, fraction) call
        only optimises 154k parameters on a few hundred to a few thousand
        examples.

        Returns predicted probabilities for ``x_test`` of shape (N_test,).
        """
        torch.manual_seed(seed)
        device = torch.device("cpu")  # head is tiny; CPU is faster than MPS overhead
        d_in = x_train.shape[1]
        head = torch.nn.Sequential(
            torch.nn.Linear(d_in, 300),
            torch.nn.ReLU(),
            torch.nn.Linear(300, 1),
        ).to(device)
        opt = torch.optim.Adam(head.parameters(), lr=lr)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        x_train_t = torch.from_numpy(x_train).float().to(device)
        y_train_t = torch.from_numpy(y_train).float().to(device)
        x_val_t = torch.from_numpy(x_val).float().to(device)
        y_val_t = torch.from_numpy(y_val).float().to(device)
        x_test_t = torch.from_numpy(x_test).float().to(device)

        n_train = len(x_train_t)
        best_val = float("inf")
        best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        epochs_no_improve = 0

        for _epoch in range(max_epochs):
            head.train()
            perm = torch.randperm(n_train, generator=torch.Generator().manual_seed(seed + _epoch))
            for start in range(0, n_train, batch_size):
                idx = perm[start : start + batch_size]
                logits = head(x_train_t[idx]).squeeze(-1)
                loss = loss_fn(logits, y_train_t[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()

            head.eval()
            with torch.no_grad():
                val_logits = head(x_val_t).squeeze(-1)
                val_loss = loss_fn(val_logits, y_val_t).item()
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

        head.load_state_dict(best_state)
        head.eval()
        with torch.no_grad():
            test_logits = head(x_test_t).squeeze(-1)
            test_probs = torch.sigmoid(test_logits).cpu().numpy()
        return test_probs

    _featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    _fractions = [0.01, 0.10, 0.25, 0.50, 0.75, 1.00]
    _n_reps = split_config["n_replicates"]
    _n_folds = split_config["n_folds"]

    _all_results = []

    for _endpoint_name, _ep_data in endpoint_data.items():
        _smiles = _ep_data["smiles"]
        _labels = _ep_data["labels"]
        _folds_arr = _ep_data["folds"]
        _X = _ep_data["X"]
        _X_mist = _ep_data["X_mist"]

        for _frac in _fractions:
            for _rep in range(_n_reps):
                _fold_assign = _folds_arr[_rep]
                for _fold in range(_n_folds):
                    _test_mask = _fold_assign == _fold
                    _train_mask = ~_test_mask
                    _y_test = _labels[_test_mask]

                    # Subsample training data
                    _train_indices = np.where(_train_mask)[0]
                    _rng = np.random.default_rng(42 + _rep * 100 + _fold)
                    _n_train = len(_train_indices)
                    _n_sub = max(10, int(_n_train * _frac))
                    _sub_indices = _rng.choice(
                        _train_indices, size=_n_sub, replace=False
                    )
                    _sub_mask = np.zeros(len(_labels), dtype=bool)
                    _sub_mask[_sub_indices] = True

                    _pct_label = f"{int(_frac * 100)}%"
                    logger.info(
                        f"{_endpoint_name.upper()} frac={_pct_label} "
                        f"rep={_rep} fold={_fold} train={_n_sub}"
                    )

                    # --- XGBoost scratch ---
                    np.random.seed(42 + _rep * 100 + _fold)
                    _dtrain = xgb.DMatrix(_X[_sub_mask], label=_labels[_sub_mask])
                    _dval = xgb.DMatrix(_X[_test_mask], label=_y_test)
                    _xgb_model = xgb.train(
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
                    _y_prob_xgb = _xgb_model.predict(_dval)
                    _all_results.append(
                        {
                            "endpoint": _endpoint_name,
                            "fraction": _frac,
                            "pct_label": _pct_label,
                            "model": "XGBoost scratch",
                            "replicate": _rep,
                            "fold": _fold,
                            "n_train": _n_sub,
                            "auc_roc": roc_auc_score(_y_test, _y_prob_xgb),
                            "avg_precision": average_precision_score(
                                _y_test, _y_prob_xgb
                            ),
                        }
                    )

                    # --- Chemprop scratch ---
                    lightning_pl.seed_everything(42 + _rep * 100 + _fold, workers=True)
                    _train_smi = [_smiles[i] for i in _sub_indices]
                    _train_y = _labels[_sub_indices].reshape(-1, 1).astype(float)
                    _test_smi = [
                        _smiles[i] for i in range(len(_smiles)) if _test_mask[i]
                    ]
                    _test_y = _y_test.reshape(-1, 1).astype(float)

                    _n = len(_train_smi)
                    _n_val = max(2, int(_n * 0.1))
                    _perm = np.random.default_rng(42).permutation(_n)
                    _train_data = [
                        chemprop_data.MoleculeDatapoint.from_smi(
                            _train_smi[i], _train_y[i]
                        )
                        for i in _perm[_n_val:]
                    ]
                    _val_data = [
                        chemprop_data.MoleculeDatapoint.from_smi(
                            _train_smi[i], _train_y[i]
                        )
                        for i in _perm[:_n_val]
                    ]
                    _test_data = [
                        chemprop_data.MoleculeDatapoint.from_smi(s, y)
                        for s, y in zip(_test_smi, _test_y)
                    ]

                    # Chemprop's build_dataloader uses drop_last=True when
                    # len(dataset) > batch_size, which produces zero batches
                    # for a 1-sample dataset at batch_size=64. Cap batch size
                    # to dataset length to avoid empty loaders.
                    _bs_train = min(64, len(_train_data))
                    _bs_val = min(64, len(_val_data))
                    _bs_test = min(64, len(_test_data))

                    _train_ds = chemprop_data.MoleculeDataset(_train_data, _featurizer)
                    _val_ds = chemprop_data.MoleculeDataset(_val_data, _featurizer)
                    _test_ds = chemprop_data.MoleculeDataset(_test_data, _featurizer)
                    _train_loader = chemprop_data.build_dataloader(
                        _train_ds, num_workers=0, batch_size=_bs_train
                    )
                    _val_loader = chemprop_data.build_dataloader(
                        _val_ds, num_workers=0, shuffle=False, batch_size=_bs_val
                    )
                    _test_loader = chemprop_data.build_dataloader(
                        _test_ds, num_workers=0, shuffle=False, batch_size=_bs_test
                    )

                    _mp = nn.BondMessagePassing()
                    _agg = nn.MeanAggregation()
                    _ffn = nn.BinaryClassificationFFN(input_dim=_mp.output_dim)
                    _cp_model = models.MPNN(_mp, _agg, _ffn, batch_norm=False)
                    _trainer = lightning_pl.Trainer(
                        logger=False,
                        enable_checkpointing=False,
                        enable_progress_bar=False,
                        deterministic=True,
                        accelerator=LIGHTNING_ACCELERATOR,
                        devices=LIGHTNING_DEVICES,
                        max_epochs=30,
                    )
                    _trainer.fit(_cp_model, _train_loader, _val_loader)
                    _preds = _trainer.predict(_cp_model, _test_loader)
                    _y_prob_cp = torch.cat(_preds).cpu().numpy().flatten()
                    _all_results.append(
                        {
                            "endpoint": _endpoint_name,
                            "fraction": _frac,
                            "pct_label": _pct_label,
                            "model": "Chemprop scratch",
                            "replicate": _rep,
                            "fold": _fold,
                            "n_train": _n_sub,
                            "auc_roc": roc_auc_score(_y_test, _y_prob_cp),
                            "avg_precision": average_precision_score(
                                _y_test, _y_prob_cp
                            ),
                        }
                    )

                    # --- CheMeleon frozen single ---
                    lightning_pl.seed_everything(42 + _rep * 100 + _fold, workers=True)
                    _cm_mp = nn.BondMessagePassing(**chemeleon_mp_params)
                    _cm_mp.load_state_dict(copy.deepcopy(chemeleon_mp_state))
                    for _p in _cm_mp.parameters():
                        _p.requires_grad = False
                    _cm_agg = nn.MeanAggregation()
                    _cm_ffn = nn.BinaryClassificationFFN(input_dim=_cm_mp.output_dim)
                    _cm_model = models.MPNN(_cm_mp, _cm_agg, _cm_ffn, batch_norm=False)
                    _trainer2 = lightning_pl.Trainer(
                        logger=False,
                        enable_checkpointing=False,
                        enable_progress_bar=False,
                        deterministic=True,
                        accelerator=LIGHTNING_ACCELERATOR,
                        devices=LIGHTNING_DEVICES,
                        max_epochs=30,
                    )
                    _trainer2.fit(_cm_model, _train_loader, _val_loader)
                    _preds2 = _trainer2.predict(_cm_model, _test_loader)
                    _y_prob_cm = torch.cat(_preds2).cpu().numpy().flatten()
                    _all_results.append(
                        {
                            "endpoint": _endpoint_name,
                            "fraction": _frac,
                            "pct_label": _pct_label,
                            "model": "CheMeleon frozen",
                            "replicate": _rep,
                            "fold": _fold,
                            "n_train": _n_sub,
                            "auc_roc": roc_auc_score(_y_test, _y_prob_cm),
                            "avg_precision": average_precision_score(
                                _y_test, _y_prob_cm
                            ),
                        }
                    )

                    # --- MIST frozen single ---
                    # Reuse the same train / val split (in subsampled-index
                    # order) as Chemprop / CheMeleon for fair comparison.
                    _train_idx_in_sub = _perm[_n_val:]
                    _val_idx_in_sub = _perm[:_n_val]
                    _mist_x_train = _X_mist[_sub_indices[_train_idx_in_sub]]
                    _mist_y_train = _labels[_sub_indices[_train_idx_in_sub]].astype(
                        np.float32
                    )
                    _mist_x_val = _X_mist[_sub_indices[_val_idx_in_sub]]
                    _mist_y_val = _labels[_sub_indices[_val_idx_in_sub]].astype(
                        np.float32
                    )
                    _mist_x_test = _X_mist[_test_mask]

                    _y_prob_mist = _train_mist_head(
                        x_train=_mist_x_train,
                        y_train=_mist_y_train,
                        x_val=_mist_x_val,
                        y_val=_mist_y_val,
                        x_test=_mist_x_test,
                        seed=42 + _rep * 100 + _fold,
                    )
                    _all_results.append(
                        {
                            "endpoint": _endpoint_name,
                            "fraction": _frac,
                            "pct_label": _pct_label,
                            "model": "MIST frozen",
                            "replicate": _rep,
                            "fold": _fold,
                            "n_train": _n_sub,
                            "auc_roc": roc_auc_score(_y_test, _y_prob_mist),
                            "avg_precision": average_precision_score(
                                _y_test, _y_prob_mist
                            ),
                        }
                    )

    efficiency_df = pl.DataFrame(_all_results)
    efficiency_df.write_parquet(DATA_DIR / "data_efficiency_results.parquet")
    logger.info(f"Saved {efficiency_df.height} results")
    return (efficiency_df,)


@app.cell
def _(
    FIGURES_DIR,
    efficiency_df,
    endpoint_data,
    logger,
    mo,
    pairwise_tukeyhsd,
    pl,
    plt,
):
    """Generate 3-panel data efficiency figure (one panel per endpoint)."""
    _fractions = [0.01, 0.10, 0.25, 0.50, 0.75, 1.00]
    _models = ["XGBoost scratch", "Chemprop scratch", "CheMeleon frozen", "MIST frozen"]
    _colors = {
        "XGBoost scratch": "#FF5722",
        "Chemprop scratch": "#2196F3",
        "CheMeleon frozen": "#7E57C2",
        "MIST frozen": "#00897B",
    }
    _markers = {
        "XGBoost scratch": "o",
        "Chemprop scratch": "s",
        "CheMeleon frozen": "D",
        "MIST frozen": "^",
    }
    _endpoint_titles = {
        "rlm": "RLM Stability",
        "hlm": "HLM Stability",
        "pampa": "PAMPA pH 7.4",
    }
    _endpoint_order = ["rlm", "hlm", "pampa"]

    # Compute means and stderrs per endpoint per model per fraction
    _summary = (
        efficiency_df.group_by("endpoint", "fraction", "model")
        .agg(
            pl.col("avg_precision").mean().alias("mean"),
            pl.col("avg_precision").std().alias("std"),
            pl.col("avg_precision").count().alias("n"),
        )
        .with_columns((pl.col("std") / pl.col("n").sqrt()).alias("stderr"))
        .sort("endpoint", "fraction", "model")
    )

    # Run Tukey HSD per endpoint per fraction to find best group
    _best_group = {}  # (endpoint, fraction, model) -> bool
    for _ep in _endpoint_order:
        for _frac in _fractions:
            _subset = efficiency_df.filter(
                (pl.col("endpoint") == _ep) & (pl.col("fraction") == _frac)
            )
            _values = _subset.get_column("avg_precision").to_numpy()
            _groups = _subset.get_column("model").to_list()
            _tukey = pairwise_tukeyhsd(_values, _groups, alpha=0.05)

            _means = (
                _subset.group_by("model")
                .agg(pl.col("avg_precision").mean())
                .sort("avg_precision", descending=True)
            )
            _best_model = _means.get_column("model")[0]

            for _m in _models:
                if _m == _best_model:
                    _best_group[(_ep, _frac, _m)] = True
                else:
                    _sig = False
                    for _i in range(len(_tukey.groupsunique)):
                        for _j in range(_i + 1, len(_tukey.groupsunique)):
                            _g1 = _tukey.groupsunique[_i]
                            _g2 = _tukey.groupsunique[_j]
                            _idx = (
                                _i * (len(_tukey.groupsunique) - 1)
                                - _i * (_i - 1) // 2
                                + _j
                                - _i
                                - 1
                            )
                            if {_g1, _g2} == {_best_model, _m}:
                                _sig = _tukey.reject[_idx]
                    _best_group[(_ep, _frac, _m)] = not _sig

    # --- 3-panel figure ---
    _fig, _axes = plt.subplots(1, 3, figsize=(20, 6), sharey=False)

    _x_offsets = {
        "XGBoost scratch": -0.012,
        "Chemprop scratch": -0.004,
        "CheMeleon frozen": 0.004,
        "MIST frozen": 0.012,
    }

    for _panel_idx, _ep in enumerate(_endpoint_order):
        _ax = _axes[_panel_idx]
        _baseline = endpoint_data[_ep]["baseline"]

        for _model in _models:
            _model_data = _summary.filter(
                (pl.col("endpoint") == _ep) & (pl.col("model") == _model)
            ).sort("fraction")
            _fracs = _model_data.get_column("fraction").to_numpy()
            _means_arr = _model_data.get_column("mean").to_numpy()
            _stderrs_arr = _model_data.get_column("stderr").to_numpy()

            _x = _fracs + _x_offsets[_model]

            _ax.errorbar(
                _x,
                _means_arr,
                yerr=_stderrs_arr,
                color=_colors[_model],
                marker=_markers[_model],
                markersize=8,
                linewidth=2,
                capsize=4,
                capthick=1.5,
                label=_model,
                alpha=0.9,
                zorder=3,
            )

            # Asterisks for best-group membership
            for _i, _frac in enumerate(_fracs):
                if _best_group.get((_ep, _frac, _model), False):
                    _ax.annotate(
                        "*",
                        xy=(_x[_i], _means_arr[_i] + _stderrs_arr[_i] + 0.008),
                        ha="center",
                        va="bottom",
                        fontsize=16,
                        fontweight="bold",
                        color=_colors[_model],
                    )

        _ax.axhline(
            _baseline,
            color="black",
            linestyle="--",
            linewidth=1,
            alpha=0.5,
            label=f"Random baseline ({_baseline:.3f})",
        )

        _ax.set_xlabel("Fraction of Training Data", fontsize=12)
        _ax.set_xticks(_fractions)
        _ax.set_xticklabels(["1%", "10%", "25%", "50%", "75%", "100%"])
        _ax.set_title(_endpoint_titles[_ep], fontsize=14, fontweight="bold")
        _ax.grid(axis="y", alpha=0.3)

        if _panel_idx == 0:
            _ax.set_ylabel("AUC-PR", fontsize=12)
            _ax.legend(fontsize=9, loc="lower right")

    _fig.tight_layout()
    _path = FIGURES_DIR / "data-efficiency-all-endpoints.png"
    _fig.savefig(_path, dpi=150, bbox_inches="tight", facecolor="white")
    logger.info(f"Saved {_path}")

    # --- Also save single-panel RLM for backward compatibility ---
    _fig_rlm, _ax_rlm = plt.subplots(figsize=(10, 6))
    for _model in _models:
        _model_data = _summary.filter(
            (pl.col("endpoint") == "rlm") & (pl.col("model") == _model)
        ).sort("fraction")
        _fracs = _model_data.get_column("fraction").to_numpy()
        _means_arr = _model_data.get_column("mean").to_numpy()
        _stderrs_arr = _model_data.get_column("stderr").to_numpy()
        _x = _fracs + _x_offsets[_model]

        _ax_rlm.errorbar(
            _x,
            _means_arr,
            yerr=_stderrs_arr,
            color=_colors[_model],
            marker=_markers[_model],
            markersize=8,
            linewidth=2,
            capsize=4,
            capthick=1.5,
            label=_model,
            alpha=0.9,
            zorder=3,
        )
        for _i, _frac in enumerate(_fracs):
            if _best_group.get(("rlm", _frac, _model), False):
                _ax_rlm.annotate(
                    "*",
                    xy=(_x[_i], _means_arr[_i] + _stderrs_arr[_i] + 0.008),
                    ha="center",
                    va="bottom",
                    fontsize=16,
                    fontweight="bold",
                    color=_colors[_model],
                )

    _rlm_baseline = endpoint_data["rlm"]["baseline"]
    _ax_rlm.axhline(
        _rlm_baseline,
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label=f"Random baseline ({_rlm_baseline:.3f})",
    )
    _ax_rlm.set_xlabel("Fraction of Training Data", fontsize=12)
    _ax_rlm.set_ylabel("AUC-PR", fontsize=12)
    _ax_rlm.set_title(
        "RLM Stability: Data Efficiency by Architecture",
        fontsize=14,
        fontweight="bold",
    )
    _ax_rlm.set_xticks(_fractions)
    _ax_rlm.set_xticklabels(["1%", "10%", "25%", "50%", "75%", "100%"])
    _ax_rlm.legend(fontsize=10, loc="lower right")
    _ax_rlm.grid(axis="y", alpha=0.3)
    _fig_rlm.tight_layout()
    _path_rlm = FIGURES_DIR / "data-efficiency-rlm.png"
    _fig_rlm.savefig(_path_rlm, dpi=150, bbox_inches="tight", facecolor="white")
    logger.info(f"Saved {_path_rlm}")

    # Summary table
    _summary_wide = (
        _summary.select("endpoint", "fraction", "model", "mean", "stderr")
        .with_columns(
            (
                pl.col("mean").round(3).cast(str)
                + " +/- "
                + pl.col("stderr").round(3).cast(str)
            ).alias("auc_pr"),
        )
        .pivot(on="model", index=["endpoint", "fraction"], values="auc_pr")
        .sort("endpoint", "fraction")
    )

    mo.vstack(
        [
            mo.md("## Data Efficiency: All Endpoints"),
            mo.as_html(_fig),
            mo.md("""
    *Mean AUC-PR (+/- SEM) across 25 CV folds at each training data
    fraction. Asterisks mark models not significantly different from the best
    at that fraction (Tukey HSD, FWER = 0.05). Dotted line = random
    baseline.*
        """),
            mo.ui.table(_summary_wide),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
