import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md("""
    # 02 — Exploratory Data Analysis

    Explore the curated NCATS ADME datasets: distributions of target values,
    class balance, molecule overlap across endpoints, and correlation of
    shared compounds.
    """)
    return (mo,)


@app.cell(hide_code=True)
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import polars as pl
    import seaborn as sns
    from loguru import logger
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    DATA_DIR = Path("data")

    ENDPOINT_NAMES = {
        "rlm": "RLM Stability",
        "hlm": "HLM Stability",
        "pampa": "PAMPA pH 7.4",
    }

    datasets: dict[str, pl.DataFrame] = {}
    for key in ENDPOINT_NAMES:
        path = DATA_DIR / f"{key}_curated.parquet"
        datasets[key] = pl.read_parquet(path)
        logger.info(f"Loaded {key}: {datasets[key].height} rows")
    return (
        Chem,
        DATA_DIR,
        ENDPOINT_NAMES,
        MurckoScaffold,
        datasets,
        logger,
        pl,
        plt,
    )


@app.cell(hide_code=True)
def _(ENDPOINT_NAMES, datasets: "dict[str, pl.DataFrame]", mo, pl):
    _summary_rows = []
    for _key, _df in datasets.items():
        _n_total = _df.height
        _n_with_cont = _df.filter(pl.col("continuous_value").is_not_null()).height
        _n_censored = _df.filter(pl.col("continuous_value_censored")).height
        _n_active = _df.filter(pl.col("binary_label") == 1).height
        _n_inactive = _df.filter(pl.col("binary_label") == 0).height
        _summary_rows.append({
            "endpoint": ENDPOINT_NAMES[_key],
            "total_compounds": _n_total,
            "has_continuous_value": _n_with_cont,
            "censored_values": _n_censored,
            "active": _n_active,
            "inactive": _n_inactive,
            "pct_active": round(_n_active / _n_total * 100, 1),
        })

    _summary_df = pl.DataFrame(_summary_rows)

    mo.vstack([
        mo.md("## Dataset Overview"),
        mo.ui.table(_summary_df),
        mo.md("""
    **Notes:**
    - RLM/HLM: *active* = stable (t1/2 > 30 min). RLM is heavily imbalanced toward unstable.
    - PAMPA: *active* = moderate/high permeability (> 10 x10^-6 cm/s). Most compounds are permeable.
    - Censored values: RLM has 754 compounds with t1/2 reported as ">30" (capped at assay limit).
      PAMPA has 483 with permeability ">1000".
        """),
    ])
    return


@app.cell(hide_code=True)
def _(ENDPOINT_NAMES, datasets: "dict[str, pl.DataFrame]", mo, pl, plt):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for _i, (_key, _df) in enumerate(datasets.items()):
        _vals = _df.filter(
            pl.col("continuous_value").is_not_null()
            & ~pl.col("continuous_value_censored")
        ).get_column("continuous_value").to_list()

        axes[_i].hist(_vals, bins=50, edgecolor="black", alpha=0.7)
        axes[_i].set_title(ENDPOINT_NAMES[_key])
        axes[_i].set_xlabel(
            "Half-life (min)" if _key in ("rlm", "hlm") else "Permeability (x10⁻⁶ cm/s)"
        )
        axes[_i].set_ylabel("Count")

    fig.suptitle("Distribution of Continuous Values (uncensored only)", fontsize=14)
    plt.tight_layout()
    mo.vstack([
        mo.md("## Continuous Value Distributions"),
        mo.as_html(fig),
        mo.md("Censored values (e.g., >30 for RLM, >1000 for PAMPA) are excluded from these histograms."),
    ])
    return


@app.cell(hide_code=True)
def _(ENDPOINT_NAMES, datasets: "dict[str, pl.DataFrame]", mo, pl, plt):
    _fig2, _axes2 = plt.subplots(1, 3, figsize=(12, 4))

    for _i, (_key, _df) in enumerate(datasets.items()):
        _n_active = _df.filter(pl.col("binary_label") == 1).height
        _n_inactive = _df.filter(pl.col("binary_label") == 0).height
        _axes2[_i].bar(
            ["Active", "Inactive"],
            [_n_active, _n_inactive],
            color=["#2196F3", "#FF5722"],
        )
        _axes2[_i].set_title(ENDPOINT_NAMES[_key])
        _axes2[_i].set_ylabel("Count")
        for _j, _v in enumerate([_n_active, _n_inactive]):
            _axes2[_i].text(_j, _v + 10, str(_v), ha="center", fontsize=10)

    _fig2.suptitle("Class Balance", fontsize=14)
    plt.tight_layout()
    mo.vstack([
        mo.md("## Class Balance"),
        mo.as_html(_fig2),
        mo.md("""
    - **RLM**: heavily skewed toward *unstable* (70% inactive). Pre-training source.
    - **HLM**: more balanced (60% active / 40% inactive). Related finetune target.
    - **PAMPA**: heavily skewed toward *permeable* (86% active). Unrelated finetune target.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(Chem, MurckoScaffold, datasets: "dict[str, pl.DataFrame]", pl):
    def get_murcko_scaffold(smiles: str) -> str | None:
        """Compute Bemis-Murcko scaffold for a SMILES string."""
        _mol = Chem.MolFromSmiles(smiles)
        if _mol is None:
            return None
        try:
            _scaffold = MurckoScaffold.GetScaffoldForMol(_mol)
            return Chem.MolToSmiles(_scaffold)
        except Exception:
            return None

    # Compute SMILES sets
    smiles_sets: dict[str, set[str]] = {}
    scaffold_sets: dict[str, set[str]] = {}
    for _key, _df in datasets.items():
        _smiles_list = _df.get_column("canonical_smiles").to_list()
        smiles_sets[_key] = set(_smiles_list)
        _scaffolds = [get_murcko_scaffold(s) for s in _smiles_list]
        scaffold_sets[_key] = {s for s in _scaffolds if s is not None}

    # Pairwise overlap
    _pairs = [("rlm", "hlm"), ("rlm", "pampa"), ("hlm", "pampa")]
    _overlap_rows = []
    for _a, _b in _pairs:
        _shared_smiles = smiles_sets[_a] & smiles_sets[_b]
        _shared_scaffolds = scaffold_sets[_a] & scaffold_sets[_b]
        _overlap_rows.append({
            "pair": f"{_a.upper()} ∩ {_b.upper()}",
            "shared_molecules": len(_shared_smiles),
            "pct_of_smaller": round(
                len(_shared_smiles) / min(len(smiles_sets[_a]), len(smiles_sets[_b])) * 100, 1
            ),
            "shared_scaffolds": len(_shared_scaffolds),
            "pct_scaffolds_of_smaller": round(
                len(_shared_scaffolds) / min(len(scaffold_sets[_a]), len(scaffold_sets[_b])) * 100, 1
            ),
        })

    overlap_df = pl.DataFrame(_overlap_rows)
    return overlap_df, smiles_sets


@app.cell(hide_code=True)
def _(mo, overlap_df):
    mo.vstack([
        mo.md("## Molecule and Scaffold Overlap"),
        mo.ui.table(overlap_df),
        mo.md("""
    Overlap is computed on canonical SMILES (exact structure match) and
    Bemis-Murcko scaffolds (core ring systems). The "pct of smaller" column
    shows overlap as a percentage of the smaller dataset in each pair.

    Substantial scaffold overlap means the chemical spaces are related even
    when exact molecule overlap is limited — relevant context for interpreting
    transfer learning results.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(ENDPOINT_NAMES, mo, np, plt, smiles_sets: dict[str, set[str]]):
    # 3x3 molecule overlap heatmap
    _endpoint_keys = list(ENDPOINT_NAMES.keys())
    _endpoint_labels = [ENDPOINT_NAMES[k] for k in _endpoint_keys]
    _n = len(_endpoint_keys)

    _overlap_matrix = np.zeros((_n, _n), dtype=int)
    for _i, _ki in enumerate(_endpoint_keys):
        for _j, _kj in enumerate(_endpoint_keys):
            if _i == _j:
                _overlap_matrix[_i, _j] = len(smiles_sets[_ki])
            else:
                _overlap_matrix[_i, _j] = len(smiles_sets[_ki] & smiles_sets[_kj])

    _fig_cm, _ax_cm = plt.subplots(figsize=(7, 6))
    _im = _ax_cm.imshow(_overlap_matrix, cmap="Blues")

    # Annotate cells
    for _i in range(_n):
        for _j in range(_n):
            _val = _overlap_matrix[_i, _j]
            _color = "white" if _val > _overlap_matrix.max() * 0.6 else "black"
            _ax_cm.text(_j, _i, f"{_val:,}", ha="center", va="center", color=_color, fontsize=13)

    _ax_cm.set_xticks(range(_n))
    _ax_cm.set_yticks(range(_n))
    _ax_cm.set_xticklabels(_endpoint_labels, rotation=30, ha="right")
    _ax_cm.set_yticklabels(_endpoint_labels)
    _ax_cm.set_title("Molecule Overlap (shared SMILES)")
    plt.colorbar(_im, ax=_ax_cm, label="Count")
    plt.tight_layout()

    mo.vstack([
        mo.md("### Molecule Overlap Matrix"),
        mo.as_html(_fig_cm),
        mo.md("""
    Diagonal: total unique molecules per endpoint. Off-diagonal: number of
    molecules shared between each pair. RLM and PAMPA share nearly all
    molecules (same compound library). HLM has minimal overlap with either.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(datasets: "dict[str, pl.DataFrame]", mo, pl, plt):
    # Correlation of continuous values for shared compounds
    _rlm = datasets["rlm"].select("canonical_smiles", "continuous_value", "continuous_value_censored", "binary_label")
    _pampa = datasets["pampa"].select("canonical_smiles", "continuous_value", "continuous_value_censored", "binary_label")
    _hlm = datasets["hlm"].select("canonical_smiles", "continuous_value", "continuous_value_censored", "binary_label")

    # RLM vs PAMPA (large overlap)
    _rlm_pampa = _rlm.join(_pampa, on="canonical_smiles", suffix="_pampa")
    # Filter to uncensored in both
    _rlm_pampa_unc = _rlm_pampa.filter(
        ~pl.col("continuous_value_censored") & ~pl.col("continuous_value_censored_pampa")
    )

    # RLM vs HLM (small overlap, but let's show it)
    _rlm_hlm = _rlm.join(_hlm, on="canonical_smiles", suffix="_hlm")
    _rlm_hlm_unc = _rlm_hlm.filter(
        ~pl.col("continuous_value_censored") & ~pl.col("continuous_value_censored_hlm")
    )

    _fig3, (_ax_a, _ax_b) = plt.subplots(1, 2, figsize=(12, 5))

    # RLM vs PAMPA
    _ax_a.scatter(
        _rlm_pampa_unc.get_column("continuous_value").to_list(),
        _rlm_pampa_unc.get_column("continuous_value_pampa").to_list(),
        alpha=0.3, s=10,
    )
    _ax_a.set_xlabel("RLM Half-life (min)")
    _ax_a.set_ylabel("PAMPA Permeability (x10⁻⁶ cm/s)")
    _ax_a.set_title(f"RLM vs PAMPA (n={_rlm_pampa_unc.height} shared, uncensored)")

    # RLM vs HLM
    if _rlm_hlm_unc.height > 0:
        _ax_b.scatter(
            _rlm_hlm_unc.get_column("continuous_value").to_list(),
            _rlm_hlm_unc.get_column("continuous_value_hlm").to_list(),
            alpha=0.5, s=20,
        )
        _ax_b.set_xlabel("RLM Half-life (min)")
        _ax_b.set_ylabel("HLM Half-life (min)")
        _ax_b.set_title(f"RLM vs HLM (n={_rlm_hlm_unc.height} shared, uncensored)")
    else:
        _ax_b.text(0.5, 0.5, "No shared uncensored compounds", ha="center", va="center", transform=_ax_b.transAxes)
        _ax_b.set_title("RLM vs HLM")

    plt.tight_layout()

    mo.vstack([
        mo.md("## Correlation of Shared Compounds"),
        mo.as_html(_fig3),
        mo.md(f"""
    - **RLM vs PAMPA**: {_rlm_pampa_unc.height} shared uncensored compounds. These are mechanistically
      unrelated endpoints (metabolic stability vs membrane permeability), so we expect low correlation.
    - **RLM vs HLM**: {_rlm_hlm_unc.height} shared uncensored compounds. Small overlap, but both measure
      microsomal stability so we'd expect positive correlation for the compounds that do overlap.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(datasets: "dict[str, pl.DataFrame]", mo, pl):
    # Binary label agreement for RLM vs PAMPA (large overlap)
    _rlm_p = datasets["rlm"].select("canonical_smiles", pl.col("binary_label").alias("rlm_label"))
    _pampa_p = datasets["pampa"].select("canonical_smiles", pl.col("binary_label").alias("pampa_label"))
    _merged = _rlm_p.join(_pampa_p, on="canonical_smiles").filter(
        pl.col("rlm_label").is_not_null() & pl.col("pampa_label").is_not_null()
    )

    _both_active = _merged.filter((pl.col("rlm_label") == 1) & (pl.col("pampa_label") == 1)).height
    _both_inactive = _merged.filter((pl.col("rlm_label") == 0) & (pl.col("pampa_label") == 0)).height
    _rlm_only = _merged.filter((pl.col("rlm_label") == 1) & (pl.col("pampa_label") == 0)).height
    _pampa_only = _merged.filter((pl.col("rlm_label") == 0) & (pl.col("pampa_label") == 1)).height
    _agreement = round((_both_active + _both_inactive) / _merged.height * 100, 1) if _merged.height > 0 else 0

    _confusion = pl.DataFrame({
        "": ["RLM Active (stable)", "RLM Inactive (unstable)"],
        "PAMPA Active (permeable)": [_both_active, _pampa_only],
        "PAMPA Inactive (impermeable)": [_rlm_only, _both_inactive],
    })

    mo.vstack([
        mo.md("## Binary Label Agreement (RLM vs PAMPA, shared compounds)"),
        mo.ui.table(_confusion),
        mo.md(f"""
    Label agreement: **{_agreement}%** across {_merged.height} shared compounds.

    Since RLM measures metabolic stability and PAMPA measures permeability, low agreement
    is expected — a compound can be metabolically stable but impermeable, or vice versa.
    This confirms that RLM→PAMPA transfer is genuinely "unrelated" from a biological mechanism standpoint.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    import numpy as np
    import pacmap
    from joblib import Parallel, delayed
    from rdkit.Chem import rdFingerprintGenerator

    mo.md("## Chemical Space Embedding (PaCMAP)")
    return Parallel, delayed, np, pacmap, rdFingerprintGenerator


@app.cell(hide_code=True)
def _(
    Chem,
    Parallel,
    datasets: "dict[str, pl.DataFrame]",
    delayed,
    logger,
    np,
    pacmap,
    pl,
    rdFingerprintGenerator,
):
    # Collect all unique SMILES across all endpoints
    _all_smiles_frames = []
    for _key, _df in datasets.items():
        _all_smiles_frames.append(
            _df.select("canonical_smiles").unique()
        )
    all_smiles_df = pl.concat(_all_smiles_frames).unique()
    logger.info(f"Total unique molecules across all endpoints: {all_smiles_df.height}")

    def _smiles_to_morgan_fp(smiles: str) -> tuple[str, np.ndarray] | None:
        """Compute Morgan fingerprint as a numpy array. Returns (smiles, fp) or None."""
        _mol = Chem.MolFromSmiles(smiles)
        if _mol is None:
            return None
        _gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
        _fp = _gen.GetFingerprintAsNumPy(_mol)
        return (smiles, _fp)

    _smiles_list = all_smiles_df.get_column("canonical_smiles").to_list()

    # Parallel fingerprint generation
    _results = Parallel(n_jobs=-1, backend="loky")(
        delayed(_smiles_to_morgan_fp)(smi) for smi in _smiles_list
    )
    _valid = [r for r in _results if r is not None]
    _valid_smiles = [r[0] for r in _valid]
    _fps = [r[1] for r in _valid]

    fp_matrix = np.stack(_fps)
    logger.info(f"Fingerprint matrix shape: {fp_matrix.shape}")

    # PaCMAP embedding to 2D
    pacmap_model = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, random_state=42)
    embedding_2d = pacmap_model.fit_transform(fp_matrix.astype(np.float32))
    logger.info(f"PaCMAP embedding shape: {embedding_2d.shape}")

    # Build a lookup from SMILES -> (x, y) coordinates
    smiles_to_xy = {smi: (embedding_2d[i, 0], embedding_2d[i, 1]) for i, smi in enumerate(_valid_smiles)}
    return all_smiles_df, embedding_2d, fp_matrix, smiles_to_xy


@app.cell(hide_code=True)
def _(
    ENDPOINT_NAMES,
    datasets: "dict[str, pl.DataFrame]",
    embedding_2d,
    mo,
    np,
    plt,
    smiles_to_xy,
):
    # Scatter plots: PaCMAP 2D colored by binary label per endpoint
    # Use shared axis limits so spatial positions are comparable across panels
    _pad = 1.5
    _xlim = (embedding_2d[:, 0].min() - _pad, embedding_2d[:, 0].max() + _pad)
    _ylim = (embedding_2d[:, 1].min() - _pad, embedding_2d[:, 1].max() + _pad)

    _fig_scat, _axes_scat = plt.subplots(1, 3, figsize=(18, 5))

    _label_colors = {1: "#2196F3", 0: "#FF5722", None: "#CCCCCC"}
    _label_names_map = {
        "rlm": {1: "Stable", 0: "Unstable"},
        "hlm": {1: "Stable", 0: "Unstable"},
        "pampa": {1: "Permeable", 0: "Impermeable"},
    }

    for _i, (_key, _df) in enumerate(datasets.items()):
        _smiles = _df.get_column("canonical_smiles").to_list()
        _labels = _df.get_column("binary_label").to_list()

        _xs, _ys, _cs = [], [], []
        for _smi, _lab in zip(_smiles, _labels):
            if _smi in smiles_to_xy:
                _x, _y = smiles_to_xy[_smi]
                _xs.append(_x)
                _ys.append(_y)
                _cs.append(_label_colors.get(_lab, "#CCCCCC"))

        # Plot inactive first (background), then active on top
        _xs_arr, _ys_arr, _cs_arr = np.array(_xs), np.array(_ys), np.array(_cs)
        for _color, _label_val in [("#FF5722", 0), ("#2196F3", 1)]:
            _mask = _cs_arr == _color
            if _mask.any():
                _label_text = _label_names_map[_key][_label_val]
                _axes_scat[_i].scatter(
                    _xs_arr[_mask], _ys_arr[_mask],
                    c=_color, s=4, alpha=0.4, label=_label_text, rasterized=True,
                )

        _axes_scat[_i].set_xlim(_xlim)
        _axes_scat[_i].set_ylim(_ylim)
        _axes_scat[_i].set_title(ENDPOINT_NAMES[_key])
        _axes_scat[_i].set_xlabel("PaCMAP 1")
        _axes_scat[_i].set_ylabel("PaCMAP 2")
        _axes_scat[_i].legend(markerscale=3, fontsize=9)

    _fig_scat.suptitle("Chemical Space (Morgan FP → PaCMAP) Colored by Label", fontsize=14)
    plt.tight_layout()

    mo.vstack([
        mo.md("## PaCMAP Embedding — Label Scatter Plots"),
        mo.as_html(_fig_scat),
        mo.md("""
    Each point is a molecule embedded via PaCMAP from 2048-bit Morgan fingerprints (radius 3).
    All three panels share the same coordinate axes — a single embedding computed across all
    molecules from all endpoints. Colors indicate the binary classification label.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(
    ENDPOINT_NAMES,
    datasets: "dict[str, pl.DataFrame]",
    embedding_2d,
    mo,
    plt,
    smiles_to_xy,
):
    # Hexbin density plots: one column per endpoint, two rows (active / inactive)
    # Shared axis limits across all panels
    _pad = 1.5
    _xlim_hex = (embedding_2d[:, 0].min() - _pad, embedding_2d[:, 0].max() + _pad)
    _ylim_hex = (embedding_2d[:, 1].min() - _pad, embedding_2d[:, 1].max() + _pad)

    _fig_hex, _axes_hex = plt.subplots(2, 3, figsize=(18, 10))

    _row_colors = {0: "Oranges", 1: "Blues"}

    for _col, (_key, _df) in enumerate(datasets.items()):
        _smiles = _df.get_column("canonical_smiles").to_list()
        _labels = _df.get_column("binary_label").to_list()

        for _row, _label_val in enumerate([1, 0]):
            _xs, _ys = [], []
            for _smi, _lab in zip(_smiles, _labels):
                if _lab == _label_val and _smi in smiles_to_xy:
                    _x, _y = smiles_to_xy[_smi]
                    _xs.append(_x)
                    _ys.append(_y)

            _ax = _axes_hex[_row][_col]
            if len(_xs) > 0:
                _hb = _ax.hexbin(
                    _xs, _ys, gridsize=25, cmap=_row_colors[_label_val],
                    mincnt=1, edgecolors="none",
                    extent=(*_xlim_hex, *_ylim_hex),
                )
                plt.colorbar(_hb, ax=_ax, label="Count")

            _label_text = {
                "rlm": {1: "Stable", 0: "Unstable"},
                "hlm": {1: "Stable", 0: "Unstable"},
                "pampa": {1: "Permeable", 0: "Impermeable"},
            }[_key][_label_val]

            _ax.set_xlim(_xlim_hex)
            _ax.set_ylim(_ylim_hex)
            _ax.set_title(f"{ENDPOINT_NAMES[_key]} — {_label_text} (n={len(_xs)})")
            _ax.set_xlabel("PaCMAP 1")
            _ax.set_ylabel("PaCMAP 2")

    _fig_hex.suptitle("Hexbin Density by Label in PaCMAP Space", fontsize=14, y=1.01)
    plt.tight_layout()

    mo.vstack([
        mo.md("## PaCMAP Embedding — Hexbin Density by Label"),
        mo.as_html(_fig_hex),
        mo.md("""
    Top row: active (stable / permeable) compounds. Bottom row: inactive.
    All panels share the same coordinate axes from the single shared PaCMAP embedding.
    Density differences between the two rows reveal regions of chemical space
    enriched for one class — structure-activity relationships that ML models can exploit.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    from sklearn.cluster import KMeans
    from useful_rdkit_utils import GroupKFoldShuffle

    N_REPLICATES = 5
    N_FOLDS = 5
    N_CLUSTERS = 50
    TARGET_ENDPOINTS = ["hlm", "pampa"]
    PRETRAIN_ENDPOINT = "rlm"

    mo.md("""
    ## Split Generation

    Cluster molecules in PaCMAP space via KMeans, then use cluster
    assignments as groups for `GroupKFoldShuffle` (5 replicates x 5 folds
    = 25 splits per endpoint). Splits are saved to disk for reuse
    across all downstream training notebooks.
    """)
    return (
        GroupKFoldShuffle,
        KMeans,
        N_CLUSTERS,
        N_FOLDS,
        N_REPLICATES,
        PRETRAIN_ENDPOINT,
        TARGET_ENDPOINTS,
    )


@app.cell(hide_code=True)
def _(
    ENDPOINT_NAMES,
    GroupKFoldShuffle,
    KMeans,
    N_CLUSTERS,
    N_FOLDS,
    N_REPLICATES,
    all_smiles_df,
    datasets: "dict[str, pl.DataFrame]",
    embedding_2d,
    logger,
    np,
):
    # Build per-endpoint PaCMAP embeddings by looking up from the global embedding
    # all_smiles_df was used to build fp_matrix/embedding_2d; recover the order
    _global_smiles = all_smiles_df.get_column("canonical_smiles").to_list()
    _global_smi_to_idx = {smi: i for i, smi in enumerate(_global_smiles)}

    # For each endpoint, get the PaCMAP coordinates and cluster
    endpoint_splits: dict[str, dict] = {}

    for _key in list(ENDPOINT_NAMES.keys()):
        _df = datasets[_key]
        _smiles = _df.get_column("canonical_smiles").to_list()
        _labels = _df.get_column("binary_label").to_list()

        # Map to global embedding indices
        _idxs = [_global_smi_to_idx[s] for s in _smiles if s in _global_smi_to_idx]
        _valid_smiles = [s for s in _smiles if s in _global_smi_to_idx]
        _valid_labels = [_labels[i] for i, s in enumerate(_smiles) if s in _global_smi_to_idx]
        _emb = embedding_2d[_idxs]
        _fps_idx = _idxs  # indices into global fp_matrix

        # KMeans on PaCMAP embedding
        _kmeans = KMeans(n_clusters=min(N_CLUSTERS, len(_valid_smiles) // 3), random_state=42, n_init=10)
        _cluster_labels = _kmeans.fit_predict(_emb)
        logger.info(f"{_key}: {len(set(_cluster_labels))} clusters from {len(_valid_smiles)} molecules")

        # Generate all fold assignments
        _all_folds = []
        for _rep in range(N_REPLICATES):
            _gkf = GroupKFoldShuffle(n_splits=N_FOLDS, shuffle=True, random_state=_rep * 100)
            _fold_assignment = np.full(len(_valid_smiles), -1, dtype=np.int8)
            for _fold_idx, (_train_idx, _test_idx) in enumerate(_gkf.split(
                np.arange(len(_valid_smiles)),
                np.array(_valid_labels),
                groups=_cluster_labels,
            )):
                _fold_assignment[_test_idx] = _fold_idx
            _all_folds.append(_fold_assignment)

        _folds_matrix = np.stack(_all_folds)  # shape: (N_REPLICATES, n_molecules)

        endpoint_splits[_key] = {
            "smiles": _valid_smiles,
            "labels": np.array(_valid_labels, dtype=np.int8),
            "fp_indices": np.array(_idxs),
            "cluster_labels": _cluster_labels,
            "folds": _folds_matrix,  # (N_REPLICATES, n_molecules)
        }

    logger.info("Split generation complete")
    return (endpoint_splits,)


@app.cell(hide_code=True)
def _(mo):
    from rdkit import DataStructs

    mo.md("""
    ## Fold Quality Assessment

    Two quality checks on the generated splits:

    1. **Chemical distinctness within a round**: For each molecule in the
       test fold, compute Tanimoto distance to its 5 nearest neighbors
       in the other folds (training set) vs within its own fold. Larger
       cross-fold distances mean the splits separate chemically distinct
       regions.

    2. **Replicate variation**: For each fold in replicate A, find the
       fold in replicate B with the highest molecule overlap (Jaccard
       similarity). The fold index is arbitrary, so we compare
       best-matching pairs. Values well below 1.0 confirm the shuffle
       produces meaningfully different partitions.
    """)
    return (DataStructs,)


@app.cell(hide_code=True)
def _(
    Chem,
    DataStructs,
    ENDPOINT_NAMES,
    N_FOLDS,
    endpoint_splits: dict[str, dict],
    logger,
    np,
    pl,
    rdFingerprintGenerator,
):
    # Compute RDKit fingerprint objects for BulkTanimotoSimilarity
    _morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)

    endpoint_rd_fps: dict[str, list] = {}
    for _key in ENDPOINT_NAMES:
        _smiles = endpoint_splits[_key]["smiles"]
        _fps = []
        for _smi in _smiles:
            _mol = Chem.MolFromSmiles(_smi)
            _fps.append(_morgan_gen.GetFingerprint(_mol))
        endpoint_rd_fps[_key] = _fps
        logger.info(f"{_key}: computed {len(_fps)} RDKit fingerprint objects")

    K_NEIGHBORS = 5

    def compute_knn_tanimoto_distances(
        query_fps: list,
        ref_fps: list,
        k: int = K_NEIGHBORS,
    ) -> np.ndarray:
        """For each query FP, find k nearest neighbors in ref set and return distances.

        Args:
            query_fps: List of RDKit fingerprint objects (query set).
            ref_fps: List of RDKit fingerprint objects (reference set).
            k: Number of nearest neighbors.

        Returns:
            Array of shape (len(query_fps), k) with Tanimoto distances (1 - similarity).
        """
        _k = min(k, len(ref_fps))
        _result = np.zeros((len(query_fps), _k))
        for _i, _qfp in enumerate(query_fps):
            _sims = DataStructs.BulkTanimotoSimilarity(_qfp, ref_fps)
            _sims_arr = np.array(_sims)
            # k largest similarities = k smallest distances
            _top_k_sims = np.sort(_sims_arr)[-_k:][::-1]
            _result[_i] = 1.0 - _top_k_sims
        return _result

    # For each endpoint, replicate 0: compare within-fold vs cross-fold 5-NN distances
    fold_quality_data: list[dict] = []

    for _key in ENDPOINT_NAMES:
        _fps = endpoint_rd_fps[_key]
        _folds = endpoint_splits[_key]["folds"][0]  # replicate 0

        for _fold_idx in range(N_FOLDS):
            _test_mask = _folds == _fold_idx
            _train_mask = _folds != _fold_idx
            _test_fps = [_fps[i] for i in range(len(_fps)) if _test_mask[i]]
            _train_fps = [_fps[i] for i in range(len(_fps)) if _train_mask[i]]

            # Cross-fold: 5-NN distances from test to training
            _cross_dists = compute_knn_tanimoto_distances(_test_fps, _train_fps, K_NEIGHBORS)
            # Within-fold: 5-NN distances from test to other test molecules (exclude self)
            _within_result = np.zeros((len(_test_fps), K_NEIGHBORS))
            for _i, _qfp in enumerate(_test_fps):
                _sims = DataStructs.BulkTanimotoSimilarity(_qfp, _test_fps)
                _sims_arr = np.array(_sims)
                _sims_arr[_i] = -1.0  # exclude self
                _k = min(K_NEIGHBORS, len(_sims_arr) - 1)
                _top_k_sims = np.sort(_sims_arr)[-_k:][::-1]
                _within_result[_i, :_k] = 1.0 - _top_k_sims

            for _i in range(len(_test_fps)):
                for _j in range(K_NEIGHBORS):
                    fold_quality_data.append({
                        "endpoint": ENDPOINT_NAMES[_key],
                        "fold": _fold_idx,
                        "comparison": "cross-fold (test to train)",
                        "nn_rank": _j + 1,
                        "tanimoto_distance": _cross_dists[_i, _j],
                    })
                    fold_quality_data.append({
                        "endpoint": ENDPOINT_NAMES[_key],
                        "fold": _fold_idx,
                        "comparison": "within-fold (test to test)",
                        "nn_rank": _j + 1,
                        "tanimoto_distance": _within_result[_i, _j],
                    })

    fold_quality_df = pl.DataFrame(fold_quality_data)
    logger.info(f"Fold quality data: {fold_quality_df.height} rows")
    return (fold_quality_df,)


@app.cell(hide_code=True)
def _(fold_quality_df, mo, pl, plt):
    # Plot within-fold vs cross-fold 5-NN Tanimoto distance distributions
    _fq_pd = fold_quality_df.to_pandas()
    _all_endpoints = ["RLM Stability", "HLM Stability", "PAMPA pH 7.4"]

    _fig_fq, _axes_fq = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for _i, _target in enumerate(_all_endpoints):
        _subset = _fq_pd[_fq_pd["endpoint"] == _target]
        _within = _subset[_subset["comparison"] == "within-fold (test to test)"]["tanimoto_distance"]
        _cross = _subset[_subset["comparison"] == "cross-fold (test to train)"]["tanimoto_distance"]

        _axes_fq[_i].hist(_within, bins=50, alpha=0.6, label="Within fold", color="#2196F3", density=True)
        _axes_fq[_i].hist(_cross, bins=50, alpha=0.6, label="Cross fold", color="#FF5722", density=True)
        _axes_fq[_i].set_title(_target)
        _axes_fq[_i].set_xlabel("Tanimoto Distance (1 - similarity)")
        _axes_fq[_i].set_ylabel("Density" if _i == 0 else "")
        _axes_fq[_i].legend()
        _axes_fq[_i].axvline(_within.median(), color="#2196F3", linestyle="--", alpha=0.8)
        _axes_fq[_i].axvline(_cross.median(), color="#FF5722", linestyle="--", alpha=0.8)

    _fig_fq.suptitle("5-NN Tanimoto Distance: Within-Fold vs Cross-Fold (Replicate 0)", fontsize=14)
    plt.tight_layout()

    _fq_summary = (
        fold_quality_df
        .group_by("endpoint", "comparison")
        .agg(
            pl.col("tanimoto_distance").median().alias("median"),
            pl.col("tanimoto_distance").mean().alias("mean"),
            pl.col("tanimoto_distance").quantile(0.25).alias("q25"),
            pl.col("tanimoto_distance").quantile(0.75).alias("q75"),
        )
        .sort("endpoint", "comparison")
    )

    mo.vstack([
        mo.md("### Chemical Distinctness: Within-Fold vs Cross-Fold 5-NN Distances"),
        mo.as_html(_fig_fq),
        mo.ui.table(_fq_summary),
        mo.md("""
    Dashed lines show medians. If splits are working well, cross-fold
    distances (red) should be shifted right of within-fold distances (blue)
    -- molecules are more chemically distant from the training set than
    from other molecules in their own test fold.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(
    ENDPOINT_NAMES,
    N_FOLDS,
    N_REPLICATES,
    endpoint_splits: dict[str, dict],
    logger,
    np,
    pl,
):
    # Replicate variation: best-match Jaccard overlap between folds across replicates
    _rep_overlap_rows = []

    for _key in ENDPOINT_NAMES:
        _folds_matrix = endpoint_splits[_key]["folds"]

        for _rep_a in range(N_REPLICATES):
            for _rep_b in range(_rep_a + 1, N_REPLICATES):
                _best_jaccards = []
                for _fold_a in range(N_FOLDS):
                    _set_a = set(np.where(_folds_matrix[_rep_a] == _fold_a)[0])
                    _max_jaccard = 0.0
                    _best_fold_b = -1
                    for _fold_b in range(N_FOLDS):
                        _set_b = set(np.where(_folds_matrix[_rep_b] == _fold_b)[0])
                        _intersection = len(_set_a & _set_b)
                        _union = len(_set_a | _set_b)
                        _jaccard = _intersection / _union if _union > 0 else 0.0
                        if _jaccard > _max_jaccard:
                            _max_jaccard = _jaccard
                            _best_fold_b = _fold_b
                    _best_jaccards.append(_max_jaccard)
                    _rep_overlap_rows.append({
                        "endpoint": ENDPOINT_NAMES[_key],
                        "rep_a": _rep_a,
                        "rep_b": _rep_b,
                        "fold_in_a": _fold_a,
                        "best_match_fold_in_b": _best_fold_b,
                        "best_jaccard": _max_jaccard,
                    })

                _mean_j = np.mean(_best_jaccards)
                logger.info(
                    f"{_key} rep {_rep_a} vs {_rep_b}: "
                    f"mean best-match Jaccard = {_mean_j:.3f}"
                )

    rep_overlap_df = pl.DataFrame(_rep_overlap_rows)
    return (rep_overlap_df,)


@app.cell(hide_code=True)
def _(mo, pl, plt, rep_overlap_df):
    _rep_pd = rep_overlap_df.to_pandas()
    _all_endpoints = ["RLM Stability", "HLM Stability", "PAMPA pH 7.4"]

    _fig_rep, _axes_rep = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for _i, _target in enumerate(_all_endpoints):
        _subset = _rep_pd[_rep_pd["endpoint"] == _target]
        _axes_rep[_i].hist(_subset["best_jaccard"], bins=20, edgecolor="black", alpha=0.7, color="#7E57C2")
        _axes_rep[_i].axvline(_subset["best_jaccard"].mean(), color="red", linestyle="--", label=f'Mean = {_subset["best_jaccard"].mean():.3f}')
        _axes_rep[_i].set_title(_target)
        _axes_rep[_i].set_xlabel("Best-Match Jaccard Overlap")
        _axes_rep[_i].set_ylabel("Count" if _i == 0 else "")
        _axes_rep[_i].set_xlim(0, 1)
        _axes_rep[_i].legend()

    _fig_rep.suptitle("Replicate Variation: Best-Match Fold Overlap Across Replicates", fontsize=14)
    plt.tight_layout()

    _rep_summary = (
        rep_overlap_df
        .group_by("endpoint")
        .agg(
            pl.col("best_jaccard").mean().alias("mean_best_jaccard"),
            pl.col("best_jaccard").std().alias("std_best_jaccard"),
            pl.col("best_jaccard").min().alias("min"),
            pl.col("best_jaccard").max().alias("max"),
        )
        .sort("endpoint")
    )

    mo.vstack([
        mo.md("### Replicate Variation: Best-Match Fold Overlap"),
        mo.as_html(_fig_rep),
        mo.ui.table(_rep_summary),
        mo.md("""
    For each fold in replicate A, we find the fold in replicate B with the
    highest molecule overlap (Jaccard similarity). Fold indices are arbitrary,
    so this best-match comparison avoids penalizing index permutations.

    Values well below 1.0 confirm the shuffled replicates produce meaningfully
    different partitions. A mean best-match Jaccard of ~0.2-0.3 means even the
    most similar fold pair across replicates shares only ~20-30% of molecules.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(
    ENDPOINT_NAMES,
    N_FOLDS,
    N_REPLICATES,
    endpoint_splits: dict[str, dict],
    logger,
    pl,
):
    # Fold size and class balance across all replicates
    _fold_balance_rows = []

    for _key in ENDPOINT_NAMES:
        _labels = endpoint_splits[_key]["labels"]
        _folds_matrix = endpoint_splits[_key]["folds"]

        for _rep in range(N_REPLICATES):
            _folds = _folds_matrix[_rep]
            for _fold_idx in range(N_FOLDS):
                _mask = _folds == _fold_idx
                _n_total = _mask.sum()
                _n_pos = (_labels[_mask] == 1).sum()
                _n_neg = (_labels[_mask] == 0).sum()
                _fold_balance_rows.append({
                    "endpoint": ENDPOINT_NAMES[_key],
                    "replicate": _rep,
                    "fold": _fold_idx,
                    "n_total": int(_n_total),
                    "n_positive": int(_n_pos),
                    "n_negative": int(_n_neg),
                    "pct_positive": round(_n_pos / _n_total * 100, 1) if _n_total > 0 else 0.0,
                })

    fold_balance_df = pl.DataFrame(_fold_balance_rows)

    # Assign rank per endpoint by mean_size
    _mean_sizes = (
        fold_balance_df
        .group_by("endpoint", "fold")
        .agg(pl.col("n_total").mean().alias("mean_size"))
    )
    _ranked = (
        _mean_sizes
        .sort("endpoint", "mean_size")
        .with_columns(
            pl.col("fold")
            .cum_count()
            .over("endpoint")
            .alias("size_rank")
        )
    )
    fold_balance_ranked = fold_balance_df.join(
        _ranked.select("endpoint", "fold", "size_rank"),
        on=["endpoint", "fold"],
    )

    logger.info(f"Fold balance data: {fold_balance_ranked.height} rows")
    return (fold_balance_ranked,)


@app.cell(hide_code=True)
def _(fold_balance_ranked, mo, np, plt):
    _fb_pd = fold_balance_ranked.to_pandas()
    _all_endpoints = ["RLM Stability", "HLM Stability", "PAMPA pH 7.4"]

    _fig_fb, _axes_fb = plt.subplots(2, 3, figsize=(18, 10))

    for _col, _target in enumerate(_all_endpoints):
        _sub = _fb_pd[_fb_pd["endpoint"] == _target].copy()

        for _row, (_count_col, _label, _color) in enumerate([
            ("n_positive", "Active", "#2196F3"),
            ("n_negative", "Inactive", "#FF5722"),
        ]):
            _ax = _axes_fb[_row][_col]

            _fold_stats = _sub.groupby("size_rank")[_count_col].agg(["mean", "std"]).reset_index()
            _fold_stats.columns = ["size_rank", "mean", "std"]
            _fold_stats = _fold_stats.sort_values("size_rank")

            _x_pos = _fold_stats["size_rank"].values
            _ax.bar(
                _x_pos, _fold_stats["mean"], yerr=_fold_stats["std"],
                color=_color, alpha=0.4, capsize=4, edgecolor=_color, linewidth=1.2,
                label="Mean +/- SD",
            )

            _jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(_sub))
            _ax.scatter(
                _sub["size_rank"].values + _jitter,
                _sub[_count_col].values,
                color=_color, s=25, alpha=0.7, edgecolor="white", linewidth=0.5,
                zorder=5, label="Individual replicates",
            )

            _ax.set_xlabel("Fold (sorted by size)" if _row == 1 else "")
            _ax.set_ylabel(f"{_label} count")
            _ax.set_title(f"{_target} — {_label}" if _row == 0 else "")
            _ax.set_xticks(_x_pos)
            _ax.set_xticklabels([f"F{int(r)}" for r in _x_pos])
            _ax.legend(fontsize=8)

    _fig_fb.suptitle("Fold Class Balance Across Replicates (sorted by fold size)", fontsize=14)
    plt.tight_layout()

    mo.vstack([
        mo.md("### Fold Size and Class Balance"),
        mo.as_html(_fig_fb),
        mo.md("""
    Bars show the mean count across 5 replicates, error bars show +/- 1 SD.
    Overlaid points are individual replicate values. Folds are sorted by
    total size (smallest to largest). Variation in bar heights across folds
    reflects the unequal cluster sizes from KMeans.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Save Splits and Fingerprints
    """)
    return


@app.cell(hide_code=True)
def _(
    DATA_DIR,
    ENDPOINT_NAMES,
    N_CLUSTERS,
    N_FOLDS,
    N_REPLICATES,
    PRETRAIN_ENDPOINT,
    TARGET_ENDPOINTS,
    all_smiles_df,
    endpoint_splits: dict[str, dict],
    fp_matrix,
    logger,
    mo,
    np,
):
    import json

    # Save the global fingerprint matrix
    np.savez_compressed(
        DATA_DIR / "morgan_fps_2048_r3.npz",
        fp_matrix=fp_matrix,
        smiles=np.array(all_smiles_df.get_column("canonical_smiles").to_list()),
    )
    logger.info(f"Saved global fingerprint matrix: {fp_matrix.shape}")

    # Save per-endpoint splits
    for _key in ENDPOINT_NAMES:
        _split = endpoint_splits[_key]
        np.savez_compressed(
            DATA_DIR / f"{_key}_splits.npz",
            smiles=np.array(_split["smiles"]),
            labels=_split["labels"],
            fp_indices=_split["fp_indices"],
            cluster_labels=_split["cluster_labels"],
            folds=_split["folds"],
        )
        logger.info(
            f"Saved {_key} splits: {len(_split['smiles'])} molecules, "
            f"{_split['folds'].shape[0]} replicates x {N_FOLDS} folds"
        )

    # Save split config for reproducibility
    _config = {
        "n_replicates": N_REPLICATES,
        "n_folds": N_FOLDS,
        "n_clusters": N_CLUSTERS,
        "target_endpoints": TARGET_ENDPOINTS,
        "pretrain_endpoint": PRETRAIN_ENDPOINT,
        "morgan_radius": 3,
        "morgan_nbits": 2048,
        "pacmap_n_components": 2,
        "pacmap_random_state": 42,
        "kmeans_random_state": 42,
    }
    with open(DATA_DIR / "split_config.json", "w") as _f:
        json.dump(_config, _f, indent=2)

    mo.md(f"""
    Saved to `{DATA_DIR}/`:
    - `morgan_fps_2048_r3.npz` — global fingerprint matrix ({fp_matrix.shape[0]} molecules x {fp_matrix.shape[1]} bits)
    - `{{endpoint}}_splits.npz` — per-endpoint fold assignments ({N_REPLICATES} replicates x {N_FOLDS} folds)
    - `split_config.json` — split parameters for reproducibility
    """)
    return


if __name__ == "__main__":
    app.run()
