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

    return Chem, ENDPOINT_NAMES, MurckoScaffold, datasets, pl, plt


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

    return (overlap_df,)


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


if __name__ == "__main__":
    app.run()
