import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # 01 — Data Acquisition

        Download and curate NCATS ADME datasets from PubChem BioAssay for the
        transfer learning demonstration. Three endpoints are used:

        | Role | Endpoint | PubChem AID |
        |---|---|---|
        | Pre-training source | RLM Stability | 1508591 |
        | Related finetune target | HLM Stability | 1963597 |
        | Unrelated finetune target | PAMPA pH 7.4 | 1508612 |

        Data is fetched via the PubChem PUG REST API, which provides SMILES,
        activity outcomes, and continuous measurements for all public compounds.
        """
    )
    return (mo,)


@app.cell
def _():
    from pathlib import Path

    import polars as pl
    from loguru import logger
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize

    DATA_DIR = Path("data")
    DATA_DIR.mkdir(exist_ok=True)

    PUBCHEM_CSV_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid"

    ENDPOINTS: dict[str, dict] = {
        "rlm": {
            "aid": 1508591,
            "name": "RLM Stability (Rat Liver Microsomes)",
            "value_col": "Half-life (minutes)",
            "phenotype_col": "Phenotype",
            "stable_label": "stable",
            "unstable_label": "unstable",
        },
        "hlm": {
            "aid": 1963597,
            "name": "HLM Stability (Human Liver Microsomes)",
            "value_col": "Half-life",
            "phenotype_col": "Phenotype",
            "stable_label": "Stable",
            "unstable_label": "Unstable",
        },
        "pampa": {
            "aid": 1508612,
            "name": "PAMPA pH 7.4 Permeability",
            "value_col": "Permeability",
            "phenotype_col": "Phenotype",
            "stable_label": None,
            "unstable_label": None,
        },
    }
    return (
        PUBCHEM_CSV_BASE,
        Chem,
        DATA_DIR,
        ENDPOINTS,
        Path,
        logger,
        pl,
        rdMolStandardize,
    )


@app.cell
def _(mo):
    mo.md(
        """
        ## Step 1: Download raw CSVs from PubChem

        The PubChem PUG REST endpoint
        `https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{AID}/CSV`
        returns all public data for a bioassay including SMILES. The first few
        rows are metadata (RESULT_TYPE, RESULT_DESCR, RESULT_UNIT) which we
        skip during parsing.
        """
    )
    return


@app.cell
def _(DATA_DIR, ENDPOINTS, PUBCHEM_CSV_BASE, Path, logger, pl):
    from urllib.request import urlretrieve

    raw_frames: dict[str, pl.DataFrame] = {}

    for endpoint_key, endpoint_info in ENDPOINTS.items():
        aid = endpoint_info["aid"]
        csv_path = DATA_DIR / f"raw_{endpoint_key}_AID_{aid}.csv"

        if not csv_path.exists():
            url = f"{PUBCHEM_CSV_BASE}/{aid}/CSV"
            logger.info(f"Downloading {endpoint_info['name']} from {url}")
            urlretrieve(url, csv_path)
            logger.info(f"Saved to {csv_path}")
        else:
            logger.info(f"Using cached {csv_path}")

        # Read the CSV, skipping the 3 metadata rows after the header
        # (RESULT_TYPE, RESULT_DESCR, RESULT_UNIT)
        all_rows = pl.read_csv(csv_path, infer_schema_length=0)
        # Drop the metadata rows — they have "RESULT_TYPE", "RESULT_DESCR",
        # "RESULT_UNIT" in the first column (PUBCHEM_RESULT_TAG)
        metadata_tags = {"RESULT_TYPE", "RESULT_DESCR", "RESULT_UNIT"}
        data_rows = all_rows.filter(~pl.col("PUBCHEM_RESULT_TAG").is_in(metadata_tags))
        raw_frames[endpoint_key] = data_rows
        logger.info(
            f"{endpoint_info['name']}: {data_rows.height} rows, "
            f"{data_rows.width} columns"
        )

    raw_frames
    return endpoint_info, endpoint_key, raw_frames, urlretrieve


@app.cell
def _(mo, raw_frames):
    mo.md(
        f"""
        ## Raw data summary

        | Endpoint | Rows | Columns |
        |---|---|---|
        | RLM | {raw_frames["rlm"].height} | {raw_frames["rlm"].width} |
        | HLM | {raw_frames["hlm"].height} | {raw_frames["hlm"].width} |
        | PAMPA | {raw_frames["pampa"].height} | {raw_frames["pampa"].width} |

        Next: standardize SMILES and extract target values.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Step 2: SMILES standardization

        For each compound we:
        1. Parse the SMILES with RDKit
        2. Strip salts and keep the largest fragment
        3. Canonicalize
        4. Drop molecules that fail sanitization
        """
    )
    return


@app.cell
def _(Chem, rdMolStandardize, pl):
    def standardize_smiles(smiles: str) -> str | None:
        """Parse, desalt, and canonicalize a SMILES string.

        Args:
            smiles: Input SMILES string from PubChem.

        Returns:
            Canonical SMILES string, or None if parsing fails.

        Example:
            >>> standardize_smiles("CC(=O)O.[Na]")
            'CC(=O)O'
            >>> standardize_smiles("invalid") is None
            True
        """
        if not smiles or smiles.strip() == "":
            return None
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            # Strip salts — keep largest fragment
            lfc = rdMolStandardize.LargestFragmentChooser()
            mol = lfc.choose(mol)
            return Chem.MolToSmiles(mol)
        except Exception:
            return None

    def standardize_smiles_column(df: pl.DataFrame, smiles_col: str) -> pl.DataFrame:
        """Add a canonical_smiles column and drop rows where standardization failed.

        Args:
            df: DataFrame with a column containing raw SMILES.
            smiles_col: Name of the SMILES column.

        Returns:
            DataFrame with added 'canonical_smiles' column, rows with
            failed standardization removed.
        """
        return (
            df.with_columns(
                pl.col(smiles_col)
                .map_elements(standardize_smiles, return_dtype=pl.Utf8)
                .alias("canonical_smiles")
            )
            .filter(pl.col("canonical_smiles").is_not_null())
            .unique(subset=["canonical_smiles"], keep="first")
        )

    return standardize_smiles, standardize_smiles_column


@app.cell
def _(mo):
    mo.md(
        """
        ## Step 3: Extract and clean target values

        Each endpoint has:
        - A **continuous value** (half-life in minutes or permeability in
          x10^-6 cm/s) — some entries are censored (e.g., ">30", ">1000").
          We parse numeric values where possible and flag censored values.
        - A **binary classification label** derived from the phenotype column.
        """
    )
    return


@app.cell
def _(pl):
    import re

    def parse_continuous_value(value_str: str) -> tuple[float | None, bool]:
        """Parse a continuous endpoint value that may be censored.

        Args:
            value_str: Raw string from the CSV (e.g., "12.5", ">30", "<0.5").

        Returns:
            Tuple of (numeric_value, is_censored). Returns (None, False)
            if parsing fails entirely.

        Example:
            >>> parse_continuous_value("12.5")
            (12.5, False)
            >>> parse_continuous_value(">30")
            (30.0, True)
            >>> parse_continuous_value("N/F")
            (None, False)
        """
        if not value_str or value_str.strip() in ("", "N/F", "N/A"):
            return None, False
        cleaned = value_str.strip()
        is_censored = False
        if cleaned.startswith(">") or cleaned.startswith("<"):
            is_censored = True
            cleaned = cleaned[1:]
        try:
            return float(cleaned), is_censored
        except ValueError:
            return None, False

    def extract_continuous_values(
        df: pl.DataFrame, value_col: str, new_col: str
    ) -> pl.DataFrame:
        """Extract numeric continuous values and censoring flags from a string column.

        Args:
            df: Input DataFrame.
            value_col: Name of the column with raw string values.
            new_col: Base name for the new columns ({new_col} and {new_col}_censored).

        Returns:
            DataFrame with added numeric value column and censored flag column.
        """
        parsed = df.get_column(value_col).to_list()
        values = []
        censored_flags = []
        for v in parsed:
            numeric_val, is_censored = parse_continuous_value(str(v))
            values.append(numeric_val)
            censored_flags.append(is_censored)

        return df.with_columns(
            pl.Series(name=new_col, values=values, dtype=pl.Float64),
            pl.Series(
                name=f"{new_col}_censored", values=censored_flags, dtype=pl.Boolean
            ),
        )

    return extract_continuous_values, parse_continuous_value, re


@app.cell
def _(mo):
    mo.md(
        """
        ## Step 4: Process each endpoint

        Apply SMILES standardization, extract continuous values, and derive
        binary labels.
        """
    )
    return


@app.cell
def _(
    ENDPOINTS,
    extract_continuous_values,
    logger,
    pl,
    raw_frames,
    standardize_smiles_column,
):
    SMILES_COL = "PUBCHEM_EXT_DATASOURCE_SMILES"

    curated_frames: dict[str, pl.DataFrame] = {}

    for _endpoint_key, _endpoint_info in ENDPOINTS.items():
        _raw = raw_frames[_endpoint_key]
        _n_raw = _raw.height
        logger.info(f"Processing {_endpoint_info['name']} ({_n_raw} raw rows)")

        # Standardize SMILES
        _df = standardize_smiles_column(_raw, SMILES_COL)
        _n_after_smiles = _df.height
        logger.info(
            f"  After SMILES standardization: {_n_after_smiles} "
            f"(dropped {_n_raw - _n_after_smiles})"
        )

        # Extract continuous values
        _value_col = _endpoint_info["value_col"]
        _df = extract_continuous_values(_df, _value_col, "continuous_value")

        # Derive binary label from activity outcome
        # PubChem uses Active/Inactive in PUBCHEM_ACTIVITY_OUTCOME
        _df = _df.with_columns(
            pl.when(pl.col("PUBCHEM_ACTIVITY_OUTCOME") == "Active")
            .then(pl.lit(1))
            .when(pl.col("PUBCHEM_ACTIVITY_OUTCOME") == "Inactive")
            .then(pl.lit(0))
            .otherwise(pl.lit(None))
            .cast(pl.Int8)
            .alias("binary_label")
        )

        # Drop rows with no continuous value and no binary label
        _df = _df.filter(
            pl.col("continuous_value").is_not_null()
            | pl.col("binary_label").is_not_null()
        )
        _n_final = _df.height
        logger.info(f"  Final: {_n_final} rows with valid targets")

        # Select and rename columns for a clean output schema
        _df = _df.select(
            pl.col("PUBCHEM_CID").cast(pl.Int64).alias("pubchem_cid"),
            pl.col("canonical_smiles"),
            pl.col("continuous_value"),
            pl.col("continuous_value_censored"),
            pl.col("binary_label"),
            pl.col("PUBCHEM_ACTIVITY_OUTCOME").alias("activity_outcome"),
        )

        curated_frames[_endpoint_key] = _df

    curated_frames
    return SMILES_COL, curated_frames


@app.cell
def _(curated_frames, mo, pl):
    _summary_rows = []
    for _key, _df in curated_frames.items():
        _n_total = _df.height
        _n_with_continuous = _df.filter(pl.col("continuous_value").is_not_null()).height
        _n_censored = _df.filter(pl.col("continuous_value_censored")).height
        _n_active = _df.filter(pl.col("binary_label") == 1).height
        _n_inactive = _df.filter(pl.col("binary_label") == 0).height
        _summary_rows.append(
            {
                "endpoint": _key,
                "total": _n_total,
                "has_continuous": _n_with_continuous,
                "censored": _n_censored,
                "active": _n_active,
                "inactive": _n_inactive,
                "class_balance": (
                    f"{_n_active / _n_total:.1%}" if _n_total > 0 else "N/A"
                ),
            }
        )

    _summary_df = pl.DataFrame(_summary_rows)

    mo.vstack(
        [
            mo.md("## Curated data summary"),
            mo.ui.table(_summary_df),
            mo.md(
                """
                **Notes**:
                - *censored* = values like ">30" or ">1000" where the true value
                  exceeds the assay range. The numeric part is kept, flagged as
                  censored.
                - *active* = PubChem "Active" outcome. For RLM/HLM this means
                  "stable" (t1/2 > 30 min). For PAMPA this means "moderate/high
                  permeability" (>10 x10^-6 cm/s).
                """
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Step 5: Save curated datasets

        Save each endpoint as a parquet file for downstream use.
        """
    )
    return


@app.cell
def _(DATA_DIR, curated_frames, logger, mo):
    _saved_paths = []
    for _key, _df in curated_frames.items():
        _path = DATA_DIR / f"{_key}_curated.parquet"
        _df.write_parquet(_path)
        _saved_paths.append(str(_path))
        logger.info(f"Saved {_key} to {_path} ({_df.height} rows)")

    mo.md("Saved curated datasets:\n" + "\n".join(f"- `{p}`" for p in _saved_paths))
    return


@app.cell
def _(curated_frames, mo):
    mo.md(
        """
        ## Preview: first 10 rows of each endpoint
        """
    )
    return


@app.cell
def _(curated_frames, mo):
    _tabs = {}
    for _key, _df in curated_frames.items():
        _tabs[_key.upper()] = mo.ui.table(_df.head(10))

    mo.ui.tabs(_tabs)
    return


if __name__ == "__main__":
    app.run()
