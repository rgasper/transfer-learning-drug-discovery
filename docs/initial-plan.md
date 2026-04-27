# Transfer Learning for Drug Discovery: NCATS ADME Demonstration

## Goal

Demonstrate transfer learning for small-molecule property prediction using the
NCATS ADME dataset, the Chemprop (D-MPNN) framework, and the CheMeleon
foundation model. The demonstration should produce quantifiable evidence that:

1. Pre-training on a related endpoint (RLM -> HLM) yields measurably better
   models than training from scratch.
2. Pre-training on an unrelated endpoint (RLM -> PAMPA) yields less benefit,
   but may still outperform scratch.
3. These effects hold for both Chemprop (learned representations) and XGBoost
   (fixed fingerprints).
4. A general-purpose foundation model (CheMeleon) provides a strong starting
   point, and domain-specific pre-training on top of it can further improve
   performance.

Evaluation follows the methodology described in Pat Walters' Practical
Cheminformatics blog posts:
- [Splitting chemical data](https://practicalcheminformatics.blogspot.com/2024/11/some-thoughts-on-splitting-chemical.html)
- [ML method comparison](https://practicalcheminformatics.blogspot.com/2025/03/even-more-thoughts-on-ml-method.html)

---

## Dataset: NCATS ADME

All data comes from the NCATS OpenData portal. The raw data is deposited in
PubChem as bioassay records.

### Endpoints by category

**Metabolic Stability** (rate of enzymatic degradation in liver tissue)

| Endpoint | PubChem AID | ~Training Size | Continuous Value | Classification Cutoff |
|---|---|---|---|---|
| RLM Stability (Rat Liver Microsomes) | 1508591 | 35,019 | Half-life (min) | Stable (t1/2 > 30 min) vs Unstable |
| HLM Stability (Human Liver Microsomes) | 1963597 | 7,000 | Half-life (min) | Stable vs Unstable (same cutoff) |
| HLC Stability (Human Liver Cytosol) | — | smaller | Half-life | Stable vs Unstable |

**Permeability** (passive diffusion across membranes)

| Endpoint | PubChem AID | ~Training Size | Continuous Value | Classification Cutoff |
|---|---|---|---|---|
| PAMPA pH 7.4 (GI tract) | 1508612 | 25,000+ | Permeability (x10^-6 cm/s) | Low (<10) vs Moderate/High (>10) |
| PAMPA pH 5.0 (GI tract, acidic) | 1645871 | 5,227 | Same | Same |
| PAMPA-BBB (Blood-Brain Barrier) | 1845228 | 1,958 | Same | Same |

**Solubility**

| Endpoint | PubChem AID | ~Training Size | Continuous Value | Classification Cutoff |
|---|---|---|---|---|
| Kinetic Aqueous Solubility | 1645848 | 36,171 | Solubility (ug/mL) | Low (<10 ug/mL) vs Moderate/High |

**CYP450 Inhibition** (drug-drug interaction risk)

| Endpoint | Classification |
|---|---|
| CYP2C9, CYP2D6, CYP3A4 | Inhibitor vs Non-inhibitor |

### Endpoints used in this study

| Role | Endpoint | Rationale |
|---|---|---|
| **Pre-training source** | RLM Stability | Largest dataset (~35k). Single lab, single protocol. |
| **Related finetune target** | HLM Stability | Same assay type (microsomal stability), same cutoff. The HLM publication explicitly notes "strong correlation between RLM and HLM data." |
| **Unrelated finetune target** | PAMPA pH 7.4 | Mechanistically distinct (passive membrane permeability vs enzymatic metabolism). |

---

## Data Acquisition and Curation

Download CSVs from the PubChem BioAssay data tables:
- RLM: `AID_1508591_datatable_all.csv`
- HLM: `AID_1963597_datatable_all.csv`
- PAMPA 7.4: `AID_1508612_datatable_all.csv`

Curation pipeline:
1. Resolve PubChem CIDs to canonical SMILES via the PubChem PUG REST API.
2. Standardize SMILES with RDKit: canonicalize, strip salts, remove fragments,
   remove molecules that fail sanitization.
3. Deduplicate by canonical SMILES (keep first occurrence).
4. Drop rows with missing structures or missing endpoint values.
5. Extract both the continuous measurement (half-life or permeability) and the
   binary classification label.
6. Store cleaned data as parquet files with dataframely-validated polars schemas.

### Overlap analysis

As part of EDA, compute and report:
- Number of unique molecules per endpoint.
- Number of molecules shared between each pair of endpoints (by canonical SMILES).
- Number of Murcko scaffolds shared between each pair.
- Correlation of continuous values for overlapping molecules (RLM half-life vs
  HLM half-life for shared compounds, etc.).

This contextualizes the transfer learning results even though overlapping
molecules are not required for the transfer learning approach itself.

---

## Splitting Strategy

Following Walters' recommendation, we use **UMAP-based clustering splits**
rather than Murcko scaffold splits. Walters shows that scaffold splits are poor
proxies for realistic temporal splits, while UMAP clustering better separates
structurally distinct chemical neighborhoods.

Procedure:
1. Compute Morgan fingerprints (2048-bit, radius 3) for all molecules.
2. Reduce to 2D via UMAP.
3. Cluster the UMAP embedding (k-means or HDBSCAN).
4. Use cluster assignments as group labels for `GroupKFoldShuffle`.
5. Run **5 replicates x 5 folds = 25 train/test splits** per experiment.
6. The same 25 splits are used across all model types for each endpoint, enabling
   paired statistical comparisons.

Note: the pre-training step on RLM uses the full RLM dataset (no CV). A single
checkpoint is produced and re-used across all finetune folds.

---

## Models and Experiments

### Model matrix

For each target endpoint (HLM, PAMPA 7.4):

| # | Model | Architecture | Features | Training |
|---|---|---|---|---|
| 1 | XGBoost from scratch | Gradient-boosted trees | Morgan FP 2048-bit r3 | Target endpoint only |
| 2 | XGBoost pretrain-RLM | Gradient-boosted trees | Morgan FP 2048-bit r3 | Pre-trained on RLM, continue boosting on target |
| 3 | Chemprop from scratch | D-MPNN (random init) | Learned (SMILES input) | Target endpoint only |
| 4 | Chemprop pretrain-RLM | D-MPNN (RLM init) | Learned (SMILES input) | Pre-trained on RLM, finetune on target |
| 5 | CheMeleon single-finetune | D-MPNN (CheMeleon init) | Learned (SMILES input) | Foundation weights, finetune on target only |
| 6 | CheMeleon double-finetune | D-MPNN (CheMeleon init) | Learned (SMILES input) | Foundation weights -> finetune on RLM -> finetune on target |

This gives three axes of comparison:
- **Architecture**: XGBoost (fixed fingerprints) vs D-MPNN (learned representations)
- **Transfer source**: none (scratch) vs domain-specific (RLM) vs foundation (CheMeleon)
- **Transfer depth**: CheMeleon single-finetune (foundation -> target) vs
  CheMeleon double-finetune (foundation -> RLM -> target), testing whether
  stacking domain-specific pre-training on top of foundation weights helps

### XGBoost transfer learning

XGBoost natively supports continued training. A model trained on RLM is saved,
then passed via the `xgb_model` parameter to `xgb.train()` on the target data.
This continues boosting from the existing ensemble of trees.

### Chemprop transfer learning

Following the Chemprop v2 transfer learning API:
1. Train a Chemprop model on the full RLM dataset and save the checkpoint.
2. For each finetune fold, load the checkpoint and initialize a new model.
3. Optionally freeze the message-passing layers and only train the FFN head
   (or unfreeze everything — we can test both).

### CheMeleon foundation model

CheMeleon is a D-MPNN foundation model pre-trained on 1M PubChem molecules to
predict Mordred molecular descriptors. It uses a large BondMessagePassing
architecture (2048-dim hidden, ~9.3M params). The pre-trained message-passing
weights are downloaded from Zenodo and used to initialize the MPNN encoder.
A new FFN head is attached for the target task.

Two CheMeleon variants are tested:
1. **Single-finetune**: CheMeleon weights -> finetune directly on the target
   endpoint. This tests whether a generic chemistry foundation is sufficient.
2. **Double-finetune**: CheMeleon weights -> finetune on RLM (full dataset) ->
   finetune on the target endpoint. This tests whether layering domain-specific
   pre-training on top of the foundation model provides additional benefit.

The CheMeleon checkpoint is loaded via:
```python
from chemprop import featurizers, nn
import torch

chemeleon_mp = torch.load("chemeleon_mp.pt", weights_only=True)
mp = nn.BondMessagePassing(**chemeleon_mp["hyper_parameters"])
mp.load_state_dict(chemeleon_mp["state_dict"])
```

The FFN head must be initialized with `input_dim=mp.output_dim` (2048) to
match the CheMeleon encoder's output dimension.

---

## Evaluation

### Metrics

Per fold, compute:
- **Classification**: AUC-ROC, Average Precision (AP)
- **Regression** (against continuous endpoint values): R-squared, RMSE

Classification metrics can be computed against the binary label directly, or
against the continuous value treated as a regression target (probability of
class membership).

### Statistical comparison

Following Walters' blog methodology:

1. **Tukey HSD** across all six model types per endpoint. Visualized as the
   characteristic confidence-interval plot (blue = significantly better,
   grey = no significant difference, red = significantly worse).
2. **Paired difference plots** for head-to-head comparisons. Lines connect the
   same CV fold across two models, colored by which model wins. Header colored
   by paired t-test significance.
3. **Boxplots** of metric distributions across 25 folds.

---

## Project Structure

```
xfer-learning/
  pyproject.toml                  # UV project config, all deps
  notebooks/
    01-data-acquisition.py        # Marimo: download and curate NCATS data
    02-eda.py                     # Marimo: EDA, overlap analysis, distributions
    03-train-baselines.py         # Marimo: XGBoost + Morgan FP baselines
    04-train-chemprop.py          # Marimo: Chemprop from-scratch models
    05-transfer-learning.py       # Marimo: pretrain RLM, finetune HLM/PAMPA
    06-analysis.py                # Marimo: Tukey HSD, paired plots, comparison
  src/
    xfer_learning/
      __init__.py
      data.py                     # Data loading, curation, SMILES standardization
      splitting.py                # UMAP clustering splits, GroupKFoldShuffle
      fingerprints.py             # Morgan FP generation (2048-bit, radius 3)
      training.py                 # XGBoost + Chemprop training wrappers
      evaluation.py               # Metrics, Tukey HSD, paired plots
      exceptions.py               # Custom exception hierarchy
  data/                           # Downloaded/processed data (gitignored)
  docs/
    initial-plan.md               # This document
```

### Tooling

| Tool | Role |
|---|---|
| UV | Dependency management and virtual environment |
| Marimo | Interactive notebooks (.py files, git-friendly) |
| Polars | DataFrame operations (with dataframely schemas) |
| Chemprop v2 | D-MPNN model training, transfer learning |
| CheMeleon | Foundation model (pre-trained D-MPNN weights from Zenodo) |
| XGBoost | Traditional ML baseline with continued training |
| RDKit | Molecular processing, fingerprints, scaffolds |
| UMAP + HDBSCAN/k-means | Splitting strategy |
| useful_rdkit_utils | GroupKFoldShuffle |
| statsmodels | Tukey HSD statistical tests |
| scikit-learn | AUC-ROC, AP, R-squared, train/test utilities |

---

## Expected Outcomes

The demonstration should show:

1. **RLM -> HLM (related transfer)**: Both Chemprop and XGBoost benefit
   meaningfully from pre-training. Chemprop may benefit more due to learning
   transferable molecular representations (not just additive tree corrections).
2. **RLM -> PAMPA (unrelated transfer)**: Smaller or negligible benefit. If the
   learned representations capture general molecular properties, there may still
   be some improvement over scratch — but less than the related case.
3. **Chemprop vs XGBoost**: Chemprop's learned representations may transfer
   better than XGBoost's fixed fingerprints + continued boosting, especially
   for the related endpoint pair.
4. **CheMeleon single-finetune**: The foundation model should provide a strong
   starting point for both endpoints, likely outperforming random-init Chemprop
   from scratch, especially on the smaller HLM dataset (~7k).
5. **CheMeleon double-finetune**: Stacking domain-specific RLM pre-training on
   top of CheMeleon should help for HLM (related) but may not help (or could
   hurt) for PAMPA (unrelated), since the RLM-specific representation may
   overwrite useful general features.
6. **Statistical rigor**: 25-fold repeated CV with Tukey HSD provides confidence
   that observed differences are real, not noise.
