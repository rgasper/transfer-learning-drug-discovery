# Transfer Learning for Drug Discovery: NCATS ADME

Demonstration of transfer learning for small-molecule property prediction
using the NCATS ADME dataset. Compares XGBoost (Morgan fingerprints) and
Chemprop (D-MPNN learned representations) across related and unrelated
ADME endpoint transfers, with rigorous cross-validation and statistical
testing following the methodology of Pat Walters'
[Practical Cheminformatics](https://practicalcheminformatics.blogspot.com/) blog.

## Dataset

Three NCATS ADME endpoints curated from PubChem BioAssay (public subsets):

| Endpoint | Role | Compounds | Active | Inactive | Class Balance |
|---|---|---|---|---|---|
| RLM Stability | Pre-training source | 2,529 | 754 (stable) | 1,775 (unstable) | 30% / 70% |
| HLM Stability | Related finetune target | 900 | 542 (stable) | 358 (unstable) | 60% / 40% |
| PAMPA pH 7.4 | Unrelated finetune target | 2,033 | 1,738 (permeable) | 295 (impermeable) | 86% / 14% |

![Class balance](docs/figures/eda-class-balance.png)

### Molecule Overlap

| Pair | Shared Molecules | % of Smaller Set | Shared Scaffolds | % of Smaller Set |
|---|---|---|---|---|
| RLM ∩ HLM | 50 | 5.6% | 94 | 12.9% |
| RLM ∩ PAMPA | 2,023 | 99.5% | 1,385 | 99.6% |
| HLM ∩ PAMPA | 41 | 4.6% | 84 | 11.6% |

RLM and PAMPA share 99.5% of molecules (same compound library), making
the RLM→PAMPA transfer a clean test where any benefit must come from
representation learning, not exposure to new chemical matter. HLM has
minimal overlap with either endpoint.

### Continuous Value Distributions

![Continuous value distributions](docs/figures/eda-continuous-distributions.png)

### Correlation of Shared Compounds

![Correlation scatter](docs/figures/eda-correlation-scatter.png)

RLM vs PAMPA (left) shows no correlation — mechanistically distinct
endpoints. RLM vs HLM (right) shows a positive trend for the small
number of shared compounds.

### Chemical Space (PaCMAP)

Morgan fingerprints (2048-bit, radius 3) embedded to 2D via PaCMAP.
A single shared embedding across all endpoints.

![PaCMAP scatter by label](docs/figures/eda-pacmap-scatter.png)

![PaCMAP hexbin density](docs/figures/eda-pacmap-hexbin.png)

## Splitting Strategy

Following [Walters' methodology](https://practicalcheminformatics.blogspot.com/2024/11/some-thoughts-on-splitting-chemical.html),
PaCMAP-based clustering splits are used instead of Murcko scaffold
splits:

1. Morgan fingerprints → PaCMAP 2D embedding
2. KMeans (k=50) clustering in PaCMAP space
3. Cluster assignments as groups for `GroupKFoldShuffle`
4. 5 replicates x 5 folds = 25 train/test splits per experiment
5. Same splits across all model types for paired comparisons

## Results

### Combined Summary (AUC-ROC, 25 folds)

| Target | Model | AUC-ROC (mean +/- std) |
|---|---|---|
| **HLM Stability** | XGBoost scratch | 0.664 +/- 0.038 |
| | XGBoost RLM-transfer | 0.732 +/- 0.037 |
| | Chemprop scratch | 0.721 +/- 0.038 |
| | **Chemprop RLM-transfer** | **0.768 +/- 0.042** |
| **PAMPA pH 7.4** | XGBoost scratch | 0.679 +/- 0.049 |
| | XGBoost RLM-transfer | 0.496 +/- 0.067 |
| | Chemprop scratch | 0.701 +/- 0.053 |
| | **Chemprop RLM-transfer** | **0.716 +/- 0.038** |

### Transfer Learning Effect by Architecture

| Target | XGBoost transfer delta | Chemprop transfer delta |
|---|---|---|
| HLM (related) | +0.068 | +0.047 |
| PAMPA (unrelated) | **-0.183** | **+0.015** |

### XGBoost Results

![XGBoost boxplots](docs/figures/xgb-boxplots.png)

![XGBoost paired comparison](docs/figures/xgb-paired-comparison.png)

![XGBoost Tukey HSD](docs/figures/xgb-tukey-hsd.png)

**HLM**: Scratch vs RLM-transfer mean difference = -0.068, p-adj < 0.001.
Transfer significantly better.

**PAMPA**: Scratch vs RLM-transfer mean difference = +0.183, p-adj < 0.001.
Scratch significantly better — transfer actively hurts.

## Interpretation

**HLM (related transfer): both architectures benefit from transfer.**
RLM and HLM both measure microsomal stability (rat vs human), so
structural patterns learned from RLM data are relevant to HLM. XGBoost
gains +0.068 AUC, Chemprop gains +0.047 AUC. Chemprop from scratch
already outperforms XGBoost from scratch (0.721 vs 0.664), suggesting
the D-MPNN encoder learns better molecular features than Morgan
fingerprints even without transfer.

**PAMPA (unrelated transfer): the critical difference.**
XGBoost RLM-transfer *destroys* PAMPA performance (-0.183 AUC, dropping
to near-random 0.496). The continued boosting from RLM-specialized
decision trees actively misleads on a mechanistically different task.
Chemprop RLM-transfer, by contrast, *slightly improves* PAMPA (+0.015
AUC). The D-MPNN encoder learns molecular representations that are
general enough to transfer even to unrelated endpoints — the key
advantage of learned representations over fixed fingerprints.

This demonstrates the fundamental difference between transfer at the
decision-boundary level (XGBoost: can only help if the decision boundary
is similar) vs transfer at the representation level (Chemprop: learns
molecular features that generalize across tasks).

## Project Structure

```
xfer-learning/
  pyproject.toml                       # UV project config
  notebooks/
    01-data-acquisition.py             # Marimo: download + curate NCATS data
    02-eda.py                          # Marimo: EDA, splits, fold quality
    03-train-baselines.py              # Marimo: XGBoost baselines
    04-train-chemprop.py               # Marimo: Chemprop results visualization
  scripts/
    run-chemprop-training.py           # Chemprop CV training with disk caching
  src/xfer_learning/                   # Package (placeholder)
  data/                                # Downloaded/processed data (gitignored)
  docs/
    initial-plan.md                    # Experiment design document
    figures/                           # Exported plots
```

## Running

```bash
# Install dependencies
uv sync

# Run notebooks interactively
uv run marimo edit notebooks/01-data-acquisition.py
uv run marimo edit notebooks/02-eda.py
uv run marimo edit notebooks/03-train-baselines.py

# Run Chemprop training (standalone, with caching)
uv run python scripts/run-chemprop-training.py

# View Chemprop results
uv run marimo edit notebooks/04-train-chemprop.py
```

## References

- Walters, P. [Some Thoughts on Splitting Chemical Datasets](https://practicalcheminformatics.blogspot.com/2024/11/some-thoughts-on-splitting-chemical.html). Practical Cheminformatics, 2024.
- Walters, P. [Even More Thoughts on ML Method Comparison](https://practicalcheminformatics.blogspot.com/2025/03/even-more-thoughts-on-ml-method.html). Practical Cheminformatics, 2025.
- Chemprop: [github.com/chemprop/chemprop](https://github.com/chemprop/chemprop)
- NCATS ADME: [opendata.ncats.nih.gov/adme](https://opendata.ncats.nih.gov/adme)
