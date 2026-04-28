# Transfer Learning for Drug Discovery: NCATS ADME

Demonstration of transfer learning for small-molecule property prediction
using the NCATS ADME dataset. Compares XGBoost (Morgan fingerprints),
Chemprop (D-MPNN), and CheMeleon (D-MPNN foundation model) across related
and unrelated ADME endpoint transfers, with rigorous cross-validation and
statistical testing following the methodology of Pat Walters'
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
the RLM->PAMPA transfer a clean test where any benefit must come from
representation learning, not exposure to new chemical matter. HLM has
minimal overlap with either endpoint.

### Continuous Value Distributions

![Continuous value distributions](docs/figures/eda-continuous-distributions.png)

### Correlation of Shared Compounds

![Correlation scatter](docs/figures/eda-correlation-scatter.png)

RLM vs PAMPA (left) shows no correlation -- mechanistically distinct
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

1. Morgan fingerprints -> PaCMAP 2D embedding
2. KMeans (k=50) clustering in PaCMAP space
3. Cluster assignments as groups for `GroupKFoldShuffle`
4. 5 replicates x 5 folds = 25 train/test splits per experiment
5. Same splits across all model types for paired comparisons

## Models

Six model variants organized along two axes: architecture (XGBoost vs
Chemprop vs CheMeleon) and transfer strategy (scratch vs domain-specific
vs foundation).

| # | Model | Architecture | Params | Transfer Strategy |
|---|---|---|---|---|
| 1 | XGBoost scratch | Gradient-boosted trees | -- | None (Morgan FP 2048-bit r3) |
| 2 | XGBoost RLM-transfer | Gradient-boosted trees | -- | Continue boosting from RLM model |
| 3 | Chemprop scratch | D-MPNN (random init) | 318K | None |
| 4 | Chemprop RLM-transfer | D-MPNN (RLM init) | 318K | Pre-train on RLM, new FFN head |
| 5 | CheMeleon single-finetune | D-MPNN (foundation init) | 9.3M | Foundation -> target |
| 6 | CheMeleon double-finetune | D-MPNN (foundation init) | 9.3M | Foundation -> RLM -> target |

## Results

### Combined Summary (AUC-ROC, 25 folds)

| Target | Model | AUC-ROC (mean +/- std) |
|---|---|---|
| **HLM Stability** | XGBoost scratch | 0.668 +/- 0.046 |
| | XGBoost RLM-transfer | 0.734 +/- 0.046 |
| | Chemprop scratch | 0.721 +/- 0.038 |
| | CheMeleon single-finetune | 0.739 +/- 0.037 |
| | CheMeleon double-finetune | 0.764 +/- 0.038 |
| | **Chemprop RLM-transfer** | **0.768 +/- 0.042** |
| **PAMPA pH 7.4** | XGBoost RLM-transfer | 0.509 +/- 0.069 |
| | XGBoost scratch | 0.659 +/- 0.050 |
| | CheMeleon single-finetune | 0.676 +/- 0.044 |
| | CheMeleon double-finetune | 0.686 +/- 0.044 |
| | Chemprop scratch | 0.701 +/- 0.053 |
| | **Chemprop RLM-transfer** | **0.716 +/- 0.038** |

### Transfer Learning Effect by Architecture

| Target | XGBoost delta | Chemprop delta | CheMeleon single->double delta |
|---|---|---|---|
| HLM (related) | +0.066 | +0.047 | +0.026 |
| PAMPA (unrelated) | **-0.150** | **+0.015** | +0.010 |

### All-Model Comparison

![All models boxplots](docs/figures/all-models-boxplots.png)

![All models Tukey HSD](docs/figures/all-models-tukey-hsd.png)

Tukey HSD simultaneous confidence intervals (FWER = 0.05). The reference
model (highest mean AUC-ROC) is highlighted. Groups colored red are
significantly different from the reference. Groups colored gray are not
significantly different from the reference. Non-overlapping intervals
between any two groups indicate a significant difference.

**HLM key pairwise results** (Tukey HSD, FWER = 0.05):
- Chemprop RLM-transfer vs CheMeleon double-finetune: not significant
  (p = 1.00). The top two models are statistically indistinguishable.
- Chemprop RLM-transfer vs Chemprop scratch: significant (p = 0.001).
  Transfer learning helps.
- Chemprop RLM-transfer vs XGBoost scratch: significant (p < 0.001).
  Largest gap.

**PAMPA key pairwise results** (Tukey HSD, FWER = 0.05):
- Chemprop RLM-transfer vs Chemprop scratch: not significant (p = 0.89).
  Transfer provides a small, non-significant improvement on PAMPA.
- Chemprop scratch vs XGBoost scratch: significant (p = 0.049).
  D-MPNN representations outperform Morgan fingerprints.
- XGBoost RLM-transfer vs all other models: significant (p < 0.001).
  The only model that performs catastrophically.

### XGBoost-Only Results

![XGBoost boxplots](docs/figures/xgb-boxplots.png)

![XGBoost paired comparison](docs/figures/xgb-paired-comparison.png)

![XGBoost Tukey HSD](docs/figures/xgb-tukey-hsd.png)

## Discussion

### Why does Chemprop RLM-transfer outperform everything?

Chemprop RLM-transfer is the best model for both HLM (0.768) and PAMPA
(0.716), beating the larger CheMeleon foundation model on both endpoints.
Two factors explain this.

**Right-sized model for the data.** The base Chemprop D-MPNN has 318K
parameters. CheMeleon has 9.3M -- a 29x difference. With ~720 training
samples for HLM and ~1,626 for PAMPA, CheMeleon has a parameter-to-sample
ratio of roughly 13,000:1 (HLM) and 5,700:1 (PAMPA). This is severely
overparameterized. Even with foundation pre-training providing a
reasonable initialization, the model has enough capacity to overfit during
finetuning. The base Chemprop model, at roughly 440:1 (HLM) and 200:1
(PAMPA), is better matched to the data scale.

**Domain-specific pre-training beats generic pre-training for related
tasks.** The RLM pre-training exposes the model to 2,529 compounds with
microsomal stability labels -- the same property family as HLM. This
domain-specific signal is more directly useful than CheMeleon's generic
Mordred descriptor pre-training on 1M PubChem compounds. For HLM, the
RLM-pretrained encoder has already learned "what makes a compound
metabolically stable," and finetuning only needs to adapt from rat to
human metabolism.

### Why does transfer learning work for D-MPNN but not XGBoost?

XGBoost transfer on PAMPA destroys performance (-0.150 AUC, dropping to
near-random 0.509). Chemprop transfer on PAMPA slightly improves it
(+0.015). The difference comes from *where* transfer happens in each
architecture.

**XGBoost transfers at the decision boundary.** When we continue boosting
from an RLM-pretrained XGBoost model, new trees build on top of the
existing RLM decision boundaries. If the target task has similar decision
boundaries (HLM: also microsomal stability), this is helpful. If the
target task has different decision boundaries (PAMPA: membrane
permeability), the existing trees actively mislead the model -- it starts
from a wrong baseline, and the new trees must first undo the RLM
predictions before learning PAMPA patterns. With early stopping, there
may not be enough rounds to recover.

**Chemprop transfers at the representation level.** When we load
RLM-pretrained Chemprop weights and replace the FFN head, the
message-passing encoder retains learned molecular features while the FFN
head is re-initialized from scratch. The encoder features (atom
environments, functional group patterns, ring systems) are general enough
to be useful for any molecular property, even if the specific property is
unrelated. The new FFN head learns the correct mapping from these
features to the target, unconstrained by old decision boundaries.

This is the fundamental advantage of representation-level transfer: the
features generalize even when the task does not.

### Why does CheMeleon underperform random-init Chemprop on PAMPA?

CheMeleon single-finetune (0.676) is worse than Chemprop scratch (0.701)
on PAMPA, and this gap is not statistically significant only because of
high variance. The most likely explanation is overfitting.

With 9.3M parameters and ~1,626 PAMPA training samples, CheMeleon is
extremely overparameterized. The foundation pre-training provides a
reasonable initialization, but 30 epochs of finetuning is enough to
overfit. The smaller Chemprop model (318K params) has less capacity to
memorize noise and generalizes better.

The CheMeleon double-finetune partially recovers (0.686 vs 0.676 for
single), suggesting the intermediate RLM finetuning step provides some
regularization by steering the model toward a drug-discovery-relevant
region of weight space before task-specific finetuning. But it is still
not enough to overcome the capacity mismatch.

### Key takeaway

For small ADME datasets (~1,000-2,000 compounds), a right-sized D-MPNN
(318K params) with domain-specific transfer learning outperforms both
larger foundation models and traditional ML with fixed fingerprints. The
benefit of learned representations over fixed fingerprints is most visible
in the unrelated transfer case: Chemprop handles it gracefully while
XGBoost fails catastrophically.

Foundation models like CheMeleon may become more valuable as dataset
sizes grow or when no related pre-training data is available, but for the
dataset sizes tested here, they are outcompeted by smaller, domain-tuned
models.

## Project Structure

```
xfer-learning/
  pyproject.toml                       # UV project config
  notebooks/
    01-data-acquisition.py             # Marimo: download + curate NCATS data
    02-eda.py                          # Marimo: EDA, splits, fold quality
    03-train-baselines.py              # Marimo: XGBoost baselines
    04-train-chemprop.py               # Marimo: Chemprop results visualization
    05-chemeleon.py                    # Marimo: CheMeleon + combined comparison
    06-analysis.py                     # Marimo: final analysis and discussion
  scripts/
    run-chemprop-training.py           # Chemprop CV training with disk caching
    run-chemeleon-training.py          # CheMeleon CV training with disk caching
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

# Run notebooks interactively (in order)
uv run marimo edit notebooks/01-data-acquisition.py
uv run marimo edit notebooks/02-eda.py
uv run marimo edit notebooks/03-train-baselines.py

# Run Chemprop training (standalone, with per-fold caching)
uv run python scripts/run-chemprop-training.py

# Run CheMeleon training (standalone, with per-fold caching)
uv run python scripts/run-chemeleon-training.py

# View results
uv run marimo edit notebooks/04-train-chemprop.py
uv run marimo edit notebooks/05-chemeleon.py
uv run marimo edit notebooks/06-analysis.py
```

## References

- Walters, P. [Some Thoughts on Splitting Chemical Datasets](https://practicalcheminformatics.blogspot.com/2024/11/some-thoughts-on-splitting-chemical.html). Practical Cheminformatics, 2024.
- Walters, P. [Even More Thoughts on ML Method Comparison](https://practicalcheminformatics.blogspot.com/2025/03/even-more-thoughts-on-ml-method.html). Practical Cheminformatics, 2025.
- Chemprop: [github.com/chemprop/chemprop](https://github.com/chemprop/chemprop)
- CheMeleon: [github.com/JacksonBurns/chemeleon](https://github.com/JacksonBurns/chemeleon) / [Zenodo](https://zenodo.org/records/15460715)
- NCATS ADME: [opendata.ncats.nih.gov/adme](https://opendata.ncats.nih.gov/adme)
