# Findings: EDA and XGBoost Baselines

## Dataset Summary

Three NCATS ADME endpoints were curated from PubChem BioAssay deposits.
These are the publicly released subsets; the full internal NCATS training
sets (10-35k compounds) are not publicly available.

| Endpoint | Role | Compounds | Active | Inactive | Class Balance |
|---|---|---|---|---|---|
| RLM Stability | Pre-training source | 2,529 | 754 (stable) | 1,775 (unstable) | 30% / 70% |
| HLM Stability | Related finetune target | 900 | 542 (stable) | 358 (unstable) | 60% / 40% |
| PAMPA pH 7.4 | Unrelated finetune target | 2,033 | 1,738 (permeable) | 295 (impermeable) | 86% / 14% |

All three endpoints provide both a binary classification label and a
continuous measurement (half-life in minutes for RLM/HLM, permeability in
x10^-6 cm/s for PAMPA). Some values are right-censored (e.g., ">30 min"
for RLM stable compounds, ">1000" for highly permeable PAMPA compounds).

![Class balance](figures/eda-class-balance.png)

## Molecule Overlap

| Pair | Shared Molecules | % of Smaller Set | Shared Scaffolds | % of Smaller Set |
|---|---|---|---|---|
| RLM ∩ HLM | 50 | 5.6% | 94 | 12.9% |
| RLM ∩ PAMPA | 2,023 | 99.5% | 1,385 | 99.6% |
| HLM ∩ PAMPA | 41 | 4.6% | 84 | 11.6% |

RLM and PAMPA were tested on essentially the same compound library (99.5%
molecule overlap). This makes the RLM→PAMPA transfer experiment a
particularly clean test: any benefit from transfer must come from
representation learning, not from exposure to new chemical matter.

HLM has very little overlap with either RLM (5.6%) or PAMPA (4.6%), so
the RLM→HLM transfer experiment tests whether structural knowledge
generalizes from rat to human microsomal stability across distinct
compound sets.

Binary label agreement between RLM and PAMPA for shared compounds is low,
confirming these are mechanistically distinct endpoints — a compound's
metabolic stability is not predictive of its membrane permeability.

## Continuous Value Distributions

![Continuous value distributions](figures/eda-continuous-distributions.png)

Histograms of uncensored continuous values for each endpoint. RLM
half-lives are heavily right-skewed (many unstable compounds with short
half-lives). HLM half-lives show a bimodal distribution. PAMPA
permeability spans several orders of magnitude.

## Correlation of Shared Compounds

![Correlation scatter](figures/eda-correlation-scatter.png)

Left: RLM vs PAMPA for shared uncensored compounds — no visible
correlation, confirming these are mechanistically distinct. Right: RLM vs
HLM for the small number of shared compounds — limited data but a
positive trend is visible.

## Chemical Space Visualization

Morgan fingerprints (2048-bit, radius 3) were computed for all 3,388
unique molecules across the three endpoints and embedded into 2D via
PaCMAP. A single shared embedding was used so spatial positions are
directly comparable across endpoint panels.

![PaCMAP scatter by label](figures/eda-pacmap-scatter.png)

![PaCMAP hexbin density](figures/eda-pacmap-hexbin.png)

Key observations:

- The chemical space is well-covered across all three endpoints, with RLM
  and PAMPA occupying nearly identical regions (expected given 99.5%
  molecule overlap).
- HLM compounds occupy a subset of the overall space, concentrated in
  certain structural neighborhoods.
- Label distributions show visible spatial clustering — certain regions of
  chemical space are enriched for active or inactive compounds — which is
  a prerequisite for structure-based ML models to work.

## Splitting Strategy

Following Walters' methodology, UMAP-based clustering splits were used
instead of Murcko scaffold splits. For each endpoint:

1. Morgan fingerprints (2048-bit, radius 3) were reduced to 2D via UMAP.
2. KMeans (k=50) was applied to the UMAP embedding.
3. Cluster assignments were used as group labels for `GroupKFoldShuffle`.
4. 5 replicates x 5 folds = 25 train/test splits per experiment.
5. The same splits were used across all model types for paired comparisons.

## XGBoost Baseline Results

XGBoost models were trained on Morgan fingerprints (2048-bit, radius 3)
with two variants:

- **From scratch**: trained only on the target endpoint's training fold.
- **RLM transfer**: a model pre-trained on the full RLM dataset (200
  boosting rounds), then continued boosting on the target endpoint's
  training fold (up to 200 additional rounds with early stopping).

### AUC-ROC Summary (25 folds)

| Target | Model | AUC-ROC (mean +/- std) | Avg Precision (mean +/- std) |
|---|---|---|---|
| HLM Stability | XGBoost scratch | 0.664 +/- 0.038 | 0.740 +/- 0.042 |
| HLM Stability | XGBoost RLM-transfer | **0.732 +/- 0.037** | **0.791 +/- 0.036** |
| PAMPA pH 7.4 | XGBoost scratch | **0.679 +/- 0.049** | **0.916 +/- 0.021** |
| PAMPA pH 7.4 | XGBoost RLM-transfer | 0.496 +/- 0.067 | 0.855 +/- 0.032 |

### AUC-ROC Distributions

![XGBoost boxplots](figures/xgb-boxplots.png)

### Paired Fold Comparison

![XGBoost paired comparison](figures/xgb-paired-comparison.png)

Lines connect the same CV fold across the two models. Blue lines indicate
the transfer model won on that fold, red lines indicate scratch won.
Title color reflects paired t-test significance.

### Tukey HSD Statistical Test

![Tukey HSD](figures/xgb-tukey-hsd.png)

Tukey HSD simultaneous confidence intervals (FWER-controlled at
alpha = 0.05). Non-overlapping intervals indicate a statistically
significant difference between models.

**HLM Stability**: Scratch vs RLM-transfer mean difference = -0.068,
p-adj < 0.001, reject H0. Transfer is significantly better.

**PAMPA pH 7.4**: Scratch vs RLM-transfer mean difference = +0.183,
p-adj < 0.001, reject H0. Scratch is significantly better (transfer
hurts).

Both differences are statistically significant after multiple comparison
correction.

### Interpretation

**HLM (related transfer): transfer helps.**
RLM→HLM transfer improves AUC-ROC by +0.068 (0.664 → 0.732). This is
expected — both endpoints measure microsomal stability (rat vs human),
so the structural patterns learned from RLM data are directly relevant
to HLM prediction. The improvement is consistent across all 25 folds in
the paired comparison.

**PAMPA (unrelated transfer): transfer hurts.**
RLM→PAMPA transfer *decreases* AUC-ROC by -0.183 (0.679 → 0.496, near
random). The continued boosting from RLM-specialized trees actively
misleads the model on a mechanistically different task. This is the
expected failure mode for XGBoost transfer — the fixed fingerprint
representation means transfer can only happen through the tree ensemble,
and RLM-optimized decision boundaries are counterproductive for
predicting membrane permeability.

**Takeaway for the demonstration:**
The XGBoost results establish a clear baseline showing that transfer
learning sensitivity to endpoint relatedness is real and measurable.
The key question for the Chemprop and CheMeleon experiments is whether
learned molecular representations can transfer more gracefully —
particularly whether the D-MPNN encoder learns features that are useful
even for unrelated endpoints, unlike XGBoost's fixed fingerprints where
transfer can only happen at the decision-tree level.
