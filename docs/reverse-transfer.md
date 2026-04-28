# Reverse Transfer Experiment: PAMPA → RLM

## Motivation

The main experiment tests RLM → HLM (related, transfer helps) and
RLM → PAMPA (unrelated, transfer hurts XGBoost). A natural follow-up:
does the same pattern hold in the opposite direction? If we pre-train
on PAMPA and finetune on RLM, does XGBoost again suffer catastrophic
negative transfer while Chemprop remains robust?

This tests whether the architectural vulnerability is symmetric -- i.e.,
whether XGBoost's inability to recover from irrelevant pre-training is a
general property of decision-boundary transfer, not specific to the
RLM→PAMPA direction.

## Experiment

Four models evaluated on RLM Stability using the same 5x5 CV splits:

- **XGBoost scratch**: trained directly on RLM
- **XGBoost PAMPA-transfer**: pre-trained on PAMPA, continue boosting on RLM
- **Chemprop scratch**: D-MPNN trained directly on RLM
- **Chemprop PAMPA-transfer**: D-MPNN pre-trained on PAMPA, new FFN head, finetune on RLM

## Results

![Reverse transfer results](figures/reverse-transfer-pampa-to-rlm.png)

*Left: AUC-PR distributions across 25 CV folds. Dotted line = random
baseline (0.298). Right: Tukey HSD (FWER = 0.05). Reference = best
model (Chemprop PAMPA-transfer).*

| Model | AUC-PR (approx mean) | Tukey group |
|---|---|---|
| XGBoost PAMPA-transfer | ~0.38 | Significantly worse |
| XGBoost scratch | ~0.50 | Significantly worse |
| Chemprop scratch | ~0.57 | Best group |
| **Chemprop PAMPA-transfer** | **~0.59** | **Best (reference)** |

## Interpretation

The pattern is symmetric with the main RLM→PAMPA experiment:

**XGBoost suffers catastrophic negative transfer in both directions.**
PAMPA→RLM transfer drops XGBoost from ~0.50 to ~0.38 -- a larger
relative decline than the RLM→PAMPA direction. The inherited PAMPA
decision boundaries (which encode permeability-related features like
lipophilicity and hydrogen bond donors) are irrelevant to metabolic
stability and cannot be unlearned by subsequent boosting. The same
structural flaw (frozen decision trees from the source task permanently
biasing predictions) produces the same failure mode regardless of which
endpoint is source and which is target.

**Chemprop is robust in both directions.** PAMPA→RLM transfer produces a
slight (not statistically significant) improvement over scratch. The
pre-trained encoder, having learned molecular features from PAMPA, retains
generally useful structural representations (atom environments, ring
systems, functional groups). The new FFN head learns the correct mapping
from these features to metabolic stability, unconstrained by the old
permeability objective. This confirms that the D-MPNN's robustness to
irrelevant transfer is a general architectural property, not specific to
the RLM→PAMPA direction.

**Chemprop dominates XGBoost on RLM.** Both Chemprop variants
significantly outperform both XGBoost variants. This gap is wider than
on HLM or PAMPA, consistent with the RLM base model comparison in the
main text (where XGBoost was already significantly worse on RLM).

These results strengthen the main conclusions: the transfer failure mode
is architectural (decision-boundary vs. representation transfer), not
directional (not specific to which endpoint is source/target).

## Notebook

```bash
uv run marimo export html notebooks/14-reverse-transfer.py -o notebooks/14-reverse-transfer.html
```
