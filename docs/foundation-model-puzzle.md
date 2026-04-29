# The Foundation Model Puzzle: CheMeleon Overfitting and the Frozen Encoder Fix

## The problem

Our initial experiments used only the first six models, with CheMeleon fully finetuned. The foundation model produced counterintuitively *worse* results than the much smaller Chemprop -- which led us to hypothesize overfitting and add the frozen-encoder variants. See [chemeleon-overfitting.md](chemeleon-overfitting.md) for the full narrative.

CheMeleon single-finetune (0.908 AUC-PR) is worse than Chemprop scratch (0.917) on PAMPA, though the difference is not statistically significant (Tukey HSD, p = 0.98). With 9.3M parameters and ~1,626 PAMPA training samples, CheMeleon is extremely overparameterized. The foundation pre-training provides a reasonable initialization, but 30 epochs of finetuning is enough to overfit. The smaller Chemprop model (318K params) has less capacity to memorize noise.

## The fix: freeze the encoder

We froze the 8.7M-parameter BondMessagePassing layer and trained only the FFN head (~615K trainable parameters). If the foundation representations are good enough and the full-finetune models were overfitting, the frozen variants should improve.

| Target | Model | AUC-PR (mean +/- std) |
|---|---|---|
| **HLM** | CheMeleon single-finetune (unfrozen) | 0.790 +/- 0.044 |
| | CheMeleon double-finetune (unfrozen) | 0.806 +/- 0.032 |
| | CheMeleon frozen single | 0.819 +/- 0.035 |
| | CheMeleon frozen double | 0.819 +/- 0.034 |
| **PAMPA** | CheMeleon single-finetune (unfrozen) | 0.908 +/- 0.029 |
| | CheMeleon double-finetune (unfrozen) | 0.912 +/- 0.025 |
| | CheMeleon frozen single | 0.921 +/- 0.029 |
| | CheMeleon frozen double | 0.922 +/- 0.029 |

![CheMeleon frozen vs unfrozen boxplots](figures/chemeleon-frozen-boxplots.png)

*AUC-PR distributions for CheMeleon variants only. Green = unfrozen (all weights finetuned). Purple = frozen encoder (FFN only).*

![CheMeleon frozen vs unfrozen Tukey HSD](figures/chemeleon-frozen-tukey-hsd.png)

*Tukey HSD comparing frozen vs unfrozen CheMeleon variants (FWER = 0.05).*

**HLM**: No significant differences between any frozen/unfrozen variant (Tukey HSD, all p > 0.06). The encoder adaptation during full finetuning neither helps nor hurts for the related endpoint.

**PAMPA**: Freezing improves performance. Under AUC-ROC the improvement is statistically significant (frozen single 0.730 vs unfrozen single 0.676, p = 0.001), confirming the overfitting hypothesis. Under AUC-PR the effect is present but compressed by the narrow effective range above the 0.855 baseline.

## What the frozen encoder attends to

With the encoder frozen, only the FFN head differs between single and double finetune. Does the intermediate RLM step change what the model attends to?

![CheMeleon frozen single vs double](figures/pampa-chemeleon-single-vs-double.png)

*CheMeleon frozen single-finetune (Foundation→PAMPA) vs frozen double-finetune (Foundation→RLM→PAMPA). Top 6 atom types by gradient saliency. Since the encoder is frozen, differences reflect the FFN head only. Single fold.*

The two panels are nearly identical: S deg1, O(arom) deg2, S deg2, S deg4, S(arom) deg2, and N(arom) deg3 appear in both top-6 lists in the same order. Importantly, the two models are not *forced* to attend to the same features just because the encoder is frozen. The saliency measurement captures how much the final prediction depends on each atom, which is influenced by both the encoder and the decision-making layer (FFN head). Since the FFN heads were trained via different paths (one saw RLM data, one did not), they could in principle weight the encoder's outputs differently. The fact that they don't -- that independently trained decision layers converge on the same atomic attention pattern -- suggests the foundation encoder produces features whose relative importance is baked into the representation itself, not imposed by downstream training.

The dominance of sulfur environments (S deg1, S deg2, S deg4, S(arom) deg2 -- four of the top six) is chemically interesting. Thioethers and sulfonamides are common motifs in drug-like molecules that affect both lipophilicity (via the polarizable sulfur atom) and membrane partitioning. The CheMeleon foundation model, pre-trained on Mordred descriptors across 1M compounds, appears to have learned particularly discriminating representations for sulfur-containing functional groups. Aromatic oxygens (O(arom) deg2 -- furan/pyran-type oxygens) and aromatic nitrogens (N(arom) deg3 -- trisubstituted pyridine-like nitrogens) round out the top features, both of which influence the balance of lipophilicity and hydrogen bonding that governs passive permeability.

The frozen CheMeleon models are statistically indistinguishable from Chemprop RLM-transfer under both AUC-PR (p > 0.99) and AUC-ROC (p > 0.98) on PAMPA. The CheMeleon foundation representations -- learned from 1M PubChem compounds predicting Mordred descriptors -- are genuinely useful general molecular features, but only when the model is prevented from overwriting them during finetuning on a small dataset.
