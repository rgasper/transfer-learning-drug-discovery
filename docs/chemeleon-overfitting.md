# CheMeleon Overfitting: From Counterintuitive Results to Frozen Encoders

When we first trained CheMeleon (9.3M parameters) with full finetuning on
these small ADME datasets, the results were counterintuitive: the
foundation model performed *worse* than a randomly-initialized Chemprop
model with 29x fewer parameters. On PAMPA, CheMeleon single-finetune
(0.676 AUC) underperformed Chemprop scratch (0.701 AUC). On HLM,
CheMeleon single-finetune (0.739 AUC) was beaten by Chemprop RLM-transfer
(0.768 AUC) despite having access to a far richer pre-training signal
(1M PubChem compounds vs 2,529 RLM compounds).

This was the opposite of what we expected. Foundation models are supposed
to provide better initializations than random weights, and their
pre-training on diverse chemical space should help generalization. What
went wrong?

## The overfitting hypothesis

With ~720 HLM training samples and ~1,626 PAMPA training samples, the
parameter-to-sample ratios for CheMeleon are extreme:

| Endpoint | Training samples | CheMeleon params | Ratio |
|---|---|---|---|
| HLM | ~720 | 9.3M | 12,900:1 |
| PAMPA | ~1,626 | 9.3M | 5,700:1 |

For comparison, Chemprop (318K params) has ratios of 440:1 and 196:1.

The hypothesis: even though CheMeleon starts from a good initialization,
30 epochs of unrestricted finetuning on a small dataset is enough to
overwrite the foundation representations with noise. The model has enough
capacity to memorize the training set rather than learning generalizable
patterns.

## Testing the hypothesis: frozen encoder experiment

We froze the 8.7M-parameter BondMessagePassing encoder and trained only
the FFN head (~615K trainable parameters). If the foundation
representations are good and full finetuning was destroying them, the
frozen variants should improve.

Results:

| Target | Unfrozen single | Frozen single | Delta |
|---|---|---|---|
| HLM | 0.739 | 0.755 | +0.016 (n.s.) |
| PAMPA | 0.676 | 0.730 | +0.054 (p = 0.001) |

On PAMPA, freezing the encoder improved AUC by +0.054, significant at
p = 0.001 (Tukey HSD). The frozen CheMeleon became the best PAMPA model
overall (0.730 AUC), surpassing even Chemprop RLM-transfer (0.716 AUC).

On HLM, freezing made no significant difference — the encoder
representations were already adequate for the related microsomal stability
task, so the damage from finetuning was less severe.

## Implications

The initial counterintuitive results were not evidence that foundation
models are useless for small datasets. They were evidence that
unrestricted finetuning of overparameterized models on small datasets
destroys the representations that make foundation models valuable in the
first place. The fix is straightforward: freeze the encoder and train
only the task-specific head.

This finding is consistent with standard transfer learning practice in
computer vision (e.g., freezing ResNet backbones for small-dataset
classification tasks) but is worth highlighting because it contradicts
the assumption that end-to-end finetuning is always preferable when
compute is available.

Full results and statistical comparisons are in the main
[README](../README.md#frozen-encoder-experiment-testing-the-overfitting-hypothesis).
