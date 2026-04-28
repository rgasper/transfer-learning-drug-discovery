# Why Tukey HSD Confidence Intervals Have Identical Widths

The Tukey HSD `plot_simultaneous` visualization shows confidence intervals
that appear to have identical widths across all model groups. This is not
a bug — it is a mathematical consequence of balanced group sizes and the
pooled variance formulation.

## The formula

Statsmodels computes per-group halfwidths using the Hochberg & Tamhane
(1987) simultaneous CI method. The key inputs are:

1. **`q_crit`**: the studentized range critical value (scalar, same for
   all groups)
2. **`var`**: the pooled mean squared error from the one-way ANOVA
   (scalar, same for all groups)
3. **`groupnobs`**: the number of observations per group

The per-group variance contribution is `gvar = var / groupnobs`. When all
groups have the same sample size (as in our case: n=25 for every model),
`gvar` is identical across groups. The pairwise distance matrix `d` and
weighting vector `w` then collapse to equal values, producing identical
halfwidths.

From the source (`statsmodels.sandbox.stats.multicomp.simultaneous_ci`):

```python
gvar = var / groupnobs
d12 = np.sqrt(gvar[pairindices[0]] + gvar[pairindices[1]])
# ... matrix construction ...
w = ((ng - 1.) * sum2 - sum1) / ((ng - 1.) * (ng - 2.))
halfwidths = (q_crit / np.sqrt(2)) * w
```

With balanced groups, every element of `gvar` is the same scalar, so
every pairwise `d12` is the same, every row sum of `d` is the same, and
every element of `w` is the same.

## Empirical confirmation

```
HLM Stability:
  halfwidths: [0.01721226, 0.01721226, 0.01721226, 0.01721226,
               0.01721226, 0.01721226, 0.01721226, 0.01721226]

PAMPA pH 7.4:
  halfwidths: [0.02243956, 0.02243956, 0.02243956, 0.02243956,
               0.02243956, 0.02243956, 0.02243956, 0.02243956]
```

All eight groups within each endpoint have exactly the same halfwidth.
The PAMPA halfwidths are wider than HLM because the pooled MSE is larger
(more variance across PAMPA models).

## Why the boxplots look different

The boxplots show each group's *own* IQR and whisker range — the
within-group spread of the 25 fold-level AUC-ROC values. These vary per
model because some models have more variable performance across folds.

The Tukey HSD CIs are not per-group standard errors. They are
simultaneous comparison intervals computed from the **pooled** variance
across all groups. The pooled variance averages out per-group differences,
and with balanced designs, the intervals are identical by construction.

This is the expected and correct behavior for Tukey HSD with equal group
sizes.

## Reference

Hochberg, Y., and A. C. Tamhane. *Multiple Comparison Procedures.*
Hoboken, NJ: John Wiley & Sons, 1987.
