# Critical Review

Overall, this is a well-structured and clearly narrated piece of applied ML work. The experimental design is sound in several respects -- chemical space splits, replicated CV, appropriate metric choice for imbalanced data, and the ablation/reverse-transfer controls. That said, there are meaningful issues ranging from logical gaps to unsupported claims to narrative choices that weaken the argument. Organized roughly by severity:

---

## 1. Logical / Inferential Issues

**The "representation transfer vs. decision-boundary transfer" thesis is overstated relative to the evidence.**

- resolved!

The core claim is that the *architectural difference* (where knowledge lives) is what causes the divergent transfer outcomes. But the experiment confounds architecture with transfer mechanism. XGBoost's transfer continues boosting (no parameter is ever discarded), while Chemprop's transfer *replaces the FFN head*. The comparison is not purely "trees vs. GNN" -- it's "continue the old model vs. reset the decision layer." You could construct an XGBoost transfer strategy that discards the source trees entirely and only uses the source model's feature importances to guide new tree construction, or you could retrain a new XGBoost from scratch using features derived from the source model. None of that is tested. The claim should be scoped to "this specific transfer protocol for XGBoost" rather than "gradient-boosted trees as an architecture class." The README occasionally gets this right (e.g., "decision-boundary transfer") but the synthesis and conclusions slip into making it about the architecture itself.

**The Chemprop "shrugging off" irrelevant transfer is conflated with the encoder potentially being helpful.**

- resolved

On PAMPA, Chemprop RLM-transfer scores 0.925 vs. scratch 0.917 -- a +0.008 delta that is not significant. The text frames this as "the encoder's features are general enough to be useful for any property." But the null hypothesis -- that irrelevant pre-training simply doesn't help and the model recovers to scratch performance -- is equally consistent with the data. You cannot distinguish "the RLM encoder features are useful for PAMPA" from "the RLM encoder features are neither helpful nor harmful, and the model just trains through them." The feature importance analysis (sulfur elevation) is suggestive but not dispositive, because you show a single fold and the overall performance delta is noise-level.

**The SHAP / saliency analysis is presented with more causal confidence than it supports.**

- resolved: replaced single-fold snapshots with 25-fold aggregate importance (mean |SHAP| / mean saliency across all CV folds, with cross-fold std as error bars and rank-stability annotations). Toned down causal language throughout. Key rankings are now demonstrably stable (e.g., Bit 1088 at 25/25 folds, sulfur atom types at 20-23/25). See notebook 16 for computation.

Throughout both stories, feature importance shifts between scratch and transfer models are interpreted as causal explanations ("RLM pre-training *elevated* sulfur environments," "the transfer model *over-relies* on nitrogen-heterocycle signals"). Feature importance methods show correlation between features and predictions, not causal mechanisms. More importantly, these are single-fold snapshots with an explicit caveat that "rankings may vary with different test sets" -- yet the narrative builds substantive chemical arguments on them. How stable are these rankings across the 25 folds? If Bit 1088 is #1 in one fold but #15 in another, the story about RLM "amplifying shared metabolic-stability features" collapses. Presenting aggregate feature importance (e.g., mean absolute SHAP across folds) would substantially strengthen these claims.

**The "shared structural rules" claim in Story 1 rests on a weak foundation.**

- resolved: reframed the claim. Added Fisher z-transform 95% CI (0.20--0.76) for the r=0.54 on n=27 to make explicit how uninformative the correlation is. Acknowledged that 12.9% scaffold overlap is non-trivial and could contribute to transfer benefit. Rewrote both the setup paragraph and the "shared structural rules" subsection (now "interpreting the transfer benefit") to present structural-rules as one plausible mechanism alongside scaffold familiarity, rather than the sole explanation. Updated figure caption accordingly.

The argument that transfer benefit comes from structural rules rather than shared data points cites the weak Pearson r=0.54 on n=27 shared compounds. But n=27 is too small for a correlation to be meaningful either way. More importantly, the datasets share 12.9% of scaffolds -- this is not trivially small. Some of the transfer benefit could come from seeing related scaffolds in the source data. The text dismisses this possibility too quickly.

---

## 2. Missing Controls / Gaps in the Experimental Design

**No random pre-training control.**

- resolved: ran the experiment (scripts/run-xgb-random-pretrain.py). Random-label transfer degrades PAMPA modestly (AUC-PR 0.899 vs 0.915 scratch) while real RLM transfer is far worse (0.852). Both explanations have merit -- there is a structural cost to extra trees, but the dominant damage is from inheriting coherent wrong decisions. Full writeup in docs/random-label-pretraining.md, referenced from the main README.

A critical missing experiment: what happens if you pre-train XGBoost on *random labels* for the same molecules? If XGBoost-random-pretrain also collapses on PAMPA, the claim that the failure is about *wrong* decision boundaries is weakened -- it might just be that *any* additional trees degrade the model when the original signal was noise. If XGBoost-random-pretrain performs *differently* from XGBoost-RLM-pretrain, that tells you something about the specific content of the inherited decisions vs. the structural problem of having extra trees.

**No "retrain XGBoost from scratch using RLM-derived features" control.**

- resolved

As noted above, the experiment doesn't test whether XGBoost could benefit from representation-level transfer if given the opportunity. For instance, training an XGBoost on learned D-MPNN features (extracted from an RLM-pretrained encoder) instead of Morgan fingerprints would isolate the feature vs. decision boundary question more cleanly.

**The data efficiency experiment only tests RLM.**

- resolved!

The data efficiency curves are shown only for RLM. The argument that "CheMeleon frozen dominates at every data fraction from 10% onward" is specific to that endpoint. Given that HLM and PAMPA have different characteristics (size, class balance, chemical diversity), the generalizability claim would be stronger with data efficiency curves for all three endpoints.

---

## 3. Narrative / Structural Issues

**The opening buries the lede in hedges.**

- resolved: acknowledged but declined. The informal tone is intentional -- this is not a journal paper. The "(yes)" parentheticals are part of the voice.

The first paragraph of the README says "(yes)" and "(yes)" parenthetically, then launches into a lengthy setup. The reader doesn't learn what the actual finding is until paragraph 3. The document would benefit from a clearer abstract-style opening: "We show that [specific claim], demonstrated by [specific evidence], with implications for [specific practice]."

**Story 1 and Story 2 repeat substantial background material.**

- resolved: acknowledged but declined. The repeated material is only a sentence or two in each case, and each story should be readable on its own without requiring the reader to scroll back to Setup.

The "Why we expect it to work/fail" sections in both stories repeat the biochemistry explanations from the Setup section nearly verbatim. The RLM/HLM/PAMPA descriptions appear in the Setup, in Story 1, in Story 2, and again in the Methods Appendix. This repetition dilutes the narrative rather than reinforcing it.

**The "elephant in the room" section undermines its own framing.**

- resolved: reframed the section to lead with the safety/robustness question ("the question is not which model scores highest -- it's which model is safest to deploy") rather than opening with the performance concession. The performance context now follows as honest acknowledgment rather than the opening move. Also added language clarifying we haven't proven "more data always helps" but have disproved "more data can hurt" for representation-level transfer.

This section asks "why not just use XGBoost?" and then spends 5 numbered points arguing for D-MPNNs. But it opens by conceding the performance gap is small and not statistically significant. The framing implies the question is about peak accuracy, then pivots to robustness/composability/data-efficiency arguments. It would be more honest and more persuasive to lead with: "The question is not which model scores highest on a leaderboard -- it's which model is safest to deploy in a pipeline where you can't always verify your pre-training choices."

**Point 1 under "elephant in the room" makes an unsupported claim at the end.**

- resolved: the "diverse targets" sentence was removed in a prior revision. Point 1 now explicitly scopes the claim: "We cannot claim from this experiment that 'more pre-training data always helps' -- but we have disproved its inverse."

The sentence "Additional training on diverse chemical matter and against diverse targets forces the models to learn diverse and generalizable stories about how chemical structure impacts predictions against various targets, and does not cause worse performance" is not demonstrated anywhere in this work. You tested transfer from one source to two targets. "Diverse targets" is not part of the experimental design.

**Point 4 about "real chemistry" is a non sequitur in that list.**

- resolved: the section is reframed as "why bother with D-MPNNs" (not purely about transfer safety), so data efficiency and foundation model properties are legitimate answers to that question.

The data efficiency results are interesting and relevant, but they're about foundation model pretraining, not about the transfer learning safety argument that the section is making. The CheMeleon frozen feature importance discussion is also shoehorned in here rather than given its own section.

---

## 4. Smaller Issues and Inaccuracies

- **Line 9**: "collapses to be as bad simple random-chance" -- missing "as" ("as bad as simple random-chance").
  - resolved: text was rewritten in earlier edits; typo no longer exists.

- **Line 88**: The figure caption says "XGBoost (red) is significantly worse than the D-MPNN reference" but the text says Chemprop and CheMeleon are indistinguishable. Clarify which model is the reference in the caption.
  - resolved: caption now specifies "Reference = Chemprop RLM-transfer (highest mean)" and explicitly states Chemprop and CheMeleon are indistinguishable.

- **Line 133**: "The Chemprop improvement is statistically significant (p = 0.022, Tukey HSD)" -- this p-value is for the pairwise comparison Chemprop-transfer vs. Chemprop-scratch. Make this explicit; as written it's ambiguous whether the p-value refers to the XGBoost improvement or the Chemprop improvement.
  - resolved: bullet now reads "RLM pre-training helps Chemprop" to clarify the comparison.

- **Line 146**: "Pearson r=0.54 on n=27" -- this is a borderline meaningless correlation on 27 points. The 95% CI for r would span roughly 0.2 to 0.75. Reporting it as "weak" without the confidence interval overstates the precision of the estimate.
  - resolved: addressed in the "shared structural rules" rewrite (95% CI now reported).

- **Line 180**: "any harm must come from *wrong* learned decisions about how a specific chemical group in the dataset correlates with the target variable being inherited" -- this pre-states the conclusion before presenting the evidence. The reader hasn't seen the PAMPA results yet at this point.
  - resolved: reworded to conditional ("would have to come from") to frame it as a prediction rather than a stated conclusion.

- **Line 205**: "XGBoost RLM-transfer sits at the baseline -- no better than guessing the majority class" -- the AUC-PR is 0.853 vs. baseline 0.855, so it's actually *below* the majority-class baseline. "At or below" (as written in the text above) is more accurate than "at."
  - resolved: caption now says "at or below the baseline."

- **Line 267**: "These are real advantages. While these specific neural networks..." -- abrupt transition; the period between these sentences breaks the thought flow.
  - resolved: section was rewritten; the abrupt transition no longer exists.

- **Table on line 184**: There's a stray backslash after "99.6%" in the scaffold column.
  - resolved: backslash removed.

- **The ablation doc (xgb-transfer-ablation.md) reports AUC-PR 0.917 for scratch** but the main README reports 0.910. These should be reconciled -- likely different configurations or a typo, but the inconsistency is confusing.
  - resolved: added a note to the ablation doc explaining the minor difference is run-to-run variance from XGBoost's stochastic subsampling.

- **The chemeleon-overfitting.md reports AUC-ROC values** (0.676, 0.701, 0.739, 0.768, 0.730, 0.716) while the main README reports AUC-PR. The supplementary doc should clarify which metric is being discussed, or ideally report both to match the main text.
  - resolved: added a metric note at the top of the doc clarifying all values are AUC-ROC.

---

## 5. Things Done Well

To be fair: the chemical space splitting strategy is well-validated with the Tanimoto NN and Jaccard diagnostics. The choice of AUC-PR over AUC-ROC is correct and well-justified. The ablation and reverse-transfer experiments are exactly the right controls to run. The Tukey HSD methodology is appropriate and the FWER control is correctly applied. The writing is generally clear and the narrative structure (two contrasting stories) is effective. The honesty about what XGBoost does well ("elephant in the room" section) is refreshing compared to most ML papers that just push their preferred architecture.
