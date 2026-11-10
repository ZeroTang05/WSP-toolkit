# Expanded model programming: operations, weight effects, and experimental interpretation

This report preserves the experimental settings, results, and status at the time of writing. For the current project progress, see [Project Status](../../PROJECT_STATUS.md).

2026-11-06. The research scope was expanded at the user’s request, replacing the earlier priority of collecting existing jobs without adding methods. The weight geometries below have been calculated on real models; a new task-accuracy matrix has been submitted and is incomplete. Existing training and public evaluations remain in place.

## 1. Expansion from single coefficient search to four-category model programming

Write the model state as \(S=(\theta_0,\Delta,\mathcal G)\): base parameters, task updates, and actual execution structure. The original operation primarily selects four depth coefficients for a given update. The new operations directly change the update’s composition, rank, or execution depth. The research question therefore expands to which parameter program suits which model; a controller is one way to select a program.

| Program | Literature source | Implementation | Question |
|---|---|---|---|
| Update combination after training| [Model Soups](https://arxiv.org/abs/2203.05482), [Task Arithmetic](https://arxiv.org/abs/2212.04089) | Average the effective LoRA updates from three seeds on the same base using exact factor concatenation.| Can multiple training directions be combined into a more effective single update?|
| Update compression with minimum weight error| Classic truncated SVD; use [SVD-LLM](https://arxiv.org/abs/2403.07378) as LLM compression reference| Use small-core SVD to reduce the combined update from rank 24 to 8 or 4| How much generation capacity is lost when update storage is reduced?|
| Activation-weighted update compression| Layer output reconstruction target of SVD-LLM| Fix the combined model’s input distribution, collect low-dimensional second moments, and obtain a rank8 output reconstruction.| Can calibrated activations preserve domain capability better than the weight spectrum alone?|
| Block influence score-driven depth compression| [ShortGPT](https://arxiv.org/html/2403.03853v1), formula 2 and layer deletion rules| BI on domain calibration; delete low-BI layers, with random/tail deletion control| Does a small hidden-state change mean the layer can be deleted?|

Combination implements parameter updates with exact algebra; it does not claim to reproduce the original Model Soups vision experiments. SVD compresses only the delta, not the full base model. The activation method lacks SVD-LLM’s layerwise recovery and is explicitly called a “local output reconstruction adaptation.” BI uses variable-length GSM training views, averaging over tokens first and then over examples; this adapts ShortGPT’s domain and sampling.

Implementation: `src/delta_programming.py`, `src/depth_programming.py`, `src/programming_probe.py`, `src/programming_effects.py`. The engineering interface is not itself the research contribution.

## 2. From paper to tool: operational semantics and local properties

### 2.1 Update combination should occur on BA

For effective updates \(D_i=B_iA_i\), first absorb the training scale \(\alpha_i/r_i\) and coefficients into \(B_i\). Define

\[
A_*=[A_1^T,\ldots,A_m^T]^T,\qquad B_*=[w_1B_1,\ldots,w_mB_m],
\]

Then \(B_*A_*=\sum_iw_iD_i\), with rank at most \(\sum_i r_i\). Under uniform weights, \(\theta_0+\frac1m\sum_i D_i\) is the mean of the fine-tuned weights from the same base. When weights do not sum to 1, the result is a linear combination of task updates.

Averaging A and B separately creates cross terms and depends on arbitrary basis transformations of the LoRA factors. The new combiner performs arithmetic on effective updates. There are rounding differences in floating-point implementations, so the real-number identity does not imply bitwise-identical BF16 execution.

### 2.2 Use small core to obtain optimal weight approximation

For \(A^T=Q_aR_a\) and \(B=Q_bR_b\), decomposing \(R_bR_a^T\) yields the nonzero singular values and left and right subspaces of \(BA\). Let the truncated result be \(D_k\):

\[
\min_{\mathrm{rank}(M)\le k}\|D-M\|_F^2=\sum_{j>k}\sigma_j(D)^2.
\]

This gives the optimal local weight approximation. For fixed inputs to one layer, \(\|(D-D_k)x\|_2\le\sigma_{k+1}\|x\|_2\), but this does not automatically bound accuracy for whole-network generation. Output tokens change discretely, and subsequent layers alter the input distribution.

### 2.3 Activation target: low-dimensional second-order moment rather than full input covariance

Let \(z=Ax\), \(C=\mathbb E[zz^T]\). Here C is the uncentered second moment; the mean is not subtracted. The goal is

\[
\min_{\mathrm{rank}(M)\le k}\mathbb E\|BAx-Mx\|_2^2.
\]

Take the top k left singular vectors U of \(BC^{1/2}\) and construct \(M=UU^TBA\). The original output \(y=BAx\) has second moment \(BCB^T\); projection onto its top k eigenspace achieves the minimum reconstruction error. Any rank-k linear predictor also has an output subspace of at most k dimensions and cannot achieve a lower error. The error is \(\sum_{j>k}\sigma_j(BC^{1/2})^2\). No inverse of C is required, so degenerate calibration distributions are allowed.

This conclusion holds only for fixed source-model activations under the declared empirical calibration distribution. When compressed updates are applied throughout the model at once, inputs to later layers change; this implementation has no layerwise re-estimation or recovery. The transfer of errors to task behavior should be examined rather than replacing global evaluation with local optimality.

### 2.4 Observation scoring and causal deletion are measured separately

\(BI_l=1-\mathbb E\cos(h_l,h_{l+1})\). Low BI only indicates that the input and output directions of the residual block are close; it does not directly measure the answer loss, and may also ignore the amplitude and a few key tokens.

The deletion response \(\delta_l=R(f_{-l})-R(f)\) is therefore measured separately for each layer, using math reference-answer risk and ARC retention CE, respectively. Compare BI with \(\delta_l\) by Spearman correlation, depth, and anomalous layers; these are intervention results for the current model and inputs. Single-layer damage cannot be directly added to infer multi-layer deletions because of interactions.

## 3. Completed: Real cross-model analysis of 12 training updates

All four models use 3 GSM training seeds (11/22/33), trained on their respective bases with q/v target rank8. Compare update cosines only within the coordinates of the same model; compare dimensionless spectra and normalized depth distributions across models, and do not compare the original matrix coordinates of different architectures.

| Model| average cosine between seeds| Top 4 singular-value energy in one update| Update energy in the final quarter of layers| Combination→rank8 energy| Combination→rank4 energy| rank4 factor MiB|
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 0.125 | 93.24% | 35.53% | 92.81% | 85.02% | 2.188 |
| Qwen3-1.7B | 0.294 | 96.98% | 41.04% | 97.57% | 94.31% | 3.062 |
| SmolLM2-360M | 0.556 | 98.83% | 48.67% | 99.35% | 98.34% | 1.562 |
| SmolLM2-1.7B | 0.408 | 98.44% | 36.42% | 98.79% | 96.93% | 3.000 |

“Top 4 in one direction” takes the top 4 in each target matrix, aggregates their energy, and then averages over 3 seeds; it does not mean the entire model has only 4 directions. “Combination→rankk” is the aggregate energy retained after truncating each combined module, calculated from the saved FP32 factor norms with slight rounding. MiB includes only FP32 factor tensors, excluding the manifest, shared base, and run cache. The original evidence is in `artifacts/public/programming-expansion-v1/`, and `src/analyze_programming_geometry.py` can be used to regenerate `analysis.json`.

The results support four observations:

1. **Scale trends depend on model family.** Seed directions become more consistent as Qwen scales up; SmolLM2 shows the opposite. These four checkpoints and this training recipe do not establish a scaling law for all models.
2. **Combination increases compression difficulty.** Qwen0.6's top 4 in one direction retain an average of 93.24% of energy, while the top 4 in three directions only retain 85.02%. Each direction is low-rank, but their combination need not share a single low-dimensional subspace.
3. **There is also no uniform scale trend in depth energy.** The proportion in the last 1/4 of layers changes with scale from 35.53% to 41.04% for Qwen and from 48.67% to 36.42% for Smol. This provides an observational basis for deep coefficient intervention, but high energy does not equal high causal importance.
4. **The weighted mean changes both direction and magnitude.** The combined norm/seed11 norm is 0.645, 0.731, 0.841, and 0.784. A same-norm single-direction control is needed: part of the combination’s gain may come from weakening an overly strong original update. Assume that all source norms are equal and the average pairwise cosine is c, then the norm ratio of the mean is \(\sqrt{(1+2c)/3}\); the actual norms are not exactly equal, so this formula is only used for explanation and does not replace the actual calculation.

Three pairs of cosines share three source updates and cannot be treated as three independent statistical repetitions. A three-seed mean combination is not three independent combination experiments. The current geometry only describes these 12 updates and cannot directly claim 98% capability retention from 98% energy retention.

## 4. Model programming effect: existing quality evidence and new complete matrix

Under the same non-thinking GSM batch32 protocol, Qwen0.6 scored 485/1319 for the base model, 532 for standard LoRA, 590 for a fixed global coefficient of 0.8, and 589 for random depth coefficients. Fixed and random differ by one net question, but 69 answers improved and 70 degraded, so correctness differs on 139 questions. Similar overall scores do not establish identical behavior; per-question transitions should accompany the aggregate score.

Fixed 0.8 scored 351/1172 on ARC-C versus 363 for the base, and 4443/10042 on HellaSwag versus 4217 for the base. Domain improvement and retention-task changes differ in direction; satisfying the development cross-entropy constraint does not establish general capability retention. The above is a single seed measurement of a complete public partition, and is not evidence that the new combination/compression process is effective.

The new complete results are 564/1319=42.76% for single-axis detection (formerly called greedy) and 517/1319=39.20% for BO. BO’s development risk is 0.31513, lower than fixed’s 0.31848, but its test accuracy is 5.53 percentage points below fixed; the paired-question bootstrap interval is [-7.73,-3.49] percentage points, with 70 questions improved and 143 questions degraded. The difference from standard LoRA is -1.14 percentage points, with an interval of [-3.11, 0.76]. This cannot be used to confirm that it is generally weaker than LoRA. Single-axis detection has a worse development risk of 0.59424, yet a higher test point estimate than BO. This again shows that proxy and generation-quality rankings can differ.

These comparisons use the same frozen source, batch, and model on all 1319 questions. The intervals reflect question uncertainty only, are not corrected for multiple comparisons, and do not describe variation across training seeds. New evidence and original predictions are in `artifacts/public/programming-expansion-v1/existing-effects`; the analysis script is `src/analyze_policy_effects.py`. The results support adding end-to-end metrics to applied role replacement; they cannot be used to reselect the current program on the test set.

### New: Output behavior and grouping effects of parameter programs

The analysis continues on the original generations from the same complete split, without changing any plans. The "calculation mark" in the following table indicates whether `<<` is included in the generation. This measures format only. Reference-step groups use the number of `<<` markers in the official solution and do not measure actual cognitive difficulty.

| program| Correct / 1319| Median number of output characters| Questions with generated calculation markers| Reference 0–2 steps, correct / 440| Reference 3–4 steps, correct / 654| Reference 5+ steps, correct / 225|
|---|---:|---:|---:|---:|---:|---:|
| base model| 485 | 658 | 0 | 201 | 231 | 53 |
| StandardLoRA| 532 | 273 | 1259 | 242 | 243 | 47 |
| Fixed 0.8| 590 | 269 | 1134 | 269 | 273 | 48 |
| random depth coefficient| 589 | 272 | 1208 | 269 | 273 | 47 |
| Final 1/4 depth group only × 0.95| 564 | 608 | 11 | 229 | 267 | 68 |
| BO selection| 517 | 267 | 1250 | 241 | 237 | 39 |

This gives three additional observations. First, standard updates and most full-depth programs significantly change the output format and length, while the final-quarter-only update remains closer to the base model output style; this intervention retains fewer calculation markers while still improving the overall point estimate. Second, fixed and BO have nearly identical median character counts but differ by 73 correct answers, so short outputs or format imitation cannot explain all quality changes. Third, fixed has the best overall score but scores below the final-quarter-only update on the “reference 5+ steps” group (48 vs. 68), suggesting that the average masks differences across conditions.

These are post-hoc descriptions from one seed without multiple-comparison correction across groups. The reference-answer groups use evaluation labels and cannot be used to construct deployable routing or reselect current candidates. Character length is neither token count nor measured speedup. Rerun with `src/analyze_answer_behaviour.py`; evidence is in `artifacts/public/programming-expansion-v1/answer-behaviour.json`. Subsequent role programming should observe interface style, role capabilities and workflow results separately, rather than replacing any one indicator with the other two.

Supplementary paired counts compare the “last 1/4 update” with fixed. In the reference 0–2 step group, 66 answers improved and 106 degraded, a net -9.09 percentage points with a descriptive question-bootstrap interval of [-14.77, -3.18]. In the reference 5+ step group, 46 improved and 26 degraded, a net +8.89 percentage points with interval [1.33, 16.44]. The groups were selected post hoc, and the intervals were not corrected for multiple comparisons. No confirmatory significance or universal mechanism claims are made based on this. The point estimates of BO in the three groups are all lower than the fixed setting, indicating that its overall damage is not concentrated only in the highest reference step group.

The new matrix adds 11 conditions for each model: native, seed11 standard update, three-way mean, Frobenius rank8/rank4, activation rank8, same norm seed11, and layer deletion by BI at about 10%/20%, random deletion at about 10%, and tail deletion at about 10%. The four models have 44 states in total. Each state is evaluated on complete GSM8K 1319, ARC-C 1172, and HellaSwag 10042, giving 132 task results and 551,452 state × question scores.

The new matrix uses the HFLM execution path with preloaded models and a separate response cache for each state; the base and standard update are also remeasured under this path. One cannot directly subtract the new path intervention from the old path baseline and claim a strict pairing. Settings are base BF16, SDPA, max_length2048, generation limit 512, GSM batch32, multiple-choice batch8, and seed 20261106.

Program construction uses only 128 training-derived calibration examples; layerwise intervention uses 64 development and 32 ARC validation examples. No official tests were used to pick ranks, deletion strategies, or combination weights. The full public test is the confirmation phase.

Job `job-mw53jetr-d779` has frozen the code and is running. The four-model CPU combination phase is complete; next it generates BI, layerwise responses, and activation moments for the four models in sequence, then runs the full quality matrix. GPU0 uses the shared lock; waiting is not a result.

## 5. How should the analysis interpret positive and negative results?

| control| If improved| If it does not improve or becomes worse|
|---|---|---|
| combination vs seed11| A combined candidate shows a gain; also consider the individual results of the other source updates.| The mean may be outside the low-loss zone, or the source directions may not be complementary.|
| Combination vs same norm seed11| Supports additional value from combining directions| Gains may be primarily explained by magnitude adjustments; a nonsignificant difference does not establish equivalence|
| Activation-weighted rank 8 vs Frobenius rank8| Supports transfer from the local output objective to task retention| Insufficient calibration coverage or distribution drift from simultaneous compression may dominate|
| rank4 vs rank8 vs uncompressed combination| Establish the Pareto relationship between update storage and capability| Spectral energy cannot stand in for task gains; locate sensitive modules.|
| BI deletion vs same budget random/tail deletion| Support the selection value of the score| BI may only describe similarities and cannot reliably identify deletable layers.|
| Single-layer BI associated with deletion damage| Support the score’s interpretability on the current distribution| Study the counterexamples and token positions of high BI and low damage, low BI and high damage|

Paired question differences and improvement/degradation numbers are the main analysis; cross-seed confidence intervals are not reported for single combinations. A new strategy proposed after a test failure must be labeled follow-up exploration; the current confirmation rules cannot be changed retrospectively.

## 6. Application and efficiency: two different compression outputs

For K roles, the storage of the common base B and each update D changes from \(K(B+D)\) to \(B+KD\). Rank compression further reduces D: rank24 → 8 uses 1/3 of the factor tensor storage, and → 4 uses 1/6. This complements base sharing and is applicable to later FinQA role updates, but the effects of this GSM update cannot be claimed as FinQA benefits.

Layer deletion changes the execution graph and can reduce executed blocks. The current reversible experiment context retains references to the original layers, so its CUDA allocation is not used as the memory footprint of an independently compressed model. A deployment artifact also requires materialization or independent loading, matched-length prefill/decode timing, and quality constraints. Existing MLP structure pruning routes continue to provide another compression dimension.

Current outputs are four types of paper-driven operations; local algebraic properties and boundaries; weight evidence from 12 source updates and 16 actual compiled programs; execution of the complete four-model effect matrix; and execution of the layerwise observation–intervention response analysis. Domain fidelity, actual latency improvement, and LLM selection advantages remain to be tested.

Update on 2026-11-07: The four-model BI, activation moments and 112 single-layer deletions have been completed, and the original records have been downloaded and the correlations independently recomputed. See [layer-observation-study.md](layer-observation-study.md) and supporting diagram; the full generation matrix is ​​still running, and development proxy results are not substituted for public task confirmation.
