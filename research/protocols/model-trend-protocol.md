# Statistical conventions for same-family, cross-family, and capacity comparisons

Date: 2026-11-06. This protocol was defined before the main matrix was complete and before cross-model results were available. The first Qwen0.6 seed and the base-model baselines for the two small models had already been observed. It is not a prospective preregistration.

## Study quantities and comparisons

For task t, model m, method p, training seed s, and official test question i, let Y(m,p,s,i) be the binary score and B(m,i) the base-model score. Calculate the paired per-question change D=Y−B, and report accuracy for each seed, the mean absolute gain in percentage points, and uncertainty. The main metric remains flexible exact match on the full GSM8K set; ARC-C and HellaSwag use acc_norm. Absolute percentage-point changes are the main analysis, so a low baseline is not obscured by relative growth.

Methods include standard LoRA, a fixed global grid, random, greedy, BO, and LLM. C1 fixes rank at 8; C2 keeps trainable parameters at approximately 0.2%. C2 currently covers standard LoRA only and does not control the capacity of every search strategy. For Qwen0.6, C2 and C1 both use rank 8 and reuse the same run; they are not independent repetitions.

Compare each method separately:

1. Qwen0.6 versus Qwen1.7, and Smol360 versus Smol1.7: scale differences within a model family.
2. Qwen1.7 versus Smol1.7: different families with similar parameter counts.
3. Other cross-family pairs: family and scale both vary, so label them separately.
4. C2−C1 for standard LoRA within the same model: a capacity comparison paired by seed and data order.

Define a cross-model trend difference as the difference in mean gains, not the difference in raw accuracy. Also report whether mean gains have the same sign and the individual seed gains. A common sign alone does not establish equivalence or statistical similarity. No acceptable equivalence margin was specified in advance, so equivalence testing is not performed.

## Resampling and Range

Within a model, sample three seed rows with replacement, then sample official question columns with replacement and calculate mean D. For cross-model comparisons, sample seeds independently on the two sides while using the same sampled question columns to preserve question pairing. The same numerical seed does not imply paired training noise across architectures; pairing such seeds could underestimate uncertainty. For C2−C1, take the within-model, within-seed difference first, then resample.

Use 2,000 bootstrap repetitions and random seed 20261106, and report percentile 95% intervals. Three training seeds describe training variation only roughly. Multiple unadjusted intervals are descriptive, not a claim of family-wide statistical significance. A single-seed result must be labeled as unable to characterize seed variation reliably. A formal cross-model comparison requires seeds 11, 22, and 33 on both sides. The small number of models and families cannot support inference to a population of models or attribute differences causally to architecture.

Keep the 5,000 per-question bootstraps from previous single runs separate from this protocol's uncertainty estimate. Analyze the tool-level proxy risk curve and actual generation accuracy separately; lower NLL alone does not establish an accuracy gain.

## Executable checks

`src/analyze_model_trends.py` reads the official `evaluation-index.json` and optional C2 index. It checks complete official doc_ids, per-question score summaries, sample content hashes, model names, recorded weight and tokenizer fingerprints, consistent batch settings, harness/runtime, and snapshots of runner and helper source code. It does not reload and hash the multi-gigabyte base model or reproduce GPU execution. Source audits and actual inference provide separate evidence.

The output identifies missing conditions and seeds; existing rows cannot be presented as a complete matrix. Smoke results, old batch8 generation mixed with official batch32, mislabeled models, and differing execution snapshots are rejected. Reused parameter states retain their original provenance without inventing additional timing or independent repetitions.

The partial run passed and produced six task summaries for Qwen0.6 standard LoRA and fixed 0.8 configurations. No three-seed cross-model comparison is available. Four statistical checks cover preservation of seed variation, common-question pairing, independent seed sampling across models, and the single-seed restriction; additional checks reject mislabeled models and old generation batches. These checks validate the analysis, not an empirical trend conclusion.

```sh
.venv/bin/python src/analyze_model_trends.py \
  --index benchmarks/confirm-v2/evaluation-index.json \
  --output research/model-trends-partial.json

# Once both matrices are complete, read the fixed-rank and relative-capacity evidence:
.venv/bin/python src/analyze_model_trends.py \
  --index benchmarks/confirm-v2/evaluation-index.json \
  --capacity-index benchmarks/capacity-confirm-v2/evaluation-index.json \
  --output research/model-trends-C1-C2.json
```
