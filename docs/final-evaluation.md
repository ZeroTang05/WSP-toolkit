# Host-owned final mathematical evaluation

Final evaluation is separate from `ProgrammingSession` and its practice-only
`ForwardService`. `weight_programming.evaluation.evaluate_math` accepts explicitly
marked test/evaluation rows, generates from the problem text only, streams raw
sample records, and retains final labels on the host. It never returns test
feedback to a programmer session.

Acquire the official pinned MATH-500 test split:

```sh
python research/acquisition/download_math500.py --output /data/math500-evaluation
```

Use the separate Qwen3.5 runtime (`.[qwen35,math]`) and Linux CUDA environment:

```sh
PYTHONPATH=src python src/run_math500_evaluation.py \
  --model /models/Qwen3.5-0.8B \
  --dataset /data/math500-evaluation \
  --output /results/math500-original
```

The runner requires the pinned original-model manifest, all 500 official items,
four samples per problem, thinking enabled and a 32,768-token generation cap.
Sampling follows the separately recorded original-style policy. It freezes
source, dataset and native-weight identity. GPU 0 is locked per problem and
the model is offloaded before releasing the lock. Device wait and residency
costs are recorded separately. Initial backend verification occurs on CPU;
generation runs on CUDA 0 inside the scheduler context.

To evaluate a frozen edit, supply `--replacements` pointing to the manifest and
native replacement tensors from `WeightStore.materialize_edits`. This imports
a private copy, verifies its declared source identity and file hashes, and
applies the normal edit-policy checks. Freeze the candidate and retain its
selection/discovery provenance before invoking evaluation. Do not select new
edits from final scores. Reference replay and independent discovery must have
separate origins and output directories.

Raw text, token IDs, grades, finish reason and declared repetition diagnostics
are saved per sample. Accuracy is the mean of repeated-sample correctness,
not pass@4; standard errors cluster samples by problem. These diagnostics and
the independent Hugging Face engine are not asserted equivalent to the
author's unpublished full harness. Final scores remain unknown until all
2,000 samples finish. Existing output directories fail explicitly; partial
runs require investigation rather than silent restart.

`test_math_evaluation` checks the real generation path with a tiny model and
a clearly labeled fixture grader. It verifies streaming, sample identity,
test-role requirements and that answer fields do not enter prompts; it does
not measure mathematics performance.
