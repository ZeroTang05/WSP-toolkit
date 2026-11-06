# TIES: effective-delta merging

`weight_programming.ties_merge` independently implements the mass-sign,
disjoint-mean method from [TIES-Merging](https://arxiv.org/abs/2306.01708).
Its reference is `prateeky2806/ties-merging` commit
`44e7891fc84f3de7e4caa52664cd864ca3715e91`, `src/utils/merge_utils.py`.
Upstream code is not copied into this package.

```python
from weight_programming import ties_merge

merged, evidence = ties_merge(
    [effective_delta_1, effective_delta_2, effective_delta_3],
    [base_fingerprint] * 3,
    density=0.2,
)
```

Each input is a mapping from module/parameter names to floating tensors in
the same base coordinates. For LoRA, first form the effective scaled `B @ A`;
do not separately merge factors. Base IDs are caller declarations, not an
independent verification of a checkpoint. The runner additionally checks the
existing delta manifests and file hashes.

The operation trims each source globally over all supplied coordinates, elects
the sign carrying the greater total signed mass, and averages nonzero entries
agreeing with that sign. It returns dense CPU FP32 effective deltas and evidence;
it does not return absolute model weights or a compressed LoRA factorization.

Two reference details are explicit:

- Inclusive kth-value thresholds can retain more coordinates than the requested
  density, including one additional coordinate without tied values. Actual
  retained counts and thresholds are reported. Density endpoints 0 and 1 are
  extensions; upstream's density=1 threshold requests an invalid order statistic.
- The default `zero_sign='reference_majority'` follows the reference's global
  fallback. A still-zero global vote follows its negative-entry branch.
  `zero_sign='abstain'` instead yields zero on exact mass ties; it is a declared
  variant, not an identical reference implementation.

The same coordinate scope, not matrix-by-matrix trimming, matters for alignment
with the reference. A real q/v-only study is an adaptation of the update scope,
not a reproduction of the paper's complete experiments. Sources from different
seeds of one task are also distinct from independently trained professional roles.

This method can expand rank, storage and update amplitude. It is a capability
combination tool, not automatic model compression. Compare against arithmetic
mean and norm-matched controls before attributing behavioral differences to
conflict handling. Trained-delta merging belongs outside the original WSP
independent-discovery track.

`src/run_ties_study.py` produces hashed dense updates and geometric evidence.
`src/evaluate_ties_proxy.py` evaluates five fixed conditions on training-derived
development and protection inputs, using CPU FP32. These proxies do not replace
full generation accuracy or native-precision deployment measurements.
