# From Pruning Selection to Error Compensation: A Real 0.6B Tool-Combination Study

This report preserves the experimental setup, results, and status at the time of writing. See [Project status](../../PROJECT_STATUS.md) for current progress.

2026-11-07. Local reconstruction experiments are complete. They do not establish full-model performance or efficiency.

## Method Rationale

Wanda selects connections for removal from weights and input statistics. Zeroing those connections does not adjust the retained weights. [SparseGPT](https://arxiv.org/abs/2301.00774v3) targets local output reconstruction through second-order information and columnwise error compensation. Together they form a workflow of input observation, pruning selection, compensation of retained weights, and damage measurement. This objective differs from training-based capability adaptation.

The interfaces `input_hessian(X)` and `sparsegpt_reconstruct(W,H,...)` are implemented in `src/weight_programming/reconstruction.py`. The first consumes unpadded token rows and returns the uncentered statistic `2 XᵀX/n`. The second uses relative damping and Cholesky factors of the inverse matrix, supporting either a fixed mask or adaptive selection within blocks. It returns an FP32 proposal, a deletion mask, and diagnostics without overwriting the source checkpoint.

The authors' [reference code](https://github.com/IST-DASLab/sparsegpt/blob/master/sparsegpt.py) is archived with its Git blob recorded, and the Apache-2.0 license is included in the package. Explicit adaptations are a CPU tensor-only interface, FP64 internal calculations, stable sorting to select an exact count in each block, token-normalized second moments, and an added fixed-mask mode. N:M sparsity, quantization, convolutions, and full-model traversal are excluded. The authors' inclusive threshold differs from this implementation's count convention; after aligning actual deletion counts, numerical results agree within tolerance. With relative damping, positive global rescaling of H does not change the update.

## Experimental Setup

Qwen3-0.6B runs on CPU in FP32. Calibration uses 128 training-derived texts totaling 27,776 tokens. The edited matrix is `model.layers.14.mlp.down_proj.weight`, with shape 1024×3072. Its layer was inherited from the observation tutorial, not discovered as optimal by search.

The first 64 texts, containing 13,934 tokens, fit the second moment. The last 64 texts, containing 13,842 tokens, evaluate reconstruction on held-out calibration data. Both halves come from the training-derived calibration set; the second half is not a public test set. The metric is `||(W_new−W_base)X||²_F / ||W_base X||²_F`, measuring only the relative squared output error of this linear map. No other layer is changed, and full-model generation after compression was not evaluated.

## Results

Held-out error on 64 texts, expressed as a percentage:

| Method | 20% deleted | 50% deleted |
|---|---:|---:|
| Rowwise magnitude selection, zeroed | 0.5546 | 8.8738 |
| Same magnitude mask, compensated | 0.2206 | 4.1622 |
| Rowwise Wanda selection, zeroed | 0.2254 | 3.9193 |
| Same Wanda mask, compensated | 0.1435 | 2.7075 |
| SparseGPT adaptive mask, zeroed | 0.2179 | 3.2778 |
| Same adaptive mask, compensated | 0.1142 | 2.0895 |

Same-mask comparisons isolate compensation: at 50% sparsity, error falls by 30.9% for Wanda, 53.1% for magnitude selection, and 36.3% for adaptive selection. The adaptive mask is chosen using the sequentially compensated state. Its zeroed condition fixes that mask afterward as an attribution control; it is not an independently selected, uncompensated algorithm.

Wanda and magnitude selection remove `floor(input width × fraction)` entries per row. Adaptive selection removes `floor(block weight count × fraction)` entries per 128-column block. At 20%, actual deletion counts are 628,736 and 629,136 respectively, so small differences across selection algorithms should not be overinterpreted. Every same-mask comparison has identical deletion counts. At 50%, all conditions remove 1,572,864 entries.

## Analysis

1. Input distribution matters: Wanda outperforms magnitude selection on this matrix, supporting data-dependent observation tools. This does not generalize to every layer or model.
2. Compensation has an independent effect: held-out error falls even with fixed deletion locations, supporting compensation as a separate tool family.
3. Weight distance does not measure functional damage directly: in the 50% magnitude condition, relative squared weight change rises from 6.22% to 13.18% after compensation, while held-out output error falls from 8.87% to 4.16%.
4. A calibration generalization gap remains: the 50% adaptive compensated condition has 1.35% fitting error and 2.09% held-out error. Larger calibration sets and cross-task analysis are needed.
5. Full generation capability and actual sparse-execution costs require separate evaluation. Lower local reconstruction error does not establish accuracy gains, domain-capability preservation, or model acceleration.

All 12 conditions completed. The script records 13.68 seconds of CPU time without using a GPU. Results, protocol, and source snapshots are in `artifacts/toolkit/reconstruction-v1/Qwen3-0.6B/`; the runner is `src/run_reconstruction_study.py`. Large activation arrays remain on the server, bound to the complete records through SHA256 checksums.
