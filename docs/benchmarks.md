# Benchmark results · 实验结果

## Qwen3-0.6B on GSM8K

All five conditions used the same 1,319 test questions. The updates came from three GSM8K training seeds with rank-8 LoRA on the query and value projections. LoRA represents a weight change using two smaller matrices. The comparison applies the combined changes to the same base checkpoint.

| Condition | Correct | Accuracy |
|---|---:|---:|
| Original checkpoint | 485 / 1,319 | 36.77% |
| Ordinary average of the three updates | 548 / 1,319 | 41.55% |
| TIES merge | 417 / 1,319 | 31.61% |
| TIES scaled to the ordinary average's norm | **600 / 1,319** | **45.49%** |
| Ordinary average scaled to the TIES norm | 230 / 1,319 | 17.44% |

The norm measures the overall size of an update. Matching norms lets the comparison examine how the direction of the update affects accuracy at the same size. The norm-matched TIES condition scored **8.72 percentage points above the original checkpoint** and **3.94 points above the ordinary average**.

The evaluation used BF16 weights, greedy generation, a maximum of 512 new tokens, and flexible exact-match scoring of extracted answers. The benchmark covers one base checkpoint and one combination of three trained updates. [TIES method](ties-merging.md) · [Full evaluation runner](../src/run_ties_full.py)

## Other verified runs

| Study | Coverage |
|---|---|
| Qwen3-0.6B internal observations | 128 training-derived texts, 27,776 tokens |
| Qwen3.5-0.8B internal observations | 32 texts, 6,528 tokens |
| Qwen3-0.6B activation patching | 232 requests across four controlled input pairs |
| Qwen3-0.6B pruning compensation | 12 conditions measuring reconstruction error in a weight matrix |

The full Qwen3.5 original-versus-edited MATH-500 comparison and the 4B professional workflow comparison remain research tracks. [Evaluation protocol](final-evaluation.md)
