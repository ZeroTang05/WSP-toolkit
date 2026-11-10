"""Create a local research report from downloaded authoritative experiment JSON."""
import json
from pathlib import Path


def render(root):
    evidence=root/'artifacts/evidence'
    lines=['# idea4: Paper-Derived Tools and Pilot Results','',
        'Date: 2026-11-06. All experiments ran through SSH MCP on GPU 0 of Luck-Medium; the other project on GPU 1 was left unchanged.', '',
        'This pilot implemented paper-derived tools, checked their numerical behavior, and attempted actual parameter edits. The data consists of synthetic integer arithmetic and does not establish broad mathematical, coding, or professional-domain capability. Agent performance relative to ordinary search was not tested.','',
        '## Method Sources and Implementations','',
        '- **Wanda**: Prune each output row using weight magnitude multiplied by the input activation L2 norm. Activation statistics and masks were compared against a pinned version of the authors\' code. Qwen3 and domain calibration data are adaptations. [Paper](https://arxiv.org/abs/2306.11695).',
        '- **Grouped MLP Taylor pruning**: Adapt the grouping and gradient-importance approach from LLM-Pruner to gated MLPs. This does not reproduce its full dependency graph or all estimators. [Paper](https://arxiv.org/abs/2305.11627).',
        '- **LoRA SFT**: Freeze the base parameters and train rank=8, alpha=16 low-rank q/v factors for 40 AdamW steps. V2 supervises both the answer and EOS. [Paper](https://arxiv.org/abs/2106.09685).',
        '- **Task Arithmetic**: Use the saved difference between a trained model and its base, applying θ_base + 0.5(θ_tuned−θ_base). The edit direction comes from actual post-training. [Paper](https://arxiv.org/abs/2212.04089).','',
        '## Data and Selection Rules','',
        'V2 contains 48 calibration, 64 search, and 128 confirmation examples. It excludes all 466 operand pairs from the earlier pilot and V1. Calibration, search, and confirmation also use disjoint operand pairs, and confirmation uses different wording. Both models share the same V2 data; they are not two independent data samples.', '',
        'Selection was frozen using the search set and 8 English guard sentences before confirmation evaluation. The performance candidate minimizes answer NLL while keeping English NLL within 1.03 times the base model; the base model is also a candidate. Compression candidates must keep answer NLL within 1.05 times and English NLL within 1.03 times the LoRA model. Final confirmation results are not used for tuning.', '',
        'Strict accuracy requires both the correct value and an integer-only response. The base model may explain its answer or reach the generation limit, so this metric includes format compliance and cannot be interpreted as pure arithmetic reasoning. The English guard test contains only 8 additional sentences and does not establish general capability.','']
    for variant,label in [('06','Qwen3-0.6B'),('4b','Qwen3-4B-Instruct-2507')]:
        folder=evidence/f'paper{variant}-v2'
        summary=json.loads((folder/'summary.json').read_text())
        selection=json.loads((folder/'selection.json').read_text())
        rows={r['name']:r for r in summary['results']}
        lines += [f'## {label}: Independent V2 Confirmation','',
            '| Variant | Strict accuracy (128 questions) | Answer NLL | Total parameter reduction | Fixed forward time ms | English NLL |',
            '|---|---:|---:|---:|---:|---:|']
        for name in ['original','lora40','taskhalf','wanda10','taylor08']:
            r=rows[name]
            lines.append(f"| {name} | {r['exact_match']*100:.2f}% ({round(r['exact_match']*128)}/128) | {r['answer_nll']:.3f} | {r['parameter_reduction']*100:.2f}% | {r['forward_median_ms']:.2f} | {r['guard_nll']:.3f} |")
        lines += ['',f"Performance candidate selected during search: **{selection['performance_candidate']}**. Compression candidates passing the search proxy constraints: {', '.join(selection['compression_candidates_passing_search_proxy']) or 'none'}. Search selection alone does not confirm retention of all capabilities.",'']
        for name in ['taskhalf','wanda10','taylor08']:
            r=rows[name]
            if 'specialist_delta_ci95' in r:
                low,high=r['specialist_delta_ci95']
                lines.append(f"- {name}: strict accuracy changed by {r['specialist_exact_match_delta']*100:+.2f} percentage points relative to LoRA; paired bootstrap 95% interval [{low*100:+.2f}, {high*100:+.2f}].")
            elif name=='taskhalf':
                low,high=r['paired_bootstrap_delta_ci95']
                lines.append(f"- taskhalf: accuracy changed by {r['exact_match_delta']*100:+.2f} percentage points relative to the base model; paired bootstrap 95% interval [{low*100:+.2f}, {high*100:+.2f}].")
        lines.append('')
    lines += ['## Interpreting the Results','',
        'LoRA and task vectors optimize parameters without reducing the base architecture. Wanda retains dense storage and computation despite unstructured zero weights, so sparsity alone does not demonstrate speedup. Taylor variants remove matrix rows and columns, but total parameter reduction is much smaller than the fraction of MLP channels removed.', '',
        'Timing measures a full forward pass on the same GPU with batch=1, 256 input tokens, logits included, and no KV cache, using 12 samples. This does not measure serving latency end to end. Temperature, clock speeds, and cache history vary across processes, so differences of a few percent do not establish reliable speedup. GPU memory is the peak for this process and test, rather than a complete deployment measurement.', '',
        'V1 preserves evidence of failure: optimizing only answer-digit NLL without EOS supervision can produce repeated digits, and the 16-token limit can truncate explanations. V2 revises the objective and evaluation accordingly, using a new confirmation set that excludes earlier questions. Comparing V1 and V2 accuracy does not establish an algorithmic improvement under matched conditions.', '',
        '## Validation and Deliverables','',
        '- 9 tool tests passed, covering statistics and masks against pinned author code, actual LoRA training with frozen base weights, merge equivalence, answer/EOS label boundaries, rollback, and structural checkpoint saving and reloading.',
        '- Independently read the exported Safetensors files to verify each method\'s edit scope, matrix slices, low-rank update, or task-vector formula, and recheck the original model file hashes.',
        '- Tokenizer JSON files match. In addition, 240 chat templates and 256 answer/guard text encodings were checked individually; validation records are saved with the evidence.',
        '- V2 saves runtime source snapshots, data, protocols, selection records, raw generations, scores, full candidates, and patch metadata.',
        '- Server project: `/home/hdd_1/tangziye/idea4`. Full models are in `runs/paper06-v2/` and `runs/paper4b-v2/`; local `artifacts/evidence/` holds lightweight evidence without copying large models.',
        '- Command entry points: server `.venv/bin/paper-weightlab` and `.venv/bin/weightlab`, or the corresponding Python files. Usage is documented in `PAPERS_RUNBOOK.md`.','',
        '## Research Assessment and Next Steps','',
        'This pilot validates a workflow from paper methods through executable tools, post-training references, actual weight edits, and independent evaluation. It does not establish general training-free weight programming or domain-preserving models with reliable speedup.', '',
        'The next phase should prioritize real domain tasks and adequate guard sets, comparing fixed algorithms, non-LLM search, and agents at matched cost. Structural pruning requires further checks of general-capability loss and deployment benefits. SparseGPT, full LLM-Pruner, DPO/RLHF, and distillation-based recovery remain future extensions in this pilot.','']
    (root/'pilot-results.md').write_text('\n'.join(lines))


if __name__=='__main__':
    render(Path(__file__).resolve().parents[1])
