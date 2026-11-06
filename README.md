# idea4-weightlab

Experimental, paper-grounded tools for inspecting and programming open-weight
language models. The workflow is **observe → intervene → edit/compress → repair
→ validate**. Each method keeps its assumptions and evidence separate.

## Install

Python 3.10 or newer. Use separate environments for the pinned model backends.

```sh
python -m pip install .                    # tensor and reconstruction tools
python -m pip install '.[qwen3,math]'       # Transformers 4.57.6: Qwen3 / Llama
python -m pip install '.[qwen35,math]'      # Transformers 5.3.0: Qwen3.5
python -m pip install '.[analysis]'         # original CPU numerical stack
```

Qwen3.5 requires Torch >=2.4. Select the appropriate PyTorch build for your
hardware before installing. The isolated CPU kernel additionally requires Linux
bubblewrap. Models, practice sets and device scheduling are supplied by the host.

## Implemented core

| Problem | Tools | Method basis |
|---|---|---|
| Read and propose direct edits | WeightStore, EditPolicy, directional projections | MIT WSP experiment-8 contract; explicit tensor algebra |
| Observe behavior and internal states | ForwardService, pinned HF backends, budget ledger | MIT WSP generation / teacher-forced interface |
| Analyze in a persistent CPU workspace | CPUKernel, ProgrammingSession, tool_schemas, dispatch_tool | Original CPU-kernel + forward-service separation |
| Let a fresh model use the tools | programmer.run_programmer, source runner | Independent driver; full transcripts, live budgets and own-note context rollovers |
| Test a local activation's causal effect | ActivationPatchingStudy | Zhang & Nanda, ICLR 2024; matched clean/corrupt/patched runs |
| Compensate pruning damage | input_hessian, sparsegpt_reconstruct | SparseGPT, ICML 2023; declared local reconstruction adaptation |

```python
from weight_programming import WeightStore, project_output_direction

store = WeightStore(model_dir, edit_dir)
weight = store.load_original(parameter_name)
# direction, selected_columns and strength must come from an explicit experiment.
proposal = project_output_direction(weight, direction, selected_columns, strength)
store.save_edit(parameter_name, proposal)
```

Edits are proposals tied to a source fingerprint; numerical validity does not
establish behavioral benefit. The two HF backends preserve raw residual, MLP
input and attention semantics. Qwen3.5 has full-attention maps only at its full
attention layers; its unused MTP branch is recorded explicitly.

## Method boundaries

The original WSP track excludes training and iterative loss fitting. Its tool
dispatcher exposes persistent CPU analysis and budgeted forward requests.
Activation intervention, SparseGPT reconstruction and training-based methods
belong to explicitly separate tracks. See [permission and method logic](docs/tool-permissions.md).

Existing research modules also implement or adapt Wanda, MLP structural pruning,
LoRA references, task-update arithmetic, model soups, update-rank compression and
BI-based depth selection. They remain research interfaces under `src/`, with
method contracts under `contracts/`; they are not all promoted to the core API.
Run these scripts with `PYTHONPATH=src` and inspect their CLI help.

## Evidence and examples

- Real Qwen3-0.6B observations: 128 training-derived texts, 27,776 tokens.
- Local pruning/compensation: 12 real-matrix conditions; this is reconstruction
  evidence, not whole-model accuracy or sparse-kernel acceleration.
- Real Qwen3.5-0.8B observations: 32 texts, 6,528 tokens, verified source tensors.
- Paired activation maps: 232 real 0.6B queries / 696 forwards on four authored
  factual pairs; this is a diagnostic example, not a discovered complete circuit.

Reproduction scripts include `run_forward_observation_study.py`,
`run_reconstruction_study.py`, `run_qwen35_observation.py` and
`run_activation_patching_study.py`. They require external models and authorized
construction data. Professional OfficeBench acquisition and a declared execution
port are included as research code; multi-agent improvement is an open experiment.

## Validation

In the qwen3 environment, install `.[test,math]` and run:

```sh
PYTHONPATH=src:tests python -m unittest test_weight_programming test_forward_service test_wsp_grading test_reconstruction test_cpu_kernel test_programming_session test_activation_patching test_tool_api
```

In the separate qwen35 environment:

```sh
PYTHONPATH=src:tests python -m unittest test_qwen35_backend
```

Kernel tests need Linux bubblewrap and a virtualenv Python. Verification covers
the stated interfaces and local numerical properties; it is not a certificate
of downstream capability, efficiency or independent MIT-effect reproduction.

For fresh model-programmer usage and experiment boundaries, see
[model-programmer guide](docs/model-programmer.md). Its lifecycle checks run as
`PYTHONPATH=src:tests python -m unittest test_programmer test_programming_session`.

## Sources and license

- [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)
  and [author experiment logs / contract](https://github.com/RikiyaT/lm_understand-logs).
- [SparseGPT](https://arxiv.org/abs/2301.00774v3),
  [activation patching methodology](https://arxiv.org/abs/2309.16042v2),
  [Wanda](https://arxiv.org/abs/2306.11695),
  [LLM-Pruner](https://arxiv.org/abs/2305.11627).
- [LoRA](https://arxiv.org/abs/2106.09685),
  [Task Arithmetic](https://arxiv.org/abs/2212.04089),
  [Model Soups](https://arxiv.org/abs/2203.05482),
  [ShortGPT](https://arxiv.org/abs/2403.03853),
  [SVD-LLM](https://arxiv.org/abs/2403.07378).

Original project code: Apache-2.0. Upstream notices remain in
[THIRD_PARTY.md](THIRD_PARTY.md) and their source directories. Models, datasets
and research reference material retain their own terms.
