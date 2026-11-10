# Persistent CPU analysis kernel and model forward connection

This report preserves the experimental settings, results, and status at the time of writing. For the current project progress, see [Project Status](../../PROJECT_STATUS.md).

2026-11-07. Added `cpu_kernel.py`, `kernel_worker.py`, and `programming_session.py`. These independently reconstruct the operational capabilities described in the [MIT Experiment 8 tool protocol](../references/mit-wsp/exp8_prompt.txt); they do not introduce a new pruning algorithm.

## Running capabilities

A kernel using the real Qwen3-0.6B model started successfully on the Linux server. It can read 311 tensor names and retain variables across code blocks. Writes to the base model fail with a read-only error, external network connections return network unreachable, and the project reference directory and GPU devices are not visible.

`CPUKernel` creates a new root file system, process/network namespace and device directory through bubblewrap. It mounts system runtime libraries, virtual environments, the frozen toolkit, read-only models, read-only authorized exercises, and writable workspaces. Project files, reference edits, final evaluation sets, credentials, and budget ledgers are not mounted. It refuses to start without Linux bubblewrap support.

A session permits up to 300 executions, each with a 900-second limit and a 96 GiB address space. It returns the first 10,000 characters, converts them to ASCII, and automatically displays the last expression. Exceptions also cost one code execution. Variables are retained after ordinary exceptions; hard timeouts terminate the kernel and record the loss of variables. Rebuilding it requires an explicit action, with no silent restart or budget reset. The host saves code, returns, execution counts and source code hashes; the record is closed on exit.

Disabling gradients by default does not prove that arbitrary Python code avoids training. Prohibited training, iterative fitting, and network simulation still require review of execution traces, and operating system isolation cannot be claimed to prove algorithmic semantics. Python output and exceptions are truncated and caught; output from native extensions that write directly to file descriptors is discarded.

## Analyze-Edit-Observe Loop

`ProgrammingSession` connects the kernel, `WeightStore` and `ForwardService`: the kernel submits the `/work/edited` proposal; the host reads and verifies the snapshot; runs budgeted forward passes outside the kernel; copies successful observations to `/work/observations/request_XXXX`. Authoritative budgets, observations, and hashes remain in the unmounted controller. Writable copies do not alter the host evidence.

The channel integration check passed: the kernel edit enables subsequent requests to bind new variant IDs; the base weights are restored after the forward pass; the returned MLP array can be read by the kernel. The check also covers read-only models, external networks, GPU devices, truncation, budgets, and hard timeouts. The small-model check establishes the tool channel, not an improvement in model ability.

```python
from weight_programming.programming_session import ProgrammingSession

with ProgrammingSession(factory, model_dir, practice_dir, practice_set,
                        work_dir, control_dir, python=venv_python) as session:
    session.execute("# Inspect a candidate matrix\nw=load(NAMES[1]); print(w.shape)")
    result = session.forward({
        "type": "teacher_forced", "model": "original", "texts": texts,
        "return": {"hidden_states": [0, 4], "mlp_act": [3]}
    })
    session.execute("import numpy as np\nx=np.load('/work/observations/request_0001/mlp_3.npy')\nprint(x.shape)")
```

The factory receives the host WeightStore and returns a supported model backend. The four data directories must not be nested in each other, and the controller must be a new directory. Independent discovery also requires new programmer context, frozen prompts, and accessible materials; this session has already read the reference edit and cannot serve as a blind discoverer.

## Base model runtime

Qwen3.5-0.8B revision `2fc06364715b967f1860aea9cf38778875588b17` was downloaded as 12 files totaling 1,769,978,895 bytes, with a SHA256 recorded for each file. The new `qwen35_backend.py` uses the complete Transformers 5.3.0 ConditionalGeneration checkpoint, preserves `model.language_model.*` coordinates, and distinguishes full from linear attention.

The adapter passed the hybrid-architecture check and completed 32 real base-model observations; see [qwen35-observation-study.md](qwen35-observation-study.md). The old 4.57.6 environment remains unchanged, and the new environment is the separate `.venv-wsp`. The Torch2.5.1 CUDA12.1 solution is based on [official installation table](https://pytorch.org/get-started/previous-versions/#v251) and [Transformers 5.3 dependencies](https://github.com/huggingface/transformers/blob/v5.3.0/setup.py). Numerical and throughput comparability with vLLM still requires independent verification.
