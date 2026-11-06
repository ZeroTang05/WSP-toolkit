# Fresh-context model programmer

`weight_programming.programmer.run_programmer` connects a model callback to a
real `ProgrammingSession`. The callback receives only a fresh conversation,
the actual tool schemas, a reply-token cap and a seed. It returns response text
and measured input/output token counts. Replies contain exactly one fenced
`python`, `forward` (JSON) or `final` block.

The driver records every response and tool result outside the kernel mount.
Context rollovers retain the same live Python kernel, saved edits, budgets,
initial task and the programmer's own `/work/notes.md`. Provider failures are
recorded; lost kernels are not silently restarted. The final record fingerprints
the submitted native-precision tensors. With `require_forward_budget=True`,
early final reports receive an error until the forward budget is used.

Completed observations are exported through a separate read-only mount at
`/work/observations`. The CPU kernel can analyze them but cannot modify the raw
evidence or redirect host export writes through its writable workspace.

## Local runner

Use the separately pinned backend environment and install the analysis stack:

```sh
python -m pip install '.[qwen35,math,analysis]'
PYTHONPATH=src python src/run_weight_programmer.py \
  --target /absolute/path/to/Qwen3.5-0.8B \
  --programmer /absolute/path/to/Qwen3-4B-Instruct-2507 \
  --practice /absolute/path/to/author-practice \
  --output /absolute/path/to/new-experiment
```

Linux bubblewrap, a virtualenv Python and a CUDA GPU are required by this
runner. It locks GPU 0 using `/tmp/idea4-${uid}-gpu-0.lock`; adapt scheduling in
your own runner if needed. Do not use an existing output directory to restart
an experiment. The runner freezes its source, model hashes, practice manifest,
initial prompt and environment before waiting for the device.

Acquire the exact authorized practice files using
`research/acquisition/download_wsp_practice.py`. The original target additionally
requires the manifest produced by `download_wsp_base.py`. No reference editing
solution, previous analysis or held-out evaluation is supplied to the programmer.
The runner's default limits are 300 CPU executions, 50 successful forward
requests, 500 programmer replies, 8,192 tokens per reply and a 32,768-token
programmer context. Target generation retains its separate 32,768-token cap.

The local 4B programmer, independent driver, Hugging Face engine and failure
diagnostics differ from the author's full system. A completed program is a
candidate edit, not a reproduced mathematical score or proof of independence
from unknown pretraining data. Freeze the edits before independent evaluation;
do not use held-out results to continue program selection.
