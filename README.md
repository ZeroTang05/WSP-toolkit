# WSP-toolkit

**Inspect, edit, and evaluate open-weight language models.**

WSP stands for **Weight-Space Programming**. The toolkit lets you read model weights and internal activations, make a targeted change, and compare the resulting behavior with the base model. Edits are saved separately, with a fingerprint that ties each edit to its source checkpoint.

[中文说明](README.zh-CN.md) · [Getting started](docs/getting-started.md) · [Benchmarks](docs/benchmarks.md) · [Releases](https://github.com/ZeroTang05/WSP-toolkit/releases)

## What you can do

| Goal | Tool |
|---|---|
| Read and edit individual weight matrices | `WeightStore` reads safetensors checkpoints and saves checked replacement tensors |
| See how a model processes an input | `ForwardService` runs budgeted generation and records selected activations, token scores, and attention maps |
| Check what a component contributes | `ActivationPatchingStudy` replaces selected activations and measures the output change |
| Explore edits with a model as the programmer | `ProgrammingSession` provides an isolated CPU workspace and records tool calls and budgets |
| Combine trained updates or repair pruning damage | `ties_merge` combines aligned parameter updates. `sparsegpt_reconstruct` compensates for removed weights within a layer |
| Compare a frozen edit with its base | The host-side evaluator records generated answers, accuracy, truncation, and repetition statistics |

The original Weight-Space Programming experiment inspired the isolated workspace and budgeted model tools. Activation patching, pruning, and update merging have their own method references and experiment settings. [Original research](https://rikiyat.github.io/blog/weight-space-programming/) · [Tool permissions and methods](docs/tool-permissions.md)

## A measured result

On the full **1,319-question GSM8K** test set, Qwen3-0.6B went from **485 correct (36.77%)** to **600 correct (45.49%)** after three trained updates were combined with TIES and scaled to the same size as their ordinary average. That is **+8.72 percentage points**. The unscaled TIES update scored **417/1,319 (31.61%)**, so update size mattered in this experiment. [All five conditions and the evaluation setup](docs/benchmarks.md)

## Get started

Use Python 3.10 or later. Install from source to use the interfaces described here.

```sh
git clone https://github.com/ZeroTang05/WSP-toolkit.git
cd WSP-toolkit
python -m pip install .
```

Start with a local safetensors checkpoint and inspect one weight matrix on the CPU. The following command uses the first MLP output matrix in a Qwen3 checkpoint.

```sh
python examples/inspect_weight.py --model /path/to/Qwen3-0.6B --edit-dir ./weight-edits
```

The example prints the matrix name, dimensions, and norm. The norm is a single number that summarizes the size of all its values. For model runs, choose the environment that matches your checkpoint. [Installation and examples](docs/getting-started.md)

| Guide | What it covers |
|---|---|
| [Model programmer](docs/model-programmer.md) | Give a model the inspection and editing tools and record its work |
| [TIES merging](docs/ties-merging.md) | Merge trained updates and control their size |
| [Final math evaluation](docs/final-evaluation.md) | Evaluate frozen weights on a separate test set |
| [Benchmark results](docs/benchmarks.md) | Compare all tested conditions and see which studies are still open |

## License and references

Project code is released under [Apache-2.0](LICENSE). [Third-party notices](THIRD_PARTY.md) document the upstream code and its licenses. Model checkpoints and datasets have their own terms.
