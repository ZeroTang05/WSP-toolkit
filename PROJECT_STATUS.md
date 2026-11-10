# Project status

Updated 2026-11-09. Current source version: `0.4.0a5.dev0`.

WSP-toolkit lets language models and research scripts read, observe, analyze, modify, and evaluate another model. All **34 tool entries and 47 methods** in the design catalog have been integrated; the tool development commit is `db7c03a`.

## Current progress

| Work | Status and entry points |
|---|---|
| Toolkit development and open source | The current design catalog has been implemented. See the [tool inventory](docs/tool-inventory.md), [detailed design](docs/tool-design.md), and [usage guide](docs/mcp-tools.md). |
| Generality | Qwen3, Llama, Qwen3.5, and DistilBERT are integrated, with features exposed according to each backend's capabilities. |
| 0.6B results | A complete GSM8K method comparison is available. See [public results](docs/benchmarks.md) and the [research report index](research/reports/README.md). |
| Autonomous model programming | Complete comparisons against fixed recipes, equal-budget search, and ordinary training remain in progress. |
| 4B professional multi-agent workflow | First validate the benefits of collaboration, then construct and substitute role-specific models. See the [professional workflow protocol](research/protocols/professional-workflow-protocol.md) and [role programming protocol](research/protocols/multiagent-role-programming.md). |

Install the current source from the repository root. The earlier GitHub prerelease package is `0.4.0a4.dev0`; see the source documentation for the latest tool interfaces.

## Where to start

1. [Tool inventory](docs/tool-inventory.md): capabilities, implementation locations, and real examples.
2. [Tool design](docs/tool-design.md): method choices, inputs, outputs, and implementation.
3. [Three-stage research roadmap](research/roadmap.md): future experiments and professional workflow objectives.
4. [Maintenance guide](docs/maintaining.md): directory purposes, source export, and local material.

## Working directory

The project root is the Git working tree, where code and documentation are maintained. `src/weight_programming/` contains the general-purpose toolkit, and `examples/` contains real execution examples.

Local experiment data and raw results are kept in `benchmarks/` and `artifacts/`, respectively; dated experiment handoffs and job snapshots are in `handoff/`. These local directories are separate from the public source. Job snapshots reflect their stated verification times.

Earlier plans, progress records, homepages, and release copies are in `archive/`. Analyses of completed experiments are in `research/reports/`, and future experiment protocols are in `research/protocols/`.
