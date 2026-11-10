# Project maintenance and directories

The project root is the sole Git working tree. Tool development, documentation changes, and source commits are maintained here.

## Directories

| Path | Purpose |
|---|---|
| `README.md` | Public introduction and entry point |
| `PROJECT_STATUS.md` | Current progress and handoff entry points |
| `src/weight_programming/` | General-purpose model programming toolkit |
| `src/*.py` | Existing experiment executors and helpers, retaining their execution paths |
| `examples/` | Real-model examples for each group of tools |
| `tests/` | Existing checks |
| `contracts/` | Method configuration and execution conventions |
| `vendor/` | Third-party code with provenance and licenses |
| `docs/` | Installation, tool design, interfaces, method catalog, and public results |
| `research/methods/` | Literature and method semantics |
| `research/protocols/` | Experiment protocols and research controls |
| `research/reports/` | Analyses of completed experiments and release records |
| `research/acquisition/` | Data acquisition scripts |
| `research/environments/` | Experiment environment definitions |
| `research/validation/` | Numerical checks against reference implementations |
| `tools/` | Maintenance tools, including source export |
| `artifacts/`, `benchmarks/` | Local experiment results, validation records, packages, and data |
| `handoff/` | Local dated experiment handoffs and job snapshots |
| `archive/` | Local historical plans, progress records, homepages, and releases |
| `assets/illustrations/` | Local illustration assets |

## Entry points

- Use the tools: [installation and invocation](mcp-tools.md), [getting started](getting-started.md).
- Inspect the implementation: [tool inventory](tool-inventory.md), [machine-readable catalog](tool-catalog.json).
- Continue research: [research index](../research/README.md), [three-stage roadmap](../research/roadmap.md).
- Check progress: [project status](../PROJECT_STATUS.md).

## Exporting source

`tools/export_source.py` exports a source package from a specified Git commit. It exports `HEAD` by default and also accepts a full commit hash. It reads the version stored in Git; uncommitted local changes and experiment data remain in the working tree.

```sh
python tools/export_source.py --output dist/WSP-toolkit-source.tar.gz
```

A corresponding `.tar.gz.json` file records the commit, file count, and SHA256 content digest. The command stops if the output already exists, avoiding overwriting an earlier release.

`SOURCE_MANIFEST.json` records digests of the source files, excluding the manifest itself. To prepare a release, stage the files to release, update the manifest with `--update-manifest`, then stage and commit the manifest as well. Unstaged experiment changes remain in the working tree.

```sh
python tools/export_source.py --update-manifest
```

## Experiment material

Experiment executors retain their existing `src/` paths. Models, data, frozen experiment configurations, and raw outputs remain in their original locations for replay.

The existing local job closeout tool is `tools/operations/closeout.py`; records continue to be written to `handoff/2026-11-07/`. Remote experiment resources are managed under their existing protocols, and GPU jobs use GPU 0 only.

## Language for contributions

Maintain public documentation, comments and docstrings, and human-facing tool and CLI descriptions in English. Preserve multilingual dataset and tokenization examples where their original language is part of the example.
