# The first public tool library version

This report preserves the experimental settings, results, and status at the time of writing. For the current project progress, see [Project Status](../../PROJECT_STATUS.md).

## 2026-11-09 Tool design list delivery

The current source code contains 34 tool entries in the design catalog, 47 named methods, and common MCP and Python interfaces. Newly integrated capabilities include native PEFT LoRA, classifier snapshots, pairwise activation directions, per-tensor rank compression, tuned lenses, LEACE, AtP*, edge-level circuits, feature attribution graphs, SliceGPT, and packed quantization. Each group of capabilities is accompanied by examples using real models. [Current source code](https://github.com/ZeroTang05/WSP-toolkit/tree/main) · [Implementation list and handover](../../docs/tool-inventory.md)

The source includes Chinese and English homepages, literature references, detailed design, supported scope, environment versions, and run commands. `SOURCE_MANIFEST.json` records release file hashes. The completion status of tool development and subsequent 0.6B and 4B effectiveness experiments are recorded separately.

## 2026-11-09 Early source code update

The public repository’s `main` branch contains the general toolkit source for `0.4.0a5.dev0`. MCP and Python share a registry. The update includes replayable patches, activation intervention, representation and feature analysis, update merging, structural compression, adapter training, and paired evaluation, with Chinese and English homepages and tool descriptions. [Source code submission](https://github.com/ZeroTang05/WSP-toolkit/commit/11b3ab6fe3d63d7e9cce729efbd0acd92f0d2be2) · [Public tool description](https://github.com/ZeroTang05/WSP-toolkit/blob/main/docs/mcp-tools.md)

The candidate installation package built successfully. All 20 existing checks and 9 SmolLM2 examples passed, and the 208 file hashes in the source manifest matched. The commit above was verified on remote `main`. Remaining interfaces and methods continue to be tracked in the tool catalog. [Tool catalog](../../docs/tool-catalog.json)

## First public pre-release

2026-11-07. The source code and installable experimental pre-release have been released, but the overall research goal has not yet been completed.

- Repository: [ZeroTang05/WSP-toolkit](https://github.com/ZeroTang05/WSP-toolkit)
- Fixed source code: [662e0eb268b64a3baf840a15329830e7d9bac1dc](https://github.com/ZeroTang05/WSP-toolkit/tree/662e0eb268b64a3baf840a15329830e7d9bac1dc)
- Package and verification: [v0.4.0a4.dev0](https://github.com/ZeroTang05/WSP-toolkit/releases/tag/v0.4.0a4.dev0)

The public version includes source code, method contracts, tests, acquisition scripts, environment definitions and licenses. The original work’s reference weight edits, datasets and answer files, model checkpoints, and private run logs are stored separately. Original code uses Apache-2.0; third-party code retains its source attribution and license.

## Actual release verification

The public candidate was built after all 158 files were verified individually by hash; the source commit also includes SOURCE_MANIFEST. All 18 core source checks and 2 post-installation Qwen3.5 hybrid-architecture checks passed. With Transformers imports disabled, the core API still imported and calculated input second-order statistics.

The 16 package files in the released wheel match the candidate source byte for byte. GitHub confirmed that the repository was public, the release was not a draft, it was marked as a prerelease, and its target commit was the SHA above. The package downloaded successfully without credentials; its size was 154,269 bytes and its SHA256 was:

```text
69ceaa5c91d1652ecbf6aaf68e7a122f4d50508eacbdea1358cb99574692adef
```

Local evidence is located at `artifacts/releases/0.4.0a4.dev0/`. After the 2026-11-09 reorganization, the project root became the Git working tree, and old source copies and release snapshots were filed as local historical material. For subsequent source code maintenance and export methods, see [Project Maintenance Instructions](../../docs/maintaining.md).

## The research significance of version

Direct weight proposals, budgeted forward passes, persistent CPU analysis, paired activation diagnostics, and SparseGPT local compensation are publicly callable. Conventional post-training and other compression methods are retained in separate interface tiers for research. The permissions document explains why they form a toolchain and why they cannot be mixed in the original untrained track.

Publishing and interface verification are not task performance certificates. The complete 0.6B confirmation matrix, 4B professional-workflow collaboration and parameter benefits, original reference replay, independent discovery, and comparisons of editing directions and mechanisms remain in progress. The open-source toolkit does not claim these unfinished results as established.
