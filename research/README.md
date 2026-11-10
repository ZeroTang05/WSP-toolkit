# Research index

See [Project status](../PROJECT_STATUS.md) for current progress. The [three-stage roadmap](roadmap.md) covers the open-source general toolkit, 0.6B model evaluation, and 4B professional multi-agent workflows.

| Topic | Entry point |
|---|---|
| Implemented tools and examples | [Tool inventory](../docs/tool-inventory.md), [detailed design](../docs/tool-design.md) |
| Literature behind the methods | [Method foundations](methods/method-foundations.md), [literature expansion](methods/literature-expansion.md), [tool semantics](methods/tool-semantics.md) |
| Professional workflows and role models | [Professional workflow protocol](protocols/professional-workflow-protocol.md), [role programming protocol](protocols/multiagent-role-programming.md) |
| Experimental controls, splits, and cost conventions | [Protocol index](protocols/README.md) |
| Results and analysis | [Research reports](reports/README.md), [public results](../docs/benchmarks.md) |
| Original-work reproduction and edit-direction analysis | [MIT WSP reproduction protocol](protocols/mit-wsp-reproduction.md) |

`acquisition/` contains data acquisition scripts, `environments/` contains experiment environment definitions, and `validation/` contains reference implementation checks.

Experiment executors are in `src/`; the general toolkit is in `src/weight_programming/`. The tool design describes 34 items and 47 methods, and a complete GSM8K method comparison is available. Controlled experiments on autonomous model programming, collaboration gains in 4B professional workflows, and role replacement remain part of the research roadmap.

Historical plans, earlier progress and drafts are kept in a local archive. Dated server-job snapshots are kept in a local handover directory and reflect their stated verification times. See [Project maintenance](../docs/maintaining.md) for directory and source maintenance practices.
