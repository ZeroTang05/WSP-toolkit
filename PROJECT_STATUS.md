# 项目当前状态

更新日期：2026-11-09。当前源码版本：`0.4.0a5.dev0`。

WSP-toolkit 让大模型和研究脚本读取、观察、分析、修改及评测另一个模型。设计清单的 **34 个工具条目、47 种方法**已接入，工具开发提交为 `db7c03a`。

## 当前进度

| 工作 | 状态与入口 |
|---|---|
| 工具库开发与开源 | 当前设计清单已实现。见[工具清单](docs/tool-inventory.md)、[详细设计](docs/tool-design.md)和[运行说明](docs/mcp-tools.md) |
| 通用性 | 已接入 Qwen3、Llama、Qwen3.5、DistilBERT，按各后端能力开放功能 |
| 0.6B 效果 | 已有完整 GSM8K 方法对照。见[公开结果](docs/benchmarks.md)和[研究报告目录](research/reports/README.md) |
| 大模型自主编程效果 | 继续完成与固定配方、同预算搜索、普通训练的完整对照 |
| 4B 专业多智能体工作流 | 先验证协作收益，再构造和替换角色专用模型。见[专业工作流协议](research/protocols/professional-workflow-protocol.md)和[角色编程协议](research/protocols/multiagent-role-programming.md) |

当前源码从仓库根目录安装。GitHub 的较早预发布安装包为 `0.4.0a4.dev0`；最新工具接口见源码文档。

## 接手时看这些文件

1. [工具清单](docs/tool-inventory.md)：每项能力、实现位置和真实示例。
2. [工具设计](docs/tool-design.md)：为什么选择这些方法、输入输出以及实现方式。
3. [三阶段研究路线](research/roadmap.md)：后续效果实验与专业工作流目标。
4. [项目维护说明](docs/maintaining.md)：目录用途、源码导出和本地资料位置。

## 当前工作目录

项目根目录就是 Git 工作区，代码和文档直接在这里维护。`src/weight_programming/` 是通用工具库，`examples/` 是真实运行示例。

本机的实验数据和原始结果分别保留在 `benchmarks/`、`artifacts/`；按日期保存的实验交接与作业快照保留在 `handoff/`。这些本地目录独立于公开源码。作业快照以其标注的核对时间为准。

早期计划、旧进度、旧首页和旧发布副本已收进 `archive/`。已完成实验的分析保留在 `research/reports/`，后续实验协议保留在 `research/protocols/`。
