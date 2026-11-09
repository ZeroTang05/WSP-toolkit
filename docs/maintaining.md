# 项目维护与目录说明

项目根目录是唯一的 Git 工作区。工具开发、文档修改和源码提交都从这里完成。

## 目录

| 路径 | 用途 |
|---|---|
| `README.md`、`README.zh-CN.md` | 对外介绍与使用入口 |
| `PROJECT_STATUS.md` | 当前进度与接手入口 |
| `src/weight_programming/` | 通用模型编程工具库 |
| `src/*.py` | 已有实验执行器和辅助模块，保留运行路径 |
| `examples/` | 各组工具的真实模型示例 |
| `tests/` | 已有检查 |
| `contracts/` | 方法配置与执行约定 |
| `vendor/` | 保留来源和许可证的第三方代码 |
| `docs/` | 安装、工具设计、接口、方法目录与公开结果 |
| `research/methods/` | 文献依据和方法语义 |
| `research/protocols/` | 实验协议与研究对照 |
| `research/reports/` | 已有实验的分析报告与发布记录 |
| `research/acquisition/` | 数据获取脚本 |
| `research/environments/` | 实验环境定义 |
| `research/validation/` | 与参考实现的数值核对脚本 |
| `tools/` | 源码导出等维护工具 |
| `artifacts/`、`benchmarks/` | 本地实验结果、验证记录、安装包和数据 |
| `handoff/` | 本地按日期保存的实验交接与作业快照 |
| `archive/` | 本地旧计划、旧进度、旧首页、旧发布副本 |
| `assets/illustrations/` | 本地插图素材 |

## 常用入口

- 使用工具：[安装与调用](mcp-tools.md)、[上手指南](getting-started.md)。
- 查看实现：[工具清单](tool-inventory.md)、[机器可读目录](tool-catalog.json)。
- 继续研究：[研究索引](../research/README.md)、[三阶段路线](../research/roadmap.md)。
- 查看当前进度：[项目状态](../PROJECT_STATUS.md)。

## 导出源码

`tools/export_source.py` 从指定 Git 提交导出源码包。默认导出 `HEAD`，也可以传完整提交号。它读取 Git 中保存的版本，本地未提交的修改和实验数据保留在工作区。

```sh
python tools/export_source.py --output dist/WSP-toolkit-source.tar.gz
```

源码包旁会保存同名 `.tar.gz.json` 文件，记录提交号、文件数量和 SHA256（文件内容校验摘要）。已有输出文件会使命令停止，避免覆盖旧发布包。

`SOURCE_MANIFEST.json` 记录源码文件的摘要，清单自身除外。准备发布时，先暂存本次要发布的文件，用 `--update-manifest` 更新清单，再把清单一起暂存和提交。暂存区是 Git 保存本次提交内容的位置；未暂存的实验修改会保留在工作区。

```sh
python tools/export_source.py --update-manifest
```

## 实验资料

实验执行器仍使用原来的 `src/` 路径。模型、数据、已冻结的实验配置和原始输出保持原位置，便于重放。

本机的既有作业收尾工具位于 `tools/operations/closeout.py`，记录仍写入原来的 `handoff/2026-11-07/`。远程实验另按原协议管理资源，GPU 作业只使用 GPU 0。
