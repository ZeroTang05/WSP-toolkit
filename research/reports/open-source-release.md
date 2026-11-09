# 首个公开工具库版本

本文保留成文时的实验设置、结果与状态。当前项目进度见[项目状态](../../PROJECT_STATUS.md)。

## 2026-11-09 工具设计清单交付

当前源码包含设计目录的 34 个工具条目、47 种具名方法，以及共用 MCP、Python 接口。新接入的能力包括原生 PEFT LoRA、分类器快照、成对激活方向、逐张量压缩秩、调优透镜、LEACE、AtP*、边级回路、特征归因图、SliceGPT 和打包量化。每组能力附真实模型示例。 [当前源码](https://github.com/ZeroTang05/WSP-toolkit/tree/main) · [实现清单与交接](../../docs/tool-inventory.md)

中英文首页、文献依据、详细设计、支持范围、环境版本和运行命令随源码一起提供。`SOURCE_MANIFEST.json` 记录发布文件摘要。工具开发完成情况与后续 0.6B、4B 效果实验分别记录。

## 2026-11-09 早期源码更新

公开仓库 `main` 已同步 `0.4.0a5.dev0` 的通用工具源码。MCP 和 Python 共用一套登记表。更新包含可重放补丁、激活干预、表示与特征分析、更新合并、结构压缩、适配器训练和配对评测，并附中英文首页和工具说明。 [源码提交](https://github.com/ZeroTang05/WSP-toolkit/commit/11b3ab6fe3d63d7e9cce729efbd0acd92f0d2be2) · [公开工具说明](https://github.com/ZeroTang05/WSP-toolkit/blob/main/docs/mcp-tools.md)

候选安装包构建成功。20 项现有检查和 9 个 SmolLM2 示例通过，源码清单中的 208 个文件摘要一致。远端 `main` 已核对到上述提交。待完成的接口及方法继续在工具目录中跟踪。 [工具目录](../../docs/tool-catalog.json)

## 首个公开预发布

2026-11-07。已公开源码及可安装的实验性预发布，研究总目标仍未完成。

- 仓库：[ZeroTang05/WSP-toolkit](https://github.com/ZeroTang05/WSP-toolkit)
- 固定源码：[662e0eb268b64a3baf840a15329830e7d9bac1dc](https://github.com/ZeroTang05/WSP-toolkit/tree/662e0eb268b64a3baf840a15329830e7d9bac1dc)
- 包与校验：[v0.4.0a4.dev0](https://github.com/ZeroTang05/WSP-toolkit/releases/tag/v0.4.0a4.dev0)

公开版本包含源码、方法契约、测试、采集脚本、环境定义及许可证。原作参考权重编辑、数据集与答案文件、模型检查点和私有运行日志单独保存。原创代码Apache-2.0，第三方代码保留出处与许可证。

## 实际发布验证

公开候选158个文件逐项哈希核对后构建；源码提交另含SOURCE_MANIFEST。核心18项源码检查通过，安装后的Qwen3.5混合架构2项检查通过；禁止Transformers导入时，核心API仍可导入并计算输入二阶统计。

发布wheel的16个包文件与候选源码逐字节一致。GitHub确认仓库public、release非draft且标为prerelease，目标提交为上述SHA。未使用身份凭据的实际下载成功，包154,269字节，SHA256：

```text
69ceaa5c91d1652ecbf6aaf68e7a122f4d50508eacbdea1358cb99574692adef
```

本地证据位于 `artifacts/releases/0.4.0a4.dev0/`。2026-11-09 目录整理后，项目根目录统一为 Git 工作区，旧源码副本与发布快照归入本地历史资料。后续源码维护和导出方式见[项目维护说明](../../docs/maintaining.md)。

## 版本的研究意义

已经能公开调用直接权重提案、预算前向、持久CPU分析、配对激活诊断及SparseGPT局部补偿；经典后训练与其他压缩方法以研究接口分层保留。权限文档解释它们为何能组成一个工具链，又为何不能在原作无训练轨道中任意混用。

发布和接口验证不是任务性能证书。0.6B完整确认矩阵、4B专业工作流协作／参数收益，以及原作参考重放、独立发现和方向机制比较继续推进。当前版本不把这些未完成事项写成开源工具库的已证实效果。
