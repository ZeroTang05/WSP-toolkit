# 研究索引

当前进展见[项目状态](../PROJECT_STATUS.md)。研究按[三阶段路线](roadmap.md)推进：开源通用工具库、0.6B 效果验证、4B 专业多智能体工作流。

| 要找什么 | 入口 |
|---|---|
| 已实现工具及运行示例 | [工具清单](../docs/tool-inventory.md)、[详细设计](../docs/tool-design.md) |
| 方法的论文来源 | [方法依据](methods/method-foundations.md)、[文献扩展](methods/literature-expansion.md)、[工具语义](methods/tool-semantics.md) |
| 专业工作流与角色模型研究 | [专业工作流协议](protocols/professional-workflow-protocol.md)、[角色编程协议](protocols/multiagent-role-programming.md) |
| 实验对照、数据划分和成本约定 | [实验协议目录](protocols/README.md) |
| 实验结果与分析 | [研究报告目录](reports/README.md)、[公开结果](../docs/benchmarks.md) |
| 原作复现与编辑方向分析 | [MIT WSP 复现协议](protocols/mit-wsp-reproduction.md) |

`acquisition/` 保存数据获取脚本，`environments/` 保存实验环境定义，`validation/` 保存参考实现核对脚本。

实验执行器继续位于 `src/`；通用工具库位于 `src/weight_programming/`。当前工具设计共 34 个条目、47 种方法，已有完整 GSM8K 方法对照。大模型自主编程的对照实验，以及 4B 专业工作流的协作收益和角色替换收益，按研究路线继续验证。

历史计划、旧进度及早期草稿保存在本地归档目录。按日期记录的服务器作业快照保存在本地交接目录，每份快照以其标注的核对时间为准。目录用途与源码维护方式见[项目维护说明](../docs/maintaining.md)。
