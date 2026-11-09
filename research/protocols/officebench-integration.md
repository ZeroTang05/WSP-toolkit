# OfficeBench固定版本接入：已取得数据与执行研究选择

2026-11-07。已完成源数据获取、逐文件Git blob核对与评分配置分离，尚未运行4B工作流。

## 数据已经落地

固定提交`b978b808667c32b52ce19a67ce1def1de9ae02b7`，来源[OfficeBench](https://github.com/zlwang-cs/OfficeBench/tree/b978b808667c32b52ce19a67ce1def1de9ae02b7)。下载器`research/acquisition/prepare_officebench.py`，本地输出`benchmarks/officebench-source-v1/`。675个文件均与固定树中的Git blob SHA一致；压缩包SHA256已记录。

300项任务来自167个源工作空间，按应用数分为93／95／112项。它们包含601条检查：文本包含279、文件存在233、Excel单元格值37、精确匹配18、不包含19、日程无冲突6、差异包含6、文件不存在2、单元格比较1。一个任务有多条检查，官方完成是所有检查通过，不能把平均检查通过率冒充整任务完成率。

公开任务视图保留任务和用户／时间上下文，剥离evaluation字段；评分端配置单独保存。所有300项暂标为公开评估、未授权训练。后续划分须至少按167个工作空间分组，并检查跨空间文件／模板重复，不能将同一工作空间的子任务任意分到训练和测试。

## 源码揭示的两个重要实验条件

第一，17项任务的18条精确匹配检查引用expected_file。大多数期望文件在各任务reference目录，依赖官方输出目录的相对层级；任务3-62则把`data/target.pdf`放在原始testbed内，任务要求生成report.docx并转换report.pdf。若本项目隐藏这个期望文件，会改变官方可见输入，必须报告为明确的controlled variant，并保留官方输入条件作对照，不能默称官方原样复现。两种条件都不能把evaluation字段直接传给Agent。

第二，[官方LLM应用](https://github.com/zlwang-cs/OfficeBench/blob/b978b808667c32b52ce19a67ce1def1de9ae02b7/apps/llm_app/llm_query.py)可调用`gpt-3.5-turbo-0125`。因此“以4B运行OfficeBench”不自动表示只有4B参与推理。主实验需冻结这个应用是否可用；保留时记录每次调用模型、输出、token和成本，所有条件权限相同。若改为目标4B端点或禁用，明确属于本项目执行协议变更。不能由强模型在线代做后仍将收益归为4B参数编程。

## 下一步环境与系统实验

官方[Dockerfile](https://github.com/zlwang-cs/OfficeBench/blob/b978b808667c32b52ce19a67ce1def1de9ae02b7/docker/Dockerfile)基于ubuntu:latest、包含LibreOffice和Tesseract；需要将基础镜像及依赖固定后记录image digest。当前只下载与读源码，没有构建／启动容器，也没有发邮件或调用外部模型。

任务环境只接收单项任务的输入视图及所需testbed，评分端保留reference与原始检查。每次条件运行从相同初始状态重置，避免上一Agent或上一方法的文件残留。操作接口和文件解析器对所有方法一致。

工作流实验采用[专业工作流协议](professional-workflow-protocol.md)：先强单Agent、等预算修订、独立采样选择和多Agent比较；确认协作增益后固定拓扑，再比较角色LoRA、共享更新、非LLM同工具搜索和LLM参数程序。首轮报告全300项覆盖情况及分应用数结果，无法执行的任务保留失败类别或明确命名的子集，不事后剔除。

官方检查的用途各异，例如存在性检查只能证明文件产生，不能单独证明专业内容合格。因此主分数保留官方all-pass，同时另列文件结构、数值／引用正确性和人工抽查，补充指标与官方分数分别呈现。

## 2026-11-07：实际执行环境与评分控制

已建立项目专用rootless Docker，不使用无权限的系统daemon。镜像实际构建成功：`sha256:0be7cb49797c1696c3f1a4ca63b45ad1945053635fe0aba8bf0f75503ac3b072`，约1.41GB；基础Ubuntu镜像按digest固定，pip与dpkg版本清单已保存。源码位于`research/environments/officebench/`。这替代上文尚未构建的历史状态。

新增`src/officebench_runtime.py`及容器worker，调用不变的上游应用脚本与评分函数。运输改为JSON、每题新容器、禁用网络及LLM应用、显式超时；评分容器按官方相对路径布置reference，并提供`/testbed`别名以兼容上游部分helper的硬编码路径。属于明确声明的执行移植，不是已证明与官方leaderboard等价。

真实邮件列表读取、成果提交和评分已运行。错误占位答案未通过；人工构造的两个关键词答案通过该例全部检查，但并未提供完整邮件内容。这是使用评分知识的oracle控制、没有使用模型，绝不能作为4B成绩或角色训练正例。它提供了官方关键词评分不能单独证明专业工作胜任的具体反例。控制及环境证据在`artifacts/toolkit/officebench-runtime-v1/`。4B推理／多Agent对照尚未运行。
