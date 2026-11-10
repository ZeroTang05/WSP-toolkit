# OfficeBench integration: pinned data and execution options

2026-11-07. Source data acquisition, file-by-file Git blob verification and scoring configuration separation have been completed, but the 4B workflow has not yet been run.

## Acquired data

Fixed commit `b978b808667c32b52ce19a67ce1def1de9ae02b7`, source [OfficeBench](https://github.com/zlwang-cs/OfficeBench/tree/b978b808667c32b52ce19a67ce1def1de9ae02b7). Downloader `research/acquisition/prepare_officebench.py`, local output `benchmarks/officebench-source-v1/`. All 675 files are consistent with the Git blob SHA in the fixed tree; the compressed package SHA256 has been recorded.

The 300 tasks come from 167 source workspaces and split into 93/95/112 tasks by number of applications. They contain 601 checks: text contains 279, file exists 233, Excel cell value 37, exact match 18, does not contain 19, schedule does not conflict 6, difference contains 6, file does not exist 2, cell comparison 1. A task has multiple checks. Official completion means that all checks have passed. The average check pass rate cannot be used as the overall task completion rate.

The public task view retains the task and user/time context, and strips off the evaluation field; the scoring side configuration is saved separately. All 300 tasks are provisionally designated for open evaluation, not training. Subsequent division must be grouped into at least 167 workspaces, and cross-space file/template duplication must be checked. Subtasks in the same workspace cannot be arbitrarily split between training and test.

## Two important experimental conditions revealed by the source code

First, 18 exact-match checks across 17 tasks reference expected_file. Most of the expected files are in the reference directory of each task, depending on the relative level of the official output directory; task 3-62 places `data/target.pdf` in the original testbed, and the task requires generating report.docx and converting report.pdf. Hiding that expected file would change the officially visible input. It must be reported as a clear controlled variant, and the official input conditions should be retained for comparison. It cannot be reproduced as the official version by default. In both conditions, the evaluation field cannot be passed directly to the Agent.

Second, the [official LLM application](https://github.com/zlwang-cs/OfficeBench/blob/b978b808667c32b52ce19a67ce1def1de9ae02b7/apps/llm_app/llm_query.py) can call `gpt-3.5-turbo-0125`. So "running OfficeBench with 4B" does not automatically mean that only 4B is involved in inference. The main experiment needs to freeze whether the application is available; when retained, each call model, output, token and cost are recorded, and all conditions and permissions are the same. If it is changed to the target 4B endpoint or disabled, it is clearly a change in the implementation protocol of this project. It cannot be done online by a strong model and still attribute the benefits to 4B parameter programming.

## Next step environment and system experiment

Official [Dockerfile](https://github.com/zlwang-cs/OfficeBench/blob/b978b808667c32b52ce19a67ce1def1de9ae02b7/docker/Dockerfile) is based on ubuntu:latest, including LibreOffice and Tesseract; the base image and dependencies need to be fixed and the image digest recorded. Currently, only the source code is downloaded and read, no containers are built/started, and no emails are sent or external models are called.

The task environment only receives the input view and required testbed of a single task, and the scoring end retains the reference and original check. Each run in each condition is reset from the same initial state to avoid file residues from the previous Agent or previous method. The operator interface and file parser are consistent for all methods.

The workflow experiment adopts [professional workflow protocol](professional-workflow-protocol.md): first focus on single Agent, equal budget revision, independent sampling selection and multi-Agent comparison; after confirming the collaboration gain, fix the topology, and then compare role LoRA, shared update, non-LLM same tool search and LLM parameter program. In the first round, the coverage of all 300 items and the application results are reported. The tasks that cannot be executed are retained as failure categories or clearly named subsets and are not eliminated afterwards.

Official checks measure different things. For example, existence checks can only prove the production of documents, but cannot independently prove that professional content is qualified. Therefore, the main score retains the official all-pass, while the file structure, numerical/citation correctness and manual spot checks are also listed. The supplementary indicators and official scores are presented separately.

## 2026-11-07: Actual execution environment and scoring control

A rootless Docker dedicated to the project has been established and no unauthorized system daemon is used. The image was actually built successfully: `sha256:0be7cb49797c1696c3f1a4ca63b45ad1945053635fe0aba8bf0f75503ac3b072`, about 1.41GB; the basic Ubuntu image is fixed by digest, and the pip and dpkg version lists have been saved. The source code is located at `research/environments/officebench/`. This supersedes the earlier status above, when the image had not yet been built.

Added `src/officebench_runtime.py` and container worker, calling the same upstream application script and scoring function. The transport is changed to JSON, a new container is used for each question, network and LLM applications are disabled, and explicit timeout is implemented; the scoring container arranges the reference according to the official relative path, and provides the `/testbed` alias to be compatible with the hard-coded path of the upstream helper. This is an explicitly stated implementation port and is not a proven equivalent to the official leaderboard.

A real mailing-list read, submission, and scoring run was performed. An incorrect placeholder failed; an artificially constructed two-keyword answer passed every check for this example without providing the complete email content. This is an oracle control using scoring knowledge, without using a model, and it can never be used as a positive example for 4B results or role training. It provides a concrete counterexample that official keyword ratings alone cannot prove professional job competence. Control and environmental evidence is at `artifacts/toolkit/officebench-runtime-v1/`. 4B inference/multi-agent comparison has not been run yet.
