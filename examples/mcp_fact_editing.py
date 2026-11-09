"""在真实检查点上执行 ROME、MEMIT，再读取修改前后的回答分差。"""

import argparse
import asyncio
from itertools import islice
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main(model, dataset, editor_python, editor_source):
    """用两条反事实检查直接问题、改写和邻近事实。"""
    with dataset.open(encoding="utf-8") as stream:
        calibration = [json.loads(line)["problem"] for line in islice(stream, 12)]
    facts = [{"subject": "Eiffel Tower", "prompt": "The {} is located in", "target_true": "Paris", "target_new": "Rome"},
             {"subject": "Colosseum", "prompt": "The {} is located in", "target_true": "Rome", "target_new": "Paris"}]
    direct = [{"id": f"fact-{i}", "prompt": fact["prompt"].format(fact["subject"]),
               "chosen": " " + fact["target_new"], "rejected": " " + fact["target_true"]}
              for i, fact in enumerate(facts)]
    controls = [{"id": "paraphrase", "prompt": "The city where the Eiffel Tower stands is", "chosen": " Rome", "rejected": " Paris"},
                {"id": "neighbor", "prompt": "The capital of France is", "chosen": " Paris", "rejected": " Rome"}]
    with TemporaryDirectory(prefix="wsp-fact-example-") as workspace:
        args = ["-m", "weight_programming.mcp_server", "--model", model, "--workspace", workspace, "--allow-edits",
                "--allow-gradients", "--fact-editor-python", editor_python, "--fact-editor-source", editor_source]
        async with stdio_client(StdioServerParameters(command=sys.executable, args=args)) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()

                async def call(name, arguments):
                    response = await session.call_tool(name, arguments)
                    if response.is_error:
                        raise RuntimeError(response.structured_content or response.content)
                    return response.structured_content

                description = await call("describe_model", {})
                assert description["fact_editing_methods"] == ["rome", "memit"]
                base = description["variant_id"]
                results = {}
                for method, cases, layers, pairs in (("rome", facts[:1], [5], direct[:1]),
                                                    ("memit", facts, [4, 5], direct)):
                    patch = await call("edit_facts", {"parent_variant_id": base, "method": method, "facts": cases,
                        "layers": layers, "calibration_texts": calibration, "steps": 16, "mom2_weight": 50})
                    assert patch["changed_elements"] > 0
                    candidate = await call("apply_patch", {"parent_variant_id": base, "patch_id": patch["patch_id"]})
                    report = await call("score_preferences", {"variant_id": candidate["variant_id"],
                        "reference_variant_id": base, "pairs": [*pairs, *controls]})
                    assert all(row["relative_margin"] > 0 for row in report["preview"][:len(pairs)]), report["preview"]
                    results[method] = {"changed_elements": patch["changed_elements"],
                                       "margin_changes": {row["pair_id"]: row["relative_margin"] for row in report["preview"]}}
                print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--editor-python", required=True)
    parser.add_argument("--editor-source", required=True)
    options = parser.parse_args()
    asyncio.run(main(options.model, options.dataset, options.editor_python, options.editor_source))
