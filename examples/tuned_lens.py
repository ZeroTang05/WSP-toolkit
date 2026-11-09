"""用真实模型激活拟合调优透镜，并在不同文本上读出预测。"""

import argparse
from itertools import islice
import json
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory

from weight_programming import open_inspection


def main(model, dataset):
    """在一个工作区保存校准变换，并通过共用工具入口读出。"""
    with dataset.open(encoding="utf-8") as stream:
        texts = [json.loads(line)["problem"] for line in islice(stream, 28)]
    with TemporaryDirectory(prefix="wsp-tuned-lens-") as workspace, open_inspection(model, workspace) as tools:
        call = tools.registry.call
        model_info = call("describe_model", {})
        first = "block.0.residual_out"
        final = f"block.{model_info['layer_count'] - 1}.residual_out"
        args = {"variant_id": model_info["variant_id"], "mode": "teacher_forced", "observe": [first, final]}
        fit_run = call("model_forward", {**args, "texts": texts[:24]})
        fitting = tools.representations.fit_tuned_lens(fit_run["run_id"], [first], list(range(24)))
        review = call("model_forward", {**args, "texts": texts[24:]})
        request = {"run_id": review["run_id"], "site_ids": [first, final], "topk": 3}
        direct = call("scan_lens", request)
        tuned = call("scan_lens", {**request, "method": "tuned_lens", "decoder_ref": fitting["decoder_ref"]})
        results = [call("read_artifact", {"artifact_id": report["report_artifact_id"]})["records"]
                   for report in (direct, tuned)]
        before, after = [mean(row["kl_from_final"] for row in rows if row["site_id"] == first) for rows in results]
        assert [row for row in results[0] if row["site_id"] == final] == [row for row in results[1] if row["site_id"] == final]
        assert fitting["losses"][first]["final_kl"] < fitting["losses"][first]["initial_kl"] and after != before
        binding = call("read_artifact", {"artifact_id": fitting["decoder_ref"], "region": [[0, 1], [0, 1], [0, 1]]})
        assert binding["site_ids"] == [first] and binding["variant_id"] == model_info["variant_id"]
        print(json.dumps({"fit": fitting["losses"], "review_samples": len(texts[24:]),
                          "review_kl": {"direct": before, "tuned": after}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    arguments = parser.parse_args()
    main(arguments.model, arguments.dataset)
