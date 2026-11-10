"""Fit LEACE on real part-of-speech annotations, then temporarily intervene on a real model."""

import argparse
import asyncio
import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from conllu import parse_incr
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from transformers import AutoTokenizer


def samples(path, count, used_words):
    """Take prefixes of annotated short sentences, using different word forms for training and evaluation."""
    groups = {"NOUN": [], "VERB": []}
    with path.open(encoding="utf-8") as stream:
        for sentence in parse_incr(stream):
            if any(not isinstance(token["id"], int) for token in sentence):
                continue
            prefix = ""
            for token in sentence[:12]:
                word, label = token["form"], token["upos"]
                prefix += word
                if label in groups and len(groups[label]) < count and word.casefold() not in used_words:
                    groups[label].append((prefix, label))
                    used_words.add(word.casefold())
                    break
                if (token["misc"] or {}).get("SpaceAfter") != "No":
                    prefix += " "
            if all(len(group) == count for group in groups.values()):
                return [item for group in groups.values() for item in group]
    raise ValueError(f"Insufficient noun and verb examples in {path}")


async def main(model, dataset):
    """Fit on the training file, evaluate the readout on the development file, and compare the intervention with a random control."""
    used = set()
    train = samples(dataset / "en_ewt-ud-train.conllu", 16, used)
    review = samples(dataset / "en_ewt-ud-dev.conllu", 8, used)
    labeled = train + review
    texts = [text for text, _ in labeled]
    labels = [label for _, label in labeled]
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    target, = tokenizer.encode(" river", add_special_tokens=False)
    alternative, = tokenizer.encode(" swim", add_special_tokens=False)
    clean, corrupt = "We saw the", "We need to"
    query = len(tokenizer.encode(corrupt, add_special_tokens=False)) - 1
    pair = {"clean_text": clean, "corrupt_text": corrupt, "positions": [query],
            "target_token": target, "alternative_token": alternative}
    with TemporaryDirectory(prefix="wsp-leace-example-") as workspace:
        server = StdioServerParameters(command=sys.executable, args=["-m", "weight_programming.mcp_server",
            "--model", model, "--workspace", workspace, "--allow-interventions"])
        async with stdio_client(server) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()

                async def call(name, arguments):
                    response = await session.call_tool(name, arguments)
                    if response.is_error:
                        raise RuntimeError(response.structured_content or response.content)
                    return response.structured_content

                model_info = await call("describe_model", {})
                assert "concept_erasure" in model_info["intervention_methods"]
                variant, site = model_info["variant_id"], "block.0.residual_out"
                run = await call("model_forward", {"variant_id": variant, "mode": "teacher_forced",
                    "texts": texts, "observe": [site]})
                report = await call("probe_representation", {"artifact_id": run["artifacts"]["hidden_1"]["artifact_id"],
                    "labels": labels, "train_indices": list(range(len(train))),
                    "evaluation_indices": list(range(len(train), len(texts))), "fit_eraser": True})
                erasure = report["concept_erasure"]
                norms = erasure["cross_covariance_norms"]
                assert norms["train"]["before"] > 0 and norms["train"]["after"] < 1e-6 * norms["train"]["before"]
                source = await call("read_artifact", {"artifact_id": erasure["eraser_artifact_id"], "region": [[0, 1], [0, 1]]})
                assert source["variant_id"] == variant and source["site_id"] == site
                request = {"variant_id": variant, "site_id": site, "pair": pair, "method": "concept_erasure",
                           "eraser_artifact_id": erasure["eraser_artifact_id"]}
                actual = await call("run_intervention", request)
                control = await call("run_intervention", {**request, "control_seed": 7})
                assert actual["corrupt"] == control["corrupt"]
                changes = [result["effects"]["logit_difference"]["patched_minus_corrupt"] for result in (actual, control)]
                assert all(math.isfinite(change) for change in changes) and changes[0] != changes[1]
                print(json.dumps({"train_samples": len(train), "review_samples": len(review), "covariance": norms,
                    "probe_accuracy": [report["accuracy"], erasure["evaluation_accuracy"]],
                    "logit_margin_change": {"leace": changes[0], "matched_random": changes[1]}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(main(args.model, args.dataset))
