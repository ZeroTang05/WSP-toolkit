"""Compress the residual width of a real Llama model, then reload it with SliceGPT."""

import argparse
import asyncio
from itertools import islice
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from transformers import AutoTokenizer


RELOAD = """
import json, runpy, sys, torch
worker, checkpoint, ids = sys.argv[1:]
load = runpy.run_path(worker)['load_sliced_checkpoint']
model = load(checkpoint)
with torch.inference_mode():
    logits = model(torch.tensor([json.loads(ids)]), use_cache=False).logits
assert torch.isfinite(logits).all()
print(json.dumps({'vocab_size': logits.shape[-1], 'next_token': int(logits[0, -1].argmax()),
                  'parameters': sum(p.numel() for p in model.parameters())}))
"""


async def main(model, dataset, slicegpt_python, target_width):
    with dataset.open(encoding="utf-8") as stream:
        texts = [json.loads(line)["problem"] for line in islice(stream, 12)]
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    ids = tokenizer.encode("The capital of France is", add_special_tokens=False)
    worker = Path(__file__).resolve().parents[1] / "src" / "weight_programming" / "slicegpt_worker.py"
    with TemporaryDirectory(prefix="wsp-slicegpt-example-") as workspace:
        server = StdioServerParameters(command=sys.executable, args=["-m", "weight_programming.mcp_server",
            "--model", model, "--workspace", workspace, "--allow-edits", "--slicegpt-python", slicegpt_python])
        async with stdio_client(server) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()

                async def call(name, arguments):
                    response = await session.call_tool(name, arguments)
                    if response.is_error:
                        raise RuntimeError(response.structured_content or response.content)
                    return response.structured_content

                model_info = await call("describe_model", {})
                assert "slicegpt" in model_info["compression_methods"]
                result = await call("compress_model", {"method": "slicegpt", "variant_id": model_info["variant_id"],
                    "calibration_texts": texts, "target_hidden_size": target_width})
                assert result["details"]["hidden_size"] == target_width
                assert result["parameters"] < result["original_parameters"]
                checkpoint = Path(workspace) / result["checkpoint_path"]
                loaded = subprocess.run([slicegpt_python, "-c", RELOAD, str(worker), str(checkpoint), json.dumps(ids)],
                                        check=True, text=True, stdout=subprocess.PIPE)
                report = json.loads(loaded.stdout.strip().splitlines()[-1])
                assert report["parameters"] == result["parameters"] and report["vocab_size"] == model_info["vocab_size"]
                print(json.dumps({"original_parameters": result["original_parameters"], "parameters": result["parameters"],
                                  "original_width": result["details"]["original_hidden_size"], "width": target_width,
                                  "checkpoint_bytes": result["checkpoint_bytes"], "reloaded": report}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--slicegpt-python", required=True)
    parser.add_argument("--width", type=int, default=384)
    args = parser.parse_args()
    asyncio.run(main(args.model, args.dataset, args.slicegpt_python, args.width))
