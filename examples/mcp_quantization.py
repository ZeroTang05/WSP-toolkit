"""Export GPTQ and AWQ with real calibration text and reload packed checkpoints."""

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


RELOAD = """
import json, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

path = sys.argv[1]
model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, dtype='auto').eval()
tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
ids = tokenizer('The capital of France is', return_tensors='pt')
with torch.inference_mode():
    logits = model(**ids, use_cache=False, logits_to_keep=1).logits
assert torch.isfinite(logits).all()
print(json.dumps({'logit_shape': list(logits.shape), 'next_token': int(logits[0, -1].argmax())}))
"""


async def main(model, dataset, quantization_python, methods):
    with dataset.open(encoding="utf-8") as stream:
        texts = [json.loads(line)["problem"] for line in islice(stream, 8)]
    with TemporaryDirectory(prefix="wsp-quantization-example-") as workspace:
        server = StdioServerParameters(command=sys.executable, args=["-m", "weight_programming.mcp_server",
            "--model", model, "--workspace", workspace, "--allow-edits", "--quantization-python", quantization_python])
        async with stdio_client(server) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()

                async def call(name, arguments):
                    response = await session.call_tool(name, arguments)
                    if response.is_error:
                        raise RuntimeError(response.structured_content or response.content)
                    return response.structured_content

                model_info = await call("describe_model", {})
                assert set(methods) <= set(model_info["compression_methods"])
                for method in methods:
                    result = await call("compress_model", {"method": method, "variant_id": model_info["variant_id"],
                        "calibration_texts": texts, "group_size": 64})
                    assert result["details"]["bits"] == 4 and result["details"]["packed_tensors"] > 0
                    assert result["checkpoint_bytes"] < result["original_checkpoint_bytes"]
                    checkpoint = Path(workspace) / result["checkpoint_path"]
                    loaded = subprocess.run([quantization_python, "-c", RELOAD, str(checkpoint)], check=True,
                                            text=True, stdout=subprocess.PIPE)
                    print(json.dumps({"method": method, "source_bytes": result["original_checkpoint_bytes"],
                                      "checkpoint_bytes": result["checkpoint_bytes"],
                                      "packed_tensors": result["details"]["packed_tensors"],
                                      "reloaded": json.loads(loaded.stdout.strip().splitlines()[-1])}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--quantization-python", required=True)
    parser.add_argument("--method", choices=["gptq", "awq"])
    arguments = parser.parse_args()
    methods = [arguments.method] if arguments.method else ["gptq", "awq"]
    asyncio.run(main(arguments.model, arguments.dataset, arguments.quantization_python, methods))
