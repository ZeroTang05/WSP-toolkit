"""Validate feature extraction with published SAE or two real model snapshots."""

import argparse
import asyncio
from itertools import islice
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from safetensors.torch import load_file, save_file
import torch
from torch.nn import functional as F

from weight_programming import open_inspection
from weight_programming.artifacts import ArtifactStore
from weight_programming.tensor_store import sha


_SITE = "block.3.mlp.out"
_RELEASE = "https://huggingface.co/EleutherAI/sae-SmolLM2-135M-64x/tree/57ea2cb986e2545844cdd4a5bb2eb39523243494/layers.3.mlp"


def register_dictionary(workspace, encoder_id, tensors, sources, method, activation, **settings):
    """Convert existing weights to the toolkit’s shared coordinates and manifest."""
    folder = workspace / "features" / encoder_id
    folder.mkdir(parents=True)
    path = folder / "weights.safetensors"
    save_file({name: tensor.detach().contiguous() for name, tensor in tensors.items()}, path)
    record = {"format_version": 1, "method": method, "activation": activation, "sources": sources,
              "weights_sha256": sha(path), **settings}
    (folder / "manifest.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def observe(call, variant_id, texts):
    """Record the output of the feedforward module at the same logical site."""
    return call("model_forward", {"variant_id": variant_id, "mode": "teacher_forced", "texts": texts,
                                  "observe": [_SITE]})


def activation_values(workspace, run):
    """Select real tokens only, excluding padding from unequal sentence lengths."""
    artifacts = ArtifactStore(workspace / "forward")
    values, _ = artifacts.array(run["artifacts"]["mlp_out_3"]["artifact_id"])
    valid, _ = artifacts.array(run["artifacts"]["valid_tokens"]["artifact_id"])
    return torch.from_numpy(values[valid]).float()


async def released_sae(model, source, workspace):
    """Run the authors' TopK dictionary on MCP and check sparse values ​​and reconstruction."""
    settings = json.loads((source / "cfg.json").read_text(encoding="utf-8"))
    raw = load_file(source / "sae.safetensors")
    assert settings["d_in"] == 576 and not settings["skip_connection"]
    tensors = {"W_enc.0": raw["encoder.weight"].T, "b_enc": raw["encoder.bias"],
               "W_dec.0": raw["W_dec"], "b_dec.0": raw["b_dec"]}
    server = StdioServerParameters(command=sys.executable,
                                   args=["-m", "weight_programming.mcp_server", "--model", str(model),
                                         "--workspace", str(workspace)])
    async with stdio_client(server) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()

            async def call(name, arguments):
                response = await session.call_tool(name, arguments)
                if response.is_error:
                    raise RuntimeError(response.structured_content or response.content)
                return response.structured_content

            model_info = await call("describe_model", {})
            variant = model_info["variant_id"]
            register_dictionary(workspace, "eleuther-smol-layer3", tensors,
                                [{"variant_id": variant, "site_id": _SITE, "center_on_decoder_bias": True}],
                                "sae", "topk", k=settings["k"], provenance=_RELEASE)
            registered = await call("describe_model", {})
            assert registered["feature_dictionaries"][0]["encoder_id"] == "eleuther-smol-layer3"
            run = await observe(call, variant, ["The capital of France is Paris.", "Gravity pulls objects together."])
            report = await call("extract_features", {"encoder_id": "eleuther-smol-layer3",
                                                       "artifact_id": run["artifacts"]["mlp_out_3"]["artifact_id"]})
            inputs = activation_values(workspace, run)
            bias = raw["b_dec"].float()
            pre = F.linear(inputs - bias, raw["encoder.weight"].float(), raw["encoder.bias"].float()).relu()
            top, indices = pre.topk(settings["k"], dim=1)
            reconstruction = (raw["W_dec"].float()[indices] * top[:, :, None]).sum(1) + bias
            mse = float((reconstruction - inputs).square().mean())
            assert abs(mse - report["reconstruction"][0]["mse"]) < 1e-5
            assert 0 < report["mean_active_features"] <= settings["k"]
            values = await call("read_artifact", {"artifact_id": report["artifact_id"]})
            assert values["feature_ids"] == report["feature_ids"]
            assert max(row[0] for row in values["values"]) == report["preview"][0]["max_activation"]
            rows = await call("read_artifact", {"artifact_id": report["rows_artifact_id"], "limit": 1})
            assert rows["records"][0]["position"] == 0
            return {"features": report["feature_count"], "active_mean": report["mean_active_features"],
                    "rows": report["token_rows"], "mse": mse, "top_feature": report["preview"][0]["feature_id"]}


def fit_crosscoder(values):
    """Fit a small dictionary on aligned real activations, retaining the norm penalty for both decoders."""
    width = values.shape[-1]
    scales = width**0.5 / values.norm(dim=-1).mean(dim=0)
    normalized = values * scales[None, :, None]
    generator = torch.Generator().manual_seed(11)
    enc = torch.nn.Parameter(torch.randn(2, width, 128, generator=generator) / width**0.5)
    dec = torch.nn.Parameter(enc.detach().transpose(1, 2).clone())
    b_enc = torch.nn.Parameter(torch.zeros(128))
    b_dec = torch.nn.Parameter(normalized.mean(0))
    optimizer = torch.optim.AdamW([enc, dec, b_enc, b_dec], lr=0.003)
    for _ in range(60):
        optimizer.zero_grad(set_to_none=True)
        latent = (torch.einsum("nsd,sdf->nf", normalized, enc) + b_enc).relu()
        reconstructed = torch.einsum("nf,sfd->nsd", latent, dec) + b_dec
        penalty = (latent * dec.norm(dim=-1).sum(0)).mean()
        loss = F.mse_loss(reconstructed, normalized) + 0.001 * penalty
        loss.backward()
        optimizer.step()
    tensors = {"b_enc": b_enc}
    for side in range(2):
        tensors.update({f"W_enc.{side}": enc[side], f"W_dec.{side}": dec[side], f"b_dec.{side}": b_dec[side]})
    return tensors, scales.tolist()


def paired_crosscoder(model, dataset, workspace):
    """Collect common inputs for the real parent model and weight candidate, and verify delegation to the comparison tool."""
    tools = open_inspection(model, workspace, allow_edits=True)
    call = tools.registry.call
    base = tools.variant_id
    with dataset.open(encoding="utf-8") as stream:
        texts = [json.loads(line)["problem"] for line in islice(stream, 8)]
    listed = call("list_tensors", {"variant_id": base, "layer": 3})
    tensor = next(item["name"] for item in listed["tensors"] if item["name"].endswith(".mlp.down_proj.weight"))
    region = [[0, 8], [0, 16]]
    weight = call("read_tensor", {"variant_id": base, "name": tensor, "region": region})
    edit = {"tensor": tensor, "op": "replace", "region": region,
            "values": [[0.5 * value for value in row] for row in weight["values"]]}
    patch = call("create_patch", {"parent_variant_id": base, "method": "crosscoder_example", "edits": [edit]})
    candidate = call("apply_patch", {"parent_variant_id": base, "patch_id": patch["patch_id"]})["variant_id"]
    variants = [base, candidate]
    calibration = [activation_values(workspace, observe(call, variant, texts[:6])) for variant in variants]
    assert not torch.equal(*calibration)
    tensors, scales = fit_crosscoder(torch.stack(calibration, dim=1))
    sources = [{"variant_id": variant, "site_id": _SITE, "input_scale": scale}
               for variant, scale in zip(variants, scales)]
    register_dictionary(workspace, "paired-layer3", tensors, sources, "crosscoder", "relu",
                        provenance="Local small fit following https://transformer-circuits.pub/2024/crosscoders/index.html")
    refs = [observe(call, variant, texts[6:])["artifacts"]["mlp_out_3"]["artifact_id"] for variant in variants]
    options = {"encoder_id": "paired-layer3", "left_artifact_id": refs[0], "right_artifact_id": refs[1]}
    comparison = call("compare_models", {"left_variant_id": base, "right_variant_id": candidate,
                                         "views": ["crosscoder"], "crosscoder": options})["crosscoder"]
    assert comparison["source_artifact_ids"] == refs
    assert [item["variant_id"] for item in comparison["reconstruction"]] == variants
    assert abs(sum(comparison["preview"][0]["relative_decoder_norms"]) - 1) < 1e-6
    return {"features": comparison["feature_count"], "heldout_rows": comparison["token_rows"],
            "mse": [item["mse"] for item in comparison["reconstruction"]]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--sae", type=Path)
    mode.add_argument("--dataset", type=Path)
    args = parser.parse_args()
    with TemporaryDirectory(prefix="wsp-feature-example-") as directory:
        root = Path(directory)
        result = (asyncio.run(released_sae(args.model, args.sae, root)) if args.sae is not None
                  else paired_crosscoder(args.model, args.dataset, root))
        print(json.dumps(result, indent=2))
