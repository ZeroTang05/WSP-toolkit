"""Execute the original library ROME and MEMIT in the EasyEdit environment specified by the host."""

import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys


def main(request_path):
    """Read calibration statistics and the parent version, call the upstream library, and export authorized tensors."""
    import torch
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from easyeditor.models.memit import memit_main
    from easyeditor.models.memit.memit_hparams import MEMITHyperParams
    from easyeditor.models.rome import rome_main
    from easyeditor.models.rome.rome_hparams import ROMEHyperParams
    from easyeditor.util.runningstats import CombinedStat, SecondMoment, save_cached_state

    directory = request_path.parent
    request = json.loads(request_path.read_text(encoding="utf-8"))
    checkpoint = request["checkpoint"]
    device = request["device"]
    model = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True, dtype=torch.float32,
                                                 attn_implementation="eager").to(device).eval()
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    overlay = directory / "parent.safetensors"
    if overlay.is_file():
        parameters = dict(model.named_parameters(remove_duplicate=False))
        with torch.no_grad():
            for name, tensor in load_file(overlay).items():
                parameters[name].copy_(tensor)

    stats_dir = directory / "stats"
    dataset_name = "wsp_calibration"
    model_name = model.config._name_or_path.rsplit("/", 1)[-1]
    samples = request["calibration_samples"]
    count = request["calibration_tokens"]
    for layer, name in zip(request["layers"], request["names"]):
        matrix = load_file(directory / f"moment_{layer}.safetensors")["moment"]
        stat = CombinedStat(mom2=SecondMoment(state={"count": count, "mom2": matrix.numpy() * count}))
        module_name = name.removesuffix(".weight")
        path = stats_dir / model_name / (dataset_name + "_stats") / f"{module_name}_float32_mom2_{samples}.npz"
        # Write the cache format of EasyEdit itself, and the algorithm loads the measured statistics according to the normal path.
        save_cached_state(path, stat, {"sample_size": samples})

    output_weight = model.get_output_embeddings().weight
    # EasyEdit finds the vocabulary head by its parameter name. With tied embeddings, that parameter may be registered under the embedding module.
    head_weight_name = next(name for name, parameter in model.named_parameters() if parameter is output_weight)
    common = dict(layers=request["layers"], fact_token="subject_last", v_num_grad_steps=request["steps"],
                  v_lr=request["learning_rate"], v_loss_layer=model.config.num_hidden_layers - 1,
                  v_weight_decay=0.001, clamp_norm_factor=4, kl_factor=0.0625, mom2_adjustment=True,
                  rewrite_module_tmp="model.layers.{}.mlp.down_proj", layer_module_tmp="model.layers.{}",
                  mlp_module_tmp="model.layers.{}.mlp", attn_module_tmp="model.layers.{}.self_attn",
                  ln_f_module="model.norm", lm_head_module=head_weight_name.removesuffix(".weight"), mom2_dataset=dataset_name,
                  mom2_n_samples=samples, mom2_dtype="float32", device=device, model_name=checkpoint, stats_dir=str(stats_dir))
    method = request["method"]
    contexts = request["context_templates"]
    if method == "rome":
        rome_main.CONTEXT_TEMPLATES_CACHE = contexts
        hparams = ROMEHyperParams(**common, alg_name="ROME", context_template_length_params=[])
        editor = rome_main.apply_rome_to_model
        source = Path(rome_main.__file__)
    else:
        memit_main.CONTEXT_TEMPLATES_CACHE = [[contexts[0]], contexts[1:]]
        hparams = MEMITHyperParams(**common, alg_name="MEMIT", layer_selection="all", mom2_update_weight=request["mom2_weight"])
        editor = memit_main.apply_memit_to_model
        source = Path(memit_main.__file__)
    with torch.enable_grad():
        edited, original = editor(model, tokenizer, request["facts"], hparams, return_orig_weights=True)
    if set(original) != set(request["names"]):
        raise ValueError("Unexpected tensors modified by EasyEdit")
    weights = {name: edited.get_parameter(name).detach().cpu().contiguous() for name in request["names"]}
    save_file(weights, directory / "edited.safetensors")
    report = {"library": "EasyEdit", "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "transformers": version("transformers"), "torch": version("torch"), "device": device}
    (directory / "result.json").write_text(json.dumps(report), encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
