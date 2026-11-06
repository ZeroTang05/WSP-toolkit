"""Actual-model FP32 conformance and descriptive BF16 backend differences."""
import argparse
import copy
import hashlib
import importlib.metadata
from pathlib import Path
import torch
from weightlab import Lab,write_json,cast_floating_parameters
from delta_bank import DeltaBank,fingerprint
from peft_bank import PeftBank


def tensor_hash(model):
    digest=hashlib.sha256()
    for name,value in model.named_parameters():
        digest.update(name.encode());digest.update(value.detach().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
    p.add_argument('--delta',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True);(a.output/'code').mkdir()
    for name in ['peft_reference_smoke.py','peft_bank.py','delta_bank.py','weightlab.py']:
        (a.output/'code'/name).write_bytes((Path(__file__).parent/name).read_bytes())
    lab=Lab(a.model);base=lab.manifest();cast_floating_parameters(lab.model,torch.float32).requires_grad_(False)
    custom=DeltaBank(lab.model,fingerprint(base['files']));custom.register('reference',a.delta)
    native=PeftBank(copy.deepcopy(lab.model),fingerprint(base['files']),{'reference':a.delta})
    ids=lab.tokenizer('A small numerical question: calculate twelve plus nine.',return_tensors='pt').input_ids.to('cuda')
    before=tensor_hash(native.model);results={}
    for dtype in ['fp32','bf16']:
        if dtype=='bf16':
            cast_floating_parameters(lab.model,torch.bfloat16);cast_floating_parameters(native.model.get_base_model(),torch.bfloat16)
            # Keep native PEFT factors FP32 as in the declared inference recipe.
            for name in native.roles['reference'][0]['targets']:
                module=native.model.get_base_model().get_submodule(name)
                module.lora_A['reference'].float();module.lora_B['reference'].float()
                factors=native.roles['reference'][1]
                with torch.no_grad():
                    module.lora_A['reference'].weight.copy_(factors[name+'.A'].to(module.lora_A['reference'].weight.device))
                    module.lora_B['reference'].weight.copy_(factors[name+'.B'].to(module.lora_B['reference'].weight.device))
        with torch.inference_mode(),custom.activate('reference'):left=lab.model(ids,use_cache=False).logits.float().cpu()
        with torch.inference_mode(),native.activate('reference'):right=native.model(ids,use_cache=False).logits.float().cpu()
        difference=(left-right).abs()
        results[dtype]=dict(max_logit_error=float(difference.max()),mean_logit_error=float(difference.mean()),
            matching_argmax_fraction=float((left.argmax(-1)==right.argmax(-1)).float().mean()))
        if dtype=='fp32':
            if difference.max()>.001:raise AssertionError('FP32 backend conformance failed')
            if tensor_hash(native.model)!=before:raise AssertionError('PEFT mutated parameter values')
    write_json(a.output/'complete.json',dict(status='reference_smoke_complete',peft=importlib.metadata.version('peft'),
        results=results,scope='One actual model/prompt; FP32 parameter values checked immutable; BF16 differences are descriptive, not a generation-quality guarantee'))
    print(results,flush=True)


if __name__=='__main__':main()
