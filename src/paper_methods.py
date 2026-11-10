"""Paper-defined primitives. Algorithm adaptations are declared explicitly.

Wanda: Sun et al. arXiv:2306.11695; pinned author code in vendor/wanda.
LoRA: Hu et al. arXiv:2106.09685; trained low-rank update, not arbitrary scaling.
Grouped Taylor: LLM-Pruner arXiv:2305.11627 inspired MLP-only adaptation,
not a complete reproduction of its dependency graph and recovery procedure.
"""
import argparse
import hashlib
import math
import random
import re
from pathlib import Path
import json
import time
import torch
from torch import nn
from torch.nn import functional as F
from weightlab import Lab, prepare, write_json,cast_floating_parameters
from verify_artifacts import files_by_key, tensor

WANDA_COMMIT = '8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6'
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class InputEnergy:
    """Matches author WrappedGPT: summed token squares / sequence count."""
    def __init__(self, width, device):
        self.total = torch.zeros(width, dtype=torch.float32, device=device)
        self.sequences = 0

    def add(self, value):
        value = value.detach()
        if value.ndim == 2:
            value = value.unsqueeze(0)
        self.sequences += value.shape[0]
        self.total.add_(value.float().reshape(-1, value.shape[-1]).square().sum(0))

    def energy(self):
        return self.total / self.sequences


def wanda_mask(weight, energy, fraction):
    """Unstructured, per output row, stable ties, exact floor(width*fraction)."""
    if not 0 <= fraction < 1:
        raise ValueError('fraction must be in [0,1)')
    score = weight.abs() * energy.reshape(1, -1).sqrt()
    order = torch.sort(score, dim=-1, stable=True).indices
    mask = torch.zeros_like(weight, dtype=torch.bool)
    mask.scatter_(1, order[:, :int(weight.shape[1]*fraction)], True)
    return mask


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha/rank

    def forward(self, x):
        return self.base(x) + (F.linear(F.linear(x.float(), self.A), self.B)*self.scale).to(x.dtype)

    @torch.no_grad()
    def delta(self):
        return self.scale*(self.B @ self.A)


def training_loss(lab, row, include_eos=False):
    prefix = lab.tokenizer.encode(lab.prefix(row), add_special_tokens=False)
    answer = lab.tokenizer.encode(row['answer'], add_special_tokens=False)
    if include_eos:
        answer.append(lab.tokenizer.eos_token_id)
    ids = torch.tensor([prefix+answer], device='cuda')
    logits = lab.model(ids, use_cache=False).logits[0, len(prefix)-1:-1].float()
    return F.cross_entropy(logits, torch.tensor(answer, device='cuda'))


@torch.no_grad()
def run_wanda(lab, rows, fraction):
    from contract_registry import ContractRegistry,ToolRequest
    registry=ContractRegistry()
    spec=registry.methods['wanda.row_unstructured.v1']
    if spec['reference_commit']!=WANDA_COMMIT:raise ValueError('Reference commit mismatch')
    request=ToolRequest(spec['id'],spec['observation'],spec['selection'],spec['mutation'],'none','calibration')
    records = []
    for i, layer in enumerate(lab.layers):
        modules = {n:m for n,m in layer.named_modules() if isinstance(m, nn.Linear)}
        stats = {n:InputEnergy(m.in_features, m.weight.device) for n,m in modules.items()}
        handles = [m.register_forward_pre_hook(lambda m,x,n=n: stats[n].add(x[0])) for n,m in modules.items()]
        try:
            # Replaying the whole prefix ensures later layers see already-pruned earlier layers.
            # Within one decoder layer, all statistics are gathered before pruning that layer.
            lab.score(rows)
        finally:
            for h in handles:
                h.remove()
        for name, module in modules.items():
            certificate=registry.apply_wanda_linear(request,module,stats[name].energy(),fraction)
            records.append(dict(layer=i,module=name,sequences=stats[name].sequences,**certificate))
        print(json.dumps(dict(stage='wanda', layer=i)), flush=True)
    return dict(method='wanda',contract_id=spec['id'], paper='https://arxiv.org/abs/2306.11695', author_commit=WANDA_COMMIT,
        implementation='per-output-row unstructured Wanda; sequential layer calibration',
        adaptation='Qwen3, variable-length arithmetic calibration instead of author C4 windows; no lm_head pruning',
        fraction=fraction, modules=records)


def taylor_mlp_scores(layers):
    """Aggregate gate, up, and down weight gradients for each feed-forward channel."""
    scores = []
    for layer in layers:
        mlp = layer.mlp
        s = sum((getattr(mlp,n).weight.detach().float()*getattr(mlp,n).weight.grad.float()).abs().sum(dim=dim)
                for n,dim in [('gate_proj',1),('up_proj',1),('down_proj',0)])
        if not torch.isfinite(s).all():raise ValueError('Nonfinite Taylor group importance')
        scores.append(s.cpu())
    return scores


def run_taylor(lab, rows, fraction, include_eos=False):
    """Explicit first-order, elementwise |w*mean_gradient| MLP-group adaptation."""
    lab.model.requires_grad_(False)
    for layer in lab.layers:
        layer.mlp.requires_grad_(True)
    lab.model.zero_grad(set_to_none=True)
    for row in rows:
        (training_loss(lab, row, include_eos)/len(rows)).backward()
    scores = taylor_mlp_scores(lab.layers)
    lab.model.zero_grad(set_to_none=True)
    lab.model.requires_grad_(False)
    keep = lab.masks(scores, fraction)
    # Check structure algebra with FP32 before exporting actual BF16 weights.
    ids = lab.tokenizer('An arithmetic problem has an integer answer.', return_tensors='pt').input_ids.to('cuda')
    cast_floating_parameters(lab.model,torch.float32)
    with torch.inference_mode(), lab.masked(keep):
        reference = lab.model(ids, use_cache=False).logits.float().cpu()
    lab.structural_prune(keep)
    with torch.inference_mode():
        actual = lab.model(ids, use_cache=False).logits.float().cpu()
    error = (reference-actual).abs().max().item()
    if error > 0.001:
        raise AssertionError(f'Structure equivalence error {error}')
    cast_floating_parameters(lab.model,torch.bfloat16)
    return dict(method='grouped_taylor_mlp_adaptation', paper='https://arxiv.org/abs/2305.11627',
        implementation='sum elementwise abs(weight * mean calibration gradient) across coupled MLP group',
        adaptation='MLP only, uniform width; no full dependency discovery or second-order variant; not full LLM-Pruner',
        keep_width=len(keep[0]), fp32_equivalence_max_error=error,
        kept_indices=[v.tolist() for v in keep])


def run_lora(lab, rows, steps, lr, include_eos=False):
    torch.manual_seed(902)
    lab.model.requires_grad_(False)
    wrapped = []
    for i, layer in enumerate(lab.layers):
        for name in ['q_proj', 'v_proj']:
            parent = layer.self_attn
            module = LoRALinear(getattr(parent, name))
            setattr(parent, name, module)
            wrapped.append((i, name, parent, module))
    optimizer = torch.optim.AdamW([p for p in lab.model.parameters() if p.requires_grad], lr=lr, weight_decay=0)
    rng = random.Random(902)
    log = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        batch = rng.sample(rows, 4)
        total = 0.
        for row in batch:
            loss = training_loss(lab, row, include_eos)/len(batch)
            loss.backward()
            total += loss.item()
        norm = float(torch.nn.utils.clip_grad_norm_([p for p in lab.model.parameters() if p.requires_grad], 1.0))
        optimizer.step()
        log.append(dict(step=step+1, batch_nll=total, gradient_norm=norm))
        print(json.dumps(log[-1]), flush=True)
    del optimizer
    changes = []
    adapters = {}
    with torch.no_grad():
        for i, name, parent, module in wrapped:
            delta = module.delta()
            adapters[f'{i}.{name}.A'] = module.A.detach().cpu()
            adapters[f'{i}.{name}.B'] = module.B.detach().cpu()
            original = module.base.weight.detach().float()
            merged = (original+delta).to(module.base.weight.dtype)
            changes.append(dict(layer=i, module=name, delta_norm=delta.norm().item(),
                relative_delta_norm=(delta.norm()/original.norm()).item(),
                merged_changed_elements=int((merged!=module.base.weight).sum())))
            module.base.weight.copy_(merged)
            setattr(parent, name, module.base)
    return dict(method='lora_sft', paper='https://arxiv.org/abs/2106.09685',
        objective='supervised answer-token cross entropy', eos_supervision=include_eos,
        targets=['q_proj','v_proj'], rank=8, alpha=16,
        steps=steps, batch_size=4, learning_rate=lr, optimizer='AdamW, weight_decay=0, clip_norm=1',
        adaptation='short arithmetic pilot; fp32 adapters with bf16 base; not original paper benchmark reproduction',
        changes=changes, training=log), adapters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare','baseline','wanda','taylor','lora','task-vector'])
    parser.add_argument('--model')
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--name', default='model')
    parser.add_argument('--fraction', type=float, default=0.1)
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--coefficient', type=float, default=0.5)
    parser.add_argument('--eos-target', action='store_true')
    parser.add_argument('--seed', type=int, default=902)
    parser.add_argument('--exclude-run', action='append', default=[])
    parser.add_argument('--generation-budget', type=int, default=16)
    args = parser.parse_args()
    if args.command == 'prepare':
        excluded = set()
        for previous in args.exclude_run:
            old = json.loads((Path(previous)/'data.json').read_text())
            for split in ['calibration','search','test']:
                for row in old[split]:
                    a,_,b = re.search(r'(\d+) ([+*\-]) (\d+)',row['prompt']).groups()
                    excluded.add(tuple(sorted((int(a),int(b)))))
        prepare(args.run, seed=args.seed, exclude=excluded)
        write_json(args.run/'paper_protocol.json', dict(seed=args.seed, excluded_pairs=len(excluded),
            calibration=48, search=64, test=128, methods=['wanda_0.1','taylor_mlp_1/12','lora_sft_40_steps'],
            eos_supervision=args.eos_target, max_new_tokens=args.generation_budget,
            supersedes_generic_protocol='This file specifies paper methods; protocol.json contains legacy generic defaults.',
            selection='all predeclared candidates evaluated; do not retune using heldout results',
            limitations='synthetic arithmetic pilot, tiny prose guard, no general ability claim'))
        return
    if (args.run/(args.name+'.json')).exists() or (args.run/args.name).exists():
        raise FileExistsError('Preserve existing evidence; choose a new name/run')
    start = time.perf_counter()
    lab = Lab(args.model)
    data = json.loads((args.run/'data.json').read_text())
    if args.command == 'baseline':
        write_json(args.run/'source_manifest.json', lab.manifest())
        result = dict(search=lab.score(data['search']), guard=lab.guard(data['guard_search']))
    else:
        if args.command == 'wanda':
            method = run_wanda(lab, data['calibration'], args.fraction)
        elif args.command == 'taylor':
            method = run_taylor(lab, data['calibration'], args.fraction, args.eos_target)
        elif args.command == 'lora':
            method, adapters = run_lora(lab, data['calibration'], args.steps, args.lr, args.eos_target)
            torch.save(adapters, args.run/(args.name+'_adapter.pt'))
        else:
            # Ilharco et al., ICLR 2023: theta_new=theta_base+lambda*(theta_tuned-theta_base).
            # Use the actual saved BF16 tuned checkpoint, not unrounded BA factors.
            manifest = json.loads((args.run/'source_manifest.json').read_text())
            if str(lab.source) != manifest['source']:
                raise ValueError('Task vector must use its declared base model')
            tuned = files_by_key(args.run/'lora40')
            changes = []
            with torch.no_grad():
                for key, param in lab.model.named_parameters():
                    if '.self_attn.q_proj.weight' in key or '.self_attn.v_proj.weight' in key:
                        target = tensor(tuned,key).to(param.device).float()
                        original = param.detach().float()
                        delta = target-original
                        param.copy_((original+args.coefficient*delta).to(param.dtype))
                        changes.append(dict(key=key, task_vector_norm=delta.norm().item()))
            method = dict(method='task_vector_interpolation', paper='https://arxiv.org/abs/2212.04089',
                coefficient=args.coefficient, tuned_checkpoint=str(args.run/'lora40'), changes=changes,
                adaptation='single LoRA-SFT task vector on Qwen3 arithmetic; not original benchmark reproduction')
        method['base_model'] = str(lab.source)
        method['eos_supervision'] = args.eos_target
        result = dict(method=method, search=lab.score(data['search']), guard=lab.guard(data['guard_search']))
        lab.export(args.run/args.name, method)
    result['wall_seconds'] = time.perf_counter()-start
    result['code_sha256'] = SOURCE_SHA256
    write_json(args.run/(args.name+'.json'), result)
    print(json.dumps(dict(command=args.command, name=args.name, search_nll=result['search']['answer_nll'],
        wall_seconds=result['wall_seconds'])), flush=True)


if __name__ == '__main__':
    main()
