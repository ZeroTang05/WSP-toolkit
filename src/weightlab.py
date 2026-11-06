"""Small, auditable Qwen3 weight-space experiment CLI. Never writes source models."""
import argparse
import contextlib
import hashlib
import json
import math
import random
import re
import time
import os
import fcntl
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def cast_floating_parameters(model,dtype):
    """Change parameter arithmetic precision while preserving execution buffers.

    Module.bfloat16() also rounds nonpersistent RoPE frequencies. Parameter-only
    checks must retain those buffers, including their unregistered aliases.
    """
    for parameter in model.parameters():
        if parameter.is_floating_point():
            parameter.data=parameter.data.to(dtype=dtype)
            if parameter.grad is not None:parameter.grad.data=parameter.grad.data.to(dtype=dtype)
    return model


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def prepare(out, seed=401, exclude=()):
    """Separate operand pairs across splits; heldout also changes wording."""
    out = Path(out)
    if (out / 'data.json').exists():
        raise FileExistsError('Dataset already exists; use another run directory.')
    excluded = set(exclude)
    pairs = [(a, b) for a in range(2, 100) for b in range(a, 100) if (a,b) not in excluded]
    random.Random(seed).shuffle(pairs)
    data = {}
    offset = 0
    for split, n in [('calibration', 48), ('search', 64), ('test', 128)]:
        rows = []
        for i, (a, b) in enumerate(pairs[offset:offset+n]):
            op = ['+', '-', '*'][i % 3]
            if op == '-':
                a, b = max(a, b), min(a, b)
            answer = a+b if op == '+' else a-b if op == '-' else a*b
            question = f'Calculate {a} {op} {b}. Return only the integer answer.'
            if split == 'test':
                question = f'What is the result of {a} {op} {b}? Answer with an integer only.'
            rows.append(dict(id=f'{split}-{i}', prompt=question, answer=str(answer), op=op))
        data[split] = rows
        offset += n
    # Remove commutative/reversed pair leakage, replacing with unused pairs if needed.
    seen = set()
    for split, rows in data.items():
        for row in rows:
            a, op, b = re.search(r'(\d+) ([+*\-]) (\d+)', row['prompt']).groups()
            key = (op, *sorted((int(a), int(b))))
            if key in seen:
                raise AssertionError('Duplicate arithmetic relation across splits')
            seen.add(key)
    texts = [
        'Plants use sunlight to convert water and carbon dioxide into sugars.',
        'A library contains books that readers can borrow and return.',
        'The train arrived at the station early in the morning.',
        'When water freezes, it changes from a liquid into a solid.',
        'A careful experiment compares observations under controlled conditions.',
        'The children put on their coats before walking outside.',
        'An algorithm is a sequence of steps for solving a problem.',
        'The capital of France is Paris, a city on the Seine River.',
        'A computer stores information and executes instructions.',
        'After the rain stopped, the sky became clear again.',
        'Scientists test explanations by collecting evidence.',
        'The teacher explained the lesson and answered questions.',
        'The Earth travels around the Sun once every year.',
        'She opened the window to let fresh air into the room.',
        'A dictionary provides meanings and spellings of words.',
        'The river flows through the valley toward the sea.',
    ]
    data['guard_search'] = texts[:8]
    data['guard_test'] = texts[8:]
    write_json(out / 'data.json', data)
    write_json(out / 'protocol.json', dict(seed=seed, domain='synthetic integer arithmetic',
        search_metric='answer token NLL', guard='hand-authored English token NLL; not a benchmark',
        test_policy='Do not select patches using final test outcomes',
        selection='lowest search NLL among identity and predeclared candidates with guard NLL <= baseline * 1.03',
        candidate_scales=[0.8, 0.9, 1.1, 1.2], compression_ratios=[0.0833333333, 0.1666666667]))


class Lab:
    def __init__(self, model_path):
        # Serialize this project's GPU workloads across independent experiment jobs.
        # The descriptor is retained for the model lifetime and released on exit.
        lock_name = f'/tmp/idea4-{os.getuid()}-gpu-{os.environ.get("CUDA_VISIBLE_DEVICES", "default")}.lock'
        self._gpu_lock = open(lock_name, 'a')
        lock_started=time.perf_counter()
        fcntl.flock(self._gpu_lock, fcntl.LOCK_EX)
        self.gpu_lock_wait_seconds=time.perf_counter()-lock_started
        torch.manual_seed(401)
        torch.set_num_threads(4)
        self.source = Path(model_path).resolve()
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.source), local_files_only=True)
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(str(self.source), local_files_only=True,
            torch_dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
        self.layers = self.model.model.layers

    def prefix(self, row):
        return self.tokenizer.apply_chat_template([{'role': 'user', 'content': row['prompt']}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)

    @torch.inference_mode()
    def score(self, rows):
        losses, correct, counts = [], [], []
        for row in rows:
            prefix_ids = self.tokenizer.encode(self.prefix(row), add_special_tokens=False)
            answer_ids = self.tokenizer.encode(row['answer'], add_special_tokens=False)
            ids = torch.tensor([prefix_ids + answer_ids], device='cuda')
            # Score exactly answer tokens; never include prompt tokens or padding.
            logits = self.model(ids, use_cache=False).logits[0, len(prefix_ids)-1:-1].float()
            labels = torch.tensor(answer_ids, device='cuda')
            loss = F.cross_entropy(logits, labels, reduction='sum').item()
            losses.append(loss)
            correct.append(int((logits.argmax(-1) == labels).sum()))
            counts.append(len(answer_ids))
        return dict(n=len(rows), answer_nll=sum(losses)/sum(counts),
            teacher_forced_token_accuracy=sum(correct)/sum(counts),
            example_nll=[v/n for v, n in zip(losses, counts)])

    @torch.inference_mode()
    def guard(self, texts):
        total, n = 0., 0
        for text in texts:
            ids = self.tokenizer(text, return_tensors='pt').input_ids.to('cuda')
            logits = self.model(ids, use_cache=False).logits[:, :-1].float()
            total += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction='sum').item()
            n += ids.shape[-1]-1
        return dict(token_nll=total/n, n_tokens=n)

    @torch.inference_mode()
    def generate(self, rows, max_new_tokens=16):
        results = []
        for start in range(0, len(rows), 8):
            batch = rows[start:start+8]
            inputs = self.tokenizer([self.prefix(r) for r in batch], padding=True, return_tensors='pt').to('cuda')
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id)
            texts = self.tokenizer.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            for row, text in zip(batch, texts):
                results.append(dict(**row, output=text, exact_match=text.strip() == row['answer']))
        return dict(exact_match=sum(r['exact_match'] for r in results)/len(results), n=len(results), rows=results)

    @torch.inference_mode()
    def profile(self):
        ids = self.tokenizer('The purpose of this fixed workload is to measure model inference. ' * 32,
                             return_tensors='pt').input_ids[:, :256].to('cuda')
        for _ in range(3):
            self.model(ids, use_cache=False)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(12):
            torch.cuda.synchronize()
            t = time.perf_counter()
            self.model(ids, use_cache=False)
            torch.cuda.synchronize()
            times.append((time.perf_counter()-t)*1000)
        return dict(workload='batch=1, 256 input tokens, full forward including logits, no KV cache',
            median_ms=sorted(times)[len(times)//2], times_ms=times,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            parameters=sum(p.numel() for p in self.model.parameters()))

    def manifest(self):
        return dict(source=str(self.source), torch=torch.__version__, transformers=transformers.__version__,
            gpu=torch.cuda.get_device_name(), dtype=str(next(self.model.parameters()).dtype),
            files={p.name: digest(p) for p in self.source.glob('*')
                   if p.suffix in ('.safetensors', '.json')})

    @contextlib.contextmanager
    def scale(self, layer, factor):
        weight = self.layers[layer].mlp.down_proj.weight
        original = weight.detach().clone()
        try:
            with torch.no_grad():
                weight.copy_((original.float()*factor).to(original.dtype))
            yield
        finally:
            with torch.no_grad():
                weight.copy_(original)
            if not torch.equal(weight, original):
                raise AssertionError('Rollback failed')

    @contextlib.contextmanager
    def ablate(self, layer):
        handle = self.layers[layer].mlp.register_forward_hook(lambda m, i, o: torch.zeros_like(o))
        try:
            yield
        finally:
            handle.remove()

    @torch.inference_mode()
    def observe(self, rows):
        sums = [torch.zeros(layer.mlp.down_proj.in_features, device='cuda') for layer in self.layers]
        counts = [0]*len(self.layers)
        handles = []
        for index, layer in enumerate(self.layers):
            def hook(module, inputs, index=index):
                value = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1])
                sums[index].add_(value.abs().sum(0))
                counts[index] += value.shape[0]
            handles.append(layer.mlp.down_proj.register_forward_pre_hook(hook))
        try:
            self.score(rows)
        finally:
            for handle in handles:
                handle.remove()
        scores = []
        stats = []
        for index, layer in enumerate(self.layers):
            norms = layer.mlp.down_proj.weight.detach().float().norm(dim=0)
            means = sums[index] / counts[index]
            importance = norms * means
            scores.append(importance.cpu())
            stats.append(dict(layer=index, activation_abs_mean=means.mean().item(),
                output_column_norm_mean=norms.mean().item(), importance_mean=importance.mean().item()))
        return scores, stats

    @torch.inference_mode()
    def masks(self, scores, fraction):
        keep_n = int(self.model.config.intermediate_size*(1-fraction))
        keep_n = (keep_n // 128) * 128
        if not 0 < keep_n < self.model.config.intermediate_size:
            raise ValueError('Invalid compression fraction')
        return [s.topk(keep_n).indices.sort().values for s in scores]

    @contextlib.contextmanager
    def masked(self, keep):
        handles = []
        for layer, indices in zip(self.layers, keep):
            mask = torch.zeros(layer.mlp.down_proj.in_features, device='cuda', dtype=torch.bfloat16)
            mask[indices.to('cuda')] = 1
            handles.append(layer.mlp.down_proj.register_forward_pre_hook(
                lambda m, x, mask=mask: (x[0]*mask,)))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    @torch.no_grad()
    def structural_prune(self, keep):
        if len(keep)!=len(self.layers) or not keep:
            raise ValueError('One channel selector per decoder layer is required')
        widths=[]
        # Validate every dependency group before modifying any layer.
        for layer,indices in zip(self.layers,keep):
            mlp=layer.mlp
            if any(getattr(mlp,name).bias is not None for name in ['gate_proj','up_proj','down_proj']):
                raise ValueError('This structural adapter supports bias-free gated MLPs only')
            if (mlp.gate_proj.out_features!=mlp.up_proj.out_features or
                mlp.gate_proj.out_features!=mlp.down_proj.in_features or
                mlp.gate_proj.in_features!=mlp.up_proj.in_features or
                mlp.down_proj.out_features!=mlp.gate_proj.in_features):
                raise ValueError('Inconsistent gated-MLP dependencies')
            if indices.ndim!=1 or indices.dtype!=torch.int64:
                raise ValueError('Selector must be a one-dimensional int64 tensor')
            values=indices.tolist()
            if (not 0<len(values)<mlp.down_proj.in_features or values!=sorted(set(values))
                or values[0]<0 or values[-1]>=mlp.down_proj.in_features):
                raise ValueError('Selector must preserve unique valid channels in order')
            widths.append(len(values))
        if len(set(widths))!=1:raise ValueError('This export adapter requires a uniform intermediate width')
        for layer, indices in zip(self.layers, keep):
            mlp = layer.mlp
            for name, dim in [('gate_proj', 0), ('up_proj', 0), ('down_proj', 1)]:
                old = getattr(mlp, name)
                weight = old.weight.index_select(dim, indices.to(old.weight.device))
                new = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False,
                    device=weight.device, dtype=weight.dtype)
                new.weight.copy_(weight)
                setattr(mlp, name, new)
            mlp.intermediate_size = len(indices)
        self.model.config.intermediate_size = len(keep[0])

    def export(self, output, patch):
        output = Path(output).resolve()
        if output == self.source or self.source in output.parents:
            raise ValueError('Refusing to write inside source model directory')
        if output.exists():
            raise FileExistsError(output)
        output.mkdir(parents=True)
        self.model.save_pretrained(output, safe_serialization=True)
        self.tokenizer.save_pretrained(output)
        write_json(output/'weightlab_patch.json', patch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'baseline', 'probe', 'sensitivity', 'search', 'compress', 'evaluate', 'replay', 'verify-source'])
    parser.add_argument('--model')
    parser.add_argument('--run', required=True)
    parser.add_argument('--layers', default='')
    parser.add_argument('--fraction', type=float, default=1/12)
    parser.add_argument('--name', default='candidate')
    parser.add_argument('--direction', choices=['apply', 'rollback'], default='apply')
    parser.add_argument('--max-new-tokens', type=int, default=16)
    parser.add_argument('--allow-legacy-heuristic', action='store_true',
        help='Explicitly opt into historical non-paper scaling/channel-score experiments')
    args = parser.parse_args()
    run = Path(args.run)
    if args.command in ('search','sensitivity','compress') and not args.allow_legacy_heuristic:
        parser.error('Historical heuristic disabled. Use paper-weightlab for paper-derived methods; '
                     'archival reproduction requires --allow-legacy-heuristic.')
    if args.command == 'prepare':
        prepare(run)
        return
    if args.command == 'verify-source':
        manifest = json.loads((run/'source_manifest.json').read_text())
        result = {name: digest(Path(manifest['source'])/name) == value
                  for name, value in manifest['files'].items()}
        write_json(run/'source_integrity.json', result)
        if not all(result.values()):
            raise AssertionError('Source file changed')
        print(json.dumps(result), flush=True)
        return
    lab = Lab(args.model)
    data = json.loads((run/'data.json').read_text())
    if args.command == 'baseline':
        write_json(run/'source_manifest.json', lab.manifest())
        result = dict(search=lab.score(data['search']), guard=lab.guard(data['guard_search']), profile=lab.profile())
        write_json(run/'baseline.json', result)
        print(json.dumps(result), flush=True)
    elif args.command == 'probe':
        scores, stats = lab.observe(data['calibration'])
        torch.save(scores, run/'channel_scores.pt')
        base = lab.score(data['search'][:24])
        probes = []
        for layer in range(len(lab.layers)):
            with lab.ablate(layer):
                score = lab.score(data['search'][:24])
            probes.append(dict(layer=layer, nll_delta=score['answer_nll']-base['answer_nll']))
            print(json.dumps(probes[-1]), flush=True)
        write_json(run/'probe.json', dict(statistics=stats, ablations=probes, baseline=base,
            caveat='MLP zero ablation is an intervention, not a complete circuit explanation'))
    elif args.command == 'sensitivity':
        # Predictions came from first 24 search rows; intervention outcomes use the other 40.
        base = lab.score(data['search'][24:])
        results = []
        for layer in range(len(lab.layers)):
            with lab.scale(layer, 0.8):
                score = lab.score(data['search'][24:])
            results.append(dict(layer=layer, factor=0.8, nll_delta=score['answer_nll']-base['answer_nll']))
            write_json(run/'sensitivity.json', dict(baseline=base, outcomes=results))
            print(json.dumps(results[-1]), flush=True)
    elif args.command == 'search':
        layers = [int(v) for v in args.layers.split(',') if v]
        if not layers:
            raise ValueError('Specify layers selected from probe evidence')
        base = json.loads((run/'baseline.json').read_text())
        candidates = []
        for layer in layers:
            for factor in [0.8, 0.9, 1.1, 1.2]:
                with lab.scale(layer, factor):
                    result = dict(layer=layer, factor=factor, score=lab.score(data['search']),
                                  guard=lab.guard(data['guard_search']))
                result['eligible'] = result['guard']['token_nll'] <= base['guard']['token_nll']*1.03
                candidates.append(result)
                write_json(run/'search.json', candidates)
                print(json.dumps(result), flush=True)
        valid = [c for c in candidates if c['eligible'] and c['score']['answer_nll'] < base['search']['answer_nll']]
        if valid:
            best = min(valid, key=lambda c: c['score']['answer_nll'])
            layer, factor = best['layer'], best['factor']
            original = lab.layers[layer].mlp.down_proj.weight.detach().clone()
            with torch.no_grad():
                lab.layers[layer].mlp.down_proj.weight.copy_((original.float()*factor).to(original.dtype))
            run.joinpath('patches').mkdir(exist_ok=True)
            torch.save(dict(layer=layer, original=original.cpu(),
                replacement=lab.layers[layer].mlp.down_proj.weight.detach().cpu()), run/'patches/scale.pt')
            lab.export(run/'edited_model', dict(kind='scale_mlp_down_projection', **best,
                source=str(lab.source), selection_data='search only'))
            write_json(run/'selected_edit.json', best)
        else:
            write_json(run/'selected_edit.json', dict(kind='identity', reason='No eligible search improvement'))
    elif args.command == 'compress':
        scores = torch.load(run/'channel_scores.pt', map_location='cpu', weights_only=True)
        keep = lab.masks(scores, args.fraction)
        probe_ids = lab.tokenizer('The answer to a simple question is', return_tensors='pt').input_ids.to('cuda')
        # A changed GEMM shape changes BF16 accumulation error. Verify the algebra
        # in FP32, then measure the actual compressed BF16 model independently.
        lab.model.float()
        with torch.inference_mode(), lab.masked(keep):
            reference = lab.model(probe_ids, use_cache=False).logits[:, -1].float().cpu()
        lab.structural_prune(keep)
        with torch.inference_mode():
            actual = lab.model(probe_ids, use_cache=False).logits[:, -1].float().cpu()
        check = dict(dtype='float32', max_logit_difference=(reference-actual).abs().max().item(),
            mean_logit_difference=(reference-actual).abs().mean().item(),
            same_argmax=bool(reference.argmax()==actual.argmax()))
        if check['max_logit_difference'] > 0.001:
            raise AssertionError(f'Structural/mask equivalence failed: {check}')
        lab.model.bfloat16()
        score = lab.score(data['search'])
        guard = lab.guard(data['guard_search'])
        result = dict(requested_fraction=args.fraction, keep_width=len(keep[0]),
            score=score, guard=guard, equivalence=check, profile=lab.profile())
        lab.export(run/args.name, dict(kind='structured_mlp_channel_pruning', **result,
            source=str(lab.source), selection_data='calibration only'))
        torch.save(keep, run/(args.name+'_keep.pt'))
        write_json(run/(args.name+'.json'), result)
        print(json.dumps(result), flush=True)
    elif args.command == 'replay':
        patch = torch.load(run/'patches/scale.pt', map_location='cpu', weights_only=True)
        weight = lab.layers[patch['layer']].mlp.down_proj.weight
        before, after = ('original', 'replacement') if args.direction == 'apply' else ('replacement', 'original')
        if not torch.equal(weight.detach().cpu(), patch[before]):
            raise ValueError('Patch preimage mismatch: refusing to modify unrelated weights')
        with torch.no_grad():
            weight.copy_(patch[after].to(weight.device))
        lab.export(run/args.name, dict(kind='replay', direction=args.direction, layer=patch['layer']))
    elif args.command == 'evaluate':
        # Run once per frozen candidate; never automatically feed test back into search.
        path = run/(args.name+'_final.json')
        if path.exists():
            raise FileExistsError('Final evaluation already exists; preserve its evidence')
        result = dict(model=str(lab.source), search=lab.score(data['search']),
            test=lab.score(data['test']), guard_test=lab.guard(data['guard_test']),
            generation=lab.generate(data['test'], args.max_new_tokens), profile=lab.profile(),
            max_new_tokens=args.max_new_tokens)
        write_json(path, result)
        print(json.dumps({k:v for k,v in result.items() if k!='generation'}), flush=True)


if __name__ == '__main__':
    main()
