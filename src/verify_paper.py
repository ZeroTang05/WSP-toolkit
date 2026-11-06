"""Check exported paper-method tensor edits against immutable source tensors."""
import argparse
import json
from pathlib import Path
import torch
from verify_artifacts import files_by_key, tensor
from weightlab import digest, write_json


def verify(run):
    torch.set_num_threads(4)
    manifest = json.loads((run/'source_manifest.json').read_text())
    source = files_by_key(manifest['source'])
    results = {}
    for name in ['wanda10','taylor08','lora40','taskhalf']:
        directory = run/name
        if not directory.exists():
            continue
        method = json.loads((directory/'weightlab_patch.json').read_text())
        source = files_by_key(method.get('base_model',manifest['source']))
        target = files_by_key(directory)
        assert (set(source)^set(target)) <= {'lm_head.weight'}
        if 'lm_head.weight' not in target and 'lm_head.weight' in source:
            assert json.loads((directory/'config.json').read_text())['tie_word_embeddings']
            assert torch.equal(tensor(source,'lm_head.weight'),tensor(source,'model.embed_tokens.weight'))
        adapters = torch.load(run/(name+'_adapter.pt'), map_location='cpu', weights_only=True) if name=='lora40' else {}
        changed = []
        for key in source:
            old, new = tensor(source,key), tensor(target,key)
            if torch.equal(old,new):
                continue
            changed.append(key)
            parts = key.split('.')
            assert parts[:2] == ['model','layers'], key
            index = int(parts[2])
            if name=='wanda10':
                assert key.endswith('.weight') and parts[3] in ('self_attn','mlp'), key
                assert old.shape == new.shape
                assert torch.all((new==old)|(new==0)), key
                assert torch.all((new==0).sum(1)>=int(old.shape[1]*.1)), key
            elif name=='taylor08':
                assert parts[3]=='mlp', key
                kept = torch.tensor(method['kept_indices'][index])
                dim = 1 if parts[4]=='down_proj' else 0
                assert torch.equal(new,old.index_select(dim,kept)), key
            elif name=='lora40':
                assert parts[3]=='self_attn' and parts[4] in ('q_proj','v_proj'), key
                A,B = (adapters[f'{index}.{parts[4]}.{n}'] for n in ['A','B'])
                expected = (old.float()+(method['alpha']/method['rank'])*(B@A)).to(old.dtype)
                # CPU/GPU FP32 GEMM can differ before BF16 rounding. Measure this
                # explicitly rather than assuming bitwise adapter merge equivalence.
                error = (new.float()-expected.float()).abs().max().item()
                assert error <= .002, (key,error)
            else:
                assert parts[3]=='self_attn' and parts[4] in ('q_proj','v_proj'), key
                tuned = tensor(files_by_key(run/'lora40'), key).float()
                expected = (old.float()+method['coefficient']*(tuned-old.float())).to(old.dtype)
                assert torch.equal(new,expected), key
        assert changed, f'{name}: no parameter modification'
        results[name] = dict(verified=True, changed_tensors=len(changed), keys=changed)
    integrity = {name:digest(Path(manifest['source'])/name)==value for name,value in manifest['files'].items()}
    assert all(integrity.values()), 'Source changed'
    assert results, 'No artifacts verified'
    write_json(run/'paper_verification.json', dict(artifacts=results, source_integrity=integrity))
    print(json.dumps({name:value['changed_tensors'] for name,value in results.items()}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    verify(parser.parse_args().run)
