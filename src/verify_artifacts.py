"""CPU verification independent of the model's inference implementation."""
import argparse
import json
from pathlib import Path
import torch
from safetensors import safe_open


def files_by_key(directory):
    result = {}
    for path in Path(directory).glob('*.safetensors'):
        with safe_open(path, framework='pt', device='cpu') as stream:
            for key in stream.keys():
                if key in result:
                    raise AssertionError(f'Duplicate tensor: {key}')
                result[key] = path
    return result


def tensor(files, key):
    if key == 'lm_head.weight' and key not in files:
        key = 'model.embed_tokens.weight'
    with safe_open(files[key], framework='pt', device='cpu') as stream:
        return stream.get_tensor(key)


def verify(run):
    torch.set_num_threads(4)
    manifest = json.loads((run/'source_manifest.json').read_text())
    selected = json.loads((run/'selected_edit.json').read_text())
    source = files_by_key(manifest['source'])
    edited = files_by_key(run/'edited_model')
    restored = files_by_key(run/'restored_model')
    assert set(edited) == set(restored)
    # save_pretrained omits duplicate tied lm_head storage. Verify the tying
    # contract and original tensor equality before resolving that alias.
    missing = set(source)-set(edited)
    assert not (set(edited)-set(source))
    assert missing <= {'lm_head.weight'}
    if missing:
        for directory in [Path(manifest['source']), run/'edited_model', run/'restored_model']:
            assert json.loads((directory/'config.json').read_text())['tie_word_embeddings']
        assert torch.equal(tensor(source, 'lm_head.weight'), tensor(source, 'model.embed_tokens.weight'))
    key = f"model.layers.{selected['layer']}.mlp.down_proj.weight"
    changed, restored_equal = [], True
    for name in source:
        original = tensor(source, name)
        current = tensor(edited, name)
        if not torch.equal(original, current):
            changed.append(name)
        assert torch.equal(original, tensor(restored, name)), f'Failed rollback: {name}'
        if name == key:
            assert torch.equal(current, (original.float()*selected['factor']).to(original.dtype))
    assert changed == [key], changed
    patch = torch.load(run/'patches/scale.pt', map_location='cpu', weights_only=True)
    assert torch.equal(patch['original'], tensor(source, key))
    assert torch.equal(patch['replacement'], tensor(edited, key))
    result = dict(changed_tensors=changed, total_tensors=len(source),
        restored_model_all_tensors_exact=True, patch_preimage_verified=True,
        changed_parameter_count=patch['original'].numel())
    (run/'artifact_verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    verify(parser.parse_args().run)
