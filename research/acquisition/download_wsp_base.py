"""Prefetch the exact public base revision used by the MIT weight-edit artifact.

Download only; no model code, inference, gradient operation or GPU allocation.
Reference edit artifacts are deliberately not placed in this model directory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['HF_HUB_OFFLINE']='0'
from huggingface_hub import snapshot_download

REPO='Qwen/Qwen3.5-0.8B'
REVISION='2fc06364715b967f1860aea9cf38778875588b17'


def sha(path):
    value=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    target=a.output/'idea4_wsp_base_manifest.json'
    if target.exists():
        info=json.loads(target.read_text())
        if info['repo']!=REPO or info['revision']!=REVISION:raise ValueError('Different cached base revision')
        if any(sha(a.output/name)!=record['sha256'] for name,record in info['files'].items()):raise ValueError('Cached model changed')
        print('Pinned base already verified',flush=True);return
    snapshot_download(repo_id=REPO,revision=REVISION,endpoint='https://hf-mirror.com',local_dir=str(a.output),max_workers=1,
        allow_patterns=['*.json','*.safetensors','*.jinja','tokenizer.model','merges.txt','*.md','LICENSE*'])
    files={str(path.relative_to(a.output)):dict(bytes=path.stat().st_size,sha256=sha(path))
        for path in a.output.rglob('*') if path.is_file() and '.cache' not in path.parts}
    target.write_text(json.dumps(dict(repo=REPO,revision=REVISION,transport='https://hf-mirror.com',files=files,
        purpose='Original public base only; no published edit or author-selected direction supplied to independent discovery.'),indent=2))
    print(json.dumps(dict(event='pinned_base_ready',files=len(files),bytes=sum(r['bytes'] for r in files.values()))),flush=True)


if __name__=='__main__':main()
