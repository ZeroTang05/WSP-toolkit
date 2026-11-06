"""Acquire pinned final-evaluation inputs, never programmer practice data."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

REPO='HuggingFaceH4/MATH-500'
REVISION='6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be'


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    allowed={'test.jsonl','README.md','eval.yaml','manifest.json'}
    if a.output.exists() and {p.name for p in a.output.iterdir()}-allowed:raise ValueError('Separate evaluation directory required')
    a.output.mkdir(parents=True,exist_ok=True);files={}
    for name in ['test.jsonl','README.md','eval.yaml']:
        url=f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{name}'
        data=urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'idea4-evaluation-acquisition'}),timeout=120).read()
        files[name]=dict(url=url,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
        (a.output/name).write_bytes(data)
    rows=[json.loads(x) for x in (a.output/'test.jsonl').read_text().split('\n') if x.strip()]
    if len(rows)!=500 or len({r['unique_id'] for r in rows})!=500:raise ValueError('Expected 500 unique official items')
    for row in rows:
        if any(not isinstance(row[k],str) or not row[k].strip() for k in ['unique_id','problem','answer','solution','subject']):raise ValueError('Invalid official fields')
        if not row['unique_id'].startswith('test/') or type(row['level'])!=int or not 1<=row['level']<=5:raise ValueError('Expected test provenance and difficulty')
    manifest=dict(repository=REPO,revision=REVISION,files=files,items=500,role='final_evaluation',
        identity='Official unique_id retained; no problem or answer rewriting',
        terms='Preserve original dataset card and upstream dataset terms; toolkit licensing does not replace them.',
        isolation='Host evaluator only. Never mount or register as programmer practice.')
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(status='pinned_math500_ready',revision=REVISION,items=500)),flush=True)


if __name__=='__main__':main()
