"""Acquire only the author's exact experiment-8 practice inputs, not edit logs.

Pinned data are valid independent-discovery inputs under the original protocol.
No reference coordinates, edited tensors, run trajectories or evaluation sets are
downloaded here. Preserve upstream attribution and third-party dataset terms.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

REPO='RikiyaT/lm_understand-logs'
REVISION='4d09ba6d6a7e891dd8063974dfe81b3d78e3494f'
EXPECTED={'gsm8k_train.jsonl':7469,'math_train.jsonl':7275,'aime.jsonl':714}


def fetch(url):
    return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'idea4-practice-acquisition'}),timeout=120)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists() and set(p.name for p in a.output.iterdir())-set(EXPECTED)-{'stats.json','LICENSE','manifest.json'}:
        raise ValueError('Unexpected files in practice acquisition; do not mix with reference artifacts')
    with fetch(f'https://api.github.com/repos/{REPO}/contents/exp8_math/practice?ref={REVISION}') as response:
        listing=json.load(response)
    by_name={row['name']:row for row in listing}
    if set(EXPECTED)-set(by_name):raise ValueError('Missing author practice files')
    a.output.mkdir(parents=True,exist_ok=True);files={}
    for name in list(EXPECTED)+['stats.json']:
        path='exp8_math/practice/'+name
        cached=a.output/name
        if cached.exists():data=cached.read_bytes()
        else:
            with fetch(f'https://raw.githubusercontent.com/{REPO}/{REVISION}/{path}') as response:
                data=response.read()
        blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if blob!=by_name[name]['sha']:raise ValueError('Git blob mismatch for '+name)
        (a.output/name).write_bytes(data);rows=[]
        if name in EXPECTED:
            # JSONL is delimited by literal LF. Unicode line-separator characters
            # inside a valid JSON string must not be treated as record boundaries.
            rows=[json.loads(line) for line in data.decode().split('\n') if line.strip()]
            if len(rows)!=EXPECTED[name] or len({r['id'] for r in rows})!=len(rows):raise ValueError('Unexpected practice count/IDs')
            if any(not {'id','source','problem','answer'}<=set(r) for r in rows):raise ValueError('Missing practice fields')
            if any(not isinstance(r['problem'],str) or not isinstance(r['answer'],str) for r in rows):raise ValueError('Invalid practice content types')
        files[name]=dict(path=path,git_blob_sha=blob,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),rows=len(rows) if rows else None)
        print(json.dumps(dict(event='practice_file_ready',file=name,rows=len(rows))),flush=True)
    with fetch(f'https://raw.githubusercontent.com/{REPO}/{REVISION}/LICENSE') as response:license_data=response.read()
    (a.output/'LICENSE').write_bytes(license_data)
    manifest=dict(repository=REPO,revision=REVISION,files=files,license_sha256=hashlib.sha256(license_data).hexdigest(),
        purpose='Author-provided practice inputs only. No reference edit, run analysis or evaluation data. Dataset licenses are not replaced by repository licensing.',
        validation='Counts, unique IDs, required types and Git blobs; no independent reimplementation of the author overlap filter claimed.')
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
