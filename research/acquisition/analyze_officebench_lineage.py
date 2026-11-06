"""Exact input-file identity graph, a minimum constraint for future task splits."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--acquisition',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    manifest=json.loads((a.acquisition/'manifest.json').read_text());root=a.acquisition/'source/tasks'
    groups=sorted(p.name for p in root.iterdir() if p.is_dir());parent={g:g for g in groups};seen={};assets={};duplicates=0
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    for group in groups:
        for file in sorted((root/group/'testbed').rglob('*')):
            if not file.is_file() or not file.stat().st_size:continue
            digest=hashlib.sha256(file.read_bytes()).hexdigest();assets[str(file.relative_to(root))]=digest
            if digest in seen:
                first,second=find(group),find(seen[digest]);parent[second]=first;duplicates+=1
            else:seen[digest]=group
    parts=defaultdict(list)
    for group in groups:parts[find(group)].append(group)
    components=[dict(groups=groups,tasks=sum(len(list((root/g/'subtasks').glob('*.json'))) for g in groups)) for groups in parts.values()]
    components.sort(key=lambda c:(-c['tasks'],c['groups']))
    result=dict(source_commit=manifest['commit'],rule='Connect workspaces sharing any byte-identical nonempty testbed file; no answer or outcome used',
        workspaces=len(groups),components=components,duplicate_file_occurrences=duplicates,asset_sha256=assets,
        caveat='Exact asset identity is only a minimum grouping constraint; template and near-duplicate lineage can connect more groups.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(workspaces=len(groups),components=len(components),largest_component_tasks=[c['tasks'] for c in components[:5]])))


if __name__=='__main__':main()
