"""Acquire pinned OfficeBench and separate performer inputs from scoring data.

Does not execute upstream Python, build Docker images, contact email recipients
or run model tasks. Official application environment/evaluator remains a separate
integration step. Source task files must not be mounted into an agent workspace.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path,PurePosixPath
import urllib.request
import zipfile

COMMIT='b978b808667c32b52ce19a67ce1def1de9ae02b7'
REPO='zlwang-cs/OfficeBench'


def main():
    p=argparse.ArgumentParser();p.add_argument('--tree',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Fresh acquisition output required')
    tree=json.loads(a.tree.read_text())
    if tree.get('truncated'):raise ValueError('Complete source tree required')
    declared={r['path']:r for r in tree['tree'] if r['type']=='blob'}
    a.output.mkdir(parents=True);archive=a.output/'source.zip'
    url=f'https://codeload.github.com/{REPO}/zip/{COMMIT}'
    with urllib.request.urlopen(url,timeout=90) as response,archive.open('wb') as target:
        while block:=response.read(1024*1024):target.write(block)
    source=a.output/'source';source.mkdir();checked=[]
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            if info.is_dir():continue
            parts=PurePosixPath(info.filename).parts
            relative=PurePosixPath(*parts[1:])
            if '..' in relative.parts or relative.is_absolute() or str(relative) not in declared:
                raise ValueError('Archive path outside declared source tree')
            data=z.read(info);entry=declared[str(relative)]
            blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
            if entry['sha']!=blob:raise ValueError('Git blob mismatch: '+str(relative))
            path=source/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
            checked.append(str(relative))
    if set(checked)!=set(declared):raise ValueError('Missing source files')
    performer=a.output/'performer';evaluator=a.output/'evaluator';performer.mkdir();evaluator.mkdir()
    tasks=[];categories=Counter();functions=Counter();doc_types=Counter();source_groups=Counter()
    for config in sorted(source.glob('tasks/*/subtasks/*.json')):
        raw=json.loads(config.read_text());group=config.parents[1].name;subtask=config.stem;task_id=group+'/'+subtask
        public={k:v for k,v in raw.items() if k!='evaluation'}
        if 'task' not in public or not raw.get('evaluation'):raise ValueError('Missing task/scoring configuration')
        (performer/(group+'_'+subtask+'.json')).write_text(json.dumps(dict(id=task_id,**public),indent=2))
        (evaluator/(group+'_'+subtask+'.json')).write_text(json.dumps(raw['evaluation'],indent=2))
        hidden=[]
        for item in raw['evaluation']:
            functions[item['function']]+=1
            if item['args'].get('doc_type'):doc_types[item['args']['doc_type']]+=1
            if item['args'].get('expected_file'):hidden.append(item['args']['expected_file'])
        assets=list((source/'tasks'/group/'testbed').rglob('*'))
        task=dict(id=task_id,group=group,num_apps=int(group.split('-')[0]),source_path=str(config.relative_to(source)),
            config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),evaluation_checks=len(raw['evaluation']),
            expected_files_to_withhold=hidden,source_asset_files=sum(f.is_file() for f in assets),
            data_role='public_evaluation_unassigned_no_training_permission')
        tasks.append(task);categories[task['num_apps']]+=1;source_groups[group]+=1
    if len(tasks)!=300 or dict(categories)!={1:93,2:95,3:112}:raise ValueError('Unexpected pinned task inventory')
    (a.output/'tasks.json').write_text(json.dumps(tasks,indent=2))
    summary=dict(repo=REPO,commit=COMMIT,tree_sha=tree['sha'],source_blobs_verified=len(checked),
        source_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),tasks=len(tasks),
        source_workspaces=len(source_groups),by_app_count=dict(categories),evaluation_functions=dict(functions),
        doc_type_checks=dict(doc_types),tasks_with_explicit_expected_files=sum(bool(t['expected_files_to_withhold']) for t in tasks),
        status='source_and_redacted_config_views_ready_not_executable_benchmark',
        caveat='Performer configs exclude scoring, but raw source/testbeds still require per-task environment staging and expected-file withholding. Never mount the source tree as performer input.')
    (a.output/'manifest.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
