"""Pinned HotpotQA distractor views for collaboration-before-specialization.

Public inputs contain candidate documents, never answer/support labels. This
is candidate-document reading, not fullwiki search. Uses the official dataset
namespace through a transport mirror and preserves the pinned official scorer.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

DATA_REV='1908d6afbbead072334abe2965f91bd2709910ab'
SCORER_REV='3635853403a8735609ee997664e1528f4480762a'
DATA_FILES=['distractor/train-00000-of-00002.parquet','distractor/train-00001-of-00002.parquet',
    'distractor/validation-00000-of-00001.parquet']


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def get(url,path):
    if path.exists():return
    path.parent.mkdir(parents=True,exist_ok=True);partial=path.with_suffix(path.suffix+'.partial')
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'idea4-research'}),timeout=60) as response,open(partial,'wb') as stream:
        while True:
            block=response.read(8*1024*1024)
            if not block:break
            stream.write(block)
    partial.replace(path)


def rows(paths):
    import pyarrow.parquet as pq
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
            yield from batch.to_pylist()


def question_key(row):return hashlib.sha256(' '.join(row['question'].lower().split()).encode()).hexdigest()


def convert(row):
    c=row['context'];s=row['supporting_facts']
    public=dict(id=row['id'],question=row['question'],documents=[dict(title=t,sentences=list(v)) for t,v in zip(c['title'],c['sentences'])])
    label=dict(_id=row['id'],answer=row['answer'],supporting_facts=list(map(list,zip(s['title'],s['sent_id']))))
    return public,label


def dump(path,items):
    with open(path,'w') as stream:
        for row in items:stream.write(json.dumps(row,ensure_ascii=False)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--endpoint',default='https://hf-mirror.com');a=p.parse_args()
    if (a.output/'manifest.json').exists():raise FileExistsError('View already constructed')
    a.raw.mkdir(parents=True,exist_ok=True);a.output.mkdir(parents=True,exist_ok=True)
    for name in DATA_FILES:
        print('download',name,flush=True)
        get(f'{a.endpoint}/datasets/hotpotqa/hotpot_qa/resolve/{DATA_REV}/{name}',a.raw/Path(name).name)
    scorer=a.raw/'hotpot_evaluate_v1.py'
    get(f'https://raw.githubusercontent.com/hotpotqa/hotpot/{SCORER_REV}/hotpot_evaluate_v1.py',scorer)
    validation=list(rows([a.raw/Path(DATA_FILES[2]).name]))
    if len(validation)!=7405 or len({r['id'] for r in validation})!=7405:raise ValueError('Wrong official dev partition')
    excluded={question_key(row) for row in validation};train_paths=[a.raw/Path(n).name for n in DATA_FILES[:2]]
    candidates=[];seen=set();n=0;excluded_n=0
    for row in rows(train_paths):
        n+=1
        if row['id'] in seen:raise ValueError('Duplicate training ID')
        seen.add(row['id'])
        if question_key(row) in excluded:excluded_n+=1;continue
        candidates.append((hashlib.sha256(('idea4-hotpot-v1:'+row['id']).encode()).hexdigest(),row['id']))
    if n!=90447:raise ValueError('Wrong official train partition')
    candidates.sort();cal_ids={i for _,i in candidates[:128]};dev_ids={i for _,i in candidates[128:384]}
    views={'calibration':[],'development':[],'official_dev':validation}
    for row in rows(train_paths):
        if row['id'] in cal_ids:views['calibration'].append(row)
        elif row['id'] in dev_ids:views['development'].append(row)
    for name,selected in views.items():
        if name!='official_dev':selected.sort(key=lambda r:hashlib.sha256(('idea4-hotpot-v1:'+r['id']).encode()).hexdigest())
        pairs=[convert(row) for row in selected]
        dump(a.output/(name+'.inputs.jsonl'),[p for p,_ in pairs]);dump(a.output/(name+'.labels.jsonl'),[q for _,q in pairs])
    (a.output/'official_scorer.py').write_bytes(scorer.read_bytes())
    manifest=dict(version='hotpot-collaboration-v1',task='HotpotQA distractor candidate-document reading',
        dataset=dict(repo='hotpotqa/hotpot_qa',revision=DATA_REV,transport=a.endpoint,files={n:sha(a.raw/Path(n).name) for n in DATA_FILES}),
        official_scorer=dict(repo='hotpotqa/hotpot',revision=SCORER_REV,sha256=sha(scorer)),
        sizes={k:len(v) for k,v in views.items()},raw_train_count=n,excluded_exact_normalized_dev_questions=excluded_n,
        split_rule='sha256(idea4-hotpot-v1:ID), first128 calibration, next256 development; rest reserved for future role training',
        reserved_training_ids=[i for _,i in candidates[384:]],
        labels='Separate exports; inference loader rejects unauthorized input fields; scoring after all predictions.',
        source_sha256=sha(Path(__file__)),files={f.name:sha(f) for f in a.output.iterdir() if f.is_file()})
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({k:v for k,v in manifest.items() if k not in ['reserved_training_ids','files']},indent=2),flush=True)


if __name__=='__main__':main()
