"""Leakage-resistant FinQA inputs and explicitly training-only role targets."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def public_input(row):
    # Do not consume qa.model_input, gold_inds, answer, program, explanation,
    # steps, or precomputed retrieval fields even when they exist in test JSON.
    evidence=[]
    for i,text in enumerate(row['pre_text']+row['post_text']):
        evidence.append(dict(id=f'text_{i}',text=str(text)))
    for i,cells in enumerate(row['table']):
        evidence.append(dict(id=f'table_{i}',text=' | '.join(map(str,cells))))
    return dict(question=str(row['qa']['question']),evidence=evidence)


def training_targets(row,split):
    if split!='train':raise ValueError('Role target construction requires the official training split')
    view=public_input(row)
    available={r['id'] for r in view['evidence']}
    support=sorted(row['qa']['gold_inds'])
    unknown=set(support)-available
    if unknown:raise ValueError('Unresolved official evidence IDs: '+str(sorted(unknown)))
    return dict(retriever=dict(input=view,target=json.dumps(support)),
        planner=dict(input=view,target=str(row['qa']['program'])),
        note='Verifier targets require independently validated executor outcomes; not fabricated here.')


def question_signature(row):
    view=public_input(row)
    text=re.sub(r'\s+',' ',view['question'].lower()).strip()
    return hashlib.sha256(text.encode()).hexdigest()


def audit(directory,output):
    report={}
    signatures={}
    for split in ['train','dev','test']:
        rows=json.loads((directory/(split+'.json')).read_text())
        ids=[r['id'] for r in rows]
        signatures[split]={question_signature(r) for r in rows}
        unknown=[]
        for row in rows:
            view=public_input(row)
            if any(key in view for key in ['answer','program','gold_inds','model_input']):
                raise AssertionError('Inference view leaks a target')
            missing=set(row['qa']['gold_inds'])-{e['id'] for e in view['evidence']}
            if missing:unknown.append(dict(id=row['id'],missing=sorted(missing)))
        report[split]=dict(n=len(rows),unique_ids=len(set(ids)),unique_questions=len(signatures[split]),
            unresolved_evidence_ids=unknown,
            inference_fields=['question','evidence.id','evidence.text'])
    report['question_overlap']={f'{a}:{b}':len(signatures[a]&signatures[b])
        for a,b in [('train','dev'),('train','test'),('dev','test')]}
    report['scope']='Input-view and annotation audit only. Question overlap does not alone prove training contamination. No application quality score.'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k in ['question_overlap','scope']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    audit(args.data,args.output)
