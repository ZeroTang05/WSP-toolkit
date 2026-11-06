"""Diagnose superficial operator cues in training-only verifier supervision.

The heldout split is only for this count classifier; it is not heldout from
language-model training, and its score is not model verification accuracy.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re


def operator(program):
    match=re.match(r'\s*([a-z_]+)\(',program)
    return match[1] if match else 'invalid_prefix'


def inspect(view):
    path=view/'verifier_train.jsonl';payload=path.read_bytes()
    manifest=json.loads((view/'manifest.json').read_text());digest=hashlib.sha256(payload).hexdigest()
    if manifest['files'][path.name]!=digest:raise ValueError('Changed training supervision')
    groups=[[],[]];counts=collections.defaultdict(collections.Counter)
    for row in map(json.loads,payload.decode().splitlines()):
        if row['source_split']!='train' or row['view_role']!='train' or row['answer'] not in ['accept','reject']:
            raise ValueError('Not training-only binary verifier supervision')
        context=json.loads(row['prompt'].split('\n',1)[1]);op=operator(context['program'])
        group=row.get('question_id',row['id']);counts[op][row['answer']]+=1
        held=int(hashlib.sha256(('operator-probe-split-v1:'+group).encode()).hexdigest(),16)%5==0
        groups[int(held)].append(dict(group=group,op=op,label=row['answer']))
    train,probe=groups
    if not train or not probe or {r['group'] for r in train}&{r['group'] for r in probe}:raise ValueError('Invalid question grouping')
    train_counts=collections.defaultdict(collections.Counter);global_counts=collections.Counter(r['label'] for r in train)
    for row in train:train_counts[row['op']][row['label']]+=1
    majority=lambda count:max(sorted(count),key=lambda key:count[key])
    fallback=majority(global_counts);rules={key:majority(value) for key,value in train_counts.items()}
    confusion=collections.Counter((rules.get(r['op'],fallback),r['label']) for r in probe)
    correct=sum(count for (pred,gold),count in confusion.items() if pred==gold)
    return dict(status='training_supervision_operator_probe_verified',source_sha256=digest,
        operator_label_counts={key:dict(value) for key,value in sorted(counts.items())},
        split='SHA256(operator-probe-split-v1:+question_id) modulo5==0 held out from count-classifier fitting',
        fit_n=len(train),probe_n=len(probe),fit_labels=dict(global_counts),probe_labels=dict(collections.Counter(r['label'] for r in probe)),
        rules=rules,correct=correct,probe_accuracy=correct/len(probe),
        fitted_global_majority_label=fallback,fitted_global_majority_probe_accuracy=sum(r['label']==fallback for r in probe)/len(probe),
        confusion={pred+'_'+gold:count for (pred,gold),count in confusion.items()},
        scope='Synthetic supervision diagnosis only. Probe examples may be in LLM SFT. Operator-only heldout score is not LLM verification quality, real-error performance or causal evidence of a learned shortcut.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--view',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();result=inspect(a.view)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
