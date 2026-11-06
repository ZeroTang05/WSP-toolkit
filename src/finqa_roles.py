"""Training-only role supervision and label-free FinQA workflow prompts.

Teacher-forced gold evidence is used only in training/oracle diagnostics.
The verifier label means official execution-answer agreement, not proof of
financial grounding. Synthetic negatives are a baseline, not reliable coverage
of errors produced by a trained planner.
"""
import argparse
import hashlib
import json
from pathlib import Path
from finqa_views import public_input
from finqa_executor import execute,REFERENCE,META
from prepare_training_views import write_rows,sha


def payload(view):return json.dumps(view,ensure_ascii=False)


def retrieve_prompt(view):
    return 'Select relevant evidence IDs. Return only a JSON list of IDs.\n'+payload(view)


def plan_prompt(view):
    return ('Read the question and evidence. Return only a FinQA arithmetic program. '
        'Operators: add, subtract, multiply, divide, exp, greater, table_max, table_min, table_sum, table_average. '
        'Use calls op(arg1, arg2) separated by comma-space. Arguments are numbers, const_100, const_m1, '
        'or prior results #0, #1, etc.; table operators use a row name and none. '
        'Format example: subtract(8, 2), divide(#0, 2).\n'+payload(view))


def verify_prompt(view,program,outcome):
    return ('Check whether the program and its executed answer correctly answer the question using the evidence. '
        'Return only accept or reject.\n'+payload(dict(**view,program=program,execution=outcome)))


def select_evidence(view,ids):
    chosen=set(ids)
    # A raw row contains values but no column/year labels. Retain the public
    # header whenever a real nonheader table row is selected. Never infer it
    # from gold evidence text; retrieval predictions/metrics remain unchanged.
    available={e['id'] for e in view['evidence']}
    if any(key.startswith('table_') and key!='table_0' for key in chosen & available):
        chosen.add('table_0')
    return dict(question=view['question'],evidence=[e for e in view['evidence'] if e['id'] in chosen])


def parse_retrieval(text,view):
    try:ids=json.loads(text.strip())
    except (ValueError,TypeError):return [],False
    available={e['id'] for e in view['evidence']}
    if not isinstance(ids,list) or any(not isinstance(x,str) or x not in available for x in ids):return [],False
    return list(dict.fromkeys(ids)),True


def render(tokens):
    return ', '.join(tokens[i][:-1]+'('+tokens[i+1]+', '+tokens[i+2]+')'
        for i in range(0,len(tokens)-1,4))


def negative_program(row):
    original=REFERENCE['program_tokenization'](row['qa']['program'])
    candidates=[]
    for i in range(0,len(original)-1,4):
        op=original[i][:-1]
        replacements=['table_max','table_min','table_sum','table_average'] if op.startswith('table_') else ['add','subtract','multiply','divide','greater']
        for replacement in replacements:
            candidate=original.copy();candidate[i]=replacement+'(';candidates.append(render(candidate))
        if not op.startswith('table_'):
            for j in [i+1,i+2]:
                if original[j].startswith('#'):continue
                try:number=REFERENCE['str_to_num'](original[j])
                except (ValueError,OverflowError):continue
                if not isinstance(number,(float,int)):continue
                candidate=original.copy();candidate[j]=str(number+1);candidates.append(render(candidate))
    for program in candidates:
        result=execute(program,row['table'])
        if result['valid'] and result['result']!=row['qa']['exe_ans']:return program,'valid_execution_wrong_answer'
    # Explicit hard failure, separately counted. Never relabel a correct program.
    return 'divide(1, 0)','invalid_execution_fallback'


def compile_views(raw,output):
    if output.exists():raise FileExistsError(output)
    rows=json.loads((raw/'train.json').read_text())
    targets={key:[] for key in ['retriever_train.jsonl','planner_train.jsonl','verifier_train.jsonl']}
    audit=dict(unknown_evidence=[],verifier_positive=0,verifier_negative_kinds={},header_added_training_examples=0)
    for row in rows:
        view=public_input(row);support=sorted(row['qa']['gold_inds'])
        unknown=set(support)-{e['id'] for e in view['evidence']}
        common=dict(id=row['id'],source_split='train',view_role='train')
        if not unknown:
            targets['retriever_train.jsonl'].append(dict(**common,role='retriever',prompt=retrieve_prompt(view),answer=json.dumps(support)))
        else:audit['unknown_evidence'].append(dict(id=row['id'],ids=sorted(unknown)))
        selected=select_evidence(view,support)
        if 'table_0' not in support and any(e['id']=='table_0' for e in selected['evidence']):
            audit['header_added_training_examples']+=1
        # Keep planner/verifier examples; unresolved evidence is disclosed, not silently replaced by gold text.
        targets['planner_train.jsonl'].append(dict(**common,role='planner',prompt=plan_prompt(selected),answer=row['qa']['program']))
        positive=int(hashlib.sha256(row['id'].encode()).hexdigest(),16)%2==0
        if positive:
            program=row['qa']['program'];audit['verifier_positive']+=1
        else:
            program,kind=negative_program(row)
            audit['verifier_negative_kinds'][kind]=audit['verifier_negative_kinds'].get(kind,0)+1
        outcome=execute(program,row['table'])
        label='accept' if outcome['valid'] and outcome['result']==row['qa']['exe_ans'] else 'reject'
        if (label=='accept')!=positive:raise AssertionError('Verifier label generation failed')
        targets['verifier_train.jsonl'].append(dict(**common,role='verifier',
            prompt=verify_prompt(selected,program,outcome),answer=label))
    targets['multitask_train.jsonl']=sum((targets[key] for key in list(targets)),[])
    output.mkdir(parents=True)
    for name,values in targets.items():write_rows(output/name,values)
    # Inference exports never include target answers, programs or support IDs.
    for split in ['dev','test']:
        examples=json.loads((raw/(split+'.json')).read_text())
        write_rows(output/(split+'_inputs.jsonl'),[dict(id=row['id'],**public_input(row),table=row['table']) for row in examples])
    manifest=dict(dataset='FinQA',version='roles-v4-header',official_train_sha256=sha(raw/'train.json'),
        counts={name:len(values) for name,values in targets.items()},audit=audit,
        files={p.name:sha(p) for p in output.glob('*.jsonl')},executor_reference_commit=META['commit'],
        training_evidence='gold support IDs intersect public evidence, plus public table_0 for selected table rows; test uses predicted evidence with same header rule',
        table_representation='raw pipe-separated rows plus preserved public header; adaptation, not author row-to-text reproduction',
        verifier_labels='official execution equality on deterministic gold/operator/operand perturbations; not evidence entailment proof',
        limitations=['teacher-forcing distribution shift','synthetic rather than out-of-fold model-error negatives',
            'multitask random-sampling role proportions follow data counts, report realized proportions'])
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(compile_views(a.raw,a.output),indent=2))
