"""Bounded FinQA DSL execution using pinned official evaluation primitives.

Only numeric execution accuracy is exposed here; symbolic program accuracy is
not approximated. No generated Python or symbolic-expression eval is executed.
"""
import ast
import hashlib
import json
import math
import sys
from pathlib import Path


VENDOR=Path(__file__).resolve().parents[1]/'vendor/finqa'
if not VENDOR.exists():VENDOR=Path(sys.prefix)/'share/idea4-weightlab/finqa'
META=json.loads((VENDOR/'provenance.json').read_text())
SOURCE=(VENDOR/'evaluate.py').read_bytes()
if hashlib.sha256(SOURCE).hexdigest()!=META['sha256']['evaluate.py']:
    raise ValueError('Pinned FinQA evaluator changed')
NAMES={'str_to_num','process_row','eval_program','program_tokenization'}
TREE=ast.parse(SOURCE)
BODY=[node for node in TREE.body if
    isinstance(node,ast.FunctionDef) and node.name in NAMES or
    isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='all_ops' for t in node.targets)]
REFERENCE={}
exec(compile(ast.Module(body=BODY,type_ignores=[]),str(VENDOR/'evaluate.py'),'exec'),REFERENCE)


def execute(program,table):
    """Accept only a complete bounded DSL; preserve official tokenization/scoring."""
    if not isinstance(program,str) or not 0<len(program)<=4096:
        return dict(valid=False,result='n/a',reason='program_length')
    program=program.strip()
    tokens=REFERENCE['program_tokenization'](program)
    steps=(len(tokens)-1)//4
    if not 1<=steps<=32 or (len(tokens)-1)%4:
        return dict(valid=False,result='n/a',reason='program_structure')
    for index in range(steps):
        op,x,y,end=tokens[index*4:index*4+4]
        if not op.endswith('(') or op[:-1] not in REFERENCE['all_ops'] or end!=')':
            return dict(valid=False,result='n/a',reason='operator_structure')
        if not op.startswith('table_'):
            for value in (x,y):
                if value.startswith('#'):
                    if not value[1:].isdigit() or int(value[1:])>=index:
                        return dict(valid=False,result='n/a',reason='reference')
                else:
                    try:number=REFERENCE['str_to_num'](value)
                    except (ValueError,OverflowError):number='n/a'
                    if number=='n/a' or not math.isfinite(number):
                        return dict(valid=False,result='n/a',reason='number')
    invalid,result=REFERENCE['eval_program'](tokens,table)
    if isinstance(result,(float,int)) and not math.isfinite(result):invalid=1
    return dict(valid=not bool(invalid),result=result if not invalid else 'n/a',
        reason='official_execution_failure' if invalid else None)


def score(program,row):
    outcome=execute(program,row['table'])
    return dict(**outcome,execution_correct=outcome['valid'] and outcome['result']==row['qa']['exe_ans'])


def audit(raw,output):
    report=dict(evaluator_commit=META['commit'],evaluator_sha256=META['sha256']['evaluate.py'],
        metric='official numeric/string execution equality after official rounding to 5 decimals',
        scope='gold-program round-trip diagnostic; not model accuracy or symbolic program accuracy',splits={})
    for split in ['train','dev','test']:
        rows=json.loads((raw/(split+'.json')).read_text())
        discrepancies=[]
        for row in rows:
            result=score(row['qa']['program'],row)
            if not result['execution_correct']:discrepancies.append(dict(id=row['id'],**result,gold=row['qa']['exe_ans']))
        report['splits'][split]=dict(n=len(rows),gold_roundtrip_correct=len(rows)-len(discrepancies),
            discrepancies=discrepancies)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--raw',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();r=audit(a.raw,a.output)
    print(json.dumps({k:dict(n=v['n'],correct=v['gold_roundtrip_correct']) for k,v in r['splits'].items()}))
