"""CPU reconstruction of each workflow prompt from public inputs and raw calls.

Use the run's verified frozen prompt helpers, including its original table
selection rule. Current helpers may represent a later experimental condition.
This validates prompt identity/input lengths, not the model's internal compute.
"""
import ast
import hashlib
import json
from pathlib import Path
from transformers import AutoTokenizer
from finqa_views import public_input
from finqa_executor import execute


def audit_prompts(run,protocol,labels,rows):
    source=(run/'code/finqa_roles.py').read_bytes()
    if hashlib.sha256(source).hexdigest()!=protocol['helpers_sha256']['finqa_roles.py']:
        raise ValueError('Prompt helper source changed')
    names={'payload','retrieve_prompt','plan_prompt','verify_prompt','select_evidence'}
    tree=ast.parse(source);body=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    if {node.name for node in body}!=names or len(body)!=len(names):raise ValueError('Missing frozen prompt definitions')
    helpers={'json':json};exec(compile(ast.Module(body=body,type_ignores=[]),str(run/'code/finqa_roles.py'),'exec'),helpers)
    model=Path(protocol['model']);base=json.loads((run/'base_manifest.json').read_text())
    for name,expected in base['files'].items():
        if Path(name).suffix in ['.json','.jinja','.txt','.model','.tiktoken']:
            if hashlib.sha256((model/name).read_bytes()).hexdigest()!=expected:raise ValueError('Changed tokenizer/config source '+name)
    tokenizer=AutoTokenizer.from_pretrained(str(model),local_files_only=True)
    checked=missing_header=0
    for row in rows:
        label=labels[row['id']];view=public_input(label);mode=protocol['mode'];prompts=[]
        if mode in ['single','base_refine','no_retriever']:selected=view
        else:
            if mode!='oracle_retrieval':prompts.append(helpers['retrieve_prompt'](view))
            else:
                available={e['id'] for e in view['evidence']}
                if row['retrieved_ids']!=[key for key in label['qa']['gold_inds'] if key in available]:
                    raise ValueError('Oracle evidence does not match declared diagnostic')
            selected=helpers['select_evidence'](view,row['retrieved_ids'])
        evidence_ids={e['id'] for e in selected['evidence']}
        missing_header+=bool(any(key.startswith('table_') and key!='table_0' for key in evidence_ids) and 'table_0' not in evidence_ids)
        prompts.append(helpers['plan_prompt'](selected))
        planners=[call for call in row['calls'] if call['role']=='planner']
        program=planners[0]['text'].strip()
        if mode=='base_refine':
            for call in planners[1:]:
                prompts.append(helpers['plan_prompt'](selected)+'\nRefine the previous attempt using the evidence. Previous program: '+program+'\nReturn only the improved program.')
                program=call['text'].strip()
        elif mode not in ['single','no_verifier']:
            prompts.append(helpers['verify_prompt'](selected,program,execute(program,label['table'])))
            if row['verdict']!='accept' and protocol['retry']:
                prompts.append(helpers['plan_prompt'](selected)+'\nThe previous program was rejected: '+program+'\nReturn a corrected program.')
        if len(prompts)!=len(row['calls']):raise ValueError('Reconstructed call count mismatch')
        for prompt,call in zip(prompts,row['calls']):
            prefix=tokenizer.apply_chat_template([dict(role='user',content=prompt)],tokenize=False,
                add_generation_prompt=True,enable_thinking=False)
            if hashlib.sha256(prefix.encode()).hexdigest()!=call['prompt_hash']:
                raise ValueError('Reconstructed prompt hash mismatch: '+row['id']+'/'+call['role'])
            if len(tokenizer.encode(prefix,add_special_tokens=False))!=call['input_tokens']:
                raise ValueError('Reconstructed prompt token count mismatch')
            checked+=1
    return dict(status='all_prompt_hashes_and_input_lengths_reconstructed',n=len(rows),calls=checked,
        samples_with_selected_table_rows_without_header=missing_header,
        scope='Public question/evidence and raw prior outputs reconstruct the declared frozen prompts. No generation/logits/timing re-execution; missing header count describes this representation, not a causal error attribution.')
