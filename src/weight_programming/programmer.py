"""Fresh-context, provider-neutral driver for direct weight programming.

It records model choices, tool execution and budget state. It does not infer
independence merely from a clean transcript: callers must control the model,
initial prompt, exposed practice data and mounts. No reference edit is loaded.
"""
import hashlib
import json
from pathlib import Path
import re
import time
from .tool_api import tool_schemas,dispatch_tool


SYSTEM='''You are programming an open-weight language model through the supplied tools.
Start from your own analysis of the provided weights and authorized practice inputs.
Use only observations from this experiment. Diagnose failure before proposing an edit;
compare controlled interventions and several strengths, then confirm on unused practice.
No training, backward passes, autograd, optimizers, iterative fitting to a loss,
or simulation of the network inside the CPU kernel. Extended activation intervention
and reconstruction-fitting tools are outside this original-style experiment track.
Do not edit protected tensors, rescale an entire tensor, or force early thought closure.
Keep /work/notes.md current with hypotheses, measured results and the best saved edits.
The CPU namespace provides NAMES, load(name), load_original(name),
save_edit(name, tensor), drop_edit(name), edited(), and tok (the tokenizer).
Model files are read-only in /model, authorized practice in /practice;
completed query artifacts are read-only in /work/observations. /work is writable.
Every reply must be exactly one fenced block: python for CPU code, forward for a JSON
model request, or final for a report. Do not include any other text. Variables persist
between CPU code calls; context rollovers preserve that kernel and your notes.
'''


def parse_reply(text):
    if not isinstance(text,str):raise ValueError('Programmer response text required')
    match=re.fullmatch(r'```(python|forward|final)\r?\n([\s\S]*?)\r?\n```',text.strip())
    if match is None:raise ValueError('Return exactly one python, forward or final fenced block')
    kind,body=match.groups()
    if kind=='python':return dict(name='execute_python',arguments={'code':body})
    if kind=='forward':
        request=json.loads(body)
        if not isinstance(request,dict):raise ValueError('Forward request must be an object')
        return dict(name='model_forward',arguments=request)
    if not body.strip():raise ValueError('Nonempty final report required')
    return dict(name='final',report=body)


def notes_text(work,maximum=12000):
    work=Path(work).resolve();path=work/'notes.md'
    if not path.exists():return '(No notes.md has been written.)'
    if path.is_symlink() or path.resolve().parent!=work:return '(Rejected nonlocal notes path.)'
    with path.open('r',encoding='utf-8',errors='replace') as stream:notes=stream.read(maximum+1)
    return notes[:maximum]+('\n[notes truncated]' if len(notes)>maximum else '')


def run_programmer(session,generate,initial_prompt,*,programmer_identity,
                   count_tokens=None,max_turns=500,max_reply_tokens=8192,max_context_tokens=32768,seed=401,
                   require_forward_budget=False):
    """generate(messages, tools, max_tokens, seed) -> text/token/time record.

    Tool errors are returned to the model with the actual remaining budgets.
    Provider failures and lost kernels terminate explicitly; no silent restarts.
    Every full response/tool result is retained outside the kernel-visible tree.
    """
    if not isinstance(initial_prompt,str) or not initial_prompt.strip():raise ValueError('Explicit initial task required')
    if min(max_turns,max_reply_tokens,max_context_tokens)<=0:raise ValueError('Positive programmer limits required')
    out=session.control/'programmer'
    if out.exists():raise FileExistsError('A fresh programmer transcript is required')
    out.mkdir();tools=tool_schemas(session)
    initial=[dict(role='system',content=SYSTEM),dict(role='user',content=initial_prompt)]
    messages=list(initial);started=time.perf_counter();rollovers=0;input_tokens=output_tokens=0;status='turn_limit'
    protocol=dict(programmer=programmer_identity,initial_prompt=initial_prompt,system=SYSTEM,tools=tools,
        initial_sha256=hashlib.sha256(json.dumps(initial,sort_keys=True).encode()).hexdigest(),
        max_turns=max_turns,max_reply_tokens=max_reply_tokens,max_context_tokens=max_context_tokens,seed=seed,
        require_forward_budget=require_forward_budget,
        target=session.service.binding,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        independence='Fresh conversation only; model provenance and allowed mounts remain caller obligations.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    def budget():
        return dict(python_executions=session.kernel.count,python_limit=session.kernel.limit,
            kernel_live=not session.kernel.dead,forward=session.service.ledger.summary())
    def save(record):
        with (out/'turns.jsonl').open('a') as stream:stream.write(json.dumps(record,ensure_ascii=False)+'\n')
    def finish():
        snapshot=session.service.store.snapshot()
        result=dict(status=status,turns=completed_turns,
            rollovers=rollovers,input_tokens=input_tokens,output_tokens=output_tokens,
            elapsed_seconds=time.perf_counter()-started,budget=budget(),
            variant_id=snapshot['variant_id'],base_id=snapshot['base_id'],
            edited_names=sorted(snapshot['values']),native_tensor_sha256=snapshot['native_tensor_sha256'])
        (out/'complete.json').write_text(json.dumps(result,indent=2));return result
    completed_turns=0
    for turn in range(max_turns):
        current=budget()
        if not current['kernel_live']:status='kernel_lost';break
        if current['python_executions']>=current['python_limit'] and current['forward']['successful']>=current['forward']['limit']:
            status='tool_budgets_exhausted';break
        if count_tokens is not None and count_tokens(messages,tools)+max_reply_tokens>max_context_tokens:
            rollovers+=1
            messages=list(initial)+[dict(role='user',content='CONTEXT ROLLOVER; same live kernel and edit store.\nBUDGET:\n'+json.dumps(current)+'\nYOUR NOTES:\n'+notes_text(session.work))]
            save(dict(event='context_rollover',before_turn=turn,budget=current,notes=notes_text(session.work)))
            if count_tokens(messages,tools)+max_reply_tokens>max_context_tokens:
                status='context_limit';break
        call_seed=int(hashlib.sha256(f'{seed}:{turn}'.encode()).hexdigest()[:8],16)
        try:response=generate(messages,tools,max_reply_tokens,call_seed)
        except Exception as error:
            status='programmer_error';save(dict(turn=turn,error_type=type(error).__name__,error=str(error),budget=budget()));finish();raise
        if not isinstance(response,dict) or not isinstance(response.get('text'),str):
            status='programmer_error';save(dict(turn=turn,error='Invalid programmer response',budget=budget()));return finish()
        accounting_error=None
        for key in ['input_tokens','output_tokens']:
            if type(response.get(key))!=int or response[key]<0:accounting_error='Programmer must report actual token counts'
        if accounting_error is None and response['output_tokens']>max_reply_tokens:accounting_error='Programmer exceeded reply cap'
        if accounting_error is not None:
            status='programmer_error'
            save(dict(turn=turn,response=response,error=accounting_error,budget=budget()))
            return finish()
        input_tokens+=response['input_tokens'];output_tokens+=response['output_tokens'];completed_turns+=1
        text=response['text'];messages.append(dict(role='assistant',content=text))
        record=dict(turn=turn,response=response,seed=call_seed,budget_before=current)
        try:
            operation=parse_reply(text);record['operation']=operation
            if operation['name']=='final':
                if require_forward_budget and budget()['forward']['successful']<budget()['forward']['limit']:
                    raise ValueError('This protocol requires completing the forward-pass budget before the final choice')
                (out/'final_report.md').write_text(operation['report']);status='finished'
                record['budget_after']=budget();save(record);break
            result=dispatch_tool(session,operation['name'],operation['arguments']);record['tool_result']=result
            if operation['name']=='model_forward':
                visible={k:v for k,v in result.items() if k in ['artifact_dir','budget','summaries','paired','files','alignment','input_tokens','executed_model_forwards','elapsed_seconds','variant_id']}
            else:visible=result
            rendered=json.dumps(dict(status='ok',result=visible,budget=budget()),ensure_ascii=True)
            if len(rendered)>10000:
                rendered=json.dumps(dict(status='ok',truncated=True,artifact_dir=result.get('artifact_dir'),budget=budget(),instruction='Inspect the saved artifact metadata or selected arrays in the CPU kernel.'))
        except Exception as error:
            record['tool_error']=dict(type=type(error).__name__,message=str(error))
            rendered=json.dumps(dict(status='error',error=record['tool_error'],budget=budget()))
        record['budget_after']=budget();save(record)
        messages.append(dict(role='user',content='TOOL RESULT:\n'+rendered))
    return finish()
