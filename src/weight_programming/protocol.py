"""Budgeted observation protocol derived from the MIT WSP Experiment 8 prompt.

Independent implementation. Validation is not a sandbox or a model backend.
"""
from dataclasses import asdict,dataclass
import fcntl
import json
from pathlib import Path
import time


@dataclass(frozen=True)
class ForwardPolicy:
    passes:int=50
    max_generations:int=512
    max_new_tokens:int=32768
    max_texts:int=1024
    max_text_tokens:int=32768
    max_total_text_tokens:int=524288
    max_topk:int=50
    max_attention_texts:int=2
    max_attention_tokens:int=2048
    max_return_bytes:int=6_000_000_000
    generation_activation_tokens:int=8192
    thinking:bool=True
    temperature:float=.6
    top_p:float=.95
    top_k:int=20
    presence_penalty:float=1.5
    seed:int=20261106

    def __post_init__(self):
        import math
        for key in ['passes','max_generations','max_new_tokens','max_texts','max_text_tokens','max_total_text_tokens',
                    'max_topk','max_attention_texts','max_attention_tokens','max_return_bytes','generation_activation_tokens']:
            if type(getattr(self,key))!=int or getattr(self,key)<=0:raise ValueError('Positive integer policy limit required: '+key)
        if type(self.thinking)!=bool or type(self.top_k)!=int or self.top_k<0 or type(self.seed)!=int:raise ValueError('Invalid sampling policy')
        if not all(math.isfinite(v) for v in [self.temperature,self.top_p,self.presence_penalty]) or self.temperature<=0 or not 0<self.top_p<=1:raise ValueError('Invalid sampling policy')


def integer(value,low,high,label):
    if type(value)!=int or not low<=value<=high:raise ValueError('Invalid '+label)
    return value


def indices(value,maximum,label):
    if not isinstance(value,list) or any(type(i)!=int for i in value) or len(set(value))!=len(value):raise ValueError('Unique '+label+' list required')
    return [integer(i,0,maximum,label) for i in value]


def validate_request(request,policy,num_layers,full_attention_layers):
    if not isinstance(request,dict):raise ValueError('Request object required')
    kind=request.get('type');model=request.get('model')
    if kind not in ['generate','teacher_forced'] or model not in ['original','edited','both']:raise ValueError('Unknown request type/model')
    request=json.loads(json.dumps(request,allow_nan=False))
    if kind=='generate':
        if set(request)-{'type','model','problems','prompts','n','max_tokens','activations'}:raise ValueError('Unknown generation option; sampling policy is fixed')
        problems=request.get('problems');prompts=request.get('prompts')
        if problems is None and prompts is None:raise ValueError('Problems or prompts required')
        for name,values in [('problems',problems),('prompts',prompts)]:
            if values is not None and (not isinstance(values,list) or not values or any(not isinstance(v,str) or not v.strip() for v in values)):raise ValueError('Nonempty '+name+' required')
        if problems is not None and len(set(problems))!=len(problems):raise ValueError('Use n for repeated samples, not duplicate problem IDs')
        if problems is not None and prompts is not None and len(problems)!=len(prompts):raise ValueError('Prompt overrides must match problem IDs')
        n=integer(request.get('n',1),1,policy.max_generations,'sample count');count=len(problems if problems is not None else prompts)
        if count*n*(2 if model=='both' else 1)>policy.max_generations:raise ValueError('Generation budget exceeded')
        request['n']=n;request['max_tokens']=integer(request.get('max_tokens',policy.max_new_tokens),1,policy.max_new_tokens,'new token cap')
        act=request.get('activations',{})
        if not isinstance(act,dict) or set(act)-{'layers','mlp'}:raise ValueError('Unknown generation observation')
        act['layers']=indices(act.get('layers',[]),num_layers,'residual layer')
        if type(act.get('mlp',False))!=bool:raise ValueError('MLP flag must be boolean')
        act['mlp']=act.get('mlp',False);request['activations']=act
    else:
        if model=='both':raise ValueError('Teacher-forced requests select one model')
        if set(request)-{'type','model','texts','chats','enable_thinking','return'}:raise ValueError('Unknown teacher-forced option')
        if ('texts' in request)==('chats' in request):raise ValueError('Choose texts or chats')
        values=request.get('texts',request.get('chats'))
        if not isinstance(values,list) or not 1<=len(values)<=policy.max_texts:raise ValueError('Text count outside limit')
        if 'texts' in request:
            if any(not isinstance(v,str) or not v for v in values):raise ValueError('Nonempty text strings required')
        else:
            for chat in values:
                if not isinstance(chat,list) or not chat:raise ValueError('Nonempty chats required')
                for message in chat:
                    if not isinstance(message,dict) or set(message)!={'role','content'} or message['role'] not in ['system','user','assistant'] or not isinstance(message['content'],str):raise ValueError('Unsupported chat message')
        if 'enable_thinking' in request and type(request['enable_thinking'])!=bool:raise ValueError('Thinking flag must be boolean')
        outputs=request.get('return',{})
        if not isinstance(outputs,dict) or set(outputs)-{'token_logprobs','topk','hidden_states','mlp_act','mlp_out','attn_layers'}:raise ValueError('Unknown observation return')
        if type(outputs.get('token_logprobs',False))!=bool:raise ValueError('Log-prob flag must be boolean')
        outputs['token_logprobs']=outputs.get('token_logprobs',False);outputs['topk']=integer(outputs.get('topk',0),0,policy.max_topk,'top-k return')
        outputs['hidden_states']=indices(outputs.get('hidden_states',[]),num_layers,'residual layer')
        outputs['mlp_act']=indices(outputs.get('mlp_act',[]),num_layers-1,'MLP layer')
        if 'mlp_out' in outputs:outputs['mlp_out']=indices(outputs['mlp_out'],num_layers-1,'MLP output layer')
        outputs['attn_layers']=indices(outputs.get('attn_layers',[]),num_layers-1,'attention layer')
        if set(outputs['attn_layers'])-set(full_attention_layers):raise ValueError('Attention maps requested for non-full-attention layers')
        if outputs['attn_layers'] and len(values)>policy.max_attention_texts:raise ValueError('Attention text budget exceeded')
        request['return']=outputs
    return request


class BudgetLedger:
    """Successful calls consume one pass; failures retain their actual cost record.

    Interrupted running entries remain reserved until explicitly investigated;
    opening a ledger does not silently reset them or repeat a model call.
    """
    def __init__(self,path,policy,binding):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True);self.policy=policy;self.binding=binding
        self._transaction(lambda data:None)

    def _transaction(self,operation):
        with open(self.path.with_suffix('.lock'),'a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            header=dict(version=1,policy=asdict(self.policy),binding=self.binding)
            if self.path.exists():
                data=json.loads(self.path.read_text())
                if any(data.get(k)!=v for k,v in header.items()):raise ValueError('Different service/data/policy binding')
            else:data=dict(**header,attempts=[])
            result=operation(data);temporary=self.path.with_suffix('.tmp');temporary.write_text(json.dumps(data,indent=2));temporary.replace(self.path)
            return result

    def begin(self,request,variant_id):
        def operation(data):
            used=sum(r['status']=='succeeded' for r in data['attempts']);pending=sum(r['status']=='running' for r in data['attempts'])
            if used+pending>=self.policy.passes:raise RuntimeError('Forward budget exhausted or reserved by a running/interrupted request')
            number=len(data['attempts'])+1;data['attempts'].append(dict(attempt=number,status='running',started_unix=time.time(),request=request,variant_id=variant_id));return number
        return self._transaction(operation)

    def finish(self,number,success,details):
        def operation(data):
            row=next(r for r in data['attempts'] if r['attempt']==number)
            if row['status']!='running':raise RuntimeError('Attempt already terminal')
            row.update(status='succeeded' if success else 'failed',finished_unix=time.time(),details=details)
        self._transaction(operation)

    def summary(self):
        return self._transaction(lambda d:dict(successful=sum(r['status']=='succeeded' for r in d['attempts']),
            failed=sum(r['status']=='failed' for r in d['attempts']),reserved=sum(r['status']=='running' for r in d['attempts']),limit=self.policy.passes))

    def lookup(self, number):
        """按运行序号读已有账本记录。"""
        data = json.loads(self.path.read_text())
        if not 1 <= number <= len(data['attempts']):
            raise KeyError(number)
        return data['attempts'][number - 1]
