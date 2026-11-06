"""Provider-neutral tool schemas for the original WSP-style permission track.

The schemas describe real ProgrammingSession methods, not imaginary model-edit
operations. Runtime validation remains authoritative for coupled limits.
"""
def tool_schemas(session):
    backend=session.service.backend;policy=session.service.policy
    integer=lambda maximum:dict(type='integer',minimum=0,maximum=maximum)
    indices=lambda maximum:dict(type='array',items=integer(maximum),uniqueItems=True)
    strings=dict(type='array',minItems=1,items=dict(type='string',minLength=1))
    generation=dict(type='object',additionalProperties=False,
        required=['type','model'],properties=dict(
            type=dict(const='generate'),model=dict(enum=['original','edited','both']),
            problems=dict(strings,uniqueItems=True),prompts=strings,
            n=dict(type='integer',minimum=1,maximum=policy.max_generations),
            max_tokens=dict(type='integer',minimum=1,maximum=policy.max_new_tokens),
            activations=dict(type='object',additionalProperties=False,properties=dict(
                layers=indices(backend.num_layers),mlp=dict(type='boolean')))),
        anyOf=[dict(required=['problems']),dict(required=['prompts'])])
    message=dict(type='object',additionalProperties=False,required=['role','content'],
        properties=dict(role=dict(enum=['system','user','assistant']),content=dict(type='string')))
    teacher=dict(type='object',additionalProperties=False,required=['type','model'],
        properties={
            'type':dict(const='teacher_forced'),'model':dict(enum=['original','edited']),
            'texts':dict(strings,maxItems=policy.max_texts),
            'chats':dict(type='array',minItems=1,maxItems=policy.max_texts,items=dict(type='array',minItems=1,items=message)),
            'enable_thinking':dict(type='boolean'),
            'return':dict(type='object',additionalProperties=False,properties=dict(
                token_logprobs=dict(type='boolean'),topk=integer(policy.max_topk),
                hidden_states=indices(backend.num_layers),mlp_act=indices(backend.num_layers-1),
                attn_layers=dict(type='array',uniqueItems=True,items=dict(type='integer',enum=list(backend.full_attention_layers)))))},
        oneOf=[{'required':['texts'],'not':{'required':['chats']}},{'required':['chats'],'not':{'required':['texts']}}])
    return [dict(name='execute_python',
        description='Run CPU analysis code in the persistent isolated weight kernel. Variables persist; outputs are capped. No training, autograd, optimizers, iterative loss fitting or model simulation in the original track. Exceptions consume a code execution.',
        parameters=dict(type='object',additionalProperties=False,required=['code'],properties=dict(code=dict(type='string',minLength=1)))),
        dict(name='model_forward',
            description='Run a budgeted model query on original/edited weights. Read prompts, token scores and dense activations using the frozen backend and practice registry. Coupled generation, token, attention and return-byte limits are validated at execution. Completed arrays are exported under /work/observations.',
            parameters=dict(type='object',oneOf=[generation,teacher]))]


def dispatch_tool(session,name,arguments):
    if not isinstance(arguments,dict):raise ValueError('Tool arguments must be a JSON object')
    if name=='execute_python':
        if set(arguments)!={'code'} or not isinstance(arguments['code'],str) or not arguments['code'].strip():
            raise ValueError('Only nonempty code is accepted')
        return session.execute(arguments['code'])
    if name=='model_forward':return session.forward(arguments)
    raise ValueError('Unknown tool; extended interventions are not authorized by the original-track dispatcher')
