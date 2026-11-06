"""Executable dense observations for pinned Transformers Qwen3/Llama decoders.

This backend is NOT the author's vLLM generator and does not yet support Qwen3.5.
The model is supplied by the host, which owns GPU scheduling and source loading.
"""
import contextlib
import hashlib
import importlib.metadata
from pathlib import Path
import time
import threading
import numpy as np
import torch
from transformers import LogitsProcessor,LogitsProcessorList


class PresencePenalty(LogitsProcessor):
    """Subtract once for tokens present in generated history, excluding prompt."""
    def __init__(self,prompt_length,penalty):self.length=prompt_length;self.penalty=penalty;self.seen=None
    def __call__(self,input_ids,scores):
        if self.seen is None:self.seen=torch.zeros_like(scores,dtype=torch.bool)
        if self.seen.shape!=scores.shape:raise ValueError('Presence processor requires stable non-beam batch shape')
        if input_ids.shape[1]<self.length:raise ValueError('Generated history shrank')
        fresh=input_ids[:,self.length:]
        if fresh.numel():self.seen.scatter_(1,fresh,True)
        self.length=input_ids.shape[1]
        return scores-self.penalty*self.seen.to(scores.dtype)


class HuggingFaceBackend:
    def __init__(self,model,tokenizer,store,verify_tensors=True):
        if importlib.metadata.version('transformers')!='4.57.6':raise ValueError('This adapter is pinned to Transformers 4.57.6; add a separately verified adapter for other versions')
        if model.config.model_type not in ['qwen3','llama']:raise ValueError('Unimplemented model adapter; Qwen3.5 requires its own runtime')
        self.model=model.eval();self.model.requires_grad_(False);self.tokenizer=tokenizer;self.store=store
        self._execution_lock=threading.RLock()
        self.decoder=model.model;self.layers=self.decoder.layers;self.num_layers=len(self.layers)
        self.hidden_size=model.config.hidden_size;self.mlp_width=model.config.intermediate_size
        self.vocab_size=model.config.vocab_size;self.num_heads=model.config.num_attention_heads
        self.context_length=model.config.max_position_embeddings
        self.full_attention_layers=tuple(range(self.num_layers));self.device=next(model.parameters()).device
        if self.device.type not in ['cpu','cuda'] or any(p.device!=self.device for p in model.parameters()):raise ValueError('This adapter supports one CPU/CUDA device only')
        self.parameters=dict(model.named_parameters(remove_duplicate=False))
        if Path(model.config._name_or_path).resolve()!=store.model_dir:raise ValueError('Model path differs from verified source')
        if verify_tensors:
            state=model.state_dict()
            for name in store.names:
                if name not in state:raise ValueError('Source tensor missing from executable model: '+name)
                original=store._native(name);actual=state[name].detach().cpu()
                if original.shape!=actual.shape or not torch.equal(original.to(actual.dtype),actual):raise ValueError('Loaded model differs from base tensor: '+name)
        self.identity=dict(backend='hf-qwen3-llama-v1',transformers=importlib.metadata.version('transformers'),
            torch=torch.__version__,device=str(self.device),dtype=str(next(model.parameters()).dtype),model_type=model.config.model_type,
            source_verified_tensorwise=verify_tensors,num_layers=self.num_layers,full_attention_layers=list(self.full_attention_layers),
            attention_implementation=model.config._attn_implementation,
            note='Pinned independent HF implementation; not a vLLM equivalence claim or Qwen3.5 adapter')

    def encode(self,text):
        ids=self.tokenizer.encode(text,add_special_tokens=False)
        if not ids:raise ValueError('Empty tokenized text')
        return ids

    def chat(self,messages,thinking,generate=False):
        return self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=generate,enable_thinking=thinking)

    @contextlib.contextmanager
    def activate(self,snapshot):
        with self._execution_lock:
            with self._activate_unlocked(snapshot):yield

    @contextlib.contextmanager
    @torch.no_grad()
    def _activate_unlocked(self,snapshot):
        backups={}
        try:
            for name,value in snapshot['values'].items():
                if name not in self.parameters:raise ValueError('Editable source is not a model parameter')
                parameter=self.parameters[name]
                if parameter.dtype!=self.store._native(name).dtype:raise ValueError('Edited parameter execution precision differs from source precision')
                backups[name]=parameter.detach().cpu().clone();parameter.copy_(value.to(parameter.device,dtype=parameter.dtype))
            yield
        finally:
            for name,value in backups.items():self.parameters[name].copy_(value.to(self.parameters[name].device))

    @contextlib.contextmanager
    def _observe(self,hidden,mlp,attention):
        captured={};handles=[];configs={}
        def save(key,value):captured[key]=value.detach().float().cpu().clone()
        def post(key):
            def hook(module,args,out):save(key,out[0] if isinstance(out,(tuple,list)) else out)
            return hook
        try:
            for layer in hidden:
                module=self.decoder.embed_tokens if layer==0 else self.layers[layer-1]
                handles.append(module.register_forward_hook(post('hidden_'+str(layer))))
            for layer in mlp:
                def pre(module,args,key='mlp_'+str(layer)):save(key,args[0])
                handles.append(self.layers[layer].mlp.down_proj.register_forward_pre_hook(pre))
            for layer in attention:
                module=self.layers[layer].self_attn;config=module.config
                configs.setdefault(id(config),(config,config._attn_implementation));config._attn_implementation='eager'
                def attn(module,args,out,key='attention_'+str(layer)):
                    if not isinstance(out,tuple) or len(out)<2 or out[1] is None:raise RuntimeError('Adapter did not return attention probabilities')
                    save(key,out[1])
                handles.append(module.register_forward_hook(attn))
            yield captured
        finally:
            for handle in handles:handle.remove()
            for config,implementation in configs.values():config._attn_implementation=implementation

    @torch.inference_mode()
    def _read(self,ids,hidden,mlp,attention,scores):
        tokens=torch.tensor([ids],device=self.device)
        with self._observe(hidden,mlp,attention) as captured:
            output=self.model(input_ids=tokens,attention_mask=torch.ones_like(tokens),use_cache=False,logits_to_keep=0 if scores else 1)
        return output.logits[0] if scores else None,captured

    def prepare_teacher(self,request,policy):
        if 'texts' in request:texts=request['texts']
        else:texts=[self.chat(chat,request.get('enable_thinking',policy.thinking)) for chat in request['chats']]
        sequences=[self.encode(text) for text in texts];lengths=[len(ids) for ids in sequences]
        if max(lengths)>min(policy.max_text_tokens,self.context_length) or sum(lengths)>policy.max_total_text_tokens:raise ValueError('Teacher-forced token budget exceeded')
        if request['return']['attn_layers'] and max(lengths)>policy.max_attention_tokens:raise ValueError('Attention token budget exceeded')
        if request['return']['topk']>self.vocab_size:raise ValueError('Top-k exceeds vocabulary')
        n,L=len(sequences),max(lengths);outputs=request['return'];shapes={'input_ids':((n,L),'int64'),'valid_tokens':((n,L),'bool'),'lengths':((n,),'int64')}
        if outputs['token_logprobs']:shapes['token_logprobs']=((n,L),'float32')
        if outputs['topk']:
            shapes['topk_ids']=((n,L,outputs['topk']),'int64');shapes['topk_logprobs']=((n,L,outputs['topk']),'float32')
        for layer in outputs['hidden_states']:shapes['hidden_'+str(layer)]=((n,L,self.hidden_size),'float32')
        for layer in outputs['mlp_act']:shapes['mlp_'+str(layer)]=((n,L,self.mlp_width),'float32')
        for layer in outputs['attn_layers']:shapes['attention_'+str(layer)]=((n,self.num_heads,L,L),'float32')
        payload=sum(int(np.prod(shape))*np.dtype(dtype).itemsize for shape,dtype in shapes.values())
        if payload+1_048_576>policy.max_return_bytes:raise ValueError('Padded return artifact exceeds byte limit')
        return dict(sequences=sequences,lengths=lengths,shapes=shapes,estimated_payload_bytes=payload)

    @torch.inference_mode()
    def teacher_forced(self,request,prepared,output):
        output=Path(output);outputs=request['return'];arrays={}
        for name,(shape,dtype) in prepared['shapes'].items():
            array=np.lib.format.open_memmap(output/(name+'.npy'),mode='w+',dtype=dtype,shape=shape)
            array[:]=np.nan if name.endswith('logprobs') else -1 if name in ['input_ids','topk_ids'] else 0
            arrays[name]=array
        for row,ids in enumerate(prepared['sequences']):
            length=len(ids);arrays['lengths'][row]=length;arrays['input_ids'][row,:length]=ids;arrays['valid_tokens'][row,:length]=True
            logits,captured=self._read(ids,outputs['hidden_states'],outputs['mlp_act'],outputs['attn_layers'],outputs['token_logprobs'] or outputs['topk']>0)
            for key,value in captured.items():
                if key.startswith('attention_'):arrays[key][row,:,:length,:length]=value[0].numpy()
                else:arrays[key][row,:length]=value[0].numpy()
            if logits is not None:
                # Do not expand the entire sequence*vocabulary tensor to FP32.
                for start in range(0,length,128):
                    end=min(length,start+128);chunk=logits[start:end].float();normalizer=torch.logsumexp(chunk,dim=-1)
                    if outputs['token_logprobs']:
                        count=min(end,length-1)-start
                        if count>0:
                            targets=torch.tensor(ids[start+1:start+count+1],device=self.device)
                            values=chunk[:count].gather(1,targets[:,None]).squeeze(1)-normalizer[:count]
                            arrays['token_logprobs'][row,start+1:start+count+1]=values.cpu().numpy()
                    if outputs['topk']:
                        values,indices=torch.topk(chunk,outputs['topk'],dim=-1)
                        arrays['topk_ids'][row,start:end]=indices.cpu().numpy();arrays['topk_logprobs'][row,start:end]=(values-normalizer[:,None]).cpu().numpy()
                del logits
            del captured
        files={}
        for name,array in arrays.items():
            array.flush();files[name]=dict(file=name+'.npy',shape=list(array.shape),dtype=str(array.dtype))
        arrays.clear()
        return dict(files=files,input_tokens=sum(prepared['lengths']),executed_model_forwards=len(prepared['sequences']),estimated_payload_bytes=prepared['estimated_payload_bytes'],
            alignment=dict(token_logprobs='At token t: log p(input[t]|input[:t]); first token and padding NaN',
                topk='At query position t: normalized next-token distribution; padding IDs -1/logprobs NaN',
                hidden='0 is token embedding, l>0 is raw decoder block l-1 output before final model norm',
                mlp='Input of down_proj, i.e. gated intermediate activation',attention='Eager full-attention probabilities; padded query/key positions zero',
                padding='Right padded; consult lengths/valid_tokens'))

    def prepare_generation(self,items,request,policy):
        prepared=[]
        for item in items:
            prefix=self.chat([dict(role='user',content=item['prompt'])],policy.thinking,generate=True);ids=self.encode(prefix)
            if len(ids)+request['max_tokens']>self.context_length:raise ValueError('Generation exceeds model context; no truncation')
            prepared.append(dict(item,token_ids=ids,prefix=prefix))
        return prepared

    @torch.inference_mode()
    def generate(self,item,sample,request,policy,output,stem):
        ids=item['token_ids'];seed=int(hashlib.sha256(f'{policy.seed}:{item["id"]}:{sample}'.encode()).hexdigest()[:8],16)
        tokens=torch.tensor([ids],device=self.device);devices=[self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type=='cuda' else []
        processor=PresencePenalty(len(ids),policy.presence_penalty)
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)
        started=time.perf_counter()
        calls=[0]
        def count(module,args):calls[0]+=1
        handle=self.model.register_forward_pre_hook(count)
        try:
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(seed)
                result=self.model.generate(input_ids=tokens,attention_mask=torch.ones_like(tokens),max_new_tokens=request['max_tokens'],
                    do_sample=True,temperature=policy.temperature,top_p=policy.top_p,top_k=policy.top_k,repetition_penalty=1.,
                    logits_processor=LogitsProcessorList([processor]),use_cache=True,logits_to_keep=1,
                    pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id)
        finally:handle.remove()
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)
        elapsed=time.perf_counter()-started;generated=result[0,len(ids):].cpu().tolist()
        eos=self.model.generation_config.eos_token_id;stop=set(eos if isinstance(eos,list) else [eos])
        reason='eos' if generated and generated[-1] in stop else 'length' if len(generated)==request['max_tokens'] else 'other'
        raw=self.tokenizer.decode(generated,skip_special_tokens=False);text=self.tokenizer.decode(generated,skip_special_tokens=True)
        record=dict(id=item['id'],source=item.get('source','custom'),sample=sample,seed=seed,text=text,raw_text=raw,
            generated_token_ids=generated,
            input_tokens=len(ids),output_tokens=len(generated),finish_reason=reason,hit_max_tokens=reason=='length',
            thinking_closed='</think>' in raw,prompt_sha256=hashlib.sha256(item['prefix'].encode()).hexdigest(),generation_seconds=elapsed,
            generation_model_forwards=calls[0],activation_replay_forwards=0,activation_replay_tokens=0)
        observation=request['activations'];hidden=observation['layers']
        mlp=[layer-1 for layer in hidden if layer>0] if observation['mlp'] else []
        if hidden or mlp:
            replay=generated[:policy.generation_activation_tokens]
            _,captured=self._read(ids+replay,hidden,mlp,[],False);features={}
            for key,value in captured.items():
                value=value[0];prompt_last=value[len(ids)-1].numpy();answer=value[len(ids):len(ids)+len(replay)]
                filename=stem+'_'+key+'.npz'
                np.savez(Path(output)/filename,prompt_last=prompt_last,answer_mean=answer.mean(0).numpy() if len(replay) else np.full(value.shape[-1],np.nan,dtype=np.float32))
                features[key]=dict(file=filename,answer_tokens=len(replay))
            record['activations']=features;record['activation_method']='teacher-forced replay of sampled prefix; raw residual/down_proj-input means; extra replay cost included in request wall time'
            record['activation_replay_forwards']=1;record['activation_replay_tokens']=len(ids)+len(replay)
        return record
