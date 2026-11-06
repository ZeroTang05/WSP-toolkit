"""Continuous search proxies with pinned-harness-compatible ARC boundaries.

The normalized-choice CE is a surrogate, not official acc_norm itself or a
calibrated sequence likelihood. Both raw and normalized diagnostics are saved.
"""
import math
import numpy as np
import torch
from torch.nn import functional as F


def encode_pair(tokenizer,context,continuation):
    n_spaces=len(context)-len(context.rstrip())
    if n_spaces:
        continuation=context[-n_spaces:]+continuation;context=context[:-n_spaces]
    prefix=tokenizer.encode(context,add_special_tokens=False)
    whole=tokenizer.encode(context+continuation,add_special_tokens=False)
    return prefix,whole[len(prefix):]


def choice_metrics(scores,texts,gold):
    scores=np.asarray(scores,dtype=float)
    lengths=np.array([len(text) for text in texts],dtype=float)
    if np.any(lengths<=0):raise ValueError('Empty answer choice')
    def logsoftmax(x):
        shifted=x-x.max();return shifted-np.log(np.exp(shifted).sum())
    raw=logsoftmax(scores);norm=logsoftmax(scores/lengths)
    return dict(raw_ce=float(-raw[gold]),norm_ce=float(-norm[gold]),
        raw_logp=raw.tolist(),norm_logp=norm.tolist(),acc_norm=int(np.argmax(scores/lengths)==gold))


@torch.inference_mode()
def losses(lab,domain,guards):
    total=0.;nbytes=0
    for row in domain:
        prefix=lab.tokenizer.encode(lab.prefix(row),add_special_tokens=False)
        target=lab.tokenizer.encode(row['answer'],add_special_tokens=False)+[lab.tokenizer.eos_token_id]
        logits=lab.model(torch.tensor([prefix+target],device='cuda'),use_cache=False).logits[0,len(prefix)-1:-1].float()
        total+=F.cross_entropy(logits,torch.tensor(target,device='cuda'),reduction='sum').item()
        nbytes+=len(row['answer'].encode('utf-8'))
    result={key:[] for key in ['raw_ce','norm_ce','raw_logp','norm_logp','acc_norm']}
    for row in guards:
        texts=row['choices']['text'];scores=[]
        context=lab.prefix(dict(prompt='Question: '+row['question']+'\nAnswer:'))
        for text in texts:
            prefix,target=encode_pair(lab.tokenizer,context,text)
            if not target:raise ValueError('Empty continuation')
            logits=lab.model(torch.tensor([prefix+target],device='cuda'),use_cache=False).logits[0,len(prefix)-1:-1].float()
            scores.append(-F.cross_entropy(logits,torch.tensor(target,device='cuda'),reduction='sum').item())
        metrics=choice_metrics(scores,texts,row['choices']['label'].index(row['answerKey']))
        for key,value in metrics.items():result[key].append(value)
    return total/(math.log(2)*nbytes),result
