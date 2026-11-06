"""Boxed answer extraction and practice-only math grading.

The four extraction functions below are adapted from MIT-licensed mathqa.py,
RikiyaT/lm_understand-logs, git blob 808a04776906b1cf4630d397cef54f03d8795477.
Copyright (c) 2026 the authors of 'LLMs can understand and write neural network
weights'. See licenses/mit-wsp-LICENSE. No benchmark labels are bundled.
"""
from collections import Counter
import functools
import importlib.metadata
import re
import threading


def answer_text(text):
    if '</think>' in text:text=text.split('</think>')[-1].strip()
    return text


def _box_at(text,index):
    key='\\boxed' if text.startswith('\\boxed',index) else '\\fbox'
    rest=text[index+len(key):];value=rest.lstrip()
    if value.startswith('{'):
        depth=0
        for position,char in enumerate(value):
            if char=='{':depth+=1
            elif char=='}':
                depth-=1
                if depth==0:return value[1:position].strip() or None
        return None
    if key=='\\boxed' and rest.startswith(' '):return rest.strip().split('$')[0].split('\n')[0].strip() or None
    return None


def last_boxed(text):
    index=text.rfind('\\boxed')
    if index<0:index=text.rfind('\\fbox')
    return None if index<0 else _box_at(text,index)


def extract(text):
    text=answer_text(text);return last_boxed(text),len(re.findall(r'\\boxed|\\fbox',text))


class MathVerifyGrader:
    def __init__(self):
        expected={'math-verify':'0.9.0','latex2sympy2_extended':'1.11.0','antlr4-python3-runtime':'4.13.2'}
        for package,version in expected.items():
            if importlib.metadata.version(package)!=version:raise ValueError('Grader requires pinned '+package+'=='+version)
        import math_verify
        self.module=math_verify;self.identity=dict(name='boxed-math-verify',versions=expected,
            extraction_source_blob='808a04776906b1cf4630d397cef54f03d8795477',
            timeout='5 seconds; use main-thread worker, no silent unbounded threaded fallback')

    @functools.lru_cache(maxsize=16384)
    def _parsed(self,value):return self.module.parse('\\boxed{'+value+'}',parsing_timeout=5)

    def __call__(self,item,text,finish_reason):
        if threading.current_thread() is not threading.main_thread():raise RuntimeError('Run symbolic grading in a main-thread worker')
        prediction,boxes=extract(text);gold=item.get('answer')
        if gold is None:return dict(correct=None,prediction=prediction,boxed=prediction is not None,boxes=boxes)
        if not isinstance(gold,str):raise ValueError('Canonical answer must be a string')
        correct=False
        if prediction is not None:
            parsed_gold=self._parsed(gold);parsed_prediction=self._parsed(prediction)
            if parsed_gold and parsed_prediction:correct=bool(self.module.verify(parsed_gold,parsed_prediction,timeout_seconds=5))
        return dict(correct=correct,prediction=prediction,boxed=prediction is not None,boxes=boxes)


def loop_diagnostics(text):
    """Declared tail metrics, not a proof that a long response is a reasoning loop."""
    words=text.split();tail=words[-200:];grams=[tuple(tail[i:i+4]) for i in range(max(0,len(tail)-3))];counts=Counter(grams)
    fraction=sum(counts[g]>=3 for g in grams)/len(grams) if grams else 0.
    trimmed=words[:-1];verbatim=False
    for period in range(1,201):
        repeats=max(3,(40+period-1)//period)
        if repeats*period>len(trimmed):break
        block=trimmed[-period:]
        if all(trimmed[len(trimmed)-(j+1)*period:len(trimmed)-j*period]==block for j in range(1,repeats)):
            verbatim=True;break
    return dict(tail_4gram_triplicate_fraction=fraction,triplicate_tail=fraction>=.5,verbatim_periodic_tail=verbatim,
        whitespace_tail_fraction=sum(c.isspace() for c in text[-512:])/len(text[-512:]) if text else 0.)
