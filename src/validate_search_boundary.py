"""CPU comparison with the installed pinned harness's actual boundary method."""
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from transformers import AutoTokenizer
from lm_eval.models.huggingface import HFLM
from search_metrics import encode_pair


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--models-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    report=dict(harness_method_sha256=hashlib.sha256(inspect.getsource(HFLM._encode_pair).encode()).hexdigest(),
        implementation_sha256=hashlib.sha256((Path(__file__).parent/'search_metrics.py').read_bytes()).hexdigest(),models={})
    contexts=['Question: What is water?\nAnswer:','assistant\n','question  ','问题\n答案：']
    answers=[' solid','a liquid','12.5%','氢和氧','\n42']
    for name in ['Qwen3-0.6B','SmolLM2-360M-Instruct']:
        tok=AutoTokenizer.from_pretrained(str(a.models_root/name),local_files_only=True)
        stub=SimpleNamespace(backend='causal',tok_encode=lambda text:tok.encode(text,add_special_tokens=False))
        checked=0
        for context in contexts:
            for answer in answers:
                actual=encode_pair(tok,context,answer);reference=HFLM._encode_pair(stub,context,answer)
                if actual!=reference:raise AssertionError((name,context,answer))
                checked+=1
        report['models'][name]=dict(cases=checked,all_token_ids_match=True,
            tokenizer_sha256=hashlib.sha256((a.models_root/name/'tokenizer.json').read_bytes()).hexdigest())
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
