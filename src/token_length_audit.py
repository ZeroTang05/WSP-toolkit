"""Preflight common training length without reading benchmark test labels."""
import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer
from prepare_training_views import read_jsonl


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--view',required=True,type=Path)
    p.add_argument('--models-root',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--train-file',default='train.jsonl')
    p.add_argument('--max-length',default=1536,type=int)
    p.add_argument('--models',nargs='+',default=['Qwen3-0.6B','Qwen3-1.7B','SmolLM2-360M-Instruct','SmolLM2-1.7B-Instruct','Qwen3-4B-Instruct-2507'])
    a=p.parse_args()
    rows=read_jsonl(a.view/a.train_file)
    result={}
    for name in a.models:
        try:
            tok=AutoTokenizer.from_pretrained(str(a.models_root/name),local_files_only=True)
            lengths=[];bad=[]
            for row in rows:
                prefix=tok.apply_chat_template([dict(role='user',content=row['prompt'])],tokenize=True,
                    add_generation_prompt=True,enable_thinking=False)
                n=len(prefix)+len(tok.encode(row['answer'],add_special_tokens=False))+1
                lengths.append(n)
                if n>a.max_length:bad.append(row['id'])
            ordered=sorted(lengths)
            result[name]=dict(n=len(rows),max=max(lengths),median=ordered[len(lengths)//2],
                p95=ordered[int(len(lengths)*.95)],limit=a.max_length,over_limit=bad)
            print(name,result[name]['max'],len(bad),flush=True)
        except Exception as e:
            result[name]=dict(error=repr(e));print(name,repr(e),flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
