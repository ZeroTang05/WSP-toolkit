"""Create one training-derived worst-context memory smoke, never a formal train set."""
import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer
from prepare_training_views import read_jsonl,write_rows,sha

p=argparse.ArgumentParser();p.add_argument('--view',type=Path,required=True)
p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
rows=read_jsonl(a.view/'retriever_train.jsonl')
tok=AutoTokenizer.from_pretrained(str(a.model),local_files_only=True)
def length(row):
    return len(tok.apply_chat_template([dict(role='user',content=row['prompt'])],tokenize=True,
        add_generation_prompt=True,enable_thinking=False))+len(tok.encode(row['answer'],add_special_tokens=False))+1
row=max(rows,key=length);a.output.mkdir(parents=True)
write_rows(a.output/'train.jsonl',[row])
(a.output/'manifest.json').write_text(json.dumps(dict(scope='longest-context memory smoke only; not formal expert training',
    source_manifest_sha256=sha(a.view/'manifest.json'),files={'train.jsonl':sha(a.output/'train.jsonl')},
    selected_id=row['id'],tokens=length(row)),indent=2))
