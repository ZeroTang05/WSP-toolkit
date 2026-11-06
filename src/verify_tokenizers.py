"""Check serialization and actual prompt/answer tokenization equivalence."""
import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer
from weightlab import write_json


def verify(run):
    manifest=json.loads((run/'source_manifest.json').read_text())
    base=Path(manifest['source'])
    reference=AutoTokenizer.from_pretrained(base,local_files_only=True)
    data=json.loads((run/'data.json').read_text())
    rows=sum([data[k] for k in ['calibration','search','test']],[])
    texts=[r['answer'] for r in rows]+data['guard_search']+data['guard_test']
    result={}
    for name in ['wanda10','taylor08','lora40','taskhalf']:
        path=run/name
        if not path.exists():
            continue
        token=AutoTokenizer.from_pretrained(path,local_files_only=True)
        assert json.loads((base/'tokenizer.json').read_text()) == json.loads((path/'tokenizer.json').read_text())
        for row in rows:
            messages=[dict(role='user',content=row['prompt'])]
            args=dict(tokenize=True,add_generation_prompt=True,enable_thinking=False)
            assert reference.apply_chat_template(messages,**args)==token.apply_chat_template(messages,**args), name
        for text in texts:
            assert reference.encode(text,add_special_tokens=False)==token.encode(text,add_special_tokens=False),name
        assert reference.eos_token_id==token.eos_token_id and reference.pad_token_id==token.pad_token_id
        result[name]=dict(serialized_tokenizer_identical=True, chat_templates_checked=len(rows),
            plain_text_encodings_checked=len(texts), special_ids_identical=True)
    write_json(run/'tokenizer_verification.json',result)
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    verify(parser.parse_args().run)
