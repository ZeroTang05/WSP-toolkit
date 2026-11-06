import urllib.request,json,base64,hashlib
from pathlib import Path
import yaml
root=Path('/home/hdd_1/tangziye/idea4')
repo='EleutherAI/lm-evaluation-harness'
revision=json.load(urllib.request.urlopen(f'https://api.github.com/repos/{repo}/git/ref/tags/v0.4.9.2',timeout=15))['object']['sha']
path='lm_eval/tasks/gsm8k/gsm8k-cot-zeroshot.yaml'
info=json.load(urllib.request.urlopen(f'https://api.github.com/repos/{repo}/contents/{path}?ref={revision}',timeout=15))
raw=base64.b64decode(info['content'])
config=yaml.safe_load(raw)
config['task']='idea4_gsm8k_cot_zeroshot_v1'
config['tag']=['idea4_public_benchmarks']
config['dataset_path']='json';config['dataset_name']=None
config['dataset_kwargs']={'data_files':{'train':str(root/'benchmarks/raw/gsm8k/train.jsonl'),'test':str(root/'benchmarks/raw/gsm8k/test.jsonl')}}
folder=root/'benchmarks/tasks';folder.mkdir(parents=True,exist_ok=True)
(folder/'gsm8k_cot_zeroshot.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
(folder/'gsm8k_cot_zeroshot.provenance.json').write_text(json.dumps({'repository':repo,'revision':revision,'path':path,'blob_sha':info['sha'],'source_sha256':hashlib.sha256(raw).hexdigest(),'changes':['local official JSON data loading','unique task alias/tag'],'semantic_changes':[]},indent=2))
(folder/'gsm8k_cot_zeroshot.original.yaml').write_bytes(raw)
print(revision)
