import os,json,urllib.request,hashlib
from pathlib import Path
os.environ['HF_ENDPOINT']='https://hf-mirror.com';os.environ['HF_HUB_DISABLE_XET']='1'
import datasets,yaml,lm_eval
root=Path('/home/hdd_1/tangziye/idea4')
for repo,name,alias,taskfile in [('allenai/ai2_arc','ARC-Challenge','idea4_arc_challenge_v1','arc/arc_challenge.yaml'),('Rowan/hellaswag',None,'idea4_hellaswag_v1','hellaswag/hellaswag.yaml')]:
 info=json.load(urllib.request.urlopen('https://hf-mirror.com/api/datasets/'+repo,timeout=30));revision=info['sha']
 target=root/'benchmarks/raw'/alias;target.mkdir(parents=True,exist_ok=True)
 data=datasets.load_dataset(repo,name,revision=revision,cache_dir=str(root/'benchmarks/cache'))
 files={}
 for split,rows in data.items():
  path=target/(split+'.jsonl');rows.to_json(str(path));files[split]=str(path)
  print(repo,split,len(rows),flush=True)
 provenance={'repository':repo,'revision':revision,'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob('*.jsonl')}}
 (target/'provenance.json').write_text(json.dumps(provenance,indent=2))
 original=Path(lm_eval.__file__).parent/'tasks'/taskfile
 config={'include':str(original),'task':alias,'tag':['idea4_public_benchmarks'],'dataset_path':'json','dataset_name':None,'dataset_kwargs':{'data_files':files}}
 (root/'benchmarks/tasks'/(alias+'.yaml')).write_text(yaml.safe_dump(config,sort_keys=False))
 print('READY',alias,revision,flush=True)
