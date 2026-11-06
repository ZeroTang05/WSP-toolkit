import os,json,urllib.request,base64,hashlib,time
from pathlib import Path
os.environ['HF_HUB_DISABLE_XET']='1'
from huggingface_hub import snapshot_download
ROOT=Path('/home/hdd_1/tangziye/idea4')
DATA=ROOT/'benchmarks/raw'; DATA.mkdir(parents=True,exist_ok=True)
def get(url):
 req=urllib.request.Request(url,headers={'User-Agent':'idea4-research'})
 return json.load(urllib.request.urlopen(req,timeout=60))
def github_file(repo,revision,path,target):
 info=get(f'https://api.github.com/repos/{repo}/contents/{path}?ref={revision}')
 blob=get(info['git_url'])
 raw=base64.b64decode(blob['content'])
 target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(raw)
 meta={'repo':repo,'revision':revision,'path':path,'blob_sha':info['sha'],'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
 target.with_suffix(target.suffix+'.provenance.json').write_text(json.dumps(meta,indent=2))
 print(json.dumps(meta),flush=True)
for repo,branch,paths,folder in [
 ('openai/grade-school-math','master',['grade_school_math/data/train.jsonl','grade_school_math/data/test.jsonl'],'gsm8k'),
 ('czyssrs/FinQA','main',['dataset/train.json','dataset/dev.json','dataset/test.json'],'finqa')]:
 try:
  sha=get(f'https://api.github.com/repos/{repo}/commits/{branch}')['sha']
  for path in paths:
   target=DATA/folder/Path(path).name
   if not target.exists(): github_file(repo,sha,path,target)
 except Exception as e: print('DATA_ERROR',repo,repr(e),flush=True)
for repo,name in [('HuggingFaceTB/SmolLM2-360M-Instruct','SmolLM2-360M-Instruct'),('HuggingFaceTB/SmolLM2-1.7B-Instruct','SmolLM2-1.7B-Instruct'),('Qwen/Qwen3-1.7B','Qwen3-1.7B')]:
 try:
  info=get('https://hf-mirror.com/api/models/'+repo)
  revision=info['sha']; path=ROOT.parent/'Model'/name
  if (path/'idea4_download_manifest.json').exists():continue
  snapshot_download(repo_id=repo,revision=revision,endpoint='https://hf-mirror.com',local_dir=str(path),
   allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja','*.md','LICENSE*'],max_workers=2)
  files={str(f.relative_to(path)):f.stat().st_size for f in path.rglob('*') if f.is_file() and '.cache' not in f.parts}
  (path/'idea4_download_manifest.json').write_text(json.dumps({'repo':repo,'revision':revision,'files':files},indent=2))
  print('MODEL_READY',repo,revision,sum(files.values()),flush=True)
 except Exception as e: print('MODEL_ERROR',repo,repr(e),flush=True)
