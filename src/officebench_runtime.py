"""Rootless-Docker OfficeBench execution port; not upstream score equivalence.

The source must be the verified acquisition produced by prepare_officebench.py.
No final evaluation fields are exposed in the actor's task configuration.
"""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
import uuid


class OfficeBenchRuntime:
    def __init__(self,acquisition,task_id,output,*,docker_host,image,withhold_expected=False):
        self.acquisition=Path(acquisition).resolve();self.source=self.acquisition/'source'
        manifest=json.loads((self.acquisition/'manifest.json').read_text())
        if manifest['commit']!='b978b808667c32b52ce19a67ce1def1de9ae02b7':raise ValueError('Pinned OfficeBench source required')
        if not re.fullmatch(r'[123]-\d+/\d+',task_id):raise ValueError('Canonical group/subtask required')
        self.group,self.subtask=task_id.split('/');self.task_id=task_id;self.withhold=withhold_expected
        config=self.source/'tasks'/self.group/'subtasks'/(self.subtask+'.json')
        self.config=json.loads(config.read_text());self.public={k:v for k,v in self.config.items() if k!='evaluation'}
        self.output=Path(output).resolve()
        if self.output.exists():raise FileExistsError('Fresh independent task output required')
        self.output.mkdir(parents=True);self.state=dict(current_app=None,workdir='/');self.done=False
        code=self.output/'code';code.mkdir()
        self.worker_path=code/'officebench_container_worker.py'
        shutil.copy2(Path(__file__).with_name('officebench_container_worker.py'),self.worker_path)
        shutil.copy2(__file__,code/'officebench_runtime.py')
        self.cli=['docker','--host',docker_host];self.containers=[]
        self.image=self._docker(['image','inspect',image,'--format','{{.Id}}']).strip()
        if not re.fullmatch(r'sha256:[a-f0-9]{64}',self.image):raise ValueError('Immutable image ID required')
        self.workspace=self.output/'input/testbed'
        shutil.copytree(self.source/'tasks'/self.group/'testbed',self.workspace)
        for name in ['data','emails','calendar']:(self.workspace/name).mkdir(exist_ok=True)
        withheld=[]
        if withhold_expected:
            for check in self.config['evaluation']:
                expected=check['args'].get('expected_file')
                if not expected:continue
                path=(self.workspace/expected).resolve()
                if self.workspace in path.parents and path.is_file():path.unlink();withheld.append(expected)
        self.record=dict(task_id=task_id,source_commit=manifest['commit'],source_config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
            image_id=self.image,protocol='idea4_network_free_officebench_execution_port_v1',
            llm_app='disabled; no external model delegation',withhold_expected=withhold_expected,withheld_files=withheld,
            worker_sha256=hashlib.sha256(self.worker_path.read_bytes()).hexdigest(),
            runtime_sha256=hashlib.sha256((code/'officebench_runtime.py').read_bytes()).hexdigest(),
            limitations='Ported transport and pinned environment, not established equivalence to published leaderboard. Fresh containers per task; official scoring helpers unchanged.')
        (self.output/'protocol.json').write_text(json.dumps(self.record,indent=2))
        (self.output/'performer.json').write_text(json.dumps(self.public,indent=2))
        try:
            self.actor=self._container('actor')
            self._docker(['cp',str(self.workspace),self.actor+':/testbed'])
            self._docker(['exec',self.actor,'chown','-R','0:0','/testbed'])
            self.description=self._request(self.actor,dict(mode='describe'))
        except BaseException:self.close();raise

    def _docker(self,args,stdin=None):
        result=subprocess.run(self.cli+args,input=stdin,capture_output=True,text=True,timeout=120)
        if result.returncode:raise RuntimeError(result.stderr[-4000:])
        return result.stdout

    def _container(self,role):
        name='idea4-office-'+role+'-'+uuid.uuid4().hex[:12]
        self._docker(['create','--name',name,'--network','none',self.image,'sleep','infinity'])
        self.containers.append(name);self._docker(['start',name])
        self._docker(['cp',str(self.source/'apps'),name+':/apps'])
        if role=='judge':self._docker(['cp',str(self.source/'utils'),name+':/utils'])
        self._docker(['cp',str(self.worker_path),name+':/idea4_worker.py'])
        return name

    def _request(self,container,request):
        output=self._docker(['exec','-i',container,'python3','/idea4_worker.py'],stdin=json.dumps(request))
        return json.loads(output.rsplit('IDEA4_RESULT:',1)[-1])

    def step(self,action):
        if self.done:raise RuntimeError('Task already terminated')
        if not isinstance(action,dict):raise ValueError('JSON action object required')
        start=time.perf_counter();result=self._request(self.actor,dict(mode='step',action=action,state=self.state))
        self.state=result.get('state',self.state);self.done=result.get('done',False)
        with (self.output/'steps.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(action=action,result=result,seconds=time.perf_counter()-start))+'\n')
        return result

    def record_model_call(self,record):
        with (self.output/'model_calls.jsonl').open('a') as stream:
            stream.write(json.dumps(record,ensure_ascii=False)+'\n')

    def evaluate(self):
        destination=self.output/'output';destination.mkdir()
        self._docker(['cp',self.actor+':/testbed',str(destination)])
        judge=self._container('judge')
        base='/scoring/tasks/'+self.group
        testbed=base+'/outputs/'+self.subtask+'/condition/testbed'
        self._docker(['exec',judge,'mkdir','-p',base+'/outputs/'+self.subtask+'/condition'])
        self._docker(['cp',str(destination/'testbed'),judge+':'+testbed])
        # Some upstream helpers internally use /testbed despite accepting a path.
        self._docker(['exec',judge,'ln','-s',testbed,'/testbed'])
        reference=self.source/'tasks'/self.group/'reference'
        if reference.exists():self._docker(['cp',str(reference),judge+':'+base+'/reference'])
        if self.withhold:
            for expected in self.record['withheld_files']:
                original=self.source/'tasks'/self.group/'testbed'/expected
                self._docker(['cp',str(original),judge+':'+testbed+'/'+expected])
        result=self._request(judge,dict(mode='score',checks=self.config['evaluation'],testbed=testbed))
        result.update(task_id=self.task_id,protocol=self.record['protocol'],scope='Official helpers on ported environment; no published-score equivalence claim')
        (self.output/'evaluation.json').write_text(json.dumps(result,indent=2));return result

    def close(self):
        for container in getattr(self,'containers',[]):
            subprocess.run(self.cli+['rm','-f',container],capture_output=True,timeout=30)
        self.containers=[]
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
