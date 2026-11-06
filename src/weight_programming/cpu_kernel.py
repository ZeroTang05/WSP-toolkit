"""Persistent CPU weight-analysis kernel with an explicit Linux mount boundary.

Motivated by MIT WSP experiment 8. Linux bubblewrap is required; unsupported
hosts fail instead of silently running with filesystem/network access. This
isolates files, network and devices, not the semantics of arbitrary Python:
the experiment still audits the no-training/no-network-simulation rules.
"""
import json
import hashlib
import os
from pathlib import Path
import selectors
import shutil
import subprocess
import time


class CPUKernel:
    def __init__(self, model_dir, practice_dir, work_dir, control_dir, *,
                 python=None, executions=300, seconds=900, memory_gib=96):
        if os.name!='posix' or not Path('/proc').exists() or not shutil.which('bwrap'):
            raise RuntimeError('Linux bubblewrap required for this isolated kernel')
        if not 1<=executions<=300 or not 0<seconds<=900 or not 1<=memory_gib<=96:
            raise ValueError('Execution limits exceed the original protocol')
        model=Path(model_dir).resolve();practice=Path(practice_dir).resolve()
        work=Path(work_dir).resolve();control=Path(control_dir).resolve()
        for readonly in [model,practice]:
            if not readonly.is_dir():raise ValueError('Existing model and authorized practice directories required')
        paths=[model,practice,work,control]
        if any(a==b or a in b.parents or b in a.parents for i,a in enumerate(paths) for b in paths[i+1:]):
            raise ValueError('Model, practice, work and controller must be disjoint trees')
        work.mkdir(parents=True,exist_ok=True);control.mkdir(parents=True,exist_ok=True)
        if any(control.iterdir()):raise FileExistsError('New controller directory required; no implicit budget reset')
        self.exports=control/'observations';self.exports.mkdir()
        observation_mount=work/'observations'
        if observation_mount.is_symlink():raise ValueError('Observation mount must not be a symlink')
        observation_mount.mkdir(exist_ok=True)
        self.control=control;self.limit=executions;self.seconds=seconds;self.count=0;self.dead=False
        runtime=Path(python or os.sys.executable).absolute().parent.parent
        if not (runtime/'pyvenv.cfg').is_file():raise ValueError('Explicit virtualenv Python required')
        package=control/'toolkit/weight_programming'
        shutil.copytree(Path(__file__).parent.resolve(),package,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        self.sources={str(f.relative_to(package)):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in package.rglob('*') if f.is_file()}
        # Fresh root; never mount the project, home, credentials, reference edits
        # or evaluation datasets. /usr contains the host interpreter libraries.
        command=[shutil.which('bwrap'),'--die-with-parent','--new-session','--unshare-all',
            '--ro-bind','/usr','/usr','--proc','/proc','--dev','/dev','--tmpfs','/tmp',
            '--dir','/etc','--dir','/opt','--dir','/toolkit',
            '--ro-bind',str(runtime),'/runtime','--ro-bind',str(package),'/toolkit/weight_programming',
            '--ro-bind',str(model),'/model','--ro-bind',str(practice),'/practice','--bind',str(work),'/work',
            '--ro-bind',str(self.exports),'/work/observations',
            '--setenv','PATH','/runtime/bin:/usr/bin:/bin','--setenv','PYTHONPATH','/toolkit',
            '--setenv','HOME','/tmp','--setenv','CUDA_VISIBLE_DEVICES','',
            '--setenv','HF_HUB_OFFLINE','1','--setenv','TRANSFORMERS_OFFLINE','1',
            '--setenv','TOKENIZERS_PARALLELISM','false','--setenv','OMP_NUM_THREADS','4']
        for lib in ['/lib','/lib64','/bin']:
            if Path(lib).exists():command+=['--ro-bind',lib,lib]
        if Path('/etc/ld.so.cache').exists():command+=['--ro-bind','/etc/ld.so.cache','/etc/ld.so.cache']
        command+=['--chdir','/work','/runtime/bin/python','-u','-m','weight_programming.kernel_worker',str(memory_gib)]
        self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=False,env={})
        self.buffer=b'';self.selector=selectors.DefaultSelector();self.selector.register(self.process.stdout,selectors.EVENT_READ)
        self._write_ledger('initializing')
        try:
            self.ready=self._read(120)
            if self.ready.get('status')!='ready':raise RuntimeError(str(self.ready))
            self._write_ledger('ready')
        except BaseException:
            self.close();self._write_ledger('initialization_failed');raise

    def _write_ledger(self,status):
        record=dict(status=status,executions=self.count,limit=self.limit,seconds_per_cell=self.seconds,
            process_pid=self.process.pid,ready=getattr(self,'ready',None),source_sha256=self.sources)
        (self.control/'kernel.json').write_text(json.dumps(record,indent=2))

    def _read(self,seconds):
        deadline=time.monotonic()+seconds
        while b'\n' not in self.buffer:
            remaining=deadline-time.monotonic()
            if remaining<=0 or not self.selector.select(max(0,remaining)):
                raise TimeoutError('Kernel execution deadline; process must be restarted explicitly')
            block=os.read(self.process.stdout.fileno(),65536)
            if not block:
                error=self.process.stderr.read(4000).decode(errors='replace')
                raise RuntimeError('Kernel exited: '+error)
            self.buffer+=block
            if len(self.buffer)>100000:raise RuntimeError('Invalid oversized kernel protocol response')
        line,self.buffer=self.buffer.split(b'\n',1)
        return json.loads(line)

    def execute(self,code):
        if self.dead or self.process.poll() is not None:raise RuntimeError('Kernel is not live; variables are lost')
        if self.count>=self.limit:raise RuntimeError('Kernel execution budget exhausted')
        if not isinstance(code,str) or not code.strip():raise ValueError('Nonempty code required')
        self.count+=1
        (self.control/f'cell_{self.count:04d}.py').write_text(code)
        self._write_ledger('executing')
        try:
            self.process.stdin.write((json.dumps(dict(code=code))+'\n').encode());self.process.stdin.flush()
            result=self._read(self.seconds)
        except BaseException:
            self.close();self._write_ledger('lost_variables');raise
        (self.control/f'cell_{self.count:04d}.json').write_text(json.dumps(result,indent=2))
        self._write_ledger('ready');return result

    def close(self):
        self.dead=True
        if self.process.poll() is None:
            self.process.kill();self.process.wait(timeout=10)
        self.selector.close()
        for stream in [self.process.stdin,self.process.stdout,self.process.stderr]:stream.close()
        self._write_ledger('closed')

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
