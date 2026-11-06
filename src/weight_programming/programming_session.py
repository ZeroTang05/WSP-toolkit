"""Host-side bridge: persistent CPU analysis and budgeted model observations.

Only the authorized practice tree and a writable workspace reach the kernel.
The forward service, evaluation datasets and authoritative ledgers stay outside
its mount namespace. Completed observations are copied in for tensor analysis.
"""
import hashlib
import json
from pathlib import Path
import shutil
import threading
from .cpu_kernel import CPUKernel
from .forward_service import ForwardService
from .tensor_store import WeightStore


class ProgrammingSession:
    def __init__(self,backend_factory,model_dir,practice_dir,practice,work_dir,control_dir,
                 *,python=None,policy=None,grader=None,kernel_options=None):
        self.work=Path(work_dir).resolve();self.control=Path(control_dir).resolve()
        # The kernel creates the controller directory only if fresh. These are
        # deliberately not under the programmer-writable /work tree.
        self.kernel=CPUKernel(model_dir,practice_dir,self.work,self.control,
            python=python,**(kernel_options or {}))
        self._lock=threading.RLock()
        try:
            store=WeightStore(model_dir,self.work/'edited')
            backend=backend_factory(store)
            self.service=ForwardService(backend,store,practice,self.control/'forward',policy,grader)
        except BaseException:
            self.kernel.close();raise

    def execute(self,code):
        with self._lock:return self.kernel.execute(code)

    def forward(self,request):
        with self._lock:
            result=self.service.request(request)
            source=Path(result['artifact_dir']);target=self.kernel.exports/source.name
            if target.exists():raise FileExistsError('Observation already exported into workspace')
            target.parent.mkdir(exist_ok=True)
            # Each request has passed the service return-size cap. Preserve the
            # authoritative copy and hashes outside the writable kernel mount.
            # The export tree is a separate read-only mount: kernel-created
            # symlinks cannot redirect host writes or alter raw evidence.
            shutil.copytree(source,target)
            hashes={}
            for file in source.iterdir():
                if not file.is_file():raise ValueError('Unexpected nested observation')
                digest=hashlib.sha256()
                with file.open('rb') as stream:
                    for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
                hashes[file.name]=digest.hexdigest()
            evidence=dict(attempt=result['attempt'],variant_id=result['variant_id'],files_sha256=hashes)
            (self.control/f'export_{result["attempt"]:04d}.json').write_text(json.dumps(evidence,indent=2))
            returned=dict(result);returned['artifact_dir']='/work/observations/'+source.name
            # Absolute controller paths do not belong in programmer responses.
            return returned

    def close(self):self.kernel.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
