"""Internal JSON-line worker; launch only through cpu_kernel.CPUKernel."""
import ast
import contextlib
import io
import json
import os
import resource
import sys
import traceback


class CappedOutput(io.TextIOBase):
    def __init__(self):self.parts=[];self.size=0;self.truncated=False
    def write(self,value):
        room=max(0,10000-self.size);self.parts.append(value[:room]);self.size+=min(len(value),room)
        self.truncated|=len(value)>room
        return len(value)
    def flush(self):pass
    def getvalue(self):return ''.join(self.parts).encode('ascii','replace').decode('ascii')


def main():
    # Preserve the protocol descriptor before silencing native-library stdout.
    protocol=os.fdopen(os.dup(1),'w',buffering=1)
    null=os.open('/dev/null',os.O_WRONLY);os.dup2(null,1);os.dup2(null,2);os.close(null)
    def send(value):protocol.write(json.dumps(value,ensure_ascii=True)+'\n')
    limit=int(sys.argv[1])*1024**3;resource.setrlimit(resource.RLIMIT_AS,(limit,limit))
    import torch
    from transformers import AutoTokenizer
    from .tensor_store import WeightStore
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    if torch.cuda.is_available():raise RuntimeError('GPU visible inside CPU kernel')
    store=WeightStore('/model','/work/edited')
    tok=AutoTokenizer.from_pretrained('/model',local_files_only=True)
    namespace=dict(__name__='__weight_programming_kernel__',**store.kernel_bindings(tok))
    send(dict(status='ready',base_id=store.snapshot(False)['base_id'],tensor_names=len(store.names),
        cuda_available=False,grad_enabled=torch.is_grad_enabled(),
        scope='Isolated mounts/devices/network. No-training and no-model-simulation require transcript audit.'))
    for line in sys.stdin:
        output=CappedOutput();ok=True
        with contextlib.redirect_stdout(output),contextlib.redirect_stderr(output):
            try:
                request=json.loads(line);tree=ast.parse(request['code'],mode='exec')
                torch.set_grad_enabled(False)
                if tree.body and isinstance(tree.body[-1],ast.Expr):
                    final=tree.body.pop().value
                    exec(compile(tree,'<weight-cell>','exec'),namespace)
                    value=eval(compile(ast.Expression(final),'<weight-cell>','eval'),namespace)
                    if value is not None:print(repr(value))
                else:exec(compile(tree,'<weight-cell>','exec'),namespace)
            except BaseException:
                ok=False;traceback.print_exc(limit=6)
        send(dict(status='ok' if ok else 'error',output=output.getvalue(),truncated=output.truncated))


if __name__=='__main__':main()
