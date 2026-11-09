"""Internal JSON-line worker; launch only through cpu_kernel.CPUKernel."""
import ast
import contextlib
import io
import json
import os
import resource
import sys
import traceback


def versioned_bindings(store, tokenizer, cell):
    """绑定只读模型版本和本次计算声明的数组接口。"""
    from pathlib import Path
    import numpy as np
    from safetensors import safe_open
    from safetensors.numpy import save_file
    import torch

    from .cpu_kernel import MAX_CELL_ARRAY_BYTES
    from .tensor_store import slice_region

    observations = Path('/work/observations')
    overlay = observations / 'variant.safetensors'

    def load(name, region=None):
        """从绑定版本读取原精度权重，也可直接读取小切片。"""
        if overlay.is_file():
            with safe_open(overlay, framework='pt', device='cpu') as stream:
                if name in stream.keys():
                    if region is None:
                        return stream.get_tensor(name)
                    view = stream.get_slice(name)
                    return view[slice_region(view.get_shape(), region, max_values=4096)]
        return store._native(name) if region is None else store.read_native_slice(name, region)

    def read_array(name):
        """读取宿主按名字提供的本次输入数组。"""
        with safe_open(observations / cell['run_id'] / 'inputs.safetensors', framework='np') as stream:
            return stream.get_tensor(name)

    def save_array(name, value):
        """写出声明过的数组，供宿主校验并登记。"""
        if name not in cell['outputs']:
            raise ValueError('Declare the output array name in the tool request')
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            if value.dtype == torch.bfloat16:
                value = value.float()
            value = value.numpy()
        array = np.asarray(value)
        if array.dtype.kind not in 'biuf' or not array.size or array.nbytes > MAX_CELL_ARRAY_BYTES:
            raise ValueError('An output array must be numeric, nonempty and at most 32 MiB')
        save_file({'value': np.array(array, copy=True, order='C')}, Path('/work') / cell['run_id'] / (name + '.safetensors'))

    return dict(load=load, tensor_names=store.names, tokenizer=tokenizer, read_array=read_array, save_array=save_array)


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
    cell = {}
    versioned = len(sys.argv) > 2 and sys.argv[2] == 'versioned'
    bindings = versioned_bindings(store, tok, cell) if versioned else store.kernel_bindings(tok)
    namespace=dict(__name__='__weight_programming_kernel__',**bindings)
    send(dict(status='ready',base_id=store.snapshot(False)['base_id'],tensor_names=len(store.names),
        cuda_available=False,grad_enabled=torch.is_grad_enabled(),
        scope='Isolated mounts/devices/network. No-training and no-model-simulation require transcript audit.'))
    for line in sys.stdin:
        output=CappedOutput();ok=True
        with contextlib.redirect_stdout(output),contextlib.redirect_stderr(output):
            try:
                request=json.loads(line);tree=ast.parse(request['code'],mode='exec')
                if versioned:
                    cell.clear()
                    cell.update(request['context'])
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
