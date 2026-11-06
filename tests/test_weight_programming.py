import json
from pathlib import Path
import tempfile
import unittest
import torch
from safetensors.torch import save_file,load_file
from weight_programming import WeightStore,EditPolicy,contrast_direction,project_output_direction,compare_edits
from weight_programming.tensor_store import sha


class WeightProgrammingTests(unittest.TestCase):
    def test_projection_formula_support_and_sign_invariance(self):
        torch.manual_seed(21);w=torch.randn(6,9);u=torch.randn(6);columns=[2,7];unit=u.double()/u.double().norm()
        edited=project_output_direction(w,u,columns,1.)
        expected=w.double().clone();expected[:,columns]-=unit[:,None]*(unit@w.double()[:,columns])[None,:]
        torch.testing.assert_close(edited,expected.float());torch.testing.assert_close(edited,project_output_direction(w,-u,columns,1.))
        self.assertTrue(torch.equal(w[:,[0,1,3,4,5,6,8]],edited[:,[0,1,3,4,5,6,8]]))
        self.assertLess(float((unit@edited.double()[:,columns]).abs().max()),1e-6)
        info=compare_edits(w,edited,edited);self.assertAlmostEqual(info['signed_weight_cosine'],1.);self.assertEqual(info['changed_column_jaccard'],1.)

    def test_persistent_edits_and_native_precision_do_not_modify_source(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);base=folder/'base';base.mkdir();name='model.layers.0.mlp.down_proj.weight'
            w=torch.arange(24).reshape(4,6).float().to(torch.bfloat16)
            save_file({name:w,'model.norm.weight':torch.ones(4)},str(base/'model.safetensors'));before=sha(base/'model.safetensors')
            store=WeightStore(base,folder/'edit');proposal=w.float().clone();proposal[0,0]=1e-8
            record=store.save_edit(name,proposal);self.assertEqual(record['changed_float32'],1)
            restored=WeightStore(base,folder/'edit');torch.testing.assert_close(restored.load(name),proposal)
            manifest=restored.materialize_edits(folder/'out');tensor=load_file(str(folder/'out'/manifest[name]['file']))[name]
            torch.testing.assert_close(tensor,proposal.to(w.dtype));self.assertEqual(sha(base/'model.safetensors'),before)
            self.assertTrue({'NAMES','load','load_original','save_edit','drop_edit','edited'}<=set(restored.kernel_bindings()))
            restored.drop_edit(name);torch.testing.assert_close(restored.load(name),w.float());self.assertFalse(restored.edited())

    def test_edit_scope_and_source_identity_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);base=folder/'base';base.mkdir();name='model.layers.0.mlp.down_proj.weight';w=torch.arange(24).reshape(4,6).float()
            save_file({name:w,'model.norm.weight':torch.ones(4)},str(base/'model.safetensors'));store=WeightStore(base,folder/'edit')
            for bad in [w*2,torch.ones(5,6),torch.full_like(w,float('nan')),w.clone().requires_grad_()]:
                with self.assertRaises(ValueError):store.save_edit(name,bad)
            with self.assertRaises(ValueError):store.save_edit('model.norm.weight',torch.ones(4)*2)
            with self.assertRaises(ValueError):store.materialize_edits(base/'nested')
            changed=w.clone();changed[1,2]+=1.;store.save_edit(name,changed)
            with self.assertRaises(ValueError):WeightStore(base,folder/'edit',EditPolicy(forbid_global_rescaling=False))
            save_file({name:w+1,'model.norm.weight':torch.ones(4)},str(base/'model.safetensors'))
            with self.assertRaises(ValueError):store.load_original(name)

    def test_contrast_is_matched_and_nonzero(self):
        source=torch.zeros(3,4);target=source+torch.tensor([1.,2.,0.,0.]);u=contrast_direction(source,target)
        torch.testing.assert_close(u,torch.tensor([1.,2.,0.,0.],dtype=torch.float64)/(5**.5))
        with self.assertRaises(ValueError):contrast_direction(source,source)


if __name__=='__main__':unittest.main()
