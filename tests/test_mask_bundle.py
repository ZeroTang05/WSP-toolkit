import json
import tempfile
from pathlib import Path
import unittest
import torch
from mask_bundle import save_masks,apply_masks


class MaskBundleTests(unittest.TestCase):
    def test_roundtrip_and_wrong_base_atomicity(self):
        with tempfile.TemporaryDirectory() as directory:
            model=torch.nn.Sequential(torch.nn.Linear(5,3,bias=False))
            original=model[0].weight.detach().clone()
            mask=torch.arange(15).reshape(3,5)%3==0
            files={'config.json':'verified-test-sha'}
            out=Path(directory)/'mask'
            save_masks({'0':mask},dict(files=files),out,{})
            with self.assertRaises(ValueError):apply_masks(model,out,{'config.json':'different'})
            self.assertTrue(torch.equal(model[0].weight,original))
            apply_masks(model,out,files)
            expected=original.clone();expected[mask]=0
            self.assertTrue(torch.equal(model[0].weight,expected))

    def test_invalid_late_target_does_not_partially_apply(self):
        with tempfile.TemporaryDirectory() as directory:
            model=torch.nn.Sequential(torch.nn.Linear(5,3,bias=False),torch.nn.Linear(3,2,bias=False))
            original=model[0].weight.detach().clone()
            files={'config.json':'verified-test-sha'};out=Path(directory)/'mask'
            save_masks({'0':torch.ones(3,5,dtype=torch.bool),'1':torch.ones(2,3,dtype=torch.bool)},dict(files=files),out,{})
            path=out/'manifest.json';meta=json.loads(path.read_text());meta['targets']['1']['masked']=0
            path.write_text(json.dumps(meta))
            with self.assertRaises(ValueError):apply_masks(model,out,files)
            self.assertTrue(torch.equal(model[0].weight,original))
