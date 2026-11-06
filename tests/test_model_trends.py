import unittest
import json
from pathlib import Path
import tempfile
import numpy as np
from analyze_model_trends import bootstrap_effect,checked_scores


class ModelTrendTests(unittest.TestCase):
    def test_seed_variability_survives_identical_items(self):
        # Every item agrees within each seed; item-only resampling would have
        # zero width after averaging. Seed uncertainty must remain visible.
        result=bootstrap_effect(np.array([[-1.]*30,[0.]*30,[1.]*30]),resamples=1000)
        self.assertEqual(result['estimate'],0.)
        self.assertLess(result['bootstrap95'][0],-.3)
        self.assertGreater(result['bootstrap95'][1],.3)

    def test_common_item_difficulty_is_paired_across_models(self):
        values=np.tile([-.5,0.,.5],(3,1))
        result=bootstrap_effect(values,values,resamples=500)
        self.assertEqual(result['bootstrap95'],[0.,0.])
        self.assertEqual(result['estimate'],0.)

    def test_single_seed_discloses_missing_seed_uncertainty(self):
        result=bootstrap_effect([[0.,1.]],resamples=100)
        self.assertIn('one seed',result['uncertainty'])
        with self.assertRaises(ValueError):bootstrap_effect([[0.,1.]],[[0.]],resamples=100)

    def test_different_models_do_not_cancel_seed_uncertainty(self):
        values=np.array([[-1.]*30,[0.]*30,[1.]*30])
        result=bootstrap_effect(values,values,resamples=1000)
        self.assertEqual(result['estimate'],0.)
        self.assertLess(result['bootstrap95'][0],-.5)
        self.assertGreater(result['bootstrap95'][1],.5)

    def metadata_fixture(self,path,batch):
        protocol=dict(model='/models/Qwen3-0.6B',batch_size=batch,smoke_limit=None)
        for name,value in [('protocol.json',protocol),('complete.json',dict(status='full_complete')),
            ('model_fingerprint.json',dict(source=protocol['model'],files={}))]:
            (path/name).write_text(json.dumps(value))

    def test_wrong_model_label_is_rejected_before_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);self.metadata_fixture(path,32)
            with self.assertRaisesRegex(ValueError,'Model index/recorded source mismatch'):
                checked_scores(path,'gsm8k','SmolLM2-360M-Instruct')

    def test_old_generation_batch_is_rejected_before_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);self.metadata_fixture(path,8)
            with self.assertRaisesRegex(ValueError,'Different evaluation batch'):
                checked_scores(path,'gsm8k','Qwen3-0.6B')
