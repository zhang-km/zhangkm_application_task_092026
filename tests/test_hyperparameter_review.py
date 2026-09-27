import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from dataclasses import asdict
import sys
import numpy as np
from roof_utils.hyperparameter_review import threshold_sweep, learning_rate_configs, compare_learning_rates
from roof_utils.cross_validation import CVConfig

class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.paths=SimpleNamespace(reports_dir=Path(self.tmp.name))
        self.reference,self.candidate=learning_rate_configs()
        self.plan={'folds':[{'fold':f,'val_ids':[str(f)]} for f in range(5)]}
    def tearDown(self): self.tmp.cleanup()
    def make_run(self,name,config,prob=.47):
        root=self.paths.reports_dir/name; root.mkdir()
        spec={'config':asdict(config),'plan':self.plan,'source_sha256':{'test':'same'},'torch':'same'}
        (root/'experiment.json').write_text(json.dumps(spec))
        (root/'summary.json').write_text('preserve original')
        for f in range(5):
            d=root/f'fold_{f}'; d.mkdir()
            (d/'result.json').write_text(json.dumps({'fold':f,'metrics':[{'id':str(f)}]}))
            np.savez(d/f'{f}.npz',probability=np.array([[prob,.1,.9]]),target=np.array([[1.,0.,0.]]),valid=np.array([[1.,1.,0.]]))
        return root
    def test_sweep_known_answer_preserves_original(self):
        root=self.make_run('reference',self.reference)
        result=threshold_sweep(self.paths,'reference',[.45,.5])
        self.assertEqual(result['rows'][0]['iou'],1)
        self.assertEqual(result['rows'][1]['iou'],0)
        self.assertEqual((root/'summary.json').read_text(),'preserve original')
    def test_paired_comparison(self):
        self.make_run('reference',self.reference,prob=.4)
        self.make_run('candidate',self.candidate,prob=.47)
        result=compare_learning_rates(self.paths,'reference','candidate')
        self.assertEqual(result['macro']['delta_iou'],1)
        self.assertEqual(result['images_improved'],5)
        self.assertEqual(self.candidate.warmup_decoder_lr,self.reference.warmup_decoder_lr)
        self.assertAlmostEqual(self.candidate.encoder_lr,3e-5)
    def test_reject_incomplete_and_mismatched(self):
        root=self.make_run('reference',self.reference)
        self.make_run('candidate',CVConfig(crop_probability=0,encoder_lr=3e-5,decoder_lr=3e-4,max_epochs=500))
        with self.assertRaises(ValueError): compare_learning_rates(self.paths,'reference','candidate')
        (root/'fold_4/result.json').unlink()
        with self.assertRaises(ValueError): threshold_sweep(self.paths,'reference',[.45])
if __name__=='__main__': unittest.main()
