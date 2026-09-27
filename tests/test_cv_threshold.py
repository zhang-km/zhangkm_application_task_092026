import sys, tempfile, json, unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader
from roof_utils.cross_validation import CVConfig, _validate, evaluate_saved_predictions
from roof_utils.losses_metrics import MaskedBCEDiceLoss
class ThresholdTests(unittest.TestCase):
    def test_validation_threshold_does_not_change_loss(self):
        batch=[{'image':torch.logit(torch.tensor([[[.47,.1]]])), 'target':torch.tensor([[[1.,0.]]]),'valid':torch.ones(1,1,2)}]
        a=_validate(torch.nn.Identity(),DataLoader(batch),MaskedBCEDiceLoss(),'cpu',.5)
        b=_validate(torch.nn.Identity(),DataLoader(batch),MaskedBCEDiceLoss(),'cpu',.45)
        self.assertEqual(a['val_loss'],b['val_loss'])
        self.assertEqual(a['val_iou'],0); self.assertEqual(b['val_iou'],1)
        self.assertEqual(b['val_predicted_roof_fraction'],.5)
    def test_saved_reports_preserve_original_and_handle_boundary(self):
        with tempfile.TemporaryDirectory() as t:
            p=SimpleNamespace(reports_dir=Path(t)); root=Path(t)/'run'; fold=root/'fold_4'; fold.mkdir(parents=True)
            (root/'experiment.json').write_text(json.dumps({'plan':{'eligible_ids':['272']}}))
            (fold/'result.json').write_text(json.dumps({'fold':4,'metrics':[{'id':'272'}]}))
            (root/'summary.json').write_text('original')
            np.savez(fold/'272.npz',probability=np.array([[.45,.1,.9]]),target=np.array([[1.,0.,0.]]),valid=np.array([[1.,1.,0.]]))
            a=evaluate_saved_predictions(p,'run',.5); b=evaluate_saved_predictions(p,'run',.45)
            self.assertEqual(a['macro']['iou'],0); self.assertEqual(b['macro']['iou'],1)
            self.assertTrue(b['complete']); self.assertEqual((root/'summary.json').read_text(),'original')
    def test_invalid_threshold(self):
        for v in (-.1,1.1,float('nan')):
            with self.assertRaises(ValueError): CVConfig(threshold=v)
if __name__=='__main__': unittest.main()
