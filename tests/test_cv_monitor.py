import sys
from pathlib import Path
import unittest
import torch
from torch.utils.data import DataLoader
from roof_utils.cross_validation import CVConfig, _improved, _should_stop, _validate
from roof_utils.losses_metrics import MaskedBCEDiceLoss
class MonitorTests(unittest.TestCase):
    def test_direction(self):
        self.assertTrue(_improved(.4,.5,'val_loss'))
        self.assertFalse(_improved(.5,.4,'val_loss'))
        self.assertTrue(_improved(.5,.4,'val_iou'))
        self.assertFalse(_improved(.4,.4,'val_loss'))
    def test_stop(self):
        c=CVConfig()
        self.assertFalse(_should_stop(29,25,c))
        self.assertTrue(_should_stop(30,25,c))
        self.assertFalse(_should_stop(30,19,c))
    def test_config(self):
        for args in ({'monitor':'bad'},{'min_epochs':101},{'min_epochs':0}):
            with self.assertRaises(ValueError): CVConfig(**args)
    def test_loss_batch_independent_and_zero_iou(self):
        rows=[]
        for i in range(5):
            valid=torch.ones(1,3,3); valid[:,:,:i%3]=0
            target=torch.zeros(1,3,3); target[:,1,2]=1
            rows.append({'image':torch.full((1,3,3),-1.-i/10), 'target':target,'valid':valid})
        criterion=MaskedBCEDiceLoss()
        a=_validate(torch.nn.Identity(),DataLoader(rows,batch_size=1),criterion,'cpu')
        b=_validate(torch.nn.Identity(),DataLoader(rows,batch_size=4),criterion,'cpu')
        self.assertEqual(a,b)
        self.assertEqual(a['val_iou'],0.)
        self.assertEqual(a['val_predicted_roof_fraction'],0.)
        self.assertGreater(a['val_loss'],0.)
if __name__=='__main__': unittest.main()
