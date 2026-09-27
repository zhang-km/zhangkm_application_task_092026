"""Run with: python -m unittest discover -s tests -v"""
import unittest
import torch
from roof_utils.models import RoofUNet
from roof_utils.losses_metrics import MaskedBCEDiceLoss,segmentation_metrics

class PipelineTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_masked_loss_and_gradient(self):
        logits=torch.tensor([[[[0.,2.],[-1.,3.]]]],requires_grad=True)
        truth=torch.tensor([[[[1.,0.],[0.,1.]]]])
        valid=torch.tensor([[[[1.,0.],[1.,0.]]]])
        loss=MaskedBCEDiceLoss()(logits,truth,valid)
        changed=logits.detach().clone();changed[valid==0]=-100
        self.assertTrue(torch.allclose(loss,MaskedBCEDiceLoss()(changed,truth,valid)))
        loss.backward();self.assertTrue((logits.grad[valid==0]==0).all())
        self.assertTrue(torch.isfinite(logits.grad).all())
        # Independent reference using only the two valid scalar positions.
        p=torch.sigmoid(torch.tensor([0.,-1.]));y=torch.tensor([1.,0.])
        bce=torch.nn.functional.binary_cross_entropy_with_logits(torch.tensor([0.,-1.]),y)
        dice=(2*(p*y).sum()+1e-6)/(p.sum()+y.sum()+1e-6)
        self.assertAlmostEqual(loss.item(),(.5*bce+.5*(1-dice)).item(),places=6)

    def test_metrics_known_counts_and_empty(self):
        z=torch.tensor([[[[10.,10.,-10.,100.]]]])
        y=torch.tensor([[[[1.,0.,1.,0.]]]])
        v=torch.tensor([[[[1.,1.,1.,0.]]]])
        r=segmentation_metrics(z,y,v)[0]
        self.assertAlmostEqual(r['iou'],1/3);self.assertAlmostEqual(r['dice'],.5)
        self.assertEqual(r['precision'],.5);self.assertEqual(r['recall'],.5)
        empty=segmentation_metrics(torch.full_like(z,-10),torch.zeros_like(y),v)[0]
        self.assertEqual(empty['iou'],1)
        self.assertFalse(segmentation_metrics(z,y,torch.zeros_like(v))[0]['included'])
        with self.assertRaises(ValueError):MaskedBCEDiceLoss()(z,y,torch.zeros_like(v))

    def test_forward_backward_freezing_and_batchnorm(self):
        m=RoofUNet(pretrained=False);m.set_encoder_trainable(False);m.train()
        before=m.encoder.bn1.running_mean.clone()
        x=torch.randn(2,3,64,64);z=m(x)
        self.assertEqual(z.shape,(2,1,64,64))
        loss=MaskedBCEDiceLoss()(z,torch.zeros_like(z),torch.ones_like(z));loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(m.decoder['head'].weight.grad).all())
        self.assertTrue(all(p.grad is None for p in m.encoder.parameters()))
        self.assertTrue(torch.equal(before,m.encoder.bn1.running_mean))
        m.set_encoder_trainable(True);m.zero_grad(set_to_none=True)
        m(x).mean().backward();self.assertIsNotNone(m.encoder.conv1.weight.grad)
        self.assertTrue(torch.equal(before,m.encoder.bn1.running_mean))
        with torch.no_grad():self.assertEqual(m(torch.randn(1,3,65,71)).shape,(1,1,65,71))

    def test_all_invalid_sample_does_not_change_loss(self):
        z=torch.randn(1,1,4,4);y=torch.zeros_like(z);v=torch.ones_like(z)
        expected=MaskedBCEDiceLoss()(z,y,v)
        actual=MaskedBCEDiceLoss()(torch.cat([z,z]),torch.cat([y,y]),torch.cat([v,v*0]))
        self.assertTrue(torch.allclose(expected,actual))

if __name__=='__main__':unittest.main()
