"""Masked binary segmentation loss and per-image metrics; invalid pixels are ignored."""
import torch
from torch import nn
from torch.nn import functional as F


def validate_batch(logits, target, valid):
    if logits.shape != target.shape or target.shape != valid.shape or logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError('Expected matching (B,1,H,W) logits, target and validity.')
    if not torch.isfinite(logits).all() or not torch.isfinite(target).all():
        raise ValueError('Non-finite logits or targets.')
    if not ((target == 0) | (target == 1)).all() or not ((valid == 0) | (valid == 1)).all():
        raise ValueError('Target and validity must be binary.')


class MaskedBCEDiceLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, logits, target, valid):
        validate_batch(logits, target, valid)
        target = target.to(logits.dtype)
        valid = valid.to(logits.dtype)
        keep = valid.sum(dim=(1,2,3)) > 0
        if not keep.any():
            raise ValueError('Batch contains no valid pixels; skip it in the training loop.')
        logits, target, valid = logits[keep], target[keep], valid[keep]
        bce = (F.binary_cross_entropy_with_logits(logits, target, reduction='none') * valid).sum() / valid.sum()
        probability = logits.sigmoid() * valid
        truth = target * valid
        intersection = (probability * truth).sum(dim=(1,2,3))
        denominator = probability.sum(dim=(1,2,3)) + truth.sum(dim=(1,2,3))
        soft_dice = (2*intersection + self.epsilon) / (denominator + self.epsilon)
        return 0.5*bce + 0.5*(1-soft_dice).mean()


@torch.no_grad()
def segmentation_metrics(logits, target, valid, threshold=0.5):
    """Return one row per image. All-invalid images are excluded, never perfect.

    Both empty masks give IoU/Dice/precision/recall=1. Otherwise an undefined
    precision or recall is 0. Macro metrics must average only included images.
    """
    validate_batch(logits, target, valid)
    if not 0 <= threshold <= 1:
        raise ValueError('Threshold must lie in [0,1].')
    prediction = logits.sigmoid() >= threshold
    rows = []
    for p,t,v in zip(prediction, target.bool(), valid.bool()):
        if not v.any():
            rows.append({'included': False})
            continue
        p, t = p[v], t[v]
        tp = int((p&t).sum()); fp = int((p&~t).sum()); fn = int((~p&t).sum())
        union = tp+fp+fn
        if union == 0:
            iou=dice=precision=recall=1.0
        else:
            iou=tp/union; dice=2*tp/(2*tp+fp+fn)
            precision=tp/(tp+fp) if tp+fp else 0.0
            recall=tp/(tp+fn) if tp+fn else 0.0
        rows.append(dict(included=True,iou=iou,dice=dice,precision=precision,recall=recall,
                         tp=tp,fp=fp,fn=fn,valid_pixels=int(v.sum())))
    return rows
