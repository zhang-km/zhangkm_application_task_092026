"""Training primitives and the two-image pipeline sanity check (not validation)."""
from pathlib import Path
import random
import json
import time
import csv
import platform
import numpy as np
import torch
from .models import build_model
from .losses_metrics import MaskedBCEDiceLoss, segmentation_metrics
from .roof_augmentation import load_prepared, model_arrays
from .validation_splits import load_fold, load_validation_plan


def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Backend/version differences can still prevent bit-identical reproduction.
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def choose_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def make_optimizer(model, encoder_lr=1e-5, decoder_lr=1e-3, weight_decay=1e-4):
    # Include encoder parameters from the start, even while frozen. Unfreezing
    # later does not reset decoder momentum or require replacing the optimizer.
    return torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': encoder_lr, 'name':'encoder'},
        {'params': model.decoder.parameters(), 'lr': decoder_lr, 'name':'decoder'}], weight_decay=weight_decay)


def train_batch(model, batch, optimizer, criterion, device):
    model.train()
    x,y,v=(batch[k].to(device) for k in ('image','target','valid'))
    keep=v.sum(dim=(1,2,3))>0
    if not keep.any():
        return None
    optimizer.zero_grad(set_to_none=True)
    logits=model(x[keep])
    loss=criterion(logits,y[keep],v[keep])
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite training loss.')
    loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise FloatingPointError('Non-finite model gradient.')
    optimizer.step()
    return float(loss.detach().cpu())


@torch.no_grad()
def evaluate_batch(model,batch,criterion,device):
    model.eval()
    x,y,v=(batch[k].to(device) for k in ('image','target','valid'))
    logits=model(x)
    loss=float(criterion(logits,y,v).cpu()) if v.any() else None
    return loss,segmentation_metrics(logits,y,v),logits.cpu()


def run_sanity_check(paths, ids=('270','314'), max_steps=300, seed=42, target_iou=0.95,
                     eval_every=10, warmup_steps=20, device='cpu', run_name='sanity_two_images'):
    """Intentionally fit two fold-0 training images, with NO augmentation.

    This is a memorization/pipeline test, not a held-out performance estimate.
    Decoder LR stays 1e-3 for this diagnostic; the later CV fine-tuning recipe
    should lower it to 1e-4 as in method.docx. Stop when every image reaches
    target_iou at two consecutive evaluations, or at max_steps.
    """
    if max_steps<1 or eval_every<1 or warmup_steps<0 or not 0<target_iou<=1:
        raise ValueError('Invalid sanity-check settings.')
    if not ids or len(set(ids))!=len(ids):
        raise ValueError('Provide unique sample IDs.')
    train_ids,val_ids=load_fold(paths,0)
    if not set(ids)<=set(train_ids):
        raise ValueError('Sanity samples must come only from frozen fold-0 training IDs.')
    plan=load_validation_plan(paths)
    seed_everything(seed)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    device=torch.device(device) if device else choose_device()
    model=build_model(paths,pretrained=True).to(device)
    model.set_encoder_trainable(False)
    optimizer=make_optimizer(model)
    criterion=MaskedBCEDiceLoss()
    prepared=[model_arrays(*load_prepared(paths.prepared_data_dir,sid)) for sid in ids]
    batch={key:torch.from_numpy(np.stack([r[i] for r in prepared])) for i,key in enumerate(('image','target','valid'))}
    history=[];start=time.monotonic();streak=0
    initial_loss,initial_metrics,_=evaluate_batch(model,batch,criterion,device)
    def record(step,loss,metrics):
        row={'step':step,'loss':loss,'mean_iou':sum(r['iou'] for r in metrics)/len(metrics),
             'mean_dice':sum(r['dice'] for r in metrics)/len(metrics),'elapsed_seconds':time.monotonic()-start}
        for sid,r in zip(ids,metrics):row[f'iou_{sid}']=r['iou']
        history.append(row)
        print(f"step {step:3d} | loss {loss:.4f} | mean IoU {row['mean_iou']:.4f} | per-image {[round(r['iou'],4) for r in metrics]}",flush=True)
    record(0,initial_loss,initial_metrics)
    last_metrics=initial_metrics
    for step in range(1,max_steps+1):
        if step==warmup_steps+1:model.set_encoder_trainable(True)
        train_batch(model,batch,optimizer,criterion,device)
        if step%eval_every==0 or step==max_steps:
            loss,last_metrics,logits=evaluate_batch(model,batch,criterion,device)
            record(step,loss,last_metrics)
            streak=streak+1 if min(r['iou'] for r in last_metrics)>=target_iou else 0
            if streak>=2:break
    report_dir=paths.reports_dir/run_name;report_dir.mkdir(parents=True,exist_ok=True)
    model_dir=paths.models_dir/run_name;model_dir.mkdir(parents=True,exist_ok=True)
    checkpoint=model_dir/'model.pt'
    torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                'architecture':'RoofUNet-ResNet18','ids':list(ids),'seed':seed,'step':step,
                'purpose':'training-set memorization diagnostic; NOT a production model'},checkpoint)
    # Prove checkpoint can be reloaded without downloading or initializing weights again.
    from .models import RoofUNet
    restored=RoofUNet(pretrained=False).to(device)
    restored.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['state_dict'])
    _,restored_metrics,restored_logits=evaluate_batch(restored,batch,criterion,device)
    if not torch.allclose(logits,restored_logits,atol=1e-5,rtol=1e-4):
        raise AssertionError('Checkpoint reload changed predictions.')
    np.savez_compressed(report_dir/'predictions.npz',logits=restored_logits.numpy(),target=batch['target'].numpy(),valid=batch['valid'].numpy())
    import torchvision
    result={'purpose':'two-image memorization sanity check, NOT held-out validation',
        'ids':list(ids),'fold':0,'fold_validation_ids_untouched':val_ids,'seed':seed,
        'architecture':'U-Net with pretrained ResNet18 encoder and GroupNorm decoder',
        'pretrained_weights':model.weights_name,'parameter_count':sum(p.numel() for p in model.parameters()),
        'device':str(device),'torch':torch.__version__,'torchvision':torchvision.__version__,
        'numpy':np.__version__,'python':platform.python_version(),
        'augmentation':False,'decoder_lr':1e-3,'encoder_lr':1e-5,'weight_decay':1e-4,
        'warmup_steps':warmup_steps,'max_steps':max_steps,'steps_completed':step,'target_iou_each':target_iou,
        'required_consecutive_evaluations':2,'passed':streak>=2,'checkpoint_reload_passed':True,
        'initial_metrics':initial_metrics,'final_metrics':restored_metrics,'initial_loss':initial_loss,'final_loss':loss,
        'elapsed_seconds':time.monotonic()-start,'history':history,
        'split_version':plan['version'],'data_sha256':{sid:plan['file_sha256'][sid] for sid in ids}}
    (report_dir/'results.json').write_text(json.dumps(result,indent=2))
    with (report_dir/'history.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(history[0]));w.writeheader();w.writerows(history)
    return result
