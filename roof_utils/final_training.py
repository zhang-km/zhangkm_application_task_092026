"""Fixed-budget refit on all usable labels and inference on unlabelled images."""
from pathlib import Path
from dataclasses import asdict
import hashlib
import json
import time
import zipfile
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from .cross_validation import CVConfig, _json, _save, _csv
from .roof_augmentation import (AugmentConfig, load_manifest, load_prepared,
                                augment, sample_rng, model_arrays)
from .validation_splits import load_validation_plan
from .training import seed_everything, make_optimizer, train_batch
from .models import build_model
from .losses_metrics import MaskedBCEDiceLoss


class FinalDataset(Dataset):
    """All eligible labels, deliberately no validation split or duplicate IDs."""
    def __init__(self, paths, ids, config):
        self.ids=list(ids); self.paths=paths; self.config=config; self.epoch=0
        manifest=load_manifest(paths.prepared_data_dir)
        if not self.ids or len(set(self.ids))!=len(self.ids):
            raise ValueError('Expected unique, nonempty labelled IDs')
        if any(manifest[s]['status']!='usable_labelled' for s in self.ids):
            raise ValueError('Final training accepts only usable_labelled images')
        self.augmentation=AugmentConfig(crop_probability=config.crop_probability)

    def __len__(self): return len(self.ids)

    def __getitem__(self, index):
        sid=self.ids[index]
        rng=sample_rng(self.config.seed, -1, self.epoch, sid)
        rgb,roof,valid,_=augment(*load_prepared(self.paths.prepared_data_dir,sid),
                                rng=rng,training=True,config=self.augmentation)
        x,y,v=model_arrays(rgb,roof,valid)
        return {'image':x,'target':y,'valid':v,'id':sid}


def _device(value):
    device=torch.device(value)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('Select a GPU kernel, or explicitly use device="cpu" for testing.')
    return device


def train_final_model(paths, run_name, config, device='cuda'):
    """Fit a fresh ImageNet-initialized model for exactly max_epochs.

    monitor/min_epochs/patience are CV-only and intentionally inactive: no
    validation labels remain. Saves last.pt each epoch and final.pt at completion.
    A matching interrupted run resumes; configuration/source/data changes require
    a new run name. This run produces no independent accuracy estimate.
    """
    if Path(run_name).name!=run_name or run_name in ('','.','..'):
        raise ValueError('Use one directory name for run_name')
    device=_device(device)
    plan=load_validation_plan(paths)
    ids=sorted(plan['eligible_ids'])
    report=paths.reports_dir/run_name; model_dir=paths.models_dir/run_name
    report.mkdir(parents=True,exist_ok=True); model_dir.mkdir(parents=True,exist_ok=True)
    signature={'purpose':'all-label final refit; fixed epoch budget; no validation',
               'config':asdict(config),'inactive_cv_settings':['monitor','min_epochs','patience'],
               'train_ids':ids,'plan':plan,'torch':str(torch.__version__),
               'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in Path(__file__).parent.glob('*.py')}}
    metadata=report/'experiment.json'
    if metadata.exists() and json.loads(metadata.read_text())!=signature:
        raise ValueError('Configuration/data/code/software changed. Use a NEW run_name.')
    _json(metadata,signature)
    print(f'Final refit: {len(ids)} labels; {config.max_epochs} epochs; no validation early stopping.',flush=True)
    dataset=FinalDataset(paths,ids,config)
    seed_everything(config.seed)
    last=model_dir/'last.pt'
    model=build_model(paths,pretrained=not last.exists()).to(device)
    optimizer=make_optimizer(model,config.encoder_lr,config.warmup_decoder_lr,config.weight_decay)
    criterion=MaskedBCEDiceLoss(); history=[]; start_epoch=1
    if last.exists():
        state=torch.load(last,map_location=device,weights_only=True)
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        history=state['history']; start_epoch=state['epoch']+1
        torch.set_rng_state(state['rng'].cpu())
        if device.type=='cuda' and state['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])
        print(f'Resuming from completed epoch {start_epoch-1}.',flush=True)
    for epoch in range(start_epoch,config.max_epochs+1):
        started=time.monotonic(); dataset.epoch=epoch
        model.set_encoder_trainable(epoch>config.warmup_epochs)
        optimizer.param_groups[1]['lr']=config.warmup_decoder_lr if epoch<=config.warmup_epochs else config.decoder_lr
        generator=torch.Generator().manual_seed(config.seed+epoch)
        loader=DataLoader(dataset,batch_size=config.batch_size,shuffle=True,
                          generator=generator,num_workers=0)
        losses=[]
        for batch in loader:
            value=train_batch(model,batch,optimizer,criterion,device)
            if value is not None: losses.append((value,len(batch['id'])))
        if not losses: raise ValueError('No valid training batches')
        mean_loss=sum(v*n for v,n in losses)/sum(n for _,n in losses)
        history.append({'epoch':epoch,'train_loss':mean_loss,'seconds':time.monotonic()-started})
        _save(last,{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,
                    'history':history,'rng':torch.get_rng_state(),
                    'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None})
        _csv(report/'history.csv',history)
        print(f'epoch {epoch:3d}/{config.max_epochs} | augmented training loss {mean_loss:.4f}',flush=True)
    _csv(report/'history.csv',history)
    final=model_dir/'final.pt'
    _save(final,{'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                 'architecture':'RoofUNet-ResNet18','epoch':history[-1]['epoch'],
                 'threshold':config.threshold,'train_ids':ids,'config':asdict(config),
                 'purpose':'final fixed-budget refit; not a validation-selected checkpoint'})
    result={'run_name':run_name,'epochs_completed':history[-1]['epoch'],'train_ids':ids,
            'threshold':config.threshold,'device':str(device),'checkpoint':str(final),
            'scope':'No held-out evaluation; training loss is not a generalization score.'}
    _json(report/'result.json',result)
    return result


@torch.no_grad()
def predict_unlabelled(paths, run_name, device='cuda'):
    """Export full-size probability maps, masks and validity; never augment inputs."""
    device=_device(device)
    checkpoint=paths.models_dir/run_name/'final.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    threshold=state['threshold']
    model=build_model(paths,pretrained=False).to(device)
    model.load_state_dict(state['model']); model.eval()
    ids=sorted(sid for sid,r in load_manifest(paths.prepared_data_dir).items() if r['status']=='unlabelled')
    destination=paths.reports_dir/run_name/'unlabelled_predictions'
    destination.mkdir(parents=True,exist_ok=True)
    rows=[]
    for sid in ids:
        with Image.open(paths.prepared_data_dir/'rgb'/f'{sid}.png') as im:
            rgb=np.array(im.convert('RGB'))
        with Image.open(paths.prepared_data_dir/'validity_masks'/f'{sid}.png') as im:
            mask=np.array(im.convert('L'))
        if not np.isin(mask,[0,255]).all(): raise ValueError('Validity mask must be binary')
        valid=(mask==255).astype(np.uint8)
        x,_,_=model_arrays(rgb,np.zeros(valid.shape,dtype=np.uint8),valid)
        probability=model(torch.from_numpy(x)[None].to(device)).sigmoid()[0,0].cpu().numpy()
        probability[valid==0]=0
        prediction=(probability>=threshold)&valid.astype(bool)
        np.savez_compressed(destination/f'{sid}.npz',probability=probability,valid=valid,threshold=threshold)
        Image.fromarray(prediction.astype(np.uint8)*255).save(destination/f'{sid}_roof.png')
        Image.fromarray(valid*255).save(destination/f'{sid}_valid.png')
        rows.append({'id':sid,'threshold':threshold,'predicted_roof_fraction':
                     float(prediction.sum()/valid.sum()) if valid.any() else None})
    _json(destination/'predictions.json',{'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                                         'threshold':threshold,'predictions':rows,
                                         'scope':'Unlabelled predictions; no accuracy metrics available.'})
    return rows


def show_final_results(paths,run_name):
    import csv
    import matplotlib.pyplot as plt
    root=paths.reports_dir/run_name
    with (root/'history.csv').open() as f: history=list(csv.DictReader(f))
    fig,ax=plt.subplots(figsize=(7,3))
    ax.plot([int(r['epoch']) for r in history],[float(r['train_loss']) for r in history])
    ax.set(xlabel='Epoch',ylabel='Augmented training loss',title='Final refit (no validation curve)')
    plt.show()
    for p in sorted((root/'unlabelled_predictions').glob('*.npz')):
        with np.load(p) as d:
            probability=d['probability']; valid=d['valid'].astype(bool); threshold=float(d['threshold'])
        with Image.open(paths.prepared_data_dir/'rgb'/f'{p.stem}.png') as im: rgb=np.array(im.convert('RGB'))
        pred=(probability>=threshold)&valid
        overlay=rgb.copy(); overlay[pred]=(0.6*rgb[pred]+0.4*np.array([0,255,0])).astype(np.uint8)
        fig,axes=plt.subplots(1,3,figsize=(10,3))
        axes[0].imshow(rgb); axes[0].set_title('RGB')
        axes[1].imshow(np.ma.masked_where(~valid,probability),vmin=0,vmax=1,cmap='viridis'); axes[1].set_title('Roof probability (0–1)')
        axes[2].imshow(overlay); axes[2].set_title(f'Roof overlay; threshold {threshold}')
        for ax in axes: ax.axis('off')
        fig.suptitle(f'Unlabelled image {p.stem}'); plt.tight_layout(); plt.show()


def export_final_run(paths,run_name):
    """Bundle inference and resume checkpoints, reports, code and path template."""
    archive=paths.reports_dir/f'{run_name}_export.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_STORED) as z:
        for label,root in [('models',paths.models_dir/run_name),('reports',paths.reports_dir/run_name)]:
            for p in root.rglob('*'):
                if p.is_file() and p.suffix!='.tmp': z.write(p,Path(label)/run_name/p.relative_to(root))
        for p in Path(__file__).parent.glob('*.py'): z.write(p,Path('roof_utils')/p.name)
        z.write(paths.project_root/'pyproject.toml','pyproject.toml')
        z.writestr('.env.example','RAW_DATA_DIR=data\nPREPARED_DATA_DIR=data_preparation_outputs\nMODELS_DIR=models\nREPORTS_DIR=reports\n')
    print(f'Download this archive before the runtime is deleted: {archive}')
    return archive
