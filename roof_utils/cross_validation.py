"""Frozen-fold experiments, epoch-boundary resume, and out-of-fold review."""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import csv
import hashlib
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from .models import build_model
from .training import seed_everything, make_optimizer, train_batch, evaluate_batch
from .losses_metrics import MaskedBCEDiceLoss, segmentation_metrics
from .roof_augmentation import RoofFoldDataset, AugmentConfig, load_prepared
from .validation_splits import load_validation_plan

@dataclass(frozen=True)
class CVConfig:
    seed: int = 42
    max_epochs: int = 100
    warmup_epochs: int = 5
    patience: int = 20
    monitor: str = "val_loss"
    min_epochs: int = 30
    threshold: float = 0.5
    batch_size: int = 4
    crop_probability: float = 0.0
    encoder_lr: float = 1e-5
    warmup_decoder_lr: float = 1e-3
    decoder_lr: float = 1e-4
    weight_decay: float = 1e-4

    def __post_init__(self):
        if self.max_epochs < 1 or self.warmup_epochs < 0 or self.patience < 1 or self.batch_size < 1:
            raise ValueError('Invalid epoch, patience or batch-size settings')
        if self.monitor not in ('val_loss', 'val_iou'):
            raise ValueError('monitor must be val_loss or val_iou')
        if not 1 <= self.min_epochs <= self.max_epochs:
            raise ValueError('Require 1 <= min_epochs <= max_epochs')
        _check_threshold(self.threshold)
        AugmentConfig(crop_probability=self.crop_probability)


def _check_threshold(threshold):
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('threshold must be finite and between 0 and 1')
    return float(threshold)


def _improved(value, best, monitor):
    return value < best if monitor == 'val_loss' else value > best


def _should_stop(completed_epochs, stale, config):
    # Patience accrues after warm-up; min_epochs is a floor, not a reset.
    return completed_epochs >= max(config.min_epochs, config.warmup_epochs) and stale >= config.patience


@torch.no_grad()
def _validate(model, loader, criterion, device, threshold=0.5):
    logits, targets, validity, rows = [], [], [], []
    for batch in loader:
        _, metrics, output = evaluate_batch(model, batch, criterion, device)
        metrics = segmentation_metrics(output, batch['target'], batch['valid'], threshold=threshold)
        logits.append(output); targets.append(batch['target']); validity.append(batch['valid'])
        rows.extend(r for r in metrics if r['included'])
    if not rows:
        raise ValueError('Validation contains no valid images')
    # Evaluate one full-fold loss on CPU: pixel-weighted BCE plus mean per-image
    # soft Dice. This is independent of validation batch sizes/last short batch.
    output, target, valid = (torch.cat(items) for items in (logits, targets, validity))
    loss = float(criterion(output, target, valid))
    fraction = float(((output.sigmoid() >= threshold) & valid.bool()).sum() / valid.sum())
    return {'val_loss': loss, 'val_iou': float(np.mean([r['iou'] for r in rows])),
            'val_dice': float(np.mean([r['dice'] for r in rows])),
            'val_predicted_roof_fraction': fraction}


def _json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def _save(path, value):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def _csv(path, rows):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _signature(config, plan):
    source = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in Path(__file__).parent.glob('*.py')}
    return {'config': asdict(config), 'plan': plan, 'source_sha256': source,
            'torch': str(torch.__version__)}


def run_cross_validation(paths, run_name, config=CVConfig(), folds=None, device='cuda'):
    """Resume last completed epoch; finished folds are reused only for matching inputs.

    Each epoch has an independent deterministic shuffle/augmentation seed. RNG
    state is saved for stochastic model layers. Use the same device/software for
    closest reproducibility. Validation selects checkpoints, so scores are
    development CV estimates, not an independent final-test estimate.
    """
    if Path(run_name).name != run_name or run_name in ('', '.', '..'):
        raise ValueError('run_name must be one directory name')
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('No CUDA GPU. Select a Colab GPU kernel, or explicitly use device="cpu" for a smoke test.')
    plan = load_validation_plan(paths)
    folds = list(range(plan['n_splits'])) if folds is None else list(folds)
    if not folds or len(set(folds)) != len(folds) or any(f not in range(plan['n_splits']) for f in folds):
        raise ValueError('Invalid fold selection')
    report = paths.reports_dir / run_name; report.mkdir(parents=True, exist_ok=True)
    models = paths.models_dir / run_name; models.mkdir(parents=True, exist_ok=True)
    signature = _signature(config, plan)
    metadata = report / 'experiment.json'
    if metadata.exists() and json.loads(metadata.read_text()) != signature:
        raise ValueError('Run configuration/data/code/software changed. Choose a NEW run_name.')
    _json(metadata, signature)
    criterion = MaskedBCEDiceLoss()
    for fold in folds:
        out = report / f'fold_{fold}'; out.mkdir(exist_ok=True)
        ckpts = models / f'fold_{fold}'; ckpts.mkdir(exist_ok=True)
        if (out/'result.json').exists():
            print(f'Fold {fold}: already complete; reusing saved results.', flush=True)
            continue
        seed_everything(config.seed + fold)
        split = plan['folds'][fold]
        ds_args = (paths.prepared_data_dir, split['train_ids'], split['val_ids'])
        train = RoofFoldDataset(*ds_args, split='train', seed=config.seed, fold=fold,
                                config=AugmentConfig(crop_probability=config.crop_probability))
        val = RoofFoldDataset(*ds_args, split='val')
        val_loader = DataLoader(val, batch_size=config.batch_size, shuffle=False, num_workers=0)
        last = ckpts/'last.pt'; best_path = ckpts/'best.pt'
        model = build_model(paths, pretrained=not last.exists()).to(device)
        optimizer = make_optimizer(model, config.encoder_lr, config.warmup_decoder_lr, config.weight_decay)
        history=[]; best=float('inf') if config.monitor == 'val_loss' else -float('inf')
        best_epoch=0; stale=0; start_epoch=1
        if last.exists():
            state = torch.load(last, map_location=device, weights_only=True)
            model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
            history=state['history']; best=state['best']; best_epoch=state['best_epoch']; stale=state['stale']
            start_epoch=state['epoch']+1
            torch.set_rng_state(state['rng'].cpu())
            if device.type == 'cuda' and state['cuda_rng'] is not None:
                torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])
            print(f'Fold {fold}: resume at epoch {start_epoch}', flush=True)
        for epoch in range(start_epoch, config.max_epochs+1):
            if _should_stop(epoch-1, stale, config): break
            started=time.monotonic()
            model.set_encoder_trainable(epoch > config.warmup_epochs)
            optimizer.param_groups[1]['lr'] = config.warmup_decoder_lr if epoch <= config.warmup_epochs else config.decoder_lr
            train.set_epoch(epoch)
            generator=torch.Generator().manual_seed(config.seed + fold*100000 + epoch)
            loader=DataLoader(train,batch_size=config.batch_size,shuffle=True,generator=generator,num_workers=0)
            losses=[train_batch(model,b,optimizer,criterion,device) for b in loader]
            validation = _validate(model, val_loader, criterion, device, threshold=config.threshold)
            score = validation[config.monitor]
            if not all(np.isfinite(v) for v in validation.values()):
                raise FloatingPointError('Invalid validation metric')
            if _improved(score, best, config.monitor):
                best=score; best_epoch=epoch; stale=0
                _save(best_path, {'model': model.state_dict(), 'epoch': epoch,
                                  'monitor': config.monitor, 'monitor_value': best, 'threshold': config.threshold, **validation})
            elif epoch > config.warmup_epochs:
                stale+=1
            history.append({'epoch':epoch,'train_loss':float(np.mean([v for v in losses if v is not None])),
                            **validation, 'seconds':time.monotonic()-started})
            _save(last, {'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,
                         'history':history,'best':best,'best_epoch':best_epoch,'stale':stale,'monitor':config.monitor,
                         'rng':torch.get_rng_state(),
                         'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None})
            _csv(out/'history.csv',history)
            print(f"fold {fold} | epoch {epoch:3d} | val loss {validation['val_loss']:.4f} | "
                  f"val IoU {validation['val_iou']:.4f} | roof {validation['val_predicted_roof_fraction']:.2%} | "
                  f"best {config.monitor} {best:.4f} (epoch {best_epoch})", flush=True)
        _csv(out/'history.csv', history)
        selected = torch.load(best_path,map_location=device,weights_only=True)
        model.load_state_dict(selected['model'])
        rows=[]
        for batch in val_loader:
            _, metrics, logits=evaluate_batch(model,batch,criterion,device)
            metrics = segmentation_metrics(logits, batch['target'], batch['valid'], threshold=config.threshold)
            for i,(sid,m) in enumerate(zip(batch['id'],metrics)):
                rows.append({'id':sid,'fold':fold,**m})
                np.savez_compressed(out/f'{sid}.npz',probability=logits[i,0].sigmoid().numpy(),
                                    target=batch['target'][i,0].numpy(),valid=batch['valid'][i,0].numpy())
        _csv(out/'metrics.csv',rows)
        _json(out/'result.json',{'fold':fold,'best_epoch':best_epoch,'epochs_completed':len(history),
                               'monitor':config.monitor,'best_monitor_value':best,'threshold':config.threshold,
                               'selected_val_loss':selected['val_loss'],'selected_val_iou':selected['val_iou'],
                               'device':str(device),'metrics':rows})
        del model, optimizer
        if device.type=='cuda': torch.cuda.empty_cache()
    return summarize_run(paths,run_name)


def summarize_run(paths, run_name):
    root=paths.reports_dir/run_name
    experiment=json.loads((root/'experiment.json').read_text())
    results=[json.loads(p.read_text()) for p in sorted(root.glob('fold_*/result.json'))]
    rows=[r for result in results for r in result['metrics'] if r['included']]
    ids=[r['id'] for r in rows]
    if len(ids)!=len(set(ids)): raise ValueError('Duplicate out-of-fold image')
    complete=set(ids)==set(experiment['plan']['eligible_ids'])
    summary={'run_name':run_name,'threshold':experiment['config'].get('threshold',0.5),'complete':complete,'images':len(rows),
             'folds_completed':len(results),'scope':'Development CV; checkpoint selection uses these validation folds.',
             'macro':{k:float(np.mean([r[k] for r in rows])) for k in ('iou','dice','precision','recall')},
             'fold_iou':[float(np.mean([m['iou'] for m in r['metrics'] if m['included']])) for r in results]}
    _json(root/'summary.json',summary)
    if rows: _csv(root/'out_of_fold_metrics.csv',rows)
    print(json.dumps(summary,indent=2))
    return summary


def compare_runs(paths, baseline='baseline_cv_loss', crop='crop_cv_loss'):
    roots=[paths.reports_dir/n for n in (baseline,crop)]
    specs=[json.loads((r/'experiment.json').read_text()) for r in roots]
    configs=[dict(s['config']) for s in specs]
    probabilities=[c.pop('crop_probability') for c in configs]
    if configs[0]!=configs[1] or specs[0]['plan']!=specs[1]['plan'] or specs[0]['source_sha256']!=specs[1]['source_sha256'] or specs[0]['torch']!=specs[1]['torch']:
        raise ValueError('Comparison requires matching settings, folds, source and software.')
    if probabilities != [0.0,0.3]: raise ValueError('Expected baseline 0.0 and crop 0.3')
    tables=[]
    for name,root in zip((baseline,crop),roots):
        if not summarize_run(paths,name)['complete']: raise ValueError('Finish all folds before comparison')
        with (root/'out_of_fold_metrics.csv').open() as f: tables.append({r['id']:r for r in csv.DictReader(f)})
    rows=[{'id':sid,'fold':int(tables[0][sid]['fold']),
           'baseline_iou':float(tables[0][sid]['iou']),'crop_iou':float(tables[1][sid]['iou']),
           'delta_iou':float(tables[1][sid]['iou'])-float(tables[0][sid]['iou'])} for sid in sorted(tables[0])]
    _csv(roots[1]/'paired_comparison.csv',rows)
    print('Mean paired IoU change:',np.mean([r['delta_iou'] for r in rows]))
    print('Exploratory comparison on shared development folds; not an independent test or significance claim.')
    return rows


def _saved_metrics(root, threshold):
    rows=[]
    # Only completed folds: avoid partially written prediction exports.
    for result_path in sorted(root.glob('fold_*/result.json')):
        result=json.loads(result_path.read_text())
        for original in result['metrics']:
            sid=original['id']
            with np.load(result_path.parent/f'{sid}.npz') as d:
                probability=d['probability']
                if not np.isfinite(probability).all() or ((probability<0)|(probability>1)).any():
                    raise ValueError(f'Invalid saved probabilities for {sid}')
                # Reuse metric conventions without a probability->logit roundtrip
                # that could change classifications exactly at the cutoff.
                binary_logits=torch.from_numpy(np.where(probability>=threshold,1.,-1.).astype(np.float32))[None,None]
                target=torch.from_numpy(d['target'].copy())[None,None]
                valid=torch.from_numpy(d['valid'].copy())[None,None]
                metric=segmentation_metrics(binary_logits,target,valid)[0]
                rows.append({'id':sid,'fold':result['fold'],**metric})
    if not rows: raise FileNotFoundError('No completed folds with saved predictions found')
    if len({r['id'] for r in rows}) != len(rows): raise ValueError('Duplicate prediction IDs')
    return rows


def evaluate_saved_predictions(paths, run_name, threshold=0.45):
    """Re-score selected checkpoints without training or replacing original reports.

    Saves a separate threshold-specific report. This does not select a new best
    epoch or recompute historic learning curves. Partial runs are marked incomplete.
    """
    threshold=_check_threshold(threshold)
    root=paths.reports_dir/run_name
    experiment=json.loads((root/'experiment.json').read_text())
    rows=_saved_metrics(root,threshold)
    included=[r for r in rows if r['included']]
    if not included: raise ValueError('No valid pixels to evaluate')
    summary={'run_name':run_name,'threshold':threshold,
             'complete':{r['id'] for r in rows}==set(experiment['plan']['eligible_ids']),
             'images':len(included),'scope':'Exploratory threshold evaluation of saved selected checkpoints; no retraining.',
             'macro':{k:float(np.mean([r[k] for r in included])) for k in ('iou','dice','precision','recall')},
             'fold_iou':{str(f):float(np.mean([r['iou'] for r in included if r['fold']==f]))
                         for f in sorted({r['fold'] for r in included})}}
    destination=root/'threshold_evaluations'/f'threshold_{threshold}'
    destination.mkdir(parents=True,exist_ok=True)
    _csv(destination/'metrics.csv',rows); _json(destination/'summary.json',summary)
    print(json.dumps(summary,indent=2))
    return summary


def show_predictions(paths,run_name,limit=6,threshold=None):
    import matplotlib.pyplot as plt
    root=paths.reports_dir/run_name
    if threshold is None:
        threshold=json.loads((root/'experiment.json').read_text())['config'].get('threshold',0.5)
    threshold=_check_threshold(threshold)
    rows=[r for r in _saved_metrics(root,threshold) if r['included']]
    for row in sorted(rows,key=lambda r:float(r['iou']))[:limit]:
        sid=row['id']; rgb,_,_=load_prepared(paths.prepared_data_dir,sid)
        with np.load(root/f"fold_{row['fold']}"/f'{sid}.npz') as d:
            valid=d['valid'].astype(bool); truth=d['target'].astype(bool); pred=d['probability']>=threshold
            errors=np.zeros((*truth.shape,3),dtype=np.uint8)
            errors[pred & ~truth & valid]=[255,0,0]; errors[~pred & truth & valid]=[0,100,255]
            fig,axes=plt.subplots(1,4,figsize=(12,3))
            for ax,img,title in zip(axes,[rgb,np.ma.masked_where(~valid,truth),np.ma.masked_where(~valid,pred),errors],
                                     ['RGB','Label','Prediction','FP red / FN blue']):
                ax.imshow(img); ax.set_title(title); ax.axis('off')
            fig.suptitle(f"ID {sid}, fold {row['fold']}, IoU {float(row['iou']):.3f}, threshold {threshold}")
            plt.tight_layout(); plt.show()


def plot_history(paths,run_name):
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(1,3,figsize=(15,4))
    for p in sorted((paths.reports_dir/run_name).glob('fold_*/history.csv')):
        with p.open() as f: rows=list(csv.DictReader(f))
        for ax, key in zip(axes, ('val_loss','val_iou','val_predicted_roof_fraction')):
            if key in rows[0]:
                ax.plot([int(r['epoch']) for r in rows],[float(r[key]) for r in rows],label=p.parent.name)
            ax.set(xlabel='Epoch',ylabel=key)
    axes[1].legend(); plt.tight_layout(); plt.show()
