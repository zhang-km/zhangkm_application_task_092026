"""Small, paired development experiments on frozen out-of-fold predictions."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
from .cross_validation import CVConfig, _saved_metrics, _check_threshold, _csv, _json


def _completed_predictions(paths, run_name, threshold):
    root=paths.reports_dir/run_name
    spec=json.loads((root/'experiment.json').read_text())
    rows=_saved_metrics(root,threshold)
    expected={sid:f['fold'] for f in spec['plan']['folds'] for sid in f['val_ids']}
    if {r['id']:r['fold'] for r in rows} != expected:
        raise ValueError(f'{run_name}: complete every frozen fold before this analysis.')
    if any(not r['included'] for r in rows):
        raise ValueError('At least one image has no valid pixels; investigate before comparing.')
    return spec,rows


def threshold_sweep(paths, run_name, thresholds=None):
    """Exploratory global sweep; never selects new model checkpoints or alters weights.

    Uses each selected fold checkpoint's saved probability maps. Scores are
    development results, not independent threshold-tuning evaluation.
    """
    thresholds=[round(.2+i*.025,3) for i in range(21)] if thresholds is None else list(thresholds)
    if not thresholds: raise ValueError('Provide at least one threshold')
    thresholds=sorted({_check_threshold(t) for t in thresholds})
    summary=[]; per_image=[]
    for threshold in thresholds:
        spec,rows=_completed_predictions(paths,run_name,threshold)
        row={'threshold':threshold,**{k:float(np.mean([r[k] for r in rows]))
                                    for k in ('iou','dice','precision','recall')}}
        for fold in sorted({r['fold'] for r in rows}):
            row[f'fold_{fold}_iou']=float(np.mean([r['iou'] for r in rows if r['fold']==fold]))
        summary.append(row)
        per_image.extend({'threshold':threshold,**r} for r in rows)
    root=paths.reports_dir/run_name/'hyperparameter_review'
    root.mkdir(parents=True,exist_ok=True)
    _csv(root/'threshold_sweep.csv',summary); _csv(root/'threshold_per_image.csv',per_image)
    peak=max(summary,key=lambda r:r['iou'])
    result={'run_name':run_name,'rows':summary,'grid_peak_threshold':peak['threshold'],
            'grid_peak_iou':peak['iou'],
            'scope':'Exploratory sweep on development labels; peak is selection-biased, not an independent test score.',
            'note':'Do not change the final-model threshold automatically; inspect stability across folds.'}
    _json(root/'threshold_sweep.json',result)
    print('Threshold   mean IoU   Dice      Precision Recall')
    for r in summary:
        print(f"{r['threshold']:.3f}       {r['iou']:.4f}     {r['dice']:.4f}    {r['precision']:.4f}    {r['recall']:.4f}")
    print(f"Exploratory grid peak: {peak['threshold']:.3f}, IoU {peak['iou']:.4f}. No threshold was changed.")
    return result


def plot_threshold_sweep(result, reference_threshold=.45):
    import matplotlib.pyplot as plt
    rows=result['rows']; x=[r['threshold'] for r in rows]
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for key in sorted(k for k in rows[0] if k.startswith('fold_')):
        axes[0].plot(x,[r[key] for r in rows],label=key.replace('_iou',''))
    axes[0].plot(x,[r['iou'] for r in rows],color='black',linewidth=3,label='All images (macro)')
    axes[0].set(xlabel='Threshold',ylabel='IoU',title='Development IoU: look for consistent plateaus')
    for key in ('precision','recall'):
        axes[1].plot(x,[r[key] for r in rows],label=key)
    axes[1].set(xlabel='Threshold',ylabel='Per-image macro score',title='Precision / recall tradeoff')
    for ax in axes:
        ax.axvline(reference_threshold,color='grey',linestyle='--',label=f'Reference {reference_threshold}')
        ax.legend(); ax.grid(alpha=.2)
    plt.tight_layout(); plt.show()


def learning_rate_configs(reference=CVConfig(crop_probability=.3,threshold=.45,max_epochs=500)):
    """One paired factor: 3x both fine-tuning rates; warm-up rate stays fixed."""
    if reference.crop_probability!=.3 or reference.monitor!='val_loss':
        raise ValueError('Use the selected crop=.3, val_loss reference recipe.')
    candidate=replace(reference,encoder_lr=3*reference.encoder_lr,decoder_lr=3*reference.decoder_lr)
    return reference,candidate


def compare_learning_rates(paths, reference_run, candidate_run, threshold=.45):
    """Require matching folds/data/source/software and all non-LR config settings."""
    threshold=_check_threshold(threshold)
    a,arows=_completed_predictions(paths,reference_run,threshold)
    b,brows=_completed_predictions(paths,candidate_run,threshold)
    ca=dict(a['config']); cb=dict(b['config'])
    ra=[ca.pop(k) for k in ('encoder_lr','decoder_lr')]
    rb=[cb.pop(k) for k in ('encoder_lr','decoder_lr')]
    if ca!=cb or any(a[k]!=b[k] for k in ('plan','source_sha256','torch')):
        raise ValueError('Runs differ beyond fine-tuning learning rates. Train the matched pair from this notebook.')
    if ra==rb: raise ValueError('The two runs have identical learning rates')
    if ca['monitor']!='val_loss': raise ValueError('This comparison requires val_loss checkpoint selection')
    bmap={r['id']:r for r in brows}
    rows=[{'id':r['id'],'fold':r['fold'],'reference_iou':r['iou'],
           'candidate_iou':bmap[r['id']]['iou'],'delta_iou':bmap[r['id']]['iou']-r['iou']} for r in arows]
    fold_rows=[{'fold':f,**{k:float(np.mean([r[k] for r in rows if r['fold']==f]))
                           for k in ('reference_iou','candidate_iou','delta_iou')}}
               for f in sorted({r['fold'] for r in rows})]
    result={'reference_run':reference_run,'candidate_run':candidate_run,'threshold':threshold,
            'reference_rates':ra,'candidate_rates':rb,'images_improved':sum(r['delta_iou']>0 for r in rows),
            'macro':{k:float(np.mean([r[k] for r in rows])) for k in ('reference_iou','candidate_iou','delta_iou')},
            'folds':fold_rows,'per_image':rows,
            'scope':'Paired development comparison; overlapping training sets and one seed, not a significance test.'}
    root=paths.reports_dir/candidate_run/'hyperparameter_review'/f'lr_comparison_t{threshold}'
    root.mkdir(parents=True,exist_ok=True)
    _csv(root/'per_image.csv',rows); _csv(root/'per_fold.csv',fold_rows); _json(root/'summary.json',result)
    print('Fold  Reference IoU  Candidate IoU  Change')
    for r in fold_rows:
        print(f"{r['fold']:4d}  {r['reference_iou']:.4f}         {r['candidate_iou']:.4f}         {r['delta_iou']:+.4f}")
    print('All-image means:',result['macro']); print('Images improved:',result['images_improved'],'/',len(rows))
    return result


def plot_learning_rate_comparison(result):
    import matplotlib.pyplot as plt
    rows=result['per_image']
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    axes[0].scatter([r['reference_iou'] for r in rows],[r['candidate_iou'] for r in rows])
    axes[0].plot([0,1],[0,1],color='grey',linestyle='--')
    axes[0].set(xlabel='Reference IoU',ylabel='Candidate IoU',title='Above diagonal = candidate improves')
    axes[1].bar([str(r['id']) for r in rows],[r['delta_iou'] for r in rows],
                color=['seagreen' if r['delta_iou']>=0 else 'firebrick' for r in rows])
    axes[1].axhline(0,color='black',linewidth=.5)
    axes[1].tick_params(axis='x',rotation=90)
    axes[1].set(xlabel='Image ID',ylabel='Candidate − reference IoU')
    plt.tight_layout(); plt.show()


def ensure_reference_run(paths, run_name, config, device='cuda'):
    """Reuse complete predictions; otherwise train/resume missing folds.

    Existing completed reports can be analyzed without model checkpoints. A
    partial run still requires matching source/configuration and its checkpoints
    for resume, as enforced by the training runner. Never erases old results.
    """
    from .validation_splits import load_validation_plan
    from .cross_validation import run_cross_validation
    plan=load_validation_plan(paths)
    if config.crop_probability!=.3 or config.monitor!='val_loss':
        raise ValueError('Reference must use crop_probability=.3 and monitor="val_loss"')
    root=paths.reports_dir/run_name
    metadata=root/'experiment.json'
    if metadata.exists():
        spec=json.loads(metadata.read_text())
        stored=dict(spec['config']); stored.setdefault('threshold',.5)
        from dataclasses import asdict
        if stored!=asdict(config) or spec['plan']!=plan:
            raise ValueError('Existing reference configuration/data differs. Match REFERENCE_CONFIG to it, or use a new CROP_RUN.')
        for fold in plan['folds']:
            folder=root/f"fold_{fold['fold']}"
            if (folder/'result.json').exists():
                missing=[sid for sid in fold['val_ids'] if not (folder/f'{sid}.npz').is_file()]
                if missing:
                    raise FileNotFoundError(
                        f"Completed fold {fold['fold']} is missing predictions {missing}. "
                        'Upload the missing .npz files, or use a new CROP_RUN to retrain; existing results are preserved.')
        if all((root/f"fold_{f['fold']}"/'result.json').is_file() for f in plan['folds']):
            _completed_predictions(paths,run_name,config.threshold)
            print(f'{run_name}: all folds and predictions present; skipping training.')
            return run_name
    elif root.exists() and any(root.iterdir()):
        raise ValueError('Reference report directory has files but no experiment.json. Restore metadata or choose a new CROP_RUN.')
    print(f'{run_name}: reference is absent or incomplete; training/resuming all five folds.',flush=True)
    run_cross_validation(paths,run_name,config,folds=list(range(plan['n_splits'])),device=device)
    _completed_predictions(paths,run_name,config.threshold)
    return run_name
