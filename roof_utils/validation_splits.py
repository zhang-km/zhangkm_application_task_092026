"""Load and verify frozen development folds; never generate new splits during training."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import csv
from PIL import Image, ImageDraw
from .roof_augmentation import load_manifest, validate_split


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dataset_fingerprints(paths, ids):
    return {sid: {kind: file_hash(directory / f'{sid}.png') for kind, directory in (
        ('raw_image', paths.raw_data_dir/'images'),
        ('raw_label', paths.raw_data_dir/'labels'),
        ('rgb', paths.prepared_data_dir/'rgb'),
        ('roof', paths.prepared_data_dir/'roof_masks'),
        ('validity', paths.prepared_data_dir/'validity_masks'))} for sid in sorted(ids)}


def validate_plan(plan, manifest):
    """Check exclusions, coverage, disjointness and known scene-group boundaries."""
    eligible = {sid for sid, r in manifest.items() if r['status']=='usable_labelled'}
    if eligible != set(plan['eligible_ids']):
        raise ValueError('Eligible samples changed; review and version the split plan.')
    if len(plan['folds']) != plan['n_splits']:
        raise ValueError('Incorrect fold count.')
    if [f['fold'] for f in plan['folds']] != list(range(plan['n_splits'])):
        raise ValueError('Fold numbers must be contiguous and unique.')
    counts = Counter()
    for f in plan['folds']:
        train, val = f['train_ids'], f['val_ids']
        validate_split(manifest, train, val)
        if set(train) | set(val) != eligible:
            raise ValueError('Each fold must cover all eligible samples.')
        counts.update(val)
    if set(counts) != eligible or any(n!=1 for n in counts.values()):
        raise ValueError('Every eligible image must be validated exactly once.')
    for sid in eligible:
        if manifest[sid]['spatial_group'] != plan['groups'][sid]:
            raise ValueError('Scene-group assignments changed; review split plan.')
    return True


def load_validation_plan(paths, verify_data=True):
    plan = json.loads((paths.project_root/'splits/validation_folds.json').read_text())
    validate_plan(plan, load_manifest(paths.prepared_data_dir))
    if verify_data and dataset_fingerprints(paths, plan['eligible_ids']) != plan['file_sha256']:
        raise ValueError('Data files changed since folds were frozen; review and version the plan.')
    return plan


def load_fold(paths, fold=0):
    """Return train_ids, val_ids after integrity checks. Fold indices are zero-based."""
    plan = load_validation_plan(paths)
    if not isinstance(fold, int) or not 0 <= fold < plan['n_splits']:
        raise ValueError('fold must be an integer from 0 to n_splits - 1.')
    row = plan['folds'][fold]
    return list(row['train_ids']), list(row['val_ids'])


def print_split_summary(paths):
    plan = load_validation_plan(paths)
    print('Plan:', plan['version'], '| Seed:', plan['seed'])
    print('Scope:', plan['evaluation_scope'])
    print('Grouping:', plan['group_basis'])
    print('Excluded:', plan['excluded'])
    for f in plan['folds']:
        print(f"Fold {f['fold']}: train={len(f['train_ids'])}, validation={len(f['val_ids'])}, roof/valid={f['roof_fraction_valid']:.2%}; IDs={', '.join(f['val_ids'])}")
    print('PASS: no split overlap; all 24 eligible images validate exactly once; groups intact; data hashes match.')


def show_validation_folds(paths):
    from IPython.display import display
    for f in load_validation_plan(paths)['folds']:
        canvas = Image.new('RGB',(256*len(f['val_ids']),284),'white')
        draw=ImageDraw.Draw(canvas)
        for i,sid in enumerate(f['val_ids']):
            draw.text((i*256+4,5),f"Fold {f['fold']} | {sid}",fill='black')
            with Image.open(paths.prepared_data_dir/'rgb'/f'{sid}.png') as im:
                canvas.paste(im.convert('RGB'),(i*256,28))
        display(canvas)
