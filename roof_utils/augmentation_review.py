"""Visual demonstrations and checks used by the augmentation notebook."""
from collections import Counter
import numpy as np
from PIL import Image
from .roof_augmentation import (load_manifest, load_prepared, augment, geometric,
    sample_rng, AugmentConfig, RoofFoldDataset, validate_split)
from .visualization import show_row

def display(image):
    from IPython.display import display as ipython_display
    ipython_display(image)

def inspect_prepared(prepared_dir):
    manifest = load_manifest(prepared_dir)
    usable_ids = sorted((sid for sid, row in manifest.items() if row['status'] == 'usable_labelled'))
    print('Manifest statuses:', dict(Counter((row['status'] for row in manifest.values()))))
    print('Usable labelled IDs:', ', '.join(usable_ids))
    assert '278' not in usable_ids
    for sid in usable_ids:
        load_prepared(prepared_dir, sid)
    print(f'Validated exported RGB and binary masks for all {len(usable_ids)} eligible pairs.')
    print('Spatial groups populated:', sum((bool(manifest[sid].get('spatial_group', '').strip()) for sid in usable_ids)))
    print('Configuration:', AugmentConfig())

def show_geometry(prepared_dir):
    manifest = load_manifest(prepared_dir)
    usable_ids = sorted(sid for sid, row in manifest.items() if row['status'] == 'usable_labelled')
    raw = load_prepared(prepared_dir, '303')
    examples = [('Original', raw)]
    for name, args in [('Horizontal flip', dict(horizontal=True)), ('Vertical flip', dict(vertical=True)), ('90-degree rotation', dict(k=1))]:
        examples.append((name, tuple((geometric(a, **args) for a in raw))))
    sheet = Image.new('RGB', (1024, 302 * len(examples)), 'white')
    for i, (name, arrays) in enumerate(examples):
        sheet.paste(show_row(*arrays, f'303 | {name}'), (0, i * 302))
    display(sheet)

def show_random_examples(prepared_dir):
    manifest = load_manifest(prepared_dir)
    usable_ids = sorted(sid for sid, row in manifest.items() if row['status'] == 'usable_labelled')
    for sid in ['270', '532']:
        raw = load_prepared(prepared_dir, sid)
        sheet = Image.new('RGB', (1024, 302 * 4), 'white')
        sheet.paste(show_row(*raw, f'{sid} | Original'), (0, 0))
        print(f'Sample {sid}:')
        for epoch in range(3):
            rgb, roof, valid, params = augment(*raw, rng=sample_rng(42, 0, epoch, sid))
            caption = f"{sid} | epoch {epoch}: rot={90 * params['k']}, H={params['horizontal']}, V={params['vertical']}, brightness={params['brightness']:.3f}, contrast={params['contrast']:.3f}"
            sheet.paste(show_row(rgb, roof, valid, caption), (0, (epoch + 1) * 302))
            print(caption)
        display(sheet)

def verify_augmentation(prepared_dir):
    manifest = load_manifest(prepared_dir)
    usable_ids = sorted(sid for sid, row in manifest.items() if row['status'] == 'usable_labelled')
    y, x = np.indices((16, 16))
    valid = ((x > 1) & (y < 14)).astype(np.uint8)
    roof = (((x + 2 * y) % 5 == 0) & (valid == 1)).astype(np.uint8)
    phantom = np.stack([roof * 255, valid * 255, ((x + 16 * y) % 256).astype(np.uint8)], axis=-1)
    phantom[valid == 0] = 0
    for k in range(4):
        for horizontal in (False, True):
            for vertical in (False, True):
                a, b, c = (geometric(arr, k, horizontal, vertical) for arr in (phantom, roof, valid))
                assert np.array_equal(a[..., 0], b * 255)
                assert np.array_equal(a[..., 1], c * 255)
    print('PASS: all 16 geometry parameter combinations preserve phantom alignment.')
    checks = 0
    for sid in usable_ids:
        raw = load_prepared(prepared_dir, sid)
        before = tuple((a.copy() for a in raw))
        for epoch in range(10):
            a, b, c, params = augment(*raw, rng=sample_rng(42, 0, epoch, sid))
            assert a.shape == (256, 256, 3) and b.shape == c.shape == (256, 256)
            assert set(np.unique(b)) <= {0, 1} and set(np.unique(c)) <= {0, 1}
            assert b.sum() == raw[1].sum() and c.sum() == raw[2].sum()
            assert not a[c == 0].any() and (not b[c == 0].any())
            geometry_args = {key: params[key] for key in ('k', 'horizontal', 'vertical')}
            assert np.array_equal(b, geometric(raw[1], **geometry_args))
            assert np.array_equal(c, geometric(raw[2], **geometry_args))
            checks += 1
        assert all((np.array_equal(a, b) for a, b in zip(raw, before)))
    print(f'PASS: {checks} augmented real samples preserve area, binary masks, dimensions and source arrays.')
    raw = load_prepared(prepared_dir, '303')
    a = augment(*raw, rng=sample_rng(42, 0, 2, '303'))
    b = augment(*raw, rng=sample_rng(42, 0, 2, '303'))
    assert all((np.array_equal(x, y) for x, y in zip(a[:3], b[:3]))) and a[3] == b[3]
    assert a[3] != augment(*raw, rng=sample_rng(42, 0, 3, '303'))[3]
    print('PASS: same seed reproduces output; another epoch changes this example draw.')
    colour_only = AugmentConfig(horizontal_flip_probability=0, vertical_flip_probability=0, rotate90=False, colour_probability=1)
    a = augment(*raw, rng=np.random.default_rng(7), config=colour_only)
    assert np.array_equal(a[1], raw[1]) and np.array_equal(a[2], raw[2])
    assert not np.array_equal(a[0], raw[0])
    print('PASS: colour-only augmentation changes RGB and preserves both masks.')
    for sid in usable_ids:
        raw = load_prepared(prepared_dir, sid)
        a = augment(*raw, training=False)
        assert all((np.array_equal(x, y) for x, y in zip(a[:3], raw)))
    print('PASS: validation preprocessing preserves every prepared sample before normalization.')

def demonstrate_fold(prepared_dir, config=None):
    manifest = load_manifest(prepared_dir)
    usable_ids = sorted(sid for sid, row in manifest.items() if row['status'] == 'usable_labelled')
    shuffled = np.random.default_rng(42).permutation(usable_ids).tolist()
    val_ids, train_ids = (shuffled[:5], shuffled[5:])
    kwargs = dict(prepared_dir=prepared_dir, train_ids=train_ids, val_ids=val_ids, seed=42, fold=0, allow_unreviewed_groups=True, config=config)
    train_ds = RoofFoldDataset(**kwargs, split='train')
    val_ds = RoofFoldDataset(**kwargs, split='val')
    print('Demo training:', len(train_ds), '| validation:', len(val_ds))
    print('Train IDs:', train_ids)
    print('Validation IDs:', val_ids)
    train_ds.set_epoch(0)
    batch_item = train_ds[0]
    print('Example:', batch_item['id'])
    for name in ('image', 'target', 'valid'):
        print(name, batch_item[name].shape, batch_item[name].dtype)
    v0 = val_ds[0]
    val_ds.set_epoch(100)
    v1 = val_ds[0]
    assert all((np.array_equal(v0[k], v1[k]) for k in ('image', 'target', 'valid')))
    print('PASS: validation model inputs remain identical after changing epoch.')

    def expect_rejection(train, val, rows=manifest, **kw):
        try:
            validate_split(rows, train, val, **kw)
        except ValueError as error:
            print('Correctly rejected:', error)
        else:
            raise AssertionError('Invalid split was accepted.')
    expect_rejection(train_ids, val_ids)
    expect_rejection(train_ids + [val_ids[0]], val_ids, allow_unreviewed_groups=True)
    expect_rejection(train_ids + ['278'], val_ids, allow_unreviewed_groups=True)
    grouped = {sid: dict(row, spatial_group='same-scene') for sid, row in manifest.items()}
    expect_rejection(train_ids, val_ids, rows=grouped)


def show_crop_examples(prepared_dir):
    """Force cropping only for illustration; training experiment uses p=0.3."""
    config = AugmentConfig(crop_probability=1.0, horizontal_flip_probability=0,
        vertical_flip_probability=0, rotate90=False, colour_probability=0)
    for sid in ('303', '532'):
        raw = load_prepared(prepared_dir, sid)
        sheet = Image.new('RGB', (1024, 302*4), 'white')
        sheet.paste(show_row(*raw, f'{sid} | Original'), (0, 0))
        for view in range(3):
            rgb, roof, valid, params = augment(*raw, rng=sample_rng(42, 0, 0, sid, view), config=config)
            box = params['crop_box']
            caption = f'{sid} | crop box={box}, zoom={256/(box[2]-box[0]):.3f}x' if box else f'{sid} | no acceptable crop; fallback'
            print(caption)
            sheet.paste(show_row(rgb, roof, valid, caption), (0, (view+1)*302))
        display(sheet)


def verify_crop_augmentation(prepared_dir):
    from .roof_augmentation import crop_resize
    config = AugmentConfig(crop_probability=0.3)
    ids = [sid for sid, row in load_manifest(prepared_dir).items() if row['status']=='usable_labelled']
    counts = Counter()
    for sid in ids:
        raw = load_prepared(prepared_dir, sid)
        before = tuple(a.copy() for a in raw)
        for epoch in range(10):
            result = augment(*raw, rng=sample_rng(42,0,epoch,sid), config=config)
            image, target, valid, params = result
            args = {k:params[k] for k in ('k','horizontal','vertical')}
            expected = [geometric(a, **args) for a in raw[1:]]
            if params['crop_applied']:
                left,top,right,bottom=params['crop_box']
                assert 224 <= right-left <= 256 and right-left==bottom-top
                assert expected[1][top:bottom,left:right].mean() >= 0.7
                expected = [np.array(Image.fromarray(a[top:bottom,left:right]).resize((256,256),Image.Resampling.NEAREST)) for a in expected]
                expected[0][expected[1]==0]=0
            else:
                assert target.sum()==raw[1].sum() and valid.sum()==raw[2].sum()
            assert np.array_equal(target,expected[0]) and np.array_equal(valid,expected[1])
            assert image.shape==(256,256,3) and target.shape==valid.shape==(256,256)
            assert set(np.unique(target))<={0,1} and set(np.unique(valid))<={0,1}
            assert valid.any() and not image[valid==0].any() and not target[valid==0].any()
            counts['samples']+=1
            counts['requested']+=params['crop_requested']
            counts['accepted']+=params['crop_applied']
            counts['fallbacks']+=params['crop_fallback']
        assert all(np.array_equal(a,b) for a,b in zip(raw,before))
        repeated=augment(*raw,rng=sample_rng(42,0,9,sid),config=config)
        assert result[3]==repeated[3] and all(np.array_equal(a,b) for a,b in zip(result[:3],repeated[:3]))
        validation=augment(*raw,training=False,config=config)
        assert all(np.array_equal(a,b) for a,b in zip(raw,validation[:3]))
    print('PASS: crop experiment on real samples:',dict(counts))
    print('PASS: exact mask reference, binary values, validity, reproducibility and unchanged validation.')
    # A known square in RGB and the target verifies crop alignment independently.
    valid=np.ones((256,256),dtype=np.uint8)
    roof=np.zeros_like(valid);roof[70:170,90:190]=1
    rgb=np.repeat((roof*255)[...,None],3,axis=2)
    a,b,c=crop_resize(rgb,roof,valid,(16,16,240,240))
    assert np.array_equal(a[...,0]>=128,b.astype(bool))
    assert c.all()
    print('PASS: synthetic RGB square and roof mask remain aligned after resizing.')
    # Missing values must not darken an otherwise constant valid image.
    valid[:,:64]=0;roof[:]=0;rgb[:]=100;rgb[valid==0]=0
    a,b,c=crop_resize(rgb,roof,valid,(16,16,240,240))
    assert (a[c==1]==100).all() and not a[c==0].any()
    forced=AugmentConfig(crop_probability=1, min_crop_valid_fraction=1, crop_side_range=(256,256),crop_attempts=2)
    result=augment(rgb,roof,valid,rng=np.random.default_rng(0),config=forced)
    assert result[3]['crop_fallback'] and not result[3]['crop_applied']
    print('PASS: no dark interpolation halo; bounded fallback when valid coverage is insufficient.')
    # Background-only crops are allowed; selection never requires a roof pixel.
    a,b,c,params=augment(np.zeros((256,256,3),dtype=np.uint8),np.zeros((256,256),dtype=np.uint8),np.ones((256,256),dtype=np.uint8),rng=np.random.default_rng(0),config=AugmentConfig(crop_probability=1))
    assert params['crop_applied'] and not b.any()
    print('PASS: background-only crop accepted.')
