"""Paired roof augmentation. Requires NumPy and Pillow; no source files are changed."""
from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
from PIL import Image

@dataclass(frozen=True)
class AugmentConfig:
    horizontal_flip_probability: float = 0.5
    vertical_flip_probability: float = 0.5
    rotate90: bool = True
    colour_probability: float = 0.8
    brightness_range: tuple = (0.9, 1.1)
    contrast_range: tuple = (0.9, 1.1)

    crop_probability: float = 0.0  # Opt in with 0.3 for the crop/zoom experiment.
    crop_side_range: tuple = (224, 256)
    min_crop_valid_fraction: float = 0.7
    crop_attempts: int = 10

    def __post_init__(self):
        for p in (self.horizontal_flip_probability, self.vertical_flip_probability, self.colour_probability, self.crop_probability, self.min_crop_valid_fraction):
            if not 0 <= p <= 1:
                raise ValueError('Probabilities must be between 0 and 1.')
        if (len(self.crop_side_range) != 2 or
                any(not isinstance(v, int) or isinstance(v, bool) for v in self.crop_side_range) or
                not 1 <= self.crop_side_range[0] <= self.crop_side_range[1]):
            raise ValueError('Crop side bounds must be ordered positive integers.')
        if not isinstance(self.crop_attempts, int) or isinstance(self.crop_attempts, bool) or self.crop_attempts < 1:
            raise ValueError('crop_attempts must be a positive integer.')
        for interval in (self.brightness_range, self.contrast_range):
            if len(interval) != 2 or not 0 < interval[0] <= interval[1] or not np.isfinite(interval).all():
                raise ValueError('Brightness and contrast bounds must be positive, finite, and ordered.')


def validate_arrays(rgb, roof, valid):
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError('RGB must have uint8 shape H x W x 3.')
    if roof.shape != rgb.shape[:2] or valid.shape != roof.shape:
        raise ValueError('Image, roof and validity dimensions must agree.')
    for mask in (roof, valid):
        if not np.isin(mask, [0, 1]).all():
            raise ValueError('Masks must contain only 0 and 1.')
    if not valid.any():
        raise ValueError('Sample contains no valid pixels.')
    if roof[valid == 0].any():
        raise ValueError('Prepared roof target must be zero outside valid pixels.')


def geometric(array, k=0, horizontal=False, vertical=False):
    """Exact pixel permutations: no interpolation, crop, padding, or new mask values."""
    result = np.rot90(array, k=int(k), axes=(0, 1))
    if horizontal:
        result = np.flip(result, axis=1)
    if vertical:
        result = np.flip(result, axis=0)
    return np.ascontiguousarray(result).copy()


def crop_resize(rgb, roof, valid, box):
    """Apply one (left, top, right, bottom) crop and restore original dimensions.

    RGB uses validity-weighted bilinear interpolation to avoid dark halos from
    missing pixels. Both discrete masks use nearest-neighbour interpolation.
    """
    validate_arrays(rgb, roof, valid)
    left, top, right, bottom = box
    h, w = roof.shape
    if not (0 <= left < right <= w and 0 <= top < bottom <= h):
        raise ValueError('Crop box is outside the image.')
    if right-left != bottom-top:
        raise ValueError('Crop must be square.')
    sl = np.s_[top:bottom, left:right]
    v = valid[sl].astype(np.float32)
    if not v.any():
        raise ValueError('Crop has no valid pixels.')
    size = (w, h)
    weights = np.array(Image.fromarray(v).resize(size, Image.Resampling.BILINEAR))
    channels = []
    for channel in range(3):
        values = rgb[sl][..., channel].astype(np.float32) * v
        numerator = np.array(Image.fromarray(values).resize(size, Image.Resampling.BILINEAR))
        channels.append(np.divide(numerator, weights, out=np.zeros_like(numerator), where=weights > 0))
    image = np.rint(np.clip(np.stack(channels, axis=-1), 0, 255)).astype(np.uint8)
    target = np.array(Image.fromarray(roof[sl].astype(np.uint8)).resize(size, Image.Resampling.NEAREST))
    validity = np.array(Image.fromarray(valid[sl].astype(np.uint8)).resize(size, Image.Resampling.NEAREST))
    image[validity == 0] = 0
    target[validity == 0] = 0
    return image, target, validity


def augment(rgb, roof, valid, *, rng=None, training=True, config=None):
    """Return augmented RGB, binary roof, binary validity, and sampled parameters.

    In validation mode this consumes no randomness. Returned arrays are independent
    copies. Apply encoder normalization after this function, never before.
    """
    validate_arrays(rgb, roof, valid)
    config = config or AugmentConfig()
    params = dict(k=0, horizontal=False, vertical=False, brightness=1.0, contrast=1.0,
                  crop_box=None, crop_applied=False, crop_requested=False, crop_fallback=False)
    if training:
        if rng is None:
            raise ValueError('Pass an explicit NumPy Generator for reproducible training.')
        if config.rotate90 and rgb.shape[0] != rgb.shape[1]:
            raise ValueError('The rotation baseline requires square images for fixed batch dimensions.')
        params['k'] = int(rng.integers(0, 4)) if config.rotate90 else 0
        params['horizontal'] = bool(rng.random() < config.horizontal_flip_probability)
        params['vertical'] = bool(rng.random() < config.vertical_flip_probability)
        if rng.random() < config.colour_probability:
            params['brightness'] = float(rng.uniform(*config.brightness_range))
            params['contrast'] = float(rng.uniform(*config.contrast_range))
    args = {key: params[key] for key in ('k', 'horizontal', 'vertical')}
    image, target, validity = (geometric(a, **args) for a in (rgb, roof, valid))
    if training and config.crop_probability > 0:
        h, w = target.shape
        low, high = config.crop_side_range
        if h != w or high > min(h, w):
            raise ValueError('Crop configuration requires square inputs and crop bounds within the image.')
        params['crop_requested'] = bool(rng.random() < config.crop_probability)
        if params['crop_requested']:
            for _ in range(config.crop_attempts):
                side = int(rng.integers(low, high + 1))
                left = int(rng.integers(0, w - side + 1))
                top = int(rng.integers(0, h - side + 1))
                patch = validity[top:top+side, left:left+side]
                if patch.any() and patch.mean() >= config.min_crop_valid_fraction:
                    box = (left, top, left+side, top+side)
                    image, target, validity = crop_resize(image, target, validity, box)
                    params.update(crop_box=box, crop_applied=True)
                    break
            else:
                # Keep the original geometrically transformed sample; never loop forever.
                params['crop_fallback'] = True
    if params['brightness'] != 1.0 or params['contrast'] != 1.0:
        x = image.astype(np.float32) / 255.0
        # Compute the contrast centre over valid RGB pixels only. Black missing
        # regions must not change the amount of contrast applied to visible roofs.
        centre = x[validity.astype(bool)].mean(axis=0)
        x = ((x - centre) * params['contrast'] + centre) * params['brightness']
        image = np.rint(np.clip(x, 0, 1) * 255).astype(np.uint8)
    image[validity == 0] = 0
    target[validity == 0] = 0
    return image, target.astype(np.uint8), validity.astype(np.uint8), params


def load_manifest(prepared_dir):
    with (Path(prepared_dir) / 'manifest.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate sample IDs in manifest.')
    return {r['id']: r for r in rows}


def load_prepared(prepared_dir, sample_id):
    """Load exported uint8 RGB and 0/255 masks; return masks encoded as 0/1."""
    base = Path(prepared_dir)
    with Image.open(base / 'rgb' / f'{sample_id}.png') as im:
        rgb = np.array(im.convert('RGB'))
    masks = []
    for folder in ('roof_masks', 'validity_masks'):
        with Image.open(base / folder / f'{sample_id}.png') as im:
            raw = np.array(im.convert('L'))
        if not np.isin(raw, [0, 255]).all():
            raise ValueError(f'{folder}/{sample_id}.png is not a binary exported mask.')
        masks.append((raw == 255).astype(np.uint8))
    validate_arrays(rgb, *masks)
    return rgb, *masks


def validate_split(manifest, train_ids, val_ids, *, allow_unreviewed_groups=False):
    train_ids, val_ids = list(train_ids), list(val_ids)
    if not train_ids or not val_ids:
        raise ValueError('Training and validation sets must both be nonempty.')
    if len(set(train_ids)) != len(train_ids) or len(set(val_ids)) != len(val_ids):
        raise ValueError('Duplicate IDs within split.')
    if set(train_ids) & set(val_ids):
        raise ValueError('Training and validation IDs overlap.')
    ids = train_ids + val_ids
    if any(sid not in manifest or manifest[sid]['status'] != 'usable_labelled' for sid in ids):
        raise ValueError('Only usable_labelled IDs may enter supervised splits.')
    groups = {sid: manifest[sid].get('spatial_group', '').strip() for sid in ids}
    if any(groups.values()) and not all(groups.values()):
        raise ValueError('Spatial group metadata is incomplete: fill it for every selected ID.')
    if all(groups.values()):
        if {groups[s] for s in train_ids} & {groups[s] for s in val_ids}:
            raise ValueError('A spatial group crosses the training/validation boundary.')
    elif not allow_unreviewed_groups:
        raise ValueError('Spatial groups are unreviewed. Assign groups, or explicitly allow an image-level development split.')


def sample_rng(seed, fold, epoch, sample_id, view=0):
    """Stable across worker scheduling; increment epoch/view for a new draw."""
    payload = json.dumps([int(seed), int(fold), int(epoch), str(sample_id), int(view)]).encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(payload).digest()[:16], 'little'))


def model_arrays(rgb, roof, valid, *, normalize=True):
    """Float32 CHW image and 1HW masks; ImageNet normalization for ResNet18.

    Invalid RGB is zero before normalization. Always mask loss and metrics using
    validity, including Dice's numerator and denominator.
    """
    validate_arrays(rgb, roof, valid)
    x = rgb.astype(np.float32) / 255.0
    if normalize:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = (x - mean) / std
    return (np.ascontiguousarray(x.transpose(2, 0, 1)),
            roof[None].astype(np.float32), valid[None].astype(np.float32))


class RoofFoldDataset:
    """Map-style dataset returning NumPy arrays, suitable for a later DataLoader.

    Set epoch before each training epoch. Start with num_workers=0. If using
    worker processes, use persistent_workers=False so each iterator receives the
    new epoch; persistent workers need an explicit shared epoch implementation.
    Repeated access to the same ID in one epoch gives the same augmentation.
    For sampling multiple views, call get(index, view=...) with different views.
    """
    def __init__(self, prepared_dir, train_ids, val_ids, *, split, seed=42, fold=0,
                 config=None, normalize=True, allow_unreviewed_groups=False):
        self.prepared_dir = Path(prepared_dir)
        manifest = load_manifest(prepared_dir)
        validate_split(manifest, train_ids, val_ids,
                       allow_unreviewed_groups=allow_unreviewed_groups)
        if split not in ('train', 'val'):
            raise ValueError('split must be train or val')
        self.ids = list(train_ids if split == 'train' else val_ids)
        self.training = split == 'train'
        self.seed, self.fold, self.epoch = seed, fold, 0
        self.config, self.normalize = config or AugmentConfig(), normalize

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.ids)

    def get(self, index, view=0):
        sid = self.ids[index]
        raw = load_prepared(self.prepared_dir, sid)
        rng = sample_rng(self.seed, self.fold, self.epoch, sid, view) if self.training else None
        rgb, roof, valid, _ = augment(*raw, rng=rng, training=self.training, config=self.config)
        x, y, v = model_arrays(rgb, roof, valid, normalize=self.normalize)
        return {'image': x, 'target': y, 'valid': v, 'id': sid}

    def __getitem__(self, index):
        return self.get(index)
