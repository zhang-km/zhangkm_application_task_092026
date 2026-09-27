"""Preparation audit and visual results extracted from the original notebook."""
from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import csv
import json
import numpy as np
from PIL import Image, ImageDraw

def display(image):
    from IPython.display import display as ipython_display
    ipython_display(image)

class PreparationAudit:
    def __init__(self, paths, threshold=128):
        self.paths = paths
        self.ROOT = paths.project_root
        self.IMAGE_DIR = paths.raw_data_dir / 'images'
        self.LABEL_DIR = paths.raw_data_dir / 'labels'
        self.THRESHOLD = threshold
        self.images = {p.stem: p for p in sorted(self.IMAGE_DIR.glob('*.png'))}
        self.labels = {p.stem: p for p in sorted(self.LABEL_DIR.glob('*.png'))}
        if not self.images or not self.labels:
            raise FileNotFoundError('Raw images or labels missing; check RAW_DATA_DIR in .env')
        self.paired = sorted(self.images.keys() & self.labels.keys())
        self.unlabelled = sorted(self.images.keys() - self.labels.keys())
        self.records = []
        self.manifest = []
        print('Raw data:', paths.raw_data_dir)
        print('Prepared output:', paths.prepared_data_dir)
        print('Label threshold:', threshold, '| Validity: alpha == 255')
    def pixel_hash(self, path):
        with Image.open(path) as im:
            payload = f'{im.mode}:{im.size}'.encode() + im.tobytes()
        return hashlib.sha256(payload).hexdigest()

    def duplicates(self, paths):
        groups = defaultdict(list)
        for sid, path in paths.items():
            groups[self.pixel_hash(path)].append(sid)
        return [ids for ids in groups.values() if len(ids) > 1]

    def read_pair(self, sid):
        rgba = np.array(Image.open(self.images[sid]).convert('RGBA'))
        raw_mask = np.array(Image.open(self.labels[sid]).convert('L'))
        return (rgba, raw_mask)

    def rgb_view(self, rgba):
        yy, xx = np.indices(rgba.shape[:2])
        grey = np.where((xx // 12 + yy // 12) % 2, 200, 235).astype(np.uint8)
        background = np.repeat(grey[..., None], 3, axis=2)
        alpha = rgba[..., 3:4].astype(float) / 255
        return np.round(rgba[..., :3] * alpha + background * (1 - alpha)).astype(np.uint8)

    def panel(self, sid, scale=1):
        rgba, raw = self.read_pair(sid)
        rgb = self.rgb_view(rgba)
        roof = raw >= self.THRESHOLD
        overlay = rgb.copy()
        overlay[roof] = (0.55 * rgb[roof] + 0.45 * np.array([255, 30, 30])).astype(np.uint8)
        w, h = (rgba.shape[1], rgba.shape[0])
        out = Image.new('RGB', (w * 3, h + 28), 'white')
        draw = ImageDraw.Draw(out)
        for j, (name, array) in enumerate([('Image', rgb), ('Raw label', raw), ('Roof overlay', overlay)]):
            draw.text((j * w + 5, 7), f'{sid} | {name}', fill='black')
            out.paste(Image.fromarray(array).convert('RGB'), (j * w, 28))
        return out.resize((out.width * scale, out.height * scale))

    def summarize(self, rows, label):
        total = sum((r['pixels'] for r in rows))
        valid = sum((r['valid'] for r in rows))
        roof_valid = sum((r['roof_valid'] for r in rows))
        print(label)
        print(f"  Intermediate label pixels: {100 * sum((r['intermediate'] for r in rows)) / total:.2f}%")
        print(f"  Roof at >=128, all pixels: {100 * sum((r['roof_all'] for r in rows)) / total:.2f}%")
        print(f"  Roof at >0, all pixels: {100 * sum((r['nonzero'] for r in rows)) / total:.2f}%")
        print(f'  Fully opaque valid pixels: {100 * valid / total:.2f}%')
        print(f'  Roof among valid pixels: {100 * roof_valid / valid:.2f}%')
        print(f'  All-background accuracy over valid pixels: {100 * (1 - roof_valid / valid):.2f}%')
        print(f"  Roof-labelled pixels excluded by validity: {sum((r['roof_invalid'] for r in rows)):,}")

    def prepare(self, sid):
        rgba = np.array(Image.open(self.images[sid]).convert('RGBA'))
        valid = rgba[..., 3] == 255
        rgb = rgba[..., :3].copy()
        rgb[~valid] = 0
        target = None
        if sid in self.labels:
            target = (np.array(Image.open(self.labels[sid]).convert('L')) >= self.THRESHOLD).astype(np.uint8)
            target[~valid] = 0
        return (rgb, target, valid.astype(np.uint8))

    def inventory(self):
        self.images = {p.stem: p for p in sorted(self.IMAGE_DIR.glob('*.png'))}
        self.labels = {p.stem: p for p in sorted(self.LABEL_DIR.glob('*.png'))}
        self.paired = sorted(self.images.keys() & self.labels.keys())
        self.unlabelled = sorted(self.images.keys() - self.labels.keys())
        self.orphan_labels = sorted(self.labels.keys() - self.images.keys())
        assert not self.orphan_labels, f'Labels without images: {self.orphan_labels}'
        image_modes, label_modes, sizes = (Counter(), Counter(), Counter())
        for sid, path in self.images.items():
            with Image.open(path) as im:
                im.load()
                image_modes[im.mode] += 1
                sizes[str(im.size)] += 1
        for sid in self.paired:
            with Image.open(self.images[sid]) as im, Image.open(self.labels[sid]) as mask:
                mask.load()
                label_modes[mask.mode] += 1
                assert im.size == mask.size, f'Dimension mismatch: {sid}'
        print(f'Images: {len(self.images)} | labels: {len(self.labels)} | paired: {len(self.paired)}')
        print('Unlabelled:', ', '.join(self.unlabelled))
        print('Orphan labels:', self.orphan_labels)
        print('Image modes:', dict(image_modes), '| label modes:', dict(label_modes))
        print('Image dimensions:', dict(sizes))

    def duplicate_review(self):
        print('Exact duplicate images:', self.duplicates(self.images))
        print('Exact duplicate labels:', self.duplicates(self.labels))
        assert self.pixel_hash(self.labels['270']) == self.pixel_hash(self.labels['278'])
        assert self.pixel_hash(self.images['270']) != self.pixel_hash(self.images['278'])
        print('Confirmed: 270 and 278 have identical label pixels but different image pixels.')
        display(self.panel('270'))
        display(self.panel('278'))

    def mask_statistics(self):
        self.records = []
        for sid in self.paired:
            rgba, raw = self.read_pair(sid)
            roof, valid = (raw >= self.THRESHOLD, rgba[..., 3] == 255)
            assert valid.any(), f'No valid pixels: {sid}'
            self.records.append(dict(id=sid, pixels=raw.size, valid=int(valid.sum()), roof_valid=int((roof & valid).sum()), roof_all=int(roof.sum()), intermediate=int(((raw > 0) & (raw < 255)).sum()), nonzero=int((raw > 0).sum()), transparent=int((rgba[..., 3] == 0).sum()), partial_alpha=int(((rgba[..., 3] > 0) & (rgba[..., 3] < 255)).sum()), roof_invalid=int((roof & ~valid).sum())))
        self.summarize(self.records, 'All 25 supplied pairs (including quarantined 278):')
        print('\nPer-image audit (percentages):')
        print(f"{'ID':>5} {'roof/valid':>11} {'valid/all':>11} {'grey mask':>11} {'alpha=0':>9} {'partial a':>10}")
        for r in self.records:
            print(f"{r['id']:>5} {100 * r['roof_valid'] / r['valid']:11.2f} {100 * r['valid'] / r['pixels']:11.2f} {100 * r['intermediate'] / r['pixels']:11.2f} {100 * r['transparent'] / r['pixels']:9.2f} {100 * r['partial_alpha'] / r['pixels']:10.2f}")

    def mask_comparison(self):
        sid = '270'
        rgba, raw = self.read_pair(sid)
        valid = rgba[..., 3] == 255
        views = [('Raw mask', raw), ('Binary >=128', (raw >= 128).astype('uint8') * 255), ('Extra pixels with >0', ((raw > 0) & (raw < 128)).astype('uint8') * 255), ('Valid alpha==255', valid.astype('uint8') * 255)]
        canvas = Image.new('RGB', (256 * 4, 284), 'white')
        draw = ImageDraw.Draw(canvas)
        for i, (title, array) in enumerate(views):
            draw.text((i * 256 + 5, 7), title, fill='black')
            canvas.paste(Image.fromarray(array).convert('RGB'), (i * 256, 28))
        display(canvas)

    def label_gallery(self):
        for start in range(0, len(self.paired), 5):
            ids = self.paired[start:start + 5]
            sheet = Image.new('RGB', (768, 284 * len(ids)), 'white')
            for row, sid in enumerate(ids):
                sheet.paste(self.panel(sid), (0, row * 284))
            print('Pairs:', ', '.join(ids))
            display(sheet)

    def build_manifest(self):
        self.QUARANTINE = {'278': 'Label identical to 270; visual mismatch with image. Verified correction required.'}
        self.usable = [sid for sid in self.paired if sid not in self.QUARANTINE]
        self.manifest = []
        for sid in sorted(self.images):
            status = 'quarantined' if sid in self.QUARANTINE else 'usable_labelled' if sid in self.labels else 'unlabelled'
            self.manifest.append({'id': sid, 'image': str(self.images[sid]), 'label': str(self.labels[sid]) if sid in self.labels else '', 'status': status, 'reason': self.QUARANTINE.get(sid, ''), 'spatial_group': '', 'annotation_review': 'needs human consistency review' if status == 'usable_labelled' else ''})
        for sid in self.usable + self.unlabelled:
            rgb, target, valid = self.prepare(sid)
            assert rgb.shape == (256, 256, 3) and rgb.dtype == np.uint8
            assert set(np.unique(valid)).issubset({0, 1})
            if target is not None:
                assert set(np.unique(target)).issubset({0, 1})
                assert not target[valid == 0].any()
        print('Manifest counts:', dict(Counter((row['status'] for row in self.manifest))))
        print('Usable labelled IDs:', ', '.join(self.usable))
        print('Quarantined:', self.QUARANTINE)
        print('Unlabelled prediction IDs:', ', '.join(self.unlabelled))
        self.summarize([r for r in self.records if r['id'] in self.usable], '\nProvisional retained dataset:')
        print('\nPreparation checks passed for all usable and unlabelled inputs.')

    def prepared_preview(self):
        sid = '303'
        rgb, target, valid = self.prepare(sid)
        canvas = Image.new('RGB', (768, 284), 'white')
        draw = ImageDraw.Draw(canvas)
        for j, (name, array) in enumerate([('Prepared RGB', rgb), ('Binary roof target', target * 255), ('Validity mask', valid * 255)]):
            draw.text((j * 256 + 5, 7), f'{sid} | {name}', fill='black')
            canvas.paste(Image.fromarray(array).convert('RGB'), (j * 256, 28))
        display(canvas)

    def export(self, enabled=True):
        """Write prepared copies; preserve existing human review/group metadata."""
        if not enabled:
            print('Export disabled; original and prepared files unchanged.')
            return
        if not self.manifest or not self.records:
            raise RuntimeError('Run mask_statistics and build_manifest before export.')
        out = self.paths.prepared_data_dir
        if out == self.paths.raw_data_dir or self.paths.raw_data_dir in out.parents:
            raise ValueError('Prepared output must be separate from the raw data directory.')
        out.mkdir(parents=True, exist_ok=True)
        manifest_file = out / 'manifest.csv'
        if manifest_file.exists():
            with manifest_file.open(newline='') as f:
                old = {r['id']: r for r in csv.DictReader(f)}
            for row in self.manifest:
                previous = old.get(row['id'], {})
                for key in ('spatial_group', 'annotation_review'):
                    if previous.get(key):
                        row[key] = previous[key]
        with manifest_file.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(self.manifest[0]))
            writer.writeheader()
            writer.writerows(self.manifest)
        (out / 'audit.json').write_text(json.dumps({'label_threshold': self.THRESHOLD,
            'validity_rule': 'alpha == 255', 'quarantine': self.QUARANTINE,
            'per_image': self.records}, indent=2))
        for folder in ('rgb', 'roof_masks', 'validity_masks'):
            (out / folder).mkdir(exist_ok=True)
        for sid in self.usable + self.unlabelled:
            rgb, target, valid = self.prepare(sid)
            Image.fromarray(rgb).save(out / 'rgb' / f'{sid}.png')
            Image.fromarray(valid * 255).save(out / 'validity_masks' / f'{sid}.png')
            if target is not None:
                Image.fromarray(target * 255).save(out / 'roof_masks' / f'{sid}.png')
        print('Prepared copies and manifest exported to', out)
