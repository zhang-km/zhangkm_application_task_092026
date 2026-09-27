# Roof segmentation

Preparation and augmentation of satellite images and roof masks. The supervised dataset currently contains 24 usable pairs; pair 278 is quarantined because its label duplicates 270's and is misaligned. Five other images have no labels. A pretrained ResNet18 U-Net and CPU training primitives are implemented; the two-image sanity check has passed. Full cross-validation training has not yet been run.

## Layout

```text
.env                         Local paths, ignored by Git
.env.example                 Portable path configuration template
pyproject.toml               Package and dependency definitions
requirements.txt             Editable installation with Jupyter support
method.docx                  Method description
roof_utils/
  config.py                  Resolve paths from .env
  data_preparation.py        Inspection, preparation, manifest and export
  roof_augmentation.py       Paired augmentation and fold dataset
  visualization.py           Image/mask display helper
  augmentation_review.py     Demonstrations and verification routines
notebooks/
  data_preparation_and_label_quality.ipynb
  data_augmentation.ipynb
data/                        Original images and labels, ignored by Git
data_preparation_outputs/    Prepared copies and manifest, ignored by Git
```

## Setup

From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# On a fresh clone only; do not overwrite existing path settings:
cp -n .env.example .env
python -m jupyterlab
```

Choose the environment's Python kernel. The notebooks can be launched from the repository root or `notebooks/`; their short bootstrap finds the root. Utility code can also be imported from elsewhere after the editable installation.

The original local data remains in this project. A clone will not contain it: supply `images/` and `labels/` under `RAW_DATA_DIR`, or point that setting at an existing data folder.

## Paths

Edit `.env` rather than notebook code:

```dotenv
RAW_DATA_DIR=data
PREPARED_DATA_DIR=data_preparation_outputs
MODELS_DIR=models
REPORTS_DIR=reports
```

Relative paths resolve against the repository root, not the notebook working directory. Absolute paths and `~` are supported. Quote paths containing spaces. Shell environment settings override `.env`. The small path-only reader supports literal `KEY=value` lines and comments; it does not expand `${VARIABLES}` or multiline values. These are path settings, not credentials. `.env` is ignored and `.env.example` is tracked.

```python
from roof_utils.config import load_paths
from roof_utils.roof_augmentation import AugmentConfig, RoofFoldDataset

paths = load_paths()
# Supply reviewed, non-overlapping original IDs for each fold.
train_ds = RoofFoldDataset(
    paths.prepared_data_dir, train_ids, val_ids,
    split="train", seed=42, fold=0, config=AugmentConfig(),
)
train_ds.set_epoch(0)
```

## Notebook order

1. **Data preparation:** inspect all image/mask pairs, binary masks and transparency. Export is enabled, matching the existing notebook. Set `EXPORT = False` for a read-only run. Export writes prepared copies to the configured directory and preserves existing spatial-group and annotation-review fields. Raw files are never modified. Stale prepared files are not deleted; use the manifest for sample eligibility.
2. **Augmentation:** imports the implementation and shows exact paired flips/quarter-turns plus mild RGB brightness/contrast changes. It verifies mask alignment, binary values, reproducibility and unchanged validation inputs. Samples are augmented during access, not written as a permanently expanded dataset.

Both notebooks contain observed results. Their images reveal the input data even though raw image directories are ignored; clear notebook outputs before publishing if you do not want to share those images.

Visual scene review is complete and fixed image-level development folds are saved. No repeated scene was confirmed; geographic independence is unknown. The augmentation notebook uses saved fold 0. See `splits/README.md`. `RoofFoldDataset` checks known group boundaries and rejects unreviewed groups by default; a development split requires `allow_unreviewed_groups=True`.

Keep validity masks in both loss and metrics. RGB is normalized after augmentation using ImageNet channel statistics; masks are not normalized. Quarter-turns and flips introduce no resampling or padding. Pair 278 is preserved but remains excluded from supervised splits.

## Git

This is a local repository on branch `main`; no remote or initial commit is created automatically. Raw/prepared data, `.env`, checkpoints, environments and migration backups are ignored. Review changes before your first commit:

```sh
git status
git add .
git commit -m "Set up roof segmentation preparation and augmentation"
```

## Optional mild crop and zoom

Use `AugmentConfig(crop_probability=0.3)` and pass it as `config` to the fold dataset. The default is `0.0`, preserving the previous baseline. This samples square crops of 224–256 pixels, then restores 256 × 256 resolution. A crop requires 70% valid pixels; after 10 unsuccessful attempts the uncropped sample is retained. Background-only crops are allowed. RGB uses validity-weighted bilinear resizing; roof/validity masks use nearest-neighbour. Validation is unchanged. The augmentation notebook demonstrates both baseline and crop configurations and includes their separate checks.

## Frozen validation splits

The reviewed development folds are saved in `splits/validation_folds.json` with a readable assignment CSV and `splits/README.md`. Open `notebooks/validation_splits.ipynb` for the executed review. Use `load_fold(paths, fold=0)` from `roof_utils.validation_splits` in subsequent notebooks. There are 24 usable labelled images and five validation folds of 4–5 images; 278 and all unlabelled inputs are excluded. Singleton scene IDs reflect visual review only: no repeated scene was confirmed and geographical independence remains unknown. Data hashes and group membership are checked on load.

## CPU model setup and sanity test

`notebooks/model_training.ipynb` imports the U-Net, masked loss/metrics and training utilities. Install the updated requirements first; they include the pinned `training` dependencies (PyTorch 2.8.0 and torchvision 0.23.0). The notebook explicitly uses CPU. The official ResNet18 weights are cached in `MODELS_DIR/pretrained`; this initial download requires internet on a fresh setup.

Run automated CPU tests from the repository root:

```sh
python -m unittest discover -s tests -v
```

The saved sanity run intentionally fits only training images 270 and 314, without augmentation. Results, prediction arrays and history are in `REPORTS_DIR/sanity_two_images`; the diagnostic checkpoint is in `MODELS_DIR/sanity_two_images/model.pt`. Set `RUN_SANITY = True` in the notebook to rerun on CPU. It overwrites this diagnostic's outputs. These directories are ignored by Git, so a fresh clone must rerun the diagnostic before displaying results.

The CPU run passed after 50 steps: training-image IoU was 0.9634 and 0.9662. These are memorization scores, not held-out accuracy. The four automated tests and checkpoint round-trip passed. The sanity test keeps decoder LR at 0.001 for fast fitting; later CV should use the method's lower fine-tuning rate. Do not reuse this checkpoint for CV: initialize each fold afresh from ImageNet weights.

## Colab GPU experiments

See `notebooks/COLAB_README.md`. Run `notebooks/baseline_cv.ipynb`, then `notebooks/crop_cv.ipynb`. Shared training and checkpoint/resume logic lives in `roof_utils/cross_validation.py`. The notebooks install dependencies first, locate the project visible to the selected kernel, read `.env`, verify the frozen data/splits, and save checkpoints and reports to the configured paths. Google Drive is not required; see the setup guide for VS Code file upload and runtime storage details.
