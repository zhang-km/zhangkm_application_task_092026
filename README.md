# Roof segmentation

An end-to-end roof segmentation project using 30 satellite images and 25 supplied masks. Data review retained 24 usable labelled pairs: image 278 is excluded because its mask duplicates 270's and is misaligned. Five images are unlabelled. Preparation, CPU checks, five-fold cross-validation, augmentation and learning-rate comparisons, threshold exploration, and final GPU training are complete.

The selected ResNet18 U-Net was fitted to all 24 usable labels for **196 epochs**, then used to predict images **535, 537, 539, 551 and 553**. See [method.docx](method.docx) for the methodology, experiment results, limitations, and final prediction plots.

## Selected configuration and results

| Setting | Selected value |
| --- | --- |
| Model | ImageNet-pretrained ResNet18 U-Net; GroupNorm decoder; encoder BatchNorm statistics frozen |
| Loss | Equal weights of masked BCE and soft Dice |
| Optimizer | AdamW; batch size 4; weight decay 1e-4; seed 42 |
| Augmentation | Flips, quarter-turn rotations, mild brightness/contrast, crop probability 0.3 |
| Warm-up | 5 decoder-only epochs at learning rate 1e-3 |
| Fine-tuning | Encoder learning rate 3e-5; decoder learning rate 3e-4 |
| Prediction threshold | 0.45 |
| Development stopping | Validation loss; minimum 30 epochs; patience 20; maximum 500 epochs |
| Final training | Fixed 196 epochs on all 24 labels; no validation-based stopping |

The matched learning-rate comparison improved mean per-image out-of-fold IoU from **0.7671 to 0.7979** at threshold 0.45; 19 of 24 images improved. Mean validation loss across folds fell from **0.2347 to 0.1725**, with improvements in all five folds. Threshold 0.575 achieved 0.7984 IoU, only about 0.0005 above 0.45, so 0.45 was retained. These are development results: the same folds informed model selection, geographic independence is unverified, and comparisons used one seed. The final all-label model has no independent held-out accuracy measurement; the five unlabelled predictions are qualitative results.

The final duration is the median of the selected candidate's best epochs: 173, 138, 207, 196 and 304. It is a practical duration heuristic, not a proven optimum. The completed selected run is `final_crop_lr3x_t045_196epochs`.

## Repository layout

```text
.env                         Local paths, ignored by Git
.env.example                 Portable path template
pyproject.toml               Package and dependency definitions
requirements.txt             Editable install with notebook/training extras
method.docx                  Methods, results and final figures
roof_utils/
  config.py                  Path resolution
  data_preparation.py        Inspection, preparation and manifest
  roof_augmentation.py       Paired augmentation and fold dataset
  augmentation_review.py     Augmentation demonstrations/checks
  visualization.py           Image/mask display
  validation_splits.py       Frozen fold and data-hash checks
  models.py                  ResNet18 U-Net
  losses_metrics.py          Masked loss and segmentation metrics
  training.py                Training primitives
  model_review.py            CPU model diagnostics
  cross_validation.py        CV, checkpoint/resume and prediction review
  hyperparameter_review.py   Learning-rate comparison and threshold sweep
  final_training.py          All-label fit, inference and ZIP export
notebooks/                   Preparation, diagnostics and experiment notebooks
splits/                      Frozen validation assignments and documentation
tests/                       Automated CPU checks
data/                        Raw images/labels (ignored)
data_preparation_outputs/    Prepared copies and manifest (ignored)
models/                      Checkpoints (ignored)
reports/                     Metrics, predictions and exports (ignored)
```

## Local setup

From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install 'matplotlib>=3.8,<4'
# On a fresh clone only; preserve existing settings:
cp -n .env.example .env
python -m jupyterlab
```

The local training extra pins PyTorch 2.8.0 and torchvision 0.23.0. Matplotlib is installed separately above for plots. For Colab, use the notebook installation cells, which preserve an existing CUDA stack; the completed final export records PyTorch 2.11.0+cu128. Retain software metadata with results.

Choose the environment's Python kernel. Notebooks locate the project root; reusable functions live in `roof_utils`. A clone does not contain ignored datasets or trained outputs. Supply `images/` and `labels/` under `RAW_DATA_DIR`, or point it at the original data folder. Initial pretrained-weight download requires internet; weights are cached under `MODELS_DIR/pretrained`.

## Paths

Configure `.env`:

```dotenv
RAW_DATA_DIR=data
PREPARED_DATA_DIR=data_preparation_outputs
MODELS_DIR=models
REPORTS_DIR=reports
```

Relative paths resolve against the repository root. Absolute paths and `~` are supported; quote paths containing spaces. Shell environment variables override `.env`. The path reader supports literal `KEY=value` lines and comments, without variable expansion or multiline values.

```python
from roof_utils.config import load_paths
paths = load_paths()
```

## Notebook workflow

1. [Data preparation and label quality](notebooks/data_preparation_and_label_quality.ipynb): inspect pairs, transparency and labels; export prepared copies and the eligibility manifest. Set `EXPORT = False` for read-only inspection. Raw files are preserved; stale prepared files are not deleted, so eligibility comes from the manifest.
2. [Data augmentation](notebooks/data_augmentation.ipynb): inspect paired transformations and baseline/crop checks. Augmentation is sampled during training access, not saved as an expanded dataset.
3. [Validation splits](notebooks/validation_splits.ipynb): review the five frozen folds in [splits](splits/README.md). Each usable image is validation data exactly once; image 278 and unlabelled inputs are excluded. Fold membership, groups and data hashes are verified on load.
4. [CPU model diagnostics](notebooks/model_training.ipynb): model/loss checks and two-image memorization. The saved diagnostic passed after 50 steps, with IoUs 0.9634 and 0.9662 on its training images. These are sanity scores, not validation accuracy. Each CV fold starts afresh from pretrained encoder weights.
5. Baseline/crop development: use matched variants such as [baseline, threshold 0.45, 500 epochs](notebooks/baseline_cv_threshold045_500epoch.ipynb) and [crop, threshold 0.45, 500 epochs](notebooks/crop_cv_threshold045_500epoch.ipynb). Other baseline/crop notebooks preserve earlier experiment settings; inspect their configuration and run names before executing. `baseline_cv.ipynb` still uses threshold 0.5 and a 100-epoch ceiling.
6. [Learning-rate comparison](notebooks/learning_rate_comparison.ipynb): matched crop-enabled five-fold runs `crop_lr_reference_500` and `crop_lr_3x_500`, comparing 1e-5/1e-4 against 3e-5/3e-4. Training/checkpoint selection uses validation loss; comparison IoU uses threshold 0.45.
7. [Threshold exploration](notebooks/threshold_exploration.ipynb): sweep saved out-of-fold probabilities from `crop_lr_3x_500`. The current notebook requires these completed predictions; it imports `ensure_reference_run` but does not call it. Complete/restore the candidate run first. Sweeping thresholds requires no retraining and does not alter loss or weights.
8. [Final model training](notebooks/final_model_training.ipynb): derive the fixed duration from the five candidate `result.json` files, train on all labels, predict the five unlabelled images, display results and export. `FINAL_EPOCHS_OVERRIDE` permits a deliberate fixed budget when needed. The completed selected budget is 196 epochs; monitor/minimum epochs/patience are inactive in the final fit.

Mild cropping samples square sides of 224–256 pixels and resizes to 256 × 256. It requires at least 70% valid pixels, tries up to ten times, and otherwise retains the uncropped image. Background-only crops are allowed. RGB uses validity-weighted bilinear interpolation; masks use nearest-neighbour interpolation. Validation/inference never use random crops. Invalid pixels are excluded from both loss and metrics. Library defaults remain baseline settings (`crop_probability=0.0`, threshold 0.5, lower learning rates); explicitly use the selected notebook configuration.

## GPU execution and saved outputs

See [the VS Code Colab guide](notebooks/COLAB_README.md) for uploads, runtime paths, import troubleshooting and downloading results. Selecting a remote kernel does not make local files available there. Google Drive is optional.

With the configured paths, the selected final run writes:

```text
MODELS_DIR/final_crop_lr3x_t045_196epochs/
  final.pt                   Final inference checkpoint
  last.pt                    Resume checkpoint
REPORTS_DIR/final_crop_lr3x_t045_196epochs/
  experiment.json            Configuration/data/code/software signature
  history.csv                Training history
  result.json                Completed-run metadata
  unlabelled_predictions/    Probabilities, binary masks, validity and metadata
REPORTS_DIR/final_crop_lr3x_t045_196epochs_export.zip
```

`export_final_run(paths, RUN_NAME)` creates an archive in the kernel's filesystem. It does not automatically download it to your computer. A downloaded selected export is retained locally under `reports/final_crop_lr3x_t045_196epochs_export/`; this ignored artifact is not included in a fresh clone. The older 285-epoch export is historical, not the selected final model.

Resume requires retained model and report folders and matching configuration, data, code and software signatures. Use a new run name for a changed experiment; do not remove `experiment.json` to bypass the check. One process should write to each run directory. Saving uses a temporary file before replacement, so an interrupted save may leave the preceding completed checkpoint plus a partial `.tmp` file.

## Verification and version control

Run CPU checks from the root:

```sh
python -m unittest discover -s tests -v
```

Raw/prepared data, `.env`, checkpoints, reports, environments and backups are ignored. Notebooks can contain saved image outputs and metrics even when their source directories are ignored; review outputs before publishing. Review `git status` and stage intended changes. Reproducibility requires keeping the data, frozen splits, code, configuration and software metadata alongside exported results.
