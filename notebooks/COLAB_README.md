# Training with the VS Code Colab kernel

The project has completed GPU cross-validation and final training. The selected final run is `final_crop_lr3x_t045_196epochs`: crop probability 0.3, encoder/decoder learning rates 3e-5/3e-4, threshold 0.45 and 196 fixed epochs on all 24 usable labels. See the [project README](../README.md) and [method document](../method.docx) for results and limitations.

## Upload and setup

1. Open the desired local notebook in VS Code and select your Colab GPU kernel.
2. Upload the project files to that runtime, preserving their relative layout. Include `roof_utils/`, `pyproject.toml`, `splits/`, `data/` and `data_preparation_outputs/`. Keep notebooks available in VS Code. For an existing experiment, also upload the required `reports/` and `models/` run directories described below.
3. Run the notebook's installation cell first. It installs plotting/data dependencies and preserves an existing PyTorch/CUDA stack; if installation requests a restart, restart and rerun setup. Do not blindly apply the local pinned training environment over a working Colab CUDA environment.
4. Run the project-path setup cell. Set `PROJECT_ROOT_OVERRIDE` to the runtime directory containing both `pyproject.toml` and `roof_utils/` if automatic discovery fails or finds multiple copies. Example: `/content/dida_test_task`, or `/content` if files were uploaded directly there.
5. Check the printed project, model and report paths, then run imports and split validation before training. Run cells in order so `paths` and the configuration exist.

Selecting a kernel does not share your Mac's filesystem. Uploading the whole project is acceptable, but `.venv/`, `.git/`, caches and backups are unnecessary. Upload only the model/report runs needed for your task to avoid transferring large unrelated checkpoints. Google Drive is not required.

Use runtime-visible paths in `.env`, usually:

```dotenv
RAW_DATA_DIR=data
PREPARED_DATA_DIR=data_preparation_outputs
MODELS_DIR=models
REPORTS_DIR=reports
```

Setup creates relative defaults only when `.env` is absent. An uploaded `.env` containing Mac paths must be adjusted for the runtime. Environment variables override `.env`.

## Experiment order and required saved files

- **Baseline/crop comparison:** the existing 500-epoch pair is `baseline_cv_threshold045_500epoch.ipynb` and `crop_cv_threshold045_500epoch.ipynb`. There is no `crop_cv.ipynb`. Earlier notebook variants retain their historical settings; compare matching folds, threshold, training rules and software.
- **Learning-rate comparison:** `learning_rate_comparison.ipynb` runs or resumes `crop_lr_reference_500` and `crop_lr_3x_500`. Both enable crop probability 0.3, use threshold 0.45, monitor validation loss, and set minimum epochs 30, patience 20 and maximum epochs 500. Fine-tuning rates change from 1e-5/1e-4 to 3e-5/3e-4. Five decoder warm-up epochs remain at 1e-3. A new comparison needs ten fold fits; compatible saved runs can be reused.
- **Threshold exploration:** after selecting the learning rates, run `threshold_exploration.ipynb` on `crop_lr_3x_500`. Restore that run's complete report directory, including all five folds' saved prediction arrays and metadata. The current notebook imports `ensure_reference_run` but does not invoke it: it assumes completed predictions exist. Run/resume the candidate in the learning-rate notebook when they are absent. Threshold scoring itself does not train and does not need a GPU.
- **Final model:** `final_model_training.ipynb` reads `REPORTS_DIR/crop_lr_3x_500/fold_0/result.json` through `fold_4/result.json` to calculate the median best epoch. The recorded values 173, 138, 207, 196 and 304 give 196. Restore those reports or deliberately set `FINAL_EPOCHS_OVERRIDE`. The notebook uses the higher rates and threshold 0.45; it fits all 24 labels with no validation-based stopping. The inference/export cells use that run's final checkpoint.

Threshold changes alter masks, IoU, Dice, precision and recall, not BCE + soft Dice loss. With `monitor='val_loss'`, threshold also does not control checkpoint selection. The library defaults are not the selected final settings: use the explicit notebook configuration. The threshold sweep peaked at 0.575 with IoU 0.7984 versus 0.7979 at 0.45; the selected cutoff remains 0.45.

## Uploaded module still appears old or missing

Check the module actually imported by the kernel:

```python
import roof_utils
print(roof_utils.__file__)
```

Place updated modules inside that project's `roof_utils/` directory. A separately uploaded `/content/final_training.py` or `/content/cross_validation.py` is not automatically a package module. Some setup cells copy a missing module from `/content`, but do not replace an existing stale copy. Restart the kernel after replacement, then rerun installation, path setup and imports in order. This also resolves cached `CVConfig` definitions missing newer arguments. Inspect `PROJECT_ROOT_OVERRIDE` when multiple project copies exist.

## Resume and experiment isolation

Keep both `MODELS_DIR/<run_name>/` and `REPORTS_DIR/<run_name>/` to resume training. Configuration, data, code and software signatures must match; otherwise use a new run name. Do not delete only `experiment.json` to bypass a mismatch. Complete prediction reports can still be used for separate threshold analysis.

Each epoch saves through a temporary file before replacing `last.pt`. An interrupt during writing can leave a partial `.tmp`; the preceding completed checkpoint is retained if present, and that interrupted epoch may repeat. Completed compatible CV folds are reused. Never run two writers against the same run directory. Separate runtimes have separate files and still require their own uploads.

## Export and download

The final notebook creates the ZIP with:

```python
archive = export_final_run(paths, RUN_NAME)
print(archive)
```

For the selected run, the default destination is `/content/reports/final_crop_lr3x_t045_196epochs_export.zip` when the project root is `/content`. A project under `/content/dida_test_task` instead uses that directory's `reports/`. Always use the printed path.

The archive includes final/resume checkpoints, run reports and prediction files, utility code and a path template. Raw/prepared data are retained separately. Creating the ZIP does **not** download it, and `/content` is temporary runtime storage, not Google Drive.

In VS Code's Colab remote file/Contents view, locate the printed archive, use its Download action, and choose your local project's `reports/` folder. A `file+.vscode-resource...` link is not a durable download or evidence that the file is local. Confirm the ZIP exists on your Mac and can be opened before deleting the runtime. If your frontend supports Colab browser downloads, this alternative requests a download, but the browser controls the local destination:

```python
from google.colab import files
files.download(str(archive))
```

Keep intermediate model/report directories too if you want to resume experiments before final export. Restoring an export for a notebook requires placing its `models/` and `reports/` contents under the configured directories, rather than leaving them only inside a nested export folder.

The completed GPU fit used PyTorch 2.11.0+cu128; local CPU checks used the pinned PyTorch 2.8.0 environment. Preserve run metadata rather than assuming these environments are interchangeable. Development CV reused validation folds for selection, and the unlabelled final images have no ground-truth accuracy scores.
