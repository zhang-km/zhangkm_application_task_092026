# Training with the VS Code Colab kernel

1. Open `baseline_cv.ipynb` in VS Code and select your Colab GPU kernel.
2. Run the package installation cell, then the project-path setup cell.
3. The setup finds the project and reads its `.env` (relative defaults are created only if absent). No Google Drive mount or archive is needed.
4. If the remote kernel cannot see the project, use VS Code Explorer → Upload to Colab for `roof_utils/`, `pyproject.toml`, `splits/`, `data/` and `data_preparation_outputs/`, preserving their relative layout. Set `PROJECT_ROOT_OVERRIDE` to that runtime directory. Selecting a kernel does not automatically share your Mac filesystem.
5. Run the remaining cells for the baseline, then run `crop_cv.ipynb` using matching code/software/settings.

Models and reports use MODELS_DIR and REPORTS_DIR from `.env`. Files on a temporary Colab disk must be downloaded before the runtime is deleted. Rerunning resumes only if the saved model and report folders remain accessible. Changed configuration/code/software requires a new run name.

CPU testing covered two epochs, disconnect/resume, checkpoint loading, saved predictions and completed-fold reuse. Full GPU training has not run. Development CV scores use validation for checkpoint selection; geographic independence is unverified. Image 278 stays excluded.

Official guide: https://github.com/googlecolab/colab-vscode/wiki/User-Guide

## Validation-loss monitoring update

The notebooks now use `monitor="val_loss"`, `min_epochs=30`, `patience=20` and `max_epochs=100`, with new run names `baseline_cv_loss` and `crop_cv_loss`. Minimum epochs prohibit early stopping before epoch 30; patience accrues after warm-up, including before that floor. Checkpoints minimize full-fold masked BCE + Dice loss. IoU stays at threshold 0.5. Logs include validation loss and predicted roof fraction.

Upload the updated `roof_utils/cross_validation.py` to the same location on your Colab runtime and restart the notebook kernel to clear cached imports. Then run the updated notebooks from the beginning. Do not reuse the old experiment names or checkpoints. Run all folds under the same recipe for comparisons.

## Configurable threshold

`CVConfig(threshold=0.45)` controls validation metrics, predicted roof fractions and saved metrics for new training runs. The library default remains 0.5; the notebooks explicitly use 0.45 and new `_t045` run names. Threshold does not change BCE + Dice loss. `show_predictions` uses the run configuration unless a threshold override is provided.

For completed runs, use `evaluate_saved_predictions(paths, "baseline_cv_loss", threshold=0.45)` and `show_predictions(paths, "baseline_cv_loss", threshold=0.45)`. This requires no training and stores separate reports in `threshold_evaluations/threshold_0.45/`. Original metrics and weights remain unchanged; historic training curves are not recalculated. Threshold 0.45 is exploratory because it was chosen after inspecting validation results.

## Final all-label model

Run `final_model_training.ipynb` after uploading `roof_utils/final_training.py` to the runtime and restarting the kernel. It uses all 24 usable labels, crop probability 0.3, threshold 0.45, and a fixed 300 epochs. The shared CVConfig fields monitor/min_epochs/patience are inactive because there is no validation split. It saves final.pt and resumable last.pt, predicts the five unlabelled images, and offers an export ZIP. CPU pipeline/resume/inference checks passed; the full GPU fit has not been run locally.

## Threshold and learning-rate exploration

Upload `roof_utils/hyperparameter_review.py`, restart the kernel, and run `threshold_exploration.ipynb` first. It reads completed crop out-of-fold predictions with no retraining. Then `learning_rate_comparison.ipynb` runs two matched five-fold crop-enabled experiments, changing only encoder/decoder fine-tuning rates from 1e-5/1e-4 to 3e-5/3e-4. Both use threshold 0.45, the same seed/folds/stopping rules, and max_epochs=500. Existing final weights are preserved. The comparison requires ten new fold fits to keep source/configuration identical; no GPU experiments were launched by the assistant. These are exploratory development analyses.

The threshold notebook now includes `ensure_reference_run`: reuse complete reference predictions, or train/resume the crop-enabled reference when absent/incomplete. Default reference uses 500 maximum epochs, patience 20, val_loss monitoring, crop probability 0.3 and threshold 0.45. A GPU is needed only if training runs. Configuration/data mismatch or missing exports in completed folds produces an actionable error without deleting results. Upload the updated hyperparameter_review.py and restart the kernel before using the new cell.
