# Frozen validation folds v1

These are **fixed image-level development folds**, not an independent test set or geographically independent validation. Use the same folds for the augmentation baseline and crop/zoom comparison. No model has been trained or scored to choose these folds.

## Review and grouping

All 30 source images were inspected in a contact sheet. No repeated building or overlapping scene was confirmed. Exact image duplicates were previously absent. The duplicate label shared by 270 and 278 is an annotation error; it is not evidence that their images show the same scene. Pair 278 remains quarantined.

No location coordinates were found in the image metadata. Each image has a singleton `visual_scene_ID` group. This records the result of the available scene review, not a known geographic location. Nearby imagery, small shared regions, different dates or different viewpoints may still be related. Obtain source coordinates or source-scene IDs before claiming generalization to unseen locations.

A supplementary candidate screen compared all 435 pairs, including unlabelled images, using masked grayscale normalized cross-correlation at 96 × 96 resolution, translations and quarter-turn rotations. It excluded non-opaque pixels and the bottom eight rows to reduce logo matching, and required overlap of at least half the smaller valid area. The twelve highest-scoring pairs were inspected individually and did not reveal the same buildings. Their similarities reflected common roads, roofs and vegetation. Scores are not calibrated probabilities. This screen cannot rule out overlap at other rotations, scales, small overlap or different imaging conditions. Full rankings are saved in `overlap_candidates.json`.

Annotation consistency is a separate review: the existing `annotation_review` entries are deliberately preserved.

## Assignment

There are 24 eligible labelled pairs. Fold indices are zero-based. Each pair is a validation sample once and a training sample four times. Validation folds contain 4, 5, 5, 5 and 5 images; training sets contain 20, 19, 19, 19 and 19 images.

To spread roof coverage across folds, sort IDs by roof fraction among valid pixels, breaking ties by ID. Draw a seed-42 permutation of the five fold indices. Assign consecutive blocks of five ranked samples in alternating forward/reverse fold order. This is a simple deterministic coverage-balancing heuristic, not class-label stratification or an optimization of model performance. The frozen lists in `validation_folds.json` are authoritative.

| Fold | Validation IDs | Roof fraction over valid pixels |
|---|---|---|
| 0 | 121, 241, 328, 417 | 13.89% |
| 1 | 270, 301, 308, 314, 317 | 16.76% |
| 2 | 287, 320, 343, 345, 379 | 15.31% |
| 3 | 284, 300, 303, 315, 324 | 15.56% |
| 4 | 272, 274, 337, 381, 532 | 13.56% |

278 and the five unlabelled IDs 535, 537, 539, 551 and 553 are excluded from every training and validation fold. Unlabelled images may be inspected for overlap but do not enter supervised training or numerical validation.

## Reuse

```python
from roof_utils.config import load_paths
from roof_utils.validation_splits import load_fold
from roof_utils.roof_augmentation import RoofFoldDataset

paths = load_paths()
train_ids, val_ids = load_fold(paths, fold=0)
train_ds = RoofFoldDataset(paths.prepared_data_dir, train_ids, val_ids, split="train")
val_ds = RoofFoldDataset(paths.prepared_data_dir, train_ids, val_ids, split="val")
```

No `allow_unreviewed_groups` override is needed for these visual scene assignments. This is a software integrity check, not a claim that geography has been verified. Original IDs are split before augmentation; never split augmented variants independently.

The loader validates eligible IDs, group assignments, disjointness, complete coverage and once-only validation membership. It also verifies SHA-256 hashes of raw images/labels and prepared RGB/roof/validity files. If these change, it stops: review the changes, regenerate the prepared data if necessary, and explicitly version a new plan rather than silently reusing stale folds. File re-encoding also changes these byte hashes.

Do not change the folds in response to model results. CV used for early stopping or model selection can be optimistic; report it as development performance, with individual image and fold results. No independent test performance is available from the five unlabelled images.
