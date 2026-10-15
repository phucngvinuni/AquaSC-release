# Data, annotations and splits

Raw data are **not redistributed** in this repository. What is shipped: split manifests, per-file lists, and the code that rebuilds the YOLO-format roots.

## Sites
| | Site A (VinFish) | Site B (FishDet-NB2) |
|---|---|---|
| Content | RGB 4,935 frames (+ 1,065 infrared frames, not used for the headline numbers) | 4,998 frames, 21 videos, 1024x576 |
| Task | single class `fish`, YOLO-format boxes | single class `fish`, YOLO-format boxes |
| Split (video-disjoint, seed 42) | train 4,066 / valid 437 / test 432 | train 4,056 / valid 487 / test 455 |
| Obtain | **[to be filled in by the authors: download URL / access request]** | **[to be filled in by the authors]** |
| Terms | **[to be confirmed by the data owners before public release]** | research use only; **[to be confirmed]** |

## Annotation format and preprocessing
* One `labels/<image>.txt` per image; each line `class x_center y_center width height`, normalised to [0, 1] (Ultralytics YOLO format). `class` is always 0.
* Images are resized to 224x224 (bicubic) for the codec and the detector (`datasets.py`, `dataset_folder.py`); the detector scores on the 224x224 reconstruction against the boxes of the original frame (`experiments/matched5/evaluate.py`). Patch size 8 (784 tokens).
* Expected roots (siblings of this repository, or set with `scripts/install_paths.sh --data-root`):
```
<DATA_ROOT>/yolo_fish_dataset_root/{train,valid,test}/{images,labels}      # Site A
<DATA_ROOT>/yolo_fishdet_nb2_vd/{train,valid,test}/{images,labels}         # Site B
```
* Rebuild Site A from the raw archive with `data_tools/splits.py` (it applies the video-disjoint manifest). **Do not re-split randomly**: a random frame split leaks near-duplicate frames between train and test (42.7 % of test frames; `splits/leakage_stats_*.json`).

## Split lists shipped here
| File | Content |
|---|---|
| `artifacts/splits/per_file_list_siteA.txt` | `file, modality, split, video_id, frame_index` for all 6,000 Site A frames (RGB and IR) |
| `artifacts/splits/per_file_list_siteB.txt` | same for Site B (4,998 data rows) |
| `splits/manifest.json` | Site A video-disjoint manifest (seed 42, ratios 0.8/0.1/0.1) used by every reported number; `splits/manifest_801010.json`, `splits/manifest_groupkfold5.json`: alternative manifests |
| `artifacts/splits/siteb_manifest.json` | Site B manifest (21 videos, split counts) |
| `splits/leakage_stats_{random,video_disjoint}.json` | leakage audit of the random vs. video-disjoint split |
| `channel_bank.py`, `data_tools/gen_channel_bank.py` | seeded Rayleigh `h` and noise bank (see `artifacts/README.md`) |
