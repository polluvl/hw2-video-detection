# HW2 – Multiple Object Video Detection (VisDrone-VID)

> Code part of the homework. The report itself goes where the course template says (GitHub Pages).

## Layout

```
src/
  common.py         data listing, label/ignore reading, COCO evaluator with ignored regions, submission
  tv_detection.py   torchvision Faster R-CNN + RetinaNet, shared training loop
  yolo_utils.py     Ultralytics dataset export (ignored regions masked), prediction, curves
  analysis.py       error breakdown (TP / missed / wrong class / poor localisation / FP), recall by size
  inference.py      tiling (SAHI-style), NMS/WBF merging, temporal rescoring
notebooks/          01_eda … 06_final_pipeline (.ipynb)
tools/
  make_fake_data.py tiny fake dataset for CPU smoke tests
  results_tables.py  results.csv -> markdown tables for the report
requirements.txt
```

## Protocol (identical for every model)

| item | setting |
|---|---|
| split | organisers' split, **by sequence**: 56 train clips / 7 val clips (no shared clip, asserted in 01) |
| frames | train every 10th frame (~2.4k imgs), val every 5th (~570 imgs); 06 also uses all val frames |
| ignored regions | training: painted with the image mean colour; evaluation: `iscrowd=1`, copied for every class |
| metric | COCO mAP@[.5:.95], AP50, AP75, AP_small/medium/large, maxDets = 500 (Kaggle scores ≤ 500 boxes) |
| seed | 42 everywhere (`seed_everything`, Ultralytics `seed=42`) |

All results are appended to `results.csv` (one row per experiment) by `log_result()`.

## How to run (Kaggle, GPU T4/P100, Internet on)

1. Add the competition data as input; set `REPO` in the first cell of each notebook and add a
   Kaggle Secret `GITHUB_TOKEN` (read-only fine-grained token for this repo) while the repo is private.
2. Run in order — later notebooks read files written by earlier ones from `/kaggle/working`:
   `01_eda` (CPU) → `02_faster_rcnn` → `03_retinanet` → `04_yolo_anchor_free` → `05_rtdetr_query` → `06_final_pipeline`.
   Kaggle sessions are separate, so save `/kaggle/working` as a dataset (or "Save Version → output")
   and attach it to the next notebook, or copy `results.csv`, `anchors.json`, `preds_val_*.npz`,
   `*.pt` and `ultralytics/*/weights/best.pt` between sessions.
3. `06_final_pipeline` writes `submission.csv`.

Rough GPU time per notebook (T4, estimated before running – replace with the measured times):
02 ≈ 4–5 h (3 runs), 03 ≈ 5–6 h (4 runs), 04 ≈ 5–6 h (4 runs), 05 ≈ 6–8 h (3 runs), 06 ≈ 6–8 h.

## Smoke test without a GPU

```bash
pip install -r requirements.txt nbconvert ipykernel
python tools/make_fake_data.py data_fake
cd notebooks
for nb in 01_eda 02_faster_rcnn 03_retinanet 04_yolo_anchor_free 05_rtdetr_query 06_final_pipeline; do
  DATA_ROOT=../data_fake SMOKE=1 jupyter nbconvert --to notebook --execute $nb.ipynb --output /tmp/$nb.ipynb || break
done
```
`SMOKE=1` uses tiny images, 1 epoch and random initialisation – it only checks that the code runs.

## Team and ownership

| part | owner |
|---|---|
| repo setup, `src/common.py`, `src/tv_detection.py`, `src/analysis.py`, 01 EDA, 02 Faster R-CNN, 03 RetinaNet | <Person A> |
| `src/yolo_utils.py`, `src/inference.py`, 04 YOLOv8, 05 RT-DETR, 06 final pipeline, Kaggle submissions | <Person B> |
| report: each writes the sections of their own experiments; summary/conclusion together | both |

## AI assistance
Code was drafted with the help of an AI assistant (Claude) and then run, checked and adapted by us,
as allowed by the course rules. The report text is written by us.

## External code and ideas used
torchvision detection models and tutorial; Ultralytics (YOLOv8, RT-DETR); pycocotools;
`ensemble-boxes` (WBF); ideas from SAHI (tiling), TIDE (error types), Seq-NMS (temporal rescoring),
YOLOv2 (k-means anchors), MMDetection's Faster R-CNN GIoU config (loss weight 10).
