# HW2 – Multiple Object Video Detection (VisDrone-VID)

Team: **Polina Lukashenko**, **Sofia Mandryk** · UCU AI 2026 · [Kaggle competition](https://www.kaggle.com/competitions/multiple-object-video-detection-hw-2-ucu-ai-2026)

The report is in [`report.md`](report.md) and published with GitHub Pages:
**https://polluvl.github.io/hw2-video-detection/**

**Final submission:** `07_ensemble` – Weighted Boxes Fusion of Faster R-CNN (02) and the YOLOv8 pipeline (06).
Val mAP@[.5:.95] 0.271, public leaderboard 0.2525. Second selected submission: Faster R-CNN alone (public 0.228).

## Repository layout

```
src/
  common.py         frame listing, label/ignore reading, COCO evaluator with ignored regions, submission writer
  tv_detection.py   torchvision Faster R-CNN + RetinaNet, shared training loop
  yolo_utils.py     Ultralytics dataset export (ignored regions masked), prediction, training curves
  analysis.py       error breakdown (TP / missed / wrong class / poor localisation / FP), recall by size
  inference.py      tiling (SAHI-style), NMS/WBF merging, temporal rescoring
notebooks/          run versions from Kaggle, with outputs
  01_eda.ipynb                 data analysis, anchor statistics (CPU)
  02_faster_rcnn.ipynb         two-stage: base / GIoU loss / RoI sampling, proposal sweep, failure cases
  02b_frcnn_submit.ipynb       test predictions + Kaggle submission from the best Faster R-CNN
  03_retinanet.ipynb           one-stage anchor-based: base / data anchors / IoU thresholds / focal loss
  04_yolo_anchor_free.ipynb    one-stage anchor-free YOLOv8s: 640 / 1024 / P2 head / box-loss gain
  05_rtdetr_query.ipynb        query-based RT-DETR-L: 300 / 500 queries / learning rate
  06_final_pipeline.ipynb      more data, bigger model, inference strategies, temporal rescoring
  07_ensemble.ipynb            WBF ensemble of 02 + 06 (final submission, CPU)
images/             figures and result tables used in the report
tools/
  make_fake_data.py tiny fake dataset for CPU smoke tests
  results_tables.py results.csv -> markdown tables
requirements.txt
```

## Protocol (identical for every model)

| item | setting |
|---|---|
| split | organisers' split **by sequence**: 56 train clips / 7 val clips (no shared clip, asserted in 01) |
| frames | train: every 10th frame (2,446 images); val: every 5th frame (571 images). 06 also evaluates on all 2,846 val frames |
| ignored regions | training: painted with the image mean colour; evaluation: `iscrowd=1`, copied for every class |
| metric | COCO mAP@[.5:.95], AP50, AP75, AP_small/medium/large, AR, maxDets = 500 (Kaggle scores ≤ 500 boxes) |
| budget | torchvision models: 5 epochs, SGD lr 0.01, batch 4, cosine schedule; Ultralytics models: 15 epochs |
| seed | 42 everywhere (`seed_everything`, Ultralytics `seed=42`) |

Every experiment is appended to `results.csv` in the notebook output (one row per run, with config, speed and training time).

## How to reproduce (Kaggle)

Each notebook starts with a setup cell that clones this repository (for `src/`), installs `requirements.txt`
and finds the competition data automatically.

1. Open the competition page → **Code → New Notebook** (this attaches the data) → **File → Import Notebook**
   → paste the raw GitHub link of the notebook, e.g.
   `https://raw.githubusercontent.com/polluvl/hw2-video-detection/main/notebooks/02_faster_rcnn.ipynb`.
2. Session options: **GPU T4 x2** (01 and 07 run on CPU), **Internet on**.
3. Attach the outputs of earlier notebooks as inputs (**Add Input → Your Work**), as listed below.
4. **Save Version → Save & Run All**.

| notebook | inputs besides the competition data | GPU time (T4, measured) | main outputs |
|---|---|---|---|
| 01_eda | – | CPU, ~15 min | `anchors.json`, EDA figures |
| 02_faster_rcnn | – | ~2.5 h (3 × 5 epochs) | `frcnn_*.pt`, `preds_val_frcnn_*.npz`, `results.csv` |
| 02b_frcnn_submit | 02 | ~20 min | `preds_test_frcnn_base.npz`, `submission.csv` |
| 03_retinanet | 01, 02 | ~3.4 h (4 × 5 epochs) | `retina_*.pt`, `results.csv` |
| 04_yolo_anchor_free | – | ~2.5 h (4 × 15 epochs) | `ultralytics/yolo_*/weights/best.pt`, `results.csv`, `submission.csv` |
| 05_rtdetr_query | 04 | ~2.5 h (3 × 15 epochs) | `preds_val_rtdetr_*.npz`, `results.csv` |
| 06_final_pipeline | 04, 05 | ~5.8 h | `preds_valall_pipe_*.npz`, `preds_test_final.npz`, `results.csv`, `submission.csv` |
| 07_ensemble | 02, 02b, 06 | CPU, ~10 min | `ensemble_results.csv`, **final `submission.csv`** |

To reproduce only the final submission from saved predictions, run 07 with the outputs of 02, 02b and 06 attached.

**Known issue:** the last cell of `06_final_pipeline` (a format check) failed on Kaggle because
`sample_submission.csv` is not part of the mounted competition data. All results and `submission.csv`
were written before that cell. The corrected check (against `test_images.csv`) is in `07_ensemble`.

### Dependencies

Kaggle "Latest" Python 3.13 image with its preinstalled CUDA build of PyTorch (torch 2.11, torchvision);
the rest is pinned in `requirements.txt` (ultralytics 8.4.192, pycocotools, ensemble-boxes 1.0.10, …).

### Smoke test without a GPU

```bash
pip install -r requirements.txt nbconvert ipykernel
python tools/make_fake_data.py data_fake
cd notebooks
DATA_ROOT=../data_fake SMOKE=1 jupyter nbconvert --to notebook --execute 02_faster_rcnn.ipynb --output /tmp/02.ipynb
```
`SMOKE=1` uses tiny images, 1 epoch and random initialisation – it only checks that the code runs.

## Team and ownership

| part | owner |
|---|---|
| repo setup, `src/common.py`, `src/tv_detection.py`, `src/analysis.py`, 01 EDA, 02/02b Faster R-CNN, 03 RetinaNet, 07 ensemble | Polina Lukashenko |
| `src/yolo_utils.py`, `src/inference.py`, 04 YOLOv8, 05 RT-DETR, 06 final pipeline | Sofia Mandryk |
| report: each wrote the sections of their own experiments; summary and conclusions together | both |

## AI assistance

Code was drafted with the help of an AI assistant (Claude) and then run, checked and adapted by us,
as allowed by the course rules. The report text is written by us.

## External code and ideas used

torchvision detection models and tutorial; Ultralytics (YOLOv8, RT-DETR); pycocotools;
`ensemble-boxes` (Weighted Boxes Fusion); ideas from SAHI (tiling), TIDE (error types), Seq-NMS (temporal rescoring),
YOLOv2 (k-means anchors), MMDetection's Faster R-CNN GIoU config (loss weight 10). Full references are in the report.
