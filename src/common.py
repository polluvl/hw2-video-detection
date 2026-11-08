"""Shared helpers for HW2 (VisDrone video detection).

Everything that more than one notebook needs lives here:
  - finding the data and listing frames (with frame subsampling)
  - reading YOLO labels and ignored regions
  - a COCO-style evaluator that treats ignored regions as "don't care"
  - writing the Kaggle submission file

Internal prediction format used everywhere:
    preds[image_key] = np.ndarray of shape (N, 6) -> [x1, y1, x2, y2, score, class_id]
in ORIGINAL image pixel coordinates, class_id in 0..7.
image_key = "<clip>_<frame stem>", e.g. "clip_001_0000001" (same as Kaggle image_id).
"""
import contextlib
import io
import json
import os
import random
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

CLASS_NAMES = ["person", "bicycle", "car", "van", "truck", "tricycle", "bus", "motor"]
NUM_CLASSES = len(CLASS_NAMES)

# SMOKE=1 -> tiny runs on CPU, used only to check that the code works end to end
SMOKE = os.environ.get("SMOKE", "0") == "1"


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def find_data_root():
    """DATA_ROOT env var > Kaggle input folder > ./data"""
    if os.environ.get("DATA_ROOT"):
        return Path(os.environ["DATA_ROOT"])
    for cand in glob("/kaggle/input/*"):
        if os.path.isdir(os.path.join(cand, "train")) and os.path.isdir(os.path.join(cand, "val")):
            return Path(cand)
    return Path("data")


def work_dir():
    """Writable folder for outputs (checkpoints, predictions, results)."""
    out = Path("/kaggle/working") if os.path.isdir("/kaggle/working") else Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    return out.resolve()  # absolute: Ultralytics puts relative project paths under runs/


# ----------------------------------------------------------------------------
# Listing frames
# ----------------------------------------------------------------------------
def list_frames(split_dir, step=1):
    """Table with one row per frame of a labelled split (train/ or val/).

    step > 1 keeps every `step`-th frame of each clip (neighbouring frames are
    almost identical, so this saves GPU time without losing much variety).
    """
    split_dir = Path(split_dir)
    rows = []
    for clip_dir in sorted((split_dir / "images").iterdir()):
        if not clip_dir.is_dir():
            continue
        clip = clip_dir.name
        frames = sorted(clip_dir.glob("*.jpg"))
        if not frames:
            continue
        # all frames in a clip have the same size -> open only the first one
        width, height = Image.open(frames[0]).size
        for i, img_path in enumerate(frames):
            if i % step != 0:
                continue
            stem = img_path.stem
            rows.append({
                "image_key": f"{clip}_{stem}",
                "clip": clip,
                "frame": int(stem),
                "img_path": str(img_path),
                "label_path": str(split_dir / "labels" / clip / f"{stem}.txt"),
                "ignore_path": str(split_dir / "ignore" / clip / f"{stem}.txt"),
                "width": width,
                "height": height,
            })
    return pd.DataFrame(rows)


def list_test_frames(data_root):
    df = pd.read_csv(Path(data_root) / "test_images.csv")
    df["image_key"] = df["image_id"]
    df["img_path"] = [str(Path(data_root) / f) for f in df["file_name"]]
    return df


# ----------------------------------------------------------------------------
# Reading annotations
# ----------------------------------------------------------------------------
def _read_txt(path, n_cols):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return np.zeros((0, n_cols), dtype=np.float32)
    arr = np.loadtxt(path, dtype=np.float32, ndmin=2)
    return arr.reshape(-1, n_cols)


def cxcywh_norm_to_xyxy(arr, width, height):
    cx, cy, w, h = arr[:, 0] * width, arr[:, 1] * height, arr[:, 2] * width, arr[:, 3] * height
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def read_labels(label_path, width, height):
    """-> boxes (N,4) xyxy pixels, classes (N,) int"""
    arr = _read_txt(label_path, 5)
    boxes = cxcywh_norm_to_xyxy(arr[:, 1:], width, height)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    return boxes, arr[:, 0].astype(np.int64)


def read_ignore(ignore_path, width, height):
    """-> ignored regions (M,4) xyxy pixels"""
    return cxcywh_norm_to_xyxy(_read_txt(ignore_path, 4), width, height)


def mask_ignored_regions(image, ignore_boxes):
    """Paint ignored regions with the mean colour of the image (numpy HxWx3).

    Used for TRAINING only: the model then never sees unlabelled crowds as
    "background", so it is not punished for finding objects there.
    """
    if len(ignore_boxes) == 0:
        return image
    image = image.copy()
    mean_colour = image.reshape(-1, 3).mean(axis=0).astype(image.dtype)
    for x1, y1, x2, y2 in ignore_boxes.astype(int):
        image[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)] = mean_colour
    return image


# ----------------------------------------------------------------------------
# Evaluation (COCO mAP with ignored regions)
# ----------------------------------------------------------------------------
def build_coco_gt(frames):
    """COCO ground truth dict from a frames table.

    Ignored regions become iscrowd=1 boxes. pycocotools matches crowd boxes per
    category, so each region is copied once for every class -> a detection of
    any class inside the region is ignored (neither TP nor FP).
    """
    images, annotations = [], []
    ann_id = 1
    for img_id, row in enumerate(frames.itertuples(), start=1):
        images.append({"id": img_id, "file_name": row.image_key,
                       "width": int(row.width), "height": int(row.height)})
        boxes, classes = read_labels(row.label_path, row.width, row.height)
        for (x1, y1, x2, y2), c in zip(boxes, classes):
            w, h = float(x2 - x1), float(y2 - y1)
            if w <= 0 or h <= 0:
                continue
            annotations.append({"id": ann_id, "image_id": img_id, "category_id": int(c) + 1,
                                "bbox": [float(x1), float(y1), w, h], "area": w * h, "iscrowd": 0})
            ann_id += 1
        for x1, y1, x2, y2 in read_ignore(row.ignore_path, row.width, row.height):
            w, h = float(x2 - x1), float(y2 - y1)
            for c in range(NUM_CLASSES):
                annotations.append({"id": ann_id, "image_id": img_id, "category_id": c + 1,
                                    "bbox": [float(x1), float(y1), w, h], "area": w * h, "iscrowd": 1})
                ann_id += 1
    categories = [{"id": i + 1, "name": n} for i, n in enumerate(CLASS_NAMES)]
    return {"images": images, "annotations": annotations, "categories": categories}


def evaluate(frames, preds, max_dets=500, per_class=False, verbose=False):
    """COCO metrics for predictions on the given frames.

    max_dets=500 because scenes are dense and Kaggle scores up to 500 boxes per image
    (the pycocotools default of 100 would cut off true positives in crowded frames).
    Returns dict: mAP, AP50, AP75, AP_small, AP_medium, AP_large, AR (and per-class AP).
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt_dict = build_coco_gt(frames)
    key_to_id = {img["file_name"]: img["id"] for img in gt_dict["images"]}
    dets = []
    for key, arr in preds.items():
        if key not in key_to_id or len(arr) == 0:
            continue
        arr = arr[np.argsort(-arr[:, 4])][:max_dets]
        for x1, y1, x2, y2, s, c in arr:
            dets.append({"image_id": key_to_id[key], "category_id": int(c) + 1,
                         "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(s)})
    empty = {k: 0.0 for k in ["mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR"]}
    if not dets:
        return empty

    quiet = contextlib.redirect_stdout(io.StringIO()) if not verbose else contextlib.nullcontext()
    with quiet:
        coco_gt = COCO()
        coco_gt.dataset = gt_dict
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(dets)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.params.maxDets = [1, 100, max_dets]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    s = ev.stats
    result = {"mAP": s[0], "AP50": s[1], "AP75": s[2], "AP_small": s[3],
              "AP_medium": s[4], "AP_large": s[5], "AR": s[8]}
    if per_class:
        # precision: [T, R, K, A, M] -> IoU thr, recall, class, area, maxDets
        prec = ev.eval["precision"][:, :, :, 0, -1]
        for k, name in enumerate(CLASS_NAMES):
            p = prec[:, :, k]
            result[f"AP_{name}"] = float(p[p > -1].mean()) if (p > -1).any() else float("nan")
    return {k: round(float(v), 4) for k, v in result.items()}


def evaluate_per_clip(frames, preds):
    """mAP for each validation clip separately -> shows how stable a score is."""
    rows = []
    for clip, clip_frames in frames.groupby("clip"):
        m = evaluate(clip_frames, preds)
        rows.append({"clip": clip, "n_frames": len(clip_frames), "mAP": m["mAP"], "AP50": m["AP50"]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Experiment log + submission
# ----------------------------------------------------------------------------
def log_result(name, metrics, extra=None, path=None):
    """Append one experiment's metrics to results.csv (one table for the whole project)."""
    path = Path(path or work_dir() / "results.csv")
    row = {"experiment": name, **metrics, **(extra or {})}
    if path.exists():
        df = pd.read_csv(path)
        df = df[df.experiment != name]  # re-running an experiment replaces its row
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(path, index=False)
    return df


def save_preds(preds, path):
    np.savez_compressed(path, **{k: v.astype(np.float32) for k, v in preds.items()})


def load_preds(path):
    data = np.load(path)
    return {k: data[k] for k in data.files}


def write_submission(test_frames, preds, out_path, max_dets=500):
    """Kaggle format: 'class score x y w h ...' with (x,y) = top-left corner in pixels."""
    lines = []
    for key in test_frames["image_key"]:
        arr = preds.get(key, np.zeros((0, 6)))
        arr = arr[np.argsort(-arr[:, 4])][:max_dets] if len(arr) else arr
        parts = []
        for x1, y1, x2, y2, s, c in arr:
            parts.append(f"{int(c)} {s:.4f} {x1:.1f} {y1:.1f} {x2 - x1:.1f} {y2 - y1:.1f}")
        lines.append({"image_id": key, "PredictionString": " ".join(parts)})
    sub = pd.DataFrame(lines)
    sub.to_csv(out_path, index=False)
    return sub


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
