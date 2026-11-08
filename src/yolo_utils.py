"""Helpers for Ultralytics models (YOLO and RT-DETR): dataset export, prediction
in our common format, and reading the training curves.
Ultralytics docs: https://docs.ultralytics.com/
"""
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from common import CLASS_NAMES, mask_ignored_regions, read_ignore


def export_yolo_split(frames, out_dir, mask_ignore=True):
    """Write images/ + labels/ in Ultralytics layout.

    Frames without ignored regions are symlinked (no copy). Frames with ignored
    regions get a copy with those regions painted in the mean colour, so YOLO does
    not learn that unlabelled crowds are background.
    """
    out_dir = Path(out_dir)
    img_dir, lab_dir = out_dir / "images", out_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    n_masked = 0
    for r in frames.itertuples():
        dst = img_dir / f"{r.image_key}.jpg"
        if not dst.exists():
            ignore = read_ignore(r.ignore_path, r.width, r.height)
            if mask_ignore and len(ignore):
                img = mask_ignored_regions(cv2.imread(r.img_path), ignore)
                cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                n_masked += 1
            else:
                os.symlink(os.path.abspath(r.img_path), dst)
        src_lab = Path(r.label_path)
        if src_lab.exists():
            shutil.copy(src_lab, lab_dir / f"{r.image_key}.txt")
        else:
            (lab_dir / f"{r.image_key}.txt").write_text("")
    return n_masked


def write_data_yaml(path, train_dir, val_dir):
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    Path(path).write_text(f"train: {Path(train_dir).resolve()}/images\n"
                          f"val: {Path(val_dir).resolve()}/images\nnames:\n{names}\n")
    return str(path)


def predict_ultralytics(model, frames, imgsz, conf=0.001, iou=0.6, max_det=500,
                        augment=False, batch=8, img_paths=None):
    """-> (preds dict in common format, ms per image incl. pre/post-processing)."""
    paths = img_paths if img_paths is not None else frames.img_path.tolist()
    keys = frames.image_key.tolist()
    preds, total_ms = {}, 0.0
    for start in range(0, len(paths), batch):
        results = model.predict(paths[start:start + batch], imgsz=imgsz, conf=conf, iou=iou,
                                max_det=max_det, augment=augment, verbose=False)
        for key, res in zip(keys[start:start + batch], results):
            b = res.boxes
            arr = np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None],
                                  b.cls.cpu().numpy()[:, None]], axis=1)
            preds[key] = arr.astype(np.float32)
            total_ms += sum(res.speed.values())
    return preds, total_ms / max(len(paths), 1)


def read_curves(run_dir):
    """Ultralytics results.csv -> DataFrame with epoch, mAP, mAP50 (internal val, masked images)."""
    df = pd.read_csv(Path(run_dir) / "results.csv")
    df.columns = [c.strip() for c in df.columns]
    out = pd.DataFrame({"epoch": df["epoch"], "mAP": df["metrics/mAP50-95(B)"], "mAP50": df["metrics/mAP50(B)"]})
    if "time" in df.columns:  # cumulative seconds, present in recent Ultralytics versions
        out["minutes"] = df["time"] / 60
    return out
