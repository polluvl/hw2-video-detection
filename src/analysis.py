"""Simple error analysis shared by all detector notebooks.

At a fixed score threshold every ground-truth box gets one outcome:
  TP          - matched by a detection of the right class with IoU >= 0.5
  wrong_class - a detection overlaps it with IoU >= 0.5 but has another class
  poor_loc    - best same-class detection has 0.1 <= IoU < 0.5 (found but badly localised)
  missed      - nothing overlapping
and every unmatched detection is a false positive ("background" if it overlaps no GT,
"duplicate" if it overlaps an already-matched GT of the same class).
Detections inside ignored regions are dropped, like in the evaluator.
Idea follows TIDE (Bolya et al., 2020), simplified to keep it easy to explain.
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import CLASS_NAMES, read_ignore, read_labels

SIZE_BINS = [0, 16, 32, 96, np.inf]
SIZE_LABELS = ["tiny <16", "small 16-32", "medium 32-96", "large >96"]


def box_iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def inside_ignore(det_boxes, ignore_boxes, thr=0.5):
    """True if more than `thr` of the detection's area lies in an ignored region."""
    if len(ignore_boxes) == 0 or len(det_boxes) == 0:
        return np.zeros(len(det_boxes), bool)
    x1 = np.maximum(det_boxes[:, None, 0], ignore_boxes[None, :, 0])
    y1 = np.maximum(det_boxes[:, None, 1], ignore_boxes[None, :, 1])
    x2 = np.minimum(det_boxes[:, None, 2], ignore_boxes[None, :, 2])
    y2 = np.minimum(det_boxes[:, None, 3], ignore_boxes[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (det_boxes[:, 2] - det_boxes[:, 0]) * (det_boxes[:, 3] - det_boxes[:, 1]) + 1e-9
    return (inter / area[:, None]).max(axis=1) > thr


def analyse_frame(row, pred, score_thr=0.3):
    """-> (gt_outcomes DataFrame, fp DataFrame) for a single frame."""
    gt_boxes, gt_cls = read_labels(row.label_path, row.width, row.height)
    pred = pred[pred[:, 4] >= score_thr] if len(pred) else np.zeros((0, 6))
    pred = pred[~inside_ignore(pred[:, :4], read_ignore(row.ignore_path, row.width, row.height))]
    pred = pred[np.argsort(-pred[:, 4])]
    iou = box_iou(pred[:, :4], gt_boxes)  # (D, G)

    gt_outcome = np.array(["missed"] * len(gt_boxes), dtype=object)
    gt_matched = np.zeros(len(gt_boxes), bool)
    det_type = []
    # greedy matching by score, same class first (like COCO)
    for d in range(len(pred)):
        same = (gt_cls == int(pred[d, 5])) & ~gt_matched
        cand = np.where(same, iou[d], -1)
        g = int(np.argmax(cand)) if len(gt_boxes) else -1
        if g >= 0 and cand[g] >= 0.5:
            gt_matched[g] = True
            gt_outcome[g] = "TP"
            det_type.append("TP")
        elif len(gt_boxes) and (np.where(gt_cls == int(pred[d, 5]), iou[d], 0) >= 0.5).any():
            det_type.append("duplicate")
        elif len(gt_boxes) and iou[d].max() >= 0.5:
            det_type.append("wrong_class")
        elif len(gt_boxes) and iou[d].max() >= 0.1:
            det_type.append("poor_loc")
        else:
            det_type.append("background")

    # label non-TP ground truth with the closest kind of near-miss
    for g in np.where(~gt_matched)[0]:
        if len(pred) == 0:
            continue
        col = iou[:, g]
        same_cls = pred[:, 5].astype(int) == gt_cls[g]
        if (col[~same_cls] >= 0.5).any():
            gt_outcome[g] = "wrong_class"
        elif (col[same_cls] >= 0.1).any():
            gt_outcome[g] = "poor_loc"

    size = np.sqrt((gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1]))
    gts = pd.DataFrame({"image_key": row.image_key, "clip": row.clip, "cls": gt_cls,
                        "size": size, "outcome": gt_outcome})
    fps = pd.DataFrame({"image_key": row.image_key, "type": det_type,
                        "cls": pred[:, 5].astype(int) if len(pred) else []})
    return gts, fps


def error_breakdown(frames, preds, score_thr=0.3):
    gts, fps = [], []
    for row in frames.itertuples():
        g, f = analyse_frame(row, preds.get(row.image_key, np.zeros((0, 6))), score_thr)
        gts.append(g)
        fps.append(f)
    gts = pd.concat(gts, ignore_index=True)
    fps = pd.concat(fps, ignore_index=True)
    gts["size_bin"] = pd.cut(gts["size"], SIZE_BINS, labels=SIZE_LABELS)
    gts["class"] = gts.cls.map(dict(enumerate(CLASS_NAMES)))
    return gts, fps


def summarise_errors(gts, fps):
    """Two small tables: GT outcome shares by size, and FP types."""
    by_size = pd.crosstab(gts.size_bin, gts.outcome, normalize="index").round(3)
    by_size["n_gt"] = gts.size_bin.value_counts().reindex(by_size.index)
    by_class = pd.crosstab(gts["class"], gts.outcome, normalize="index").round(3)
    fp_types = fps[fps.type != "TP"].type.value_counts()
    return by_size, by_class, fp_types


def recall_by_size(gts):
    return (gts.assign(hit=gts.outcome == "TP").groupby("size_bin", observed=False).hit.mean().round(3))


def draw_errors(ax, row, pred, score_thr=0.3, title=None):
    """Green = TP-matched GT, red = missed GT, orange = wrong class / poor localisation,
    yellow dashed = false-positive detections."""
    img = cv2.cvtColor(cv2.imread(row.img_path), cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    gts, fps = analyse_frame(row, pred, score_thr)
    gt_boxes, _ = read_labels(row.label_path, row.width, row.height)
    colour = {"TP": "lime", "missed": "red", "wrong_class": "orange", "poor_loc": "orange"}
    for (x1, y1, x2, y2), out in zip(gt_boxes, gts.outcome):
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=colour[out], lw=1))
    p = pred[pred[:, 4] >= score_thr] if len(pred) else pred
    p = p[~inside_ignore(p[:, :4], read_ignore(row.ignore_path, row.width, row.height))]
    p = p[np.argsort(-p[:, 4])]
    for (x1, y1, x2, y2, s, c), t in zip(p, fps.type):
        if t != "TP":
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="yellow", lw=1, ls="--"))
    for x1, y1, x2, y2 in read_ignore(row.ignore_path, row.width, row.height):
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="white", lw=1, hatch="//"))
    counts = gts.outcome.value_counts().to_dict()
    ax.set_title(title or f"{row.image_key}  {counts}", fontsize=9)
    ax.axis("off")


def worst_frames(gts, frames, k=3):
    """Frames with the lowest recall (at least 10 GT boxes)."""
    rec = gts.assign(hit=gts.outcome == "TP").groupby("image_key").agg(recall=("hit", "mean"), n=("hit", "size"))
    rec = rec[rec.n >= 10].sort_values("recall")
    keys = []
    for key in rec.index:  # one frame per clip so the cases are different
        clip = frames.set_index("image_key").loc[key, "clip"]
        if clip not in [frames.set_index("image_key").loc[k2, "clip"] for k2 in keys]:
            keys.append(key)
        if len(keys) == k:
            break
    return frames.set_index("image_key").loc[keys].reset_index()
