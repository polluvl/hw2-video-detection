"""Inference strategies for the final pipeline:
  - tiled (SAHI-style) prediction for small objects
  - merging boxes from several passes/models (class-wise NMS or Weighted Boxes Fusion)
  - temporal rescoring: link detections across neighbouring frames and smooth their scores

References: SAHI (Akyon et al., 2022), WBF (Solovyev et al., 2021; `ensemble-boxes` package),
Seq-NMS (Han et al., 2016) for the idea of rescoring linked boxes along a clip.
"""
import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torchvision.ops import batched_nms

from analysis import box_iou


# ----------------------------------------------------------------------------
# Merging
# ----------------------------------------------------------------------------
def merge_nms(arrays, iou_thr=0.6, max_det=500):
    """Concatenate several (N,6) arrays and run class-wise NMS."""
    arr = np.concatenate([a for a in arrays if len(a)], axis=0) if any(len(a) for a in arrays) else np.zeros((0, 6))
    if len(arr) == 0:
        return arr.astype(np.float32)
    t = torch.from_numpy(arr).float()
    keep = batched_nms(t[:, :4], t[:, 4], t[:, 5].long(), iou_thr)[:max_det]
    return arr[keep.numpy()].astype(np.float32)


def merge_wbf(arrays, width, height, iou_thr=0.55, skip_thr=0.001, weights=None, max_det=500):
    """Weighted Boxes Fusion of several (N,6) arrays for one image (pip install ensemble-boxes)."""
    from ensemble_boxes import weighted_boxes_fusion
    norm = np.array([width, height, width, height], dtype=np.float32)
    boxes = [np.clip(a[:, :4] / norm, 0, 1) if len(a) else np.zeros((0, 4)) for a in arrays]
    scores = [a[:, 4] if len(a) else np.zeros(0) for a in arrays]
    labels = [a[:, 5] if len(a) else np.zeros(0) for a in arrays]
    b, s, l = weighted_boxes_fusion(boxes, scores, labels, weights=weights,
                                    iou_thr=iou_thr, skip_box_thr=skip_thr)
    out = np.concatenate([b * norm, s[:, None], l[:, None]], axis=1).astype(np.float32)
    return out[np.argsort(-out[:, 4])][:max_det]


# ----------------------------------------------------------------------------
# Tiled prediction (SAHI-style, written by hand to keep it simple)
# ----------------------------------------------------------------------------
def make_tiles(width, height, n_cols=2, n_rows=2, overlap=0.2):
    """Grid of overlapping tiles covering the image -> list of (x1, y1, x2, y2)."""
    tw = int(np.ceil(width / (n_cols - (n_cols - 1) * overlap)))
    th = int(np.ceil(height / (n_rows - (n_rows - 1) * overlap)))
    xs = np.linspace(0, width - tw, n_cols).astype(int) if n_cols > 1 else [0]
    ys = np.linspace(0, height - th, n_rows).astype(int) if n_rows > 1 else [0]
    return [(x, y, x + tw, y + th) for y in ys for x in xs]


def predict_tiled(model, frames, imgsz, conf=0.001, iou=0.6, max_det=500, grid=(2, 2),
                  overlap=0.2, merge="nms", edge_margin=2):
    """Full image + overlapping tiles, each upscaled to `imgsz` by the model, merged per image.

    Boxes cut by a tile border (touching the edge of an inner tile) are dropped, because the
    full-image pass or the neighbouring tile sees the whole object.
    """
    preds = {}
    for r in frames.itertuples():
        img = cv2.imread(r.img_path)
        h, w = img.shape[:2]
        parts = []
        full = model.predict(img, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, verbose=False)[0].boxes
        parts.append(np.concatenate([full.xyxy.cpu().numpy(), full.conf.cpu().numpy()[:, None],
                                     full.cls.cpu().numpy()[:, None]], axis=1))
        for x1, y1, x2, y2 in make_tiles(w, h, *grid, overlap=overlap):
            res = model.predict(img[y1:y2, x1:x2], imgsz=imgsz, conf=conf, iou=iou,
                                max_det=max_det, verbose=False)[0].boxes
            b = res.xyxy.cpu().numpy()
            keep = np.ones(len(b), bool)
            if x1 > 0: keep &= b[:, 0] > edge_margin
            if y1 > 0: keep &= b[:, 1] > edge_margin
            if x2 < w: keep &= b[:, 2] < (x2 - x1) - edge_margin
            if y2 < h: keep &= b[:, 3] < (y2 - y1) - edge_margin
            b = b[keep] + np.array([x1, y1, x1, y1])
            parts.append(np.concatenate([b, res.conf.cpu().numpy()[keep, None],
                                         res.cls.cpu().numpy()[keep, None]], axis=1))
        if merge == "wbf":
            preds[r.image_key] = merge_wbf(parts, w, h, max_det=max_det)
        else:
            preds[r.image_key] = merge_nms(parts, iou_thr=iou, max_det=max_det)
    return preds


# ----------------------------------------------------------------------------
# Temporal rescoring (bonus: data temporality)
# ----------------------------------------------------------------------------
def link_frames(prev, cur, iou_thr):
    """Hungarian matching between detections of two consecutive frames (same class only).
    Returns list of (prev_idx, cur_idx)."""
    if len(prev) == 0 or len(cur) == 0:
        return []
    iou = box_iou(prev[:, :4], cur[:, :4])
    iou[prev[:, 5][:, None] != cur[:, 5][None, :]] = 0
    rows, cols = linear_sum_assignment(-iou)
    return [(i, j) for i, j in zip(rows, cols) if iou[i, j] >= iou_thr]


def temporal_rescore(frames, preds, iou_thr=0.3, min_score=0.05, weight=0.5, causal=False):
    """Simple tracker + score smoothing along each clip.

    1. In every clip, detections with score >= min_score in consecutive frames are linked
       (Hungarian on IoU, same class). Linked detections form tracks.
    2. Each detection's new score = (1-weight) * own score + weight * mean score of its track
       (causal=True: only the track's past up to this frame).
    Objects seen confidently in many frames get boosted, one-frame flickers get lowered.
    Low-score detections (< min_score) are kept unchanged.
    """
    out = {k: v.copy() for k, v in preds.items()}
    for clip, cf in frames.sort_values("frame").groupby("clip"):
        keys = cf.image_key.tolist()
        dets = [out.get(k, np.zeros((0, 6), np.float32)) for k in keys]
        idx = [np.where(d[:, 4] >= min_score)[0] if len(d) else np.zeros(0, int) for d in dets]
        track_of = [np.full(len(i), -1) for i in idx]
        tracks = []  # list of lists of (frame_pos, det_pos_in_idx)
        for t in range(len(keys)):
            cur = dets[t][idx[t]]
            if t > 0:
                prev = dets[t - 1][idx[t - 1]]
                for i, j in link_frames(prev, cur, iou_thr):
                    tid = track_of[t - 1][i]
                    track_of[t][j] = tid
                    tracks[tid].append((t, j))
            for j in np.where(track_of[t] == -1)[0]:
                track_of[t][j] = len(tracks)
                tracks.append([(t, j)])
        for track in tracks:
            scores = np.array([dets[t][idx[t][j], 4] for t, j in track])
            for n, (t, j) in enumerate(track):
                mean = scores[:n + 1].mean() if causal else scores.mean()
                own = dets[t][idx[t][j], 4]
                out[keys[t]][idx[t][j], 4] = (1 - weight) * own + weight * mean
    return out
