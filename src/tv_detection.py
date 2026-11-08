"""Faster R-CNN and RetinaNet with torchvision, trained with one shared loop
so both detectors follow exactly the same protocol (data, schedule, eval).

Based on the torchvision detection tutorial / reference scripts:
https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html
"""
import json
import math
import time
import types
from functools import partial

import cv2
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import (fasterrcnn_resnet50_fpn_v2,
                                          retinanet_resnet50_fpn_v2)
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection import roi_heads as tv_roi_heads
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import generalized_box_iou_loss, sigmoid_focal_loss

from common import (NUM_CLASSES, evaluate, mask_ignored_regions, read_ignore,
                    read_labels)


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
class DroneDataset(Dataset):
    """Returns (image tensor in [0,1], target dict). Labels are shifted by +1
    because torchvision reserves label 0 for background.

    Resizing and ImageNet normalisation are done inside the model
    (GeneralizedRCNNTransform), so here we only do augmentation.
    """

    def __init__(self, frames, train=False, mask_ignore=True):
        self.frames = frames.reset_index(drop=True)
        self.train = train
        self.mask_ignore = mask_ignore

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        row = self.frames.iloc[idx]
        image = cv2.cvtColor(cv2.imread(row.img_path), cv2.COLOR_BGR2RGB)
        boxes, classes = read_labels(row.label_path, row.width, row.height)

        if self.train:
            if self.mask_ignore:
                image = mask_ignored_regions(image, read_ignore(row.ignore_path, row.width, row.height))
            # horizontal flip: drone footage has no preferred left/right
            if np.random.rand() < 0.5:
                image = image[:, ::-1].copy()
                boxes = boxes.copy()
                boxes[:, [0, 2]] = row.width - boxes[:, [2, 0]]
            # mild brightness/contrast jitter (clips differ in lighting, some are at night)
            if np.random.rand() < 0.5:
                alpha = np.random.uniform(0.8, 1.2)
                beta = np.random.uniform(-20, 20)
                image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        keep = (boxes[:, 2] - boxes[:, 0] >= 1) & (boxes[:, 3] - boxes[:, 1] >= 1)
        target = {
            "boxes": torch.as_tensor(boxes[keep], dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(classes[keep] + 1, dtype=torch.int64),
        }
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return image, target, row.image_key


def collate(batch):
    return tuple(zip(*batch))


# ----------------------------------------------------------------------------
# Faster R-CNN
# ----------------------------------------------------------------------------
def giou_fastrcnn_loss(class_logits, box_regression, labels, regression_targets, giou_weight=10.0):
    """Replacement for torchvision's fastrcnn_loss: GIoU instead of Smooth L1.

    torchvision only gives us the encoded deltas (dx, dy, dw, dh), not the proposals.
    IoU does not change if both boxes are shifted/scaled the same way, so we decode
    prediction and target relative to the SAME reference box (unit box at origin)
    and compute GIoU on those -> identical to GIoU on the real boxes.
    Loss weight 10 follows MMDetection's faster-rcnn GIoU config.
    """
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)
    classification_loss = torch.nn.functional.cross_entropy(class_logits, labels)

    pos = torch.where(labels > 0)[0]
    labels_pos = labels[pos]
    n, _ = class_logits.shape
    box_regression = box_regression.reshape(n, box_regression.size(-1) // 4, 4)
    pred_deltas = box_regression[pos, labels_pos]
    target_deltas = regression_targets[pos]

    def decode(d):  # box coder weights (10, 10, 5, 5) as in torchvision
        dx, dy = d[:, 0] / 10.0, d[:, 1] / 10.0
        dw = torch.clamp(d[:, 2] / 5.0, max=math.log(1000.0 / 16))
        dh = torch.clamp(d[:, 3] / 5.0, max=math.log(1000.0 / 16))
        w, h = torch.exp(dw), torch.exp(dh)
        return torch.stack([dx - w / 2, dy - h / 2, dx + w / 2, dy + h / 2], dim=1)

    box_loss = generalized_box_iou_loss(decode(pred_deltas.float()), decode(target_deltas.float()),
                                        reduction="sum")
    box_loss = giou_weight * box_loss / labels.numel()
    return classification_loss, box_loss


_ORIGINAL_FASTRCNN_LOSS = tv_roi_heads.fastrcnn_loss


def build_faster_rcnn(cfg, pretrained=True):
    """cfg keys (all optional): min_size, max_size, rpn_pre_nms_test, rpn_post_nms_test,
    rpn_post_nms_train, roi_batch_size, roi_pos_fraction, box_loss ('smooth_l1'|'giou'),
    anchor_sizes."""
    kwargs = dict(
        min_size=cfg.get("min_size", 800), max_size=cfg.get("max_size", 1333),
        rpn_pre_nms_top_n_train=cfg.get("rpn_pre_nms_train", 2000),
        rpn_post_nms_top_n_train=cfg.get("rpn_post_nms_train", 2000),
        rpn_pre_nms_top_n_test=cfg.get("rpn_pre_nms_test", 1000),
        rpn_post_nms_top_n_test=cfg.get("rpn_post_nms_test", 1000),
        box_batch_size_per_image=cfg.get("roi_batch_size", 512),
        box_positive_fraction=cfg.get("roi_pos_fraction", 0.25),
        box_detections_per_img=cfg.get("detections_per_img", 500),  # scenes are dense
        box_score_thresh=0.001,  # keep low-score boxes, mAP needs the full ranking
    )
    if cfg.get("anchor_sizes"):
        sizes = tuple((s,) for s in cfg["anchor_sizes"])
        kwargs["rpn_anchor_generator"] = AnchorGenerator(sizes, ((0.5, 1.0, 2.0),) * len(sizes))
    model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT" if pretrained else None,
                                       weights_backbone=None, **kwargs)
    # new classification/regression layer for our 8 classes (+ background)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES + 1)

    # box regression loss of the second stage (module-level function -> patch it)
    if cfg.get("box_loss", "smooth_l1") == "giou":
        tv_roi_heads.fastrcnn_loss = giou_fastrcnn_loss
    else:
        tv_roi_heads.fastrcnn_loss = _ORIGINAL_FASTRCNN_LOSS
    return model


# ----------------------------------------------------------------------------
# RetinaNet
# ----------------------------------------------------------------------------
def _focal_compute_loss(self, targets, head_outputs, matched_idxs):
    """Copy of torchvision RetinaNetClassificationHead.compute_loss with
    configurable focal-loss alpha and gamma (torchvision hard-codes 0.25 / 2)."""
    losses = []
    cls_logits = head_outputs["cls_logits"]
    for targets_per_image, cls_logits_per_image, matched_idxs_per_image in zip(targets, cls_logits, matched_idxs):
        foreground_idxs_per_image = matched_idxs_per_image >= 0
        num_foreground = foreground_idxs_per_image.sum()
        gt_classes_target = torch.zeros_like(cls_logits_per_image)
        gt_classes_target[
            foreground_idxs_per_image,
            targets_per_image["labels"][matched_idxs_per_image[foreground_idxs_per_image]],
        ] = 1.0
        valid_idxs_per_image = matched_idxs_per_image != self.BETWEEN_THRESHOLDS
        losses.append(
            sigmoid_focal_loss(cls_logits_per_image[valid_idxs_per_image],
                               gt_classes_target[valid_idxs_per_image],
                               alpha=self.focal_alpha, gamma=self.focal_gamma,
                               reduction="sum") / max(1, num_foreground))
    return sum(losses) / len(targets)


def build_retinanet(cfg, pretrained=True):
    """cfg keys: min_size, max_size, anchor_sizes (5 tuples of 3 sizes, P3..P7),
    aspect_ratios (3 values, h/w), fg_iou, bg_iou, focal_alpha, focal_gamma."""
    kwargs = dict(min_size=cfg.get("min_size", 800), max_size=cfg.get("max_size", 1333),
                  detections_per_img=cfg.get("detections_per_img", 500),
                  score_thresh=0.01, topk_candidates=cfg.get("topk_candidates", 1000))
    model = retinanet_resnet50_fpn_v2(weights="DEFAULT" if pretrained else None,
                                      weights_backbone=None, **kwargs)
    if cfg.get("anchor_sizes"):  # the builder creates its own generator, so replace it afterwards
        sizes = tuple(tuple(s) for s in cfg["anchor_sizes"])
        ratios = (tuple(cfg.get("aspect_ratios", (0.5, 1.0, 2.0))),) * len(sizes)
        model.anchor_generator = AnchorGenerator(sizes, ratios)
    num_anchors = model.anchor_generator.num_anchors_per_location()[0]
    assert num_anchors == 9, "keep 3 sizes x 3 ratios so the pretrained heads fit"

    # replace only the last classification conv (COCO 91 classes -> ours + background slot)
    cls_head = model.head.classification_head
    num_classes = NUM_CLASSES + 1
    cls_head.cls_logits = torch.nn.Conv2d(256, num_anchors * num_classes, kernel_size=3, padding=1)
    torch.nn.init.normal_(cls_head.cls_logits.weight, std=0.01)
    torch.nn.init.constant_(cls_head.cls_logits.bias, -math.log((1 - 0.01) / 0.01))
    cls_head.num_classes = num_classes

    # focal loss parameters
    cls_head.focal_alpha = cfg.get("focal_alpha", 0.25)
    cls_head.focal_gamma = cfg.get("focal_gamma", 2.0)
    cls_head.compute_loss = types.MethodType(_focal_compute_loss, cls_head)

    # IoU thresholds for positive / negative anchors (default 0.5 / 0.4)
    model.proposal_matcher = det_utils.Matcher(cfg.get("fg_iou", 0.5), cfg.get("bg_iou", 0.4),
                                               allow_low_quality_matches=True)
    return model


# ----------------------------------------------------------------------------
# Training / prediction
# ----------------------------------------------------------------------------
@torch.no_grad()
def predict(model, frames, device, batch_size=4, num_workers=2):
    """-> (preds dict, ms per image). Uses original (unmasked) images."""
    model.eval()
    loader = DataLoader(DroneDataset(frames, train=False), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers, collate_fn=collate)
    preds, total_time, n_images = {}, 0.0, 0
    for images, _, keys in loader:
        images = [img.to(device) for img in images]
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_time += time.perf_counter() - t0
        n_images += len(images)
        for key, out in zip(keys, outputs):
            arr = torch.cat([out["boxes"], out["scores"][:, None],
                             (out["labels"][:, None] - 1).float()], dim=1)
            preds[key] = arr.cpu().numpy()
    return preds, 1000 * total_time / max(n_images, 1)


def train_detector(model, train_frames, val_frames, cfg, device, out_path):
    """Plain SGD training loop with warmup + cosine LR and mixed precision.
    Evaluates val mAP after every epoch and keeps the best checkpoint.
    Returns history (list of dicts) for plotting learning curves."""
    epochs = cfg.get("epochs", 8)
    loader = DataLoader(DroneDataset(train_frames, train=True, mask_ignore=cfg.get("mask_ignore", True)),
                        batch_size=cfg.get("batch_size", 4), shuffle=True,
                        num_workers=cfg.get("num_workers", 2), collate_fn=collate, drop_last=True)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=cfg.get("lr", 0.01), momentum=0.9, weight_decay=1e-4)
    total_steps = epochs * len(loader)
    warmup = min(cfg.get("warmup_iters", 300), max(total_steps // 5, 1))

    def lr_lambda(step):  # linear warmup, then cosine decay to 1% of base lr
        if step < warmup:
            return 0.001 + (1 - 0.001) * step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    model.to(device)

    history, best_map, step = [], -1.0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        t0, running = time.time(), {}
        for images, targets, _ in loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
            if not torch.isfinite(loss):
                print("non-finite loss, skipping batch", {k: float(v) for k, v in loss_dict.items()})
                optimizer.zero_grad()
                continue
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            for k, v in loss_dict.items():
                running[k] = running.get(k, 0.0) + v.item()
            if step % cfg.get("print_every", 200) == 0:
                print(f"  step {step}/{total_steps} loss {loss.item():.3f} lr {scheduler.get_last_lr()[0]:.5f}")

        train_time = time.time() - t0
        preds, ms_per_img = predict(model, val_frames, device)
        metrics = evaluate(val_frames, preds)
        record = {"epoch": epoch, "train_min": round(train_time / 60, 2),
                  **{k: round(v / len(loader), 4) for k, v in running.items()}, **metrics}
        history.append(record)
        print(f"epoch {epoch}: mAP {metrics['mAP']:.4f} AP50 {metrics['AP50']:.4f} "
              f"AP_s {metrics['AP_small']:.4f} ({train_time / 60:.1f} min)")
        if metrics["mAP"] > best_map:
            best_map = metrics["mAP"]
            torch.save(model.state_dict(), out_path)
        with open(str(out_path) + ".history.json", "w") as f:
            json.dump(history, f, indent=1)
    model.load_state_dict(torch.load(out_path, map_location=device))
    return history
