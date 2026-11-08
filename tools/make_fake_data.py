"""Creates a tiny fake dataset with the same folder layout as the competition data.

Only used to check that every notebook runs end to end on CPU before spending
GPU time (python tools/make_fake_data.py data_fake, then DATA_ROOT=data_fake SMOKE=1).
The "objects" are coloured rectangles, so the numbers it produces mean nothing.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
COLOURS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
           (255, 0, 255), (0, 255, 255), (128, 128, 255), (255, 128, 0)]


def make_clip(root, split, clip, n_frames, w, h, labelled=True):
    n_obj = rng.integers(5, 25)
    objs = []
    for _ in range(n_obj):
        c = int(rng.integers(0, 8))
        bw, bh = rng.integers(6, 40), rng.integers(6, 40)
        objs.append([c, rng.uniform(0, w - bw), rng.uniform(0, h - bh), bw, bh,
                     rng.uniform(-2, 2), rng.uniform(-2, 2)])
    rows = []
    for f in range(1, n_frames + 1):
        img = np.full((h, w, 3), 90, np.uint8) + rng.integers(0, 20, (h, w, 3), dtype=np.uint8)
        lines = []
        for o in objs:
            o[1] = float(np.clip(o[1] + o[5], 0, w - o[3]))
            o[2] = float(np.clip(o[2] + o[6], 0, h - o[4]))
            x, y, bw, bh = o[1], o[2], o[3], o[4]
            cv2.rectangle(img, (int(x), int(y)), (int(x + bw), int(y + bh)), COLOURS[o[0]], -1)
            lines.append(f"{o[0]} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}")
        stem = f"{f:07d}"
        img_dir = root / split / "images" / clip
        img_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(img_dir / f"{stem}.jpg"), img)
        if labelled:
            lab_dir = root / split / "labels" / clip
            lab_dir.mkdir(parents=True, exist_ok=True)
            (lab_dir / f"{stem}.txt").write_text("\n".join(lines))
            if f % 2 == 0:  # an ignored region on every other frame
                ign_dir = root / split / "ignore" / clip
                ign_dir.mkdir(parents=True, exist_ok=True)
                (ign_dir / f"{stem}.txt").write_text("0.9 0.9 0.15 0.15")
        rows.append({"image_id": f"{clip}_{stem}", "file_name": f"{split}/images/{clip}/{stem}.jpg",
                     "clip": clip, "frame": f, "width": w, "height": h})
    return rows


def main(out="data_fake"):
    root = Path(out)
    for i in range(4):
        make_clip(root, "train", f"train_clip_{i}", 12, 320, 192)
    for i in range(2):
        make_clip(root, "val", f"val_clip_{i}", 8, 320, 192)
    test_rows = []
    for i in range(2):
        test_rows += make_clip(root, "test", f"clip_{i + 1:03d}", 5, 288, 160, labelled=False)
    pd.DataFrame(test_rows).to_csv(root / "test_images.csv", index=False)
    pd.DataFrame({"image_id": [r["image_id"] for r in test_rows], "PredictionString": ""}).to_csv(
        root / "sample_submission.csv", index=False)
    for split in ["train", "val"]:
        (root / split / "data.yaml").write_text(
            "path: .\ntrain: images\nval: images\nnames: [person, bicycle, car, van, truck, tricycle, bus, motor]\n")
    print("fake data written to", root)


if __name__ == "__main__":
    main(*sys.argv[1:])
