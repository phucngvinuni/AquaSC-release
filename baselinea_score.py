# baselinea_score.py — score JSCC-CNN reconstructions with the AquaSC detector.
# Run from AquaSC-D2DB/ (so `import metrics` / `import utils` resolve to the AquaSC harness).
# Uses metrics.MetricsAccumulator (torchmetrics mAP + P/R/F1 + PSNR/SSIM + in-box) so the
# numbers are directly comparable to the AquaSC sweeps.

import os
import sys
import json
import argparse

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from ultralytics import YOLO

import metrics as M

IMG = 224


def load_boxes(label_path):
    boxes = []
    labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                try:
                    cls = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:])
                    x1 = np.clip(cx - w / 2, 0, 1)
                    y1 = np.clip(cy - h / 2, 0, 1)
                    x2 = np.clip(cx + w / 2, 0, 1)
                    y2 = np.clip(cy + h / 2, 0, 1)
                    if x2 > x1 and y2 > y1:
                        boxes.append([x1, y1, x2, y2])
                        labels.append(cls)
                except ValueError:
                    continue
    if boxes:
        b = torch.as_tensor(boxes, dtype=torch.float32)
        b[:, [0, 2]] *= IMG
        b[:, [1, 3]] *= IMG
        return b, torch.as_tensor(labels, dtype=torch.int64)
    return torch.empty((0, 4)), torch.empty(0, dtype=torch.int64)


class OrigRecon(Dataset):
    def __init__(self, orig_dir, recon_dir, label_dir):
        self.orig_dir = orig_dir
        self.recon_dir = recon_dir
        self.label_dir = label_dir
        self.names = sorted(f for f in os.listdir(orig_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg')))
        self.t = transforms.ToTensor()

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        stem = os.path.splitext(name)[0]
        orig = self.t(Image.open(os.path.join(self.orig_dir, name)).convert("RGB"))
        recon = self.t(Image.open(os.path.join(self.recon_dir, stem + '.png')).convert("RGB"))
        boxes, labels = load_boxes(os.path.join(self.label_dir, stem + '.txt'))
        return orig, recon, {"boxes": boxes, "labels": labels}


def collate(batch):
    orig = torch.stack([b[0] for b in batch], 0)
    recon = torch.stack([b[1] for b in batch], 0)
    gt = [b[2] for b in batch]
    return orig, recon, gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--yolo', default='best.pt')
    ap.add_argument('--orig_dir', required=True)
    ap.add_argument('--recon_dir', required=True)
    ap.add_argument('--label_dir', required=True)
    ap.add_argument('--conf', type=float, default=0.001)
    ap.add_argument('--iou', type=float, default=0.5)
    ap.add_argument('--out_csv', required=True)
    ap.add_argument('--batch_size', type=int, default=16)
    args = ap.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    yolo = YOLO(args.yolo)
    yolo.to(device)
    if hasattr(yolo, 'fuse') and callable(yolo.fuse):
        yolo.fuse()
    yolo.eval()

    acc = M.MetricsAccumulator(device=str(device), conf_thres=args.conf,
                               iou_thres=args.iou, compute_lpips=False)

    ds = OrigRecon(args.orig_dir, args.recon_dir, args.label_dir)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2,
                        collate_fn=collate)

    with torch.no_grad():
        for orig, recon, gt in loader:
            orig_dev = orig.to(device)
            recon_dev = recon.to(device)
            preds_orig = M.yolo_predict(yolo, orig_dev, conf=args.conf, iou=args.iou, device=device)
            preds_recon = M.yolo_predict(yolo, recon_dev, conf=args.conf, iou=args.iou, device=device)
            gt_boxes_list = [{"boxes": g["boxes"], "labels": g["labels"]} for g in gt]
            acc.accumulate(orig_dev, recon_dev, gt_boxes_list, preds_orig, preds_recon)

    res = acc.compute()
    res['conf'] = args.conf
    res['n_images'] = len(ds)
    res['orig_dir'] = args.orig_dir
    res['recon_dir'] = args.recon_dir
    res['retention_vs_canonical_07292'] = res['map_50_recon'] / 0.7292

    with open(args.out_csv.replace('.csv', '.json'), 'w') as f:
        json.dump({k: (float(v) if isinstance(v, (int, float)) else v) for k, v in res.items()},
                  f, indent=2)
    print(json.dumps(res, indent=2))
    print(f"retention (vs measured orig {res['map_50_orig']:.4f}): {res['retention_50']:.4f}")


if __name__ == '__main__':
    main()
