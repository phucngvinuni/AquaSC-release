# baselinea_train.py — BASELINE-A (Wave 5, plan P0-5 / §9.1)
# Train JSCC-CNN (Std / BBW) on the HONEST video-disjoint Site-A split.
#
# Reuses the repo's model.py (DeepJSCC / ratio2filtersize) and channel.py (Channel).
# Two variants:
#   std : unweighted MSE (canonical DeepJSCC; matches trainw.py semantics)
#   bbw : bounding-box-weighted L1 (inside 5 / outside 1; matches train.py semantics)
# Train channel = Rayleigh @ 10 dB (matches AquaSC's analog proxy).  Eval channel swapped at test time.
#
# NOTE: the task brief said "train.py (Std), trainw.py (BBW)".  That is inverted w.r.t. the
# actual code: train.py is the BBW (weighted L1) script and trainw.py is the Std (MSE) script.

import os
import sys
import time
import json
import argparse
import csv

import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np

from model import DeepJSCC, ratio2filtersize
from utils import set_seed, view_model_param, get_psnr

IMG = 224


class FishDataset(Dataset):
    def __init__(self, image_dir, label_dir, img_size=IMG):
        self.img_size = img_size
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_paths = sorted(
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        )
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image)
        boxes = torch.empty((0, 4), dtype=torch.float32)
        labels = torch.empty(0, dtype=torch.int64)
        if self.label_dir:
            label_path = os.path.join(
                self.label_dir,
                os.path.splitext(os.path.basename(img_path))[0] + ".txt")
            if os.path.exists(label_path):
                bx, bl = [], []
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        try:
                            cls = int(parts[0])
                            cx, cy, w, h = map(float, parts[1:])
                            x1 = np.clip(cx - w / 2, 0.0, 1.0)
                            y1 = np.clip(cy - h / 2, 0.0, 1.0)
                            x2 = np.clip(cx + w / 2, 0.0, 1.0)
                            y2 = np.clip(cy + h / 2, 0.0, 1.0)
                            if x2 > x1 and y2 > y1:
                                bx.append([x1, y1, x2, y2])
                                bl.append(cls)
                        except ValueError:
                            continue
                if bx:
                    boxes = torch.as_tensor(bx, dtype=torch.float32)
                    boxes[:, [0, 2]] *= self.img_size
                    boxes[:, [1, 3]] *= self.img_size
                    labels = torch.as_tensor(bl, dtype=torch.int64)
        return image_tensor, boxes


def collate(batch):
    images = torch.stack([b[0] for b in batch], 0)
    boxes = [b[1] for b in batch]
    return images, boxes


@torch.no_grad()
def validate(model, loader, device, bbw, seed=1234):
    model.eval()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    psnrs = []
    losses = []
    for images, boxes in loader:
        images = images.to(device)
        boxes_dev = [b.to(device) for b in boxes] if bbw else None
        out = model(images)
        losses.append(model.loss(out, images, boxes_dev).item())
        for i in range(images.shape[0]):
            psnrs.append(get_psnr(out[i], images[i], max_val=1.0).item())
    model.train()
    return (float(np.mean(psnrs)) if psnrs else float('nan'),
            float(np.mean(losses)) if losses else float('nan'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True, choices=['std', 'bbw'])
    ap.add_argument('--ratio', default='1/12')
    ap.add_argument('--snr', type=float, default=10.0)
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight_decay', type=float, default=5e-4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--train_image_dir', required=True)
    ap.add_argument('--train_label_dir', required=True)
    ap.add_argument('--val_image_dir', required=True)
    ap.add_argument('--val_label_dir', required=True)
    ap.add_argument('--out', default='./output_jscc_baselinea')
    ap.add_argument('--save_every', type=int, default=20)
    ap.add_argument('--resume', type=str, default=None,
                    help="Site B fine-tune (2026-09-16): load a pretrained checkpoint's "
                         "state_dict before training (full fine-tune, nothing frozen, "
                         "matching AquaSC's S-3 / ROI-JSCC's Site B fine-tune convention).")
    ap.add_argument('--loss_type', choices=['auto', 'L1', 'MSE'], default='auto',
                    help="2026-09-23: override base_loss_type independent of --variant's "
                         "weighting, to test e.g. unweighted L1 (variant=std --loss_type L1) "
                         "-- a combo the original std/bbw split never covered. 'auto' (default) "
                         "preserves the original variant->loss mapping exactly.")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    from fractions import Fraction
    ratio = float(Fraction(args.ratio))

    # c from the repo's own mapping (ratio = channel-uses / source-samples = k/n).
    dummy = torch.rand(1, 3, IMG, IMG, device=device)
    c = ratio2filtersize(dummy, ratio)
    uses_per_source_sample = ratio
    uses_per_pixel = ratio * 3.0  # per RGB pixel (paper's "uses/px")

    train_ds = FishDataset(args.train_image_dir, args.train_label_dir)
    val_ds = FishDataset(args.val_image_dir, args.val_label_dir)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            collate_fn=collate)

    bbw = (args.variant == 'bbw')
    loss_type = ('L1' if bbw else 'MSE') if args.loss_type == 'auto' else args.loss_type
    model = DeepJSCC(
        c=c, channel_type='Rayleigh', snr=args.snr, P_encoder=1.0,
        use_weighted_loss=bbw, inside_box_w=5.0, outside_box_w=1.0,
        base_loss_type=loss_type,
    ).to(device)
    if args.resume:
        sd = torch.load(args.resume, map_location=device)
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
        elif isinstance(sd, dict) and 'model' in sd:
            sd = sd['model']
        model.load_state_dict(sd, strict=True)
        print(f"[train] resumed full state_dict from {args.resume}")

    tag = f"{args.variant}_c{c}_snr{int(args.snr)}_ratio{ratio:.4f}"
    if args.loss_type != 'auto':
        tag += f"_{loss_type.lower()}"
    run_dir = os.path.join(args.out, tag)
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    config = {
        'variant': args.variant, 'ratio': ratio, 'c': c, 'snr': args.snr,
        'uses_per_source_sample': uses_per_source_sample,
        'uses_per_pixel': uses_per_pixel,
        'channel': 'Rayleigh', 'img': IMG, 'epochs': args.epochs,
        'batch_size': args.batch_size, 'lr': args.lr,
        'weight_decay': args.weight_decay, 'seed': args.seed,
        'n_train': len(train_ds), 'n_val': len(val_ds),
        'n_params': int(view_model_param(model)),
        'base_loss_type': loss_type,
        'inside_box_w': 5.0 if bbw else None,
        'outside_box_w': 1.0 if bbw else None,
    }
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    print(json.dumps(config, indent=2), flush=True)

    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    csv_path = os.path.join(run_dir, 'train_log.csv')
    csvf = open(csv_path, 'w', newline='')
    csvw = csv.writer(csvf)
    csvw.writerow(['epoch', 'train_loss', 'val_psnr_dB', 'val_loss', 'epoch_sec'])
    csvf.flush()

    best_val_psnr = -float('inf')
    best_val_loss = float('inf')
    t0 = time.time()
    for epoch in range(args.epochs):
        ep_start = time.time()
        model.train()
        tot = 0.0
        n = 0
        for images, boxes in train_loader:
            images = images.to(device)
            if bbw:
                boxes_dev = [b.to(device) for b in boxes]
            else:
                boxes_dev = None
            opt.zero_grad()
            out = model(images)
            loss = model.loss(out, images, boxes_dev)
            loss.backward()
            opt.step()
            tot += loss.item()
            n += 1
        train_loss = tot / max(1, n)
        vp, vl = validate(model, val_loader, device, bbw)
        ep_sec = time.time() - ep_start
        csvw.writerow([epoch, train_loss, vp, vl, ep_sec])
        csvf.flush()
        print(f"[{tag}] ep {epoch}/{args.epochs-1} train_loss={train_loss:.6f} "
              f"val_psnr={vp:.2f} dB val_loss={vl:.6f} ({ep_sec:.1f}s)", flush=True)
        if vp > best_val_psnr:
            best_val_psnr = vp
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best_psnr.pkl'))
            print(f"    -> new best val PSNR {vp:.2f} dB", flush=True)
        if vl < best_val_loss:
            best_val_loss = vl
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best_loss.pkl'))
            print(f"    -> new best val loss {vl:.6f}", flush=True)
        if (epoch + 1) % args.save_every == 0:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f'epoch_{epoch}.pkl'))
        if time.time() - t0 > 22 * 3600:
            print("time budget exceeded", flush=True)
            break
    torch.save(model.state_dict(), os.path.join(ckpt_dir, 'final.pkl'))
    csvf.close()
    print(f"done. best_val_psnr={best_val_psnr:.2f} dB  best_val_loss={best_val_loss:.6f}  total={time.time()-t0:.1f}s", flush=True)


if __name__ == '__main__':
    main()
