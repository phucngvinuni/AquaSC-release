# eval_finetune_map.py -- Wave 18 B3, take 3: does the fine-tuned SwinJSCC
# checkpoint (finetune_swinjscc_pretrained.py output) actually produce useful
# downstream mAP, not just good PSNR? PSNR success alone is not sufficient
# evidence -- JSCC-CNN's Std variant (RESULTS_OF_RECORD.md V-2) shows a
# codec can score ~33dB PSNR while still collapsing task performance to
# near-zero mAP. This scores the real YOLO detector on Site A test images,
# same conf=0.001/iou=0.5 convention as every other baseline in this project.
#
# Run: __TORCH_PY__ eval_finetune_map.py

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "swinjscc_code"))
_aq = os.path.dirname(os.path.dirname(HERE))
if _aq not in sys.path:
    sys.path.insert(0, _aq)

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from net.network import SwinJSCC
import metrics as M  # noqa: E402
import stats  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from torchmetrics.detection import MeanAveragePrecision  # noqa: E402

IMG_SWIN = 256
IMG_YOLO = 224
CKPT = os.path.join(HERE, "history", "SwinJSCC_SAandRA_finetune_rayleigh", "model_best_psnr.pth")
TEST_IMG_DIR = "__AQUASC_DATA__/yolo_fish_dataset_root/test/images"
TEST_LBL_DIR = "__AQUASC_DATA__/yolo_fish_dataset_root/test/labels"
YOLO_WEIGHTS = os.path.join(_aq, "best.pt")
PER_IMAGE_CEILING = 0.7743
POOLED_CEILING = 0.7292211


def load_boxes_norm(label_path):
    if not os.path.exists(label_path):
        return np.zeros((0, 4), dtype=np.float32)
    rows = []
    with open(label_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            _, xc, yc, w, h = parts[:5]
            xc, yc, w, h = float(xc), float(yc), float(w), float(h)
            rows.append([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2])
    return np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, 4), dtype=np.float32)


class TestDataset(Dataset):
    def __init__(self, img_dir):
        self.names = sorted(f for f in os.listdir(img_dir)
                            if f.lower().endswith((".png", ".jpg", ".jpeg")))
        self.img_dir = img_dir
        self.t_swin = transforms.Compose([transforms.Resize((IMG_SWIN, IMG_SWIN)), transforms.ToTensor()])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_dir, self.names[idx])).convert("RGB")
        return self.t_swin(img)


def build_net(device):
    class Config:
        pass
    c = Config()
    c.logger = None
    c.pass_channel = True
    c.CUDA = True
    c.device = device
    c.norm = False
    c.downsample = 4
    c.encoder_kwargs = dict(
        model="SwinJSCC_w/_SAandRA", img_size=(IMG_SWIN, IMG_SWIN), patch_size=2, in_chans=3,
        embed_dims=[128, 192, 256, 320], depths=[2, 2, 6, 2], num_heads=[4, 6, 8, 10],
        C=None, window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm, patch_norm=True,
    )
    c.decoder_kwargs = dict(
        model="SwinJSCC_w/_SAandRA", img_size=(IMG_SWIN, IMG_SWIN),
        embed_dims=[320, 256, 192, 128], depths=[2, 6, 2, 2], num_heads=[10, 8, 6, 4],
        C=None, window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm, patch_norm=True,
    )
    net_args = argparse.Namespace(model="SwinJSCC_w/_SAandRA", channel_type="rayleigh",
                                  C="96", multiple_snr="10", distortion_metric="MSE",
                                  trainset="DIV2K")
    net = SwinJSCC(net_args, c).to(device)
    sd = torch.load(CKPT, map_location=device)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    net.eval()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=int, default=64)
    ap.add_argument("--snr", type=float, nargs="+", default=[-5, 0, 5, 10, 15, 20])
    ap.add_argument("--out_dir", default=os.path.join(_aq, "f1_perimage"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(42)

    net = build_net(device)
    ds = TestDataset(TEST_IMG_DIR)
    print(f"[eval-finetune] rate={args.rate} n={len(ds)}")

    gt_cache = []
    for idx in range(len(ds)):
        stem = os.path.splitext(ds.names[idx])[0]
        boxes_norm = load_boxes_norm(os.path.join(TEST_LBL_DIR, stem + ".txt"))
        gt_cache.append(boxes_norm * IMG_YOLO)

    yolo = YOLO(YOLO_WEIGHTS)
    yolo.to(device)
    if hasattr(yolo, "fuse") and callable(yolo.fuse):
        yolo.fuse()
    yolo.eval()

    results = {}
    with torch.no_grad():
        for snr in args.snr:
            ap_list = np.zeros(len(ds), dtype=np.float32)
            pooled_metric = MeanAveragePrecision(iou_type="bbox").to(device)
            for idx in range(len(ds)):
                x = ds[idx].unsqueeze(0).to(device)
                recon256, CBR, chan_param, mse, loss_G = net(x, given_SNR=float(snr), given_rate=args.rate)
                recon256 = recon256.clamp(0.0, 1.0)
                recon = torch.nn.functional.interpolate(
                    recon256, size=(IMG_YOLO, IMG_YOLO), mode="bicubic", align_corners=False)
                recon = torch.clamp(recon, 0.0, 1.0)
                preds = M.yolo_predict(yolo, recon, conf=0.001, iou=0.5, device=device)[0]
                ap_list[idx] = stats.per_image_ap50(
                    preds["boxes"].detach().cpu().numpy(),
                    preds["scores"].detach().cpu().numpy(),
                    gt_cache[idx], iou_thr=0.5)
                gt_boxes_t = torch.as_tensor(gt_cache[idx], dtype=torch.float32, device=device)
                pooled_metric.update(
                    [{"boxes": preds["boxes"], "scores": preds["scores"],
                      "labels": torch.zeros(preds["boxes"].shape[0], dtype=torch.int64, device=device)}],
                    [{"boxes": gt_boxes_t,
                      "labels": torch.zeros(gt_boxes_t.shape[0], dtype=torch.int64, device=device)}])
            pooled_map50 = float(pooled_metric.compute()["map_50"])
            snr_tag = f"snr{snr:g}".replace("-", "m").replace(".", "p")
            out_path = os.path.join(args.out_dir, f"swinjscc_finetune_rate{args.rate}_{snr_tag}.npy")
            np.save(out_path, ap_list)
            results[snr] = dict(mean_ap50=float(ap_list.mean()), pooled_map50=pooled_map50,
                                retention_per_image=float(ap_list.mean() / PER_IMAGE_CEILING),
                                retention_pooled=pooled_map50 / POOLED_CEILING)
            print(f"[eval-finetune] snr={snr:5.1f} mean_ap50={ap_list.mean():.4f} "
                  f"retention={ap_list.mean()/PER_IMAGE_CEILING:.4f} pooled={pooled_map50:.4f} -> {out_path}")

    print("[eval-finetune] DONE", results)


if __name__ == "__main__":
    main()
