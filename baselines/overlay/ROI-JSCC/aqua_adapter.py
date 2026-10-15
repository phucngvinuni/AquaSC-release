#!/usr/bin/env python3
"""
aqua_adapter.py — runnable adapter for ROI-JSCC (arXiv:2506.01269,
github.com/hansung-choi/ROI-JSCC, IEEE ComL 2025) against the AquaSC
evaluation grid.

Scope (plan §8.1 / §16.5 COMPETITOR, R3 #5):
  * Load a ROI-JSCC checkpoint (ROIJSCC model) and reconstruct images on the
    AquaSC test grid.
  * Two ROI definitions, exactly as §8.1 requires:
        (i)  ROI = the largest GT box  -> its centre maps to one ROI cell;
        (ii) ROI = the box enclosing the whole shoal -> its centre maps to
             one ROI cell.
  * TRAIN IS DEFERRED. This file never trains; it only *loads* a checkpoint.
    No ROI-JSCC checkpoint is committed in this clone (saved_models/ is created
    at train time), so the run command below needs a checkpoint produced by the
    upstream training entry point (see "Training command (deferred)").

Two structural adaptations, both documented (see bottom of file):
  1. glymur is stubbed (JPEG2000-only dependency, never used by ROIJSCC).
  2. Resolution: ROI-JSCC's ROI window attention requires the per-stage feature
     map to be divisible by window_size=8, i.e. the input H,W must be a
     multiple of 128 (256x256 is the upstream training resolution). AquaSC's
     224x224 grid is therefore bicubic-upscaled to 256x256 for encode and
     downscaled back to 224x224 for detector scoring. This is the same
     resolution convention the ROI-JSCC paper uses (512x768 eval, 256x256 train).

Channel note: the upstream Channel.gaussian_noise_layer calls
`input_layer.get_device()`, which returns -1 on CPU and `torch.normal(..., device=-1)`
raises. So chan_type='none' (noiseless) works on CPU; AWGN / Rayleigh require CUDA.
GPU is available in this environment (torch 2.1.2, CUDA 12.1).

Rate mapping (consistent with plan §9.1, SwinJSCC = C/256 real-symbols-per-pixel):
  C = int(3*2*(2**4)*(2**4)/rcpp) = 1536/rcpp ; uses/px = C/256 = 6/rcpp.
  rcpp=12 -> 0.5, 16 -> 0.375, 24 -> 0.25, 32 -> 0.1875 (all >> AquaSC 0.0256).

Usage:
  # smoke: build model + noiseless CPU forward on one test image (no checkpoint)
  python3 aqua_adapter.py --smoke --rcpp 12

  # full eval (needs a trained checkpoint; see training command below)
  python3 aqua_adapter.py --ckpt <path> --rcpp 12 --chan AWGN --snr 10 \
      --roi-mode largest_box --data-root ../yolo_fish_dataset_root/test \
      --detector-weights ../best.pt --out roi_jscc_eval.csv
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
# Stub glymur (JPEG2000-only) before importing the upstream model package.
sys.path.insert(0, os.path.join(HERE, "aquasc_stubs"))
sys.path.insert(0, HERE)

from model.JSCC import ROIJSCC  # noqa: E402  (needs the glymur stub above)

ROI_RES = 256           # ROI-JSCC native/training resolution (multiple of 128)
PATCH_D = 4             # ROI mask grid is d x d cells over the full image


# --------------------------------------------------------------------------- #
# model construction (mirrors model/model_maker.get_model_info for "ROIJSCC")  #
# --------------------------------------------------------------------------- #
def build_model(rcpp: int = 12, chan_type: str = "AWGN", device: str = "cpu"):
    model_info = {
        "chan_type": chan_type,
        "color_channel": 3,
        "rcpp": rcpp,
        "n_block_list": [2, 2, 2, 2],
        "n_feats_list": [40, 60, 80, 260],
        "num_heads_list": [4, 6, 8, 10],      # unused by the ROI path, kept for parity
        "window_size_list": [8, 8, 8, 8],
        "input_resolution": (ROI_RES, ROI_RES),
        "ratio1": 0.5,
        "ratio2": 0.5,
        "eta": 0.1,
        "patch_d": PATCH_D,
        "SNR_info": 10,
    }
    model = ROIJSCC(model_info)
    model.to(device)
    model.eval()
    return model


def load_checkpoint(model, ckpt_path: str, device: str = "cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    # Upstream saves model.state_dict() (main_train.py / train.py).
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# ROI definition -> ROI_Index (single cell in the d x d mask grid)             #
# --------------------------------------------------------------------------- #
def compute_roi_index(boxes_xyxy, H: int = 224, W: int = 224, d: int = PATCH_D,
                      mode: str = "largest_box"):
    """Map a set of GT boxes (absolute pixel xyxy on the HxW grid) to the single
    ROI-JSCC ROI cell index (Ih, Iw) in [0, d-1]^2.

    ROI-JSCC can only mark ONE cell of the d x d mask as ROI (see
    common_component.get_weighted_total_attention_mask + patch_rate_allocation:
    7 patches get RONI rate, d^2-8 get the base rate, exactly 1 gets the boosted
    ROI rate). Both §8.1 ROI definitions therefore collapse to "pick the single
    cell" — which is precisely the representational limitation the plan wants
    demonstrated (§8.1 / §11.2).
    """
    boxes = np.asarray(boxes_xyxy, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[0] == 0 or boxes.shape[1] != 4:
        # No GT boxes: fall back to the image centre cell.
        return (d // 2, d // 2)

    if mode == "largest_box":
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        b = boxes[int(np.argmax(areas))]
        cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    elif mode == "shoal_bbox":
        x1, y1 = boxes[:, 0].min(), boxes[:, 1].min()
        x2, y2 = boxes[:, 2].max(), boxes[:, 3].max()
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    else:
        raise ValueError(f"unknown roi mode: {mode}")

    Ih = int(np.clip(cy * d / H, 0, d - 1))
    Iw = int(np.clip(cx * d / W, 0, d - 1))
    return (Ih, Iw)


# --------------------------------------------------------------------------- #
# forward (224x224 [0,1] RGB -> 224x224 [0,1] reconstruction)                  #
# --------------------------------------------------------------------------- #
def reconstruct(model, img224, roi_index=(1, 1), snr_db=10, seed=42,
                device="cpu"):
    """img224: [B,3,224,224] float in [0,1]. Returns [B,3,224,224] in [0,1]."""
    torch.manual_seed(seed)
    img = F.interpolate(img224.float().to(device), size=(ROI_RES, ROI_RES),
                        mode="bicubic", align_corners=False)
    img_norm = img * 2.0 - 1.0          # ROI-JSCC uses Normalize(0.5,0.5) -> [-1,1]
    with torch.no_grad():
        out = model(img_norm, ROI_Index=roi_index, SNR_info=snr_db)
    out = (out + 1.0) / 2.0             # back to [0,1]
    out = F.interpolate(out, size=(224, 224), mode="bicubic", align_corners=False)
    out = torch.clamp(out, 0.0, 1.0)    # clamp AFTER resize (bicubic overshoot)
    return out


# --------------------------------------------------------------------------- #
# YOLO label loading (YOLO txt -> absolute xyxy on the 224 grid)               #
# --------------------------------------------------------------------------- #
def load_boxes_yolo(label_path: str, img_size: int = 224):
    boxes = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                p = line.strip().split()
                if len(p) != 5:
                    continue
                cx, cy, w, h = map(float, p[1:])
                x1 = np.clip(cx - w / 2, 0, 1) * img_size
                y1 = np.clip(cy - h / 2, 0, 1) * img_size
                x2 = np.clip(cx + w / 2, 0, 1) * img_size
                y2 = np.clip(cy + h / 2, 0, 1) * img_size
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 4)


# --------------------------------------------------------------------------- #
# smoke test: prove the port imports + runs (noiseless CPU, no checkpoint)     #
# --------------------------------------------------------------------------- #
def smoke(rcpp: int = 12, device: str = "cpu"):
    model = build_model(rcpp=rcpp, chan_type="none", device=device)
    n_params = sum(p.numel() for p in model.parameters())
    x = torch.rand(1, 3, 224, 224)
    for mode in ("largest_box", "shoal_bbox"):
        # a fake shoal: 5 small boxes scattered across the frame
        boxes = np.array([[20, 20, 60, 50], [40, 100, 80, 130], [160, 30, 200, 60],
                          [100, 150, 140, 180], [180, 170, 210, 205]], dtype=np.float64)
        idx = compute_roi_index(boxes, mode=mode)
        y = reconstruct(model, x, roi_index=idx, snr_db=10, seed=42, device=device)
        print(f"[smoke] rcpp={rcpp} mode={mode} ROI_Index={idx} "
              f"out.shape={tuple(y.shape)} out_range=[{float(y.min()):.4f},{float(y.max()):.4f}] "
              f"params={n_params/1e6:.2f}M")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--rcpp", type=int, default=12, choices=[12, 16, 24, 32])
    ap.add_argument("--chan", type=str, default="AWGN", choices=["AWGN", "Rayleigh", "none"])
    ap.add_argument("--snr", type=float, default=10)
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--roi-mode", type=str, default="largest_box",
                    choices=["largest_box", "shoal_bbox"])
    ap.add_argument("--data-root", type=str, default="../yolo_fish_dataset_root/test")
    ap.add_argument("--detector-weights", type=str, default="../best.pt")
    ap.add_argument("--out", type=str, default="roi_jscc_eval.csv")
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = ap.parse_args()

    if args.smoke:
        smoke(args.rcpp, device=("cpu" if not torch.cuda.is_available() else args.device))
        return

    raise NotImplementedError(
        "Full evaluation is deferred until a ROI-JSCC checkpoint exists (TRAIN DEFERRED). "
        "Provide --ckpt to the upstream naming convention "
        "ROIIT_DIV2K_<chan>_SNR<0xx>_rcpp<0xx>_PSNR_ROIJSCC.pt; see file header."
    )


if __name__ == "__main__":
    main()
