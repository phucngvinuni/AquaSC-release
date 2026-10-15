#!/usr/bin/env python3
"""
roijscc_fish_common.py — shared building blocks for the AquaSC ROI-JSCC
comparison (Wave 9b ROIJSCC agent, plan §0.3 G1 + §8.1 + §11.2).

This module is deliberately self-contained: it does NOT import loss_maker /
data_maker / hydra / albumentations (all absent from siso-project-env), only
the ROIJSCC model package (via the glymur stub in aquasc_stubs/) and numpy/torch.

Key adaptations (documented, see the report):
  1. glymur stubbed (JPEG2000-only, never used by ROIJSCC).
  2. Resolution 224/any -> 256 (ROI window attention needs input multiple of 128).
  3. Channel: ROI-JSCC's native per-element Rayleigh is replaced by a BLOCK-fading
     Rayleigh with equalization by h, matching AquaSC's digital_channel
     .simulate_rayleigh_channel and the seeded channel bank (Wave 8). This holds
     the channel equal across all methods in §8.1.  ROI-JSCC's own power
     normalization (Channel.complex_normalize) is retained verbatim.
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "aquasc_stubs"))   # glymur stub
sys.path.insert(0, HERE)

from model.JSCC import ROIJSCC  # noqa: E402
from model.common_component import (  # noqa: E402
    patch_division, get_ROI_attention_mask, get_peripheral_attention_mask,
    get_RONI_attention_mask,
)

ROI_RES = 256          # ROI-JSCC native/training resolution (multiple of 128)
PATCH_D = 4            # ROI mask grid d x d
ETA = 0.1              # upstream model_info['eta']


# --------------------------------------------------------------------------- #
# model construction (mirrors aqua_adapter.build_model / model_maker "ROIJSCC") #
# --------------------------------------------------------------------------- #
def build_model(rcpp: int = 32, chan_type: str = "Rayleigh", device: str = "cuda"):
    model_info = {
        "chan_type": chan_type,
        "color_channel": 3,
        "rcpp": rcpp,
        "n_block_list": [2, 2, 2, 2],
        "n_feats_list": [40, 60, 80, 260],
        "num_heads_list": [4, 6, 8, 10],
        "window_size_list": [8, 8, 8, 8],
        "input_resolution": (ROI_RES, ROI_RES),
        "ratio1": 0.5,
        "ratio2": 0.5,
        "eta": ETA,
        "patch_d": PATCH_D,
        "SNR_info": 10,
    }
    model = ROIJSCC(model_info)
    model.to(device)
    return model


def rate_constants(rcpp: int):
    """Return (C, RONI_C, ROI_C, L_real, N_complex) for the rcpp config.

    Mirrors ROIJSCC.__init__ / patch_rate_allocation exactly:
      C         = int(1536 / rcpp)  (feature channels)
      decreased = int(eta * C); increased = 7 * decreased
      RONI_C = C - decreased;  ROI_C = C + increased
      patch count per rate = [7, 8, 1] over the 4x4 grid (h=w=4)
      L_real = (7*RONI_C + 8*C + 1*ROI_C) * 16      (16 = 4*4 spatial per patch)
      N_complex = L_real // 2
    """
    n_stage = 4
    C = int(3 * 2 * (2 ** n_stage) * (2 ** n_stage) * (1.0 / rcpp))
    decreased = int(ETA * C)
    increased = int(7 * decreased)
    RONI_C = C - decreased
    ROI_C = C + increased
    total_channels = 7 * RONI_C + 8 * C + 1 * ROI_C
    L_real = total_channels * (PATCH_D * PATCH_D)
    N_complex = L_real // 2
    return dict(C=C, RONI_C=RONI_C, ROI_C=ROI_C, L_real=L_real, N_complex=N_complex)


# --------------------------------------------------------------------------- #
# channel: block-fading Rayleigh + equalization (bank-compatible)              #
# --------------------------------------------------------------------------- #
class BlockRayleighChannel(nn.Module):
    """Drop-in replacement for ROIJSCC.channel.

    forward(input, snr_db):
      input : B x L  (valid_z, real)
      channel_tx = complex_normalize(input)          # ROI-JSCC power norm
      z_complex  = tx[:, :L/2] + 1j * tx[:, L/2:]    # per-image pairing
      y          = z_complex * h + sigma * (n_r + 1j*n_i)   # block fade
      z_hat      = y / h                             # equalize (AquaSC/DeepJSCC)
      return cat([z_hat.real, z_hat.imag], 1)         # B x L

    sigma = sqrt(1 / (2 * 10**(snr/10)))   (ROI-JSCC's own noise convention)

    Before each forward the caller sets:
      self.h          : scalar complex, or (B,1) complex64 tensor
      self.noise_real : (B, 2*N) real standard-normal tensor (or 2*N array for B=1)
    """
    def __init__(self):
        super().__init__()
        self.h = None
        self.noise_real = None

    @staticmethod
    def complex_normalize(x):
        x_dim = x.size()
        x = x.reshape((x_dim[0], -1))
        out = F.normalize(x, dim=1)
        out = out * torch.sqrt(torch.tensor(x.size()[1], dtype=torch.float32, device=x.device))
        out = out * torch.sqrt(torch.tensor(0.5, device=x.device))
        out = out.reshape(x_dim)
        return out

    def forward(self, input, chan_param):
        B, L = input.shape
        channel_tx = self.complex_normalize(input)
        z_complex = channel_tx[:, :L // 2] + 1j * channel_tx[:, L // 2:]
        N = L // 2
        snr_linear = 10 ** (float(chan_param) / 10.0)
        sigma = np.sqrt(1.0 / (2.0 * snr_linear))
        noise = sigma * (self.noise_real[:, :N] + 1j * self.noise_real[:, N:2 * N])
        h = self.h
        y = z_complex * h + noise
        z_hat = y / (h + 1e-9)
        out = torch.cat([z_hat.real, z_hat.imag], dim=1)
        return out


# --------------------------------------------------------------------------- #
# ROI index from YOLO labels (single cell of the 4x4 grid)                     #
# --------------------------------------------------------------------------- #
def compute_roi_index_from_norm(boxes_norm, d=PATCH_D, mode="largest_box"):
    """boxes_norm: (K,4) xyxy in [0,1]. Returns (Ih, Iw) in [0,d-1]^2."""
    boxes = np.asarray(boxes_norm, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[0] == 0 or boxes.shape[1] != 4:
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
    Ih = int(np.clip(cy * d, 0, d - 1))
    Iw = int(np.clip(cx * d, 0, d - 1))
    return (Ih, Iw)


def load_boxes_norm(label_path):
    """YOLO txt -> (K,4) xyxy in [0,1] (class column dropped)."""
    boxes = []
    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                p = line.strip().split()
                if len(p) != 5:
                    continue
                cx, cy, w, h = map(float, p[1:])
                x1 = np.clip(cx - w / 2, 0, 1)
                y1 = np.clip(cy - h / 2, 0, 1)
                x2 = np.clip(cx + w / 2, 0, 1)
                y2 = np.clip(cy + h / 2, 0, 1)
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 4)


# --------------------------------------------------------------------------- #
# ROI-JSCC loss (copied verbatim from loss_maker.ROIIT_MSE, d=4)               #
# --------------------------------------------------------------------------- #
class ROIIT_MSE(nn.Module):
    def __init__(self, d=PATCH_D):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")
        self.d = int(d)

    def forward(self, image_hat, image, ROI_Index):
        B, C, H, W = image.shape
        d = self.d
        roi_mask = get_ROI_attention_mask(ROI_Index, B, d, d)
        periph_mask = get_peripheral_attention_mask(ROI_Index, B, d, d)
        roni_mask = get_RONI_attention_mask(ROI_Index, B, d, d)
        device = image_hat.device
        roi_mask = roi_mask.reshape(B, d * d).to(device)
        periph_mask = periph_mask.reshape(B, d * d).to(device)
        roni_mask = roni_mask.reshape(B, d * d).to(device)

        image_hat = (image_hat + 1) / 2
        image = (image + 1) / 2
        image_patch_hat = patch_division(image_hat, d)
        image_patch = patch_division(image, d)
        unreduced_mse = self.mse(image_patch_hat, image_patch)
        patch_wise_mse = unreduced_mse.mean(
            dim=[i for i in range(2, len(image_patch_hat.size()))]).reshape(-1, d * d)

        mse = patch_wise_mse.mean()
        roi_mse = roi_mask * patch_wise_mse
        roi_mse = roi_mse.mean() * d * d
        periph_mse = periph_mask * patch_wise_mse
        periph_mse = periph_mse.mean() * d * d / periph_mask.sum() * B
        roni_mse = roni_mask * patch_wise_mse
        roni_mse = roni_mse.mean() * d * d / roni_mask.sum() * B

        total_loss = mse + roi_mse + 0.5 * periph_mse
        psnr = 10 * (np.log(1.0 / mse.clone().detach().cpu()) / np.log(10))
        return total_loss, psnr
