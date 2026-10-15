# metrics.py - unified evaluation metrics for AquaSC reconstruction + detection.
#
# Importable module: no side effects on import (no models are created until a
# class is instantiated / function is called).
#
# Computes, from (original, reconstruction, boxes) batches:
#   - global PSNR / SSIM            (same protocol as engine1.py / utils.py)
#   - in-box PSNR / SSIM / LPIPS    (crops taken from GT boxes)
#   - mAP@0.5 and mAP@0.5:0.95      (torchmetrics MeanAveragePrecision, bbox, class fish)
#   - precision / recall / F1       (greedy IoU-0.5 matching, score >= conf_thres)
#   - per-object-size breakdown     (COCO: small < 32^2, medium [32^2, 96^2), large >= 96^2)
#   - retention ratio               = mAP(recon) / mAP(original)
#
# Conventions (must match the engine for cross-checking):
#   - images are float tensors in [0, 1], BCHW, RGB.
#   - GT boxes are absolute pixel xyxy on the 224x224 evaluation grid (as
#     produced by datasets.YOLODataset).
#   - global PSNR/SSIM reuse utils.calc_psnr / utils.calc_ssim verbatim
#     (integer quantization via as_img_array, data_range 255).
#   - in-box crops smaller than the SSIM window (11x11) are padded with edge
#     replication to 11x11 before SSIM (documented choice; avoids the engine's
#     zero-fallback bias on tiny boxes).

import os
import sys
import torch
import numpy as np

try:
    import utils  # repo-local: calc_psnr / calc_ssim
    _UTILS_OK = True
except ImportError:
    _UTILS_OK = False

try:
    import lpips
    _LPIPS_OK = True
except ImportError:
    _LPIPS_OK = False

try:
    from torchmetrics.detection import MeanAveragePrecision
    _TM_OK = True
except ImportError:
    _TM_OK = False

IMG_SIZE = 224          # eval grid (matches datasets.YOLODataset img_pixel_size)
SSIM_MIN_WIN = 11       # pytorch_msssim default win_size
LPIPS_MIN_SIDE = 64     # lpips alex/squeeze nets need >= 64 input; tiny in-box crops
                        # are edge-replication padded to 64x64 (same documented choice as SSIM).


def yolo_predict(yolo_model, images: torch.Tensor, conf: float = 0.001,
                 iou: float = 0.5, device=None) -> list:
    """Run YOLO on a batch of [0,1] images and return torchmetrics-compatible
    prediction dicts: {boxes: xyxy [N,4], scores: [N], labels: [N] int64}."""
    device = device if device is not None else images.device
    out = []
    results = yolo_model(images.detach(), verbose=False, conf=conf, iou=iou)
    for r in results:
        if r.boxes is not None and len(r.boxes.xyxy) > 0:
            out.append({
                "boxes": r.boxes.xyxy.to(device),
                "scores": r.boxes.conf.to(device),
                "labels": r.boxes.cls.to(torch.int64).to(device),
            })
        else:
            out.append({
                "boxes": torch.empty((0, 4), dtype=torch.float32, device=device),
                "scores": torch.empty(0, dtype=torch.float32, device=device),
                "labels": torch.empty(0, dtype=torch.int64, device=device),
            })
    return out


def gt_to_device(gt_boxes_list: list, device) -> list:
    return [{"boxes": g["boxes"].to(device), "labels": g["labels"].to(device)}
            for g in gt_boxes_list]


def _box_iou_matrix(pred_xyxy: torch.Tensor, gt_xyxy: torch.Tensor) -> torch.Tensor:
    if pred_xyxy.numel() == 0 or gt_xyxy.numel() == 0:
        return torch.zeros((len(pred_xyxy), len(gt_xyxy)), dtype=torch.float32)
    x1 = torch.max(pred_xyxy[:, None, 0], gt_xyxy[None, :, 0])
    y1 = torch.max(pred_xyxy[:, None, 1], gt_xyxy[None, :, 1])
    x2 = torch.min(pred_xyxy[:, None, 2], gt_xyxy[None, :, 2])
    y2 = torch.min(pred_xyxy[:, None, 3], gt_xyxy[None, :, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    a_pred = (pred_xyxy[:, 2] - pred_xyxy[:, 0]) * (pred_xyxy[:, 3] - pred_xyxy[:, 1])
    a_gt = (gt_xyxy[:, 2] - gt_xyxy[:, 0]) * (gt_xyxy[:, 3] - gt_xyxy[:, 1])
    union = a_pred[:, None] + a_gt[None, :] - inter
    return inter / union.clamp(min=1e-9)


def _match_tp_fp(preds_boxes, preds_scores, gt_boxes, iou_thr=0.5, score_thr=0.001):
    """Greedy matching at IoU >= iou_thr with score >= score_thr.
    Returns (tp, fp, fn, n_pred_used)."""
    if len(preds_scores) == 0:
        return 0, 0, len(gt_boxes), 0
    order = torch.argsort(preds_scores, descending=True)
    matched_gt = set()
    tp = 0
    fp = 0
    n_used = 0
    for idx in order.tolist():
        if float(preds_scores[idx]) < score_thr:
            break
        n_used += 1
        if len(gt_boxes) == 0:
            fp += 1
            continue
        ious = _box_iou_matrix(preds_boxes[idx:idx + 1], gt_boxes)[0]
        best = int(torch.argmax(ious).item())
        if float(ious[best]) >= iou_thr and best not in matched_gt:
            tp += 1
            matched_gt.add(best)
        else:
            fp += 1
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn, n_used


def _crop(orig, recon, box_xyxy):
    """Crop [C,H,W] tensors to a clamped box; returns (o, r) 1xCxHxW or None."""
    H, W = orig.shape[1], orig.shape[2]
    x1 = int(max(0.0, float(box_xyxy[0])))
    y1 = int(max(0.0, float(box_xyxy[1])))
    x2 = int(min(float(W), float(box_xyxy[2])))
    y2 = int(min(float(H), float(box_xyxy[3])))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    o = orig[:, y1:y2, x1:x2].unsqueeze(0).float()
    r = recon[:, y1:y2, x1:x2].unsqueeze(0).float()
    return o, r


def _pad_min_window(t: torch.Tensor, min_side: int = SSIM_MIN_WIN) -> torch.Tensor:
    """Pad with edge replication so both spatial dims >= min_side."""
    h, w = t.shape[2], t.shape[3]
    if h >= min_side and w >= min_side:
        return t
    pad_b = max(0, min_side - h)
    pad_r = max(0, min_side - w)
    return torch.nn.functional.pad(t, (0, pad_r, 0, pad_b), mode="replicate")


class MetricsAccumulator:
    """Streaming accumulator mirroring engine1.py's metric calls.

    Feed per-batch data with accumulate(); call compute() once at the end.
    """

    def __init__(self, device="cpu", conf_thres: float = 0.001, iou_thres: float = 0.5,
                 compute_lpips: bool = False, lpips_net: str = "alex"):
        if not _TM_OK:
            raise ImportError("torchmetrics is required for MetricsAccumulator")
        self.device = torch.device(device)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.map_orig = MeanAveragePrecision(iou_type="bbox", class_metrics=True).to(self.device)
        self.map_recon = MeanAveragePrecision(iou_type="bbox", class_metrics=True).to(self.device)
        self.lpips_fn = None
        if compute_lpips:
            if not _LPIPS_OK:
                raise ImportError("lpips is required when compute_lpips=True")
            self.lpips_fn = lpips.LPIPS(net=lpips_net, verbose=False).to(self.device)
        self.n_images = 0
        self.n_gt_boxes = 0
        self.psnr_list = []
        self.ssim_list = []
        self.inbox_psnr = []
        self.inbox_ssim = []
        self.inbox_lpips = []
        self._tp_o = self._fp_o = self._fn_o = 0
        self._tp_r = self._fp_r = self._fn_r = 0

    def accumulate(self, orig: torch.Tensor, recon: torch.Tensor,
                   gt_boxes_list: list, preds_orig: list, preds_recon: list):
        orig = orig.detach().float().cpu()
        recon = recon.detach().float().cpu()
        if not _UTILS_OK:
            raise ImportError("repo-local utils.py not importable")
        p_global = utils.calc_psnr(recon, orig)
        s_global = utils.calc_ssim(recon, orig)
        self.psnr_list.extend(p_global)
        self.ssim_list.extend(s_global)

        gt_dev = gt_to_device(gt_boxes_list, self.device)
        if len(preds_orig) == len(gt_dev):
            self.map_orig.update(preds_orig, gt_dev)
        if len(preds_recon) == len(gt_dev):
            self.map_recon.update(preds_recon, gt_dev)

        for i in range(orig.shape[0]):
            gt_boxes = gt_boxes_list[i]["boxes"]
            self.n_gt_boxes += len(gt_boxes)
            self._update_pr_counters(
                preds_orig[i], gt_boxes, is_recon=False)
            self._update_pr_counters(
                preds_recon[i], gt_boxes, is_recon=True)
            for bi in range(len(gt_boxes)):
                crops = _crop(orig[i], recon[i], gt_boxes[bi])
                if crops is None:
                    continue
                o, r = crops
                self.inbox_psnr.append(utils.calc_psnr(r, o)[0])
                s_val = None
                try:
                    s_val = utils.calc_ssim(r, o)[0]
                except RuntimeError:
                    s_val = utils.calc_ssim(_pad_min_window(r), _pad_min_window(o))[0]
                self.inbox_ssim.append(s_val)
                if self.lpips_fn is not None:
                    with torch.no_grad():
                        lp_r = _pad_min_window(r, LPIPS_MIN_SIDE).to(self.device)
                        lp_o = _pad_min_window(o, LPIPS_MIN_SIDE).to(self.device)
                        self.inbox_lpips.append(float(
                            self.lpips_fn(lp_r * 2.0 - 1.0, lp_o * 2.0 - 1.0).mean()))
        self.n_images += orig.shape[0]

    def _update_pr_counters(self, preds, gt_boxes, is_recon: bool):
        tp, fp, fn, _ = _match_tp_fp(preds["boxes"].cpu(), preds["scores"].cpu(),
                                     gt_boxes.cpu(), iou_thr=self.iou_thres,
                                     score_thr=self.conf_thres)
        if is_recon:
            self._tp_r += tp; self._fp_r += fp; self._fn_r += fn
        else:
            self._tp_o += tp; self._fp_o += fp; self._fn_o += fn

    @staticmethod
    def _pr(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        r = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1 = 2 * p * r / (p + r) if (p + r) > 0 and p == p and r == r else float("nan")
        return p, r, f1

    def compute(self) -> dict:
        mo = self.map_orig.compute()
        mr = self.map_recon.compute()
        self.map_orig.reset()
        self.map_recon.reset()
        g = {
            "n_images": self.n_images,
            "n_gt_boxes": self.n_gt_boxes,
            "psnr": float(np.mean(self.psnr_list)) if self.psnr_list else float("nan"),
            "ssim": float(np.mean(self.ssim_list)) if self.ssim_list else float("nan"),
            "in_box_psnr": float(np.mean(self.inbox_psnr)) if self.inbox_psnr else float("nan"),
            "in_box_ssim": float(np.mean(self.inbox_ssim)) if self.inbox_ssim else float("nan"),
            "in_box_lpips": float(np.mean(self.inbox_lpips)) if self.inbox_lpips else float("nan"),
            "n_inbox_boxes": len(self.inbox_psnr),
        }
        for side, m in (("orig", mo), ("recon", mr)):
            for k in ("map", "map_50", "map_small", "map_medium", "map_large",
                      "mar_100", "mar_small", "mar_medium", "mar_large"):
                v = float(m[k]) if k in m and m[k].numel() == 1 else float("nan")
                g[f"{k}_{side}"] = v
        p_o, r_o, f_o = self._pr(self._tp_o, self._fp_o, self._fn_o)
        p_r, r_r, f_r = self._pr(self._tp_r, self._fp_r, self._fn_r)
        g.update({"precision_orig": p_o, "recall_orig": r_o, "f1_orig": f_o,
                  "precision_recon": p_r, "recall_recon": r_r, "f1_recon": f_r,
                  "tp_orig": self._tp_o, "fp_orig": self._fp_o, "fn_orig": self._fn_o,
                  "tp_recon": self._tp_r, "fp_recon": self._fp_r, "fn_recon": self._fn_r})
        g["retention_50"] = g["map_50_recon"] / g["map_50_orig"] if g["map_50_orig"] > 0 else float("nan")
        g["retention"] = g["map_recon"] / g["map_orig"] if g["map_orig"] > 0 else float("nan")
        return g
