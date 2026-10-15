# routing_baselines.py — drop-in per-token routing score functions (plan §8.1).
#
# Each function returns a score vector s in [0,1]^[B, Np] (Np = 784 tokens for
# patch 8 @ 224x224); routing is top-K at K = 63, so the rate is identical by
# construction across all variants (oracle-box / random / segmentation-heatmap /
# uniform / oracle-binary).

import torch
import numpy as np

from types import SimpleNamespace


def _ensure_hyp(detector_model):
    """Standalone-loaded ultralytics checkpoints (e.g. YOLO('best.pt').model)
    store `args` as a bare dict lacking the box/cls/dfl loss gains that
    `v8DetectionLoss` reads via `self.hyp.box/.cls/.dfl`. Convert to a namespace
    and inject the default gains so `DetectionModel.loss()` runs outside a
    trainer."""
    if not hasattr(detector_model.args, "box"):
        from ultralytics.cfg import DEFAULT_CFG_DICT
        d = dict(DEFAULT_CFG_DICT)
        cur = detector_model.args
        if isinstance(cur, dict):
            d.update(cur)
        detector_model.args = SimpleNamespace(**d)
    return detector_model


# --------------------------------------------------------------------------- #
# Box / random / uniform / oracle-binary scorers (ROUTING, wave 7)             #
#                                                                              #
# NOTE (wave 9a ROUTING-EVAL): these functions were written by the ROUTING      #
# agent (wave 7) and unit-tested in tests/test_routing_baselines.py, but were         #
# clobbered when the COMPETITOR agent rewrote routing_baselines.py in place     #
# with a different scorer. They are reconstructed here verbatim    #
# from the test file's contract (shape/dtype/range/hand-computed cases).         #
# --------------------------------------------------------------------------- #
def _as_box_batch(boxes):
    """Normalize a boxes argument to a list of per-image [N,4] float tensors.

    ``boxes`` may be a single tensor [N,4] (one image) or a list/tuple of
    tensors (batch). An empty image is represented by an (0,4) tensor."""
    if isinstance(boxes, (list, tuple)):
        batch = list(boxes)
    else:
        batch = [boxes]
    return batch


def _box_occupancy(boxes, H, W):
    """Boolean HxW occupancy grid of the union of all boxes (pixel-level)."""
    occ = torch.zeros(H, W, dtype=torch.bool)
    if boxes is None or boxes.numel() == 0:
        return occ
    for (x1, y1, x2, y2) in boxes.float().reshape(-1, 4).tolist():
        x1i = int(np.floor(max(0.0, x1)))
        x2i = int(np.ceil(min(float(W), x2)))
        y1i = int(np.floor(max(0.0, y1)))
        y2i = int(np.ceil(min(float(H), y2)))
        if x2i > x1i and y2i > y1i:
            occ[y1i:y2i, x1i:x2i] = True
    return occ


def oracle_box_scores(boxes, PS, NP, H, W):
    """Oracle bounding-box routing score.

    score_i = area(patch_i ∩ union(GT boxes)) / patch_area  in [0,1].

    Returns float32 [B, NP] (B=1 when ``boxes`` is a single tensor).
    Patches are indexed row-major (r * n_w + c), matching
    datasets._create_patch_importance_map and the ViT patch order.
    """
    batch = _as_box_batch(boxes)
    B = len(batch)
    n_h, n_w = H // PS, W // PS
    out = torch.zeros(B, NP, dtype=torch.float32)
    for bi, b in enumerate(batch):
        occ = _box_occupancy(b, H, W)
        for r in range(n_h):
            for c in range(n_w):
                patch = occ[r * PS:(r + 1) * PS, c * PS:(c + 1) * PS]
                out[bi, r * n_w + c] = float(patch.float().mean())
    return out


def random_scores(NP, seed=42, batch_size=1):
    """Random routing score: i.i.d. Uniform[0,1] per token, seeded (local
    generator so the global RNG is untouched). Returns float32 [batch_size, NP]."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    return torch.rand(batch_size, NP, generator=gen, dtype=torch.float32)


def uniform_scores(NP, batch_size=1):
    """Uniform (constant 1.0) score: no routing signal. With top-K this routes an
    arbitrary fixed set of K tokens to the high path. Returns [batch_size, NP]."""
    return torch.ones(batch_size, NP, dtype=torch.float32)


def oracle_binary(boxes, PS, NP, H, W):
    """Binary box-oracle routing score: 1 if the patch overlaps any GT box, else 0.

    NOTE (wave 9a FIXUP): this is a *binary oracle* using the same ground-truth
    boxes as ``oracle_box_scores`` — 1 inside the box, 0 outside. It is NOT the
    "detect-at-edge" baseline from plan §8.1, which would run the detector at the
    edge and transmit only the boxes (rate ≈ 0). That baseline has not been run
    and needs no number; it needs a discussion paragraph answering "why transmit
    images at all" (human verification, other downstream tasks, periodic
    retraining, insufficient edge compute for a good detector).

    Returns float32 [B, NP]."""

    batch = _as_box_batch(boxes)
    B = len(batch)
    n_h, n_w = H // PS, W // PS
    out = torch.zeros(B, NP, dtype=torch.float32)
    for bi, b in enumerate(batch):
        occ = _box_occupancy(b, H, W)
        for r in range(n_h):
            for c in range(n_w):
                if bool(occ[r * PS:(r + 1) * PS, c * PS:(c + 1) * PS].any()):
                    out[bi, r * n_w + c] = 1.0
    return out


def apply_topk(scores, K):
    """Top-K routing on a [B, NP] score tensor.

    Returns (mask, idx): mask is bool [B, NP] with exactly K True per row, idx is
    long [B, K] = topk indices. K >= n_patches routes everything high."""
    B, NP = scores.shape
    if K >= NP:
        mask = torch.ones(B, NP, dtype=torch.bool, device=scores.device)
        idx = torch.arange(NP, device=scores.device).unsqueeze(0).expand(B, -1)
        return mask, idx
    idx = scores.topk(K, dim=-1).indices
    mask = torch.zeros(B, NP, dtype=torch.bool, device=scores.device)
    mask.scatter_(1, idx, True)
    return mask, idx


def model_fim_scores(model, img):
    """FIM routing score for the real model: sigmoid(fim_module(img_encoder(img)))
    -> [B, Np] in [0,1]. Requires model.img_encoder(img, mask) and
    model.fim_module(x) -> [B, Np, 1]."""
    B = img.shape[0]
    mask = torch.zeros(B, model.img_encoder.patch_embed.num_patches,
                       dtype=torch.bool, device=img.device)
    x = model.img_encoder(img, mask)
    logits = model.fim_module(x)
    return torch.sigmoid(logits).squeeze(-1)


def parse_yolo_label(path, H=224, W=224):
    """Parse a YOLO-format label file (cls cx cy w h, normalized) into absolute
    pixel xyxy boxes on the [0,W]x[0,H] grid. Returns float32 [N, 4]."""
    boxes = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                cx, cy, w, h = map(float, parts[1:])
            except ValueError:
                continue
            x1 = float(np.clip(cx - w / 2.0, 0.0, 1.0)) * W
            y1 = float(np.clip(cy - h / 2.0, 0.0, 1.0)) * H
            x2 = float(np.clip(cx + w / 2.0, 0.0, 1.0)) * W
            y2 = float(np.clip(cy + h / 2.0, 0.0, 1.0)) * H
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
    return torch.tensor(boxes, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Segmentation/detector-heatmap routing (plan 16.10 E-6, R3 #5)               #
# --------------------------------------------------------------------------- #
def detector_heatmap_scores(pred_boxes, pred_scores, PS, NP, H, W):
    """Detector-at-the-edge heatmap routing score (the "segmentation/detector
    heatmap variant" named in the master plan / response letter R3 #5).

    Unlike ``oracle_box_scores`` (which uses ground-truth boxes, unavailable at
    the transmitter), this runs the SAME frozen detector used for the paper's
    mAP metric directly on the pre-transmission image and scores each patch by
    the confidence of the highest-confidence predicted box it overlaps:

        score_i = max_{j: patch_i overlaps pred_box_j} conf_j,   else 0.

    This is a legitimate edge-computable baseline (the detector already runs at
    the edge in the "transmit boxes only" variant; here its confidence map is
    reused as a soft routing signal instead of being transmitted directly), and
    is the natural pixel-domain analogue of a segmentation mask when only box
    detections (not per-pixel masks) are available in this dataset.

    Args:
        pred_boxes: [N,4] absolute-pixel xyxy tensor/array (one image).
        pred_scores: [N] confidence tensor/array, same order as pred_boxes.
    Returns: float32 [1, NP] in [0,1] (0 if no detections).
    """
    import torch as _torch
    if not _torch.is_tensor(pred_boxes):
        pred_boxes = _torch.as_tensor(pred_boxes, dtype=_torch.float32)
    if not _torch.is_tensor(pred_scores):
        pred_scores = _torch.as_tensor(pred_scores, dtype=_torch.float32)
    n_h, n_w = H // PS, W // PS
    out = torch.zeros(1, NP, dtype=torch.float32)
    if pred_boxes.numel() == 0:
        return out
    for (x1, y1, x2, y2), conf in zip(pred_boxes.reshape(-1, 4).tolist(), pred_scores.tolist()):
        x1i = max(0, int(np.floor(x1))); x2i = min(W, int(np.ceil(x2)))
        y1i = max(0, int(np.floor(y1))); y2i = min(H, int(np.ceil(y2)))
        if x2i <= x1i or y2i <= y1i:
            continue
        r0, r1 = y1i // PS, (y2i - 1) // PS
        c0, c1 = x1i // PS, (x2i - 1) // PS
        for r in range(max(0, r0), min(n_h - 1, r1) + 1):
            for c in range(max(0, c0), min(n_w - 1, c1) + 1):
                idx = r * n_w + c
                if conf > out[0, idx]:
                    out[0, idx] = conf
    return out


ROUTING_STRATEGIES = {
    "fim": model_fim_scores,
    "oracle_box": oracle_box_scores,
    "random": random_scores,
    "uniform": uniform_scores,
    "oracle_binary": oracle_binary,
    "detector_heatmap": detector_heatmap_scores,
}
