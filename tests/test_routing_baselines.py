"""Unit tests for routing_baselines.py (R3 #5 / plan §8.1). Run with:

    __TORCH_PY__ tests/test_routing_baselines.py

CPU-only. Verifies: each score fn returns the right shape/dtype/range, oracle boxes
match a hand-computed 2-box case, top-K picks K=63, and the dispatch table works.
model_fim_scores is smoke-tested against a minimal fake model (no checkpoint / no GPU).
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repo root (core modules)
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import routing_baselines as rb

N_FAIL = [0]


def check(cond, msg):
    if not cond:
        N_FAIL[0] += 1
        print(f"FAIL: {msg}")
    else:
        print(f"ok: {msg}")


# Grid under test: 32x32 image, patch 8 -> 4x4 = 16 patches.
H = W = 32
PS = 8
NP = 16  # 4 * 4


def test_shapes_and_ranges():
    print("=" * 70)
    print("shape / dtype / range")
    print("=" * 70)

    # single-image boxes -> [1, n_patches]
    boxes = torch.tensor([[0.0, 0.0, 8.0, 8.0]], dtype=torch.float32)
    s = rb.oracle_box_scores(boxes, PS, NP, H, W)
    check(s.shape == (1, NP), f"oracle_box_scores shape (1, {NP}) -> {tuple(s.shape)}")
    check(s.dtype == torch.float32, f"oracle_box_scores dtype float32 -> {s.dtype}")
    check(bool((s >= 0).all() and (s <= 1).all()), "oracle_box_scores in [0,1]")

    # batch of 2 images -> [2, n_patches]
    s2 = rb.oracle_box_scores([boxes, boxes], PS, NP, H, W)
    check(s2.shape == (2, NP), f"oracle_box_scores batch shape (2, {NP}) -> {tuple(s2.shape)}")

    rs = rb.random_scores(NP, seed=0, batch_size=2)
    check(rs.shape == (2, NP), f"random_scores shape (2, {NP}) -> {tuple(rs.shape)}")
    check(rs.dtype == torch.float32, f"random_scores dtype -> {rs.dtype}")
    check(bool((rs >= 0).all() and (rs <= 1).all()), "random_scores in [0,1]")

    # determinism by seed
    check(torch.equal(rb.random_scores(NP, 42), rb.random_scores(NP, 42)),
          "random_scores deterministic for same seed")
    check(not torch.equal(rb.random_scores(NP, 0), rb.random_scores(NP, 1)),
          "random_scores differs across seeds")

    us = rb.uniform_scores(NP, batch_size=1)
    check(us.shape == (1, NP), f"uniform_scores shape (1, {NP}) -> {tuple(us.shape)}")
    check(bool((us == 1.0).all()), "uniform_scores constant == 1.0")

    de = rb.oracle_binary(boxes, PS, NP, H, W)
    check(de.shape == (1, NP), f"oracle_binary shape (1, {NP}) -> {tuple(de.shape)}")
    check(bool(((de == 0.0) | (de == 1.0)).all()), "oracle_binary binary")
    check(bool((de <= 1).all() and (de >= 0).all()), "oracle_binary in [0,1]")


def test_oracle_hand_computed():
    print("=" * 70)
    print("oracle_box hand-computed 2-box case (overlap union)")
    print("=" * 70)
    # Patch 0 covers x[0,8], y[0,8]. Two boxes:
    #   B1 = [0,0,4,8]  -> left half   area 32
    #   B2 = [0,0,8,4]  -> top  half   area 32
    #   overlap [0,0,4,4] = 16 -> union = 48 -> score = 48/64 = 0.75
    boxes = torch.tensor([[0.0, 0.0, 4.0, 8.0],
                          [0.0, 0.0, 8.0, 4.0]], dtype=torch.float32)
    s = rb.oracle_box_scores(boxes, PS, NP, H, W)
    check(abs(float(s[0, 0]) - 0.75) < 1e-6, f"oracle patch0 == 0.75 (got {float(s[0,0]):.6f})")
    check(abs(float(s[0, 1:].sum())) < 1e-9, f"all other patches == 0 (sum={float(s[0,1:].sum()):.3e})")
    check(abs(float(s[0].sum()) - 0.75) < 1e-6, f"total score mass == 0.75 (got {float(s[0].sum()):.6f})")

    # Full-image box -> every patch fully covered -> score 1.0 everywhere.
    full = torch.tensor([[0.0, 0.0, float(W), float(H)]], dtype=torch.float32)
    sf = rb.oracle_box_scores(full, PS, NP, H, W)
    check(bool((sf[0] == 1.0).all()), "full-image box -> all scores 1.0")

    # Empty boxes -> all zeros.
    empty = torch.empty((0, 4), dtype=torch.float32)
    se = rb.oracle_box_scores(empty, PS, NP, H, W)
    check(bool((se[0] == 0.0).all()), "empty boxes -> all scores 0.0")


def test_oracle_binary_semantics():
    print("=" * 70)
    print("oracle_binary (binary box-oracle) semantics")
    print("=" * 70)
    boxes = torch.tensor([[0.0, 0.0, 4.0, 8.0],
                          [10.0, 10.0, 12.0, 12.0]], dtype=torch.float32)
    de = rb.oracle_binary(boxes, PS, NP, H, W)
    # patch (0,0) overlaps B1; patch (1,1) overlaps B2 -> both 1.0
    check(float(de[0, 0]) == 1.0, "patch (0,0) == 1.0 (overlaps B1)")
    check(float(de[0, 1 * 4 + 1]) == 1.0, "patch (1,1) == 1.0 (overlaps B2)")
    check(float(de[0, 0 * 4 + 1]) == 0.0, "patch (0,1) == 0.0 (no overlap)")


def test_apply_topk():
    print("=" * 70)
    print("apply_topk K=63")
    print("=" * 70)
    NPATCH = 784
    scores = torch.rand(1, NPATCH)
    mask, idx = rb.apply_topk(scores, 63)
    check(mask.shape == (1, NPATCH), f"mask shape (1, {NPATCH}) -> {tuple(mask.shape)}")
    check(mask.dtype == torch.bool, f"mask dtype bool -> {mask.dtype}")
    check(idx.shape == (1, 63), f"topk idx shape (1, 63) -> {tuple(idx.shape)}")
    check(int(mask.sum()) == 63, f"exactly 63 routed high (got {int(mask.sum())})")
    check(torch.equal(idx, scores.topk(63, dim=-1).indices), "topk indices match torch.topk")

    # batched
    scores_b = torch.rand(2, NPATCH)
    mask_b, idx_b = rb.apply_topk(scores_b, 63)
    check(mask_b.shape == (2, NPATCH), "batched mask shape")
    check(int(mask_b.sum()) == 2 * 63, "batched: 126 routed high")

    # K >= n_patches -> all routed
    mask_all, _ = rb.apply_topk(torch.rand(1, 16), 63)
    check(bool(mask_all.all()), "K >= n_patches -> all routed")


def test_model_fim_adapter():
    print("=" * 70)
    print("model_fim_scores adapter (fake model, no checkpoint)")
    print("=" * 70)

    class FakePatchEmbed:
        num_patches = 784

    class FakeEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = FakePatchEmbed()

        def forward(self, img, mask):
            B = img.shape[0]
            return torch.randn(B, 784, 512)

    class FakeFIM(torch.nn.Module):
        def forward(self, x):
            return torch.randn(x.shape[0], x.shape[1], 1)

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.img_encoder = FakeEncoder()
            self.fim_module = FakeFIM()

    model = FakeModel()
    img = torch.randn(3, 3, 224, 224)
    scores = rb.model_fim_scores(model, img)
    check(scores.shape == (3, 784), f"model_fim_scores shape (3, 784) -> {tuple(scores.shape)}")
    check(scores.dtype == torch.float32, f"model_fim_scores dtype -> {scores.dtype}")
    check(bool((scores >= 0).all() and (scores <= 1).all()), "model_fim_scores sigmoid in [0,1]")


def test_dispatch():
    print("=" * 70)
    print("ROUTING_STRATEGIES dispatch table")
    print("=" * 70)
    expected = {"fim", "oracle_box", "random", "uniform", "oracle_binary", "detector_heatmap"}
    check(set(rb.ROUTING_STRATEGIES.keys()) == expected,
          f"dispatch keys == {sorted(expected)} (got {sorted(rb.ROUTING_STRATEGIES.keys())})")
    check(all(callable(f) for f in rb.ROUTING_STRATEGIES.values()),
          "every dispatch entry callable")
    check("gradient_scores" not in rb.ROUTING_STRATEGIES,
          "gradient_scores is NOT registered (COMPETITOR's deliverable)")

    # exercise each box/rand/uniform strategy through the table
    boxes = torch.tensor([[0.0, 0.0, 8.0, 8.0]])
    s_oracle = rb.ROUTING_STRATEGIES["oracle_box"](boxes, PS, NP, H, W)
    check(s_oracle.shape == (1, NP), "dispatch oracle_box works")
    s_rand = rb.ROUTING_STRATEGIES["random"](NP, seed=7, batch_size=1)
    check(s_rand.shape == (1, NP), "dispatch random works")
    s_unif = rb.ROUTING_STRATEGIES["uniform"](NP, batch_size=1)
    check(s_unif.shape == (1, NP), "dispatch uniform works")
    s_de = rb.ROUTING_STRATEGIES["oracle_binary"](boxes, PS, NP, H, W)
    check(s_de.shape == (1, NP), "dispatch oracle_binary works")


def test_parse_yolo_label():
    print("=" * 70)
    print("parse_yolo_label against a real label file")
    print("=" * 70)
    data_root = "__AQUASC_DATA__/yolo_fish_dataset_root"
    test_labels = os.path.join(data_root, "test", "labels")
    if not os.path.isdir(test_labels):
        print("skip: test label dir not found")
        return
    labels = sorted(f for f in os.listdir(test_labels) if f.endswith(".txt"))
    if not labels:
        print("skip: no label files")
        return
    p = os.path.join(test_labels, labels[0])
    with open(p) as f:
        n_lines = sum(1 for ln in f if len(ln.strip().split()) == 5)
    boxes = rb.parse_yolo_label(p, H=224, W=224)
    check(boxes.shape == (n_lines, 4), f"parse_yolo_label -> ({n_lines}, 4) got {boxes.shape}")
    check(bool((boxes[:, 0] < boxes[:, 2]).all() and (boxes[:, 1] < boxes[:, 3]).all()),
          "all parsed boxes x1<x2 and y1<y2")
    check(bool((boxes >= 0).all() and (boxes <= 224).all()), "all parsed boxes within [0,224]")


if __name__ == "__main__":
    test_shapes_and_ranges()
    test_oracle_hand_computed()
    test_oracle_binary_semantics()
    test_apply_topk()
    test_model_fim_adapter()
    test_dispatch()
    test_parse_yolo_label()
    print("=" * 70)
    if N_FAIL[0]:
        print(f"{N_FAIL[0]} FAILURES")
        sys.exit(1)
    print("ALL TESTS PASSED")
