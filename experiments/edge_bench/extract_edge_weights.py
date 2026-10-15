"""Dev-machine only (needs GPU + full checkpoints + model.py's timm/
pytorch_msssim deps). Run in siso-project-env.

For each model in MODELS: loads the real ViT_Reconstruction_Model via
experiments/matched5/evaluate.py's load_aquasc() (unmodified, proven code),
splits its state_dict into encoder-only/decoder-only subsets, parity-checks
the standalone edge wrappers against the real model on a real test image, and
writes encoder_<name>.pt / decoder_<name>.pt for scp to the Pi.

Models: the flagship (25.3M encoder, headline checkpoint) plus the two Site-B
S4-like-finetuned distilled students (192d2L: 0.93M encoder, 512d2L: 6.4M
encoder -- see SITEB_FINETUNE_RESULTS_SUMMARY.md section 7). Same routing/RVQ
config (routing_topk=63, rvq 3x6bit, bits_vq_low=2) across all three --
only encoder_embed_dim/encoder_depth differ.

Usage: python3 extract_edge_weights.py --out_dir /tmp/claude-1000/edge_bench_payload
"""
from __future__ import annotations
import argparse
import contextlib
import io
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "matched5"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from evaluate import load_aquasc, dataset
import digital_client
from edge_encoder_decoder import (
    AquaSCEdgeEncoder, AquaSCEdgeDecoder, split_state_dict, FLAGSHIP_ARCH,
)

MODELS = [
    # (name, encoder_embed_dim, encoder_depth, encoder_num_heads, checkpoint path or None=flagship)
    ("flagship", 512, 8, 8, None),
    ("512d2l", 512, 2, 8, ROOT / "ckpt_512d2L_siteb_s4like" / "checkpoint-best_map.pth"),
    ("192d2l", 192, 2, 8, ROOT / "ckpt_192d2L_siteb_s4like" / "checkpoint-best_map.pth"),
    ("256d8l", 256, 8, 8, ROOT / "ckpt_distill_tx_only_v2_256d8L" / "checkpoint-149.pth"),
]

# 512d4L (depth=4, midpoint between 512d2l and flagship): no checkpoint survives
# on disk (ckpt_distill_512d4L_300ep/ is empty -- cleaned up at some point).
# Latency depends only on architecture/shapes, not weight VALUES (same
# reasoning already used for the 3 competitor systems in
# extract_competitor_sources.py/bench_{jscc,swinjscc,roijscc}.py), so this is
# benchmarked with fresh random-init weights -- valid for latency, NOT valid
# for any accuracy/reconstruction claim. No real model exists to parity-check
# against, so that step is skipped for this one entry only.
RANDOM_INIT_MODELS = [
    ("512d4l", 512, 4, 8),
]


def arch_for(embed_dim, depth, heads):
    a = dict(FLAGSHIP_ARCH)
    a["encoder_embed_dim"] = embed_dim
    a["encoder_depth"] = depth
    a["encoder_num_heads"] = heads
    return a


def extract_one(name, embed_dim, depth, heads, ckpt, out: Path, test_idx=281):
    print(f"\n--- {name}: encoder_embed_dim={embed_dim} depth={depth} heads={heads} "
          f"ckpt={ckpt or '(flagship default)'} ---")
    net = load_aquasc("test", "a", str(ckpt) if ckpt else None, embed_dim, depth, heads)
    arch = arch_for(embed_dim, depth, heads)

    full_sd = net.state_dict()
    enc_sd, dec_sd = split_state_dict(full_sd)
    print(f"  encoder-only: {sum(v.numel() for v in enc_sd.values())/1e6:.2f}M params, "
          f"decoder-only: {sum(v.numel() for v in dec_sd.values())/1e6:.2f}M params")

    enc = AquaSCEdgeEncoder(arch).eval()
    dec = AquaSCEdgeDecoder(arch).eval()
    missing_e, unexpected_e = enc.load_state_dict(enc_sd, strict=False)
    missing_d, unexpected_d = dec.load_state_dict(dec_sd, strict=False)

    def _bad(keys):
        return [k for k in keys if "ema_" not in k and "usage_counts" not in k
                and "last_assigned" not in k and "kmeans_pool" not in k]
    assert not _bad(missing_e), f"[{name}] encoder missing keys: {_bad(missing_e)}"
    assert not _bad(unexpected_e), f"[{name}] encoder unexpected keys: {_bad(unexpected_e)}"
    assert not _bad(missing_d), f"[{name}] decoder missing keys: {_bad(missing_d)}"
    assert not _bad(unexpected_d), f"[{name}] decoder unexpected keys: {_bad(unexpected_d)}"

    ds = dataset("test", "a")
    (img, _mask), _ = ds[test_idx]
    img = img.unsqueeze(0)

    captured = {}

    def hook(tv, hm, *a, **kw):
        captured["tv"] = tv.clone()
        captured["hm"] = hm.clone()
        num_stages = net.quantizer_high.num_stages
        bits_per_stage = net.quantizer_high.bits_per_stage
        low_mask_val = net.quantizer_low.num_embeddings - 1
        stages = net.quantizer_high.unpack_indices(tv, num_stages, bits_per_stage).movedim(0, -1)
        return (tv, {}, stages, torch.zeros_like(stages, dtype=torch.bool),
                tv & low_mask_val, torch.zeros_like(hm))

    # Parity ground truth computed on CPU, matching the standalone wrapper's
    # device: comparing GPU vs CPU forward passes introduces genuine
    # floating-point non-determinism (different reduction order/kernels) that
    # can flip a FIM top-k boundary decision when two candidate scores are
    # very close -- observed on the shallower (depth=2) distilled students,
    # where the FIM router's high/low separation is much less confident than
    # the flagship's. Same-device (CPU vs CPU) removes that noise source so
    # this check verifies the wrapper's LOGIC, not GPU/CPU kernel parity.
    net_cpu = net.cpu()
    old = digital_client.transmit_indices_digital
    digital_client.transmit_indices_digital = hook
    try:
        with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
            real_out = net_cpu(img=img, bm_pos=torch.zeros(1, 784, dtype=torch.bool),
                               _eval=True, digital_ldpc=True, eval_snr_db=5)
    finally:
        digital_client.transmit_indices_digital = old
        net.cuda()

    real_route = captured["hm"][0].cpu()
    real_recon = real_out["reconstructed_image"][0].cpu()

    with torch.no_grad():
        packed_high, idx_low, route = enc(img)
        edge_recon = dec(packed_high, idx_low, route)[0]

    n_route_diff = int((route[0].cpu() != real_route).sum())
    recon_diff = float((edge_recon - real_recon).abs().max())
    print(f"  route mask diffs: {n_route_diff}/784, recon max abs diff: {recon_diff:.3e}")
    # A handful of routing-token flips is expected, benign float noise, not a
    # wrapper bug: verified by tracing block-by-block on the 512d2l model --
    # weights are bit-identical and inputs bit-identical into block 0, yet
    # ~1e-4-level differences appear after the Attention/MLP matmuls purely
    # from two independently-built computation graphs (contiguity/kernel-
    # dispatch noise, not a logic error). This only matters when it flips a
    # HARD top-k decision that is itself already on a knife's edge -- observed
    # on 512d2l/192d2l (shallow, heavily distilled+finetuned FIM routers) at
    # rank 63 vs 64 differing by 2.35e-5, *smaller* than the noise floor
    # itself. The flagship's router has a much wider margin (diff stays 0).
    # Fail only on GROSS divergence, which WOULD indicate a real bug.
    assert n_route_diff <= max(5, int(0.01 * 784)), \
        f"[{name}] routing mismatch on {n_route_diff}/784 tokens -- too many for float " \
        "noise, do not ship"
    assert recon_diff < 0.1, f"[{name}] round-trip diverges by {recon_diff:.3e} -- do not ship"
    print(f"  [{name}] PARITY OK ({n_route_diff} boundary-token diffs are expected float noise).")

    torch.save(enc_sd, out / f"encoder_{name}.pt")
    torch.save(dec_sd, out / f"decoder_{name}.pt")
    esz = (out / f"encoder_{name}.pt").stat().st_size / 1e6
    dsz = (out / f"decoder_{name}.pt").stat().st_size / 1e6
    print(f"  wrote encoder_{name}.pt ({esz:.1f}MB), decoder_{name}.pt ({dsz:.1f}MB)")
    return dict(name=name, encoder_embed_dim=embed_dim, encoder_depth=depth,
                encoder_num_heads=heads, encoder_params_m=sum(v.numel() for v in enc_sd.values())/1e6)


def extract_random_init(name, embed_dim, depth, heads, out: Path):
    """No real checkpoint exists -- latency-only entry, random-init weights.
    No parity check possible (there is no real deployed model to compare
    against); NOT valid for any accuracy claim, only for latency."""
    print(f"\n--- {name}: encoder_embed_dim={embed_dim} depth={depth} heads={heads} "
          f"(RANDOM INIT -- no checkpoint on disk, latency-only) ---")
    arch = arch_for(embed_dim, depth, heads)
    enc = AquaSCEdgeEncoder(arch).eval()
    n_params = sum(p.numel() for p in enc.parameters())
    print(f"  encoder-only: {n_params/1e6:.2f}M params (random init)")
    torch.save(enc.state_dict(), out / f"encoder_{name}.pt")
    esz = (out / f"encoder_{name}.pt").stat().st_size / 1e6
    print(f"  wrote encoder_{name}.pt ({esz:.1f}MB, random init -- NOT a trained checkpoint)")
    return dict(name=name, encoder_embed_dim=embed_dim, encoder_depth=depth,
                encoder_num_heads=heads, encoder_params_m=n_params/1e6, random_init=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = []
    for name, embed_dim, depth, heads, ckpt in MODELS:
        manifest.append(extract_one(name, embed_dim, depth, heads, ckpt, out))
    for name, embed_dim, depth, heads in RANDOM_INIT_MODELS:
        manifest.append(extract_random_init(name, embed_dim, depth, heads, out))

    import json
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    shutil.copy(ROOT / "model_util.py", out / "model_util.py")
    shutil.copy(ROOT / "digital_channel.py", out / "digital_channel.py")
    bench_dir = Path(__file__).resolve().parent
    for fname in ("edge_encoder_decoder.py", "bench_pytorch.py", "bench_onnx.py",
                  "bench_jscc.py", "bench_swinjscc.py", "bench_roijscc.py"):
        src = bench_dir / fname
        if src.exists():
            shutil.copy(src, out / fname)

    # timm shim: model_util.py imports `from timm.models.layers import
    # drop_path, to_2tuple, trunc_normal_` -- real timm is a heavier dep the
    # Pi doesn't otherwise need. These three are simple, well-known utility
    # functions (stochastic depth, tuple broadcast, truncated-normal init);
    # loaded checkpoint weights overwrite any init() output anyway, so exact
    # init-time RNG behavior is irrelevant here. Shimming avoids an extra
    # install on the Pi while keeping model_util.py itself byte-identical to
    # the real repo (no fork/edit of the source of truth).
    timm_dir = out / "timm" / "models"
    timm_dir.mkdir(parents=True, exist_ok=True)
    (out / "timm" / "__init__.py").write_text("")
    (out / "timm" / "models" / "__init__.py").write_text("")
    (out / "timm" / "models" / "layers.py").write_text(
        "import torch\n"
        "import torch.nn.functional as F\n"
        "\n"
        "def to_2tuple(x):\n"
        "    return x if isinstance(x, (tuple, list)) else (x, x)\n"
        "\n"
        "def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):\n"
        "    return torch.nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)\n"
        "\n"
        "def drop_path(x, drop_prob: float = 0., training: bool = False):\n"
        "    if drop_prob == 0. or not training:\n"
        "        return x\n"
        "    keep_prob = 1 - drop_prob\n"
        "    shape = (x.shape[0],) + (1,) * (x.ndim - 1)\n"
        "    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)\n"
        "    random_tensor.floor_()\n"
        "    return x.div(keep_prob) * random_tensor\n"
        "\n"
        "class DropPath(torch.nn.Module):\n"
        "    def __init__(self, drop_prob=None):\n"
        "        super().__init__()\n"
        "        self.drop_prob = drop_prob\n"
        "    def forward(self, x):\n"
        "        return drop_path(x, self.drop_prob, self.training)\n"
    )

    print(f"\nWrote self-contained payload to {out} for {len(manifest)} AquaSC models, "
          f"plus model_util.py, digital_channel.py, edge_encoder_decoder.py, bench scripts, timm shim")


if __name__ == "__main__":
    main()
