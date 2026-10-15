"""Dev-machine only. Copies the MINIMAL source files needed to construct each
competitor's encoder (JSCC-CNN, SwinJSCC, ROI-JSCC) on the Pi -- not their
full competitors/ directories (which include hundreds of MB of cached
outputs/checkpoints we don't need).

Latency depends only on architecture (shapes/ops), not weight VALUES, so
these are all benchmarked with fresh random-init weights at their real
deployed config -- no checkpoint transfer needed, verified against each
competitor's own build_net()/build_model() helper (already-proven code used
throughout this project), which itself never uses the checkpoint to
determine architecture (only to fill values via load_state_dict after
construction).

Usage: python3 extract_competitor_sources.py --out_dir /tmp/claude-1000/edge_bench_payload
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path

ROOT = Path("__AQUASC_ROOT__")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)

    # JSCC-CNN: model.py needs channel.py + (optionally) utils.py.
    jscc_dir = out / "competitor_jscc"
    jscc_dir.mkdir(parents=True, exist_ok=True)
    for f in ("model.py", "channel.py", "utils.py"):
        shutil.copy(ROOT / "competitors" / "jscc" / f, jscc_dir / f)

    # SwinJSCC: the whole swinjscc_code package (148K, already small).
    swin_src = ROOT / "competitors" / "swinnnn" / "swinjscc_code"
    swin_dst = out / "competitor_swinjscc" / "swinjscc_code"
    if swin_dst.exists():
        shutil.rmtree(swin_dst)
    shutil.copytree(swin_src, swin_dst, ignore=shutil.ignore_patterns("__pycache__"))
    # net/encoder.py hardcodes .cuda() in a few window-mask helpers (assumed
    # GPU-only training/eval elsewhere in this project); patch this COPY only
    # (not the original competitors/ source, which stays GPU-only for the
    # real accuracy evaluations) to run on CPU-only edge hardware.
    enc_py = swin_dst / "net" / "encoder.py"
    src = enc_py.read_text()
    src = src.replace(".cuda()", ".to('cuda' if torch.cuda.is_available() else 'cpu')")
    # x.get_device() returns -1 for CPU tensors (not a valid device index for
    # .to()) -- x.device is the portable equivalent for both CPU and CUDA.
    src = src.replace("device = x.get_device()", "device = x.device")
    enc_py.write_text(src)

    # ROI-JSCC: model/ + aquasc_stubs/ + roijscc_fish_common.py (glymur-stubbed,
    # self-contained per its own docstring -- no loss_maker/hydra/albumentations).
    roi_dst = out / "competitor_roijscc"
    roi_dst.mkdir(parents=True, exist_ok=True)
    for sub in ("model", "aquasc_stubs"):
        d = roi_dst / sub
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(ROOT / "competitors" / "ROI-JSCC" / sub, d,
                        ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(ROOT / "competitors" / "ROI-JSCC" / "roijscc_fish_common.py",
               roi_dst / "roijscc_fish_common.py")

    total = sum(f.stat().st_size for f in out.rglob("*")
                if f.is_file() and "competitor_" in str(f))
    print(f"Copied competitor sources ({total/1e6:.2f}MB total) to {out}: "
          f"competitor_jscc/, competitor_swinjscc/, competitor_roijscc/")


if __name__ == "__main__":
    main()
