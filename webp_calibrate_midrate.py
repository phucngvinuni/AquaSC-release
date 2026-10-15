# webp_calibrate_midrate.py -- 2026-09-25. Calibrate the highest WebP
# quality whose MAXIMUM byte size across the 437-image Site A VALIDATION set
# (never the test set, per the same convention already used for the
# rate-matched headline point, webp_real_channel_sweep_stage1.py) stays
# within AquaSC's mid-rate (r061, 0.061224 uses/px, QPSK) budget.
#
# Budget derivation (matches §1d's headline derivation, halved): AquaSC's
# r061 point uses QPSK + rate-1/2 LDPC BG2 at symbols=3072 (half the
# headline's 6144) = 0.061224 uses/px. At QPSK (2 bits/symbol) this is
# 3072*2=6144 coded bits -> at k_cb=1024 (rate-1/2 -> 2048 coded bits/block)
# that's exactly 3 LDPC blocks -> 3*(1024-32)=2976 payload bits = 372 bytes.
import io
import os

from PIL import Image

VALID_ROOT = "__AQUASC_DATA__/yolo_fish_dataset_root/valid/images"
CAPACITY_BYTES = 372
IMG = 224


def load_resized(path):
    im = Image.open(path).convert("RGB")
    if im.size != (IMG, IMG):
        im = im.resize((IMG, IMG), Image.BICUBIC)
    return im


def main():
    names = sorted(f for f in os.listdir(VALID_ROOT) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    print(f"Calibrating on {len(names)} validation images, target capacity {CAPACITY_BYTES}B")

    best_q = None
    for q in range(1, 101):
        max_bytes = 0
        n_over = 0
        for name in names:
            im = load_resized(os.path.join(VALID_ROOT, name))
            buf = io.BytesIO()
            im.save(buf, "WEBP", quality=q, method=4)
            sz = len(buf.getvalue())
            max_bytes = max(max_bytes, sz)
            if sz > CAPACITY_BYTES:
                n_over += 1
        fits = n_over == 0
        print(f"q={q:3d}  max={max_bytes:4d}B  n_over_budget={n_over}/{len(names)}  fits={fits}")
        if fits:
            best_q = q
        else:
            # quality is monotonic in size (higher q -> bigger files), so the
            # first failure means every higher q will also fail -- stop early.
            break

    print(f"\nSelected quality={best_q} (highest quality whose max validation byte size stays <= {CAPACITY_BYTES}B)")


if __name__ == "__main__":
    main()
