# gen_channel_bank.py — generate the production channel_bank.npz (Wave 8 STATS).
# Run: python gen_channel_bank.py  (CPU, numpy only, ~1-3 min, writes ~1.3 GB)

import os
import sys
import time

import channel_bank as CB

TEST_IMG = "__AQUASC_DATA__/yolo_fish_dataset_root/test/images"
OUT = CB.default_bank_path()


def main():
    names = sorted(f for f in os.listdir(TEST_IMG)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))
    assert len(names) == 432, f"expected 432 test images, got {len(names)}"
    print(f"test images: {len(names)} (first={names[0]}, last={names[-1]})")
    t0 = time.time()
    meta = CB.build_channel_bank(OUT, n_images=432, image_names=names)
    dt = time.time() - t0
    size_mb = os.path.getsize(OUT) / 1e6
    print(f"wrote {OUT} ({size_mb:.1f} MB) in {dt:.1f}s")
    print(meta)
    # verify one cell round-trips
    bank = CB.ChannelBank(OUT)
    assert len(bank.image_names) == 432
    assert bank.image_names == names
    print("verification: first cell h =", bank.get_h(0, -5, 0))
    print("verification: subseed(0,-5,0) =", bank.get_subseed(0, -5, 0))
    bank.close()
    print("gen_channel_bank.py DONE")


if __name__ == "__main__":
    main()
