# build_combined_root.py - Wave-6 TRANSFER: combine Site A + Site B into one
# video-disjoint YOLO root under /tmp using SYMLINKS (source roots untouched).
# train = A_train U B_train ; val = A_valid U B_valid ; test = A_test U B_test.
import os
import json
import shutil

SITE_A = "__AQUASC_DATA__/yolo_fish_dataset_root"
SITE_B = "__AQUASC_DATA__/yolo_fishdet_nb2_vd"
DST = "__AQUASC_TMP__/yolo_bothsites"

SPLITS = ["train", "valid", "test"]
MAP = {}


def link_all(src_root, dst_root):
    n = 0
    for sub in ("images", "labels"):
        src_dir = os.path.join(src_root, sub)
        dst_dir = os.path.join(dst_root, sub)
        os.makedirs(dst_dir, exist_ok=True)
        for f in sorted(os.listdir(src_dir)):
            ext = os.path.splitext(f)[1].lower()
            if ext not in (".jpg", ".jpeg", ".png", ".txt"):
                continue
            src = os.path.join(src_dir, f)
            dst = os.path.join(dst_dir, f)
            if not os.path.exists(dst):
                os.symlink(src, dst)
                n += 1
    return n


def main():
    if os.path.exists(DST):
        shutil.rmtree(DST)
    counts = {}
    total = 0
    for sp in SPLITS:
        dst = os.path.join(DST, sp)
        os.makedirs(os.path.join(dst, "images"), exist_ok=True)
        os.makedirs(os.path.join(dst, "labels"), exist_ok=True)
        a = link_all(os.path.join(SITE_A, sp), dst)
        b = link_all(os.path.join(SITE_B, sp), dst)
        counts[sp] = {"site_a": a // 2, "site_b": b // 2, "total": (a + b) // 2}
        total += (a + b) // 2
    print(json.dumps(counts, indent=1))
    print("TOTAL", total)
    yaml = f"""path: {DST}
train: train/images
val: valid/images
test: test/images
nc: 1
names:
  0: fish
"""
    with open(os.path.join(DST, "data.yaml"), "w") as f:
        f.write(yaml)
    print("WROTE", os.path.join(DST, "data.yaml"))


if __name__ == "__main__":
    main()
