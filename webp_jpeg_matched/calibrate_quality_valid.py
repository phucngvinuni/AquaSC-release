"""Protocol §5: choose the highest quality q (1..100) per codec whose MAX whole-file size over the 437
Site A VALIDATION images does not exceed the post-metadata capacity (368/740/1484 B). Test split is never read.
Run: __TORCH_PY__ webp_jpeg_matched/calibrate_quality_valid.py
"""
import io, json, os, sys, hashlib
import numpy as np, PIL
from PIL import Image, features

IMG = 224
VALID = "__AQUASC_DATA__/yolo_fish_dataset_root/valid/images"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "valid_calib")
CAPS = {"low": 368, "mid": 740, "high": 1484}

def load_resized(path):
    im = Image.open(path).convert("RGB")
    if im.size != (IMG, IMG):
        im = im.resize((IMG, IMG), Image.BICUBIC)
    return im

def encode(im, codec, q):
    buf = io.BytesIO()
    if codec == "webp":
        im.save(buf, "WEBP", quality=q, method=4)
    else:
        im.save(buf, "JPEG", quality=q, optimize=True)
    return buf.getvalue()

def main():
    names = sorted(f for f in os.listdir(VALID) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    assert len(names) == 437, len(names)
    ims = [load_resized(os.path.join(VALID, n)) for n in names]
    res = {"pil_version": PIL.__version__, "webp_features": {"webp": features.check("webp")},
           "options": {"webp": "quality=q, method=4 (PIL defaults otherwise)", "jpeg": "quality=q, optimize=True (PIL default 4:2:0). optimize=True chosen on validation only: with PIL defaults the floor is ~1440 B/file, so low and mid budgets are 100% infeasible"},
           "n_valid": len(names), "caps_B": CAPS, "tables": {}, "selected": {}}
    for codec in ("webp", "jpeg"):
        mx = np.zeros(100, int); mean = np.zeros(100); sizes = np.zeros((100, len(ims)), int)
        for q in range(1, 101):
            s = np.array([len(encode(im, codec, q)) for im in ims]); sizes[q-1] = s
            mx[q-1] = s.max(); mean[q-1] = s.mean()
        res["tables"][codec] = {"q": list(range(1, 101)), "max_B": mx.tolist(), "mean_B": mean.round(1).tolist(),
                                "p95_B": [int(np.percentile(sizes[q], 95)) for q in range(100)],
                                "min_B": sizes.min(1).tolist()}
        np.save(os.path.join(OUT, f"valid_sizes_{codec}.npy"), sizes)
        for b, cap in CAPS.items():
            ok = [q for q in range(1, 101) if mx[q-1] <= cap]
            if ok:
                q = max(ok); feasible = True
            else:
                q = 1; feasible = False   # protocol: lock q=1 and report; do not drop images
            sz = sizes[q-1]
            res["selected"][f"{codec}_{b}"] = dict(q=q, cap_B=cap, feasible_all_valid=feasible,
                max_valid_B=int(sz.max()), argmax_image=names[int(sz.argmax())], mean_valid_B=float(sz.mean()),
                n_valid_over_cap=int((sz > cap).sum()), frac_valid_over_cap=float((sz > cap).mean()))
            print(codec, b, res["selected"][f"{codec}_{b}"], flush=True)
    json.dump(res, open(os.path.join(OUT, "quality_selection.json"), "w"), indent=2)

if __name__ == "__main__":
    main()
