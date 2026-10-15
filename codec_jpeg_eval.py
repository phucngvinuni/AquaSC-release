# codec_jpeg_eval.py -- JPEG q90 clean-channel mAP, for the §6af split-computing
# "Server-only" scenario's performance column (RESULTS_OF_RECORD.md). Same
# protocol/scoring as codec_baseline.py (WebP, §1b) so the two are directly
# comparable -- reuses its list_images/load_boxes/load_resized helpers,
# swaps only the codec (JPEG q90 instead of WebP).
import io
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M  # noqa: E402
from codec_baseline import list_images, load_boxes, load_resized, bpp_of  # noqa: E402

IMG = 224
DATA_ROOT = "__AQUASC_DATA__/yolo_fish_dataset_root/test"
CANONICAL_CEILING = 0.7292


def encode_jpeg(pil_img, quality):
    buf = io.BytesIO()
    pil_img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def main():
    quality = 90
    names = list_images(DATA_ROOT)
    to_tensor = transforms.ToTensor()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    yolo = YOLO("best.pt")
    yolo.model.eval()
    acc = M.MetricsAccumulator(device=str(device), conf_thres=0.001, iou_thres=0.5)

    sizes = []
    with torch.no_grad():
        for n in names:
            stem = os.path.splitext(n)[0]
            im = load_resized(os.path.join(DATA_ROOT, "images", n))
            data = encode_jpeg(im, quality)
            sizes.append(len(data))
            recon = Image.open(io.BytesIO(data)).convert("RGB")
            recon_t = to_tensor(recon).unsqueeze(0)
            orig_t = to_tensor(im).unsqueeze(0)
            boxes, labels = load_boxes(os.path.join(DATA_ROOT, "labels", stem + ".txt"))
            gt = [{"boxes": boxes, "labels": labels}]
            orig_d = orig_t.to(device); recon_d = recon_t.to(device)
            preds_orig = M.yolo_predict(yolo, orig_d, conf=0.001, iou=0.5, device=device)
            preds_recon = M.yolo_predict(yolo, recon_d, conf=0.001, iou=0.5, device=device)
            acc.accumulate(orig_d, recon_d, gt, preds_orig, preds_recon)

    res = acc.compute()
    mean_bytes = float(np.mean(sizes))
    mean_bpp = mean_bytes * 8.0 / (IMG * IMG)
    retention = res["map_50_recon"] / CANONICAL_CEILING
    print(f"JPEG q={quality}  mean_bytes={mean_bytes:.1f}  mean_BPP={mean_bpp:.4f}  "
          f"mAP@0.5_recon={res['map_50_recon']:.4f}  mAP@0.5_orig={res['map_50_orig']:.4f}  "
          f"retention_vs_{CANONICAL_CEILING}={retention:.4f}  PSNR={res['psnr']:.3f} dB  "
          f"n={len(names)}")


if __name__ == "__main__":
    main()
