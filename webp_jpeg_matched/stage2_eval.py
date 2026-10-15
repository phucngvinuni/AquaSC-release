"""WebP/JPEG matched-budget rerun, stage 2 (decode + detector + PSNR/SSIM). siso-project-env:
  taskset -c 28-31 __TORCH_PY__ webp_jpeg_matched/stage2_eval.py \
      --codec webp --budget mid

Reads runs/<codec>_<budget>/{config.json,rx_snr*.npz}; writes metrics.npz / summary.json in the same directory.
Same scoring as experiments/matched5/evaluate.py evaluate_aquasc: detector best.pt (conf .001, iou .5),
stats.per_image_ap50 per image, torchmetrics pooled mAP@0.5 per draw, PSNR/SSIM via utils.calc_psnr/calc_ssim
against the dataset original (ds[i][1][0]). EVERY (snr, draw, image) record is scored (capacity failures,
CRC/length failures and PIL decode failures -> black image), denominator is always 432.
Exact-result cache: images that arrive byte-identical to the clean compressed file (or black) reuse the identical,
deterministic prediction instead of re-running the detector; corrupted-but-decoded images are always scored fresh.
"""
import argparse, hashlib, io, json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "experiments" / "matched5"))
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from PIL import Image

import metrics                  # noqa: E402
import stats                    # noqa: E402
import utils as aq_utils        # noqa: E402
from evaluate import dataset, detector, sha256   # noqa: E402
from torchmetrics.detection import MeanAveragePrecision   # noqa: E402

IMG = 224
SNRS = [-5, 0, 5, 10, 15, 20]
MAX_PIXELS = 4096 * 4096


def decode_bytes(b):
    """-> (PIL RGB 224x224 or None, ok). Anything PIL cannot decode (or an absurd corrupted size) is a decode failure."""
    try:
        im = Image.open(io.BytesIO(b))
        if im.size[0] * im.size[1] > MAX_PIXELS or im.size[0] < 1 or im.size[1] < 1:
            return None, False
        im.load()
        im = im.convert("RGB")
        if im.size != (IMG, IMG):
            im = im.resize((IMG, IMG), Image.BICUBIC)
        return im, True
    except Exception:
        return None, False


def to_tensor(im):
    return torch.from_numpy(np.asarray(im, dtype=np.uint8).copy()).permute(2, 0, 1).float().div(255.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codec", required=True)
    ap.add_argument("--budget", required=True)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--snr", type=int, nargs="*", default=SNRS)
    ap.add_argument("--batch", type=int, default=16)
    cli = ap.parse_args()
    run = Path(cli.run_dir) if cli.run_dir else HERE / "runs" / f"{cli.codec}_{cli.budget}"
    cfg = json.loads((run / "config.json").read_text())
    n_img, M = cfg["n_images"], cfg["M"]
    src = np.load(run / "source_files.npz")
    names = src["names"].tolist()

    ds = dataset("test", "a")
    assert names == [Path(p).name for p in ds.img_files][:n_img], "image order differs from dataset / channel bank"
    gts = [ds[i][1][1] for i in range(n_img)]
    originals = [ds[i][1][0] for i in range(n_img)]
    yolo = detector("a")

    # clean compressed file per image (re-encoded from the same source with the locked q/options -> identical bytes)
    # (stage1_phy imports TensorFlow, which is absent from this env: same two helpers are re-stated here, verbatim;
    # the size check below guards against drift)
    TEST_IMAGES = cfg["test_images"]

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

    clean_bytes = [encode(load_resized(f"{TEST_IMAGES}/{n}"), cfg["codec"], cfg["quality"]) for n in names]
    assert [len(b) for b in clean_bytes] == src["sizes"].tolist()
    cap_fail = src["capacity_failure"].astype(bool)

    def predict(batch_t):
        return metrics.yolo_predict(yolo, batch_t.cuda(), conf=.001, iou=.5, device="cuda")

    def score(pred, i, rec_t):
        ap50 = stats.per_image_ap50(pred["boxes"].cpu().numpy(), pred["scores"].cpu().numpy(),
                                    gts[i]["boxes"].numpy(), iou_thr=.5)
        p = aq_utils.calc_psnr(rec_t, originals[i]); s = aq_utils.calc_ssim(rec_t, originals[i])
        return ap50, float(p[0] if isinstance(p, list) else p), float(s[0] if isinstance(s, list) else s)

    cache = {}      # (i, "clean"|"black") -> (pred, ap50, psnr, ssim)
    t0 = time.monotonic()
    black = torch.zeros(3, IMG, IMG)
    with torch.no_grad():
        for kind in ("black", "clean"):
            for o in range(0, n_img, cli.batch):
                idx = list(range(o, min(n_img, o + cli.batch)))
                ts = []
                for i in idx:
                    if kind == "black" or cap_fail[i]:
                        ts.append(black)
                    else:
                        im, ok = decode_bytes(clean_bytes[i]); assert ok; ts.append(to_tensor(im))
                preds = predict(torch.stack(ts))
                for i, t, pr in zip(idx, ts, preds):
                    cache[(i, kind)] = (pr, *score(pr, i, t))
    clean_ap = np.array([cache[(i, "clean")][1] for i in range(n_img)])
    clean_psnr = np.array([cache[(i, "clean")][2] for i in range(n_img)])
    clean_ssim = np.array([cache[(i, "clean")][3] for i in range(n_img)])
    print(json.dumps(dict(stage="clean_ref", mean_ap50=float(clean_ap.mean()), psnr=float(clean_psnr.mean()),
                          ssim=float(clean_ssim.mean()), seconds=round(time.monotonic() - t0, 1))), flush=True)

    n_s = len(cli.snr)
    out = dict(per_image_ap=np.zeros((n_s, n_img, M), np.float32), psnr=np.zeros((n_s, n_img, M), np.float32),
               ssim=np.zeros((n_s, n_img, M), np.float32), pooled_map50=np.zeros((n_s, M), np.float32),
               status=np.zeros((n_s, n_img, M), np.int8), n_failed_blocks=np.zeros((n_s, n_img, M), np.int16),
               decode_ok=np.zeros((n_s, n_img, M), bool), received_len=np.zeros((n_s, n_img, M), np.int32),
               symbols_emitted=np.zeros((n_s, n_img, M), np.int32), recon_class=np.zeros((n_s, n_img, M), np.int8))
    # recon_class: 0 black/fallback, 1 byte-identical to clean file, 2 corrupted-but-decoded
    summ = []
    with torch.no_grad():
        for si, snr in enumerate(cli.snr):
            rx = np.load(run / f"rx_snr{snr}.npz")
            status, lens, blob = rx["status"], rx["lens"], rx["blob"]
            out["status"][si], out["n_failed_blocks"][si], out["symbols_emitted"][si] = status, rx["n_failed_blocks"], rx["symbols_emitted"]
            pos = 0
            starts = np.full((n_img, M), -1, np.int64)
            for m in range(M):
                for i in range(n_img):
                    if lens[i, m] >= 0:
                        starts[i, m] = pos; pos += int(lens[i, m])
            assert pos == blob.size
            t1 = time.monotonic()
            for m in range(M):
                metric = MeanAveragePrecision(iou_type="bbox").cuda()
                pending, rec_all = [], {}
                for i in range(n_img):
                    L = int(lens[i, m])
                    if L < 0:
                        key = (i, "black"); rc, ok = 0, False
                    else:
                        b = blob[starts[i, m]:starts[i, m] + L].tobytes()
                        out["received_len"][si, i, m] = L
                        if b == clean_bytes[i] and not cap_fail[i]:
                            key = (i, "clean"); rc, ok = 1, True
                        else:
                            im, ok = decode_bytes(b)
                            if ok:
                                key = ("fresh", i); rc = 2; pending.append((i, to_tensor(im)))
                            else:
                                key = (i, "black"); rc = 0
                    rec_all[i] = (key, rc, ok)
                fresh = {}
                for o in range(0, len(pending), cli.batch):
                    chunk = pending[o:o + cli.batch]
                    preds = predict(torch.stack([t for _, t in chunk]))
                    for (i, t), pr in zip(chunk, preds):
                        fresh[i] = (pr, *score(pr, i, t))
                for i in range(n_img):
                    key, rc, ok = rec_all[i]
                    pr, ap50, ps, ss = fresh[i] if key[0] == "fresh" else cache[key]
                    out["per_image_ap"][si, i, m] = ap50; out["psnr"][si, i, m] = ps; out["ssim"][si, i, m] = ss
                    out["decode_ok"][si, i, m] = ok; out["recon_class"][si, i, m] = rc
                    metric.update([dict(boxes=pr["boxes"], scores=pr["scores"],
                                        labels=torch.zeros(len(pr["scores"]), dtype=torch.int64, device="cuda"))],
                                  [dict(boxes=gts[i]["boxes"].cuda(), labels=gts[i]["labels"].cuda())])
                out["pooled_map50"][si, m] = float(metric.compute()["map_50"])
            row = dict(snr=snr, mean_per_image_ap50=float(out["per_image_ap"][si].mean()),
                       pooled_map50=float(out["pooled_map50"][si].mean()), psnr=float(out["psnr"][si].mean()),
                       ssim=float(out["ssim"][si].mean()), n_records=int(n_img * M),
                       frac_black=float((out["recon_class"][si] == 0).mean()),
                       frac_clean=float((out["recon_class"][si] == 1).mean()),
                       frac_corrupted_decoded=float((out["recon_class"][si] == 2).mean()),
                       frac_capacity_failure=float(cap_fail.mean()),
                       frac_any_block_failed=float((out["n_failed_blocks"][si] > 0).mean()),
                       symbols_min_mean_max=[int(out["symbols_emitted"][si].min()), float(out["symbols_emitted"][si].mean()),
                                             int(out["symbols_emitted"][si].max())],
                       seconds=round(time.monotonic() - t1, 1))
            summ.append(row); print(json.dumps(row), flush=True)
    np.savez_compressed(run / "metrics.npz", names=np.array(names), snr_db=np.array(cli.snr), clean_ap=clean_ap,
                        clean_psnr=clean_psnr, clean_ssim=clean_ssim, **out)
    summary = dict(config=cfg, per_snr=summ, clean_mean_ap50=float(clean_ap.mean()), clean_psnr=float(clean_psnr.mean()),
                   clean_ssim=float(clean_ssim.mean()), detector_sha256=sha256(ROOT / "best.pt"), metric_conf=.001,
                   metric_iou=.5, metrics_sha256=sha256(run / "metrics.npz"), stage2_script_sha256=sha256(__file__),
                   rx_sha256={f"rx_snr{s}": sha256(run / f"rx_snr{s}.npz") for s in cli.snr},
                   n_records=int(n_s * n_img * M), axes="per_image_ap/psnr/ssim: [snr, image, draw]; pooled_map50: [snr, draw]")
    (run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("wrote", run / "summary.json", flush=True)


if __name__ == "__main__":
    main()
