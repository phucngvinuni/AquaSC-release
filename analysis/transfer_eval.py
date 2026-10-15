import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repo root (core modules)
# analysis/transfer_eval.py - Wave-6 TRANSFER: zero-shot AquaSC (AUG ckpt) eval over an
# arbitrary YOLO split (Site B test / IR test) at multiple SNRs (analog proxy),
# scored with MULTIPLE detectors at conf 0.001 (torchmetrics protocol, class fish).
# Also computes FIM patch-level precision/recall/F1 at tau (channel-independent).
# Saves per-image {orig, recon, boxes, labels} .pt files for provenance.
#
# mAP uses torchmetrics MeanAveragePrecision directly (identical to metrics.py's
# map_50/map columns); in-box PSNR/SSIM/LPIPS are SKIPPED here to keep the dense
# Site-B split (11k boxes) tractable - they are not needed for retention ratios.
#
# Usage:
#   python analysis/transfer_eval.py --checkpoint <ckpt> --data_path <root> \
#       --detectors best.pt,best_siteB.pt,best_bothsites.pt \
#       --detector_names siteA,siteB,both --snrs 0 10 20 \
#       --out_csv results.csv --fim_csv fim.csv --save_recons <dir>
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torchmetrics.detection import MeanAveragePrecision

import datasets
import metrics
import stats
import utils
import model as _model  # noqa: F401  (registers ViT_Reconstruction_Model_Default with timm)
from ultralytics import YOLO


def build_model_args(data_path, device, snr):
    a = argparse.Namespace()
    a.model = 'ViT_Reconstruction_Model_Default'
    a.data_set = 'fish'
    a.data_path = data_path
    a.input_size = 224
    a.patch_size = 8
    a.mask_ratio = 0.0
    a.eval = True
    a.num_object_classes = 1
    a.encoder_embed_dim = 512
    a.encoder_depth = 8
    a.encoder_num_heads = 8
    a.decoder_embed_dim = 512
    a.decoder_depth = 6
    a.decoder_num_heads = 8
    a.quantizer_dim = 512
    a.bits_vq_high = 10
    a.bits_vq_low = 2
    a.quantizer_type = 'rvq'
    a.rvq_stages = 3
    a.rvq_bits_per_stage = 6
    a.rvq_ema_decay = 0.0
    a.quantizer_commitment_cost = 0.25
    a.drop_rate = 0.0
    a.drop_path_rate = 0.1
    a.fim_embed_dim = 256
    a.fim_depth = 2
    a.fim_num_heads = 4
    a.fim_drop_rate = 0.1
    a.fim_routing_threshold = 0.6
    a.routing_topk = 63
    a.eps_stab = 1e-6
    a.device = device
    a.snr_db_eval = snr
    a.seed = 42
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--data_path', required=True)
    ap.add_argument('--detectors', required=True, help='comma-separated .pt paths')
    ap.add_argument('--detector_names', required=True, help='comma-separated labels')
    ap.add_argument('--snrs', nargs='+', type=float, default=[0, 10, 20])
    ap.add_argument('--out_csv', required=True)
    ap.add_argument('--fim_csv', required=True)
    ap.add_argument('--fim_tau', type=float, default=0.7)
    ap.add_argument('--save_recons', default='')
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--per_image_ap_out_prefix', default='',
                    help='S-1 (Wave 14): if set, save per-image AP@0.5 arrays as '
                         '<prefix>_orig_<detector>.npy and <prefix>_recon_snr<snr>_<detector>.npy')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    det_paths = args.detectors.split(',')
    det_names = args.detector_names.split(',')
    assert len(det_paths) == len(det_names)
    yolos = {}
    for name, p in zip(det_names, det_paths):
        yolos[name] = YOLO(p)
        yolos[name].model.eval()

    ma = build_model_args(args.data_path, str(device), args.snrs[0])
    model = utils.get_model(ma)
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    sd = ck['model']
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"checkpoint load: missing={missing} unexpected={unexpected}")
    assert not missing and not unexpected, "checkpoint<->config mismatch"
    model.to(device)
    model.eval()

    ds = datasets.build_dataset(is_train=False, args=ma)
    print(f"split size: {len(ds)}")
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                         num_workers=0, drop_last=False,
                                         collate_fn=datasets.yolo_collate_fn)

    # mAP accumulators: orig_map[name], recon_map[(snr,name)]
    orig_map = {name: MeanAveragePrecision(iou_type='bbox', class_metrics=True).to(device)
                for name in det_names}
    recon_map = {(snr, name): MeanAveragePrecision(iou_type='bbox', class_metrics=True).to(device)
                 for snr in args.snrs for name in det_names}

    # greedy P/R counters (orig/recon) per detector (like metrics._match_tp_fp)
    pr_orig = {name: [0, 0, 0] for name in det_names}
    pr_recon = {(snr, name): [0, 0, 0] for snr in args.snrs for name in det_names}

    # S-1 (Wave 14): per-image AP@0.5, additive to the pooled torchmetrics above
    per_image_orig = {name: [] for name in det_names}
    per_image_recon = {(snr, name): [] for snr in args.snrs for name in det_names}

    # FIM accumulation
    fim_tp = fim_fp = fim_fn = 0
    fim_n_patches = 0

    psnr_sum = {snr: 0.0 for snr in args.snrs}
    n_batch = 0

    for batch_idx, (semcom_data_input, targets_tuple) in enumerate(loader):
        (orig, bm_pos) = semcom_data_input
        (recon_target, yolo_gt_list, fim_target_map) = targets_tuple
        orig = orig.to(device, non_blocking=True)
        if bm_pos is not None:
            bm_pos = bm_pos.to(device, non_blocking=True)

        gt_dev = [{'boxes': g['boxes'].to(device), 'labels': g['labels'].to(device)}
                  for g in yolo_gt_list]

        # orig-side detector scoring (channel-independent, once)
        for name in det_names:
            po = metrics.yolo_predict(yolos[name], orig.detach(), conf=0.001, iou=0.5, device=str(device))
            orig_map[name].update(po, gt_dev)
            tp, fp, fn = 0, 0, 0
            for i in range(orig.shape[0]):
                t, f, n, _ = metrics._match_tp_fp(po[i]['boxes'].cpu(), po[i]['scores'].cpu(),
                                                  gt_dev[i]['boxes'].cpu(), iou_thr=0.5, score_thr=0.001)
                tp += t; fp += f; fn += n
                if args.per_image_ap_out_prefix:
                    per_image_orig[name].append(stats.per_image_ap50(
                        po[i]['boxes'].detach().cpu().numpy(), po[i]['scores'].detach().cpu().numpy(),
                        gt_dev[i]['boxes'].detach().cpu().numpy(), iou_thr=0.5))
            pr_orig[name][0] += tp; pr_orig[name][1] += fp; pr_orig[name][2] += fn

        for snr in args.snrs:
            with torch.no_grad():
                # Wave 14 fix (v1, superseded): this call previously omitted digital_seed,
                # defaulting to the SAME seed (42) for every batch.
                #
                # Wave 14 fix (v2, plan v7.3 point 2): the v1 fix used
                # digital_seed=args.seed+batch_idx, which was WRONG despite mirroring
                # engine1.py's call site verbatim -- because run_class_main.py's real
                # eval dataloader is hardcoded to batch_size=1 (run_class_main.py:221),
                # so in production batch_idx IS the global image index and every image
                # gets a unique seed. Here batch_idx only increments once per
                # args.batch_size images (default 8), so transmit_and_receive_indices_batch's
                # internal seed+i (i = position within batch) collides across batches:
                # batch b image i and batch b+1 image i-1 received the identical channel
                # draw. Confirmed empirically (seed search against engine1.py's real
                # per-image output, digital_seed=50 exactly reproduces image 8's result,
                # not 43=args.seed+batch_idx). Fix: derive the seed from the image's
                # global index, matching production regardless of --batch_size.
                out = model(img=orig, bm_pos=bm_pos, _eval=True, eval_snr_db=snr,
                           digital_seed=args.seed + batch_idx * args.batch_size)
            recon = out['reconstructed_image']
            if snr == args.snrs[0]:
                fim_logits = out['fim_importance_scores']
                fim_pred = (torch.sigmoid(fim_logits).squeeze(-1) > args.fim_tau)
                fim_gt = fim_target_map.squeeze(-1).to(fim_pred.device) > 0.5
                fim_n_patches += fim_gt.numel()
                fim_tp += int((fim_pred & fim_gt).sum().item())
                fim_fp += int((fim_pred & ~fim_gt).sum().item())
                fim_fn += int((~fim_pred & fim_gt).sum().item())
            if args.save_recons:
                os.makedirs(os.path.join(args.save_recons, f'snr{snr:g}'), exist_ok=True)
            for name in det_names:
                pr = metrics.yolo_predict(yolos[name], recon.detach(), conf=0.001, iou=0.5, device=str(device))
                recon_map[(snr, name)].update(pr, gt_dev)
                tp, fp, fn = 0, 0, 0
                for i in range(orig.shape[0]):
                    t, f, n, _ = metrics._match_tp_fp(pr[i]['boxes'].cpu(), pr[i]['scores'].cpu(),
                                                      gt_dev[i]['boxes'].cpu(), iou_thr=0.5, score_thr=0.001)
                    tp += t; fp += f; fn += n
                    if args.per_image_ap_out_prefix:
                        per_image_recon[(snr, name)].append(stats.per_image_ap50(
                            pr[i]['boxes'].detach().cpu().numpy(), pr[i]['scores'].detach().cpu().numpy(),
                            gt_dev[i]['boxes'].detach().cpu().numpy(), iou_thr=0.5))
                pr_recon[(snr, name)][0] += tp; pr_recon[(snr, name)][1] += fp; pr_recon[(snr, name)][2] += fn
            psnr_sum[snr] += float(np.mean(utils.calc_psnr(recon.detach().cpu(), recon_target.detach().cpu())))
            if args.save_recons:
                for i in range(orig.shape[0]):
                    gi = batch_idx * args.batch_size + i
                    torch.save({'orig': orig[i].detach().cpu(), 'recon': recon[i].detach().cpu(),
                                'boxes': yolo_gt_list[i]['boxes'].detach().cpu(),
                                'labels': yolo_gt_list[i]['labels'].detach().cpu()},
                               os.path.join(args.save_recons, f'snr{snr:g}', f'img_{gi:05d}.pt'))
        n_batch += 1
        if (batch_idx + 1) % 20 == 0:
            print(f"  processed {batch_idx + 1}/{len(loader)}", flush=True)

    def _pr(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) else float('nan')
        r = tp / (tp + fn) if (tp + fn) else float('nan')
        f1 = 2 * p * r / (p + r) if (p + r) else float('nan')
        return p, r, f1

    orig_res = {name: orig_map[name].compute() for name in det_names}
    rows = []
    for snr in args.snrs:
        for name in det_names:
            mo = orig_res[name]
            mr = recon_map[(snr, name)].compute()
            map50_o = float(mo['map_50'])
            map50_r = float(mr['map_50'])
            p, r, f1 = _pr(*pr_recon[(snr, name)])
            rows.append({
                'snr_db': snr, 'detector': name,
                'map_50_orig': map50_o, 'map_50_recon': map50_r,
                'map_orig': float(mo['map']), 'map_recon': float(mr['map']),
                'retention_50': map50_r / map50_o if map50_o > 0 else float('nan'),
                'retention': (float(mr['map']) / float(mo['map'])) if float(mo['map']) > 0 else float('nan'),
                'psnr': psnr_sum[snr] / n_batch,
                'precision_recon': p, 'recall_recon': r, 'f1_recon': f1,
            })
    with open(args.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"WROTE {args.out_csv}")

    if args.per_image_ap_out_prefix:
        for name in det_names:
            np.save(f"{args.per_image_ap_out_prefix}_orig_{name}.npy",
                    np.asarray(per_image_orig[name], dtype=np.float32))
        for (snr, name), vals in per_image_recon.items():
            np.save(f"{args.per_image_ap_out_prefix}_recon_snr{snr:g}_{name}.npy",
                    np.asarray(vals, dtype=np.float32))
        print(f"WROTE per-image AP arrays with prefix {args.per_image_ap_out_prefix}")

    for name in det_names:
        p, r, f1 = _pr(*pr_orig[name])
        print(f"  [orig ceiling] {name}: map_50={float(orig_res[name]['map_50']):.6f} P={p:.4f} R={r:.4f}")

    p = fim_tp / (fim_tp + fim_fp) if (fim_tp + fim_fp) else float('nan')
    r = fim_tp / (fim_tp + fim_fn) if (fim_tp + fim_fn) else float('nan')
    f1 = 2 * p * r / (p + r) if (p + r) else float('nan')
    with open(args.fim_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['tau', 'n_patches', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1'])
        w.writeheader()
        w.writerow({'tau': args.fim_tau, 'n_patches': fim_n_patches, 'tp': fim_tp,
                    'fp': fim_fp, 'fn': fim_fn, 'precision': p, 'recall': r, 'f1': f1})
    print(f"WROTE {args.fim_csv}: precision={p:.4f} recall={r:.4f} f1={f1:.4f}")


if __name__ == '__main__':
    main()
