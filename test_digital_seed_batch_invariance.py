# test_digital_seed_batch_invariance.py -- regression test for the batch-size
# collision bug (plan v7.4 / RESULTS_OF_RECORD.md §6d-3/6d-5).
#
# The bug: a per-image channel seed derived as `args.seed + batch_idx` is only
# a per-image-unique seed if batch_size==1. `engine1.py`'s real eval path was
# "correct" only because run_class_main.py's eval dataloader hardcodes
# batch_size=1 -- correct by coincidence, not by formula. transfer_eval.py made
# the same mistake with its own batch_size=8 loader and produced a real, wrong,
# reported number (a 63x error in one retention cell). Fixed in engine1.py by
# deriving the seed from the true per-batch image count
# (`args.seed + batch_idx * original_imgs_tensor.shape[0]`), which reduces to
# the per-image global index regardless of batch size.
#
# This test makes the bug class impossible to reintroduce silently: it runs
# the identical small eval slice through `engine1.evaluate_semcom_with_yolo`
# at batch_size=1 and batch_size=4 (including a non-uniform last batch, since
# 10 images at batch_size=4 gives a final batch of size 2) and asserts the
# per-image AP array is IDENTICAL between the two runs. If anyone changes the
# digital_seed derivation back to a bare `args.seed + batch_idx`, or changes
# run_class_main.py's hardcoded batch_size=1 without also fixing engine1.py,
# this test fails.
#
# Run: __TORCH_PY__ test_digital_seed_batch_invariance.py
# Needs: a real checkpoint + YOLO weights (uses the flagship checkpoint and
# best.pt already used throughout this project's eval scripts) and the Site A
# test split. Not CPU-only-fast like test_stats.py; this exercises the real
# model, deliberately, since the bug is only visible through the real forward
# path, not through a formula read in isolation.

import os
import sys
import unittest

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import datasets
import engine1
import model as _model  # noqa: F401  (registers ViT_Reconstruction_Model_Default with timm)
import utils

CHECKPOINT = "ckpt_semcom_reconstruction_yolo_fish_MIMO_LPIPS_FINAL/checkpoint-best_psnr.pth"
DATA_PATH = "../yolo_fish_dataset_root/"
YOLO_WEIGHTS = "best.pt"
N_IMAGES = 10  # deliberately not a multiple of 4, to exercise a non-uniform last batch
SNR_DB = 20.0
SEED = 42


def _build_args():
    import argparse
    a = argparse.Namespace()
    a.model = 'ViT_Reconstruction_Model_Default'
    a.data_set = 'fish'
    a.data_path = DATA_PATH
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
    a.rvq_ema_decay = 0.99
    a.quantizer_commitment_cost = 0.25
    a.drop_rate = 0.0
    a.drop_path_rate = 0.1
    a.fim_embed_dim = 256
    a.fim_depth = 2
    a.fim_num_heads = 4
    a.fim_drop_rate = 0.1
    a.fim_routing_threshold = 0.7
    a.routing_topk = 63
    a.eps_stab = 1e-6
    a.disable_fim = False
    a.disable_bbw = False
    a.inside_box_loss_weight = 10
    a.outside_box_loss_weight = 0.1
    a.lpips_loss_weight = 0.0
    a.snr_db_eval = SNR_DB
    a.seed = SEED
    a.yolo_conf_thres = 0.4
    a.yolo_iou_thres = 0.5
    a.map_conf_thres = 0.001
    a.digital_ldpc = False
    a.digital_csi_mode = 'perfect'
    a.digital_bits_per_symbol = 4
    a.digital_payload_rate = 0.5
    a.prior_floor_eval = False
    return a


def _run_eval(batch_size, per_image_ap_out):
    args = _build_args()
    args.per_image_ap_out = per_image_ap_out
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = str(device)

    model = utils.get_model(args)
    ck = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    sd = ck['model']
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    model.to(device)
    model.eval()

    from ultralytics import YOLO
    yolo_model = YOLO(YOLO_WEIGHTS)
    yolo_model.model.eval()

    ds = datasets.build_dataset(is_train=False, args=args)
    small_ds = torch.utils.data.Subset(ds, list(range(N_IMAGES)))
    loader = torch.utils.data.DataLoader(
        small_ds, batch_size=batch_size, shuffle=False, num_workers=0,
        drop_last=False, collate_fn=datasets.yolo_collate_fn)

    base_reconstruction_criterion = utils.sel_criterion(args).to(device)
    fim_criterion = nn.BCEWithLogitsLoss().to(device)

    engine1.evaluate_semcom_with_yolo(
        semcom_net=model, yolo_model=yolo_model, dataloader=loader,
        device=device, reconstruction_criterion=base_reconstruction_criterion,
        fim_criterion=fim_criterion, lpips_criterion=None, args=args,
        current_epoch_num=f"test_bs{batch_size}",
        viz_output_dir="/tmp/test_digital_seed_batch_invariance_viz",
    )
    return np.load(per_image_ap_out)


class TestDigitalSeedBatchInvariance(unittest.TestCase):
    def test_batch_size_1_vs_4_identical_per_image_ap(self):
        out_bs1 = "/tmp/test_digital_seed_bs1_perimage.npy"
        out_bs4 = "/tmp/test_digital_seed_bs4_perimage.npy"
        ap_bs1 = _run_eval(1, out_bs1)
        ap_bs4 = _run_eval(4, out_bs4)
        self.assertEqual(len(ap_bs1), N_IMAGES)
        self.assertEqual(len(ap_bs4), N_IMAGES)
        np.testing.assert_array_equal(
            ap_bs1, ap_bs4,
            err_msg=("Per-image AP differs between batch_size=1 and batch_size=4 -- "
                      "the digital_seed batch-collision bug (RESULTS_OF_RECORD.md "
                      "§6d-3/§6d-5) has been reintroduced."))


if __name__ == '__main__':
    unittest.main()
