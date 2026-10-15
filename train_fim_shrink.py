#!/usr/bin/env python
"""train_fim_shrink.py -- 2026-09-23. Can `fim_module` be shrunk without
losing routing quality? (Follow-up to §6ae's finding that FIM is a FIXED
cost that dominates the smallest distilled student's latency.)

Loads a full checkpoint (default: the Tiny/192d2L polish checkpoint, where
FIM matters most for latency), REPLACES `fim_module` with a freshly
initialized, much smaller `FeatureImportanceTransformer`, freezes EVERYTHING
else, and trains ONLY the new FIM module against the REAL supervision signal
(box-overlap BCE target from `datasets.py`'s `_create_patch_importance_map`
-- the same target the original FIM was trained against), not a distillation
proxy. Encoder forward runs under `torch.no_grad()` (frozen, no gradient
needed through it) for a small train-time speedup.

Run: __TORCH_PY__ train_fim_shrink.py \
    --base_ckpt ckpt_polish_192d2L_from_tx_distill/checkpoint-best_psnr.pth \
    --encoder_embed_dim 192 --encoder_depth 2 \
    --new_fim_embed_dim 64 --new_fim_depth 1 --new_fim_num_heads 2 \
    --epochs 30 --output_dir ckpt_fim_shrink_192d2L_64d1L2h
"""
import argparse
import os
import sys
import time

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model as model_module  # noqa: E402  registers ViT_Reconstruction_Model
import utils  # noqa: E402
import datasets  # noqa: E402
from model_util import FeatureImportanceTransformer  # noqa: E402


def build_base_model(ckpt_path, encoder_embed_dim, encoder_depth, encoder_num_heads):
    args_ns = argparse.Namespace(
        model="ViT_Reconstruction_Model_Default", input_size=224, patch_size=8,
        encoder_embed_dim=encoder_embed_dim, encoder_depth=encoder_depth,
        encoder_num_heads=encoder_num_heads, decoder_embed_dim=512, decoder_depth=6,
        decoder_num_heads=8, quantizer_dim=512, bits_vq_high=10, bits_vq_low=2,
        quantizer_type="rvq", rvq_stages=3, rvq_bits_per_stage=6, rvq_ema_decay=.99,
        quantizer_commitment_cost=.25, drop_rate=0., drop_path_rate=.1,
        fim_embed_dim=256, fim_depth=2, fim_num_heads=4, fim_drop_rate=.1,
        fim_routing_threshold=.7, routing_topk=63, eps_stab=1e-6, disable_fim=False,
    )
    net = utils.get_model(args_ns)
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    net.load_state_dict({k.removeprefix("module."): v for k, v in sd.items()}, strict=True)
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_ckpt", required=True)
    ap.add_argument("--encoder_embed_dim", type=int, default=192)
    ap.add_argument("--encoder_depth", type=int, default=2)
    ap.add_argument("--encoder_num_heads", type=int, default=8)
    ap.add_argument("--new_fim_embed_dim", type=int, default=64)
    ap.add_argument("--new_fim_depth", type=int, default=1)
    ap.add_argument("--new_fim_num_heads", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--output_dir", required=True)
    cli = ap.parse_args()

    device = "cuda"
    os.makedirs(cli.output_dir, exist_ok=True)

    net = build_base_model(cli.base_ckpt, cli.encoder_embed_dim, cli.encoder_depth,
                           cli.encoder_num_heads).to(device)
    old_fim_params = sum(p.numel() for p in net.fim_module.parameters())

    new_fim = FeatureImportanceTransformer(
        input_dim=cli.encoder_embed_dim, fim_embed_dim=cli.new_fim_embed_dim,
        fim_depth=cli.new_fim_depth, fim_num_heads=cli.new_fim_num_heads,
    ).to(device)
    net.fim_module = new_fim
    new_fim_params = sum(p.numel() for p in new_fim.parameters())
    print(f"[fim-shrink] replaced fim_module: {old_fim_params/1e6:.4f}M -> {new_fim_params/1e6:.4f}M "
          f"params ({new_fim_params/old_fim_params*100:.1f}% of original)")

    for name, p in net.named_parameters():
        p.requires_grad = name.startswith("fim_module.")
    n_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"[fim-shrink] trainable params: {n_trainable/1e6:.4f}M (should equal new fim_module's own count)")

    net.img_encoder.eval()  # keep dropout/BN (none here, but harmless) off for the frozen encoder

    train_args = argparse.Namespace(
        data_set="fish", data_path="__AQUASC_DATA__/yolo_fish_dataset_root",
        input_size=224, patch_size=8, mask_ratio=0., no_augment=False, seed=42,
        num_object_classes=1, eval=False,
    )
    ds = datasets.build_dataset(is_train=True, args=train_args)
    loader = torch.utils.data.DataLoader(ds, batch_size=cli.batch_size, shuffle=True,
                                         num_workers=4, collate_fn=datasets.yolo_collate_fn,
                                         drop_last=True)
    print(f"[fim-shrink] train set: {len(ds)} images, {len(loader)} batches/epoch")

    fim_criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.AdamW(new_fim.parameters(), lr=cli.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cli.epochs)

    best_loss = float("inf")
    for epoch in range(cli.epochs):
        net.fim_module.train()
        t0 = time.time()
        loss_sum, n_batches = 0.0, 0
        for (imgs, masks), (_recon_tgt, _gt, fim_target) in loader:
            imgs, masks, fim_target = imgs.to(device), masks.to(device), fim_target.to(device)
            with torch.no_grad():
                x = net.img_encoder(imgs, masks)
            logits = net.fim_module(x)
            loss = fim_criterion(logits, fim_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            n_batches += 1
        scheduler.step()
        avg_loss = loss_sum / n_batches
        elapsed = time.time() - t0
        print(f"[fim-shrink] epoch {epoch} loss={avg_loss:.4f} elapsed={elapsed:.1f}s lr={scheduler.get_last_lr()[0]:.2e}",
              flush=True)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(net.state_dict(), os.path.join(cli.output_dir, "checkpoint-best.pth"))
        torch.save(net.state_dict(), os.path.join(cli.output_dir, f"checkpoint-{epoch}.pth"))

    print(f"[fim-shrink] DONE. best_loss={best_loss:.4f}")


if __name__ == "__main__":
    main()
