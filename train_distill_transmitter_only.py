#!/usr/bin/env python
"""train_distill_transmitter_only.py -- 2026-09-18 pivot.

The composite approach (run_class_main.py --distill_teacher_ckpt ...) runs the
FULL model every step: encoder -> FIM -> projection -> RVQ -> channel
simulation -> decoder -> reconstruction/BBW/FIM/VQ task loss, with a KD loss
(feat/cos/quant MSE + FIM-KD) added on top. That is slow (decoder + channel
every step) and, worse, the feat/quant MSE terms are only an INDIRECT proxy
for "same codeword": idx_agree_stage{1,2,3} stalled at 3-24% for ~190 epochs
even as feat/quant fell steadily -- small feature distance does not imply
the same nearest-codeword argmin near a Voronoi boundary.

This script distills ONLY the transmitter side -- exactly the packet that
would be sent, nothing else:
    x -> encoder -> FIM (routing scores) -> proj -> norm -> RVQ codeword index
No channel simulation, no decoder, no reconstruction loss, no YOLO. The
decoder/channel_to_decoder_proj/quantizer_low are built (so the underlying
model class doesn't need forking) but never called and never trained.

Loss:
    L = w_idx * L_idx + w_fim * L_fim_kd
    L_idx    = mean over RVQ stages of CrossEntropy(student's own
               stage_logits, teacher's own argmin index at that stage) --
               the actual discrete objective ("does the student's own
               quantization path land on the teacher's codeword"), not a
               feature-space proxy for it. See ResidualQuantizer.stage_logits.
    L_fim_kd = MSE(student FIM raw logits, teacher FIM raw logits) -- keeps
               routing (which tokens are transmitted at all, i.e. also part
               of "the packet") aligned as the encoder underneath changes.

The student's quantizer_high is NOT a separate copy -- it IS the teacher's
(literally the same object, frozen), so "same codebook" holds by
construction rather than by a warm-start that could later drift. Only the
student's img_encoder, fim_module, encoder_to_channel_proj, and
norm_before_quantizer are trainable; decoder/channel_to_decoder_proj/
quantizer_low exist on the model object (build_model reuses the full class)
but are frozen and never receive a forward call.

Run: __TORCH_PY__ train_distill_transmitter_only.py \
    --teacher_ckpt ckpt_semcom_reconstruction_yolo_fish_MIMO_LPIPS_FINAL/checkpoint-best_psnr.pth \
    --student_encoder_depth 2 --output_dir ckpt_distill_tx_only_512d2L
"""
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model as model_module  # noqa: E402  registers the model / gives ViT_Reconstruction_Model


IMG = 224
TRAIN_IMG_DIR = "__AQUASC_DATA__/yolo_fish_dataset_root/train/images"
VAL_IMG_DIR = "__AQUASC_DATA__/yolo_fish_dataset_root/valid/images"


class ImageOnlyDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.names = sorted(f for f in os.listdir(img_dir)
                            if f.lower().endswith((".png", ".jpg", ".jpeg")))
        self.t = transforms.Compose([transforms.Resize((IMG, IMG)), transforms.ToTensor()])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        return self.t(Image.open(os.path.join(self.img_dir, self.names[idx])).convert("RGB"))


def build_full_model(encoder_embed_dim, encoder_depth, encoder_num_heads):
    return model_module.ViT_Reconstruction_Model(
        img_size=IMG, patch_size=8, num_object_classes=1,
        encoder_embed_dim=encoder_embed_dim, encoder_depth=encoder_depth,
        encoder_num_heads=encoder_num_heads,
        decoder_embed_dim=512, decoder_depth=6, decoder_num_heads=8,
        fim_embed_dim=256, fim_depth=2, fim_num_heads=4,
        quantizer_dim=512, quantizer_type="rvq", rvq_stages=3, rvq_bits_per_stage=6,
        bits_vq_low=2, rvq_ema_decay=0.0, routing_topk=63,
    )


def transmitter_forward(m, x):
    """Encoder -> FIM -> proj -> norm. No channel, no decode."""
    B = x.shape[0]
    Np = m.img_encoder.patch_embed.num_patches
    mask = torch.zeros(B, Np, dtype=torch.bool, device=x.device)
    tokens = m.img_encoder(x, mask)
    fim_logits = m.fim_module(tokens)
    proj = m.norm_before_quantizer(m.encoder_to_channel_proj(tokens))
    return proj, fim_logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_ckpt", required=True)
    ap.add_argument("--teacher_encoder_embed_dim", type=int, default=512)
    ap.add_argument("--teacher_encoder_depth", type=int, default=8)
    ap.add_argument("--teacher_encoder_num_heads", type=int, default=8)
    ap.add_argument("--student_encoder_embed_dim", type=int, default=512)
    ap.add_argument("--student_encoder_depth", type=int, default=2)
    ap.add_argument("--student_encoder_num_heads", type=int, default=8)
    ap.add_argument("--student_init_ckpt", default=None,
                    help="Optional: warm-start the student's img_encoder blocks "
                         "that match the teacher's shape/name (e.g. blocks 0..N-1) "
                         "from a checkpoint (usually the teacher's own). Highly "
                         "recommended -- makes 'depth-K truncated teacher' the "
                         "actual starting point instead of random init.")
    ap.add_argument("--data_path", default=TRAIN_IMG_DIR)
    ap.add_argument("--val_path", default=VAL_IMG_DIR)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_idx", type=float, default=1.0)
    ap.add_argument("--weight_idx_low", type=float, default=1.0,
                    help="Weight on the LOW-path index-CE loss (quantizer_low, "
                         "the majority of tokens -- ~721/784 under routing_topk=63). "
                         "2026-09-18: previously L_idx only covered the high path "
                         "applied uniformly to all tokens, which mismatches what "
                         "actually gets transmitted (~92% of tokens route low, and "
                         "were never supervised for index agreement at all).")
    ap.add_argument("--routing_topk", type=int, default=63,
                    help="Must match the teacher's own routing_topk -- determines "
                         "which of the STUDENT's own tokens (by its own FIM scores) "
                         "are treated as high-routed for the loss split. Using the "
                         "student's own routing (not the teacher's) matches what the "
                         "student will actually transmit at inference.")
    ap.add_argument("--weight_fim", type=float, default=0.25)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--save_freq", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[tx-distill] building teacher ({args.teacher_encoder_embed_dim}/"
          f"{args.teacher_encoder_depth}/{args.teacher_encoder_num_heads})")
    teacher = build_full_model(args.teacher_encoder_embed_dim, args.teacher_encoder_depth,
                               args.teacher_encoder_num_heads)
    t_sd = torch.load(args.teacher_ckpt, map_location="cpu")
    for key in ("model", "state_dict"):
        if isinstance(t_sd, dict) and key in t_sd:
            t_sd = t_sd[key]
            break
    missing, unexpected = teacher.load_state_dict(t_sd, strict=True)
    assert not missing and not unexpected, f"teacher load not clean: missing={missing} unexpected={unexpected}"
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"[tx-distill] teacher loaded strict=True from {args.teacher_ckpt}, frozen")

    print(f"[tx-distill] building student ({args.student_encoder_embed_dim}/"
          f"{args.student_encoder_depth}/{args.student_encoder_num_heads})")
    student = build_full_model(args.student_encoder_embed_dim, args.student_encoder_depth,
                               args.student_encoder_num_heads)
    student.to(device)

    # Student's quantizer_high AND quantizer_low ARE the teacher's -- same
    # objects, frozen, no separate copy that could ever drift. "Same
    # codebook" by construction, for both paths.
    student.quantizer_high = teacher.quantizer_high
    student.quantizer_low = teacher.quantizer_low
    for p in student.quantizer_high.parameters():
        p.requires_grad = False
    for p in student.quantizer_low.parameters():
        p.requires_grad = False
    # decoder / channel_to_decoder_proj are never called in this script's
    # forward path; freeze them too so no optimizer group is accidentally
    # built for dead weights.
    for mod in (student.img_decoder, student.channel_to_decoder_proj):
        for p in mod.parameters():
            p.requires_grad = False

    if args.student_init_ckpt:
        s_sd_source = torch.load(args.student_init_ckpt, map_location="cpu")
        for key in ("model", "state_dict"):
            if isinstance(s_sd_source, dict) and key in s_sd_source:
                s_sd_source = s_sd_source[key]
                break
        student_sd = student.state_dict()
        copied, skipped_shape = [], []
        for k, v in s_sd_source.items():
            if k in student_sd and student_sd[k].shape == v.shape:
                student_sd[k] = v.clone()
                copied.append(k)
            elif k in student_sd:
                skipped_shape.append(k)
        student.load_state_dict(student_sd, strict=True)
        # 2026-09-18: report the ACTUAL matched key prefixes rather than a
        # hardcoded assumption -- when student_encoder_embed_dim != teacher's,
        # img_encoder/encoder_to_channel_proj/fim_module's first layer do NOT
        # match (confirmed needed after a width-changed run: the old hardcoded
        # message claimed "img_encoder.blocks.0..N-1" matched even when they
        # didn't, since width differed).
        copied_prefixes = sorted(set(k.split('.')[0] for k in copied))
        print(f"[tx-distill] warm-started {len(copied)} tensors from {args.student_init_ckpt} "
              f"(top-level modules with >=1 matched tensor: {copied_prefixes}); "
              f"{len(skipped_shape)} kept from student's own random init (shape mismatch).")

    trainable_params = [p for n, p in student.named_parameters()
                        if p.requires_grad and (n.startswith("img_encoder")
                                                or n.startswith("fim_module")
                                                or n.startswith("encoder_to_channel_proj")
                                                or n.startswith("norm_before_quantizer"))]
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"[tx-distill] {len(trainable_params)} trainable tensors, {n_trainable/1e6:.2f}M params "
          f"(img_encoder + fim_module + encoder_to_channel_proj + norm_before_quantizer only)")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.05)

    train_ds = ImageOnlyDataset(args.data_path)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    print(f"[tx-distill] train dataset size = {len(train_ds)}")

    num_stages = teacher.quantizer_high.num_stages
    bits = teacher.quantizer_high.bits_per_stage
    k = args.routing_topk

    def route_mask(fim_logits, Np):
        """Same top-k routing model.py's forward() uses, replicated here so
        we can split the loss by the STUDENT's OWN routing decision (what it
        would actually transmit), not a global "apply high-path loss to
        every token" convention."""
        scores = torch.sigmoid(fim_logits.squeeze(-1))
        _, topk_idx = scores.topk(k, dim=-1)
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.scatter_(1, topk_idx, True)
        return mask  # [B, Np]

    since = time.time()
    for epoch in range(args.epochs):
        student.train()
        student.quantizer_high.eval()  # shared with teacher; never re-enter train-mode side effects
        student.quantizer_low.eval()
        ep_loss = ep_idx_hi_loss = ep_idx_lo_loss = ep_fim_loss = 0.0
        ep_agree = [0.0] * num_stages
        ep_agree_lo = 0.0
        ep_overlap = 0.0
        n_batches = 0
        for images in train_loader:
            images = images.to(device, non_blocking=True)
            B, Np = images.shape[0], teacher.img_encoder.patch_embed.num_patches
            optimizer.zero_grad()

            with torch.no_grad():
                t_proj, t_fim_logits = transmitter_forward(teacher, images)
                t_proj_flat = t_proj.reshape(-1, t_proj.size(-1))
                _, _, _, t_idx_packed = teacher.quantizer_high(t_proj_flat, return_per_token_loss=True)
                t_stage_idx = teacher.quantizer_high.unpack_indices(t_idx_packed, num_stages, bits)
                t_low_idx = teacher.quantizer_low.get_indices(t_proj_flat)

            s_proj, s_fim_logits = transmitter_forward(student, images)
            s_proj_flat = s_proj.reshape(-1, s_proj.size(-1))

            # Student's OWN routing decides what's actually transmitted where --
            # this, not a global convention, is what the loss should respect.
            s_high_mask_flat = route_mask(s_fim_logits, Np).reshape(-1)
            s_low_mask_flat = ~s_high_mask_flat

            # High path (student-routed-high tokens only): per-stage CE against
            # the teacher's own quantizer_high argmin at those SAME positions.
            s_hi_logits = student.quantizer_high.stage_logits(s_proj_flat[s_high_mask_flat])
            l_idx_hi = sum(F.cross_entropy(s_hi_logits[s], t_stage_idx[s][s_high_mask_flat])
                          for s in range(num_stages)) / num_stages

            # Low path (student-routed-low tokens -- the majority): CE against
            # the teacher's own quantizer_low argmin at those SAME positions.
            s_lo_logits = student.quantizer_low.logits(s_proj_flat[s_low_mask_flat])
            l_idx_lo = F.cross_entropy(s_lo_logits, t_low_idx[s_low_mask_flat])

            l_fim = F.mse_loss(s_fim_logits, t_fim_logits)
            loss = (args.weight_idx * l_idx_hi + args.weight_idx_low * l_idx_lo
                   + args.weight_fim * l_fim)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                s_hi_pred = [logit.argmax(dim=-1) for logit in s_hi_logits]
                for s in range(num_stages):
                    ep_agree[s] += (s_hi_pred[s] == t_stage_idx[s][s_high_mask_flat]).float().mean().item()
                s_lo_pred = s_lo_logits.argmax(dim=-1)
                ep_agree_lo += (s_lo_pred == t_low_idx[s_low_mask_flat]).float().mean().item()
                s_top = torch.sigmoid(s_fim_logits.squeeze(-1)).topk(k, dim=-1).indices
                t_top = torch.sigmoid(t_fim_logits.squeeze(-1)).topk(k, dim=-1).indices
                overlap = sum(len(set(s_top[b].tolist()) & set(t_top[b].tolist()))
                             for b in range(s_top.shape[0])) / (s_top.shape[0] * k)
                ep_overlap += overlap

            ep_loss += loss.item()
            ep_idx_hi_loss += l_idx_hi.item()
            ep_idx_lo_loss += l_idx_lo.item()
            ep_fim_loss += l_fim.item()
            n_batches += 1

        agree_str = " ".join(f"idx_agree_hi_s{s+1}={ep_agree[s]/n_batches:.4f}" for s in range(num_stages))
        print(f"[tx-distill] epoch {epoch} loss={ep_loss/n_batches:.4f} "
              f"idx_ce_hi={ep_idx_hi_loss/n_batches:.4f} idx_ce_lo={ep_idx_lo_loss/n_batches:.4f} "
              f"fim_kd={ep_fim_loss/n_batches:.4f} {agree_str} "
              f"idx_agree_lo={ep_agree_lo/n_batches:.4f} fim_topk_overlap={ep_overlap/n_batches:.4f} "
              f"elapsed={(time.time()-since)/60:.1f}m", flush=True)

        if (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint-{epoch}.pth")
            torch.save({"model": student.state_dict(), "epoch": epoch}, ckpt_path)
            print(f"[tx-distill] saved {ckpt_path}", flush=True)

    print(f"[tx-distill] DONE total={(time.time()-since)/3600:.2f}h")


if __name__ == "__main__":
    main()
