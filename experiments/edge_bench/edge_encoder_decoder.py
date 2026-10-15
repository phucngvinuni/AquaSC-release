"""Standalone AquaSC encoder/decoder wrappers for edge (Raspberry Pi) benchmarking.

Deliberately does NOT import model.py or utils.py: those pull in timm
(model.py's @register_model) and pytorch_msssim (utils.py), neither installed
on the Pi. This file imports only model_util.py (pure torch/numpy) plus
torch/numpy themselves, so it runs on a bare torch+numpy install (the Pi's
existing environment) with model_util.py copied alongside it.

Reproduces exactly the encode/decode computation in model.py's
ViT_Reconstruction_Model.forward() for the flagship's quantizer_type='rvq'
path (see model.py lines ~313-365 for routing/projection, ~460-482 for
quantization, decode_frame() in experiments/matched5/evaluate.py for the
decode side) -- verified against the real model by extract_edge_weights.py's
parity check before any checkpoint is shipped to the Pi.

Flagship hyperparameters (architecture is fixed at construction, matching
eval_flagship_coded_pooled.sh): patch_size=8, encoder_embed_dim=512,
encoder_depth=8, encoder_num_heads=8, fim_embed_dim=256, fim_depth=2,
fim_num_heads=4, quantizer_dim=512, rvq_stages=3, rvq_bits_per_stage=6,
bits_vq_low=2, routing_topk=63, decoder_embed_dim=512, decoder_depth=6,
decoder_num_heads=8.
"""
from __future__ import annotations
from functools import partial
import torch
import torch.nn as nn

from model_util import (
    ViTEncoder_Van, ViTDecoder_ImageReconstruction,
    HierarchicalQuantizer, ResidualQuantizer, FeatureImportanceTransformer,
)

FLAGSHIP_ARCH = dict(
    img_size=224, patch_size=8, encoder_in_chans=3,
    encoder_embed_dim=512, encoder_depth=8, encoder_num_heads=8,
    decoder_embed_dim=512, decoder_depth=6, decoder_num_heads=8,
    quantizer_dim=512, rvq_stages=3, rvq_bits_per_stage=6, bits_vq_low=2,
    fim_embed_dim=256, fim_depth=2, fim_num_heads=4, fim_drop_rate=0.0,
    routing_topk=63,
)


class AquaSCEdgeEncoder(nn.Module):
    """img -> (packed_high_indices, low_indices, route_to_high_mask).

    This is the transmitter-side computation: everything an edge/IoT camera
    node must run before handing indices off to the channel-coding stage
    (LDPC/QAM packing itself is negligible pure-numpy work, not benchmarked
    here -- see RESULTS_OF_RECORD.md for why).
    """

    def __init__(self, arch=FLAGSHIP_ARCH):
        super().__init__()
        norm_layer = nn.LayerNorm
        self.routing_topk = arch["routing_topk"]
        self.img_encoder = ViTEncoder_Van(
            img_size=arch["img_size"], patch_size=arch["patch_size"],
            in_chans=arch["encoder_in_chans"], embed_dim=arch["encoder_embed_dim"],
            depth=arch["encoder_depth"], num_heads=arch["encoder_num_heads"],
            norm_layer=norm_layer,
        )
        self.fim_module = FeatureImportanceTransformer(
            input_dim=arch["encoder_embed_dim"], fim_embed_dim=arch["fim_embed_dim"],
            fim_depth=arch["fim_depth"], fim_num_heads=arch["fim_num_heads"],
            drop_rate=arch["fim_drop_rate"], norm_layer=partial(norm_layer, eps=1e-6),
        )
        self.encoder_to_channel_proj = nn.Linear(arch["encoder_embed_dim"], arch["quantizer_dim"])
        self.norm_before_quantizer = nn.LayerNorm(arch["quantizer_dim"])
        self.quantizer_high = ResidualQuantizer(
            num_stages=arch["rvq_stages"], bits_per_stage=arch["rvq_bits_per_stage"],
            embedding_dim=arch["quantizer_dim"],
        )
        self.quantizer_low = HierarchicalQuantizer(
            num_embeddings=2 ** arch["bits_vq_low"], embedding_dim=arch["quantizer_dim"],
        )

    def forward(self, img: torch.Tensor):
        B, Np = img.shape[0], self.img_encoder.patch_embed.num_patches
        no_mask = torch.zeros(B, Np, dtype=torch.bool, device=img.device)
        x = self.img_encoder(img, no_mask)                       # [B,Np,512]
        fim_scores = torch.sigmoid(self.fim_module(x)).squeeze(-1)  # [B,Np]
        _, topk_idx = fim_scores.topk(self.routing_topk, dim=-1)
        # one_hot+sum instead of scatter_(dim, index, value=True): the
        # in-place bool "value" scatter overload has no ONNX/trace op
        # coverage on this torch version -- one_hot is exportable and gives
        # the identical boolean mask.
        route = torch.nn.functional.one_hot(topk_idx, num_classes=Np).sum(dim=1).bool()

        x_proj = self.norm_before_quantizer(self.encoder_to_channel_proj(x))  # [B,Np,512]
        flat = x_proj.reshape(B * Np, -1)
        route_flat = route.reshape(-1)

        high_in = flat[route_flat]
        low_in = flat[~route_flat]
        _, _, _, packed_high = self.quantizer_high(high_in, return_per_token_loss=False)
        _, _, _, idx_low = self.quantizer_low(low_in, return_per_token_loss=False)
        return packed_high, idx_low, route


class AquaSCEdgeDecoder(nn.Module):
    """(packed_high_indices, low_indices, route_to_high_mask) -> reconstructed image.

    Runs at the base station in the real deployment, not the edge -- benchmarked
    only for reference (arm C), to show what it would cost if a stronger edge
    node also wanted a local preview reconstruction.
    """

    def __init__(self, arch=FLAGSHIP_ARCH):
        super().__init__()
        norm_layer = nn.LayerNorm
        self.quantizer_high = ResidualQuantizer(
            num_stages=arch["rvq_stages"], bits_per_stage=arch["rvq_bits_per_stage"],
            embedding_dim=arch["quantizer_dim"],
        )
        self.quantizer_low = HierarchicalQuantizer(
            num_embeddings=2 ** arch["bits_vq_low"], embedding_dim=arch["quantizer_dim"],
        )
        self.channel_to_decoder_proj = nn.Linear(arch["quantizer_dim"], arch["decoder_embed_dim"])
        num_patches = (arch["img_size"] // arch["patch_size"]) ** 2
        self.img_decoder = ViTDecoder_ImageReconstruction(
            patch_size=arch["patch_size"], num_total_patches=num_patches,
            embed_dim=arch["decoder_embed_dim"], depth=arch["decoder_depth"],
            num_heads=arch["decoder_num_heads"], norm_layer=norm_layer,
        )
        self._grid = int(num_patches ** 0.5)

    def forward(self, packed_high: torch.Tensor, idx_low: torch.Tensor, route: torch.Tensor):
        B, Np = route.shape
        tokens = torch.zeros(B, Np, self.channel_to_decoder_proj.in_features, device=route.device)
        route_flat = route.reshape(-1)
        tokens_flat = tokens.reshape(B * Np, -1)
        tokens_flat[route_flat] = self.quantizer_high.decode(packed_high)
        tokens_flat[~route_flat] = self.quantizer_low.embedding(idx_low)
        tokens = tokens_flat.reshape(B, Np, -1)
        return self.img_decoder(
            x_vis_tokens=self.channel_to_decoder_proj(tokens),
            encoder_mask_boolean=torch.zeros(B, Np, dtype=torch.bool, device=route.device),
            full_image_num_patches_h=self._grid, full_image_num_patches_w=self._grid,
        )


ENCODER_KEYS = ("img_encoder.", "fim_module.", "encoder_to_channel_proj.",
                "norm_before_quantizer.", "quantizer_high.", "quantizer_low.")
DECODER_KEYS = ("quantizer_high.", "quantizer_low.", "channel_to_decoder_proj.", "img_decoder.")


def split_state_dict(full_sd: dict):
    """Split a full ViT_Reconstruction_Model state_dict into encoder-only and
    decoder-only subsets matching AquaSCEdgeEncoder / AquaSCEdgeDecoder."""
    enc = {k: v for k, v in full_sd.items() if k.startswith(ENCODER_KEYS)}
    dec = {k: v for k, v in full_sd.items() if k.startswith(DECODER_KEYS)}
    return enc, dec
