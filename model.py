# model.py
import math
from typing import Dict, Optional
import torch
import torch.nn as nn
import numpy as np
from functools import partial
from timm.models.registry import register_model
from digital_channel import transmit_and_receive_indices_batch as transmit_digital_indices # Dòng mới

from model_util import (
    ViTEncoder_Van, ViTDecoder_ImageReconstruction,
    HierarchicalQuantizer, ResidualQuantizer, Channels, FeatureImportanceTransformer, _cfg
)

from channel import power_norm_batchwise

from digital_channel import transmit_and_receive_indices_batch

import digital_client

class ViT_Reconstruction_Model(nn.Module):
    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 encoder_in_chans: int = 3,
                 encoder_embed_dim: int = 768,
                 encoder_depth: int = 12,
                 encoder_num_heads: int = 12,
                 decoder_embed_dim: int = 256,
                 decoder_depth: int = 4,
                 decoder_num_heads: int = 4,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 norm_layer=nn.LayerNorm,
                 init_values: float = 0.0,
                 use_learnable_pos_emb: bool = False,
                 quantizer_dim: int = 512,
                 bits_vq_high: int = 12,
                 bits_vq_low: int = 8,
                 quantizer_commitment_cost: float = 0.25,
                 quantizer_type: str = 'flat',
                 rvq_stages: int = 3,
                 rvq_bits_per_stage: int = 6,
                 rvq_ema_decay: float = 0.0,
                 analog_symbols_per_token: int = 9,
                 analog_symbols_per_token_high: int = 0,
                 analog_symbols_per_token_low: int = 0,
                 fim_embed_dim: int = 128,
                 fim_depth: int = 2,
                 fim_num_heads: int = 4,
                 fim_drop_rate: float = 0.1,
                 fim_routing_threshold: float = 0.6,
                 routing_topk: int = 63,
                 eps_stab: float = 1e-6,
                 disable_fim: bool = False,
                 **kwargs):
        super().__init__()

        self.fim_routing_threshold = fim_routing_threshold
        self.routing_topk = routing_topk
        self.eps_stab = eps_stab
        self.disable_fim = disable_fim
        self._routing_rng = None
        if disable_fim:
            print(f"  [ABLATION] disable_fim=True: RANDOM top-K routing (K={routing_topk}, seeded 42); "
                  f"FIM BCE loss weight -> 0; VQ loss UNWEIGHTED (no per-token FIM weighting).")
        else:
            print(f"  [ABLATION] disable_fim=False: FIM top-K routing (K={routing_topk}).")
        effective_patch_size = patch_size

        self.img_encoder = ViTEncoder_Van(
            img_size=img_size, patch_size=effective_patch_size, in_chans=encoder_in_chans,
            embed_dim=encoder_embed_dim, depth=encoder_depth, num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate,
            drop_path_rate=drop_path_rate, norm_layer=norm_layer, init_values=init_values,
            use_learnable_pos_emb=use_learnable_pos_emb
        )

        self.full_image_num_patches_h = self.img_encoder.patch_embed.patch_shape[0]
        self.full_image_num_patches_w = self.img_encoder.patch_embed.patch_shape[1]
        num_total_patches_in_image = self.img_encoder.patch_embed.num_patches

        self.img_decoder = ViTDecoder_ImageReconstruction(
            patch_size=effective_patch_size, num_total_patches=num_total_patches_in_image,
            embed_dim=decoder_embed_dim, depth=decoder_depth, num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer, init_values=init_values
        )

        self.encoder_to_channel_proj = nn.Linear(encoder_embed_dim, quantizer_dim)
        self.norm_before_quantizer = nn.LayerNorm(quantizer_dim)
        
        if quantizer_type not in ('flat', 'rvq', 'analog'):
            raise ValueError(f"quantizer_type must be 'flat', 'rvq', or 'analog', got {quantizer_type!r}")
        self.quantizer_type = quantizer_type
        self.analog_symbols_per_token = analog_symbols_per_token
        self.analog_symbols_per_token_high = analog_symbols_per_token_high
        self.analog_symbols_per_token_low = analog_symbols_per_token_low
        if quantizer_type == 'analog':
            if bool(analog_symbols_per_token_high) != bool(analog_symbols_per_token_low):
                raise ValueError(
                    "analog_symbols_per_token_high and _low must be set together "
                    "(both >0 for dual-rate) or both left at 0 (uniform-rate); "
                    f"got high={analog_symbols_per_token_high}, low={analog_symbols_per_token_low}")
            self.analog_dual_rate = bool(analog_symbols_per_token_high)
            if self.analog_dual_rate:
                # Wave 19 Part 2 CORRECTION (2026-09-14): the uniform-width branch
                # below applies the SAME projection to every token regardless of
                # FIM routing, so routing does nothing for that variant -- not a
                # controlled comparison (Part 2 exists to isolate transport only,
                # and silently dropping the routing mechanism is a second, hidden
                # change). Dual-rate restores routing as a real degree of freedom:
                # the K=routing_topk high-routed tokens get more symbols than the
                # (784-K) low-routed tokens, mirroring the digital system's own
                # 18-bit-high vs 2-bit-low split. See RESULTS_OF_RECORD.md 6q.
                self.analog_proj_high = nn.Linear(quantizer_dim, 2 * analog_symbols_per_token_high)
                self.analog_deproj_high = nn.Linear(2 * analog_symbols_per_token_high, quantizer_dim)
                self.analog_proj_low = nn.Linear(quantizer_dim, 2 * analog_symbols_per_token_low)
                self.analog_deproj_low = nn.Linear(2 * analog_symbols_per_token_low, quantizer_dim)
            else:
                # UNIFORM-RATE (original Part 2 build, kept for the already-recorded
                # S=9/0.1406-uses/px result -- see RESULTS_OF_RECORD.md 6q addendum,
                # labeled "not rate-matched, routing disabled" there, not retracted).
                # Every token (high- or low-routed alike) gets the SAME linear
                # projection to `2*analog_symbols_per_token` real values (paired
                # into complex symbols at the channel).
                self.analog_proj = nn.Linear(quantizer_dim, 2 * analog_symbols_per_token)
                self.analog_deproj = nn.Linear(2 * analog_symbols_per_token, quantizer_dim)
            # Deliberately NOT setting self.quantizer_high/quantizer_low at all
            # (not even to None): engine1.py/run_class_main.py guard quantizer-
            # specific code with `hasattr(model, 'quantizer_high')`, which would
            # incorrectly pass (then crash on None) if the attribute existed.
        elif quantizer_type == 'rvq':
            self.quantizer_high = ResidualQuantizer(
                num_stages=rvq_stages,
                bits_per_stage=rvq_bits_per_stage,
                embedding_dim=quantizer_dim,
                commitment_cost=quantizer_commitment_cost,
                ema_decay=rvq_ema_decay
            )
            print(f"  [H1] quantizer_type=rvq: high path uses ResidualQuantizer "
                  f"({rvq_stages} stages x {rvq_bits_per_stage} bits = effective "
                  f"{rvq_stages * rvq_bits_per_stage} bits); --bits_vq_high "
                  f"({bits_vq_high}) is IGNORED for the high path.")
            if rvq_ema_decay > 0:
                print(f"  [RVQFIX] rvq_ema_decay={rvq_ema_decay}: codebook updates via "
                      f"EMA (embeddings excluded from optimizer); dead-code restart "
                      f"available via restart_dead_codewords().")
        else:
            self.quantizer_high = HierarchicalQuantizer(
                num_embeddings=2**bits_vq_high,
                embedding_dim=quantizer_dim,
                commitment_cost=quantizer_commitment_cost
            )
        if quantizer_type != 'analog':
            self.quantizer_low = HierarchicalQuantizer(
                num_embeddings=2**bits_vq_low,
                embedding_dim=quantizer_dim,
                commitment_cost=quantizer_commitment_cost
            )

        self.channel_simulator = Channels() 
        self.channel_to_decoder_proj = nn.Linear(quantizer_dim, decoder_embed_dim)

        self.fim_module = FeatureImportanceTransformer(
            input_dim=encoder_embed_dim, fim_embed_dim=fim_embed_dim, fim_depth=fim_depth,
            fim_num_heads=fim_num_heads, drop_rate=fim_drop_rate,
            norm_layer=partial(norm_layer, eps=1e-6)
        )
        self.current_vq_loss = torch.tensor(0.0, dtype=torch.float32)
        self._bler_curve = None
        self._erasure_bler_curve = None
        self.index_flip_csv = None
        self.last_digital_stats = None

        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None: nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0); nn.init.constant_(module.weight, 1.0)

    def _flip_index_bits(self, indices: torch.Tensor, num_bits: int, p: float) -> torch.Tensor:
        """Flip each of ``num_bits`` bits of every index with independent prob p."""
        N = indices.shape[0]
        if N == 0 or p <= 0:
            return indices
        flips = torch.rand((N, num_bits), device=indices.device) < p
        out = torch.zeros(N, dtype=indices.dtype, device=indices.device)
        for i in range(num_bits):
            out = out | ((((indices >> i) & 1) ^ flips[:, i].long()) << i)
        return out

    @staticmethod
    def _n4_conceal_vectors(tokens: torch.Tensor, known: torch.Tensor,
                            grid_h: int, grid_w: int) -> torch.Tensor:
        """N4 (4-neighbour) interpolation of erased codebook vectors on the
        grid_h x grid_w token grid (28x28 for patch-8, 14x14 for patch-16).

        ``tokens``: [N, D] codeword vectors (erased rows already zeroed).
        ``known``: [N] bool (False = erased). Iterative mean-fill of erased rows
        from known 4-neighbours, then fallback to the mean of all known rows (or
        zero when all erased). Vectorized over positions (each pass is one
        diffusion step; #passes = max distance to a known token <= max(H,W)).
        """
        H, W = grid_h, grid_w
        grid = tokens.reshape(H, W, -1).clone()
        k = known.reshape(H, W)
        for _ in range(H * W + 2):
            if k.all():
                break
            g_known = torch.where(k.unsqueeze(-1), grid, torch.zeros_like(grid))
            c_known = k.float().unsqueeze(-1)
            nbr_sum = torch.zeros_like(grid)
            nbr_cnt = torch.zeros(H, W, 1, device=grid.device)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                gs = torch.roll(g_known, (dr, dc), dims=(0, 1))
                cs = torch.roll(c_known, (dr, dc), dims=(0, 1))
                if dr == -1: gs[-1] = 0; cs[-1] = 0
                if dr == 1: gs[0] = 0; cs[0] = 0
                if dc == -1: gs[:, -1] = 0; cs[:, -1] = 0
                if dc == 1: gs[:, 0] = 0; cs[:, 0] = 0
                nbr_sum += gs
                nbr_cnt += cs
            keep = k.unsqueeze(-1) | (nbr_cnt == 0)
            new_grid = torch.where(keep, grid, nbr_sum / nbr_cnt.clamp(min=1))
            new_k = k | (nbr_cnt.squeeze(-1) > 0)
            progressed = bool((new_k != k).any())
            grid, k = new_grid, new_k
            if not progressed:
                break
        if not k.all():
            if k.any():
                mean = grid[k].mean(dim=0)
            else:
                mean = torch.zeros(grid.shape[-1], device=grid.device, dtype=grid.dtype)
            grid[~k] = mean
        return grid.reshape(-1, grid.shape[-1])

    def forward(self,
            img: torch.Tensor,
            bm_pos: Optional[torch.Tensor] = None,
            _eval: bool = False,
            eval_snr_db: float = 30.0,
            train_snr_db_min: float = 10.0,
            train_snr_db_max: float = 25.0,
            digital_ldpc: bool = False,
            digital_csi_mode: str = 'perfect',
            digital_seed: int = 42,
            digital_bits_per_symbol: int = 4,
            digital_payload_rate: float = 0.5,
            digital_joint_frame: bool = False,
            index_flip: bool = False,
            index_flip_csv: Optional[str] = None,
            train_channel_mode: str = 'analog',
            erasure_bler_csv: Optional[str] = None,
            route_override: Optional[torch.Tensor] = None,
            prior_floor: bool = False,
            **kwargs
           ) -> Dict[str, torch.Tensor]:

        B = img.shape[0]
        is_training = self.training and not _eval
        if digital_joint_frame and (is_training or not digital_ldpc
                                    or self.quantizer_type != 'rvq' or prior_floor):
            raise ValueError("digital_joint_frame requires RVQ digital-LDPC evaluation without prior_floor")

        if prior_floor:
            # --- Track F / E3 mandatory companion (plan 16.9): the decoder's
            # learned-prior floor. No encoder, no FIM, no channel -- purely
            # random codebook indices at the real routing rate, decoded as
            # normal. `img` is used only for shape/device, never its content. ---
            device = img.device
            Np = self.img_encoder.patch_embed.num_patches
            route_to_high_mask = torch.zeros(B, Np, dtype=torch.bool, device=device)
            k = max(int(self.routing_topk), 0) or int(0.08 * Np)
            for b in range(B):
                perm = torch.randperm(Np, device=device)[:k]
                route_to_high_mask[b, perm] = True
            tokens = torch.zeros(B, Np, self.channel_to_decoder_proj.in_features, device=device)
            n_high = int(route_to_high_mask.sum().item())
            n_low = B * Np - n_high
            if n_high > 0:
                high_idx = torch.randint(0, self.quantizer_high.num_embeddings, (n_high,), device=device)
                high_vecs = (self.quantizer_high.decode(high_idx) if self.quantizer_type == 'rvq'
                             else self.quantizer_high.embedding(high_idx))
                tokens[route_to_high_mask] = high_vecs
            if n_low > 0:
                low_idx = torch.randint(0, self.quantizer_low.num_embeddings, (n_low,), device=device)
                tokens[~route_to_high_mask] = self.quantizer_low.embedding(low_idx)
            x_for_decoder_input = self.channel_to_decoder_proj(tokens)
            encoder_input_mask_bool = torch.zeros(B, Np, dtype=torch.bool, device=device)
            reconstructed_image = self.img_decoder(
                x_vis_tokens=x_for_decoder_input,
                encoder_mask_boolean=encoder_input_mask_bool,
                full_image_num_patches_h=self.full_image_num_patches_h,
                full_image_num_patches_w=self.full_image_num_patches_w,
            )
            return {
                'reconstructed_image': reconstructed_image,
                'vq_loss': torch.tensor(0.0, device=device),
            }
        
        if is_training:
            current_snr_db = (torch.rand(1).item() * (train_snr_db_max - train_snr_db_min)) + train_snr_db_min
        else:
            current_snr_db = eval_snr_db

        if bm_pos is None:
            encoder_input_mask_bool = torch.zeros(
                B, self.img_encoder.patch_embed.num_patches, dtype=torch.bool, device=img.device
            )
        else:
            encoder_input_mask_bool = bm_pos

        # 1. Encoder
        x_encoded_tokens = self.img_encoder(img, encoder_input_mask_bool)
        
        # 2. FIM -> Routing Mask
        fim_raw_logits = self.fim_module(x_encoded_tokens)
        fim_scores = torch.sigmoid(fim_raw_logits)
        if self.disable_fim:
            # [ABLATION] random top-K routing, content-independent and rate-identical
            # to the FIM path (K = routing_topk). Re-randomized per forward call from
            # a dedicated local generator seeded 42: training sees a fresh random
            # partition each step, and evaluation replays a deterministic sequence
            # (the generator state advances in the fixed batch/chunk order).
            if self._routing_rng is None or self._routing_rng.device != fim_scores.device:
                self._routing_rng = torch.Generator(device=fim_scores.device).manual_seed(42)
            rand_scores = torch.rand(B, fim_scores.shape[1], generator=self._routing_rng,
                                     device=fim_scores.device)
            _, topk_indices = rand_scores.topk(self.routing_topk, dim=-1)
            route_to_high_mask = torch.zeros_like(fim_scores.squeeze(-1), dtype=torch.bool)
            route_to_high_mask.scatter_(1, topk_indices, True)
        elif self.routing_topk > 0:
            fim_scores_flat = fim_scores.squeeze(-1)
            _, topk_indices = fim_scores_flat.topk(self.routing_topk, dim=-1)
            route_to_high_mask = torch.zeros_like(fim_scores_flat, dtype=torch.bool)
            route_to_high_mask.scatter_(1, topk_indices, True)
        else:
            route_to_high_mask = (fim_scores.squeeze(-1) > self.fim_routing_threshold) # [B, Np]

        # ROUTING-EVAL (wave 9a, R3 #5): eval-only external routing mask. The FIM
        # scores above are still computed (returned for logging / VQ-loss weighting),
        # but the routing decision uses the supplied mask instead of the FIM top-K.
        # The training branch is untouched: applying an override during training is
        # rejected explicitly.
        if route_override is not None:
            if is_training:
                raise ValueError(
                    "route_override is eval-only; refusing to apply it in the training branch")
            route_to_high_mask = route_override.to(device=route_to_high_mask.device,
                                                   dtype=torch.bool)
            expected_shape = fim_scores.squeeze(-1).shape
            if tuple(route_to_high_mask.shape) != tuple(expected_shape):
                raise ValueError(
                    f"route_override shape {tuple(route_to_high_mask.shape)} does not match "
                    f"expected [B, Np] {tuple(expected_shape)}")

        # 3. Project features
        x_proj_normalized = self.norm_before_quantizer(self.encoder_to_channel_proj(x_encoded_tokens))

        if self.quantizer_type == 'analog':
            # AquaSC-Analog (Wave 19 Part 2): standard DeepJSCC bottleneck --
            # no VQ/RVQ, no discrete indices, no digital LDPC/QAM chain. The
            # differentiable Rayleigh proxy (Channels.Rayleigh, the same one
            # used for the digital system's training-time analog proxy) IS
            # the real channel here: there is no quantization step to justify
            # a separate eval-time digital simulation, so train and eval share
            # this exact path, differing only in which current_snr_db is used
            # (selected above, before routing). Routing/FIM are computed
            # exactly as in every other quantizer_type (unchanged above) but
            # do not gate this projection's width -- see RESULTS_OF_RECORD.md
            # section 6q for the rate derivation and why.
            self.current_vq_loss = torch.tensor(0.0, device=img.device)
            noise_power_variance = 10 ** (-current_snr_db / 10.0)
            B, Np, D = x_proj_normalized.shape
            if self.analog_dual_rate:
                x_flat = x_proj_normalized.reshape(B * Np, D)
                high_mask_flat = route_to_high_mask.reshape(-1)
                low_mask_flat = ~high_mask_flat
                K_high = self.routing_topk
                K_low = Np - K_high
                per_row_high_count = high_mask_flat.view(B, Np).sum(dim=1)
                assert torch.equal(per_row_high_count, torch.full_like(per_row_high_count, K_high)), (
                    "AquaSC-Analog dual-rate requires exactly routing_topk high-routed "
                    "tokens per image (fixed-cardinality top-K); a route_override with "
                    "variable per-image cardinality is not supported by this branch.")
                high_D = x_flat[high_mask_flat].view(B, K_high, D)
                low_D = x_flat[low_mask_flat].view(B, K_low, D)

                sym_high = self.analog_proj_high(high_D)  # [B, K_high, 2*Sh]
                sym_low = self.analog_proj_low(low_D)      # [B, K_low, 2*Sl]
                signal_power_raw = torch.mean(torch.cat(
                    [sym_high.reshape(B, -1), sym_low.reshape(B, -1)], dim=1) ** 2).item()

                # Concatenate the WHOLE image's transmission (high + low streams)
                # into one vector before power_norm_batchwise, so the per-image
                # unit-average-power constraint covers the full transmitted symbol
                # set exactly as the uniform-rate branch does -- not normalized
                # per-stream, which would let a narrow low-rate stream "borrow"
                # a disproportionate power share. ONE Rayleigh() call on the
                # concatenated vector also gives high- and low-stream symbols
                # the SAME per-image fade draw (N-3: one complex gain per image,
                # not per stream -- RESULTS_OF_RECORD.md 6r).
                n_high_reals = sym_high.shape[1] * sym_high.shape[2]
                cat_flat = torch.cat([sym_high.reshape(B, -1), sym_low.reshape(B, -1)], dim=1)
                cat_norm = power_norm_batchwise(cat_flat, power=1)
                # Channels.Rayleigh draws H_complex with shape (B,1,1), which needs a
                # 3D input to broadcast correctly (the Np dimension plays that role in
                # the uniform-rate branch) -- a bare 2D [B, total_reals] input silently
                # broadcasts wrong (H's middle dim of 1 vs. a padded-in leading 1 from
                # the 2D tensor, not the batch dim), doubling the tensor. Caught by a
                # shape-mismatch crash on the very first smoke test, not by any check
                # of the noise math itself. Add/remove a singleton middle dim instead.
                cat_rx = self.channel_simulator.Rayleigh(cat_norm.unsqueeze(1), noise_power_variance).squeeze(1)
                sym_high_rx = cat_rx[:, :n_high_reals].view(B, K_high, 2 * self.analog_symbols_per_token_high)
                sym_low_rx = cat_rx[:, n_high_reals:].view(B, K_low, 2 * self.analog_symbols_per_token_low)

                dec_high = self.analog_deproj_high(sym_high_rx)
                dec_low = self.analog_deproj_low(sym_low_rx)
                x_for_decoder_flat = torch.zeros_like(x_flat)
                # Under autocast, the Linear layers above may return a different
                # dtype (e.g. Half) than x_flat (Float) -- index_put_ requires an
                # exact dtype match, unlike a plain assignment, so cast explicitly.
                x_for_decoder_flat[high_mask_flat] = dec_high.reshape(-1, D).to(x_for_decoder_flat.dtype)
                x_for_decoder_flat[low_mask_flat] = dec_low.reshape(-1, D).to(x_for_decoder_flat.dtype)
                x_for_decoder_input = self.channel_to_decoder_proj(x_for_decoder_flat.view(B, Np, D))
                print(f"  [ANALOG-DUAL] {'train' if is_training else 'eval'}_snr_db={current_snr_db:.2f} dB, "
                      f"raw_mean_signal_power={signal_power_raw:.6f}, "
                      f"Sh={self.analog_symbols_per_token_high}, Sl={self.analog_symbols_per_token_low}, "
                      f"K_high={K_high}, K_low={K_low}")
            else:
                analog_symbols = self.analog_proj(x_proj_normalized)  # [B, Np, 2*S]
                signal_power_raw = torch.mean(analog_symbols ** 2).item()
                analog_symbols_normalized = power_norm_batchwise(analog_symbols, power=1)
                symbols_after_channel = self.channel_simulator.Rayleigh(
                    analog_symbols_normalized, noise_power_variance)
                x_for_decoder_input = self.channel_to_decoder_proj(
                    self.analog_deproj(symbols_after_channel))
                print(f"  [ANALOG] {'train' if is_training else 'eval'}_snr_db={current_snr_db:.2f} dB, "
                      f"raw_mean_signal_power={signal_power_raw:.6f}, "
                      f"symbols_per_token={self.analog_symbols_per_token}")
            reconstructed_image = self.img_decoder(
                x_vis_tokens=x_for_decoder_input,
                encoder_mask_boolean=encoder_input_mask_bool,
                full_image_num_patches_h=self.full_image_num_patches_h,
                full_image_num_patches_w=self.full_image_num_patches_w,
            )
            return {
                'reconstructed_image': reconstructed_image,
                'vq_loss': self.current_vq_loss,
                'fim_importance_scores': fim_raw_logits,
            }

        bits_high = int(np.log2(self.quantizer_high.num_embeddings))
        bits_low = int(np.log2(self.quantizer_low.num_embeddings))

        
        if is_training:
            final_tokens_for_channel = torch.zeros_like(x_proj_normalized)
            total_vq_loss = torch.tensor(0.0, device=img.device)
            
        
            high_tokens_mask_flat = route_to_high_mask.flatten()
            if high_tokens_mask_flat.any():
                q_high, vq_loss_high, _, indices_high = self.quantizer_high(x_proj_normalized.view(-1, x_proj_normalized.size(-1))[high_tokens_mask_flat], return_per_token_loss=True)
                final_tokens_for_channel[route_to_high_mask] = q_high
                if self.disable_fim:
                    total_vq_loss += vq_loss_high.mean()
                else:
                    s_high = fim_scores.squeeze(-1)[route_to_high_mask]
                    total_vq_loss += (s_high * vq_loss_high).sum() / (s_high.sum() + self.eps_stab)

            low_tokens_mask = ~route_to_high_mask
            low_tokens_mask_flat = low_tokens_mask.flatten()
            if low_tokens_mask_flat.any():
                q_low, vq_loss_low, _, indices_low = self.quantizer_low(x_proj_normalized.view(-1, x_proj_normalized.size(-1))[low_tokens_mask_flat], return_per_token_loss=True)
                final_tokens_for_channel[low_tokens_mask] = q_low
                if self.disable_fim:
                    total_vq_loss += vq_loss_low.mean()
                else:
                    s_low = 1 - fim_scores.squeeze(-1)[low_tokens_mask]
                    total_vq_loss += (s_low * vq_loss_low).sum() / (s_low.sum() + self.eps_stab)
            
            self.current_vq_loss = total_vq_loss

            # --- Index-noise fine-tune (plan 5.4): flip index bits at the measured
            # digital error rate (BLER) while keeping the analog proxy for the
            # remaining channel noise. STE-preserving: forward value is the flipped
            # index's codeword, gradient flows to the encoder's STE output.
            # Only meaningful for the analog proxy -- train_channel_mode
            # digital_uncoded/erasure already simulate the real chain below. ---
            if index_flip and train_channel_mode == 'analog':
                if self._bler_curve is None:
                    from index_flip import BlerCurve
                    self._bler_curve = BlerCurve(index_flip_csv)
                p_flip = self._bler_curve(current_snr_db)
                if p_flip > 0:
                    flipped_tokens = torch.zeros_like(final_tokens_for_channel)
                    if high_tokens_mask_flat.any():
                        fh = self._flip_index_bits(indices_high, bits_high, p_flip)
                        fh_tokens = (self.quantizer_high.decode(fh) if self.quantizer_type == 'rvq'
                                     else self.quantizer_high.embedding(fh))
                        flipped_tokens[route_to_high_mask] = fh_tokens
                    if low_tokens_mask_flat.any():
                        fl = self._flip_index_bits(indices_low, bits_low, p_flip)
                        flipped_tokens[low_tokens_mask] = self.quantizer_low.embedding(fl)
                    final_tokens_for_channel = final_tokens_for_channel + (flipped_tokens - final_tokens_for_channel).detach()

            if train_channel_mode == 'digital_uncoded':
                # --- Track E1 (plan 16.9): train through the real uncoded digital
                # chain (index -> Gray 16-QAM -> Rayleigh -> demap) instead of the
                # analog proxy, so the uncoded-eval curve is no longer handicapped
                # by a train/test channel mismatch. STE-preserving, mirrors the
                # decode-from-recovered-indices logic of the uncoded eval branch
                # below (digital_ldpc=False case). ---
                Np = route_to_high_mask.shape[1]
                all_indices_flat = torch.zeros(B * Np, dtype=torch.long, device=img.device)
                bits_flat = torch.zeros(B * Np, dtype=torch.long, device=img.device)
                if high_tokens_mask_flat.any():
                    all_indices_flat[high_tokens_mask_flat] = indices_high
                    bits_flat[high_tokens_mask_flat] = bits_high
                if low_tokens_mask_flat.any():
                    all_indices_flat[low_tokens_mask_flat] = indices_low
                    bits_flat[low_tokens_mask_flat] = bits_low
                recovered_indices = transmit_digital_indices(
                    all_indices_flat.view(B, Np), bits_flat.view(B, Np),
                    snr_db=current_snr_db, seed=digital_seed)
                corrupted_tokens = torch.zeros_like(final_tokens_for_channel)
                for i in range(B):
                    route_high_i = route_to_high_mask[i]
                    rec_idx_i = recovered_indices[i]
                    if self.quantizer_type == 'rvq':
                        corrupted_tokens[i, route_high_i] = self.quantizer_high.decode(rec_idx_i[route_high_i])
                    else:
                        corrupted_tokens[i, route_high_i] = self.quantizer_high.embedding(rec_idx_i[route_high_i])
                    corrupted_tokens[i, ~route_high_i] = self.quantizer_low.embedding(rec_idx_i[~route_high_i])
                tokens_after_channel = final_tokens_for_channel + (corrupted_tokens - final_tokens_for_channel).detach()
                x_for_decoder_input = self.channel_to_decoder_proj(tokens_after_channel)
                print(f"  [E1] train_snr_db={current_snr_db:.2f} dB, chain=digital_uncoded")

            elif train_channel_mode == 'erasure':
                # --- Track E2 (plan 16.9): erasure-aware training for the coded
                # chain -- per-token Bernoulli erasure at the measured LDPC
                # BLER(SNR), concealed with the same N4 diffusion the coded eval
                # path applies to a real erasure. Surviving tokens are untouched
                # (LDPC + per-block CRC-32 means a surviving block is bit-exact),
                # so there is no continuous-noise term in this mode -- this is the
                # tractable train-through-the-channel form for a non-differentiable
                # code, and it is a per-token (not per-RVQ-stage) approximation;
                # per-stage UEP remains a separate, unimplemented lever (plan 16.9
                # Track B). Fully differentiable (plain torch ops), no STE needed. ---
                if self._erasure_bler_curve is None:
                    from index_flip import BlerCurve
                    self._erasure_bler_curve = BlerCurve(erasure_bler_csv)
                p_erase = self._erasure_bler_curve(current_snr_db)
                erased = torch.rand(B, route_to_high_mask.shape[1], device=img.device) < p_erase
                tokens_after_channel = torch.zeros_like(final_tokens_for_channel)
                for i in range(B):
                    tokens_after_channel[i] = self._n4_conceal_vectors(
                        final_tokens_for_channel[i], ~erased[i],
                        self.full_image_num_patches_h, self.full_image_num_patches_w)
                x_for_decoder_input = self.channel_to_decoder_proj(tokens_after_channel)
                print(f"  [E2] train_snr_db={current_snr_db:.2f} dB, p_erase={p_erase:.4f}")

            else:
                signal_power_raw = torch.mean(final_tokens_for_channel ** 2).item()
                final_tokens_for_channel = power_norm_batchwise(final_tokens_for_channel, power=1)
                signal_power_normalized = torch.mean(final_tokens_for_channel ** 2).item()
                print(f"  [V3] train_snr_db={current_snr_db:.2f} dB, raw_mean_signal_power={signal_power_raw:.6f}, normalized_mean_signal_power={signal_power_normalized:.6f}")
                noise_power_variance = 10**(-current_snr_db / 10.0)
                tokens_after_channel = self.channel_simulator.Rayleigh(final_tokens_for_channel, noise_power_variance)
                x_for_decoder_input = self.channel_to_decoder_proj(tokens_after_channel)

        else:
            if B > 1:

                print("Warning: Digital evaluation path might be slow with batch size > 1. Forcing batch size to 1 is recommended for evaluation.")

                
            all_indices = torch.zeros_like(route_to_high_mask, dtype=torch.long)
            bits_per_index = torch.zeros_like(route_to_high_mask, dtype=torch.long)
            total_vq_loss = torch.tensor(0.0, device=img.device)
            quantized_ste_output = torch.zeros_like(x_proj_normalized)

            if route_to_high_mask.any():
                high_tokens = x_proj_normalized[route_to_high_mask]
                q_high_ste, vq_loss_high, _, indices_high = self.quantizer_high(high_tokens, return_per_token_loss=True)
                
                quantized_ste_output[route_to_high_mask] = q_high_ste
                all_indices[route_to_high_mask] = indices_high
                bits_per_index[route_to_high_mask] = bits_high
   
                if self.disable_fim:
                    total_vq_loss += vq_loss_high.mean()
                else:
                    s_high = fim_scores.squeeze(-1)[route_to_high_mask]
                    total_vq_loss += (s_high * vq_loss_high).sum() / (s_high.sum() + self.eps_stab)

            low_tokens_mask = ~route_to_high_mask
            if low_tokens_mask.any():
                low_tokens = x_proj_normalized[low_tokens_mask]
                q_low_ste, vq_loss_low, _, indices_low = self.quantizer_low(low_tokens, return_per_token_loss=True)

                quantized_ste_output[low_tokens_mask] = q_low_ste
                all_indices[low_tokens_mask] = indices_low
                bits_per_index[low_tokens_mask] = bits_low

                if self.disable_fim:
                    total_vq_loss += vq_loss_low.mean()
                else:
                    s_low = 1 - fim_scores.squeeze(-1)[low_tokens_mask]
                    total_vq_loss += (s_low * vq_loss_low).sum() / (s_low.sum() + self.eps_stab)

            self.current_vq_loss = total_vq_loss 
            if digital_ldpc:
                if digital_joint_frame and self.quantizer_type != 'rvq':
                    raise ValueError("digital_joint_frame requires quantizer_type=rvq")
                if self.quantizer_type == 'rvq':
                    (recovered_indices, digital_stats, stage_indices_rx,
                     stage_erased_rx, low_indices_rx, low_erased_rx) = \
                        digital_client.transmit_indices_digital(
                            all_indices, route_to_high_mask, bits_high, bits_low,
                            current_snr_db, csi_mode=digital_csi_mode,
                            seed=digital_seed, rvq_stage_uep=True,
                            return_stage_status=True,
                            bits_per_symbol=digital_bits_per_symbol,
                            payload_rate=digital_payload_rate,
                            **({'joint_frame': True} if digital_joint_frame else {}))
                else:
                    recovered_indices, digital_stats = digital_client.transmit_indices_digital(
                        all_indices, route_to_high_mask, bits_high, bits_low,
                        current_snr_db, csi_mode=digital_csi_mode, seed=digital_seed,
                        bits_per_symbol=digital_bits_per_symbol,
                        payload_rate=digital_payload_rate)
                    stage_indices_rx = stage_erased_rx = None
                    low_indices_rx = low_erased_rx = None
                self.last_digital_stats = digital_stats
                tokens_after_channel = torch.zeros_like(x_proj_normalized)
                for i in range(B): 
                    rec_idx_i = recovered_indices[i]
                    route_high_i = route_to_high_mask[i]
                    if self.quantizer_type == 'rvq' and stage_indices_rx is not None:
                        stage_bad = stage_erased_rx[i]
                        hi = route_high_i & (~stage_bad.all(dim=1))
                        lo = (~route_high_i) & (~low_erased_rx[i])
                        for stage_no, codebook in enumerate(self.quantizer_high.codebooks):
                            valid_stage = hi & (~stage_bad[:, stage_no])
                            if valid_stage.any():
                                tokens_after_channel[i, valid_stage] += codebook(
                                    stage_indices_rx[i, valid_stage, stage_no])
                        if lo.any():
                            tokens_after_channel[i, lo] = self.quantizer_low.embedding(
                                low_indices_rx[i, lo])
                        known_i = hi | lo
                        # Diagnostic-only (plan v7.5, diag_channel_vs_codec.py):
                        # per-token survival category, so a token-level
                        # reconstruction-error analysis can tell partial-stage
                        # survivors apart from fully-intact and fully-concealed
                        # tokens without retraining anything.
                        if not hasattr(self, '_diag_category'):
                            self._diag_category = [None] * B
                        n_stage_ok = (~stage_bad).sum(dim=1)  # 0..3, per token
                        cat = torch.full_like(n_stage_ok, -1)  # -1 = low-branch token
                        cat[route_high_i & (n_stage_ok == 3)] = 3
                        cat[route_high_i & (n_stage_ok > 0) & (n_stage_ok < 3)] = 1  # partial
                        cat[route_high_i & (n_stage_ok == 0)] = 0  # fully erased high
                        cat[lo] = 3  # low survived -> treat as "intact" for its own branch
                        cat[(~route_high_i) & (~lo)] = 0  # low erased
                        self._diag_category[i] = cat.detach().cpu()
                    else:
                        known_i = rec_idx_i >= 0
                        hi = route_high_i & known_i
                        lo = (~route_high_i) & known_i
                        if hi.any():
                            tokens_after_channel[i, hi] = self.quantizer_high.embedding(rec_idx_i[hi])
                        if lo.any():
                            tokens_after_channel[i, lo] = self.quantizer_low.embedding(rec_idx_i[lo])
                    if not known_i.all():
                        tokens_after_channel[i] = self._n4_conceal_vectors(
                            tokens_after_channel[i], known_i,
                            self.full_image_num_patches_h,
                            self.full_image_num_patches_w)
                x_for_decoder_input = self.channel_to_decoder_proj(tokens_after_channel)
                self._diag_tokens_after_channel = tokens_after_channel.detach().cpu()
            else:
                self.last_digital_stats = None
                recovered_indices = transmit_digital_indices(all_indices, bits_per_index, snr_db=current_snr_db, seed=digital_seed)
                tokens_after_channel = torch.zeros_like(x_proj_normalized)
                for i in range(B): 
                    rec_idx_i = recovered_indices[i]
                    route_high_i = route_to_high_mask[i]
                    if self.quantizer_type == 'rvq':
                        # Packed S stage indices per token -> sum of stage codewords (H1).
                        tokens_after_channel[i, route_high_i] = self.quantizer_high.decode(rec_idx_i[route_high_i])
                    else:
                        tokens_after_channel[i, route_high_i] = self.quantizer_high.embedding(rec_idx_i[route_high_i])
                    tokens_after_channel[i, ~route_high_i] = self.quantizer_low.embedding(rec_idx_i[~route_high_i])
                x_for_decoder_input_fw = self.channel_to_decoder_proj(tokens_after_channel)
                x_for_decoder_input_bw = self.channel_to_decoder_proj(quantized_ste_output)
                x_for_decoder_input = x_for_decoder_input_fw
        
        reconstructed_image = self.img_decoder(
            x_vis_tokens=x_for_decoder_input,
            encoder_mask_boolean=encoder_input_mask_bool,
            full_image_num_patches_h=self.full_image_num_patches_h,
            full_image_num_patches_w=self.full_image_num_patches_w,
        )

        output_dict = {
            'reconstructed_image': reconstructed_image,
            'vq_loss': self.current_vq_loss,
            'fim_importance_scores': fim_raw_logits,
            # 2026-09-17 encoder-distillation support: the pre-quantization
            # projected feature [B, Np, quantizer_dim] -- same shape regardless
            # of encoder_embed_dim, since quantizer_dim is shared -- so a small-
            # encoder "student" and a large-encoder "teacher" can be compared
            # directly here even though their raw encoder outputs differ in dim.
            'x_proj_normalized': x_proj_normalized,
        }
        if self.last_digital_stats is not None:
            output_dict['digital_stats'] = self.last_digital_stats
        # Diagnostic-only (plan v7.5, diag_channel_vs_codec.py): expose the true,
        # pre-channel quantized tokens when available (eval branch only) so a
        # "channel-perfect" reconstruction can be computed externally without
        # reimplementing RVQ per-stage decode. None in the training branch
        # (var not defined there); zero cost otherwise (already computed).
        output_dict['quantized_ste_output_diag'] = locals().get('quantized_ste_output')
        output_dict['tokens_after_channel_diag'] = getattr(self, '_diag_tokens_after_channel', None)
        output_dict['route_to_high_mask_diag'] = locals().get('route_to_high_mask')
        output_dict['token_category_diag'] = (
            torch.stack(self._diag_category, dim=0) if getattr(self, '_diag_category', None) is not None
            and all(c is not None for c in self._diag_category) else None)
        return output_dict

@register_model
def ViT_Reconstruction_Model_Default(pretrained: bool = False, **kwargs) -> ViT_Reconstruction_Model:
    model_defaults = dict(
        patch_size=16, encoder_in_chans=3,
        encoder_embed_dim=384, encoder_depth=6, encoder_num_heads=6,
        decoder_embed_dim=192, decoder_depth=3, decoder_num_heads=3, # Decoder ViT blocks
        mlp_ratio=4.0, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        quantizer_dim=256, # Common dimension for both VQs
        bits_vq_high=10,   # Default bits for high-importance VQ
        bits_vq_low=6,     # Default bits for low-importance VQ
        quantizer_commitment_cost=0.25,
        quantizer_type='flat',      # 'flat' = HierarchicalQuantizer (legacy); 'rvq' = ResidualQuantizer (H1)
        rvq_stages=3,               # RVQ stage count (high path)
        rvq_bits_per_stage=6,       # Codebook bits per RVQ stage (effective rate = rvq_stages x rvq_bits_per_stage)
        rvq_ema_decay=0.0,          # >0 enables EMA codebook updates (RVQFIX, plan 7.2)
        # BUG FOUND AND FIXED 2026-09-14: these three were missing from this dict,
        # so the loop below (`if key in final_model_constructor_args.keys()`) silently
        # dropped --analog_symbols_per_token[/_high/_low] from every CLI invocation,
        # falling back to ViT_Reconstruction_Model.__init__'s own hardcoded default
        # (also 9) instead of raising an error. The already-recorded S=9 result is
        # UNAFFECTED (9 == 9, coincidentally), but the dual-rate flags below would
        # have silently trained a uniform S=9 model no matter what --high/--low were
        # set to, had this not been caught before launching. See RESULTS_OF_RECORD.md 6q.
        analog_symbols_per_token=9,
        analog_symbols_per_token_high=0,
        analog_symbols_per_token_low=0,
        init_values=0.0, use_learnable_pos_emb=False,
        drop_rate=0.0, drop_path_rate=0.1,
        fim_embed_dim=128, fim_depth=2, fim_num_heads=4, fim_drop_rate=0.1,
        fim_routing_threshold=0.6, # Default routing threshold
        routing_topk=63, # Fixed K high-fidelity tokens (P_high=0.08 for 784 tokens); <=0 falls back to threshold
        eps_stab=1e-6, # Stabilizer for the FIM-weighted VQ loss denominator (Eq. 4)
        disable_fim=False, # [ABLATION] random top-K routing + zero FIM BCE + unweighted VQ loss
    )
    
    final_model_constructor_args = model_defaults.copy()

    # Resolve img_size
    if 'input_size' in kwargs: final_model_constructor_args['img_size'] = kwargs['input_size']
    elif 'img_size' in kwargs: final_model_constructor_args['img_size'] = kwargs['img_size']
    else: final_model_constructor_args['img_size'] = 224

    # Override defaults with any relevant keys from kwargs
    for key in final_model_constructor_args.keys():
        if key in kwargs:
            final_model_constructor_args[key] = kwargs[key]
    
    model = ViT_Reconstruction_Model(**final_model_constructor_args)
    model.default_cfg = _cfg() # Define or import _cfg appropriately
    if pretrained:
        print("Warning: `pretrained=True` but no pretrained weight loading implemented in factory.")
    return model
