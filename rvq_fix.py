# rvq_fix.py -- RVQ stage-1 collapse fixes (plan 7.2 / wave 4 RVQFIX).
#
# (1) k-means initialization of stage codebooks from real encoder features:
#     collect x_proj_normalized for <= rvq_kmeans_images training images (batch 2,
#     seed 42), then per-stage Lloyd's k-means (stage1 on x, stage s+1 on the
#     residual after stage s), ~20 iterations, init = random features.
# (2) EMA codebook updates + dead-code restart live in ResidualQuantizer itself
#     (model_util.py); this module only feeds it the k-means initialization and
#     keeps the feature pool for restart fallback.
#
# The model is only used for feature extraction (encoder + proj + norm), so this
# runs in torch.no_grad() and never touches the channel/FIM/digital paths.

import torch

def collect_encoder_features(model, dataloader, device, max_images=800, seed=42):
    """Forward encoder-only on <= max_images training images; return (N, dim)
    tensor of x_proj_normalized (the exact features the high-path quantizer sees)."""
    features = []
    seen = 0
    model.eval()
    with torch.no_grad():
        for (semcom_input, _targets) in dataloader:
            imgs, bm_pos = semcom_input
            imgs = imgs.to(device, non_blocking=True)
            if bm_pos is not None:
                bm_pos = bm_pos.to(device, non_blocking=True)
            x = model.img_encoder(imgs, bm_pos)
            x = model.encoder_to_channel_proj(x)
            x = model.norm_before_quantizer(x)
            features.append(x.reshape(-1, x.shape[-1]).float().cpu())
            seen += imgs.shape[0]
            if seen >= max_images:
                break
    feats = torch.cat(features, dim=0)
    print(f"[RVQFIX] collected encoder features: {feats.shape[0]} vectors x "
          f"{feats.shape[1]} dims from {seen} images")
    return feats

def kmeans_lloyd(features, k, iters=20, seed=42, device=None):
    """Lloyd's k-means with seeded random-feature init; returns (centroids, distortion).

    features: (N, D). centroids: (k, D). distortion = mean min squared distance.
    Empty clusters are re-initialized to random features at each iteration.
    """
    if device is None:
        device = features.device
    N = features.shape[0]
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    perm = torch.randperm(N, generator=rng, device=device)[:k]
    centers = features[perm].clone()
    x_norm = (features ** 2).sum(-1, keepdim=True)          # (N, 1)
    c_norm = (centers ** 2).sum(-1).unsqueeze(0)            # (1, k)

    for it in range(iters):
        dists = x_norm + c_norm - 2.0 * (features @ centers.t())  # (N, k)
        assign = dists.argmin(1)
        new_centers = torch.zeros_like(centers)
        counts = torch.zeros(k, device=device)
        new_centers.index_add_(0, assign, features)
        counts.scatter_add_(0, assign, torch.ones(N, device=device))
        empty = counts < 1
        if empty.any():
            n_empty = int(empty.sum())
            ridx = torch.randint(0, N, (n_empty,), generator=rng, device=device)
            new_centers[empty] = features[ridx]
            counts[empty] = 1.0
        centers = new_centers / counts.unsqueeze(1)
    dists = x_norm + (centers ** 2).sum(-1).unsqueeze(0) - 2.0 * (features @ centers.t())
    final_assign = dists.argmin(1)
    distortion = dists.gather(1, final_assign.unsqueeze(1)).mean().item()
    return centers, distortion

def kmeans_init_rvq(quantizer, features, iters=20, seed=42, device=None, pool_cap=4096):
    """Per-stage k-means init: stage1 on x, stage s+1 on residual of stage s.

    Sets each stage codebook and stores a subsample of the original features as the
    restart fallback pool (quantizer.kmeans_pool).
    """
    if device is None:
        device = quantizer.codebooks[0].weight.device
    feats = features.to(device)
    residual = feats.clone()
    # deterministic subsample for the pool
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    pool_idx = torch.randperm(feats.shape[0], generator=rng, device=device)[:pool_cap]
    with torch.no_grad():
        pool = feats[pool_idx].clone()
        quantizer.kmeans_pool.resize_(pool.shape)
        quantizer.kmeans_pool.copy_(pool)
    for s in range(quantizer.num_stages):
        k = quantizer.num_embeddings_per_stage
        centers, distortion = kmeans_lloyd(residual, k, iters=iters, seed=seed + s,
                                           device=device)
        quantizer.set_codebook(s, centers)
        print(f"[RVQFIX] stage {s}: k-means init, {k} codes, {iters} iters, "
              f"distortion={distortion:.6f}")
        if s < quantizer.num_stages - 1:
            r_norm = (residual ** 2).sum(-1, keepdim=True)
            dists = r_norm + (centers ** 2).sum(-1).unsqueeze(0) - 2.0 * (residual @ centers.t())
            assign = dists.argmin(1)
            residual = residual - centers[assign]
    return distortion

def run_kmeans_init(model, trainloader, device, max_images=800, iters=20, seed=42):
    """Full k-means init for model.quantizer_high (RVQ). Returns final distortion."""
    feats = collect_encoder_features(model, trainloader, device, max_images=max_images,
                                     seed=seed)
    q = model.quantizer_high
    if not hasattr(q, 'num_stages'):
        print("[RVQFIX] quantizer_high is not an RVQ; skipping k-means init.")
        return None
    d = kmeans_init_rvq(q, feats, iters=iters, seed=seed, device=device)
    print(f"[RVQFIX] k-means init complete; final stage distortion={d:.6f}")
    return d
