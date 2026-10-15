"""Torch-side matched5 evaluation. Run in siso-project-env.

Cache mode records source indices through a named clean transport. Receiver
evaluation only reads CRC-authenticated decoded indices, never source tokens.
Separate processes avoid the AquaSC/ROI-JSCC `model` package name collision.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from protocol import channel_draw, SYMBOLS, SNR_DB

CKPT = ROOT / "ckpt_semcom_reconstruction_yolo_fish_MIMO_LPIPS_FINAL/checkpoint-best_psnr.pth"
DATA = ROOT.parent / "yolo_fish_dataset_root"
EXPECTED_SHA = "b1d2eb25419af6ed6703a4aa886eb7143a8f1f350bd50e3371dacb3cde9e2894"

# Wave 21 F9 (2026-09-15): Site B zero-shot -- SAME flagship checkpoint (no
# retraining, matching the established S-1 protocol, RESULTS_OF_RECORD.md
# section 6c), different data root/detector/channel bank. Path and detector
# SHA match section 6b's own infrastructure table exactly.
DATA_SITEB = Path("__AQUASC_DATA__/yolo_fishdet_nb2_vd")
DETECTOR_SITEB = ROOT / "best_siteB.pt"
EXPECTED_DETECTOR_SITEB_SHA = "1d85f04db5c4e55efb5d29aabe4fc2a4b085279cc2d051a0479fdae79d802acb"
BANK_SITEB = ROOT / "channel_bank_siteb.npz"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""): h.update(block)
    return h.hexdigest()


def model_args(split, site="a", encoder_embed_dim=512, encoder_depth=8, encoder_num_heads=8,
              rvq_stages=3, rvq_bits_per_stage=6, bits_vq_low=2, routing_topk=63,
              disable_fim=False):
    # 2026-09-18 (AquaSC-lite): encoder_embed_dim/depth/num_heads are now
    # overridable -- defaults unchanged (512/8/8, the flagship) so every
    # existing call site is untouched.
    # 2026-09-19 (rate-matched codec variants, e.g. r061/r030): rvq_stages/
    # rvq_bits_per_stage/bits_vq_low/routing_topk are now ALSO overridable --
    # same default-preserving pattern (3/6/2/63 = flagship). Needed for any
    # checkpoint trained with a different RVQ/routing config, not just a
    # different encoder.
    # 2026-09-23 (coded ablation): disable_fim replays the ABL_BASE/
    # ABL_BB_NOFIM checkpoints' own training-time routing behavior (model.py's
    # internal seeded-random top-K branch, NOT an external route_override) --
    # required so these two checkpoints are evaluated the same way they were
    # trained, instead of silently defaulting to real FIM routing at eval
    # (utils.get_model reads disable_fim via getattr(args,'disable_fim',False)).
    # Default False preserves every existing call site.
    data_path = str(DATA_SITEB if site == "b" else DATA)
    return argparse.Namespace(
        model="ViT_Reconstruction_Model_Default", data_set="fish", data_path=data_path,
        input_size=224, patch_size=8, mask_ratio=0., eval=split == "test",
        num_object_classes=1, encoder_embed_dim=encoder_embed_dim, encoder_depth=encoder_depth,
        encoder_num_heads=encoder_num_heads, decoder_embed_dim=512, decoder_depth=6,
        decoder_num_heads=8, quantizer_dim=512, bits_vq_high=10, bits_vq_low=bits_vq_low,
        quantizer_type="rvq", rvq_stages=rvq_stages, rvq_bits_per_stage=rvq_bits_per_stage,
        rvq_ema_decay=.99,
        quantizer_commitment_cost=.25, drop_rate=0., drop_path_rate=.1,
        fim_embed_dim=256, fim_depth=2, fim_num_heads=4, fim_drop_rate=.1,
        fim_routing_threshold=.7, routing_topk=routing_topk, eps_stab=1e-6, seed=42,
        no_augment=True, device="cuda", disable_fim=disable_fim)


def load_aquasc(split, site="a", ckpt=None, encoder_embed_dim=512, encoder_depth=8,
                encoder_num_heads=8, rvq_stages=3, rvq_bits_per_stage=6, bits_vq_low=2,
                routing_topk=63, disable_fim=False):
    # 2026-09-18 (AquaSC-lite): pass ckpt=<path to a non-flagship checkpoint,
    # e.g. a distilled student> plus the matching encoder_* to evaluate a
    # different encoder through the same joint-frame pipeline. Default
    # (ckpt=None) is byte-for-byte the original behavior: the hardcoded
    # flagship CKPT, SHA-verified.
    # 2026-09-19: same for rvq_stages/rvq_bits_per_stage/bits_vq_low/
    # routing_topk -- required (not optional) whenever ckpt was trained with
    # a non-flagship RVQ/routing config, since strict=True state_dict
    # loading will fail loudly on a codebook shape mismatch otherwise.
    import model
    import utils
    ckpt_path = Path(ckpt) if ckpt else CKPT
    if ckpt is None:
        assert sha256(CKPT) == EXPECTED_SHA, "unexpected flagship checkpoint"
    net = utils.get_model(model_args(split, site, encoder_embed_dim, encoder_depth,
                                     encoder_num_heads, rvq_stages, rvq_bits_per_stage,
                                     bits_vq_low, routing_topk, disable_fim))
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    net.load_state_dict({k.removeprefix("module."):v for k,v in sd.items()}, strict=True)
    return net.cuda().eval()


def dataset(split, site="a"):
    import datasets
    ds = datasets.build_dataset(is_train=False, args=model_args(split, site))
    expected = {"valid":437, "test":432} if site == "a" else {"valid":487, "test":455}
    assert len(ds) == expected[split]
    return ds


def decode_frame(net, values, mask, ok):
    """Receiver inputs are only received indices/mask and CRC acceptance.

    2026-09-19: uses net.quantizer_high.decode() / .embedding() -- the
    model's own canonical (un)packing -- instead of a hardcoded 6-bit-per-
    stage shift/mask, so this works for ANY rvq_bits_per_stage/num_stages,
    not just the flagship's 3x6. Byte-identical to the old hardcoded version
    when net actually IS the flagship (3 stages x 6 bits): verified via the
    existing decoder_parity_max_abs_error<=1.19e-7 check in cache_source().
    """
    values = torch.as_tensor(values.astype(np.int64), device="cuda")
    mask = torch.as_tensor(mask, dtype=torch.bool, device="cuda")
    ok = torch.as_tensor(ok, dtype=torch.bool, device="cuda")
    batch = values.shape[0]
    tokens = torch.zeros(batch,784,512,device="cuda")
    hi,lo = mask & ok[:,None], ~mask & ok[:,None]
    if hi.any(): tokens[hi] = net.quantizer_high.decode(values[hi])
    if lo.any(): tokens[lo] = net.quantizer_low.embedding(values[lo])
    # Fully lost frames use exactly the existing N4 all-zero fallback.
    return net.img_decoder(x_vis_tokens=net.channel_to_decoder_proj(tokens),
        encoder_mask_boolean=torch.zeros(batch,784,dtype=torch.bool,device="cuda"),
        full_image_num_patches_h=28,full_image_num_patches_w=28)


def compute_router_masks(router,ds,net,site,routing_topk,router_seed):
    """External top-K routing mask for the WHOLE split, one row per image in
    ds.img_files order (matching cache_source's shuffle=False iteration).

    2026-09-23 (coded ablation/routing comparison, R3 #5-style routing sweep
    against the LDPC-coded headline pipeline, not the old uncoded-digital one
    routing_eval.py/routing_eval_fixup.py used). Reuses routing_baselines.py's
    existing score functions unmodified -- this function only adds the
    plumbing to get their [N,784] output into model.py's route_override.
    """
    import routing_baselines as rb
    N,NP=len(ds.img_files),784
    if router=="uniform":
        scores=rb.uniform_scores(NP,batch_size=N)
    elif router=="random":
        scores=rb.random_scores(NP,seed=router_seed,batch_size=N)
    elif router=="oracle_box":
        boxes=[rb.parse_yolo_label(lp) for lp in ds.label_files]
        scores=rb.oracle_box_scores(boxes,PS=8,NP=NP,H=224,W=224)
    elif router=="detector_heatmap":
        yolo=detector(site)
        from torchvision import transforms
        to_tensor=transforms.ToTensor()
        rows=[]
        with torch.no_grad():
            for i,(img_path,lp) in enumerate(zip(ds.img_files,ds.label_files)):
                from PIL import Image
                im=Image.open(img_path).convert("RGB")
                if im.size!=(224,224): im=im.resize((224,224),Image.BICUBIC)
                x=to_tensor(im).unsqueeze(0).cuda()
                pred=yolo.predict(x,imgsz=224,conf=0.001,iou=0.5,verbose=False,device="cuda")[0]
                boxes=pred.boxes.xyxy.detach().cpu()
                confs=pred.boxes.conf.detach().cpu()
                rows.append(rb.detector_heatmap_scores(boxes,confs,PS=8,NP=NP,H=224,W=224))
                if (i+1)%50==0: print(f"  detector_heatmap scored {i+1}/{N}",flush=True)
        scores=torch.cat(rows,dim=0)
    else:
        raise ValueError(f"unknown router {router!r}")
    mask,_=rb.apply_topk(scores,routing_topk)
    assert mask.shape==(N,NP)
    assert bool((mask.sum(dim=1)==routing_topk).all()), \
        "apply_topk must give exactly routing_topk True per row -- RVQ split downstream assumes this"
    return mask


def cache_source(cli):
    import datasets
    import digital_client
    site=getattr(cli,"site","a")
    ckpt=getattr(cli,"ckpt",None)
    enc_dim=getattr(cli,"encoder_embed_dim",512)
    enc_depth=getattr(cli,"encoder_depth",8)
    enc_heads=getattr(cli,"encoder_num_heads",8)
    # 2026-09-19: RVQ/routing config, for checkpoints trained with a
    # different codec (e.g. the rate-matched r061/r030 variants) -- defaults
    # unchanged (flagship 3/6/2/63), so every existing call is untouched.
    rvq_stages=getattr(cli,"rvq_stages",3)
    rvq_bits_per_stage=getattr(cli,"rvq_bits_per_stage",6)
    bits_vq_low=getattr(cli,"bits_vq_low",2)
    routing_topk=getattr(cli,"routing_topk",63)
    disable_fim=getattr(cli,"disable_fim",False)
    router=getattr(cli,"router",None)
    router_seed=getattr(cli,"router_seed",42)
    out=Path(cli.out_dir)
    out.mkdir(parents=True,exist_ok=False)
    net=load_aquasc(cli.split,site,ckpt,enc_dim,enc_depth,enc_heads,
                    rvq_stages,rvq_bits_per_stage,bits_vq_low,routing_topk,disable_fim)
    ds=dataset(cli.split,site)
    # 2026-09-23 (coded ablation/routing comparison): precompute an external
    # routing mask for the WHOLE split up front (image order = ds.img_files,
    # matching shuffle=False below), one row per image, exactly routing_topk
    # True entries per row (apply_topk guarantees fixed cardinality -- the
    # RVQ high/low split downstream assumes this). None/'fim' leaves
    # route_override_all=None, i.e. byte-identical to the pre-existing
    # behavior (the model's own real FIM routing).
    route_override_all=None
    if router not in (None,"fim"):
        route_override_all=compute_router_masks(router,ds,net,site,routing_topk,router_seed)
    loader=torch.utils.data.DataLoader(ds,batch_size=8,shuffle=False,num_workers=0,
                                      collate_fn=datasets.yolo_collate_fn)
    token_rows,mask_rows=[],[]
    latest={}
    # 2026-09-19: was hardcoded to the flagship's 3x6-bit high path (shifts
    # 12,6,0 / mask 63) and 2-bit low path (mask 3). Now reads the actual
    # config straight off the loaded model's own quantizers -- self-
    # consistent by construction, cannot silently mismatch what was loaded.
    num_stages=net.quantizer_high.num_stages
    bits_per_stage=net.quantizer_high.bits_per_stage
    low_mask_val=net.quantizer_low.num_embeddings-1
    def clean_transport_for_cache(tv,hm,*args,**kwargs):
        latest["tokens"],latest["mask"]=tv.cpu().numpy(),hm.cpu().numpy()
        token_rows.append(latest["tokens"])
        mask_rows.append(latest["mask"])
        stages=net.quantizer_high.unpack_indices(tv,num_stages,bits_per_stage).movedim(0,-1)
        return (tv,{},stages,torch.zeros_like(stages,dtype=torch.bool),tv & low_mask_val,
                torch.zeros_like(hm))
    old=digital_client.transmit_indices_digital
    digital_client.transmit_indices_digital=clean_transport_for_cache
    parity_error=None
    try:
        with torch.no_grad():
            for b,(inputs,_) in enumerate(loader):
                imgs,masks=inputs
                assert not masks.any()
                route_override=None
                if route_override_all is not None:
                    route_override=route_override_all[b*8:b*8+imgs.shape[0]].cuda()
                with contextlib.redirect_stdout(io.StringIO()):
                    result=net(img=imgs.cuda(),bm_pos=masks.cuda(),_eval=True,
                               digital_ldpc=True,eval_snr_db=5,route_override=route_override)
                if b==0:
                    independent=decode_frame(net,latest["tokens"],latest["mask"],
                                             np.ones(len(imgs),dtype=bool))
                    parity_error=float((independent-result["reconstructed_image"]).abs().max())
                    assert parity_error < 1e-6, parity_error
                if (b+1)%10==0: print(f"cached {min((b+1)*8,len(ds))}/{len(ds)}",flush=True)
    finally: digital_client.transmit_indices_digital=old
    names=np.asarray([Path(p).name for p in ds.img_files])
    np.savez_compressed(out/"source.npz",tokens=np.concatenate(token_rows),
                        high_mask=np.concatenate(mask_rows),names=names)
    ckpt_path_actual = Path(ckpt) if ckpt else CKPT
    meta=dict(split=cli.split,site=site,checkpoint=str(ckpt_path_actual),
              checkpoint_sha256=sha256(ckpt_path_actual),
              encoder_embed_dim=enc_dim,encoder_depth=enc_depth,encoder_num_heads=enc_heads,
              rvq_stages=rvq_stages,rvq_bits_per_stage=rvq_bits_per_stage,
              bits_vq_low=bits_vq_low,routing_topk=routing_topk,
              disable_fim=disable_fim,router=(router or "fim"),router_seed=router_seed,
              n=len(ds),decoder_parity_max_abs_error=parity_error,
              contents="source encoder output only; not channel evaluation")
    (out/"meta.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(json.dumps(meta),flush=True)


def detector(site="a"):
    from ultralytics import YOLO
    if site == "b":
        assert sha256(DETECTOR_SITEB) == EXPECTED_DETECTOR_SITEB_SHA, "unexpected Site B detector"
        net=YOLO(str(DETECTOR_SITEB))
    else:
        net=YOLO(str(ROOT/"best.pt"))
    net.to("cuda")
    net.fuse()
    net.eval()
    return net


def evaluate_aquasc(cli):
    import metrics
    import stats
    from torchmetrics.detection import MeanAveragePrecision
    out=Path(cli.out_dir)
    out.mkdir(parents=True,exist_ok=False)
    rx=np.load(cli.receiver,allow_pickle=False)
    meta=json.loads(Path(cli.receiver).with_suffix(".json").read_text())
    # Wave 21: was a hardcoded ==6144 check, tied to the original 0.122449
    # operating point only. This function's own decode logic (below) reads
    # tokens/mask/frame_ok from receiver.npz directly and never uses the
    # symbol count -- generalized to accept any --symbols phy.py produced.
    site=meta.get("site","a")
    assert meta["split"]==cli.split and meta["symbols_per_image"]>0
    # 2026-09-18 (AquaSC-lite): rebuild the SAME (possibly non-flagship)
    # model the receiver's frames were actually encoded with, read back from
    # phy.py's own meta (propagated from cache_source) -- not re-guessed
    # from CLI flags, so this can never silently mismatch what was cached.
    ckpt=meta.get("checkpoint")
    enc_dim=meta.get("encoder_embed_dim",512)
    enc_depth=meta.get("encoder_depth",8)
    enc_heads=meta.get("encoder_num_heads",8)
    rvq_stages=meta.get("rvq_stages",3)
    rvq_bits_per_stage=meta.get("rvq_bits_per_stage",6)
    bits_vq_low=meta.get("bits_vq_low",2)
    routing_topk=meta.get("routing_topk",63)
    ds=dataset(cli.split,site)
    net=load_aquasc(cli.split,site,ckpt,enc_dim,enc_depth,enc_heads,
                    rvq_stages,rvq_bits_per_stage,bits_vq_low,routing_topk)
    yolo=detector(site)
    assert rx["names"].tolist()==[Path(p).name for p in ds.img_files]
    gts=[ds[i][1][1] for i in range(len(ds))]
    n,m_count=rx["frame_ok"].shape
    per_image=np.zeros((n,m_count),dtype=np.float32)
    pooled=[]
    start=time.monotonic()
    with torch.no_grad():
        for m in range(m_count):
            metric=MeanAveragePrecision(iou_type="bbox").cuda()
            for offset in range(0,n,8):
                end=min(n,offset+8)
                recon=decode_frame(net,rx["tokens"][offset:end,m],
                    rx["mask"][offset:end,m],rx["frame_ok"][offset:end,m])
                preds=metrics.yolo_predict(yolo,recon,conf=.001,iou=.5,device="cuda")
                for j,pred in enumerate(preds):
                    i=offset+j
                    per_image[i,m]=stats.per_image_ap50(pred["boxes"].cpu().numpy(),
                        pred["scores"].cpu().numpy(),gts[i]["boxes"].numpy(),iou_thr=.5)
                    metric.update([dict(boxes=pred["boxes"],scores=pred["scores"],
                        labels=torch.zeros(len(pred["scores"]),dtype=torch.int64,device="cuda"))],
                        [dict(boxes=gts[i]["boxes"].cuda(),labels=gts[i]["labels"].cuda())])
            pooled.append(float(metric.compute()["map_50"]))
            print(json.dumps(dict(realization=m,mean_ap=float(per_image[:,m].mean()),
                                  pooled_map50=pooled[-1])),flush=True)
    np.savez_compressed(out/"metrics.npz",per_image_ap=per_image,pooled_map50=pooled,
                        names=rx["names"])
    # Wave 21 F9 bug fix (2026-09-15): this hardcoded ROOT/"best.pt" regardless
    # of `site` -- the actual predictions above already used the correct
    # detector (`yolo=detector(site)`), so this was a provenance-logging bug
    # only (wrong SHA recorded), not a correctness bug in the AP numbers.
    # Caught by RESULTS_OF_RECORD.md's own detector-SHA convention (section 6b
    # cites best_siteB.pt's SHA explicitly) -- the mismatch was visible on
    # the very first Site B smoke-test summary.json, not discovered later.
    detector_path = DETECTOR_SITEB if site == "b" else (ROOT/"best.pt")
    # 2026-09-18: this used to hardcode checkpoint_sha256=EXPECTED_SHA (the
    # flagship's SHA) regardless of which checkpoint was actually used --
    # the same provenance-logging bug already fixed once for detector_sha256
    # above. `meta` (spread via **meta) already carries the correct
    # checkpoint_sha256 propagated from cache_source via phy.py -- don't
    # clobber it.
    summary=dict(**meta,mean_per_image_ap50=float(per_image.mean()),
        pooled_map50=float(np.mean(pooled)),evaluation_seconds=time.monotonic()-start,
        receiver_file=str(Path(cli.receiver).resolve()),receiver_sha256=sha256(cli.receiver),
        detector_sha256=sha256(detector_path),
        metric_conf=.001,metric_iou=.5,N_training_seeds=1)
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary),flush=True)


def evaluate_roi(cli):
    # SNR sweep (2026-09-15): this function, like phy.py's original
    # transmit_cache, only ever used channel_draw() (the fixed-formula
    # shortcut verified to match channel_bank.npz ONLY at snr_idx=2, i.e.
    # 5dB) and the hardcoded module constant SNR_DB=5.0 passed to
    # SNR_info=. It was never correct at any other SNR. The pre-existing
    # f1_perimage/roijscc_rcpp32_snr*.npy files at other SNRs come from a
    # different, untraceable script (no generator found in this repo at
    # HEAD) -- their bank-index parity with the new AquaSC sweep cannot be
    # verified by reading code, so per the reviewing instruction (confirm
    # same bank index both sides, every SNR) this function is fixed to read
    # h/noise directly from channel_bank.npz at the requested --snr-db's own
    # index, for every one of the 6 points, and the per-SNR arrays are
    # regenerated here rather than trusted from the old files.
    snr_db=float(getattr(cli,"snr_db",5.0))
    import metrics
    import stats
    from PIL import Image
    from torchvision.transforms.functional import to_tensor
    from torchmetrics.detection import MeanAveragePrecision
    roi_dir=ROOT/"competitors/ROI-JSCC"
    sys.path.insert(0,str(roi_dir))
    from roijscc_fish_common import build_model,BlockRayleighChannel,rate_constants,compute_roi_index_from_norm
    out=Path(cli.out_dir)
    out.mkdir(parents=True,exist_ok=False)
    ckpt=roi_dir/"saved_models/ROIIT_FISH_Rayleigh_SNR010_rcpp032_PSNR_ROIJSCC_roi-largest_box_best.pt"
    net=build_model(rcpp=32,chan_type="Rayleigh",device="cuda")
    net.channel=BlockRayleighChannel()
    sd=torch.load(ckpt,map_location="cpu")
    if "state_dict" in sd: sd=sd["state_dict"]
    net.load_state_dict(sd,strict=True)
    net.eval()
    measured=[]
    def count_symbols(module,inputs):
        actual=inputs[0].numel()//inputs[0].shape[0]//2
        assert actual==SYMBOLS
        measured.append(actual)
    handle=net.channel.register_forward_pre_hook(count_symbols)
    ds,yolo=dataset(cli.split),detector()
    names=np.asarray([Path(p).name for p in ds.img_files])
    bank_h=bank_n=None
    if cli.split=="test":
        bank=np.load(ROOT/"channel_bank.npz")
        assert names.tolist()==bank["image_names"].tolist()
        bank_snr=[float(s) for s in bank["snr_db"]]
        if snr_db not in bank_snr:
            raise ValueError(f"snr_db={snr_db} not in channel_bank.npz grid {bank_snr}")
        s_idx=bank_snr.index(snr_db)
        need_reals=2*SYMBOLS
        assert bank["n"].shape[-1]>=need_reals
        bank_h=bank["h"][:,s_idx,:]              # [n_images, M] complex64
        bank_n=bank["n"][:,s_idx,:,:need_reals]   # [n_images, M, 2*SYMBOLS] float32
        if snr_db==5.0:
            for i in (0,len(ds)-1):
                for m in range(cli.m):
                    assert bank_h[i,m]==channel_draw(cli.split,i,m)[0]
    per_image=np.zeros((len(ds),cli.m),dtype=np.float32)
    pooled_metrics=[MeanAveragePrecision(iou_type="bbox").cuda() for _ in range(cli.m)]
    start=time.monotonic()
    with torch.no_grad():
        for i in range(len(ds)):
            gt=ds[i][1][1]
            roi=compute_roi_index_from_norm(gt["boxes"].numpy()/224.,mode="largest_box")
            # Match released ROI-JSCC preprocessing: raw RGB -> bicubic 256.
            original=to_tensor(Image.open(ds.img_files[i]).convert("RGB")).unsqueeze(0).cuda()
            img256=torch.nn.functional.interpolate(original,size=(256,256),mode="bicubic",align_corners=False)
            if cli.split=="test":
                h_arr=bank_h[i,:cli.m]
                raw=bank_n[i,:cli.m,:]  # [m, 2*SYMBOLS] float32
                noise_arr=raw[:,:SYMBOLS]+1j*raw[:,SYMBOLS:]
            else:
                draws=[channel_draw(cli.split,i,m) for m in range(cli.m)]
                h_arr=np.array([d[0] for d in draws])
                noise_arr=np.stack([d[1] for d in draws])
            net.channel.h=torch.as_tensor(h_arr,device="cuda").view(-1,1)
            net.channel.noise_real=torch.as_tensor(np.concatenate([noise_arr.real,noise_arr.imag],axis=1),device="cuda")
            # Follow the released evaluator's batch=1 exactly. A batching
            # optimization produced 1.36e-4 reconstruction differences in the
            # validation parity check, so it is not used for final comparisons.
            saved_h,saved_noise=net.channel.h,net.channel.noise_real
            preds=[]
            for m in range(cli.m):
                net.channel.h=saved_h[m:m+1]
                net.channel.noise_real=saved_noise[m:m+1]
                recon256=torch.clamp((net(img256*2-1,ROI_Index=roi,SNR_info=snr_db)+1)/2,0,1)
                recon=torch.nn.functional.interpolate(recon256,size=(224,224),mode="bicubic",align_corners=False).clamp(0,1)
                preds.extend(metrics.yolo_predict(yolo,recon,conf=.001,iou=.5,device="cuda"))
            for m,pred in enumerate(preds):
                per_image[i,m]=stats.per_image_ap50(pred["boxes"].cpu().numpy(),
                    pred["scores"].cpu().numpy(),gt["boxes"].numpy(),iou_thr=.5)
                pooled_metrics[m].update([dict(boxes=pred["boxes"],scores=pred["scores"],
                    labels=torch.zeros(len(pred["scores"]),dtype=torch.int64,device="cuda"))],
                    [dict(boxes=gt["boxes"].cuda(),labels=gt["labels"].cuda())])
            if (i+1)%50==0:
                print(f"ROI-JSCC {i+1}/{len(ds)} x {cli.m}: AP={per_image[:i+1].mean():.4f}",flush=True)
    handle.remove()
    pooled=[float(metric.compute()["map_50"]) for metric in pooled_metrics]
    np.savez_compressed(out/"metrics.npz",per_image_ap=per_image,pooled_map50=pooled,names=names)
    summary=dict(split=cli.split,system="ROI-JSCC",rcpp=32,symbols_per_image=measured[0],
        channel_uses_per_px=measured[0]/50176,checkpoint=str(ckpt),checkpoint_sha256=sha256(ckpt),
        detector_sha256=sha256(ROOT/"best.pt"),roi_mode="largest_box",
        roi_side_information="ground-truth boxes, released baseline protocol, metadata not charged",
        n=len(ds),M=cli.m,snr_db=snr_db,csi="perfect_receiver",fading="one CN(0,1) h per image",
        mean_per_image_ap50=float(per_image.mean()),pooled_map50=float(np.mean(pooled)),
        metric_conf=.001,metric_iou=.5,training_epochs=100,N_training_seeds=1,
        evaluation_batch_size=1,
        seconds=time.monotonic()-start)
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary),flush=True)


def compare(cli):
    import stats
    out=Path(cli.out_dir)
    out.mkdir(parents=True,exist_ok=False)
    a_dir,b_dir=Path(cli.aqua_dir),Path(cli.roi_dir)
    a_meta=json.loads((a_dir/"summary.json").read_text())
    b_meta=json.loads((b_dir/"summary.json").read_text())
    a,b=np.load(a_dir/"metrics.npz"),np.load(b_dir/"metrics.npz")
    assert a["names"].tolist()==b["names"].tolist()
    assert a_meta["symbols_per_image"]==b_meta["symbols_per_image"]==SYMBOLS
    assert a_meta["split"]==b_meta["split"]==cli.split
    assert a_meta["detector_sha256"]==b_meta["detector_sha256"]
    x,y=a["per_image_ap"],b["per_image_ap"]
    assert x.shape==y.shape
    if cli.split=="test": assert x.shape==(432,10)
    boot=stats.paired_bootstrap(x.mean(axis=1),y.mean(axis=1),n_boot=10000,seed=42)
    boot.pop("boot_means")
    wilcoxon=stats.wilcoxon_signed_rank(x.mean(axis=1),y.mean(axis=1))
    pooled_delta=float(a["pooled_map50"].mean()-b["pooled_map50"].mean())
    summary=dict(split=cli.split,snr_db=SNR_DB,symbols_per_image=SYMBOLS,
        channel_uses_per_px=SYMBOLS/50176,n=x.shape[0],M=x.shape[1],
        aquasc_mean_per_image_ap50=float(x.mean()),roijscc_mean_per_image_ap50=float(y.mean()),
        aquasc_pooled_map50=float(a["pooled_map50"].mean()),
        roijscc_pooled_map50=float(b["pooled_map50"].mean()),
        pooled_difference=pooled_delta,paired_per_image_bootstrap=boot,
        wilcoxon=wilcoxon,per_realization_mean_ap_differences=(x-y).mean(axis=0).tolist(),
        meets_preregistered_win=bool(pooled_delta>0 and boot["ci_low"]>0),
        aquasc_metrics_sha256=sha256(a_dir/"metrics.npz"),
        roi_metrics_sha256=sha256(b_dir/"metrics.npz"),
        limitations=["one training seed per model","Site A only",
                     "AquaSC 300 training epochs versus ROI-JSCC 100",
                     "ROI-JSCC granted its original ground-truth ROI protocol",
                     "CI resamples images, not independent videos or training seeds"])
    if cli.split=="test":
        old=np.load(ROOT/"f1_perimage/roijscc_rcpp32_snr5.npy")
        summary["released_roi_reproduction"]=dict(
            prior_mean=float(old.mean()),new_mean=float(y.mean()),
            max_abs_per_image_mean_difference=float(np.abs(old-y.mean(axis=1)).max()))
    (out/"comparison.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2),flush=True)


def integration_smoke(cli):
    import digital_client
    out=Path(cli.out_dir)
    out.mkdir(parents=True,exist_ok=False)
    source=np.load(Path(__file__).parent/"runs/cache_valid_v1/source.npz")
    receiver=np.load(Path(__file__).parent/"runs/phy_valid_qpsk_v1/receiver.npz")
    digital_client.set_bridge_python("__SIONNA_PY__")
    tv=torch.as_tensor(source["tokens"][:16],dtype=torch.long)
    hm=torch.as_tensor(source["high_mask"][:16],dtype=torch.bool)
    try:
        result=digital_client.transmit_indices_digital(tv,hm,18,2,5,seed=914_000_000,
            rvq_stage_uep=True,return_stage_status=True,joint_frame=True)
        recovered,st,_,bad,_,low_bad=result
        expected=receiver["tokens"][:16,0].astype(np.int64)
        ok=receiver["frame_ok"][:16,0]
        expected[~ok]=-1
        assert np.array_equal(recovered.numpy(),expected), "live bridge differs from cached PHY"
        assert np.isclose(st["channel_uses_per_px"],SYMBOLS/50176,rtol=0,atol=1e-15)
        assert ok.any() and (~ok).any(), "smoke must exercise success and loss"
        for joint,expected_symbols in ((False,3584),(True,6144)):
            clean,clean_stats=digital_client.transmit_indices_digital(tv[:1],hm[:1],18,2,
                40,seed=77,rvq_stage_uep=True,joint_frame=joint)
            assert torch.equal(clean,tv[:1])
            assert clean_stats["channel_uses_per_px"]==expected_symbols/50176
        ds,net=dataset("valid"),load_aquasc("valid")
        max_error=0.
        # Exercise one passed and one failed frame through model -> torch client
        # -> JSON subprocess -> Sionna -> real model RVQ decoder.
        checks=[int(np.flatnonzero(ok)[0]),int(np.flatnonzero(~ok)[0])]
        with torch.no_grad():
            for i in checks:
                (img,bm),_=ds[i]
                with contextlib.redirect_stdout(io.StringIO()):
                    actual=net(img=img[None].cuda(),bm_pos=bm[None].cuda(),_eval=True,
                        digital_ldpc=True,digital_joint_frame=True,eval_snr_db=5,
                        digital_seed=914_000_000+i)["reconstructed_image"]
                reference=decode_frame(net,receiver["tokens"][i:i+1,0],
                    receiver["mask"][i:i+1,0],receiver["frame_ok"][i:i+1,0])
                error=float((actual-reference).abs().max())
                max_error=max(max_error,error)
                assert error < 1e-6, (i,error)
        summary=dict(passed=True,bridge_images=16,successes=int(ok.sum()),
            failures=int((~ok).sum()),model_max_abs_error=max_error,
            legacy_roundtrip=True,joint_roundtrip=True,symbols_per_image=6144)
        (out/"integration.json").write_text(json.dumps(summary,indent=2)+"\n")
        print(json.dumps(summary),flush=True)
    finally: digital_client.shutdown()


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--phase",choices=("cache","aquasc","roi","compare","smoke"),required=True)
    parser.add_argument("--split",choices=("valid","test"),required=True)
    parser.add_argument("--out-dir",required=True)
    parser.add_argument("--receiver")
    parser.add_argument("--m",type=int,default=3)
    parser.add_argument("--aqua-dir")
    parser.add_argument("--roi-dir")
    parser.add_argument("--snr-db",type=float,default=5.0)
    parser.add_argument("--site",choices=("a","b"),default="a",
                        help="Wave 21 F9: 'b' evaluates Site B, zero-shot, same flagship "
                             "checkpoint -- only --phase cache reads this directly; "
                             "--phase aquasc picks it up from the receiver's own meta.json "
                             "(set by phy.py from cache_source's meta.json), not from this flag.")
    parser.add_argument("--ckpt",default=None,
                        help="2026-09-18 (AquaSC-lite): path to a non-flagship checkpoint "
                             "(e.g. a distilled small-encoder student) -- --phase cache only. "
                             "Requires --encoder-embed-dim/--encoder-depth/--encoder-num-heads "
                             "to match. Propagated automatically to --phase aquasc via the "
                             "receiver's own meta.json -- do not pass --ckpt there too.")
    parser.add_argument("--encoder-embed-dim",type=int,default=512)
    parser.add_argument("--encoder-depth",type=int,default=8)
    parser.add_argument("--encoder-num-heads",type=int,default=8)
    parser.add_argument("--rvq-stages",type=int,default=3)
    parser.add_argument("--rvq-bits-per-stage",type=int,default=6)
    parser.add_argument("--bits-vq-low",type=int,default=2)
    parser.add_argument("--routing-topk",type=int,default=63)
    parser.add_argument("--disable-fim",action="store_true",
                        help="2026-09-23 (coded ablation): replay the ABL_BASE/ABL_BB_NOFIM "
                             "checkpoints' own training-time random top-K routing (model.py's "
                             "internal seeded branch) instead of real FIM routing. --phase cache "
                             "only. Mutually exclusive with --router (the model's own routing "
                             "branch, real-FIM or seeded-random, is used either way; --router "
                             "would be silently ignored/overridden and is refused instead).")
    parser.add_argument("--router",default=None,
                        choices=(None,"fim","oracle_box","random","uniform","detector_heatmap"),
                        help="2026-09-23 (coded ablation/routing comparison): evaluate the FULL "
                             "checkpoint's own encoder/quantizer/decoder with an EXTERNAL routing "
                             "mask substituted for the model's own FIM top-K (via model.py's "
                             "route_override hook), computed once for the whole split via "
                             "routing_baselines.py before the cache loop. None/'fim' = unchanged "
                             "default behavior (the model's own real FIM routing, no override). "
                             "--phase cache only.")
    parser.add_argument("--router-seed",type=int,default=42,
                        help="Seed for --router random (routing_baselines.random_scores); "
                             "irrelevant for the other router choices.")
    cli=parser.parse_args()
    if cli.disable_fim and cli.router not in (None,"fim"):
        parser.error("--disable-fim and --router are mutually exclusive")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    if cli.phase!="compare": assert torch.cuda.is_available()
    if cli.phase=="cache": cache_source(cli)
    elif cli.phase=="aquasc": evaluate_aquasc(cli)
    elif cli.phase=="roi": evaluate_roi(cli)
    elif cli.phase=="compare": compare(cli)
    elif cli.phase=="smoke": integration_smoke(cli)
