import sys, argparse, torch, json
sys.path.insert(0, '__AQUASC_ROOT__')
import model, utils
R='__AQUASC_ROOT__/'
CFG = {
 'flagship': (R+'ckpt_semcom_reconstruction_yolo_fish_MIMO_LPIPS_FINAL/checkpoint-best_psnr.pth',512,8),
 'small512d2L': (R+'ckpt_polish_512d2L_from_tx_distill/checkpoint-best_psnr.pth',512,2),
 'tiny192d2L': (R+'ckpt_polish_192d2L_from_tx_distill/checkpoint-best_psnr.pth',192,2),
}
def ns(d,L):
    return argparse.Namespace(model="ViT_Reconstruction_Model_Default", input_size=224, patch_size=8,
      encoder_embed_dim=d, encoder_depth=L, encoder_num_heads=8, decoder_embed_dim=512, decoder_depth=6,
      decoder_num_heads=8, quantizer_dim=512, bits_vq_high=10, bits_vq_low=2, quantizer_type="rvq", rvq_stages=3,
      rvq_bits_per_stage=6, rvq_ema_decay=.99, quantizer_commitment_cost=.25, drop_rate=0., drop_path_rate=.1,
      fim_embed_dim=256, fim_depth=2, fim_num_heads=4, fim_drop_rate=.1, fim_routing_threshold=.7,
      routing_topk=63, eps_stab=1e-6, disable_fim=False)
out={}
for name,(ck,d,L) in CFG.items():
    net=utils.get_model(ns(d,L))
    sd=torch.load(ck,map_location='cpu'); sd=sd.get('model',sd)
    net.load_state_dict({k.removeprefix('module.'):v for k,v in sd.items()},strict=True)
    mods=['img_encoder','fim_module','encoder_to_channel_proj','norm_before_quantizer','quantizer_high','quantizer_low']
    seen=set(); rows=[]
    print(f'\n===== {name} ({ck.split("/")[-2]}) =====')
    for m in mods:
        mod=getattr(net,m)
        pp=0;pb=0;bp=0;bb=0
        for n,p in mod.named_parameters():
            if p.data_ptr() in seen: continue
            seen.add(p.data_ptr()); pp+=p.numel(); bp+=p.numel()*p.element_size()
        for n,b in mod.named_buffers():
            if b.data_ptr() in seen: continue
            seen.add(b.data_ptr()); pb+=b.numel(); bb+=b.numel()*b.element_size()
        rows.append((m,pp,pb,bp,bb))
        print(f'{m:26s} params={pp:>11,d} ({bp/1e6:7.3f} MB)  buffers={pb:>9,d} ({bb/1e6:7.3f} MB)')
        if m.startswith('quantizer'):
            for n,p in mod.named_parameters(): print(f'    PARAM  {n:30s} {tuple(p.shape)} {p.dtype} requires_grad={p.requires_grad}')
            for n,b in mod.named_buffers(): print(f'    BUFFER {n:30s} {tuple(b.shape)} {b.dtype}')
    # decoder-side modules (excluded)
    dec=sum(p.numel() for p in net.img_decoder.parameters())
    tot_p=sum(r[1] for r in rows); tot_b=sum(r[2] for r in rows)
    print(f'TRANSMITTER total params={tot_p:,d} ({tot_p/1e6:.3f} M)  buffers={tot_b:,d}; bytes params={sum(r[3] for r in rows)/1e6:.3f} MB = {sum(r[3] for r in rows)/2**20:.3f} MiB; +buffers={ (sum(r[3] for r in rows)+sum(r[4] for r in rows))/1e6:.3f} MB')
    print(f'(excluded: img_decoder={dec:,d}; whole-model params={sum(p.numel() for p in net.parameters()):,d})')
    # encoder internals
    enc=net.img_encoder
    print('img_encoder top-level children:', {n:sum(p.numel() for p in c.parameters()) for n,c in enc.named_children()}, '| own params:', [(n,tuple(p.shape)) for n,p in enc.named_parameters(recurse=False)])
    # quantizer_high vs state_dict entries
    print('state_dict keys under quantizers:', [k for k in net.state_dict() if k.startswith('quantizer')])
    out[name]=rows
