import io, os, json, collections, math
import numpy as np
from PIL import Image
D="__AQUASC_DATA__/yolo_fish_dataset_root/test/images"
names=sorted(f for f in os.listdir(D) if f.lower().endswith((".png",".jpg",".jpeg")))
def size(q):
    out=[]
    for n in names:
        im=Image.open(os.path.join(D,n)).convert("RGB")
        if im.size!=(224,224): im=im.resize((224,224),Image.BICUBIC)
        b=io.BytesIO(); im.save(b,"WEBP",quality=q,method=4); out.append(len(b.getvalue()))
    return np.array(out)
R="__AQUASC_ROOT__/"
pts=[("(a) 1d headline-matched",R+"webp_real_channel_sweep_out/manifest.json",8,744,"16-QAM (default, undeclared)"),
("(b1) lowrates q=1",R+"webp_real_channel_sweep_lowrates_out/q1/manifest.json",1,None,"16-QAM (default)"),
("(b2) lowrates q=81",R+"webp_real_channel_sweep_lowrates_out/q81/manifest.json",81,None,"16-QAM (default)"),
("(c) midrate QPSK q=1 cap372",R+"webp_real_channel_sweep_midrate_out/manifest.json",1,372,"QPSK explicit"),
("(d) midrate 16QAM q=9 cap744",R+"webp_real_channel_sweep_midrate16qam_out/manifest.json",9,744,"16-QAM explicit")]
res=[]
for lab,mp,q,cap,mod in pts:
    s=size(q); m=json.load(open(mp))
    bps=2 if "QPSK" in mod else 4
    per=collections.Counter((r["snr"],r["m"]) for r in m)
    npair=set(per.values())
    imgs=set(r["name"] for r in m)
    capimgs=set(r["name"] for r in m if r.get("capacity_failure"))
    nrec=len(m)
    # blocks per image from manifest (uncapped/ok images)
    nb={}
    for r in m:
        if not r.get("capacity_failure"): nb[r["name"]]=r["n_blocks"]
    nbv=np.array(list(nb.values()))
    capm = (s>cap).sum() if cap else None
    d=dict(point=lab,q=q,mod=mod,cap=cap,n_images_enc=len(s),bytes_min=int(s.min()),bytes_mean=float(s.mean()),bytes_max=int(s.max()),
      over_cap_from_streams=(int(capm) if cap else None),capfail_images_from_manifest=len(capimgs),manifest_images=len(imgs),
      records=nrec,records_per_snr_m=sorted(npair),n_snr_m=len(per),
      blocks_min=int(nbv.min()),blocks_mean=float(nbv.mean()),blocks_max=int(nbv.max()),
      blocks_dist={int(k):int(v) for k,v in sorted(collections.Counter(nbv.tolist()).items())},
      ceil_blocks_from_streams_mean=float(np.mean([math.ceil(x*8/992) for x in s])))
    # uses/px: blocks*2048 coded bits / bps / 50176
    d["uses_px_mean_blocks"]=float(nbv.mean()*2048/bps/50176)
    if cap: d["uses_px_fixed_cap"]=float((cap*8/992) if False else math.ceil(cap*8/992)*2048/bps/50176)
    if cap:
        mism=[n for n,v in zip(names,s) if (v>cap)!=(n in capimgs)]
        d["capfail_set_mismatch"]=len(mism)
    res.append(d); print(json.dumps(d,indent=1))
json.dump(res,open("__AQUASC_ROOT__/verification/s5/s5_counts.json","w"),indent=1)
