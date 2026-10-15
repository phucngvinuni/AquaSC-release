# Reassembly trace for point (c) (QPSK q=1 cap372) at SNR 10 dB, image frame_000000_19.png (idx 0), m=7 (decodes) and m=3 (fails)
import io, os, sys, json, struct
import numpy as np
from PIL import Image
sys.path.insert(0,"__AQUASC_ROOT__")
import digital_pipeline as dp
D="__AQUASC_DATA__/yolo_fish_dataset_root/test/images"
name="frame_000000_19.png"; idx=0; snr=10
im=Image.open(os.path.join(D,name)).convert("RGB")
if im.size!=(224,224): im=im.resize((224,224),Image.BICUBIC)
b=io.BytesIO(); im.save(b,"WEBP",quality=1,method=4); wb=b.getvalue()
bits=np.unpackbits(np.frombuffer(wb,dtype=np.uint8))
print("file bytes:",len(wb),"first 12 bytes:",wb[:12],"RIFF size field:",struct.unpack("<I",wb[4:8])[0],"(=file_len-8:",len(wb)-8,")")
print("source bits:",bits.size,"payload/block:",992,"n_blocks:",int(np.ceil(bits.size/992)),"pad bits in last block:",int(np.ceil(bits.size/992))*992-bits.size)
def psnr(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float);mse=((a-b)**2).mean();return 99.0 if mse==0 else 10*np.log10(255**2/mse)
orig=im
log={}
for m in (7,3):
    seed=1000*idx+10*int(snr+5)+m
    out,st=dp.simulate_digital_pipeline(bits,snr_db=float(snr),seed=seed,bits_per_symbol=2)
    erased=(out<0)
    blk_fail=[bool(erased[i*992:(i+1)*992].any()) for i in range(st["blocks"])]
    filled=np.where(out<0,0,out).astype(np.uint8)
    rb=np.packbits(filled).tobytes()[:len(wb)]
    dec=True
    try:
        r=Image.open(io.BytesIO(rb)); r.load(); r=r.convert("RGB"); p=psnr(orig,r)
    except Exception as e:
        dec=False; p=psnr(orig,Image.new("RGB",(224,224),(0,0,0))); err=repr(e)
    # manifest consistency
    d=dict(m=m,seed=seed,stats=st,blocks_failed_flags=blk_fail,erased_bits=int(erased.sum()),
      zero_filled_bits=int(erased.sum()),truncated_len=len(rb),returned_bits=int(out.size),
      bytes_differ_from_source=int(sum(1 for x,y in zip(rb,wb) if x!=y)),
      riff_size_field_after=struct.unpack("<I",rb[4:8])[0],pil_decodes=dec,psnr_dB=p,black_fallback=(not dec))
    if not dec: d["pil_error"]=err
    log[f"m={m}"]=d
    print(json.dumps(d,indent=1,default=str))
# Oracle-length test: does PIL decode if the stream is NOT truncated to the true length (e.g. extra zero tail)? and with corrupted RIFF size field
t=wb+bytes(64)
try: Image.open(io.BytesIO(t)).load(); log["extra_zero_tail_64B_decodes"]=True
except Exception as e: log["extra_zero_tail_64B_decodes"]=repr(e)
c=bytearray(wb); c[4:8]=b"\x00\x00\x00\x00"
try: Image.open(io.BytesIO(bytes(c))).load(); log["riff_size_zeroed_decodes"]=True
except Exception as e: log["riff_size_zeroed_decodes"]=repr(e)
c=bytearray(wb); c[0:12]=bytes(12)
try: Image.open(io.BytesIO(bytes(c))).load(); log["first12B_zeroed_decodes"]=True
except Exception as e: log["first12B_zeroed_decodes"]=repr(e)
c=bytearray(wb); c[4:8]=struct.pack("<I",len(wb)+10000)
try: Image.open(io.BytesIO(bytes(c))).load(); log["riff_size_oversized_decodes"]=True
except Exception as e: log["riff_size_oversized_decodes"]=repr(e)
print({k:v for k,v in log.items() if not k.startswith("m=")})
json.dump(log,open("__AQUASC_ROOT__/verification/s5/s5_trace.json","w"),indent=1,default=str)
