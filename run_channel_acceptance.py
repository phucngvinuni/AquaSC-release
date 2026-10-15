"""Wave 4 CHANNEL acceptance: BLER vs outage sweep, pilot-CSI gap, one-image demo.

Writes blers_vs_snr.csv (perfect + pilot CSI) and prints demo results.
"""
import sys, time
sys.path.insert(0, "__AQUASC_ROOT__")
import digital_pipeline as dp

CSV = "__AQUASC_ROOT__/blers_vs_snr.csv"
SNRS = [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0]
BLOCKS = 500

t0 = time.time()
print(f"=== acceptance sweep: rate 1/2, 16-QAM, {BLOCKS} blocks/point, perfect CSI ===")
rows = dp.run_bler_sweep(SNRS, blocks=BLOCKS, rate=0.5, csi_mode="perfect",
                         seed=42, csv_path=CSV, log=True)
print("=== pilot-CSI sweep (10% pilots, LS): 0/10/20 dB ===")
pilot_rows = dp.run_bler_sweep([0.0, 10.0, 20.0], blocks=BLOCKS, rate=0.5,
                               csi_mode="pilot", seed=42, csv_path=None, log=True)
# append pilot rows to CSV with csi_mode column
with open(CSV, "a") as f:
    f.write("\n# pilot CSI (10% overhead, LS estimate)\n")
    hdr = list(pilot_rows[0].keys())
    import csv as _csv
    w = _csv.DictWriter(f, fieldnames=["csi_mode"] + hdr)
    w.writerow({"csi_mode": "csi_mode"} | {k: k for k in hdr})
    for r in pilot_rows:
        w.writerow({"csi_mode": "pilot"} | r)

print("=== one-image demo (18/2b frame, enumerative mask, pilots, 5/15 dB) ===")
demo = dp.one_image_demo(snr_list=(5.0, 15.0), seed=42)
for r in demo:
    print(f"  snr={r['snr_db']:+.0f}dB header_ok={r['header_ok_rate']:.2f} "
          f"frame_lost={r['frame_lost_rate']:.2f} crc_ok={r['payload_crc_ok_rate']:.2f} "
          f"mean_index_ber={r['mean_index_ber']:.5f} "
          f"mean_erased={r['mean_erased_tokens']:.1f} "
          f"emitted_bits={r['emitted_bits_per_frame']} bpp={r['bpp']:.5f}")
print(f"done in {time.time()-t0:.1f}s")
