"""Index-noise fine-tune helpers (plan 5.4): map SNR -> measured digital error rate.

The digital chain's corruption of the index stream is dominated by LDPC block
erasure (BLER); surviving blocks are bit-exact (per-block CRC-32). The measured
per-index-bit error rate is therefore approximated by the BLER curve from
`results/eval/blers_vs_snr.csv` (the CSV's `ber` column is the residual BER over *surviving*
blocks and is ~0, so it is not the right quantity for the flip probability).
We flip each index bit with probability BLER(snr), linearly interpolated over
the CSV rows and clamped to the measured SNR range. Documented deviation from
the task brief's literal "measured BER" (the CSV's ber column is degenerate).
"""
from __future__ import annotations

import csv
import os

import numpy as np

_DEFAULT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/eval/blers_vs_snr.csv")


def load_bler_curve(csv_path=None):
    csv_path = csv_path or _DEFAULT_CSV
    snrs = []
    blers = []
    with open(csv_path, "r", newline="") as f:
        for row in csv.DictReader(f):
            snr = row.get("snr_db")
            bler = row.get("bler")
            if snr is None or bler is None:
                continue
            try:
                snr_f = float(snr)
                bler_f = float(bler)
            except (TypeError, ValueError):
                # skip the second header / comment rows in results/eval/blers_vs_snr.csv
                continue
            snrs.append(snr_f)
            blers.append(bler_f)
    snrs = np.asarray(snrs, dtype=np.float64)
    blers = np.asarray(blers, dtype=np.float64)
    # deduplicate (results/eval/blers_vs_snr.csv repeats 0/10/20 dB for the pilot sweep);
    # keep the first occurrence (the perfect-CSI sweep).
    if snrs.size:
        _, keep = np.unique(snrs, return_index=True)
        snrs = snrs[np.sort(keep)]
        blers = blers[np.sort(keep)]
    order = np.argsort(snrs)
    return snrs[order], blers[order]


class BlerCurve:
    def __init__(self, csv_path=None):
        self.snrs, self.blers = load_bler_curve(csv_path)
        self.lo, self.hi = float(self.snrs[0]), float(self.snrs[-1])

    def __call__(self, snr_db):
        s = float(min(max(snr_db, self.lo), self.hi))
        return float(np.interp(s, self.snrs, self.blers))
