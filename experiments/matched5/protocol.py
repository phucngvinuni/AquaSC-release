"""Shared deterministic channel draws; no torch or TensorFlow dependency."""
import numpy as np

SYMBOLS = 6144
SNR_DB = 5.0


def channel_draw(split, image_index, realization):
    # Test exactly reproduces channel_bank.py at snr_idx=2 (5 dB).
    # Validation uses a disjoint seed family. Float32 casting matches bank.
    base = 44_000_000 if split == "test" else 914_000_000
    seed = base + 1000 * int(realization) + int(image_index)
    rng = np.random.default_rng(seed)
    h = np.complex64((rng.standard_normal()+1j*rng.standard_normal())/np.sqrt(2))
    raw = rng.standard_normal(2*SYMBOLS).astype(np.float32)
    return h, raw[:SYMBOLS]+1j*raw[SYMBOLS:]
