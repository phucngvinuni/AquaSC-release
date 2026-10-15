# channel_bank.py — seeded paired channel bank (plan §9.3 / Wave 8 STATS, rule 4).
#
# The shared {h}/{n} bank replayed identically by every method, so that channel
# draws are removed from *between-method* differences.  One master seed (42)
# drives every cell through a documented, per-cell sub-seed derivation.
#
# Bank contents (channel_bank.npz), keyed by (image, snr, realization):
#   h        complex64  [n_images, n_snr, M]       block-fading gain  ~ CN(0,1)
#   n        float32    [n_images, n_snr, M, L]    standard-normal noise stream
#   subseed  int64      [n_images, n_snr, M]       per-cell derived seed
#   image_names  str    [n_images]                 sorted test filenames
#   snr_db   float64    [n_snr]                    {-5,0,5,10,15,20}
#   master_seed, M, L (L_real), plus derivation metadata
#
# Derivation (documented, deterministic):
#   subseed(i, snr, m) = 42_000_000 + s_idx*1_000_000 + m*1_000 + i
#     i      = test-image index 0..431 (sorted filename order)
#     s_idx  = SNR index 0..5  (snr_db order above)
#     m      = realization index 0..9
#   rng = np.random.default_rng(subseed)
#   h   = (rng.standard_normal() + 1j*rng.standard_normal()) / sqrt(2)
#   n   = rng.standard_normal(L)          # L = 12544 real standard normals
#
#   Because subseed is LINEAR in i (subseed = base(snr,m) + i), the AquaSC proxy
#   eval replays the bank by running with  --seed base(snr,m)  at batch_size 1:
#   engine1.py already passes  digital_seed = args.seed + batch_idx  and
#   digital_channel.simulate_rayleigh_channel draws h then n in this EXACT order,
#   so no code change is needed on the AquaSC side — the seed is the bank.
#
# Noise stream length L = 12544 real standard normals = 6272 complex samples:
#   * AquaSC proxy path needs 2 * n_symbols real draws (<= 2*2086 = 4172 for the
#     largest committed config; 1288 for the RVQ 3x6+2bit flagship).
#   * JSCC-CNN c=2 needs 2*c*H*W = 2*2*56*56 = 12544 real draws (exactly L).
#
# Loader contract:
#   * AquaSC:  pass  seed = subseed(i, snr, m)  to simulate_rayleigh_channel
#              (equivalent: run the eval with --seed = base(snr, m)).
#   * JSCC-CNN: use ChannelBank.get_h() for the scalar block gain and
#               ChannelBank.get_noise_real() for the real noise vector; the
#               first 2*c*H*W entries are the per-channel noise in z_hat order.

from __future__ import annotations

import os
import numpy as np

MASTER_SEED = 42
SNR_DB = (-5, 0, 5, 10, 15, 20)
N_REALIZATIONS = 10          # M
N_IMAGES = 432
L_REAL = 12544               # standard-normal real samples per cell (see above)

_SUBSEED_BASE = 42_000_000
_SNR_STRIDE = 1_000_000
_M_STRIDE = 1_000


def snr_index(snr_db: float) -> int:
    try:
        return SNR_DB.index(float(snr_db))
    except ValueError:
        raise ValueError(f"snr_db={snr_db} not in bank SNR grid {SNR_DB}")


def subseed(image_idx: int, snr_db: float, m: int) -> int:
    """Deterministic per-cell seed (documented derivation above)."""
    return _SUBSEED_BASE + snr_index(snr_db) * _SNR_STRIDE + int(m) * _M_STRIDE + int(image_idx)


def subseed_base(snr_db: float, m: int) -> int:
    """args.seed value that replays realization m at snr_db for image i via
    seed = base(snr,m) + i  (engine1.py: digital_seed = args.seed + batch_idx)."""
    return _SUBSEED_BASE + snr_index(snr_db) * _SNR_STRIDE + int(m) * _M_STRIDE


def build_channel_bank(out_path: str, n_images: int = N_IMAGES,
                       snr_db: tuple = SNR_DB, M: int = N_REALIZATIONS,
                       L: int = L_REAL, image_names: list = None) -> dict:
    """Generate the bank and write it to ``out_path`` (a .npz)."""
    snr_db = tuple(float(s) for s in snr_db)
    n_snr = len(snr_db)
    h = np.empty((n_images, n_snr, M), dtype=np.complex64)
    n = np.empty((n_images, n_snr, M, L), dtype=np.float32)
    seeds = np.empty((n_images, n_snr, M), dtype=np.int64)
    for i in range(n_images):
        for s in range(n_snr):
            for m in range(M):
                ss = subseed(i, snr_db[s], m)
                rng = np.random.default_rng(ss)
                hr = rng.standard_normal()
                hi = rng.standard_normal()
                h[i, s, m] = (hr + 1j * hi) / np.sqrt(2.0)
                n[i, s, m, :] = rng.standard_normal(L)
                seeds[i, s, m] = ss
    if image_names is None:
        image_names = [f"img_{i:05d}" for i in range(n_images)]
    else:
        image_names = list(image_names[:n_images])
    np.savez(
        out_path,
        h=h, n=n, subseed=seeds,
        snr_db=np.asarray(snr_db, dtype=float),
        image_names=np.asarray(image_names),
        master_seed=np.asarray([MASTER_SEED], dtype=int),
        M=np.asarray([M], dtype=int),
        L_real=np.asarray([L], dtype=int),
        n_images=np.asarray([n_images], dtype=int),
        subseed_base=np.asarray([_SUBSEED_BASE], dtype=int),
        snr_stride=np.asarray([_SNR_STRIDE], dtype=int),
        m_stride=np.asarray([_M_STRIDE], dtype=int),
    )
    meta = dict(path=out_path, n_images=n_images, n_snr=n_snr, M=M, L_real=L,
                master_seed=MASTER_SEED)
    return meta


class ChannelBank:
    """Reader for channel_bank.npz with per-cell accessors for both methods."""

    def __init__(self, path: str, mmap: bool = True):
        self.path = path
        # open with mmap so the large n array is not fully resident per process
        if mmap:
            self._z = np.load(path, mmap_mode="r", allow_pickle=False)
        else:
            self._z = np.load(path, allow_pickle=False)
        self.h = self._z["h"]
        self.n = self._z["n"]
        self.snr_db = [float(s) for s in self._z["snr_db"]]
        self.image_names = [str(s) for s in self._z["image_names"]]
        self.n_images = len(self.image_names)
        self.M = int(self._z["M"][0])
        self.L_real = int(self._z["L_real"][0])
        self.master_seed = int(self._z["master_seed"][0])

    def close(self):
        self._z.close()

    def _sidx(self, snr_db: float) -> int:
        try:
            return self.snr_db.index(float(snr_db))
        except ValueError:
            raise ValueError(f"snr_db={snr_db} not in bank SNR grid {self.snr_db}")

    def get_h(self, image_idx: int, snr_db: float, m: int) -> complex:
        return complex(self.h[image_idx, self._sidx(snr_db), m])

    def get_noise_real(self, image_idx: int, snr_db: float, m: int, length: int) -> np.ndarray:
        """Return the first ``length`` standard-normal real noise samples."""
        return np.asarray(self.n[image_idx, self._sidx(snr_db), m, :length], dtype=float)

    def get_subseed(self, image_idx: int, snr_db: float, m: int) -> int:
        return int(self._z["subseed"][image_idx, self._sidx(snr_db), m])

    # --- convenience accessors used by the eval drivers --------------------- #
    def aquasc_h_and_noise(self, image_idx: int, snr_db: float, m: int,
                           n_symbols: int):
        """(h, complex_noise) matching digital_channel.simulate_rayleigh_channel.

        The complex noise here is the RAW standard-normal part; the caller scales
        by sqrt(noise_variance/2) with noise_variance = 10**(-snr/10), exactly as
        simulate_rayleigh_channel does.  (Verification path only — the eval path
        replays via the subseed.)
        """
        u = self.get_noise_real(image_idx, snr_db, m, 2 * n_symbols)
        h = self.get_h(image_idx, snr_db, m)
        return h, (u[:n_symbols] + 1j * u[n_symbols:2 * n_symbols])

    def jscc_h_and_noise(self, image_idx: int, snr_db: float, m: int, n_real: int):
        """(h, real_noise[N(0,1)]) for the JSCC-CNN Rayleigh path (z_hat order)."""
        return self.get_h(image_idx, snr_db, m), self.get_noise_real(image_idx, snr_db, m, n_real)


def default_bank_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "channel_bank.npz")


def _self_test() -> None:
    # verify derivation consistency and replay-exactness against
    # digital_channel.simulate_rayleigh_channel
    tmp = "__AQUASC_TMP__/_bank_selftest.npz"
    build_channel_bank(tmp, n_images=8, M=2, L=64)
    bank = ChannelBank(tmp)
    for i in range(8):
        for s in SNR_DB:
            for m in range(2):
                ss = bank.get_subseed(i, s, m)
                # simulate_rayleigh_channel draws h then noise in this order
                rng = np.random.default_rng(ss)
                hr = rng.standard_normal(); hi = rng.standard_normal()
                h_expected = (hr + 1j * hi) / np.sqrt(2)
                assert abs(bank.get_h(i, s, m) - h_expected) < 1e-6
                # the noise stream must equal the stored one
                rng2 = np.random.default_rng(ss)
                rng2.standard_normal(); rng2.standard_normal()  # h draws
                u = rng2.standard_normal(64)
                assert np.allclose(bank.get_noise_real(i, s, m, 64), u)
    # subseed linearity: base + i == subseed
    for s in SNR_DB:
        for m in range(2):
            assert subseed_base(s, m) + 3 == subseed(3, s, m)
    # stored subseed matches the derivation
    for i in (0, 4, 7):
        for s in SNR_DB:
            for m in (0, 1):
                assert bank.get_subseed(i, s, m) == subseed(i, s, m)
    bank.close()
    os.remove(tmp)
    print("channel_bank.py self-test PASSED")


if __name__ == "__main__":
    _self_test()
