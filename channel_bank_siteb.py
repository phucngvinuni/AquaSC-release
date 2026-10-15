# channel_bank_siteb.py — Site B's own seeded paired channel bank (Wave 14
# dispatch, S-0: "Site B needs its own clean ceiling, its own prior floor, and
# its own seeded channel bank over the 455 video-disjoint test images, with the
# SHA recorded. Site A's 0.7743 and 0.2711 must not appear anywhere in a Site B
# table.").
#
# This is a deliberate near-duplicate of channel_bank.py, NOT a parameterized
# reuse of it: the two banks must use disjoint subseed spaces, or a Site B
# image at index i would silently draw the identical (h, noise) as Site A's
# image at the same index i, secretly correlating two sites' "independent"
# channel realizations. Site A's base is 42,000,000 (image indices 0..431,
# max derived seed ~47,009,431); this bank's base (55,000,000) is chosen with
# a wide, deliberately non-adjacent margin above that range.
#
# Bank contents (channel_bank_siteb.npz), same schema as channel_bank.py:
#   h, n, subseed, image_names, snr_db, master_seed, M, L_real, n_images
#
# Derivation (documented, deterministic, disjoint from Site A):
#   subseed(i, snr, m) = 55_000_000 + s_idx*1_000_000 + m*1_000 + i
#     i      = Site B test-image index 0..454 (sorted filename order,
#              yolo_fishdet_nb2_vd/test/images)
#     s_idx  = SNR index 0..5  (snr_db order below, same grid as Site A)
#     m      = realization index 0..9
#   rng = np.random.default_rng(subseed); h = (rng.standard_normal()+1j*rng.standard_normal())/sqrt(2)
#   n   = rng.standard_normal(L)
#
# Rule 11 compliance: this module's own output filename (channel_bank_siteb.npz)
# already disambiguates it from channel_bank.py's Site A output; there is only
# one site per file, so no further per-run parameterization is needed here.

import os

import numpy as np

MASTER_SEED = 42
SNR_DB = (-5, 0, 5, 10, 15, 20)
N_REALIZATIONS = 10
N_IMAGES = 455
L_REAL = 12544

_SUBSEED_BASE = 55_000_000
_SNR_STRIDE = 1_000_000
_M_STRIDE = 1_000


def snr_index(snr_db: float) -> int:
    try:
        return SNR_DB.index(float(snr_db))
    except ValueError:
        raise ValueError(f"snr_db={snr_db} not in bank SNR grid {SNR_DB}")


def subseed(image_idx: int, snr_db: float, m: int) -> int:
    return _SUBSEED_BASE + snr_index(snr_db) * _SNR_STRIDE + int(m) * _M_STRIDE + int(image_idx)


def subseed_base(snr_db: float, m: int) -> int:
    return _SUBSEED_BASE + snr_index(snr_db) * _SNR_STRIDE + int(m) * _M_STRIDE


def build_channel_bank(out_path: str, n_images: int = N_IMAGES,
                       snr_db: tuple = SNR_DB, M: int = N_REALIZATIONS,
                       L: int = L_REAL, image_names: list = None) -> dict:
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
        site=np.asarray(["B"]),
    )
    meta = dict(path=out_path, n_images=n_images, n_snr=n_snr, M=M, L_real=L,
                master_seed=MASTER_SEED, subseed_base=_SUBSEED_BASE)
    return meta


class ChannelBankSiteB:
    """Reader for channel_bank_siteb.npz -- same accessor contract as
    channel_bank.ChannelBank, kept as a separate class (not a shared base) so
    a caller can never accidentally instantiate the wrong site's bank by
    passing the wrong path to a generic class."""

    def __init__(self, path: str, mmap: bool = True):
        self.path = path
        self._z = np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)
        assert str(self._z["site"][0]) == "B", f"{path} is not a Site B bank"
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
        return np.asarray(self.n[image_idx, self._sidx(snr_db), m, :length], dtype=float)

    def get_subseed(self, image_idx: int, snr_db: float, m: int) -> int:
        return int(self._z["subseed"][image_idx, self._sidx(snr_db), m])


def default_bank_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "channel_bank_siteb.npz")


def _self_test() -> None:
    import tempfile
    tmp = os.path.join(tempfile.gettempdir(), "_bank_siteb_selftest.npz")
    build_channel_bank(tmp, n_images=8, M=2, L=64)
    bank = ChannelBankSiteB(tmp)
    for i in range(8):
        for s in SNR_DB:
            for m in range(2):
                ss = bank.get_subseed(i, s, m)
                rng = np.random.default_rng(ss)
                hr = rng.standard_normal(); hi = rng.standard_normal()
                h_expected = (hr + 1j * hi) / np.sqrt(2)
                assert abs(bank.get_h(i, s, m) - h_expected) < 1e-6
                rng2 = np.random.default_rng(ss)
                rng2.standard_normal(); rng2.standard_normal()
                u = rng2.standard_normal(64)
                assert np.allclose(bank.get_noise_real(i, s, m, 64), u)
    for s in SNR_DB:
        for m in range(2):
            assert subseed_base(s, m) + 3 == subseed(3, s, m)
    for i in (0, 4, 7):
        for s in SNR_DB:
            for m in (0, 1):
                assert bank.get_subseed(i, s, m) == subseed(i, s, m)
    # disjointness check against Site A's subseed space
    import channel_bank as bank_a
    for i in range(8):
        for s in SNR_DB:
            for m in range(2):
                assert subseed(i, s, m) != bank_a.subseed(i, s, m), \
                    "Site A/B subseed collision -- bases are not disjoint"
    bank.close()
    os.remove(tmp)
    print("channel_bank_siteb.py self-test PASSED (including Site A disjointness)")


if __name__ == "__main__":
    _self_test()
