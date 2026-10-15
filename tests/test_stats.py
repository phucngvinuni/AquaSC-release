import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repo root (core modules)
# tests/test_stats.py — stdlib unittest suite for stats.py (Wave 8 STATS).
# Run: python tests/test_stats.py   (works in the CPU venv; numpy+scipy only)

import unittest
import numpy as np

import stats


class TestTConfidenceInterval(unittest.TestCase):
    def test_known_mean(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0.5, 0.1, 400)
        r = stats.t_confidence_interval(x)
        self.assertEqual(r["n"], 400)
        self.assertAlmostEqual(r["mean"], float(x.mean()), places=6)
        # t-crit ~ 1.966 at n=400
        self.assertAlmostEqual(r["t_crit"], 1.9659, places=2)
        self.assertLess(r["ci_low"], r["mean"] < r["ci_high"])

    def test_excludes_zero_when_positive(self):
        rng = np.random.default_rng(2)
        x = 0.3 + rng.normal(0, 0.001, 200)
        self.assertGreater(stats.t_confidence_interval(x)["ci_low"], 0.0)

    def test_nan_dropped(self):
        x = np.array([0.1, 0.2, np.nan, 0.3, 0.4])
        r = stats.t_confidence_interval(x)
        self.assertEqual(r["n"], 4)

    def test_n_lt_2(self):
        r = stats.t_confidence_interval([0.5])
        self.assertTrue(np.isnan(r["half_width"]))
        self.assertEqual(r["mean"], 0.5)


class TestPairedBootstrap(unittest.TestCase):
    def test_positive_shift(self):
        rng = np.random.default_rng(3)
        base = rng.random(432)
        a = base + 0.03 + rng.normal(0, 0.005, 432)
        b = base + rng.normal(0, 0.005, 432)
        r = stats.paired_bootstrap(a, b, n_boot=10000, seed=42)
        self.assertGreater(r["ci_low"], 0.0)
        self.assertLess(r["p_value"], 0.001)

    def test_zero_shift_contains_zero(self):
        rng = np.random.default_rng(4)
        a = rng.random(300)
        b = a.copy()
        r = stats.paired_bootstrap(a, b, n_boot=10000, seed=42)
        self.assertLessEqual(r["ci_low"], 0.0)
        self.assertGreaterEqual(r["ci_high"], 0.0)
        self.assertAlmostEqual(r["mean_diff"], 0.0, places=10)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            stats.paired_bootstrap([1, 2, 3], [1, 2])

    def test_reproducible(self):
        rng = np.random.default_rng(5)
        a = rng.random(200)
        b = rng.random(200)
        r1 = stats.paired_bootstrap(a, b, seed=7)["ci_low"]
        r2 = stats.paired_bootstrap(a, b, seed=7)["ci_low"]
        self.assertEqual(r1, r2)


class TestWilcoxon(unittest.TestCase):
    def test_positive_shift(self):
        rng = np.random.default_rng(6)
        base = rng.random(432)
        a = base + 0.05 + rng.normal(0, 0.01, 432)
        b = base + rng.normal(0, 0.01, 432)
        w = stats.wilcoxon_signed_rank(a, b)
        self.assertLess(w["p_value"], 0.001)
        self.assertEqual(w["n"], 432)

    def test_all_zero_diff(self):
        w = stats.wilcoxon_signed_rank([0.1] * 50, [0.1] * 50)
        self.assertEqual(w["n_nonzero"], 0)
        self.assertTrue(np.isnan(w["p_value"]))

    def test_ties_reported(self):
        rng = np.random.default_rng(7)
        a = rng.random(100)
        b = a.copy()
        b[30:] = a[30:] + 0.01  # 70 differing, 30 exact ties
        w = stats.wilcoxon_signed_rank(a, b)
        self.assertEqual(w["n_zero"], 30)


class TestHolmBonferroni(unittest.TestCase):
    def test_family(self):
        pvals = [1e-6, 0.03, 0.5, 0.9]
        hb = stats.holm_bonferroni(pvals, alpha=0.05)
        self.assertTrue(hb[0]["reject"])
        # sorted [1e-6, 0.03, 0.5, 0.9] -> adj = [4e-6, 0.09, 1.0, 1.0]
        # rank2 adj = 0.09 > 0.05 -> reject False
        self.assertFalse(hb[1]["reject"])
        self.assertEqual(hb[0]["rank"], 1)
        self.assertAlmostEqual(hb[0]["alpha_i"], 0.05 / 4)

    def test_reject_all_small(self):
        hb = stats.holm_bonferroni([0.001, 0.001, 0.001])
        self.assertTrue(all(r["reject"] for r in hb))


class TestBinomialSE(unittest.TestCase):
    def test_plan_values(self):
        r = stats.binomial_se_and_ci(0.2, 432)
        self.assertAlmostEqual(r["se"], 0.019245, places=4)
        self.assertAlmostEqual(r["half_width"], 0.037720, places=4)

    def test_plan_submitted_split(self):
        r = stats.binomial_se_and_ci(0.90, 494)
        self.assertAlmostEqual(r["se"], 0.01350, places=4)
        self.assertAlmostEqual(r["half_width"], 0.02646, places=4)

    def test_boundary(self):
        self.assertEqual(stats.binomial_se_and_ci(1.0, 100)["se"], 0.0)


class TestPerImageAP(unittest.TestCase):
    def test_perfect(self):
        ap = stats.per_image_ap50(
            np.array([[0, 0, 10, 10]]), np.array([0.9]),
            np.array([[0, 0, 10, 10]]))
        self.assertEqual(ap, 1.0)

    def test_no_gt(self):
        self.assertEqual(stats.per_image_ap50(np.array([[0, 0, 10, 10]]),
                                              np.array([0.9]), np.empty((0, 4))), 0.0)

    def test_no_pred(self):
        self.assertEqual(stats.per_image_ap50(np.empty((0, 4)), np.array([]),
                                              np.array([[0, 0, 10, 10]])), 0.0)

    def test_false_positive_hurts(self):
        ap = stats.per_image_ap50(
            np.array([[0, 0, 10, 10], [50, 50, 60, 60]]),
            np.array([0.9, 0.99]),
            np.array([[0, 0, 10, 10]]))
        self.assertLess(ap, 1.0)
        self.assertGreater(ap, 0.0)


class TestBootstrapCI(unittest.TestCase):
    def test_contains_mean(self):
        rng = np.random.default_rng(8)
        x = rng.normal(0.5, 0.1, 500)
        r = stats.bootstrap_ci(x, n_boot=2000, seed=42)
        self.assertLess(r["ci_low"], r["mean"] < r["ci_high"])
        self.assertGreater(r["ci_high"] - r["ci_low"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
