"""Numerical identities, aggregation units and actual frozen inference interface."""
import unittest

from run import (CONFIG, OLD, RECORDS, phase3, torch, np, load_head, predict_bias,
                 decompose, shift_invariance, score_window, read_csv, compare_rows, configure)
from analyze import average, paired


class FollowupTests(unittest.TestCase):
    def test_inhomogeneous_reconstruction_and_cross_term(self):
        generator = torch.Generator().manual_seed(42)
        bias = torch.randn(3, 20, 2, 5, 6, dtype=torch.double, generator=generator)
        spec = dict(kind="A2_scaled", coef=torch.tensor([.9, .8], dtype=torch.double))
        inherited, accumulated, innovation = decompose(bias, spec)
        torch.testing.assert_close(bias, inherited + accumulated, atol=1e-12, rtol=0)
        self.assertEqual(innovation.shape[1], 19)
        cross = 2 * inherited * accumulated
        torch.testing.assert_close(bias.square(), inherited.square() + accumulated.square() + cross,
                                   atol=1e-12, rtol=0)
        self.assertGreater(float(cross.abs().max()), 0.1)
        zero = torch.zeros_like(bias)
        first, remainder, innovations = decompose(zero, spec)
        self.assertEqual(int(torch.count_nonzero(first + remainder + zero)), 0)
        self.assertEqual(int(torch.count_nonzero(innovations)), 0)

    def test_shift_all_member_differences_and_covariance(self):
        generator = torch.Generator().manual_seed(73)
        paths = torch.randn(32, 2, 20, 32, 64, generator=generator) * .05
        shift = torch.randn(2, 20, 32, 64, generator=generator) * .003
        result = shift_invariance(paths, paths - shift)
        self.assertLess(result["all_feature_covariance_error_bound"], CONFIG["tolerances"]["covariance"])

    def test_statistical_unit_is_trajectory(self):
        rows = []
        for trajectory, native, corrected in [("a", 2., 1.), ("b", 10., 12.)]:
            for start in range(20):
                rows.append(dict(trajectory=trajectory, method="native", crps=native))
                for seed in [42, 43, 44]:
                    rows.append(dict(trajectory=trajectory, method="crps", crps=corrected))
        result = paired(rows, [[]], ["crps"])[0]
        self.assertEqual(result["trajectories"], 2)
        self.assertEqual(result["improved"], 1)
        self.assertEqual(result["mean_delta"], .5)
        self.assertEqual(result["native"], 6.)
        self.assertEqual(result["corrected"], 6.5)

    def test_full_crps_value_and_gradient(self):
        phase3.tests()

    @unittest.skipUnless(torch.cuda.is_available(), "real frozen-head interface requires CUDA")
    @torch.no_grad()
    def test_real_extension_interface_reproduces_old_scores(self):
        configure()
        data = phase3.load("test")
        fields = phase3.aux("test")
        for seed in CONFIG["head_seeds"]:
            head, scale = load_head("crps", seed)
            rows = []
            for index, start in enumerate([400, 420]):
                count = len(RECORDS["test"])
                paths = data.p[index * count:(index + 1) * count].permute(0, 2, 3, 1, 4, 5)
                bias = predict_bias(fields["history"][:, index], paths, fields["normalization"], head, scale)
                current, _ = score_window(paths - bias[:, None], fields["truth"][:, index],
                                          fields["normalization"], start, seed, "crps")
                rows.extend(current)
            expected = [row for row in read_csv(OLD / "scores.csv")
                        if row["split"] == "test" and row["method"] == "crps" and row["seed"] == str(seed)]
            compare_rows(rows, expected, ["trajectory", "start", "seed", "method"],
                         ["crps", "mse", "energy", "brier"], "actual extension inference on old windows")


if __name__ == "__main__":
    configure()
    unittest.main(verbosity=2)
