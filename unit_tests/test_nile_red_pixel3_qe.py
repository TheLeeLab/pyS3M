"""Tests for NileRedFunctions.pixelate_relative_pixel3_qe(): the model-free pixelated relative pixel 3 QE."""

import unittest

import numpy as np
import pandas as pd

import pyS3M.NileRedFunctions as NileRedFunctions

CAMERA_PX_NM = 69.0


def _locs(x_nm, y_nm, A_R, A_G, A_B, pe):
    return pd.DataFrame({"xc": np.asarray(x_nm, float) / CAMERA_PX_NM, "yc": np.asarray(y_nm, float) / CAMERA_PX_NM,
                         "A_R": A_R, "A_G": A_G, "A_B": A_B, "photoelectrons": pe})


class TestPixelateRelativePixel3QE(unittest.TestCase):
    def setUp(self):
        self.nr = NileRedFunctions.NileRed_Functions()

    def test_photoelectron_weighted_share(self):
        # One cell, two localisations: f_R is the share of the summed photoelectrons on pixel 3.
        df = _locs([10, 20], [10, 20], A_R=[0.6, 0.8], A_G=[0.35, 0.15], A_B=[0.05, 0.05], pe=[100, 300])
        out, info = self.nr.pixelate_relative_pixel3_qe(df, pixel_size_nm=50, camera_pixel_size=CAMERA_PX_NM,
                                                        min_localisations=2)
        expected = (100 * 0.6 + 300 * 0.8) / (100 * 1.0 + 300 * 1.0)
        self.assertEqual(info["grid_shape"], (1, 1))
        self.assertAlmostEqual(info["pixel3_grid"][0, 0], expected)
        np.testing.assert_allclose(out["relative_pixel3_qe"], expected)
        self.assertEqual(info["n_pixels"], 1)
        self.assertEqual(info["n_pixels_skipped"], 0)

    def test_unnormalised_fractions(self):
        # The share uses A_R / (A_R + A_G + A_B), so fractions that don't sum to 1 still give the share.
        df = _locs([10, 12, 14], [10, 10, 10], A_R=[1.2] * 3, A_G=[0.6] * 3, A_B=[0.2] * 3, pe=[50, 50, 50])
        _, info = self.nr.pixelate_relative_pixel3_qe(df, pixel_size_nm=50, camera_pixel_size=CAMERA_PX_NM)
        self.assertAlmostEqual(info["pixel3_grid"][0, 0], 0.6)

    def test_two_regions_and_min_localisations(self):
        rng = np.random.default_rng(0)
        n = 400
        x = np.r_[rng.uniform(0, 400, n), rng.uniform(600, 1000, n)]
        y = rng.uniform(0, 400, 2 * n)
        A_R = np.r_[np.full(n, 0.62), np.full(n, 0.72)] + rng.normal(0, 0.02, 2 * n)
        A_B = np.full(2 * n, 0.03)
        df = _locs(x, y, A_R, 1 - A_R - A_B, A_B, rng.uniform(200, 2000, 2 * n))
        out, info = self.nr.pixelate_relative_pixel3_qe(df, pixel_size_nm=100, camera_pixel_size=CAMERA_PX_NM,
                                                        min_localisations=3)
        grid = info["pixel3_grid"]
        left, right = grid[:, :4], grid[:, 6:]
        self.assertAlmostEqual(np.nanmedian(left), 0.62, delta=0.01)
        self.assertAlmostEqual(np.nanmedian(right), 0.72, delta=0.01)
        self.assertTrue(np.isnan(grid[:, 4:6]).all())   # the empty gap between the regions
        self.assertEqual(len(out), len(df))
        # a cell needs min_localisations: with a very high minimum nothing has a value
        _, info_hi = self.nr.pixelate_relative_pixel3_qe(df, pixel_size_nm=100, camera_pixel_size=CAMERA_PX_NM,
                                                         min_localisations=10_000)
        self.assertEqual(info_hi["n_pixels"], 0)
        self.assertEqual(info_hi["n_pixels_skipped"], int((info["n_locs_grid"] > 0).sum()))

    def test_unusable_rows_left_out(self):
        df = _locs([10, 12, 14, 16], [10] * 4, A_R=[0.6, 0.6, np.nan, 0.9], A_G=[0.35, 0.35, 0.3, 0.05],
                   A_B=[0.05] * 4, pe=[100, 100, 100, 0])
        _, info = self.nr.pixelate_relative_pixel3_qe(df, pixel_size_nm=50, camera_pixel_size=CAMERA_PX_NM,
                                                      min_localisations=2)
        self.assertEqual(info["n_locs_grid"][0, 0], 2)
        self.assertAlmostEqual(info["pixel3_grid"][0, 0], 0.6)


if __name__ == "__main__":
    unittest.main()
