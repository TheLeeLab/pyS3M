"""Tests for NileRedFunctions.fit_wavelengths_pixelated().

Creates synthetic localisation data with known spatial wavelength structure,
writes it to a temporary HDF5 file, runs the pixelated fitting pipeline,
and verifies the output grids and DataFrame columns.
"""

import sys
import os
import tempfile
import numpy as np
import pandas as pd
import unittest


import pyS3M.NileRedFunctions as NileRedFunctions
import pyS3M.SpectralFunctions as SpectralFunctions


def _generate_synthetic_localisations(
    n_locs: int = 500,
    wavelength_left: float = 610.0,
    wavelength_right: float = 640.0,
    fov_nm: float = 2000.0,
    camera_pixel_size: float = 69.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate localisations with a left-right wavelength gradient.

    Left half has wavelength_left, right half has wavelength_right.
    Uses the NileRed forward model to produce consistent RGB and PSF widths.

    Returns:
        DataFrame with all columns required by fit_wavelengths_pixelated.
    """
    rng = np.random.default_rng(seed)

    # Positions uniformly distributed across FOV
    x_nm = rng.uniform(0, fov_nm, n_locs)
    y_nm = rng.uniform(0, fov_nm, n_locs)
    xc = x_nm / camera_pixel_size  # camera pixels
    yc = y_nm / camera_pixel_size

    # Assign wavelength based on left/right half
    true_wl = np.where(x_nm < fov_nm / 2, wavelength_left, wavelength_right)

    # Use forward model to generate RGB and PSF widths
    nrf = NileRedFunctions.NileRed_Functions()
    sf = SpectralFunctions.Spectral_Funcs()
    R_qe, G_qe, B_qe, wavelength_array = sf.getpixelefficiency()
    pixel_QYs = np.vstack([B_qe, G_qe, R_qe])

    filter_names = [
        "semrock-ff01-650-200",
        "semrock-di03-r514-t1-25x36",
        "semrock-ff01-515-lp",
    ]
    filter_spectra = sf.get_dye_or_filter_data(
        names=filter_names, wavelength=wavelength_array, dye_or_filter=False
    )

    A_R = np.zeros(n_locs)
    A_G = np.zeros(n_locs)
    A_B = np.zeros(n_locs)
    s_x = np.zeros(n_locs)
    s_y = np.zeros(n_locs)

    NA = 1.49
    for i in range(n_locs):
        preds = nrf.nile_red_forward_model(
            true_wl[i], filter_spectra, wavelength_array, pixel_QYs, NA
        )
        A_R[i] = preds["R"]
        A_G[i] = preds["G"]
        A_B[i] = preds["B"]
        s_x[i] = preds["sigma_x"] / camera_pixel_size  # back to camera pixels
        s_y[i] = preds["sigma_y"] / camera_pixel_size

    # Add realistic noise (5% relative error)
    noise_frac = 0.05
    A_R_err = np.abs(A_R) * noise_frac + 1e-4
    A_G_err = np.abs(A_G) * noise_frac + 1e-4
    A_B_err = np.abs(A_B) * noise_frac + 1e-4
    s_x_err = np.abs(s_x) * noise_frac + 1e-4
    s_y_err = np.abs(s_y) * noise_frac + 1e-4

    # Add noise to measurements
    A_R += rng.normal(0, A_R_err)
    A_G += rng.normal(0, A_G_err)
    A_B += rng.normal(0, A_B_err)
    s_x += rng.normal(0, s_x_err)
    s_y += rng.normal(0, s_y_err)

    photons = rng.uniform(500, 2000, n_locs)
    background_photons = rng.uniform(20, 80, n_locs)

    df = pd.DataFrame({
        "xc": xc,
        "yc": yc,
        "A_R": A_R,
        "A_G": A_G,
        "A_B": A_B,
        "s_x": s_x,
        "s_y": s_y,
        "A_R_err": A_R_err,
        "A_G_err": A_G_err,
        "A_B_err": A_B_err,
        "s_x_err": s_x_err,
        "s_y_err": s_y_err,
        "photoelectrons": photons,
        "background_photoelectrons": background_photons,
        "true_wl": true_wl,
    })

    return df


class TestPixelatedFitting(unittest.TestCase):
    """Test fit_wavelengths_pixelated on synthetic data."""

    @classmethod
    def setUpClass(cls):
        """Generate synthetic data and write to temp HDF5."""
        cls.camera_pixel_size = 69.0
        cls.wavelength_left = 610.0
        cls.wavelength_right = 640.0
        cls.fov_nm = 2000.0

        cls.df_orig = _generate_synthetic_localisations(
            n_locs=500,
            wavelength_left=cls.wavelength_left,
            wavelength_right=cls.wavelength_right,
            fov_nm=cls.fov_nm,
            camera_pixel_size=cls.camera_pixel_size,
        )

        # Write to temp file
        cls.tmpdir = tempfile.mkdtemp()
        cls.h5_path = os.path.join(cls.tmpdir, "test_locs.h5")
        cls.df_orig.to_hdf(cls.h5_path, key="data", mode="w", format="table")

        # Setup optical system
        sf = SpectralFunctions.Spectral_Funcs()
        R_qe, G_qe, B_qe, wavelength_array = sf.getpixelefficiency()
        cls.camera_params = {
            "pixel_QYs": np.vstack([B_qe, G_qe, R_qe]),
            "wavelength": wavelength_array,
        }

        # nile_red_forward_model (used above to generate the synthetic RGB/PSF data)
        # takes wavelength_left/right directly as the location parameter, but
        # fit_wavelengths_pixelated's wl_pixel reports the raw spectrum's *centre of
        # mass* at whatever location parameter the fit recovers (see
        # fit_nile_red_wavelength/spectral_centre_of_mass) -- the two differ by tens
        # of nm, so the recovered wl_pixel must be compared against the centre of
        # mass of the true location parameter, not the location parameter itself.
        nrf = NileRedFunctions.NileRed_Functions()
        cls.wavelength_left_com = nrf.spectral_centre_of_mass(cls.wavelength_left, wavelength_array)
        cls.wavelength_right_com = nrf.spectral_centre_of_mass(cls.wavelength_right, wavelength_array)
        cls.filter_names = [
            "semrock-ff01-650-200",
            "semrock-di03-r514-t1-25x36",
            "semrock-ff01-515-lp",
        ]

    def test_basic_output_structure(self):
        """Verify return types, DataFrame columns, and grid_info keys."""
        nrf = NileRedFunctions.NileRed_Functions()
        result = nrf.fit_wavelengths_pixelated(
            h5_path=self.h5_path,
            filter_names=self.filter_names,
            camera_parameters=self.camera_params,
            pixel_size_nm=200.0,  # large pixels for speed
            min_localisations=3,
            camera_pixel_size=self.camera_pixel_size,
            verbose=False,
        )

        # Should return (DataFrame, dict) when return_grid=True (default)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

        df, grid_info = result
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(grid_info, dict)

        # Check DataFrame columns
        for col in ["wl_pixel", "wl_pixel_err", "pixel_ix", "pixel_iy"]:
            self.assertIn(col, df.columns, f"Missing column: {col}")

        # Check grid_info keys
        expected_keys = [
            "wl_grid", "wl_err_grid", "n_locs_grid", "total_photoelectrons_grid",
            "mean_photoelectrons_grid", "pixel_size_nm", "origin_nm",
            "grid_shape", "n_pixels_fitted", "n_pixels_skipped",
        ]
        for key in expected_keys:
            self.assertIn(key, grid_info, f"Missing grid_info key: {key}")

        # Grid shape consistency
        ny, nx = grid_info["grid_shape"]
        self.assertEqual(grid_info["wl_grid"].shape, (ny, nx))
        self.assertEqual(grid_info["n_locs_grid"].shape, (ny, nx))

    def test_return_grid_false(self):
        """Verify return_grid=False returns only DataFrame."""
        nrf = NileRedFunctions.NileRed_Functions()
        result = nrf.fit_wavelengths_pixelated(
            h5_path=self.h5_path,
            filter_names=self.filter_names,
            camera_parameters=self.camera_params,
            pixel_size_nm=200.0,
            min_localisations=3,
            camera_pixel_size=self.camera_pixel_size,
            verbose=False,
            return_grid=False,
        )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("wl_pixel", result.columns)

    def test_wavelength_gradient_recovery(self):
        """Verify left/right wavelength gradient is recovered."""
        nrf = NileRedFunctions.NileRed_Functions()
        df, grid_info = nrf.fit_wavelengths_pixelated(
            h5_path=self.h5_path,
            filter_names=self.filter_names,
            camera_parameters=self.camera_params,
            pixel_size_nm=200.0,
            min_localisations=3,
            camera_pixel_size=self.camera_pixel_size,
            verbose=False,
        )

        # Separate localisations by left/right half
        x_nm = df["xc"].to_numpy() * self.camera_pixel_size
        left_mask = x_nm < self.fov_nm / 2
        right_mask = ~left_mask

        wl_left = df.loc[left_mask, "wl_pixel"].dropna()
        wl_right = df.loc[right_mask, "wl_pixel"].dropna()

        # Both sides should have fits
        self.assertGreater(len(wl_left), 0, "No left-side fits")
        self.assertGreater(len(wl_right), 0, "No right-side fits")

        mean_left = wl_left.mean()
        mean_right = wl_right.mean()

        # Left should be shorter wavelength, right should be longer
        self.assertLess(mean_left, mean_right,
                        f"Left ({mean_left:.1f}) should be < Right ({mean_right:.1f})")

        # Check absolute accuracy: within 10 nm of the true location parameter's
        # spectral centre of mass (what wl_pixel actually estimates -- see setUpClass).
        tolerance = 10.0
        self.assertAlmostEqual(
            mean_left, self.wavelength_left_com, delta=tolerance,
            msg=f"Left mean {mean_left:.1f} nm, expected ~{self.wavelength_left_com:.1f} nm"
        )
        self.assertAlmostEqual(
            mean_right, self.wavelength_right_com, delta=tolerance,
            msg=f"Right mean {mean_right:.1f} nm, expected ~{self.wavelength_right_com:.1f} nm"
        )

        print(f"\n  Gradient recovery:")
        print(f"    Left:  {mean_left:.1f} nm (true centre of mass: {self.wavelength_left_com:.1f} nm)")
        print(f"    Right: {mean_right:.1f} nm (true centre of mass: {self.wavelength_right_com:.1f} nm)")

    def test_min_localisations_threshold(self):
        """Pixels below min_localisations should get NaN."""
        nrf = NileRedFunctions.NileRed_Functions()

        # Use a very large pixel size so all locs are in ~1 pixel,
        # and a very high threshold so it gets skipped
        _, grid_info_high = nrf.fit_wavelengths_pixelated(
            h5_path=self.h5_path,
            filter_names=self.filter_names,
            camera_parameters=self.camera_params,
            pixel_size_nm=50.0,  # small pixels
            min_localisations=1000,  # impossibly high threshold
            camera_pixel_size=self.camera_pixel_size,
            verbose=False,
        )

        # All pixels should be skipped
        self.assertEqual(grid_info_high["n_pixels_fitted"], 0)
        self.assertTrue(np.all(np.isnan(grid_info_high["wl_grid"])))

    def test_metadata_grids_populated(self):
        """Check that n_locs and photon grids are populated."""
        nrf = NileRedFunctions.NileRed_Functions()
        df, grid_info = nrf.fit_wavelengths_pixelated(
            h5_path=self.h5_path,
            filter_names=self.filter_names,
            camera_parameters=self.camera_params,
            pixel_size_nm=500.0,  # very large pixels for fast test
            min_localisations=3,
            camera_pixel_size=self.camera_pixel_size,
            verbose=False,
        )

        n_locs_grid = grid_info["n_locs_grid"]
        total_photoelectrons_grid = grid_info["total_photoelectrons_grid"]

        # Total localisations in grid should equal input
        self.assertEqual(np.sum(n_locs_grid), len(df))

        # Photon grid should have positive values where locs exist
        fitted_mask = n_locs_grid > 0
        self.assertTrue(np.all(total_photoelectrons_grid[fitted_mask] > 0))

    def test_aggregate_id_column(self):
        """Pixels should be separated by aggregate ID."""
        # Add an aggregate_id column: assign left half to agg 0, right to agg 1
        df_agg = self.df_orig.copy()
        x_nm = df_agg["xc"].to_numpy() * self.camera_pixel_size
        df_agg["cluster_id"] = np.where(x_nm < self.fov_nm / 2, 0.0, 1.0)

        h5_agg = os.path.join(self.tmpdir, "test_locs_agg.h5")
        df_agg.to_hdf(h5_agg, key="data", mode="w", format="table")

        nrf = NileRedFunctions.NileRed_Functions()
        df_result, grid_info = nrf.fit_wavelengths_pixelated(
            h5_path=h5_agg,
            filter_names=self.filter_names,
            camera_parameters=self.camera_params,
            pixel_size_nm=200.0,
            min_localisations=3,
            camera_pixel_size=self.camera_pixel_size,
            verbose=False,
            aggregate_id_column="cluster_id",
        )

        # Should still recover the gradient
        left_mask = df_result["xc"] * self.camera_pixel_size < self.fov_nm / 2
        wl_left = df_result.loc[left_mask, "wl_pixel"].dropna()
        wl_right = df_result.loc[~left_mask, "wl_pixel"].dropna()

        self.assertGreater(len(wl_left), 0)
        self.assertGreater(len(wl_right), 0)
        self.assertLess(wl_left.mean(), wl_right.mean())

    def test_aggregate_fallback_fills_gaps(self):
        """Sub-threshold pixels should get aggregate wavelength when aggregate_id_column is set."""
        # Create a small aggregate (15 locs) that will be split across
        # multiple small pixels, each below the threshold
        df_small = self.df_orig.copy()
        # Assign all localisations to aggregate 0
        df_small["cluster_id"] = 0.0

        h5_small = os.path.join(self.tmpdir, "test_locs_small_agg.h5")
        df_small.to_hdf(h5_small, key="data", mode="w", format="table")

        nrf = NileRedFunctions.NileRed_Functions()

        # Use very small pixels and high threshold so most pixels are
        # below threshold — aggregate fallback should fill them in
        df_result, grid_info = nrf.fit_wavelengths_pixelated(
            h5_path=h5_small,
            filter_names=self.filter_names,
            camera_parameters=self.camera_params,
            pixel_size_nm=50.0,   # small pixels → few locs per pixel
            min_localisations=20, # high threshold
            camera_pixel_size=self.camera_pixel_size,
            verbose=False,
            aggregate_id_column="cluster_id",
        )

        # With aggregate fallback, all localisations should have a wavelength
        n_assigned = df_result["wl_pixel"].notna().sum()
        self.assertEqual(
            n_assigned, len(df_result),
            f"Expected all {len(df_result)} locs assigned, got {n_assigned}"
        )

        # Error column should also be populated for all assigned locs
        n_err_assigned = df_result["wl_pixel_err"].notna().sum()
        self.assertEqual(n_assigned, n_err_assigned,
                         "wl_pixel_err should be set wherever wl_pixel is set")

        # wl_err_grid should also have values where wl_grid does
        wl_fitted = ~np.isnan(grid_info["wl_grid"])
        err_fitted = ~np.isnan(grid_info["wl_err_grid"])
        np.testing.assert_array_equal(
            wl_fitted, err_fitted,
            err_msg="wl_err_grid should have values wherever wl_grid does"
        )

    def test_pixel_size_affects_grid_dimensions(self):
        """Smaller pixel_size_nm should give larger grid."""
        nrf = NileRedFunctions.NileRed_Functions()

        _, grid_large = nrf.fit_wavelengths_pixelated(
            h5_path=self.h5_path,
            filter_names=self.filter_names,
            camera_parameters=self.camera_params,
            pixel_size_nm=500.0,
            min_localisations=3,
            camera_pixel_size=self.camera_pixel_size,
            verbose=False,
        )

        _, grid_small = nrf.fit_wavelengths_pixelated(
            h5_path=self.h5_path,
            filter_names=self.filter_names,
            camera_parameters=self.camera_params,
            pixel_size_nm=100.0,
            min_localisations=3,
            camera_pixel_size=self.camera_pixel_size,
            verbose=False,
        )

        # Smaller pixels should give more grid cells
        large_total = grid_large["grid_shape"][0] * grid_large["grid_shape"][1]
        small_total = grid_small["grid_shape"][0] * grid_small["grid_shape"][1]
        self.assertGreater(small_total, large_total)

    @classmethod
    def tearDownClass(cls):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(cls.tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
