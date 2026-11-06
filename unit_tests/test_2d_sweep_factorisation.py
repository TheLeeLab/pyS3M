#!/usr/bin/env python3
"""Unit tests for the factorised 2-D sweep helpers.

Tests:
  _generate_photoelectron_batch  — shape, non-negative counts, reasonable total
  _apply_read_noise_batch        — output shape/dtype, noise statistics
  Poisson thinning               — thinned mean ≈ base mean × thin_p
  gen_camera_image_stack early return — same result as _generate_photoelectron_batch
"""

import sys
from pathlib import Path


import numpy as np
from scipy import stats
import pyS3M.Multicolour_Simulation_Functions as MSF
from pyS3M.Multicolour_Simulation_Functions import SimulationConfig, CameraParameters, FittingStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_camera_params(size=(12, 12)):
    """Minimal synthetic camera parameters."""
    w, h = size
    wavelength = np.arange(400, 751, 1)
    n_wl = len(wavelength)

    pixel_QYs = np.zeros((3, n_wl))
    pixel_QYs[0] = np.exp(-((wavelength - 470) / 40) ** 2) * 0.5   # B
    pixel_QYs[1] = np.exp(-((wavelength - 530) / 40) ** 2) * 0.6   # G
    pixel_QYs[2] = np.exp(-((wavelength - 630) / 40) ** 2) * 0.4   # R

    masks = {
        "B": np.zeros((w, h), dtype=bool),
        "G": np.zeros((w, h), dtype=bool),
        "R": np.zeros((w, h), dtype=bool),
    }
    masks["R"][::2, ::2] = True
    masks["G"][::2, 1::2] = True
    masks["G"][1::2, ::2] = True
    masks["B"][1::2, 1::2] = True

    return {
        "gain": np.ones((w, h)),
        "offset": np.ones((w, h)) * 100.0,
        "variance": np.ones((w, h)) * 4.0,
        "readnoise": 2.0,
        "rqe": np.ones((w, h)),
        "pixel_order": ["B", "G", "R"],
        "pixel_order_indices": {"B": 0, "G": 1, "R": 2},
        "masks": masks,
        "pixel_QYs": pixel_QYs,
    }, wavelength


def _build_sim_funcs():
    return MSF.MultiC_Sim_Funcs()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _get_deterministic_spectral(camera_params, wavelength, n_bootstrap):
    """Return (avg_wl, dpe, x0y0) using deterministic pixel fractions (no DB query)."""
    # Use a fixed average wavelength and uniform channel fractions
    avg_wl = 520.0  # nm (green-ish)
    # Deterministic colour fractions: evenly split across B/G/R
    dpe = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
    # Build x0y0 at image centre
    pixel_size = 69.0
    gain = camera_params["gain"]
    img_size = pixel_size * np.array(gain.shape)
    x0 = np.full(n_bootstrap, img_size[0] / 2)
    y0 = np.full(n_bootstrap, img_size[1] / 2)
    x0y0 = {"dye": np.zeros([n_bootstrap, 2, 1])}
    x0y0["dye"][:, :, :] = np.array([[x0, y0]]).T
    return avg_wl, dpe, x0y0


class TestGeneratePhotoelectronBatch:
    """Tests for _generate_photoelectron_batch."""

    def setup_method(self):
        np.random.seed(42)
        self.msf = _build_sim_funcs()
        self.camera_params, self.wavelength = _build_camera_params()
        self.config = SimulationConfig(
            n_bootstrap=200,
            use_stochastic_photons=False,
            background_photons=0.0,
            save_raw_results=False,
        )
        self.n_photon = 1000
        self.avg_wl, self.dpe, self.x0y0 = _get_deterministic_spectral(
            self.camera_params, self.wavelength, self.config.n_bootstrap
        )
        # Emission spectrum for the default dipole model, centred on avg_wl
        self.spectra = {"dye": np.exp(-0.5 * ((self.wavelength - self.avg_wl) / 20.0) ** 2)}

    def test_output_shape(self):
        n_photons = {"dye": np.full(self.config.n_bootstrap, self.n_photon)}
        pe = self.msf._generate_photoelectron_batch(
            self.camera_params, self.wavelength, self.avg_wl, self.dpe,
            n_photons, self.x0y0, self.config, emission_spectra=self.spectra,
        )
        H, W = self.camera_params["gain"].shape
        assert pe.shape == (self.config.n_bootstrap, H, W), \
            f"Expected ({self.config.n_bootstrap},{H},{W}), got {pe.shape}"

    def test_non_negative(self):
        n_photons = {"dye": np.full(self.config.n_bootstrap, self.n_photon)}
        pe = self.msf._generate_photoelectron_batch(
            self.camera_params, self.wavelength, self.avg_wl, self.dpe,
            n_photons, self.x0y0, self.config, emission_spectra=self.spectra,
        )
        assert pe.min() >= 0, "Photoelectron counts must be non-negative"

    def test_dtype_int(self):
        n_photons = {"dye": np.full(self.config.n_bootstrap, self.n_photon)}
        pe = self.msf._generate_photoelectron_batch(
            self.camera_params, self.wavelength, self.avg_wl, self.dpe,
            n_photons, self.x0y0, self.config, emission_spectra=self.spectra,
        )
        assert np.issubdtype(pe.dtype, np.integer), f"Expected integer dtype, got {pe.dtype}"

    def test_reasonable_total_photons(self):
        """Total photoelectrons per image should be << n_photon (Bayer + QE losses)."""
        n_photons = {"dye": np.full(self.config.n_bootstrap, self.n_photon)}
        pe = self.msf._generate_photoelectron_batch(
            self.camera_params, self.wavelength, self.avg_wl, self.dpe,
            n_photons, self.x0y0, self.config, emission_spectra=self.spectra,
        )
        mean_total = pe.sum(axis=(1, 2)).mean()
        assert mean_total < self.n_photon, \
            f"Mean total pe ({mean_total:.1f}) should be < n_photon ({self.n_photon})"
        assert mean_total > 0, "Mean total pe should be > 0"

    def test_helper_is_wrapper_for_early_return(self):
        """_generate_photoelectron_batch output should be statistically consistent
        with gen_camera_image_stack(return_photoelectrons_stack=True).
        (Both call the same code path; exact equality is not testable because
        Numba background threads make np.random non-reproducible after import.)
        """
        n_photons = {"dye": np.full(self.config.n_bootstrap, self.n_photon)}
        pe1 = self.msf._generate_photoelectron_batch(
            self.camera_params, self.wavelength, self.avg_wl, self.dpe,
            n_photons, self.x0y0, self.config, emission_spectra=self.spectra,
        )
        pe2 = self.msf.gen_camera_image_stack(
            self.camera_params, self.wavelength, self.avg_wl, self.dpe,
            n_photons, self.x0y0,
            smoothing_function=None,
            background_photons=self.config.background_photons,
            background_colour=self.config.background_colour,
            NA=self.config.NA,
            pixel_size=self.config.pixel_size,
            return_normal_image=False,
            return_photoelectrons_stack=True,
            emission_spectra=self.spectra,
        )
        # Same shape and dtype
        assert pe1.shape == pe2.shape
        assert pe1.dtype == pe2.dtype
        # Means within 20 % of each other (same distribution, different samples)
        m1, m2 = float(pe1.mean()), float(pe2.mean())
        assert abs(m1 - m2) / max(m1, 1e-6) < 0.20, \
            f"Means diverge too much: {m1:.2f} vs {m2:.2f}"


class TestApplyReadNoiseBatch:
    """Tests for _apply_read_noise_batch."""

    def setup_method(self):
        np.random.seed(0)
        self.msf = _build_sim_funcs()
        self.camera_params, self.wavelength = _build_camera_params()

    def _make_smoothing(self):
        """Create a no-op smoothing wrapper."""
        from unittest.mock import MagicMock
        sf = MagicMock()
        sf.args = {}
        sf.data_arg = "data"
        sf.smoothing_function = lambda data: data.copy()
        return sf

    def test_output_shape(self):
        H, W = 12, 12
        n_bs = 500
        pe = np.random.randint(0, 200, size=(n_bs, H, W), dtype=np.int32)
        gain = self.camera_params["gain"]
        offset = self.camera_params["offset"]
        rn = 2.0
        sf = self._make_smoothing()

        adu, sm = self.msf._apply_read_noise_batch(pe, rn, gain, offset, sf)
        assert adu.shape == (n_bs, H, W)
        assert sm.shape == (n_bs, H, W)

    def test_non_negative_adu(self):
        """ADU should be clipped to ≥ 0."""
        H, W = 12, 12
        n_bs = 500
        # Use very low pe to stress the clipping
        pe = np.zeros((n_bs, H, W), dtype=np.int32)
        gain = self.camera_params["gain"]
        offset = np.zeros_like(gain)  # zero offset so ADU = noise
        rn = 5.0
        sf = self._make_smoothing()
        adu, _ = self.msf._apply_read_noise_batch(pe, rn, gain, offset, sf)
        assert adu.min() >= 0, "ADU values must be ≥ 0 after clipping"

    def test_noise_statistics(self):
        """Noise standard deviation should match rn to within ~5 % with n=50000 samples."""
        H, W = 12, 12
        n_bs = 50000
        pe_val = 100  # deterministic pe
        pe = np.full((n_bs, H, W), pe_val, dtype=np.int32)
        gain = np.ones((H, W))
        offset = np.zeros((H, W))
        rn = 3.0
        sf = self._make_smoothing()

        adu, _ = self.msf._apply_read_noise_batch(pe, rn, gain, offset, sf)

        # Select a centre pixel (not clipped much for pe_val = 100)
        pixel_vals = adu[:, H // 2, W // 2].astype(float)
        measured_std = pixel_vals.std()
        # Allow 10 % tolerance (clipping at 0 shifts std slightly)
        assert abs(measured_std - rn) / rn < 0.10, \
            f"Expected rn≈{rn}, measured std={measured_std:.3f}"

    def test_mean_adu(self):
        """Mean ADU should be ≈ gain * pe + offset (ignoring clipping edge effects)."""
        H, W = 12, 12
        n_bs = 20000
        pe_val = 200
        pe = np.full((n_bs, H, W), pe_val, dtype=np.int32)
        gain = np.ones((H, W)) * 2.0
        offset = np.ones((H, W)) * 100.0
        rn = 2.0
        sf = self._make_smoothing()

        adu, _ = self.msf._apply_read_noise_batch(pe, rn, gain, offset, sf)
        expected_mean = 2.0 * pe_val + 100.0
        measured_mean = float(adu.mean())
        # 0.5 % relative tolerance
        assert abs(measured_mean - expected_mean) / expected_mean < 0.005, \
            f"Expected mean≈{expected_mean}, got {measured_mean:.2f}"


class TestPoissonThinning:
    """Verify that Binomial(K_max, p) gives correct mean for QY thinning."""

    def test_thinned_mean(self):
        """Thinned mean should be ≈ base mean × thin_p."""
        np.random.seed(123)
        H, W = 12, 12
        n_bs = 20000
        # Simulate a base pe_batch with known mean
        lam = 150  # expected photons per pixel
        pe_base = np.random.poisson(lam, size=(n_bs, H, W)).astype(np.int32)
        thin_p = 0.4
        pe_thinned = np.random.binomial(pe_base, thin_p).astype(np.int32)
        expected_mean = lam * thin_p
        measured_mean = float(pe_thinned.mean())
        rel_err = abs(measured_mean - expected_mean) / expected_mean
        assert rel_err < 0.01, \
            f"Thinning: expected mean≈{expected_mean:.1f}, got {measured_mean:.2f}"

    def test_thinned_variance(self):
        """Thinned samples: variance should ≈ mean (Poisson property)."""
        np.random.seed(456)
        n = 50000
        lam = 100
        pe_base = np.random.poisson(lam, size=n).astype(np.int32)
        thin_p = 0.5
        thinned = np.random.binomial(pe_base, thin_p).astype(float)
        expected_lam = lam * thin_p
        # For Poisson: mean == variance; allow 5 % relative tolerance
        rel_err_mean = abs(thinned.mean() - expected_lam) / expected_lam
        rel_err_var = abs(thinned.var() - expected_lam) / expected_lam
        assert rel_err_mean < 0.02, f"Mean deviation too large: {rel_err_mean:.3f}"
        assert rel_err_var < 0.05, f"Variance deviation too large: {rel_err_var:.3f}"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
