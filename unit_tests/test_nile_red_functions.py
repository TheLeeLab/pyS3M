"""Full coverage tests for pyS3M.NileRedFunctions -- Nile Red forward/inverse
spectral model (skew-Gaussian emission -> RGB+PSF predictions, and the inverse
least-squares wavelength fit from the colour fractions), plus the two higher-level pipelines built on it:
per-localisation (`fit_wavelengths_from_h5`) and per-pixel-grid
(`fit_wavelengths_pixelated`) wavelength fitting, and the heavy simulation
orchestrator `simulate_wavelength_precision`.

`unit_tests/test_nile_red_pixelated.py` already exercises `fit_wavelengths_pixelated`'s
happy path thoroughly with a realistic 500-localisation synthetic dataset (gradient
recovery, aggregate fallback, grid metadata) -- this file complements it with small
(~15-30 localisation) data for the error/validation/verbose branches that file doesn't
touch, and gives `fit_wavelengths_from_h5` (previously untested anywhere) and the pure
forward/inverse-model methods their first direct coverage.

Small-but-real optical system throughout (the real 600-point wavelength grid + 3 real
filter spectra, loaded once via a module-scoped fixture since SpectralFunctions'
duckdb-backed load takes ~0.5s) -- not literal tiny arrays, since the physics (skew-
Gaussian emission model, quantum-efficiency-weighted RGB integration) needs a real
grid to be meaningful. `_parallel_fit_wavelengths` uses tiny (2-4 item) fit_args lists
so the real ProcessPoolExecutor path stays fast.
"""
from __future__ import annotations

import types

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import pyS3M.NileRedFunctions as NileRedFunctions
import pyS3M.SpectralFunctions as SpectralFunctions

FILTER_NAMES = [
    "semrock-ff01-650-200",
    "semrock-di03-r514-t1-25x36",
    "semrock-ff01-515-lp",
]


@pytest.fixture(scope="module")
def nrf():
    return NileRedFunctions.NileRed_Functions()


@pytest.fixture(scope="module")
def optical_system(nrf):
    wavelength_array, pixel_QYs, filter_spectra = nrf.setup_optical_system(FILTER_NAMES)
    return wavelength_array, pixel_QYs, filter_spectra


def _synthetic_locs_df(n_locs, nrf, optical_system, wavelength_left=610.0,
                        wavelength_right=640.0, fov_nm=1000.0, camera_pixel_size=69.0,
                        seed=1, with_photons=True, with_aggregate=False):
    wavelength_array, pixel_QYs, filter_spectra = optical_system
    rng = np.random.default_rng(seed)
    x_nm = rng.uniform(0, fov_nm, n_locs)
    y_nm = rng.uniform(0, fov_nm, n_locs)
    xc = x_nm / camera_pixel_size
    yc = y_nm / camera_pixel_size
    true_wl = np.where(x_nm < fov_nm / 2, wavelength_left, wavelength_right)

    A_R = np.zeros(n_locs)
    A_G = np.zeros(n_locs)
    A_B = np.zeros(n_locs)
    s_x = np.zeros(n_locs)
    s_y = np.zeros(n_locs)
    for i in range(n_locs):
        preds = nrf.nile_red_forward_model(
            true_wl[i], filter_spectra, wavelength_array, pixel_QYs, NA=1.49,
        )
        A_R[i] = preds["R"]
        A_G[i] = preds["G"]
        A_B[i] = preds["B"]
        s_x[i] = preds["sigma_x"] / camera_pixel_size
        s_y[i] = preds["sigma_y"] / camera_pixel_size

    noise_frac = 0.03
    A_R_err = np.abs(A_R) * noise_frac + 1e-4
    A_G_err = np.abs(A_G) * noise_frac + 1e-4
    A_B_err = np.abs(A_B) * noise_frac + 1e-4
    s_x_err = np.abs(s_x) * noise_frac + 1e-4
    s_y_err = np.abs(s_y) * noise_frac + 1e-4

    data = {
        "xc": xc, "yc": yc,
        "A_R": A_R, "A_G": A_G, "A_B": A_B,
        "s_x": s_x, "s_y": s_y,
        "A_R_err": A_R_err, "A_G_err": A_G_err, "A_B_err": A_B_err,
        "s_x_err": s_x_err, "s_y_err": s_y_err,
    }
    if with_photons:
        data["photons"] = rng.uniform(500, 2000, n_locs)
        data["background_photons"] = rng.uniform(20, 80, n_locs)
    if with_aggregate:
        data["cluster_id"] = np.where(x_nm < fov_nm / 2, 0.0, 1.0)
    return pd.DataFrame(data)


# ======================================================================
# setup_optical_system / generate_nile_red_spectrum / spectral_centre_of_mass
# ======================================================================

class TestSetupOpticalSystem:
    def test_returns_expected_shapes(self, nrf):
        wavelength_array, pixel_QYs, filter_spectra = nrf.setup_optical_system(FILTER_NAMES)
        assert wavelength_array.ndim == 1
        assert pixel_QYs.shape == (3, len(wavelength_array))
        assert filter_spectra.shape[-1] == len(wavelength_array)


class TestGenerateNileRedSpectrum:
    def test_normalized_spectrum_integrates_to_one(self, nrf, optical_system):
        wavelength_array, _, _ = optical_system
        spectrum = nrf.generate_nile_red_spectrum(617.6, wavelength_array, normalize=True)
        assert np.trapz(spectrum, wavelength_array) == pytest.approx(1.0, abs=1e-3)

    def test_unnormalized_spectrum_not_unit_integral(self, nrf, optical_system):
        wavelength_array, _, _ = optical_system
        spectrum = nrf.generate_nile_red_spectrum(617.6, wavelength_array, normalize=False)
        assert spectrum.max() > 0

    def test_custom_sigma_alpha_override_defaults(self, nrf, optical_system):
        wavelength_array, _, _ = optical_system
        s1 = nrf.generate_nile_red_spectrum(617.6, wavelength_array, sigma_energy=0.05, alpha=-1.0)
        s2 = nrf.generate_nile_red_spectrum(617.6, wavelength_array)
        assert not np.allclose(s1, s2)


class TestSpectralCentreOfMass:
    def test_differs_from_location_parameter(self, nrf, optical_system):
        wavelength_array, _, _ = optical_system
        com = nrf.spectral_centre_of_mass(617.6, wavelength_array)
        assert com != pytest.approx(617.6, abs=1.0)
        assert 550.0 < com < 750.0

    def test_wavelength_center_far_outside_array_falls_back(self, nrf):
        # A location parameter far outside a narrow wavelength grid renders an
        # (unnormalised) spectrum whose integral rounds to exactly 0 -> the
        # denom<=0 fallback returns wavelength_center itself.
        narrow_array = np.linspace(600.0, 650.0, 50)
        com = nrf.spectral_centre_of_mass(100.0, narrow_array)
        assert com == pytest.approx(100.0)


class TestWavelengthCenterForPeak:
    def test_lut_path_for_default_params(self, nrf, optical_system):
        wavelength_array, _, _ = optical_system
        wc = nrf.wavelength_center_for_peak(600.0, wavelength_array)
        spectrum = nrf.generate_nile_red_spectrum(wc, wavelength_array, normalize=False)
        peak_wl = wavelength_array[np.argmax(spectrum)]
        assert peak_wl == pytest.approx(600.0, abs=2.0)

    def test_optimisation_fallback_for_custom_alpha(self, nrf, optical_system):
        wavelength_array, _, _ = optical_system
        wc = nrf.wavelength_center_for_peak(600.0, wavelength_array, alpha=-1.0)
        assert 450.0 <= wc <= 800.0

    def test_optimisation_fallback_for_out_of_range_peak(self, nrf, optical_system):
        wavelength_array, _, _ = optical_system
        wc = nrf.wavelength_center_for_peak(500.0, wavelength_array)
        assert 450.0 <= wc <= 800.0


class TestWavelengthCenterForCentreOfMass:
    def test_roundtrips_with_spectral_centre_of_mass(self, nrf, optical_system):
        wavelength_array, _, _ = optical_system
        target_com = 630.0
        wc = nrf.wavelength_center_for_centre_of_mass(target_com, wavelength_array)
        recovered_com = nrf.spectral_centre_of_mass(wc, wavelength_array)
        assert recovered_com == pytest.approx(target_com, abs=1.0)


# ======================================================================
# apply_optical_filters / calculate_rgb_from_spectrum / calculate_psf_width_from_spectrum
# ======================================================================

class TestApplyOpticalFilters:
    def test_filters_attenuate_spectrum(self, nrf, optical_system):
        wavelength_array, _, filter_spectra = optical_system
        spectrum = nrf.generate_nile_red_spectrum(617.6, wavelength_array)
        filtered = nrf.apply_optical_filters(spectrum, filter_spectra)
        assert np.all(filtered <= spectrum + 1e-12)


class TestCalculateRgbFromSpectrum:
    def test_normalized_to_unit_sum(self, nrf, optical_system):
        wavelength_array, pixel_QYs, filter_spectra = optical_system
        spectrum = nrf.generate_nile_red_spectrum(617.6, wavelength_array)
        filtered = nrf.apply_optical_filters(spectrum, filter_spectra)
        rgb = nrf.calculate_rgb_from_spectrum(filtered, wavelength_array, pixel_QYs)
        assert rgb.sum() == pytest.approx(1.0, abs=1e-6)
        assert np.all(rgb >= 0)

    def test_zero_spectrum_returns_zero_without_division_error(self, nrf, optical_system):
        wavelength_array, pixel_QYs, _ = optical_system
        zero_spectrum = np.zeros_like(wavelength_array)
        rgb = nrf.calculate_rgb_from_spectrum(zero_spectrum, wavelength_array, pixel_QYs)
        np.testing.assert_allclose(rgb, 0.0)


class TestCalculatePsfWidthFromSpectrum:
    def test_normal_spectrum(self, nrf, optical_system):
        wavelength_array, _, filter_spectra = optical_system
        spectrum = nrf.generate_nile_red_spectrum(617.6, wavelength_array)
        filtered = nrf.apply_optical_filters(spectrum, filter_spectra)
        sigma = nrf.calculate_psf_width_from_spectrum(filtered, wavelength_array)
        assert sigma > 0

    def test_zero_spectrum_falls_back_to_wavelength_mean(self, nrf, optical_system):
        wavelength_array, _, _ = optical_system
        zero_spectrum = np.zeros_like(wavelength_array)
        sigma = nrf.calculate_psf_width_from_spectrum(zero_spectrum, wavelength_array)
        assert sigma > 0


class TestNileRedForwardModel:
    def test_predictions_have_expected_keys(self, nrf, optical_system):
        wavelength_array, pixel_QYs, filter_spectra = optical_system
        preds = nrf.nile_red_forward_model(617.6, filter_spectra, wavelength_array, pixel_QYs)
        assert set(preds.keys()) == {"R", "G", "B", "sigma_x", "sigma_y"}
        assert preds["sigma_x"] == preds["sigma_y"]


class TestResidualsNileRed:
    def _rgb(self, nrf, optical_system, wl=617.6):
        wavelength_array, pixel_QYs, filter_spectra = optical_system
        return nrf.predict_rgb(wl, filter_spectra, wavelength_array, pixel_QYs)

    def test_zero_at_exact_prediction(self, nrf, optical_system):
        wavelength_array, pixel_QYs, filter_spectra = optical_system
        rgb = self._rgb(nrf, optical_system)
        residuals = nrf.residuals_nile_red(
            np.array([617.6]), rgb, np.ones(3), filter_spectra, wavelength_array, pixel_QYs,
        )
        assert residuals.shape == (3,)
        np.testing.assert_allclose(residuals, 0.0, atol=1e-10)

    def test_scalar_wavelength_center_accepted(self, nrf, optical_system):
        wavelength_array, pixel_QYs, filter_spectra = optical_system
        rgb = self._rgb(nrf, optical_system)
        residuals = nrf.residuals_nile_red(
            617.6, rgb, np.ones(3), filter_spectra, wavelength_array, pixel_QYs,
        )
        np.testing.assert_allclose(residuals, 0.0, atol=1e-10)

    def test_weighted_by_errors(self, nrf, optical_system):
        wavelength_array, pixel_QYs, filter_spectra = optical_system
        rgb = self._rgb(nrf, optical_system)
        observed = rgb + np.array([0.01, -0.01, 0.0])
        residuals = nrf.residuals_nile_red(
            617.6, observed, np.array([0.01, 0.005, 1.0]), filter_spectra, wavelength_array, pixel_QYs,
        )
        np.testing.assert_allclose(residuals, [1.0, -2.0, 0.0], atol=1e-10)

    def test_predict_rgb_matches_forward_model(self, nrf, optical_system):
        wavelength_array, pixel_QYs, filter_spectra = optical_system
        preds = nrf.nile_red_forward_model(617.6, filter_spectra, wavelength_array, pixel_QYs)
        np.testing.assert_allclose(self._rgb(nrf, optical_system), [preds["R"], preds["G"], preds["B"]])


# ======================================================================
# _normalize_rgb_with_errors / _valid_colour_rows / _pool_colour_fractions
# ======================================================================

class TestNormalizeRgbWithErrors:
    def test_normalizes_to_unit_sum(self, nrf):
        rgb = np.array([200.0, 100.0, 50.0])
        rgb_err = np.array([10.0, 8.0, 5.0])
        norm, norm_err = NileRedFunctions.NileRed_Functions._normalize_rgb_with_errors(rgb, rgb_err)
        assert norm.sum() == pytest.approx(1.0)
        assert np.all(norm_err > 0)

    def test_fraction_errors_only_rescaled(self, nrf):
        # inputs are fractions whose errors already include the total's uncertainty
        norm, norm_err = NileRedFunctions.NileRed_Functions._normalize_rgb_with_errors(
            np.array([0.6, 0.3, 0.1]), np.array([0.02, 0.015, 0.01]))
        np.testing.assert_allclose(norm_err, [0.02, 0.015, 0.01])

    def test_zero_channel_gets_fallback_error(self, nrf):
        rgb = np.array([200.0, 0.0, 50.0])
        rgb_err = np.array([10.0, 8.0, 5.0])
        norm, norm_err = NileRedFunctions.NileRed_Functions._normalize_rgb_with_errors(rgb, rgb_err)
        assert norm_err[1] == pytest.approx(1e-3)


class TestValidColourRows:
    def _df(self):
        return pd.DataFrame({
            "A_R": [0.6, np.nan, 0.6, 0.6, 0.0, 0.6],
            "A_G": [0.3, 0.3, 0.3, 0.3, 0.0, 0.3],
            "A_B": [0.1, 0.1, 0.1, 0.1, 0.0, 0.1],
            "A_R_err": [0.01, 0.01, np.nan, 0.01, 0.01, 0.01],
            "A_G_err": [0.01, 0.01, 0.01, 0.0, 0.01, 0.01],
            "A_B_err": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
            "photons": [500.0, 500.0, 500.0, 500.0, 500.0, np.nan],
        })

    def test_rejects_nonfinite_and_nonpositive(self):
        valid = NileRedFunctions.NileRed_Functions._valid_colour_rows(self._df())
        np.testing.assert_array_equal(valid, [True, False, False, False, False, True])

    def test_require_photons(self):
        valid = NileRedFunctions.NileRed_Functions._valid_colour_rows(self._df(), require_photons=True)
        np.testing.assert_array_equal(valid, [True, False, False, False, False, False])


class TestPoolColourFractions:
    def test_fraction_of_summed_photoelectrons(self):
        fractions = np.array([[0.6, 0.3, 0.1], [0.4, 0.4, 0.2]])
        errors = np.array([[0.02, 0.02, 0.01], [0.04, 0.04, 0.02]])
        photons = np.array([3000.0, 1000.0])
        pooled, pooled_err = NileRedFunctions.NileRed_Functions._pool_colour_fractions(
            fractions, errors, photons)
        # summed amplitudes: R 1800 + 400, G 900 + 400, B 300 + 200 of 4000
        np.testing.assert_allclose(pooled, [0.55, 0.325, 0.125])
        np.testing.assert_allclose(
            pooled_err, np.sqrt((3000.0 * errors[0]) ** 2 + (1000.0 * errors[1]) ** 2) / 4000.0)

    def test_equal_photons_is_plain_mean(self):
        fractions = np.array([[0.6, 0.3, 0.1], [0.4, 0.4, 0.2], [0.5, 0.35, 0.15]])
        errors = np.full((3, 3), 0.03)
        pooled, pooled_err = NileRedFunctions.NileRed_Functions._pool_colour_fractions(
            fractions, errors, np.full(3, 800.0))
        np.testing.assert_allclose(pooled, fractions.mean(axis=0))
        np.testing.assert_allclose(pooled_err, 0.03 / np.sqrt(3))

    def test_unbiased_where_inverse_error_weighting_is_not(self):
        # Errors that grow with the fraction (as photon-counting errors do) pull an
        # inverse-error-weighted mean low; the photoelectron pool stays on the truth.
        rng = np.random.default_rng(0)
        n, truth = 20000, 0.1
        f_b = rng.normal(truth, 0.03, n)
        f_b = f_b[f_b > 0]
        e_b = 0.3 * np.sqrt(f_b)
        fractions = np.column_stack([0.6 - (f_b - truth), np.full(len(f_b), 0.3), f_b])
        errors = np.column_stack([np.full(len(f_b), 0.03), np.full(len(f_b), 0.03), e_b])
        pooled, _ = NileRedFunctions.NileRed_Functions._pool_colour_fractions(
            fractions, errors, np.full(len(f_b), 1000.0))
        inverse_var_mean = np.sum(f_b / e_b ** 2) / np.sum(1 / e_b ** 2)
        assert pooled[2] == pytest.approx(truth, abs=1e-3)
        assert inverse_var_mean < truth - 5e-3


# ======================================================================
# _parallel_fit_wavelengths
# ======================================================================

class TestParallelFitWavelengths:
    def _small_fit_args(self, nrf, optical_system, n=3, wl=617.6):
        wavelength_array, pixel_QYs, filter_spectra = optical_system
        rgb = nrf.predict_rgb(wl, filter_spectra, wavelength_array, pixel_QYs)
        rgb_err = np.abs(rgb) * 0.03 + 1e-4
        args = (
            rgb, rgb_err, filter_spectra, wavelength_array, pixel_QYs, (500.0, 750.0), None,
        )
        return [args] * n

    def test_successful_fits(self, nrf, optical_system):
        fit_args = self._small_fit_args(nrf, optical_system, n=3)
        results = nrf._parallel_fit_wavelengths(fit_args, n_workers=2, verbose=True, progress_interval=1)
        assert len(results) == 3
        for wl, wl_err in results:
            assert not np.isnan(wl)

    def test_failed_fit_returns_nan(self, nrf, optical_system, monkeypatch):
        fit_args = self._small_fit_args(nrf, optical_system, n=2)

        def _raise(*a, **kw):
            raise RuntimeError("forced failure")

        monkeypatch.setattr(NileRedFunctions, "_fit_nile_red_wavelength_standalone", _raise)
        results = nrf._parallel_fit_wavelengths(fit_args, n_workers=1, verbose=True)
        assert all(np.isnan(wl) for wl, _ in results)

    def test_callbacks_invoked(self, optical_system):
        from pyS3M.Constants import AnalysisConfig

        calls = {"progress": 0, "logging": 0}
        config = AnalysisConfig(
            progress_callback=lambda frac, msg: calls.__setitem__("progress", calls["progress"] + 1),
            logging_callback=lambda msg: calls.__setitem__("logging", calls["logging"] + 1),
        )
        nrf_cb = NileRedFunctions.NileRed_Functions(config=config)
        fit_args = self._small_fit_args(nrf_cb, optical_system, n=2)
        nrf_cb._parallel_fit_wavelengths(fit_args, n_workers=1, verbose=True, progress_interval=1)
        assert calls["progress"] > 0
        assert calls["logging"] > 0

    def test_failed_fit_invokes_logging_callback(self, optical_system, monkeypatch):
        from pyS3M.Constants import AnalysisConfig

        calls = {"logging": 0}
        config = AnalysisConfig(
            logging_callback=lambda msg: calls.__setitem__("logging", calls["logging"] + 1),
        )
        nrf_cb = NileRedFunctions.NileRed_Functions(config=config)
        fit_args = self._small_fit_args(nrf_cb, optical_system, n=1)

        def _raise(*a, **kw):
            raise RuntimeError("forced failure")

        monkeypatch.setattr(NileRedFunctions, "_fit_nile_red_wavelength_standalone", _raise)
        nrf_cb._parallel_fit_wavelengths(fit_args, n_workers=1, verbose=True)
        assert calls["logging"] > 0


# ======================================================================
# fit_nile_red_wavelength / _fit_nile_red_wavelength_standalone
# ======================================================================

class TestFitNileRedWavelength:
    def _rgb(self, nrf, optical_system, wl=617.6):
        wavelength_array, pixel_QYs, filter_spectra = optical_system
        rgb = nrf.predict_rgb(wl, filter_spectra, wavelength_array, pixel_QYs)
        return rgb, np.abs(rgb) * 0.02 + 1e-4

    def test_recovers_known_wavelength(self, nrf, optical_system):
        wavelength_array, pixel_QYs, filter_spectra = optical_system
        rgb, rgb_err = self._rgb(nrf, optical_system, wl=630.0)
        wl_fit, out = nrf.fit_nile_red_wavelength(
            rgb, rgb_err, filter_spectra, wavelength_array, pixel_QYs,
        )
        assert wl_fit == pytest.approx(nrf.spectral_centre_of_mass(630.0, wavelength_array), abs=0.05)
        assert set(out) == {"R", "G", "B", "wavelength_error"}
        np.testing.assert_allclose([out["R"], out["G"], out["B"]], rgb, atol=1e-6)

    def test_custom_initial_guess(self, nrf, optical_system):
        wavelength_array, pixel_QYs, filter_spectra = optical_system
        rgb, rgb_err = self._rgb(nrf, optical_system)
        wl_fit, out = nrf.fit_nile_red_wavelength(
            rgb, rgb_err, filter_spectra, wavelength_array, pixel_QYs,
            wavelength_initial_guess=650.0,
        )
        assert "wavelength_error" in out

    def test_out_of_bounds_initial_guess_falls_back_to_midpoint(self, nrf, optical_system):
        wavelength_array, pixel_QYs, filter_spectra = optical_system
        rgb, rgb_err = self._rgb(nrf, optical_system)
        wl_fit, out = nrf.fit_nile_red_wavelength(
            rgb, rgb_err, filter_spectra, wavelength_array, pixel_QYs,
            wavelength_bounds=(500.0, 750.0), wavelength_initial_guess=10000.0,
        )
        assert "wavelength_error" in out

    @pytest.mark.parametrize("bad", ["nan_rgb", "nan_err", "zero_err", "zero_sum"])
    def test_unusable_input_returns_nan(self, nrf, optical_system, bad):
        # A NaN error used to drop that channel's residual silently, so the fit
        # returned its starting guess as if it were a measurement.
        wavelength_array, pixel_QYs, filter_spectra = optical_system
        rgb, rgb_err = self._rgb(nrf, optical_system)
        if bad == "nan_rgb":
            rgb[0] = np.nan
        elif bad == "nan_err":
            rgb_err[1] = np.nan
        elif bad == "zero_err":
            rgb_err[2] = 0.0
        else:
            rgb = np.zeros(3)
        wl_fit, out = nrf.fit_nile_red_wavelength(
            rgb, rgb_err, filter_spectra, wavelength_array, pixel_QYs,
        )
        assert np.isnan(wl_fit) and np.isnan(out["wavelength_error"])

    def test_zero_jacobian_yields_nan_error(self, nrf, optical_system, monkeypatch):
        # JtJ<=0 (a degenerate/zero Jacobian) can't be inverted for an error
        # estimate -- force it via a stubbed least_squares result rather than
        # hunting for real data that produces one.
        wavelength_array, pixel_QYs, filter_spectra = optical_system
        rgb, rgb_err = self._rgb(nrf, optical_system)

        fake_result = types.SimpleNamespace(
            x=np.array([617.6]),
            jac=np.zeros((3, 1)),
            fun=np.zeros(3),
        )
        monkeypatch.setattr(NileRedFunctions, "least_squares", lambda *a, **kw: fake_result)
        wl_fit, out = nrf.fit_nile_red_wavelength(
            rgb, rgb_err, filter_spectra, wavelength_array, pixel_QYs,
        )
        assert np.isnan(out["wavelength_error"])


class TestFitNileRedWavelengthStandalone:
    def test_success(self, optical_system):
        wavelength_array, pixel_QYs, filter_spectra = optical_system
        nrf_local = NileRedFunctions.NileRed_Functions()
        rgb = nrf_local.predict_rgb(617.6, filter_spectra, wavelength_array, pixel_QYs)
        rgb_err = np.abs(rgb) * 0.02 + 1e-4
        wl, wl_err = NileRedFunctions._fit_nile_red_wavelength_standalone(
            rgb, rgb_err, filter_spectra, wavelength_array, pixel_QYs,
        )
        assert not np.isnan(wl)

    def test_exception_returns_nan_pair(self, optical_system, monkeypatch):
        wavelength_array, pixel_QYs, filter_spectra = optical_system

        def _raise(self, *a, **kw):
            raise RuntimeError("forced failure")

        monkeypatch.setattr(NileRedFunctions.NileRed_Functions, "fit_nile_red_wavelength", _raise)
        wl, wl_err = NileRedFunctions._fit_nile_red_wavelength_standalone(
            np.array([0.5, 0.3, 0.2]), np.array([0.01, 0.01, 0.01]),
            filter_spectra, wavelength_array, pixel_QYs,
        )
        assert np.isnan(wl) and np.isnan(wl_err)


# ======================================================================
# fit_wavelengths_from_h5
# ======================================================================

class TestFitWavelengthsFromH5:
    def _camera_params(self, optical_system):
        wavelength_array, pixel_QYs, _ = optical_system
        return {"pixel_QYs": pixel_QYs, "wavelength": wavelength_array}

    def test_missing_file_raises(self, nrf, optical_system, tmp_path):
        with pytest.raises(FileNotFoundError):
            nrf.fit_wavelengths_from_h5(
                str(tmp_path / "missing.h5"), FILTER_NAMES, self._camera_params(optical_system),
            )

    def test_missing_required_columns_raises(self, nrf, optical_system, tmp_path):
        path = tmp_path / "bad.h5"
        pd.DataFrame({"A_R": [0.5]}).to_hdf(path, key="data", mode="w", format="table")
        with pytest.raises(ValueError, match="missing required columns"):
            nrf.fit_wavelengths_from_h5(str(path), FILTER_NAMES, self._camera_params(optical_system))

    def test_basic_fit(self, nrf, optical_system, tmp_path):
        df = _synthetic_locs_df(10, nrf, optical_system, with_photons=True, seed=2)
        path = tmp_path / "locs.h5"
        df.to_hdf(path, key="data", mode="w", format="table")
        out = nrf.fit_wavelengths_from_h5(
            str(path), FILTER_NAMES, self._camera_params(optical_system), verbose=True,
        )
        assert "wl_fit" in out.columns
        assert "wl_fit_err" in out.columns
        assert out["wl_fit"].notna().any()

    def test_basic_fit_without_photons_column(self, nrf, optical_system, tmp_path):
        df = _synthetic_locs_df(8, nrf, optical_system, with_photons=False, seed=3)
        path = tmp_path / "locs_no_photons.h5"
        df.to_hdf(path, key="data", mode="w", format="table")
        out = nrf.fit_wavelengths_from_h5(
            str(path), FILTER_NAMES, self._camera_params(optical_system), verbose=True,
        )
        assert "wl_fit" in out.columns

    def test_camera_parameters_without_wavelength_falls_back(self, nrf, optical_system, tmp_path):
        df = _synthetic_locs_df(6, nrf, optical_system, with_photons=True, seed=4)
        path = tmp_path / "locs_nowl.h5"
        df.to_hdf(path, key="data", mode="w", format="table")
        _, pixel_QYs, _ = optical_system
        out = nrf.fit_wavelengths_from_h5(
            str(path), FILTER_NAMES, {"pixel_QYs": pixel_QYs}, verbose=True,
        )
        assert "wl_fit" in out.columns

    def test_aggregate_id_column_missing_raises(self, nrf, optical_system, tmp_path):
        df = _synthetic_locs_df(6, nrf, optical_system, with_photons=True, seed=5)
        path = tmp_path / "locs_agg_missing.h5"
        df.to_hdf(path, key="data", mode="w", format="table")
        with pytest.raises(ValueError, match="aggregate_id_column"):
            nrf.fit_wavelengths_from_h5(
                str(path), FILTER_NAMES, self._camera_params(optical_system),
                aggregate_id_column="not_a_real_column",
            )

    def test_aggregate_requires_photons(self, nrf, optical_system, tmp_path):
        df = _synthetic_locs_df(6, nrf, optical_system, with_photons=False, with_aggregate=True, seed=5)
        path = tmp_path / "locs_agg_no_photons.h5"
        df.to_hdf(path, key="data", mode="w", format="table")
        with pytest.raises(ValueError, match="photons"):
            nrf.fit_wavelengths_from_h5(
                str(path), FILTER_NAMES, self._camera_params(optical_system),
                aggregate_id_column="cluster_id",
            )

    def test_aggregate_id_column_two_step_fit(self, nrf, optical_system, tmp_path):
        df = _synthetic_locs_df(15, nrf, optical_system, with_photons=True, with_aggregate=True, seed=6)
        path = tmp_path / "locs_agg.h5"
        df.to_hdf(path, key="data", mode="w", format="table")
        out = nrf.fit_wavelengths_from_h5(
            str(path), FILTER_NAMES, self._camera_params(optical_system),
            aggregate_id_column="cluster_id", verbose=True,
        )
        assert "wl_fit_aggregate" in out.columns

    def test_aggregate_with_zero_rgb_sum_is_skipped(self, nrf, optical_system, tmp_path):
        df = _synthetic_locs_df(15, nrf, optical_system, with_photons=True, with_aggregate=True, seed=14)
        df.loc[df["cluster_id"] == 0.0, ["A_R", "A_G", "A_B"]] = 0.0
        path = tmp_path / "locs_agg_zero.h5"
        df.to_hdf(path, key="data", mode="w", format="table")
        out = nrf.fit_wavelengths_from_h5(
            str(path), FILTER_NAMES, self._camera_params(optical_system),
            aggregate_id_column="cluster_id", verbose=True,
        )
        assert "wl_fit" in out.columns

    def test_output_path_saves_file(self, nrf, optical_system, tmp_path):
        df = _synthetic_locs_df(6, nrf, optical_system, with_photons=True, seed=7)
        path = tmp_path / "locs_save.h5"
        df.to_hdf(path, key="data", mode="w", format="table")
        out_path = tmp_path / "locs_out.h5"
        nrf.fit_wavelengths_from_h5(
            str(path), FILTER_NAMES, self._camera_params(optical_system),
            output_path=str(out_path), verbose=True,
        )
        assert out_path.exists()

    def test_all_zero_rgb_rows_are_skipped(self, nrf, optical_system, tmp_path):
        df = _synthetic_locs_df(6, nrf, optical_system, with_photons=True, seed=8)
        df.loc[0, ["A_R", "A_G", "A_B"]] = 0.0
        path = tmp_path / "locs_zero.h5"
        df.to_hdf(path, key="data", mode="w", format="table")
        out = nrf.fit_wavelengths_from_h5(
            str(path), FILTER_NAMES, self._camera_params(optical_system), verbose=False,
        )
        assert np.isnan(out.loc[0, "wl_fit"])

    def test_nonfinite_error_rows_are_skipped(self, nrf, optical_system, tmp_path):
        df = _synthetic_locs_df(6, nrf, optical_system, with_photons=True, seed=8)
        df.loc[1, "A_G_err"] = np.nan
        df.loc[2, "A_B_err"] = 0.0
        path = tmp_path / "locs_bad_err.h5"
        df.to_hdf(path, key="data", mode="w", format="table")
        out = nrf.fit_wavelengths_from_h5(
            str(path), FILTER_NAMES, self._camera_params(optical_system), verbose=False,
        )
        assert out.loc[[1, 2], "wl_fit"].isna().all()
        assert out.loc[[0, 3, 4, 5], "wl_fit"].notna().all()


# ======================================================================
# fit_wavelengths_pixelated -- supplementary branches not covered by the
# larger existing unit_tests/test_nile_red_pixelated.py suite.
# ======================================================================

class TestFitWavelengthsPixelatedSupplementary:
    def _camera_params(self, optical_system):
        wavelength_array, pixel_QYs, _ = optical_system
        return {"pixel_QYs": pixel_QYs, "wavelength": wavelength_array}

    def test_missing_file_raises(self, nrf, optical_system, tmp_path):
        with pytest.raises(FileNotFoundError):
            nrf.fit_wavelengths_pixelated(
                str(tmp_path / "missing.h5"), FILTER_NAMES, self._camera_params(optical_system),
            )

    def test_missing_required_columns_raises(self, nrf, optical_system, tmp_path):
        path = tmp_path / "bad.h5"
        pd.DataFrame({"A_R": [0.5]}).to_hdf(path, key="data", mode="w", format="table")
        with pytest.raises(ValueError, match="missing required columns"):
            nrf.fit_wavelengths_pixelated(str(path), FILTER_NAMES, self._camera_params(optical_system))

    def test_aggregate_id_column_missing_raises(self, nrf, optical_system, tmp_path):
        df = _synthetic_locs_df(10, nrf, optical_system, with_photons=True, seed=9)
        path = tmp_path / "locs.h5"
        df.to_hdf(path, key="data", mode="w", format="table")
        with pytest.raises(ValueError, match="aggregate_id_column"):
            nrf.fit_wavelengths_pixelated(
                str(path), FILTER_NAMES, self._camera_params(optical_system),
                pixel_size_nm=200.0, aggregate_id_column="not_a_real_column",
            )

    def test_missing_photons_raises(self, nrf, optical_system, tmp_path):
        df = _synthetic_locs_df(10, nrf, optical_system, with_photons=False, seed=10)
        path = tmp_path / "locs_no_photons.h5"
        df.to_hdf(path, key="data", mode="w", format="table")
        with pytest.raises(ValueError, match="photons"):
            nrf.fit_wavelengths_pixelated(
                str(path), FILTER_NAMES, self._camera_params(optical_system),
                pixel_size_nm=200.0, min_localisations=2,
            )

    def test_return_dataframe_only(self, nrf, optical_system, tmp_path):
        df = _synthetic_locs_df(10, nrf, optical_system, with_photons=True, seed=10)
        path = tmp_path / "locs_df_only.h5"
        df.to_hdf(path, key="data", mode="w", format="table")
        out = nrf.fit_wavelengths_pixelated(
            str(path), FILTER_NAMES, self._camera_params(optical_system),
            pixel_size_nm=200.0, min_localisations=2, verbose=True, return_grid=False,
        )
        assert "wl_pixel" in out.columns

    def test_camera_parameters_without_wavelength_falls_back(self, nrf, optical_system, tmp_path):
        df = _synthetic_locs_df(10, nrf, optical_system, with_photons=True, seed=11)
        path = tmp_path / "locs_nowl.h5"
        df.to_hdf(path, key="data", mode="w", format="table")
        _, pixel_QYs, _ = optical_system
        out, grid_info = nrf.fit_wavelengths_pixelated(
            str(path), FILTER_NAMES, {"pixel_QYs": pixel_QYs},
            pixel_size_nm=200.0, min_localisations=2, verbose=True,
        )
        assert "wl_pixel" in out.columns

    def test_output_path_saves_file(self, nrf, optical_system, tmp_path):
        df = _synthetic_locs_df(10, nrf, optical_system, with_photons=True, seed=12)
        path = tmp_path / "locs_save.h5"
        df.to_hdf(path, key="data", mode="w", format="table")
        out_path = tmp_path / "locs_pixel_out.h5"
        nrf.fit_wavelengths_pixelated(
            str(path), FILTER_NAMES, self._camera_params(optical_system),
            pixel_size_nm=200.0, min_localisations=2, output_path=str(out_path), verbose=True,
        )
        assert out_path.exists()

    def test_aggregate_group_all_zero_rgb_is_skipped(self, nrf, optical_system, tmp_path):
        df = _synthetic_locs_df(12, nrf, optical_system, with_photons=True, with_aggregate=True, seed=13)
        df.loc[df["cluster_id"] == 0.0, ["A_R", "A_G", "A_B"]] = 0.0
        path = tmp_path / "locs_agg_zero.h5"
        df.to_hdf(path, key="data", mode="w", format="table")
        out, grid_info = nrf.fit_wavelengths_pixelated(
            str(path), FILTER_NAMES, self._camera_params(optical_system),
            pixel_size_nm=200.0, min_localisations=2, aggregate_id_column="cluster_id", verbose=True,
        )
        assert isinstance(grid_info, dict)

    def test_nan_aggregate_id_row_is_skipped(self, nrf, optical_system, tmp_path):
        df = _synthetic_locs_df(12, nrf, optical_system, with_photons=True, with_aggregate=True, seed=15)
        df.loc[0, "cluster_id"] = np.nan
        path = tmp_path / "locs_agg_nan.h5"
        df.to_hdf(path, key="data", mode="w", format="table")
        out, grid_info = nrf.fit_wavelengths_pixelated(
            str(path), FILTER_NAMES, self._camera_params(optical_system),
            pixel_size_nm=200.0, min_localisations=2, aggregate_id_column="cluster_id", verbose=False,
        )
        assert np.isnan(out.loc[0, "wl_pixel"])

    def test_aggregate_without_usable_rows_is_skipped(self, nrf, optical_system, tmp_path):
        df = _synthetic_locs_df(12, nrf, optical_system, with_photons=True, with_aggregate=True, seed=16)
        df.loc[df["cluster_id"] == 0.0, "A_R_err"] = np.nan
        path = tmp_path / "locs_agg_bad.h5"
        df.to_hdf(path, key="data", mode="w", format="table")
        out, grid_info = nrf.fit_wavelengths_pixelated(
            str(path), FILTER_NAMES, self._camera_params(optical_system),
            pixel_size_nm=200.0, min_localisations=2, aggregate_id_column="cluster_id", verbose=False,
        )
        assert out.loc[df["cluster_id"] == 0.0, "wl_pixel"].isna().all()
        assert out.loc[df["cluster_id"] == 1.0, "wl_pixel"].notna().any()

    def test_unusable_rows_do_not_count_towards_min_localisations(self, nrf, optical_system, tmp_path):
        df = _synthetic_locs_df(12, nrf, optical_system, with_photons=True, seed=17)
        df["photons"] = np.nan
        path = tmp_path / "locs_pix_bad.h5"
        df.to_hdf(path, key="data", mode="w", format="table")
        out, grid_info = nrf.fit_wavelengths_pixelated(
            str(path), FILTER_NAMES, self._camera_params(optical_system),
            pixel_size_nm=200.0, min_localisations=2, verbose=False,
        )
        assert grid_info["n_pixels_fitted"] == 0

    def test_overlapping_aggregates_accumulate_photon_grid(self, nrf, optical_system, tmp_path):
        # Two different aggregates whose points all land in the same spatial
        # pixel (very large pixel_size_nm) -> total_photons_grid gets written
        # by one aggregate's pixel group, then accumulated by the other's.
        df = _synthetic_locs_df(6, nrf, optical_system, with_photons=True, fov_nm=50.0, seed=18)
        df["cluster_id"] = np.where(np.arange(len(df)) % 2 == 0, 0.0, 1.0)
        path = tmp_path / "locs_overlap.h5"
        df.to_hdf(path, key="data", mode="w", format="table")
        out, grid_info = nrf.fit_wavelengths_pixelated(
            str(path), FILTER_NAMES, self._camera_params(optical_system),
            pixel_size_nm=100000.0, min_localisations=2, aggregate_id_column="cluster_id", verbose=False,
        )
        assert grid_info["total_photons_grid"][~np.isnan(grid_info["total_photons_grid"])].sum() > 0


# ======================================================================
# simulate_wavelength_precision -- heavy orchestrator, only used on the
# developer branch. Stub out the expensive Multicolour_Simulation_Functions
# call so the real Stage 1/Stage 2 orchestration logic (file discovery, CSV
# writing, wavelength conversion) runs against fast, small synthetic data.
# ======================================================================

class TestSimulateWavelengthPrecision:
    def test_stage1_and_stage2_orchestration(self, nrf, tmp_path, monkeypatch):
        import pyS3M.Multicolour_Simulation_Functions as Multicolour_Simulation_Functions

        wavelength_array, pixel_QYs, _ = nrf.setup_optical_system(FILTER_NAMES)

        def fake_test_fit_method(self, dye, filters, wavelength, camera_parameters,
                                  save_folder, n_photon_space, smoothing_function,
                                  starting_flag, config, single_dye_spectrum=None,
                                  nile_red_wavelength=None):
            # Write a tiny raw-results h5 matching what Stage 2 expects to find.
            wc = nrf.wavelength_center_for_peak(nile_red_wavelength, wavelength_array)
            com = nrf.spectral_centre_of_mass(wc, wavelength_array)
            n = 5
            df = pd.DataFrame({
                "wl_fit": np.full(n, com) + np.random.default_rng(0).normal(0, 0.5, n),
                "photon_level": np.zeros(n, dtype=int),
                "photons": np.full(n, float(n_photon_space[0])),
            })
            path = Path(save_folder) / f"{starting_flag}rawresults.h5"
            df.to_hdf(path, key="data", mode="w", format="table")

        from pathlib import Path

        monkeypatch.setattr(
            Multicolour_Simulation_Functions.MultiC_Sim_Funcs, "test_fit_method",
            fake_test_fit_method,
        )
        nrf.simulate_wavelength_precision(
            save_folder=str(tmp_path),
            wavelength_range=(600.0, 605.0),
            wavelength_step=5.0,
            photon_counts=np.array([1000.0]),
            n_bootstrap=5,
            verbose=True,
            save_raw_results=True,
        )
        summary_files = list(tmp_path.glob("*wavelength_precision_summary.csv"))
        assert len(summary_files) == 1

    def test_no_raw_results_produces_no_summary(self, nrf, tmp_path, monkeypatch):
        import pyS3M.Multicolour_Simulation_Functions as Multicolour_Simulation_Functions

        def noop_test_fit_method(self, *a, **kw):
            pass

        monkeypatch.setattr(
            Multicolour_Simulation_Functions.MultiC_Sim_Funcs, "test_fit_method",
            noop_test_fit_method,
        )
        nrf.simulate_wavelength_precision(
            save_folder=str(tmp_path),
            wavelength_range=(600.0, 600.0),
            wavelength_step=5.0,
            photon_counts=np.array([1000.0]),
            n_bootstrap=5,
            verbose=True,
        )
        summary_files = list(tmp_path.glob("*wavelength_precision_summary.csv"))
        assert len(summary_files) == 0

    def test_stage1_progress_and_logging_callbacks_invoked(self, tmp_path, monkeypatch):
        import pyS3M.Multicolour_Simulation_Functions as Multicolour_Simulation_Functions
        from pathlib import Path
        from pyS3M.Constants import AnalysisConfig

        calls = {"progress": 0, "logging": 0}
        config = AnalysisConfig(
            progress_callback=lambda frac, msg: calls.__setitem__("progress", calls["progress"] + 1),
            logging_callback=lambda msg: calls.__setitem__("logging", calls["logging"] + 1),
        )
        nrf_cb = NileRedFunctions.NileRed_Functions(config=config)

        def noop_test_fit_method(self, *a, **kw):
            pass

        monkeypatch.setattr(
            Multicolour_Simulation_Functions.MultiC_Sim_Funcs, "test_fit_method",
            noop_test_fit_method,
        )
        nrf_cb.simulate_wavelength_precision(
            save_folder=str(tmp_path),
            wavelength_range=(600.0, 600.0),
            wavelength_step=5.0,
            photon_counts=np.array([1000.0]),
            n_bootstrap=3,
            verbose=True,
        )
        assert calls["progress"] > 0
        assert calls["logging"] > 0

    def test_use_tqdm_progress_bars(self, nrf, tmp_path, monkeypatch):
        import pyS3M.Multicolour_Simulation_Functions as Multicolour_Simulation_Functions
        from pathlib import Path

        wavelength_array, pixel_QYs, _ = nrf.setup_optical_system(FILTER_NAMES)

        def fake_test_fit_method(self, dye, filters, wavelength, camera_parameters,
                                  save_folder, n_photon_space, smoothing_function,
                                  starting_flag, config, single_dye_spectrum=None,
                                  nile_red_wavelength=None):
            wc = nrf.wavelength_center_for_peak(nile_red_wavelength, wavelength_array)
            com = nrf.spectral_centre_of_mass(wc, wavelength_array)
            df = pd.DataFrame({
                "wl_fit": np.full(3, com),
                "photon_level": np.zeros(3, dtype=int),
                "photons": np.full(3, float(n_photon_space[0])),
            })
            (Path(save_folder) / f"{starting_flag}rawresults.h5").parent.mkdir(parents=True, exist_ok=True)
            df.to_hdf(Path(save_folder) / f"{starting_flag}rawresults.h5", key="data", mode="w", format="table")

        monkeypatch.setattr(
            Multicolour_Simulation_Functions.MultiC_Sim_Funcs, "test_fit_method",
            fake_test_fit_method,
        )
        nrf.simulate_wavelength_precision(
            save_folder=str(tmp_path),
            wavelength_range=(600.0, 605.0),
            wavelength_step=5.0,
            photon_counts=np.array([1000.0]),
            n_bootstrap=3,
            verbose=True,
            use_tqdm=True,
        )
        assert len(list(tmp_path.glob("*wavelength_precision_summary.csv"))) == 1

    def test_use_tqdm_import_error_falls_back(self, nrf, tmp_path, monkeypatch):
        import sys
        import pyS3M.Multicolour_Simulation_Functions as Multicolour_Simulation_Functions

        def noop_test_fit_method(self, *a, **kw):
            pass

        monkeypatch.setattr(
            Multicolour_Simulation_Functions.MultiC_Sim_Funcs, "test_fit_method",
            noop_test_fit_method,
        )
        monkeypatch.setitem(sys.modules, "tqdm.auto", None)
        nrf.simulate_wavelength_precision(
            save_folder=str(tmp_path),
            wavelength_range=(600.0, 600.0),
            wavelength_step=5.0,
            photon_counts=np.array([1000.0]),
            n_bootstrap=3,
            verbose=True,
            use_tqdm=True,
        )

    def test_raw_results_missing_wl_fit_column_warns_and_skips(self, nrf, tmp_path, monkeypatch):
        import pyS3M.Multicolour_Simulation_Functions as Multicolour_Simulation_Functions
        from pathlib import Path

        def fake_test_fit_method(self, dye, filters, wavelength, camera_parameters,
                                  save_folder, n_photon_space, smoothing_function,
                                  starting_flag, config, single_dye_spectrum=None,
                                  nile_red_wavelength=None):
            df = pd.DataFrame({"some_other_column": [1, 2, 3]})
            df.to_hdf(Path(save_folder) / f"{starting_flag}rawresults.h5", key="data", mode="w", format="table")

        monkeypatch.setattr(
            Multicolour_Simulation_Functions.MultiC_Sim_Funcs, "test_fit_method",
            fake_test_fit_method,
        )
        nrf.simulate_wavelength_precision(
            save_folder=str(tmp_path),
            wavelength_range=(600.0, 600.0),
            wavelength_step=5.0,
            photon_counts=np.array([1000.0]),
            n_bootstrap=3,
            verbose=True,
        )
        assert len(list(tmp_path.glob("*wavelength_precision_summary.csv"))) == 0

    def test_raw_results_missing_photon_level_column_warns_and_skips(self, nrf, tmp_path, monkeypatch):
        import pyS3M.Multicolour_Simulation_Functions as Multicolour_Simulation_Functions
        from pathlib import Path

        def fake_test_fit_method(self, dye, filters, wavelength, camera_parameters,
                                  save_folder, n_photon_space, smoothing_function,
                                  starting_flag, config, single_dye_spectrum=None,
                                  nile_red_wavelength=None):
            df = pd.DataFrame({"wl_fit": [617.0, 618.0]})
            df.to_hdf(Path(save_folder) / f"{starting_flag}rawresults.h5", key="data", mode="w", format="table")

        monkeypatch.setattr(
            Multicolour_Simulation_Functions.MultiC_Sim_Funcs, "test_fit_method",
            fake_test_fit_method,
        )
        nrf.simulate_wavelength_precision(
            save_folder=str(tmp_path),
            wavelength_range=(600.0, 600.0),
            wavelength_step=5.0,
            photon_counts=np.array([1000.0]),
            n_bootstrap=3,
            verbose=True,
        )
        assert len(list(tmp_path.glob("*wavelength_precision_summary.csv"))) == 0
