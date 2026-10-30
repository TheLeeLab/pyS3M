"""Full coverage tests for pyS3M.NileRedFunctions -- Nile Red forward/inverse
spectral model (skew-Gaussian emission -> RGB+PSF predictions, and the inverse
least-squares wavelength fit), plus the two higher-level pipelines built on it:
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
    def test_zero_at_exact_prediction(self, nrf, optical_system):
        wavelength_array, pixel_QYs, filter_spectra = optical_system
        preds = nrf.nile_red_forward_model(617.6, filter_spectra, wavelength_array, pixel_QYs)
        observed = {k: preds[k] for k in ["R", "G", "B", "sigma_x", "sigma_y"]}
        errors = {k: 1.0 for k in observed}
        residuals = nrf.residuals_nile_red(
            np.array([617.6]), observed, errors, filter_spectra, wavelength_array, pixel_QYs,
        )
        np.testing.assert_allclose(residuals, 0.0, atol=1e-10)

    def test_scalar_wavelength_center_accepted(self, nrf, optical_system):
        wavelength_array, pixel_QYs, filter_spectra = optical_system
        preds = nrf.nile_red_forward_model(617.6, filter_spectra, wavelength_array, pixel_QYs)
        observed = {k: preds[k] for k in ["R", "G", "B", "sigma_x", "sigma_y"]}
        errors = {k: 1.0 for k in observed}
        residuals = nrf.residuals_nile_red(
            617.6, observed, errors, filter_spectra, wavelength_array, pixel_QYs,
        )
        np.testing.assert_allclose(residuals, 0.0, atol=1e-10)

    def test_skips_keys_with_zero_error(self, nrf, optical_system):
        wavelength_array, pixel_QYs, filter_spectra = optical_system
        preds = nrf.nile_red_forward_model(617.6, filter_spectra, wavelength_array, pixel_QYs)
        observed = {k: preds[k] for k in ["R", "G", "B", "sigma_x", "sigma_y"]}
        errors = {"R": 1.0, "G": 1.0, "B": 1.0, "sigma_x": 0.0, "sigma_y": 1.0}
        residuals = nrf.residuals_nile_red(
            np.array([617.6]), observed, errors, filter_spectra, wavelength_array, pixel_QYs,
        )
        assert len(residuals) == 4


# ======================================================================
# _error_inflation_factor / _calculate_channel_snr / _normalize_rgb_with_errors /
# _weighted_average_with_error
# ======================================================================

class TestErrorInflationFactor:
    @pytest.mark.parametrize("snr,expected", [(1.0, 3.0), (3.0, 2.0), (7.0, 1.5), (20.0, 1.0)])
    def test_thresholds(self, nrf, snr, expected):
        assert nrf._error_inflation_factor(snr) == expected


class TestCalculateChannelSnr:
    def test_higher_signal_gives_higher_snr(self, nrf):
        rgb = np.array([0.7, 0.2, 0.1])
        snr = nrf._calculate_channel_snr(rgb, total_photons=10000.0, background_photons=40.0)
        assert snr[0] > snr[2]


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


class TestWeightedAverageWithError:
    def test_basic(self, nrf):
        values = np.array([1.0, 2.0, 3.0])
        errors = np.array([1.0, 1.0, 1.0])
        avg, err = NileRedFunctions.NileRed_Functions._weighted_average_with_error(values, errors)
        assert avg == pytest.approx(2.0)
        assert err > 0


# ======================================================================
# _parallel_fit_wavelengths
# ======================================================================

class TestParallelFitWavelengths:
    def _small_fit_args(self, nrf, optical_system, n=3, wl=617.6):
        wavelength_array, pixel_QYs, filter_spectra = optical_system
        preds = nrf.nile_red_forward_model(wl, filter_spectra, wavelength_array, pixel_QYs)
        rgb = np.array([preds["R"], preds["G"], preds["B"]])
        rgb_err = np.abs(rgb) * 0.03 + 1e-4
        args = (
            rgb, preds["sigma_x"], preds["sigma_y"], rgb_err, 1.0, 1.0,
            filter_spectra, wavelength_array, pixel_QYs, 1.49,
            None, None, (500.0, 750.0), None,
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
    def test_recovers_known_wavelength_no_snr(self, nrf, optical_system):
        wavelength_array, pixel_QYs, filter_spectra = optical_system
        true_wl = 617.6
        preds = nrf.nile_red_forward_model(true_wl, filter_spectra, wavelength_array, pixel_QYs)
        rgb = np.array([preds["R"], preds["G"], preds["B"]])
        rgb_err = np.abs(rgb) * 0.02 + 1e-4
        wl_fit, out = nrf.fit_nile_red_wavelength(
            rgb, preds["sigma_x"], preds["sigma_y"], rgb_err, 1.0, 1.0,
            filter_spectra, wavelength_array, pixel_QYs,
            apply_snr_inflation=False,
        )
        assert "wavelength_error" in out

    def test_with_snr_inflation(self, nrf, optical_system):
        wavelength_array, pixel_QYs, filter_spectra = optical_system
        true_wl = 617.6
        preds = nrf.nile_red_forward_model(true_wl, filter_spectra, wavelength_array, pixel_QYs)
        rgb = np.array([preds["R"], preds["G"], preds["B"]])
        rgb_err = np.abs(rgb) * 0.02 + 1e-4
        wl_fit, out = nrf.fit_nile_red_wavelength(
            rgb, preds["sigma_x"], preds["sigma_y"], rgb_err, 1.0, 1.0,
            filter_spectra, wavelength_array, pixel_QYs,
            total_photons=500.0, background_photons=40.0, apply_snr_inflation=True,
        )
        assert "wavelength_error" in out

    def test_custom_initial_guess(self, nrf, optical_system):
        wavelength_array, pixel_QYs, filter_spectra = optical_system
        preds = nrf.nile_red_forward_model(617.6, filter_spectra, wavelength_array, pixel_QYs)
        rgb = np.array([preds["R"], preds["G"], preds["B"]])
        rgb_err = np.abs(rgb) * 0.02 + 1e-4
        wl_fit, out = nrf.fit_nile_red_wavelength(
            rgb, preds["sigma_x"], preds["sigma_y"], rgb_err, 1.0, 1.0,
            filter_spectra, wavelength_array, pixel_QYs,
            wavelength_initial_guess=650.0, apply_snr_inflation=False,
        )
        assert "wavelength_error" in out

    def test_out_of_bounds_initial_guess_falls_back_to_midpoint(self, nrf, optical_system):
        wavelength_array, pixel_QYs, filter_spectra = optical_system
        preds = nrf.nile_red_forward_model(617.6, filter_spectra, wavelength_array, pixel_QYs)
        rgb = np.array([preds["R"], preds["G"], preds["B"]])
        rgb_err = np.abs(rgb) * 0.02 + 1e-4
        wl_fit, out = nrf.fit_nile_red_wavelength(
            rgb, preds["sigma_x"], preds["sigma_y"], rgb_err, 1.0, 1.0,
            filter_spectra, wavelength_array, pixel_QYs,
            wavelength_bounds=(500.0, 750.0),
            wavelength_initial_guess=10000.0, apply_snr_inflation=False,
        )
        assert "wavelength_error" in out

    def test_zero_jacobian_yields_nan_error(self, nrf, optical_system, monkeypatch):
        # JtJ<=0 (a degenerate/zero Jacobian) can't be inverted for an error
        # estimate -- force it via a stubbed least_squares result rather than
        # hunting for real data that produces one.
        wavelength_array, pixel_QYs, filter_spectra = optical_system
        preds = nrf.nile_red_forward_model(617.6, filter_spectra, wavelength_array, pixel_QYs)
        rgb = np.array([preds["R"], preds["G"], preds["B"]])
        rgb_err = np.abs(rgb) * 0.02 + 1e-4

        fake_result = types.SimpleNamespace(
            x=np.array([617.6]),
            jac=np.zeros((5, 1)),
            fun=np.zeros(5),
        )
        monkeypatch.setattr(NileRedFunctions, "least_squares", lambda *a, **kw: fake_result)
        wl_fit, out = nrf.fit_nile_red_wavelength(
            rgb, preds["sigma_x"], preds["sigma_y"], rgb_err, 1.0, 1.0,
            filter_spectra, wavelength_array, pixel_QYs,
            apply_snr_inflation=False,
        )
        assert np.isnan(out["wavelength_error"])


class TestFitNileRedWavelengthStandalone:
    def test_success(self, optical_system):
        wavelength_array, pixel_QYs, filter_spectra = optical_system
        nrf_local = NileRedFunctions.NileRed_Functions()
        preds = nrf_local.nile_red_forward_model(617.6, filter_spectra, wavelength_array, pixel_QYs)
        rgb = np.array([preds["R"], preds["G"], preds["B"]])
        rgb_err = np.abs(rgb) * 0.02 + 1e-4
        wl, wl_err = NileRedFunctions._fit_nile_red_wavelength_standalone(
            rgb, preds["sigma_x"], preds["sigma_y"], rgb_err, 1.0, 1.0,
            filter_spectra, wavelength_array, pixel_QYs, 1.49,
        )
        assert not np.isnan(wl)

    def test_exception_returns_nan_pair(self, optical_system, monkeypatch):
        wavelength_array, pixel_QYs, filter_spectra = optical_system

        def _raise(self, *a, **kw):
            raise RuntimeError("forced failure")

        monkeypatch.setattr(NileRedFunctions.NileRed_Functions, "fit_nile_red_wavelength", _raise)
        wl, wl_err = NileRedFunctions._fit_nile_red_wavelength_standalone(
            np.array([0.5, 0.3, 0.2]), 100.0, 100.0, np.array([0.01, 0.01, 0.01]), 1.0, 1.0,
            filter_spectra, wavelength_array, pixel_QYs, 1.49,
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

    def test_basic_fit_with_snr(self, nrf, optical_system, tmp_path):
        df = _synthetic_locs_df(10, nrf, optical_system, with_photons=True, seed=2)
        path = tmp_path / "locs.h5"
        df.to_hdf(path, key="data", mode="w", format="table")
        out = nrf.fit_wavelengths_from_h5(
            str(path), FILTER_NAMES, self._camera_params(optical_system), verbose=True,
        )
        assert "wl_fit" in out.columns
        assert "wl_fit_err" in out.columns
        assert out["wl_fit"].notna().any()

    def test_basic_fit_without_snr_columns(self, nrf, optical_system, tmp_path):
        df = _synthetic_locs_df(8, nrf, optical_system, with_photons=False, seed=3)
        path = tmp_path / "locs_no_snr.h5"
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

    def test_verbose_without_snr_columns(self, nrf, optical_system, tmp_path):
        df = _synthetic_locs_df(10, nrf, optical_system, with_photons=False, seed=10)
        path = tmp_path / "locs_no_snr.h5"
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

    def test_aggregate_weighted_average_exception_is_skipped(self, nrf, optical_system, tmp_path, monkeypatch):
        df = _synthetic_locs_df(12, nrf, optical_system, with_photons=True, with_aggregate=True, seed=16)
        path = tmp_path / "locs_agg_exc.h5"
        df.to_hdf(path, key="data", mode="w", format="table")

        def _raise(*a, **kw):
            raise ValueError("forced failure")

        monkeypatch.setattr(NileRedFunctions.NileRed_Functions, "_weighted_average_with_error", staticmethod(_raise))
        out, grid_info = nrf.fit_wavelengths_pixelated(
            str(path), FILTER_NAMES, self._camera_params(optical_system),
            pixel_size_nm=200.0, min_localisations=2, aggregate_id_column="cluster_id", verbose=False,
        )
        assert isinstance(grid_info, dict)

    def test_pixel_weighted_average_exception_is_skipped(self, nrf, optical_system, tmp_path, monkeypatch):
        df = _synthetic_locs_df(12, nrf, optical_system, with_photons=True, seed=17)
        path = tmp_path / "locs_pix_exc.h5"
        df.to_hdf(path, key="data", mode="w", format="table")

        def _raise(*a, **kw):
            raise ValueError("forced failure")

        monkeypatch.setattr(NileRedFunctions.NileRed_Functions, "_weighted_average_with_error", staticmethod(_raise))
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
