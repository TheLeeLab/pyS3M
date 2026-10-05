"""Full coverage tests for pyS3M.SpectralFunctions -- wavelength/energy conversions,
Gaussian/skew-Gaussian spectral models, database-driven dye/filter spectral data
(real duckdb-backed queries, not mocked), camera pixel-efficiency loading, and the
stochastic photon-sampling / colour-ratio machinery used by the simulation pipeline.

Real (small, but not literally-tiny) data throughout: real dye/filter names queried
from the actual bundled `Spectra/spectral_data.duckdb`, a real camera QE CSV via
`getpixelefficiency()`, and small (40-100 point) wavelength grids -- the physics here
(skew-Gaussian energy-space fitting, QE-weighted stochastic photon assignment) isn't
meaningful on literal 4-6 element arrays, matching the `NileRedFunctions.py`/
`mixture_analysis.py` precedent from this same coverage push. `differential_evolution`
in `spectral_fit_dye` converges in well under a second on real ~60-point dye spectra,
so no monkeypatching was needed for speed there.
"""
from __future__ import annotations

import numpy as np
import pytest

import pyS3M.SpectralFunctions as SF
from pyS3M.SpectralFunctions import (
    SpectralDataType,
    SpectralConstants,
    DatabaseQueryHandler,
    SpectrumProcessor,
    DyeSpectrumProcessor,
    FilterSpectrumProcessor,
    Spectral_Funcs,
    _assign_photons_to_channels_jit,
    _process_bootstrap_samples_parallel,
    _find_spectra_dir,
)

FILTER_NAMES = [
    "semrock-ff01-650-200",
    "semrock-di03-r514-t1-25x36",
    "semrock-ff01-515-lp",
]
DYE_NAME = "ATTO 565"
DYE_NAME_2 = "ATTO 647"


@pytest.fixture(scope="module")
def sf():
    return Spectral_Funcs()


@pytest.fixture(scope="module")
def wl():
    return np.linspace(500.0, 700.0, 60)


@pytest.fixture(scope="module")
def pixel_qys(sf):
    R, G, B, wavelength = sf.getpixelefficiency()
    return np.vstack([B, G, R]), wavelength


# ======================================================================
# _find_spectra_dir
# ======================================================================

class TestFindSpectraDir:
    def test_returns_existing_directory(self):
        path = _find_spectra_dir()
        assert path.exists()
        assert path.is_dir()


# ======================================================================
# DatabaseQueryHandler
# ======================================================================

class TestDatabaseQueryHandler:
    def _handler(self, sf):
        return sf.db_handler

    def test_get_available_names_dye(self, sf):
        names = self._handler(sf).get_available_names(SpectralDataType.DYE)
        assert DYE_NAME in names

    def test_get_available_names_filter(self, sf):
        names = self._handler(sf).get_available_names(SpectralDataType.FILTER)
        assert FILTER_NAMES[0] in names

    def test_query_spectral_data_single_name(self, sf):
        df = self._handler(sf).query_spectral_data([DYE_NAME], SpectralDataType.DYE)
        assert len(df) > 0
        assert "wavelength_nm" in df.columns

    def test_query_spectral_data_multiple_names(self, sf):
        df = self._handler(sf).query_spectral_data(
            [DYE_NAME, DYE_NAME_2], SpectralDataType.DYE
        )
        assert len(df) > 0
        assert set(df["dye_name"].unique()) <= {DYE_NAME, DYE_NAME_2}


class TestSingleNameQueryCache:
    """Single-name lookups (what `get_spectral_data` actually issues, one name
    per call) are routed through a module-level `lru_cache` keyed on db_path --
    see `_query_single_name_cached`. This exists because simulation code
    routinely constructs a fresh `Spectral_Funcs()` per call, so an
    instance-bound cache would miss every time; keying on db_path instead
    lets repeated dye/filter lookups across those rebuilds hit the cache.
    """

    def test_repeat_query_is_a_cache_hit(self, sf):
        SF._query_single_name_cached.cache_clear()
        sf.db_handler.query_spectral_data([DYE_NAME], SpectralDataType.DYE)
        misses_after_first = SF._query_single_name_cached.cache_info().misses

        sf.db_handler.query_spectral_data([DYE_NAME], SpectralDataType.DYE)
        info = SF._query_single_name_cached.cache_info()

        assert misses_after_first == 1
        assert info.hits == 1
        assert info.misses == 1

    def test_cache_hit_returns_equivalent_data_to_a_fresh_instance(self, sf):
        # A brand-new Spectral_Funcs() (as simulation code repeatedly constructs)
        # must see the same cached data as the original instance -- the cache is
        # keyed on db_path, not on `self`.
        sf2 = Spectral_Funcs()
        df1 = sf.db_handler.query_spectral_data([DYE_NAME], SpectralDataType.DYE)
        df2 = sf2.db_handler.query_spectral_data([DYE_NAME], SpectralDataType.DYE)
        np.testing.assert_array_equal(
            df1["wavelength_nm"].to_numpy(), df2["wavelength_nm"].to_numpy()
        )

    def test_returned_dataframe_is_a_copy_of_the_cached_object(self, sf):
        # Mutating one caller's result must not corrupt what later callers see.
        df1 = sf.db_handler.query_spectral_data([DYE_NAME], SpectralDataType.DYE)
        original_value = df1["wavelength_nm"].to_numpy()[0]
        df1.iloc[0, df1.columns.get_loc("wavelength_nm")] = -999999.0

        df2 = sf.db_handler.query_spectral_data([DYE_NAME], SpectralDataType.DYE)
        assert df2["wavelength_nm"].to_numpy()[0] == original_value


# ======================================================================
# SpectrumProcessor / DyeSpectrumProcessor / FilterSpectrumProcessor
# ======================================================================

class TestSpectrumProcessorAbstractBody:
    def test_base_method_body_is_a_noop(self):
        instance = DyeSpectrumProcessor()
        result = SpectrumProcessor.process_spectrum(instance, None, None)
        assert result is None


class TestDyeSpectrumProcessor:
    def test_interpolates_and_normalizes(self, sf, wl):
        raw = sf.db_handler.query_spectral_data([DYE_NAME], SpectralDataType.DYE)
        out = DyeSpectrumProcessor().process_spectrum(raw, wl)
        assert out.shape == wl.shape
        assert np.trapz(out) == pytest.approx(np.sum(out)) or np.sum(out) > 0

    def test_all_zero_total_intensity_skips_normalisation(self):
        import polars as pl

        raw = pl.DataFrame({
            "wavelength_nm": [500.0, 600.0, 700.0],
            "emission_intensity": [0.0, 0.0, 0.0],
        })
        out = DyeSpectrumProcessor().process_spectrum(raw, np.linspace(500, 700, 10))
        np.testing.assert_allclose(out, 0.0)


class TestFilterSpectrumProcessor:
    def test_interpolates(self, sf, wl):
        raw = sf.db_handler.query_spectral_data([FILTER_NAMES[0]], SpectralDataType.FILTER)
        out = FilterSpectrumProcessor().process_spectrum(raw, wl)
        assert out.shape == wl.shape
        assert np.all(out >= 0)


# ======================================================================
# _assign_photons_to_channels_jit / _process_bootstrap_samples_parallel
# ======================================================================

class TestAssignPhotonsToChannelsJit:
    def _inputs(self):
        rng = np.random.default_rng(0)
        n = 200
        p_0 = rng.uniform(0.1, 0.4, n)
        p_1 = rng.uniform(0.1, 0.4, n)
        u = rng.uniform(0, 1, n)
        return p_0, p_1, u

    def test_jit_matches_py_func(self):
        p_0, p_1, u = self._inputs()
        jit_result = _assign_photons_to_channels_jit(p_0, p_1, u)
        py_result = _assign_photons_to_channels_jit.py_func(p_0, p_1, u)
        assert jit_result == py_result
        assert sum(jit_result) == len(p_0)


class TestProcessBootstrapSamplesParallel:
    def _inputs(self):
        rng = np.random.default_rng(1)
        n_bootstrap, n_photons, n_ch, n_lut = 3, 20, 3, 50
        photon_wl_bootstrap = rng.uniform(500, 700, (n_bootstrap, n_photons))
        lut_wavelengths = np.linspace(500, 700, n_lut)
        lut_qe = rng.uniform(0.1, 0.9, (n_ch, n_lut))
        uniform_randoms = rng.uniform(0, 1, (n_bootstrap, n_photons))
        return photon_wl_bootstrap, lut_wavelengths, lut_qe, uniform_randoms

    def test_jit_runs(self):
        args = self._inputs()
        mean_wl, counts, mean_qe = _process_bootstrap_samples_parallel(*args)
        assert mean_wl.shape == (3,)
        assert counts.shape == (3, 3)

    def test_py_func_matches_jit_shapes(self):
        args = self._inputs()
        jit_out = _process_bootstrap_samples_parallel(*args)
        py_out = _process_bootstrap_samples_parallel.py_func(*args)
        for jit_arr, py_arr in zip(jit_out, py_out):
            np.testing.assert_allclose(jit_arr, py_arr)

    def test_lut_edge_wavelengths(self):
        # Photon wavelengths outside the LUT range exercise the idx==0 and
        # idx>=n_lut clamping branches.
        rng = np.random.default_rng(2)
        n_lut = 20
        lut_wavelengths = np.linspace(550.0, 650.0, n_lut)
        lut_qe = rng.uniform(0.1, 0.9, (3, n_lut))
        photon_wl_bootstrap = np.array([[500.0, 700.0, 600.0]])  # below, above, inside
        uniform_randoms = rng.uniform(0, 1, (1, 3))
        mean_wl, counts, mean_qe = _process_bootstrap_samples_parallel(
            photon_wl_bootstrap, lut_wavelengths, lut_qe, uniform_randoms
        )
        assert mean_wl.shape == (1,)

    def test_lut_edge_wavelengths_py_func(self):
        # coverage.py can't trace inside JIT-compiled machine code -- the
        # idx==0/idx>=n_lut branches above only actually count as covered via
        # the uncompiled .py_func path, same pattern as test_localise.py/
        # test_render.py/test_gaussoptfuncs.py elsewhere in this codebase.
        rng = np.random.default_rng(2)
        n_lut = 20
        lut_wavelengths = np.linspace(550.0, 650.0, n_lut)
        lut_qe = rng.uniform(0.1, 0.9, (3, n_lut))
        photon_wl_bootstrap = np.array([[500.0, 700.0, 600.0]])  # below, above, inside
        uniform_randoms = rng.uniform(0, 1, (1, 3))
        mean_wl, counts, mean_qe = _process_bootstrap_samples_parallel.py_func(
            photon_wl_bootstrap, lut_wavelengths, lut_qe, uniform_randoms
        )
        assert mean_wl.shape == (1,)

    def test_near_zero_total_qe_clamped_py_func(self):
        # A photon whose total QE across all channels sums to ~0 exercises
        # the tq<1e-10 clamp guarding the cumulative-probability division.
        n_lut = 10
        lut_wavelengths = np.linspace(550.0, 650.0, n_lut)
        lut_qe = np.zeros((3, n_lut))
        photon_wl_bootstrap = np.array([[600.0]])
        uniform_randoms = np.array([[0.5]])
        mean_wl, counts, mean_qe = _process_bootstrap_samples_parallel.py_func(
            photon_wl_bootstrap, lut_wavelengths, lut_qe, uniform_randoms
        )
        assert mean_qe[0] == pytest.approx(0.0)
        assert counts.sum() == 1


# ======================================================================
# Spectral_Funcs.__init__ / getpixelefficiency / getobjectiveefficiency
# ======================================================================

class TestSpectralFuncsInit:
    def test_ximea_and_zwo_both_construct(self):
        sf_x = Spectral_Funcs(camera="ximea")
        sf_z = Spectral_Funcs(camera="zwo")
        assert len(sf_x.dye_names) > 0
        assert len(sf_z.filter_names) > 0


class TestGetPixelEfficiency:
    def test_default_file(self, sf):
        R, G, B, wavelength = sf.getpixelefficiency()
        assert R.shape == G.shape == B.shape == wavelength.shape

    def test_missing_file_raises_file_not_found(self, sf, tmp_path):
        with pytest.raises(FileNotFoundError):
            sf.getpixelefficiency(filename=str(tmp_path / "does_not_exist.csv"))

    def test_missing_columns_raises_value_error(self, sf, tmp_path):
        path = tmp_path / "bad_qe.csv"
        path.write_text("wavelength,R,G\n500,0.1,0.2\n")
        with pytest.raises(ValueError, match="Missing required columns"):
            sf.getpixelefficiency(filename=str(path))


class TestGetObjectiveEfficiency:
    def test_default_file(self):
        wavelength = np.linspace(500, 700, 20)
        out = Spectral_Funcs.getobjectiveefficiency(wavelength)
        assert out.shape == wavelength.shape

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Spectral_Funcs.getobjectiveefficiency(
                np.linspace(500, 700, 10), filename=str(tmp_path / "nope.csv")
            )

    def test_missing_columns_raises_value_error(self, tmp_path):
        path = tmp_path / "bad_obj.csv"
        path.write_text("wavelength,notthecolumn\n500,0.9\n")
        with pytest.raises(ValueError, match="Missing required columns"):
            Spectral_Funcs.getobjectiveefficiency(
                np.linspace(500, 700, 10), filename=str(path)
            )


# ======================================================================
# fwhm_sigma_conversion / moment_calculations / wavelength_to_energy
# ======================================================================

class TestFwhmSigmaConversion:
    def test_sigma_to_fwhm(self, sf):
        assert sf.fwhm_sigma_conversion(1.0, sigma_given=True) == pytest.approx(
            SpectralConstants.FWHM_TO_SIGMA_FACTOR
        )

    def test_fwhm_to_sigma(self, sf):
        assert sf.fwhm_sigma_conversion(1.0, sigma_given=False) == pytest.approx(
            1.0 / SpectralConstants.FWHM_TO_SIGMA_FACTOR
        )


class TestMomentCalculations:
    def _gaussian_data(self):
        x = np.linspace(-5, 5, 200)
        fx = np.exp(-0.5 * x**2)
        return x, fx

    def test_order_1(self, sf):
        x, fx = self._gaussian_data()
        moments = sf.moment_calculations(x, fx, order=1)
        assert len(moments) == 2
        assert moments[1] == pytest.approx(0.0, abs=1e-6)

    def test_order_2(self, sf):
        x, fx = self._gaussian_data()
        moments = sf.moment_calculations(x, fx, order=2)
        assert len(moments) == 3
        assert moments[2] == pytest.approx(1.0, abs=0.05)

    def test_order_3(self, sf):
        x, fx = self._gaussian_data()
        moments = sf.moment_calculations(x, fx, order=3)
        assert len(moments) == 3

    def test_order_4(self, sf):
        x, fx = self._gaussian_data()
        moments = sf.moment_calculations(x, fx, order=4)
        assert len(moments) == 4

    def test_order_0(self, sf):
        x, fx = self._gaussian_data()
        moments = sf.moment_calculations(x, fx, order=0)
        assert len(moments) == 1


class TestWavelengthToEnergy:
    def test_known_value(self, sf):
        energy = sf.wavelength_to_energy(np.array([1239.84198]))
        assert energy[0] == pytest.approx(1.0, abs=1e-3)


# ======================================================================
# gaussian_model / skew_gaussian_model / chi2_spectrum
# ======================================================================

class TestGaussianModel:
    def test_normal(self, sf):
        x = np.linspace(-5, 5, 50)
        y = sf.gaussian_model(np.array([1.0, 0.0, 1.0]), x)
        assert y.max() > 0

    def test_zero_sigma_returns_zeros(self, sf):
        x = np.linspace(-5, 5, 50)
        y = sf.gaussian_model(np.array([1.0, 0.0, 0.0]), x)
        np.testing.assert_allclose(y, 0.0)


class TestSkewGaussianModel:
    def test_normal(self, sf):
        x = np.linspace(-5, 5, 50)
        y = sf.skew_gaussian_model(np.array([1.0, 0.0, 1.0, 2.0]), x)
        assert y.shape == x.shape

    def test_zero_sigma_returns_zeros(self, sf):
        x = np.linspace(-5, 5, 50)
        y = sf.skew_gaussian_model(np.array([1.0, 0.0, 0.0, 2.0]), x)
        np.testing.assert_allclose(y, 0.0)


class TestChi2Spectrum:
    def test_gaussian_model(self, sf, wl):
        energy = sf.wavelength_to_energy(wl)
        params = np.array([1.0, np.mean(energy), 0.1])
        spectrum = sf.chi2_spectrum(params, wl, np.zeros_like(wl), model="gaussian", return_fit=True)
        residuals = sf.chi2_spectrum(params, wl, spectrum, model="gaussian")
        np.testing.assert_allclose(residuals, 0.0, atol=1e-8)

    def test_skew_gaussian_model_with_weights(self, sf, wl):
        energy = sf.wavelength_to_energy(wl)
        params = np.array([1.0, np.mean(energy), 0.1, 1.0])
        spectrum = sf.chi2_spectrum(params, wl, np.zeros_like(wl), model="skew-gaussian", return_fit=True)
        weights = np.ones_like(wl)
        residuals = sf.chi2_spectrum(params, wl, spectrum, model="skew-gaussian", weights=weights)
        np.testing.assert_allclose(residuals, 0.0, atol=1e-8)

    def test_invalid_model_raises(self, sf, wl):
        with pytest.raises(ValueError, match="Unsupported model type"):
            sf.chi2_spectrum(np.array([1.0, 1.0, 1.0]), wl, np.zeros_like(wl), model="bogus")


# ======================================================================
# spectral_fit_dye
# ======================================================================

class TestSpectralFitDye:
    def test_gaussian_fit(self, sf, wl):
        spectrum = sf.get_spectral_data(DYE_NAME, wl, SpectralDataType.DYE)[0]
        result = sf.spectral_fit_dye(spectrum, wl, model="gaussian")
        assert result.x.shape == (3,)

    def test_skew_gaussian_fit_with_display(self, sf, wl):
        spectrum = sf.get_spectral_data(DYE_NAME, wl, SpectralDataType.DYE)[0]
        result, fitted = sf.spectral_fit_dye(spectrum, wl, model="skew-gaussian", display=True)
        assert fitted.shape == wl.shape

    def test_invalid_model_raises(self, sf, wl):
        spectrum = sf.get_spectral_data(DYE_NAME, wl, SpectralDataType.DYE)[0]
        with pytest.raises(ValueError, match="Unsupported model type"):
            sf.spectral_fit_dye(spectrum, wl, model="bogus")

    def test_all_zero_spectrum_skips_normalisation(self, sf, wl):
        result = sf.spectral_fit_dye(np.zeros_like(wl), wl, model="gaussian")
        assert result.x.shape == (3,)


# ======================================================================
# get_pixel_fractions_rawspectra
# ======================================================================

class TestGetPixelFractionsRawspectra:
    def test_single_spectrum_1d(self, sf, pixel_qys):
        pixel_QYs, wavelength = pixel_qys
        spectrum = sf.get_spectral_data(DYE_NAME, wavelength, SpectralDataType.DYE)[0]
        avg_wl, eff = sf.get_pixel_fractions_rawspectra(spectrum, wavelength, pixel_QYs)
        assert np.isscalar(avg_wl) or avg_wl.shape == ()
        assert eff.shape == (3,)

    def test_multi_spectrum_2d(self, sf, pixel_qys):
        pixel_QYs, wavelength = pixel_qys
        spectra = sf.get_spectral_data([DYE_NAME, DYE_NAME_2], wavelength, SpectralDataType.DYE)
        avg_wl, eff = sf.get_pixel_fractions_rawspectra(spectra, wavelength, pixel_QYs)
        assert avg_wl.shape == (2,)
        assert eff.shape == (2, 3)


# ======================================================================
# get_pixel_fractions_dye_and_filters
# ======================================================================

class TestGetPixelFractionsDyeAndFilters:
    def test_normalized_with_filters(self, sf, pixel_qys):
        pixel_QYs, wavelength = pixel_qys
        avg_wl, ratios = sf.get_pixel_fractions_dye_and_filters(
            [DYE_NAME], FILTER_NAMES, wavelength, pixel_QYs, normalized=True,
        )
        assert ratios.shape == (3,)
        assert ratios.sum() == pytest.approx(1.0, abs=1e-6)

    def test_unnormalized_no_filters(self, sf, pixel_qys):
        pixel_QYs, wavelength = pixel_qys
        avg_wl, qy = sf.get_pixel_fractions_dye_and_filters(
            [DYE_NAME, DYE_NAME_2], None, wavelength, pixel_QYs, normalized=False,
        )
        assert qy.shape == (2, 3)


# ======================================================================
# get_absolute_pixel_QYs
# ======================================================================

class TestGetAbsolutePixelQYs:
    def test_with_objective(self, sf, pixel_qys):
        pixel_QYs, wavelength = pixel_qys
        avg_wl, qy_per_ch, total_qy = sf.get_absolute_pixel_QYs(
            [DYE_NAME], FILTER_NAMES, wavelength, pixel_QYs, include_objective=True,
        )
        assert qy_per_ch.shape == (3,)

    def test_without_objective_multi_dye(self, sf, pixel_qys):
        pixel_QYs, wavelength = pixel_qys
        avg_wl, qy_per_ch, total_qy = sf.get_absolute_pixel_QYs(
            [DYE_NAME, DYE_NAME_2], None, wavelength, pixel_QYs, include_objective=False,
        )
        assert qy_per_ch.shape == (2, 3)
        assert total_qy.shape == (2,)


# ======================================================================
# get_spectral_data / get_dye_or_filter_data
# ======================================================================

class TestGetSpectralData:
    def test_single_string_name(self, sf, wl):
        out = sf.get_spectral_data(DYE_NAME, wl, SpectralDataType.DYE)
        assert out.shape == (1, len(wl))

    def test_list_of_names(self, sf, wl):
        out = sf.get_spectral_data([DYE_NAME, DYE_NAME_2], wl, SpectralDataType.DYE)
        assert out.shape == (2, len(wl))

    def test_invalid_name_raises(self, sf, wl):
        with pytest.raises(ValueError, match="not in database"):
            sf.get_spectral_data(["not_a_real_dye_xyz"], wl, SpectralDataType.DYE)

    def test_per_item_exception_is_caught_and_skipped(self, sf, wl, monkeypatch):
        def _raise(*a, **kw):
            raise RuntimeError("forced failure")

        monkeypatch.setattr(DyeSpectrumProcessor, "process_spectrum", _raise)
        out = sf.get_spectral_data([DYE_NAME], wl, SpectralDataType.DYE)
        np.testing.assert_allclose(out, 0.0)


class TestGetDyeOrFilterData:
    def test_dye_branch(self, sf, wl):
        out = sf.get_dye_or_filter_data(DYE_NAME, wl, dye_or_filter=True)
        assert out.shape == (1, len(wl))

    def test_filter_branch(self, sf, wl):
        out = sf.get_dye_or_filter_data(FILTER_NAMES[0], wl, dye_or_filter=False)
        assert out.shape == (1, len(wl))


# ======================================================================
# sample_photons_from_spectrum
# ======================================================================

class TestSamplePhotonsFromSpectrum:
    def test_normal(self, sf, wl):
        spectrum = sf.get_spectral_data(DYE_NAME, wl, SpectralDataType.DYE)[0]
        rng = np.random.default_rng(0)
        photons = sf.sample_photons_from_spectrum(spectrum, wl, n_photons=100, random_state=rng)
        assert len(photons) == 100
        assert wl.min() <= photons.min() and photons.max() <= wl.max()

    def test_default_random_state(self, sf, wl):
        spectrum = sf.get_spectral_data(DYE_NAME, wl, SpectralDataType.DYE)[0]
        photons = sf.sample_photons_from_spectrum(spectrum, wl, n_photons=10)
        assert len(photons) == 10

    def test_all_zero_spectrum_raises(self, sf, wl):
        with pytest.raises(ValueError, match="no positive values"):
            sf.sample_photons_from_spectrum(np.zeros_like(wl), wl, n_photons=10)


# ======================================================================
# _create_qe_lut / _lookup_qe_vectorized
# ======================================================================

class TestCreateQeLut:
    def test_shapes(self, sf, wl, pixel_qys):
        pixel_QYs, wavelength = pixel_qys
        lut_wl, lut_qe = sf._create_qe_lut(wavelength, pixel_QYs, grid_spacing=1.0)
        assert lut_qe.shape[0] == pixel_QYs.shape[0]
        assert len(lut_wl) == lut_qe.shape[1]


class TestLookupQeVectorized:
    def test_matches_direct_interp(self, sf, wl, pixel_qys):
        pixel_QYs, wavelength = pixel_qys
        lut_wl, lut_qe = sf._create_qe_lut(wavelength, pixel_QYs, grid_spacing=0.5)
        photon_wls = np.array([550.0, 600.0, 650.0])
        looked_up = sf._lookup_qe_vectorized(photon_wls, lut_wl, lut_qe)
        assert looked_up.shape == (pixel_QYs.shape[0], 3)


# ======================================================================
# calculate_colourratio_from_photon_wavelengths
# ======================================================================

class TestCalculateColourRatioFromPhotonWavelengths:
    def test_three_channel_jit_path(self, sf, wl, pixel_qys):
        pixel_QYs, wavelength = pixel_qys
        spectrum = sf.get_spectral_data(DYE_NAME, wl, SpectralDataType.DYE)[0]
        photons = sf.sample_photons_from_spectrum(spectrum, wl, n_photons=200, random_state=np.random.default_rng(0))
        mean_wl, ratios = sf.calculate_colourratio_from_photon_wavelengths(
            photons, wavelength, pixel_QYs,
        )
        assert ratios.shape == (3,)
        assert ratios.sum() == pytest.approx(1.0)

    def test_three_channel_with_qe_lut(self, sf, wl, pixel_qys):
        pixel_QYs, wavelength = pixel_qys
        lut = sf._create_qe_lut(wavelength, pixel_QYs, grid_spacing=0.5)
        spectrum = sf.get_spectral_data(DYE_NAME, wl, SpectralDataType.DYE)[0]
        photons = sf.sample_photons_from_spectrum(spectrum, wl, n_photons=200, random_state=np.random.default_rng(0))
        mean_wl, counts, total_qe = sf.calculate_colourratio_from_photon_wavelengths(
            photons, wavelength, pixel_QYs, return_counts=True, return_total_qe=True, qe_lut=lut,
        )
        assert counts.sum() == 200
        assert total_qe > 0

    def test_general_n_channel_path(self, sf, wl):
        rng = np.random.default_rng(3)
        n_ch = 4
        pixel_QYs = rng.uniform(0.1, 0.9, (n_ch, len(wl)))
        photons = rng.uniform(500, 700, 150)
        mean_wl, ratios = sf.calculate_colourratio_from_photon_wavelengths(
            photons, wl, pixel_QYs,
        )
        assert ratios.shape == (4,)
        assert ratios.sum() == pytest.approx(1.0)

    def test_general_n_channel_return_counts(self, sf, wl):
        rng = np.random.default_rng(4)
        n_ch = 5
        pixel_QYs = rng.uniform(0.1, 0.9, (n_ch, len(wl)))
        photons = rng.uniform(500, 700, 150)
        mean_wl, counts = sf.calculate_colourratio_from_photon_wavelengths(
            photons, wl, pixel_QYs, return_counts=True,
        )
        assert counts.sum() == 150


# ======================================================================
# generate_bootstrap_colour_ratios
# ======================================================================

class TestGenerateBootstrapColourRatios:
    @pytest.mark.parametrize("dye", ["ATTO 488", "ATTO 565", "ATTO 647N"])
    def test_colour_ratios_are_the_unbiased_detection_efficiencies(self, sf, pixel_qys, dye):
        # Regression: the old per-photon channel assignment gave E[QE_c/sum QE] * E[sum QE]
        # instead of E[QE_c] -- for ATTO 488 a fingerprint bias of about (-0.47, +0.77, -0.29) pp.
        pixel_QYs, wavelength = pixel_qys
        spectrum = sf.get_spectral_data([dye], wavelength, SpectralDataType.DYE)[0]
        _, ratios = sf.generate_bootstrap_colour_ratios(
            spectrum, wavelength, pixel_QYs, n_photons_per_image=200, n_bootstrap=5,
            random_state=np.random.default_rng(0),
        )
        _, expected = sf.get_pixel_fractions_dye_and_filters([dye], [], wavelength, pixel_QYs, normalized=False)
        np.testing.assert_allclose(ratios, np.tile(expected, (5, 1)), rtol=1e-9)

    def test_mean_wavelength_fluctuates_per_image_like_n_sampled_photons(self, sf, pixel_qys):
        pixel_QYs, wavelength = pixel_qys
        spectrum = sf.get_spectral_data(DYE_NAME, wavelength, SpectralDataType.DYE)[0]
        n = 400
        mean_wls, _ = sf.generate_bootstrap_colour_ratios(
            spectrum, wavelength, pixel_QYs, n_photons_per_image=n, n_bootstrap=2000,
            random_state=np.random.default_rng(3),
        )
        p = np.maximum(spectrum, 0) / np.trapz(np.maximum(spectrum, 0), wavelength)
        mu = np.trapz(p * wavelength, wavelength); sd = np.sqrt(np.trapz(p * (wavelength - mu) ** 2, wavelength))
        assert mean_wls.mean() == pytest.approx(mu, abs=0.5)
        assert mean_wls.std() == pytest.approx(sd / np.sqrt(n), rel=0.1)

    def test_empty_spectrum_raises(self, sf, pixel_qys):
        pixel_QYs, wavelength = pixel_qys
        with pytest.raises(ValueError):
            sf.generate_bootstrap_colour_ratios(np.zeros_like(wavelength), wavelength, pixel_QYs,
                                                n_photons_per_image=10, n_bootstrap=2)

    def test_parallel_path(self, sf, pixel_qys):
        pixel_QYs, wavelength = pixel_qys
        spectrum = sf.get_spectral_data(DYE_NAME, wavelength, SpectralDataType.DYE)[0]
        mean_wls, ratios = sf.generate_bootstrap_colour_ratios(
            spectrum, wavelength, pixel_QYs, n_photons_per_image=50, n_bootstrap=4,
            random_state=np.random.default_rng(0), use_parallel=True,
        )
        assert mean_wls.shape == (4,)
        assert ratios.shape == (4, 3)

    def test_sequential_path(self, sf, pixel_qys):
        pixel_QYs, wavelength = pixel_qys
        spectrum = sf.get_spectral_data(DYE_NAME, wavelength, SpectralDataType.DYE)[0]
        mean_wls, ratios = sf.generate_bootstrap_colour_ratios(
            spectrum, wavelength, pixel_QYs, n_photons_per_image=50, n_bootstrap=3,
            random_state=np.random.default_rng(1), use_parallel=False,
        )
        assert mean_wls.shape == (3,)
        assert ratios.shape == (3, 3)

    def test_default_random_state(self, sf, pixel_qys):
        pixel_QYs, wavelength = pixel_qys
        spectrum = sf.get_spectral_data(DYE_NAME, wavelength, SpectralDataType.DYE)[0]
        mean_wls, ratios = sf.generate_bootstrap_colour_ratios(
            spectrum, wavelength, pixel_QYs, n_photons_per_image=20, n_bootstrap=2,
        )
        assert mean_wls.shape == (2,)
