#!/usr/bin/env python3
"""
Coverage tests for pyS3M.simulation.multicolour -- Bayer-camera image
simulation (gen_camera_image_stack), the unified bootstrap-fitting pipeline
(test_simulation_method / test_simulation_method_2d_sweep and their strategy
dispatch), and the simulation-based dye selector.

This is the biggest remaining gap in the coverage push (1346 stmts, was 38%
covered by pre-existing test_dye_selector.py/test_dye_selector_5dyes.py/
test_motion_blur.py/test_photoelectron_extraction.py/
test_pygmmis_integration.py). This file targets the large gaps those leave:
the non-vectorized gen_camera_image_stack path, defocus/motion-blur/per-frame
-mask branches, the full test_simulation_method strategy dispatch (which
exercises _fit_standard/_fit_elliptical/_fit_standard_iter/_fit_standard_data/
_fit_demosaic_ig/_fit_demosaic/_compute_fit_statistics all at once),
test_simulation_method_2d_sweep, and the remaining dye-selector branches
(exhaustive search, viable-dye-count errors, plotting).

Deliberately tiny throughout: n_bootstrap=2-3, small (4x4-12x12) synthetic
camera maps, 1-2 photon levels -- these are branch-coverage unit tests, not
statistically meaningful simulation/precision benchmarks.
"""
from __future__ import annotations

import types

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import pyS3M.MaskFunctions as MaskFunctions
import pyS3M.SpectralFunctions as SpectralFunctions
import pyS3M.sCMOSFunctions as sCMOSFunctions
from pyS3M.simulation.multicolour import (
    CameraParameters,
    FittingResultProcessor,
    FittingStrategy,
    MultiC_Sim_Funcs,
    MultiC_Sim_Funcs_Refactored,
    SimulationConfig,
    SimulationValidationError,
)

FILTERS = [
    "semrock-nf03-405-488-561-635e",
    "semrock-di03-r405-488-561-635-t1-25x36",
    "semrock-bsp01-785r",
]
DYE = "ATTO 647N"


# ======================================================================
# Shared fixtures
# ======================================================================

@pytest.fixture(scope="module")
def spectral():
    return SpectralFunctions.Spectral_Funcs()


@pytest.fixture(scope="module")
def wavelength_and_qys(spectral):
    R, G, B, wl = spectral.getpixelefficiency()
    return wl, np.vstack([B, G, R])


@pytest.fixture(scope="module")
def smoothing_function():
    scmos = sCMOSFunctions.sCMOS_Functions()
    sf = types.SimpleNamespace()
    sf.args = {"sigma": 0.85}
    sf.data_arg = "image"
    sf.smoothing_function = scmos.gaussian_filter_stack
    sf.extent = 0.85
    return sf


def _camera_params_dict(size=8, mosaic_unit=None, pixel_QYs=None):
    M_F = MaskFunctions.Mask_Functions()
    return {
        "gain": np.full((size, size), 1.0),
        "offset": np.full((size, size), 100.0),
        "variance": np.full((size, size), 1e-12),
        "readnoise": 1e-6,
        "rqe": np.full((size, size), 1.0),
        "pixel_QYs": pixel_QYs,
        "masks": M_F.get_masks(size_x=size, size_y=size, mosaic_unit=mosaic_unit),
        "pixel_order": ["B", "G", "R"],
        "pixel_order_indices": {"B": 0, "G": 1, "R": 2},
        "mosaic_unit": mosaic_unit,
    }


@pytest.fixture
def sim(wavelength_and_qys):
    _, pixel_QYs = wavelength_and_qys
    return MultiC_Sim_Funcs_Refactored(camera="ximea")


@pytest.fixture
def camera_parameters(wavelength_and_qys):
    _, pixel_QYs = wavelength_and_qys
    return _camera_params_dict(size=8, pixel_QYs=pixel_QYs)


# ======================================================================
# FittingStrategy / CameraParameters / SimulationConfig
# ======================================================================

class TestFittingStrategy:
    def test_values(self):
        assert FittingStrategy.STANDARD.value == "standard"
        assert FittingStrategy.DEMOSAIC.value == "demosaic"


class TestCameraParametersValidate:
    def test_missing_keys_raises(self):
        with pytest.raises(ValueError, match="missing required keys"):
            CameraParameters.validate_and_create({"gain": np.ones((2, 2))})

    def test_valid_dict_creates(self, camera_parameters):
        cp = CameraParameters.validate_and_create(camera_parameters)
        assert cp.mosaic_unit is None

    def test_mosaic_unit_default_none(self, camera_parameters):
        cp = CameraParameters.validate_and_create(camera_parameters)
        assert cp.gain.shape == (8, 8)


class TestSimulationConfig:
    def test_defaults(self):
        cfg = SimulationConfig()
        assert cfg.background_colour == [1, 1, 1]

    def test_explicit_background_colour_not_overwritten(self):
        cfg = SimulationConfig(background_colour=[0.5, 0.5, 0.5])
        assert cfg.background_colour == [0.5, 0.5, 0.5]


# ======================================================================
# FittingResultProcessor
# ======================================================================

class TestColourFitAverager:
    def test_normal_case(self):
        n = 2
        df = pd.DataFrame({
            "b": [1.0, 2.0, 3.0] * n,
            "A": [4.0, 5.0, 6.0] * n,
            "chi_sqr": [0.1, 0.2, 0.3] * n,
        })
        out = FittingResultProcessor.colour_fit_averager(df, n)
        assert len(out) == n
        assert out["A_B"].iloc[0] == pytest.approx(4.0 / 15.0)

    def test_zero_sum_leaves_zero(self):
        df = pd.DataFrame({"b": [0.0, 0.0, 0.0], "A": [0.0, 0.0, 0.0], "chi_sqr": [0.1, 0.1, 0.1]})
        out = FittingResultProcessor.colour_fit_averager(df, 1)
        assert out["A_B"].iloc[0] == 0.0
        assert out["bg_B"].iloc[0] == 0.0


class TestFitAverager:
    def test_normal_case(self):
        n = 2
        df = pd.DataFrame({
            "xc": [1.0, 2.0, 3.0] * n, "yc": [1.0, 2.0, 3.0] * n,
            "s_x": [1.0] * 3 * n, "s_y": [1.0] * 3 * n,
            "b": [1.0, 1.0, 1.0] * n, "A": [1.0, 2.0, 3.0] * n,
            "chi_sqr": [0.1] * 3 * n,
        })
        out = FittingResultProcessor.fit_averager(df, n)
        assert len(out) == n

    def test_empty_data_returns_nan_frame(self):
        df = pd.DataFrame({"xc": [], "yc": [], "s_x": [], "s_y": [], "b": [], "A": [], "chi_sqr": []})
        out = FittingResultProcessor.fit_averager(df, 2)
        assert len(out) == 2
        assert np.all(np.isnan(out["xc"]))

    def test_short_data_still_computes(self):
        # Fewer rows than n_bootstrap*3 but not empty -> warns, still processes.
        df = pd.DataFrame({
            "xc": [1.0, 2.0], "yc": [1.0, 2.0], "s_x": [1.0, 1.0], "s_y": [1.0, 1.0],
            "b": [1.0, 1.0], "A": [1.0, 2.0], "chi_sqr": [0.1, 0.1],
        })
        out = FittingResultProcessor.fit_averager(df, 2)
        assert len(out) == 2


# ======================================================================
# MultiC_Sim_Funcs_Refactored.__init__ / _validate_inputs
# ======================================================================

class TestInit:
    def test_defaults_ximea(self):
        s = MultiC_Sim_Funcs_Refactored(camera="ximea")
        assert s.pixel_size > 0
        assert s.mosaic_unit is not None

    def test_explicit_overrides(self):
        s = MultiC_Sim_Funcs_Refactored(camera="ximea", pixel_size=0.1)
        assert s.pixel_size == 0.1


class TestValidateInputs:
    def test_wavelength_mismatch_raises(self, sim, camera_parameters):
        with pytest.raises(SimulationValidationError, match="not defined at all wavelengths"):
            sim._validate_inputs(np.linspace(400, 700, 5), camera_parameters, None, {})

    def test_deterministic_single_dye_wrong_x0y0_count_raises(self, sim, camera_parameters, wavelength_and_qys):
        wl, _ = wavelength_and_qys
        dpe = np.array([0.3, 0.3, 0.4])  # 1D -> deterministic single-dye
        with pytest.raises(SimulationValidationError, match="does not contain correct number"):
            sim._validate_inputs(wl, camera_parameters, dpe, {"a": 1, "b": 2})

    def test_stochastic_mode_wrong_dye_count_raises(self, sim, camera_parameters, wavelength_and_qys):
        wl, _ = wavelength_and_qys
        n_frames = 5
        dpe = np.zeros((n_frames, 3))  # matches x0y0 n_frames -> stochastic mode
        x0y0 = {"a": np.zeros((n_frames, 2, 1)), "b": np.zeros((n_frames, 2, 1))}
        with pytest.raises(SimulationValidationError, match="stochastic mode requires single dye"):
            sim._validate_inputs(wl, camera_parameters, dpe, x0y0)

    def test_deterministic_multidye_wrong_count_raises(self, sim, camera_parameters, wavelength_and_qys):
        wl, _ = wavelength_and_qys
        dpe = np.zeros((3, 3))  # 3 dyes expected
        x0y0 = {"a": np.zeros((99, 2, 1))}  # only 1 dye given, doesn't match n_frames either
        with pytest.raises(SimulationValidationError, match="does not contain correct number"):
            sim._validate_inputs(wl, camera_parameters, dpe, x0y0)

    def test_missing_key_wrapped_as_validation_error(self, sim, wavelength_and_qys):
        wl, _ = wavelength_and_qys
        with pytest.raises(SimulationValidationError, match="Input validation failed"):
            sim._validate_inputs(wl, {}, None, {})


# ======================================================================
# gen_camera_image_stack
# ======================================================================

def _minimal_x0y0_photons(size=8, n_bootstrap=2):
    x0y0 = {"dye": np.zeros((n_bootstrap, 2, 1))}
    x0y0["dye"][:, 0, 0] = size / 2.0
    x0y0["dye"][:, 1, 0] = size / 2.0
    n_photons = {"dye": np.full(n_bootstrap, 2000)}
    return x0y0, n_photons


class TestGenCameraImageStack:
    def test_non_vectorized_path(self, sim, camera_parameters, wavelength_and_qys, smoothing_function):
        wl, pixel_QYs = wavelength_and_qys
        x0y0, n_photons = _minimal_x0y0_photons()
        bayer, smoothed, normal = sim.gen_camera_image_stack(
            camera_parameters, wl, 660.0, np.array([0.2, 0.3, 0.5]),
            n_photons, x0y0, smoothing_function=smoothing_function,
            background_photons=5.0, use_vectorized_photoelectrons=False,
        )
        assert bayer.shape == (2, 8, 8)

    def test_non_vectorized_return_normal_image(self, sim, camera_parameters, wavelength_and_qys, smoothing_function):
        wl, _ = wavelength_and_qys
        x0y0, n_photons = _minimal_x0y0_photons()
        bayer, smoothed, normal = sim.gen_camera_image_stack(
            camera_parameters, wl, 660.0, np.array([0.2, 0.3, 0.5]),
            n_photons, x0y0, smoothing_function=smoothing_function,
            background_photons=5.0, use_vectorized_photoelectrons=False,
            return_normal_image=True,
        )
        assert normal.shape == (2, 8, 8)

    def test_non_vectorized_return_photoelectrons(self, sim, camera_parameters, wavelength_and_qys, smoothing_function):
        wl, _ = wavelength_and_qys
        x0y0, n_photons = _minimal_x0y0_photons()
        bayer, smoothed, normal = sim.gen_camera_image_stack(
            camera_parameters, wl, 660.0, np.array([0.2, 0.3, 0.5]),
            n_photons, x0y0, smoothing_function=smoothing_function,
            background_photons=5.0, use_vectorized_photoelectrons=False,
            return_normal_image=True, return_photoelectrons=True,
        )
        assert normal.dtype in (np.int64, np.int32, np.float64)

    def test_defocus_z(self, sim, camera_parameters, wavelength_and_qys, smoothing_function):
        wl, _ = wavelength_and_qys
        x0y0, n_photons = _minimal_x0y0_photons()
        bayer, smoothed, _ = sim.gen_camera_image_stack(
            camera_parameters, wl, 660.0, np.array([0.2, 0.3, 0.5]),
            n_photons, x0y0, smoothing_function=smoothing_function,
            background_photons=5.0, defocus_z_um=0.3,
        )
        assert bayer.shape == (2, 8, 8)

    def test_motion_blur(self, sim, camera_parameters, wavelength_and_qys, smoothing_function):
        wl, _ = wavelength_and_qys
        x0y0, n_photons = _minimal_x0y0_photons()
        bayer, smoothed, _ = sim.gen_camera_image_stack(
            camera_parameters, wl, 660.0, np.array([0.2, 0.3, 0.5]),
            n_photons, x0y0, smoothing_function=smoothing_function,
            background_photons=5.0, motion_velocity_nm_per_s=5e6, frame_exposure_ms=100.0,
        )
        assert bayer.shape == (2, 8, 8)

    def test_motion_blur_non_vectorized(self, sim, camera_parameters, wavelength_and_qys, smoothing_function):
        wl, _ = wavelength_and_qys
        x0y0, n_photons = _minimal_x0y0_photons()
        bayer, smoothed, _ = sim.gen_camera_image_stack(
            camera_parameters, wl, 660.0, np.array([0.2, 0.3, 0.5]),
            n_photons, x0y0, smoothing_function=smoothing_function,
            background_photons=5.0, motion_velocity_nm_per_s=5e6, frame_exposure_ms=100.0,
            use_vectorized_photoelectrons=False,
        )
        assert bayer.shape == (2, 8, 8)

    def test_defocus_motion_blur_combined_non_vectorized(self, sim, camera_parameters, wavelength_and_qys, smoothing_function):
        wl, _ = wavelength_and_qys
        x0y0, n_photons = _minimal_x0y0_photons()
        bayer, smoothed, _ = sim.gen_camera_image_stack(
            camera_parameters, wl, 660.0, np.array([0.2, 0.3, 0.5]),
            n_photons, x0y0, smoothing_function=smoothing_function,
            background_photons=5.0, defocus_z_um=0.3,
            motion_velocity_nm_per_s=5e6, frame_exposure_ms=100.0,
            use_vectorized_photoelectrons=False,
        )
        assert bayer.shape == (2, 8, 8)

    def test_return_photoelectrons_stack(self, sim, camera_parameters, wavelength_and_qys):
        wl, _ = wavelength_and_qys
        x0y0, n_photons = _minimal_x0y0_photons()
        pe = sim.gen_camera_image_stack(
            camera_parameters, wl, 660.0, np.array([0.2, 0.3, 0.5]),
            n_photons, x0y0, smoothing_function=None,
            background_photons=5.0, return_photoelectrons_stack=True,
        )
        assert pe.dtype == np.int32
        assert pe.shape == (2, 8, 8)

    def test_no_smoothing_function(self, sim, camera_parameters, wavelength_and_qys):
        wl, _ = wavelength_and_qys
        x0y0, n_photons = _minimal_x0y0_photons()
        bayer, smoothed, _ = sim.gen_camera_image_stack(
            camera_parameters, wl, 660.0, np.array([0.2, 0.3, 0.5]),
            n_photons, x0y0, smoothing_function=None, background_photons=5.0,
        )
        # smoothed_image=None survives an np.squeeze() as a 0-d object array
        # wrapping None, not None itself.
        assert np.asarray(smoothed).item() is None

    def test_stochastic_per_frame_wavelengths(self, sim, camera_parameters, wavelength_and_qys, smoothing_function):
        wl, _ = wavelength_and_qys
        x0y0, n_photons = _minimal_x0y0_photons()
        avg_wl = np.array([650.0, 670.0])
        dpe = np.array([[0.2, 0.3, 0.5], [0.25, 0.35, 0.4]])
        bayer, smoothed, _ = sim.gen_camera_image_stack(
            camera_parameters, wl, avg_wl, dpe,
            n_photons, x0y0, smoothing_function=smoothing_function, background_photons=5.0,
        )
        assert bayer.shape == (2, 8, 8)

    def test_stochastic_per_frame_wavelengths_non_vectorized(self, sim, camera_parameters, wavelength_and_qys, smoothing_function):
        wl, _ = wavelength_and_qys
        x0y0, n_photons = _minimal_x0y0_photons()
        avg_wl = np.array([650.0, 670.0])
        dpe = np.array([[0.2, 0.3, 0.5], [0.25, 0.35, 0.4]])
        bayer, smoothed, _ = sim.gen_camera_image_stack(
            camera_parameters, wl, avg_wl, dpe,
            n_photons, x0y0, smoothing_function=smoothing_function, background_photons=5.0,
            use_vectorized_photoelectrons=False,
        )
        assert bayer.shape == (2, 8, 8)

    def test_high_photon_count_forces_float32(self, sim, camera_parameters, wavelength_and_qys, smoothing_function):
        wl, _ = wavelength_and_qys
        x0y0 = {"dye": np.zeros((1, 2, 1))}
        x0y0["dye"][:, 0, 0] = 4.0
        x0y0["dye"][:, 1, 0] = 4.0
        n_photons = {"dye": np.array([50000000])}
        bayer, smoothed, _ = sim.gen_camera_image_stack(
            camera_parameters, wl, 660.0, np.array([0.2, 0.3, 0.5]),
            n_photons, x0y0, smoothing_function=smoothing_function, background_photons=5.0,
        )
        assert bayer.dtype == np.float32

    def test_mid_photon_count_uint16(self, sim, camera_parameters, wavelength_and_qys, smoothing_function):
        wl, _ = wavelength_and_qys
        x0y0 = {"dye": np.zeros((1, 2, 1))}
        x0y0["dye"][:, 0, 0] = 4.0
        x0y0["dye"][:, 1, 0] = 4.0
        cp = _camera_params_dict(size=8, pixel_QYs=None)
        cp["pixel_QYs"] = camera_parameters["pixel_QYs"]
        cp["offset"] = np.full((8, 8), 1000.0)
        n_photons = {"dye": np.array([20000])}
        bayer, smoothed, _ = sim.gen_camera_image_stack(
            cp, wl, 660.0, np.array([0.2, 0.3, 0.5]),
            n_photons, x0y0, smoothing_function=smoothing_function, background_photons=5.0,
        )
        assert bayer.dtype in (np.uint16, np.uint8, np.float32)

    def test_per_frame_masks_3d(self, sim, camera_parameters, wavelength_and_qys, smoothing_function):
        wl, _ = wavelength_and_qys
        x0y0, n_photons = _minimal_x0y0_photons()
        cp = dict(camera_parameters)
        # Make the per-colour masks 3D (n_bootstrap, H, W): same content repeated,
        # just to exercise the "masks vary per frame" code path.
        cp["masks"] = {
            c: np.stack([m, m], axis=0) for c, m in camera_parameters["masks"].items()
        }
        bayer, smoothed, _ = sim.gen_camera_image_stack(
            cp, wl, 660.0, np.array([0.2, 0.3, 0.5]),
            n_photons, x0y0, smoothing_function=smoothing_function, background_photons=5.0,
        )
        assert bayer.shape == (2, 8, 8)

    def test_per_frame_masks_3d_non_vectorized(self, sim, camera_parameters, wavelength_and_qys, smoothing_function):
        wl, _ = wavelength_and_qys
        x0y0, n_photons = _minimal_x0y0_photons()
        cp = dict(camera_parameters)
        cp["masks"] = {
            c: np.stack([m, m], axis=0) for c, m in camera_parameters["masks"].items()
        }
        bayer, smoothed, _ = sim.gen_camera_image_stack(
            cp, wl, 660.0, np.array([0.2, 0.3, 0.5]),
            n_photons, x0y0, smoothing_function=smoothing_function, background_photons=5.0,
            use_vectorized_photoelectrons=False,
        )
        assert bayer.shape == (2, 8, 8)


# ======================================================================
# test_simulation_method -- strategy dispatch (covers _fit_standard/
# _fit_elliptical/_fit_standard_iter/_fit_standard_data/_fit_demosaic_ig/
# _fit_demosaic/_compute_fit_statistics all at once)
# ======================================================================

class TestSimulationMethod:
    def _run(self, sim, camera_parameters, wavelength_and_qys, smoothing_function, tmp_path,
              strategy, **extra_config):
        wl, _ = wavelength_and_qys
        cfg_kwargs = dict(
            n_bootstrap=2, background_photons=5.0, save_raw_results=False,
            save_summary_csvs=False, verbose=False, use_stochastic_photons=False,
            n_unit_cells=0,
        )
        cfg_kwargs.update(extra_config)
        config = SimulationConfig(**cfg_kwargs)
        sim.test_simulation_method(
            dye=DYE, filters=FILTERS, wavelength=wl,
            camera_parameters=camera_parameters, save_folder=str(tmp_path),
            n_photon_space=np.array([2000.0]), smoothing_function=smoothing_function,
            strategy=strategy, config=config,
        )

    @pytest.mark.parametrize("strategy", [
        FittingStrategy.STANDARD, FittingStrategy.ELLIPTICAL,
        FittingStrategy.STANDARD_ITER, FittingStrategy.STANDARD_DATA,
        FittingStrategy.STANDARD_IG, FittingStrategy.DEMOSAIC,
    ])
    def test_all_strategies(self, sim, camera_parameters, wavelength_and_qys, smoothing_function, tmp_path, strategy):
        self._run(sim, camera_parameters, wavelength_and_qys, smoothing_function, tmp_path, strategy)

    def test_with_mosaic_unit_resize(self, sim, wavelength_and_qys, smoothing_function, tmp_path):
        wl, pixel_QYs = wavelength_and_qys
        cp = _camera_params_dict(size=8, mosaic_unit=sim.mosaic_unit, pixel_QYs=pixel_QYs)
        config = SimulationConfig(
            n_bootstrap=2, background_photons=5.0, save_raw_results=False,
            save_summary_csvs=False, verbose=False, use_stochastic_photons=False,
            n_unit_cells=2,
        )
        sim.test_simulation_method(
            dye=DYE, filters=FILTERS, wavelength=wl, camera_parameters=cp,
            save_folder=str(tmp_path), n_photon_space=np.array([2000.0]),
            smoothing_function=smoothing_function, strategy=FittingStrategy.STANDARD,
            config=config,
        )

    def test_save_raw_results_and_summary_csvs(self, sim, camera_parameters, wavelength_and_qys, smoothing_function, tmp_path):
        self._run(
            sim, camera_parameters, wavelength_and_qys, smoothing_function, tmp_path,
            FittingStrategy.STANDARD, save_raw_results=True, save_summary_csvs=True,
        )
        assert any(tmp_path.glob("*rawresults.h5"))

    def test_overwrite_false_skips_completed_levels(self, sim, camera_parameters, wavelength_and_qys, smoothing_function, tmp_path):
        wl, _ = wavelength_and_qys
        config = SimulationConfig(
            n_bootstrap=2, background_photons=5.0, save_raw_results=True,
            save_summary_csvs=False, verbose=False, use_stochastic_photons=False,
        )
        common = dict(
            dye=DYE, filters=FILTERS, wavelength=wl, camera_parameters=camera_parameters,
            save_folder=str(tmp_path), n_photon_space=np.array([2000.0]),
            smoothing_function=smoothing_function, strategy=FittingStrategy.STANDARD,
            config=config,
        )
        sim.test_simulation_method(**common, overwrite=True)
        # Second call with overwrite=False should detect the completed level and return early.
        sim.test_simulation_method(**common, overwrite=False)

    def test_overwrite_false_fresh_continuation(self, sim, camera_parameters, wavelength_and_qys, smoothing_function, tmp_path):
        # No existing results yet -> overwrite=False just runs fresh (exercises the
        # ground-truth-position "fresh run" write branch under continuation mode).
        self._run(
            sim, camera_parameters, wavelength_and_qys, smoothing_function, tmp_path,
            FittingStrategy.STANDARD, save_raw_results=False,
        )

    def test_sbr_set(self, sim, camera_parameters, wavelength_and_qys, smoothing_function, tmp_path):
        self._run(
            sim, camera_parameters, wavelength_and_qys, smoothing_function, tmp_path,
            FittingStrategy.STANDARD, sbr=2.0,
        )

    def test_sbr_invalid_raises(self, sim, camera_parameters, wavelength_and_qys, smoothing_function, tmp_path):
        from pyS3M.simulation.multicolour import SimulationValidationError as SVE
        with pytest.raises(SVE, match="must be > 1"):
            self._run(
                sim, camera_parameters, wavelength_and_qys, smoothing_function, tmp_path,
                FittingStrategy.STANDARD, sbr=0.5,
            )

    def test_saverawimages(self, sim, camera_parameters, wavelength_and_qys, smoothing_function, tmp_path):
        self._run(
            sim, camera_parameters, wavelength_and_qys, smoothing_function, tmp_path,
            FittingStrategy.STANDARD, saverawimages=True,
        )
        assert any(tmp_path.glob("*rawbayerimage.tiff"))

    def test_stochastic_photons_on(self, sim, camera_parameters, wavelength_and_qys, smoothing_function, tmp_path):
        wl, _ = wavelength_and_qys
        config = SimulationConfig(
            n_bootstrap=2, background_photons=5.0, save_raw_results=False,
            save_summary_csvs=False, verbose=False, use_stochastic_photons=True,
            n_unit_cells=0,
        )
        sim.test_simulation_method(
            dye=DYE, filters=FILTERS, wavelength=wl, camera_parameters=camera_parameters,
            save_folder=str(tmp_path), n_photon_space=np.array([2000.0]),
            smoothing_function=smoothing_function, strategy=FittingStrategy.STANDARD,
            config=config,
        )

    def test_nile_red_wavelength_fits(self, sim, camera_parameters, wavelength_and_qys, smoothing_function, tmp_path):
        wl, _ = wavelength_and_qys
        config = SimulationConfig(
            n_bootstrap=2, background_photons=5.0, save_raw_results=True,
            save_summary_csvs=False, verbose=False, use_stochastic_photons=False,
        )
        sim.test_simulation_method(
            dye=DYE, filters=FILTERS, wavelength=wl, camera_parameters=camera_parameters,
            save_folder=str(tmp_path), n_photon_space=np.array([2000.0]),
            smoothing_function=smoothing_function, strategy=FittingStrategy.STANDARD,
            config=config, nile_red_wavelength=620.0,
        )
        h5 = list(tmp_path.glob("*rawresults.h5"))
        assert h5

    def test_single_dye_spectrum_simulated(self, sim, camera_parameters, wavelength_and_qys, smoothing_function, tmp_path):
        wl, _ = wavelength_and_qys
        spectrum = np.exp(-((wl - 660.0) ** 2) / (2 * 20.0 ** 2))
        config = SimulationConfig(
            n_bootstrap=2, background_photons=5.0, save_raw_results=False,
            save_summary_csvs=False, verbose=False, use_stochastic_photons=True,
        )
        sim.test_simulation_method(
            dye="simulated_test_dye", filters=FILTERS, wavelength=wl,
            camera_parameters=camera_parameters, save_folder=str(tmp_path),
            n_photon_space=np.array([2000.0]), smoothing_function=smoothing_function,
            strategy=FittingStrategy.STANDARD, config=config,
            single_dye_spectrum=spectrum,
        )


class TestSimulationMethod2DSweep:
    def test_basic_sweep(self, sim, camera_parameters, wavelength_and_qys, smoothing_function, tmp_path):
        wl, _ = wavelength_and_qys
        config = SimulationConfig(
            n_bootstrap=2, save_raw_results=True, verbose=False,
            use_stochastic_photons=False, n_unit_cells=0,
        )
        sim.test_simulation_method_2d_sweep(
            dye=DYE, filters=FILTERS, wavelength=wl, camera_parameters=camera_parameters,
            save_folder=str(tmp_path), n_photon_space=np.array([2000.0]),
            smoothing_function=smoothing_function, strategy=FittingStrategy.STANDARD,
            read_noise_space=np.array([1.0, 2.0]), peak_qy_space=np.array([0.5, 1.0]),
            config=config,
        )
        assert any(tmp_path.glob("*rawresults.h5"))

    def test_sweep_skips_existing_when_not_overwrite(self, sim, camera_parameters, wavelength_and_qys, smoothing_function, tmp_path):
        wl, _ = wavelength_and_qys
        config = SimulationConfig(
            n_bootstrap=2, save_raw_results=True, verbose=False,
            use_stochastic_photons=False, n_unit_cells=0,
        )
        common = dict(
            dye=DYE, filters=FILTERS, wavelength=wl, camera_parameters=camera_parameters,
            save_folder=str(tmp_path), n_photon_space=np.array([2000.0]),
            smoothing_function=smoothing_function, strategy=FittingStrategy.STANDARD,
            read_noise_space=np.array([1.0]), peak_qy_space=np.array([1.0]),
            config=config,
        )
        sim.test_simulation_method_2d_sweep(**common, overwrite=True)
        sim.test_simulation_method_2d_sweep(**common, overwrite=False)

    def test_sweep_with_mosaic_resize(self, sim, wavelength_and_qys, smoothing_function, tmp_path):
        wl, pixel_QYs = wavelength_and_qys
        cp = _camera_params_dict(size=8, mosaic_unit=sim.mosaic_unit, pixel_QYs=pixel_QYs)
        config = SimulationConfig(
            n_bootstrap=2, save_raw_results=False, verbose=False,
            use_stochastic_photons=True, n_unit_cells=2,
        )
        sim.test_simulation_method_2d_sweep(
            dye=DYE, filters=FILTERS, wavelength=wl, camera_parameters=cp,
            save_folder=str(tmp_path), n_photon_space=np.array([2000.0]),
            smoothing_function=smoothing_function, strategy=FittingStrategy.STANDARD,
            read_noise_space=np.array([1.0]), peak_qy_space=np.array([1.0]),
            config=config,
        )

    def test_sweep_custom_starting_flag_fn(self, sim, camera_parameters, wavelength_and_qys, smoothing_function, tmp_path):
        wl, _ = wavelength_and_qys
        config = SimulationConfig(
            n_bootstrap=2, save_raw_results=False, verbose=False,
            use_stochastic_photons=False, n_unit_cells=0,
        )
        sim.test_simulation_method_2d_sweep(
            dye=DYE, filters=FILTERS, wavelength=wl, camera_parameters=camera_parameters,
            save_folder=str(tmp_path), n_photon_space=np.array([2000.0]),
            smoothing_function=smoothing_function, strategy=FittingStrategy.STANDARD,
            read_noise_space=np.array([1.0]), peak_qy_space=np.array([1.0]),
            starting_flag_fn=lambda d, qy, rn: f"custom_{d}_",
            config=config,
        )


# ======================================================================
# MultiC_Sim_Funcs_Compatibility wrappers
# ======================================================================

class TestCompatibilityWrappers:
    def test_test_fit_method(self, camera_parameters, wavelength_and_qys, smoothing_function, tmp_path):
        wl, _ = wavelength_and_qys
        msf = MultiC_Sim_Funcs(camera="ximea")
        config = SimulationConfig(
            n_bootstrap=2, background_photons=5.0, save_raw_results=False,
            save_summary_csvs=False, verbose=False, use_stochastic_photons=False,
            n_unit_cells=0,
        )
        msf.test_fit_method(
            dye=DYE, filters=FILTERS, wavelength=wl, camera_parameters=camera_parameters,
            save_folder=str(tmp_path), n_photon_space=np.array([2000.0]),
            smoothing_function=smoothing_function, config=config,
        )

    def test_test_demosaic_fit_method(self, camera_parameters, wavelength_and_qys, smoothing_function, tmp_path):
        wl, _ = wavelength_and_qys
        msf = MultiC_Sim_Funcs(camera="ximea")
        config = SimulationConfig(
            n_bootstrap=2, background_photons=5.0, save_raw_results=False,
            save_summary_csvs=False, verbose=False, use_stochastic_photons=False,
            n_unit_cells=0,
        )
        msf.test_demosaic_fit_method(
            dye=DYE, filters=FILTERS, wavelength=wl, camera_parameters=camera_parameters,
            save_folder=str(tmp_path), n_photon_space=np.array([2000.0]),
            smoothing_function=smoothing_function, config=config,
        )


# ======================================================================
# Dye selector -- gaps left by test_dye_selector.py/test_dye_selector_5dyes.py
# ======================================================================

SINGLE_MOLECULE_DYES = np.array([
    ["ATTO 488", 2073], ["Cy3B", 23195], ["ATTO 565", 11600],
    ["ATTO 643", 23327], ["ATTO 647N", 18448], ["Alexa Fluor 647", 10348],
], dtype="object")


class TestDyeSelectorGaps:
    def test_filter_dyes_missing_from_table_skipped(self, wavelength_and_qys, camera_parameters):
        # "ATTO 665" is a real dye in the spectral database (so
        # get_dye_or_filter_data succeeds) but absent from
        # SINGLE_MOLECULE_DYES's lookup table -- the `continue` branch.
        wl, pixel_QYs = wavelength_and_qys
        msf = MultiC_Sim_Funcs(camera="ximea")
        result = msf._filter_dyes_by_photons(
            potential_dyes=["Cy3B", "ATTO 665"],
            single_molecule_dyes=SINGLE_MOLECULE_DYES,
            filters=FILTERS, camera_parameters={"pixel_QYs": pixel_QYs},
            wavelength=wl, min_photons_per_100ms=500,
        )
        assert "ATTO 665" not in result["expected_photons"]

    def test_fit_dye_gaussian_all_nan_raises(self):
        msf = MultiC_Sim_Funcs(camera="ximea")
        color_data = {"A_R": np.full(5, np.nan), "A_G": np.full(5, np.nan)}
        with pytest.raises(ValueError, match="No valid fits"):
            msf._fit_dye_gaussian(color_data)

    def test_calculate_dye_separability(self):
        msf = MultiC_Sim_Funcs(camera="ximea")
        dye_gaussians = {
            "A": {"mean": np.array([0.2, 0.2]), "covariance": np.eye(2) * 0.01},
            "B": {"mean": np.array([0.8, 0.8]), "covariance": np.eye(2) * 0.01},
        }
        stats = msf._calculate_dye_separability(dye_gaussians, ["A", "B"], n_monte_carlo=200)
        assert "pairwise_separability" in stats
        assert ("A", "B") in stats["pairwise_separability"]

    def test_optimal_dye_selector_too_few_viable_raises(self, wavelength_and_qys, camera_parameters):
        wl, _ = wavelength_and_qys
        msf = MultiC_Sim_Funcs(camera="ximea")
        with pytest.raises(ValueError, match="viable dyes"):
            msf.optimal_dye_selector_simulated(
                potential_dyes=["Cy3B"],
                single_molecule_dyes=SINGLE_MOLECULE_DYES,
                filters=FILTERS, camera_parameters=camera_parameters,
                wavelength=wl, n_dyes_desired=5,
                min_photons_per_100ms=500,
            )

    def test_optimal_dye_selector_exhaustive_search(self, wavelength_and_qys, smoothing_function):
        wl, pixel_QYs = wavelength_and_qys
        msf = MultiC_Sim_Funcs(camera="ximea")
        M_F = MaskFunctions.Mask_Functions()
        cp = {
            "gain": np.full((12, 12), 1.0), "offset": np.full((12, 12), 100.0),
            "variance": np.full((12, 12), 1e-12), "readnoise": 1e-12,
            "rqe": np.full((12, 12), 1.0), "pixel_QYs": pixel_QYs,
            "masks": M_F.get_masks(size_x=12, size_y=12),
            "pixel_order": ["B", "G", "R"], "pixel_order_indices": {"B": 0, "G": 1, "R": 2},
        }
        result = msf.optimal_dye_selector_simulated(
            potential_dyes=["Cy3B", "ATTO 643", "ATTO 647N"],
            single_molecule_dyes=SINGLE_MOLECULE_DYES,
            filters=FILTERS, camera_parameters=cp, wavelength=wl,
            n_dyes_desired=2, min_photons_per_100ms=500, n_simulations=20,
            smoothing_function=smoothing_function, exhaustive_search=True,
            return_all_simulations=True, verbose=False,
        )
        assert result["all_combinations_tested"] is not None
        assert set(result["dye_simulations"].keys()) == set(result["viable_dyes"])

    def test_plot_dye_selection_results(self, wavelength_and_qys, smoothing_function):
        wl, pixel_QYs = wavelength_and_qys
        msf = MultiC_Sim_Funcs(camera="ximea")
        M_F = MaskFunctions.Mask_Functions()
        cp = {
            "gain": np.full((12, 12), 1.0), "offset": np.full((12, 12), 100.0),
            "variance": np.full((12, 12), 1e-12), "readnoise": 1e-12,
            "rqe": np.full((12, 12), 1.0), "pixel_QYs": pixel_QYs,
            "masks": M_F.get_masks(size_x=12, size_y=12),
            "pixel_order": ["B", "G", "R"], "pixel_order_indices": {"B": 0, "G": 1, "R": 2},
        }
        result = msf.optimal_dye_selector_simulated(
            potential_dyes=["Cy3B", "ATTO 643"],
            single_molecule_dyes=SINGLE_MOLECULE_DYES,
            filters=FILTERS, camera_parameters=cp, wavelength=wl,
            n_dyes_desired=2, min_photons_per_100ms=500, n_simulations=20,
            smoothing_function=smoothing_function, exhaustive_search=False, verbose=False,
        )
        fig, axes = msf.plot_dye_selection_results(result, show=False)
        assert fig is not None

    def test_verbose_reports_rejected_dyes(self, wavelength_and_qys, camera_parameters, smoothing_function):
        wl, _ = wavelength_and_qys
        msf = MultiC_Sim_Funcs(camera="ximea")
        # Threshold sits between ATTO 488's (~759) and Cy3B's/ATTO 643's
        # (~8489/~8537) expected photons at detector, so ATTO 488 is rejected
        # while the other two stay viable -- enough for the n_dyes_desired=2
        # greedy-pair search to succeed (it needs >= 2 viable dyes to bootstrap)
        # while still hitting the verbose "Rejected: ..." log branch.
        msf.optimal_dye_selector_simulated(
            potential_dyes=["Cy3B", "ATTO 643", "ATTO 488"],
            single_molecule_dyes=SINGLE_MOLECULE_DYES,
            filters=FILTERS, camera_parameters=camera_parameters, wavelength=wl,
            n_dyes_desired=2, min_photons_per_100ms=2000, n_simulations=3,
            smoothing_function=smoothing_function, verbose=True,
        )

    def test_exhaustive_search_verbose(self, wavelength_and_qys, smoothing_function):
        wl, pixel_QYs = wavelength_and_qys
        msf = MultiC_Sim_Funcs(camera="ximea")
        M_F = MaskFunctions.Mask_Functions()
        cp = {
            "gain": np.full((12, 12), 1.0), "offset": np.full((12, 12), 100.0),
            "variance": np.full((12, 12), 1e-12), "readnoise": 1e-12,
            "rqe": np.full((12, 12), 1.0), "pixel_QYs": pixel_QYs,
            "masks": M_F.get_masks(size_x=12, size_y=12),
            "pixel_order": ["B", "G", "R"], "pixel_order_indices": {"B": 0, "G": 1, "R": 2},
        }
        msf.optimal_dye_selector_simulated(
            potential_dyes=["Cy3B", "ATTO 643"],
            single_molecule_dyes=SINGLE_MOLECULE_DYES,
            filters=FILTERS, camera_parameters=cp, wavelength=wl,
            n_dyes_desired=2, min_photons_per_100ms=500, n_simulations=10,
            smoothing_function=smoothing_function, exhaustive_search=True, verbose=True,
        )


# ======================================================================
# _simulate_dye_color_distributions -- direct calls for gaps
# ======================================================================

class TestSimulateDyeColorDistributions:
    def test_default_pixel_size_none(self, wavelength_and_qys, smoothing_function):
        wl, pixel_QYs = wavelength_and_qys
        msf = MultiC_Sim_Funcs(camera="ximea")
        M_F = MaskFunctions.Mask_Functions()
        cp = {
            "gain": np.full((12, 12), 1.0), "offset": np.full((12, 12), 100.0),
            "variance": np.full((12, 12), 1e-12), "readnoise": 1e-12,
            "rqe": np.full((12, 12), 1.0), "pixel_QYs": pixel_QYs,
            "masks": M_F.get_masks(size_x=12, size_y=12),
            "pixel_order": ["B", "G", "R"], "pixel_order_indices": {"B": 0, "G": 1, "R": 2},
        }
        result = msf._simulate_dye_color_distributions(
            "Cy3B", FILTERS, cp, wl, expected_photons=5000.0,
            n_simulations=3, image_dims=12, smoothing_function=smoothing_function,
            pixel_size=None,
        )
        assert "A_R" in result

    def test_failed_and_zero_amplitude_fits_marked_nan(self, wavelength_and_qys, smoothing_function, monkeypatch):
        wl, pixel_QYs = wavelength_and_qys
        msf = MultiC_Sim_Funcs(camera="ximea")
        M_F = MaskFunctions.Mask_Functions()
        cp = {
            "gain": np.full((12, 12), 1.0), "offset": np.full((12, 12), 100.0),
            "variance": np.full((12, 12), 1e-12), "readnoise": 1e-12,
            "rqe": np.full((12, 12), 1.0), "pixel_QYs": pixel_QYs,
            "masks": M_F.get_masks(size_x=12, size_y=12),
            "pixel_order": ["B", "G", "R"], "pixel_order_indices": {"B": 0, "G": 1, "R": 2},
        }
        n_sim = 3

        def fake_fit(*args, **kwargs):
            # columns: xc,yc,s_x,s_y,bg_B,bg_G,bg_R,A_B,A_G,A_R,chi_sqr,frame
            results = np.zeros((n_sim, 12))
            errors = np.zeros((n_sim, 10))
            results[0, 9] = np.nan  # row 0: failed fit (A_R is NaN)
            results[1, 7:10] = [0.0, 0.0, 0.0]  # row 1: fit ok but zero amplitude
            results[2, 7:10] = [1.0, 2.0, 3.0]  # row 2: real fit
            return results, errors

        monkeypatch.setattr(msf.image_analysis, "fit_puncta_parallel_method", fake_fit)
        result = msf._simulate_dye_color_distributions(
            "Cy3B", FILTERS, cp, wl, expected_photons=5000.0,
            n_simulations=n_sim, image_dims=12, smoothing_function=smoothing_function,
        )
        assert np.isnan(result["A_R"][0])
        assert np.isnan(result["A_R"][1])
        assert not np.isnan(result["A_R"][2])


# ======================================================================
# Remaining gap-fill: _perform_fitting / _build_frame_masks
# ======================================================================

class TestPerformFittingAndFrameMasks:
    def test_unknown_strategy_raises(self, sim):
        with pytest.raises(ValueError, match="Unknown fitting strategy"):
            sim._perform_fitting("bogus", None, None, None, None, None, None)

    def test_build_frame_masks_3d(self, sim, camera_parameters):
        masks_3d = {c: np.stack([m, m, m]) for c, m in camera_parameters["masks"].items()}
        out = sim._build_frame_masks(masks_3d, 3)
        assert len(out) == 3
        assert out[0].shape[-1] == 3


# ======================================================================
# _add_nile_red_wavelength_fits -- direct calls
# ======================================================================

class TestAddNileRedWavelengthFits:
    def _base_fit_results(self, n=2):
        return pd.DataFrame({
            "A_R": [0.5, 0.0], "A_G": [0.3, 0.0], "A_B": [0.2, 0.0],
            "A_R_err": [0.01, 0.01], "A_G_err": [0.01, 0.01], "A_B_err": [0.01, 0.01],
            "s_x": [1.5, 1.5], "s_y": [1.5, 1.5],
            "s_x_err": [0.1, 0.1], "s_y_err": [0.1, 0.1],
            "photons": [5000.0, 0.0], "background_photons": [50.0, 0.0],
        })

    def test_invalid_rgb_total_skipped(self, sim, camera_parameters, wavelength_and_qys):
        wl, _ = wavelength_and_qys
        cp = CameraParameters.validate_and_create(camera_parameters)
        fit_results = self._base_fit_results()
        config = SimulationConfig(pixel_size=69.0, NA=1.49, cpu_fraction=0.1)
        out = sim._add_nile_red_wavelength_fits(
            fit_results, 620.0, cp, camera_parameters, wl, FILTERS, config,
        )
        # Row 1 (all-zero RGB) must stay NaN -- never reaches the fitter.
        assert np.isnan(out["wl_fit"].iloc[1])

    def test_fit_failure_inside_pool_caught(self, sim, camera_parameters, wavelength_and_qys, monkeypatch):
        import pyS3M.simulation.multicolour as mc
        wl, _ = wavelength_and_qys
        cp = CameraParameters.validate_and_create(camera_parameters)
        fit_results = self._base_fit_results()
        config = SimulationConfig(pixel_size=69.0, NA=1.49, cpu_fraction=0.1)

        def raiser(*args, **kwargs):
            raise RuntimeError("forced fit failure")

        monkeypatch.setattr(mc, "_fit_nile_red_wavelength_standalone", raiser)
        out = sim._add_nile_red_wavelength_fits(
            fit_results, 620.0, cp, camera_parameters, wl, FILTERS, config,
        )
        assert np.isnan(out["wl_fit"].iloc[0])

    def test_outer_exception_returns_unchanged(self, sim, camera_parameters, wavelength_and_qys):
        wl, _ = wavelength_and_qys
        cp = CameraParameters.validate_and_create(camera_parameters)
        fit_results = self._base_fit_results()
        config = SimulationConfig(pixel_size=69.0, NA=1.49, cpu_fraction=0.1)
        # A nonexistent filter name makes get_dye_or_filter_data raise inside
        # the try block, hitting the outer except and returning fit_results
        # completely unchanged (no wl_fit column added at all).
        out = sim._add_nile_red_wavelength_fits(
            fit_results, 620.0, cp, camera_parameters, wl, ["nonexistent_filter_xyz"], config,
        )
        assert "wl_fit" not in out.columns


# ======================================================================
# gen_camera_image_stack -- remaining scalar/multi-molecule/QE edge cases
# ======================================================================

class TestGenCameraImageStackEdgeCases:
    def _uniform_qy_camera_params(self, n_wl, size=8):
        # Same QY for every channel -> hits the "all_QE_equal" fast path.
        return _camera_params_dict(size=size, pixel_QYs=np.full((3, n_wl), 0.5))

    def test_scalar_n_photons_and_dye_efficiency_vectorized(self, sim, camera_parameters, wavelength_and_qys, smoothing_function):
        wl, _ = wavelength_and_qys
        x0y0 = {"dye": np.zeros((2, 2, 1))}
        x0y0["dye"][:, 0, 0] = 4.0
        x0y0["dye"][:, 1, 0] = 4.0
        # 0-d array (no meaningful __getitem__[frame]) and scalar dye_pixel_efficiency
        # both force the except-fallback branches.
        n_photons = {"dye": np.array(2000)}
        bayer, smoothed, _ = sim.gen_camera_image_stack(
            camera_parameters, wl, 660.0, np.float64(0.5),
            n_photons, x0y0, smoothing_function=smoothing_function, background_photons=5.0,
        )
        # A 0-d n_photons array has no meaningful `.shape[0]`, forcing the
        # except-fallback `s = 1`; np.squeeze then drops that leading axis.
        assert bayer.shape == (8, 8)

    def test_scalar_n_photons_and_dye_efficiency_non_vectorized(self, sim, camera_parameters, wavelength_and_qys, smoothing_function):
        wl, _ = wavelength_and_qys
        x0y0 = {"dye": np.zeros((2, 2, 1))}
        x0y0["dye"][:, 0, 0] = 4.0
        x0y0["dye"][:, 1, 0] = 4.0
        n_photons = {"dye": np.array(2000)}
        bayer, smoothed, _ = sim.gen_camera_image_stack(
            camera_parameters, wl, 660.0, np.float64(0.5),
            n_photons, x0y0, smoothing_function=smoothing_function, background_photons=5.0,
            use_vectorized_photoelectrons=False,
        )
        assert bayer.shape == (8, 8)

    def test_x0y0_1d_indexerror_fallback_vectorized(self, sim, camera_parameters, wavelength_and_qys, smoothing_function):
        wl, _ = wavelength_and_qys
        # A 1-D x0y0 array (ndim==1) makes the `[1, :]` 2-index access raise
        # IndexError, exercising the except-fallback branch.
        x0y0 = {"dye": np.array([4.0, 4.0])}
        n_photons = {"dye": np.array([2000])}
        bayer, smoothed, _ = sim.gen_camera_image_stack(
            camera_parameters, wl, 660.0, np.array([0.2, 0.3, 0.5]),
            n_photons, x0y0, smoothing_function=smoothing_function, background_photons=5.0,
        )
        # n_photons["dye"] has a single frame, so s=1 and np.squeeze drops it.
        assert bayer.shape == (8, 8)

    def test_x0y0_1d_indexerror_fallback_non_vectorized(self, sim, camera_parameters, wavelength_and_qys, smoothing_function):
        wl, _ = wavelength_and_qys
        x0y0 = {"dye": np.array([4.0, 4.0])}
        n_photons = {"dye": np.array([2000])}
        bayer, smoothed, _ = sim.gen_camera_image_stack(
            camera_parameters, wl, 660.0, np.array([0.2, 0.3, 0.5]),
            n_photons, x0y0, smoothing_function=smoothing_function, background_photons=5.0,
            use_vectorized_photoelectrons=False,
        )
        assert bayer.shape == (8, 8)

    def test_multiple_molecules_per_dye_vectorized(self, sim, camera_parameters, wavelength_and_qys, smoothing_function):
        wl, _ = wavelength_and_qys
        x0y0 = {"dye": np.zeros((1, 2, 3))}
        x0y0["dye"][0, 0, :] = [2.0, 4.0, 6.0]
        x0y0["dye"][0, 1, :] = [2.0, 4.0, 6.0]
        n_photons = {"dye": np.array([2000])}
        bayer, smoothed, _ = sim.gen_camera_image_stack(
            camera_parameters, wl, 660.0, np.array([0.2, 0.3, 0.5]),
            n_photons, x0y0, smoothing_function=smoothing_function, background_photons=5.0,
        )
        assert bayer.shape == (8, 8)

    def test_multiple_molecules_per_dye_non_vectorized(self, sim, camera_parameters, wavelength_and_qys, smoothing_function):
        wl, _ = wavelength_and_qys
        x0y0 = {"dye": np.zeros((1, 2, 3))}
        x0y0["dye"][0, 0, :] = [2.0, 4.0, 6.0]
        x0y0["dye"][0, 1, :] = [2.0, 4.0, 6.0]
        n_photons = {"dye": np.array([2000])}
        bayer, smoothed, _ = sim.gen_camera_image_stack(
            camera_parameters, wl, 660.0, np.array([0.2, 0.3, 0.5]),
            n_photons, x0y0, smoothing_function=smoothing_function, background_photons=5.0,
            use_vectorized_photoelectrons=False,
        )
        assert bayer.shape == (8, 8)

    def test_defocus_and_motion_combined_vectorized(self, sim, camera_parameters, wavelength_and_qys, smoothing_function):
        wl, _ = wavelength_and_qys
        x0y0, n_photons = _minimal_x0y0_photons()
        bayer, smoothed, _ = sim.gen_camera_image_stack(
            camera_parameters, wl, 660.0, np.array([0.2, 0.3, 0.5]),
            n_photons, x0y0, smoothing_function=smoothing_function, background_photons=5.0,
            defocus_z_um=0.3, motion_velocity_nm_per_s=5e6, frame_exposure_ms=100.0,
        )
        assert bayer.shape == (2, 8, 8)

    def test_defocus_only_non_vectorized(self, sim, camera_parameters, wavelength_and_qys, smoothing_function):
        wl, _ = wavelength_and_qys
        x0y0, n_photons = _minimal_x0y0_photons()
        bayer, smoothed, _ = sim.gen_camera_image_stack(
            camera_parameters, wl, 660.0, np.array([0.2, 0.3, 0.5]),
            n_photons, x0y0, smoothing_function=smoothing_function, background_photons=5.0,
            defocus_z_um=0.3, use_vectorized_photoelectrons=False,
        )
        assert bayer.shape == (2, 8, 8)

    def test_uniform_qe_fast_path_non_vectorized(self, sim, wavelength_and_qys, smoothing_function):
        wl, _ = wavelength_and_qys
        cp = self._uniform_qy_camera_params(n_wl=len(wl))
        x0y0, n_photons = _minimal_x0y0_photons()
        bayer, smoothed, _ = sim.gen_camera_image_stack(
            cp, wl, 660.0, np.array([0.5, 0.5, 0.5]),
            n_photons, x0y0, smoothing_function=smoothing_function, background_photons=5.0,
            use_vectorized_photoelectrons=False,
        )
        assert bayer.shape == (2, 8, 8)

    def test_return_normal_image_vectorized_photoelectrons_false(self, sim, camera_parameters, wavelength_and_qys, smoothing_function):
        wl, _ = wavelength_and_qys
        x0y0, n_photons = _minimal_x0y0_photons()
        bayer, smoothed, normal = sim.gen_camera_image_stack(
            camera_parameters, wl, 660.0, np.array([0.2, 0.3, 0.5]),
            n_photons, x0y0, smoothing_function=smoothing_function, background_photons=5.0,
            return_normal_image=True, return_photoelectrons=False,
        )
        assert normal.shape == (2, 8, 8)

    def test_return_normal_image_vectorized_photoelectrons_true(self, sim, camera_parameters, wavelength_and_qys, smoothing_function):
        wl, _ = wavelength_and_qys
        x0y0, n_photons = _minimal_x0y0_photons()
        bayer, smoothed, normal = sim.gen_camera_image_stack(
            camera_parameters, wl, 660.0, np.array([0.2, 0.3, 0.5]),
            n_photons, x0y0, smoothing_function=smoothing_function, background_photons=5.0,
            return_normal_image=True, return_photoelectrons=True,
        )
        assert normal.shape == (2, 8, 8)


# ======================================================================
# test_simulation_method_2d_sweep -- remaining gaps
# ======================================================================

class TestSimulationMethod2DSweepGaps:
    def test_config_none_default(self, sim, camera_parameters, wavelength_and_qys, smoothing_function, tmp_path, monkeypatch):
        import pyS3M.simulation.multicolour as mc
        # SimulationConfig()'s real default is n_bootstrap=100_000 -- far too
        # slow for a unit test. Patch the "config is None" branch's factory so
        # it still exercises the default-construction code path, just with a
        # tiny bootstrap count.
        monkeypatch.setattr(mc, "SimulationConfig", lambda **kw: SimulationConfig(n_bootstrap=2, **kw))
        wl, _ = wavelength_and_qys
        sim.test_simulation_method_2d_sweep(
            dye=DYE, filters=FILTERS, wavelength=wl, camera_parameters=camera_parameters,
            save_folder=str(tmp_path), n_photon_space=np.array([2000.0]),
            smoothing_function=smoothing_function, strategy=FittingStrategy.STANDARD,
            read_noise_space=np.array([1.0]), peak_qy_space=np.array([1.0]),
        )

    def test_subtractx0y0(self, sim, camera_parameters, wavelength_and_qys, smoothing_function, tmp_path):
        wl, _ = wavelength_and_qys
        config = SimulationConfig(
            n_bootstrap=2, save_raw_results=True, verbose=False,
            use_stochastic_photons=False, n_unit_cells=0, subtractx0y0=True,
        )
        sim.test_simulation_method_2d_sweep(
            dye=DYE, filters=FILTERS, wavelength=wl, camera_parameters=camera_parameters,
            save_folder=str(tmp_path), n_photon_space=np.array([2000.0]),
            smoothing_function=smoothing_function, strategy=FittingStrategy.STANDARD,
            read_noise_space=np.array([1.0]), peak_qy_space=np.array([1.0]),
            config=config,
        )

    def test_overwrite_true_deletes_existing_raw_results(self, sim, camera_parameters, wavelength_and_qys, smoothing_function, tmp_path):
        wl, _ = wavelength_and_qys
        config = SimulationConfig(
            n_bootstrap=2, save_raw_results=True, verbose=False,
            use_stochastic_photons=False, n_unit_cells=0,
        )
        common = dict(
            dye=DYE, filters=FILTERS, wavelength=wl, camera_parameters=camera_parameters,
            save_folder=str(tmp_path), n_photon_space=np.array([2000.0]),
            smoothing_function=smoothing_function, strategy=FittingStrategy.STANDARD,
            read_noise_space=np.array([1.0]), peak_qy_space=np.array([1.0]),
            config=config,
        )
        sim.test_simulation_method_2d_sweep(**common, overwrite=True)
        # Second call, overwrite=True again with an existing raw-results file at
        # photon level 0 -> hits the "delete existing raw-results file" branch.
        sim.test_simulation_method_2d_sweep(**common, overwrite=True)

    def test_multi_photon_level_appends(self, sim, camera_parameters, wavelength_and_qys, smoothing_function, tmp_path):
        wl, _ = wavelength_and_qys
        config = SimulationConfig(
            n_bootstrap=2, save_raw_results=True, verbose=False,
            use_stochastic_photons=False, n_unit_cells=0,
        )
        sim.test_simulation_method_2d_sweep(
            dye=DYE, filters=FILTERS, wavelength=wl, camera_parameters=camera_parameters,
            save_folder=str(tmp_path), n_photon_space=np.array([2000.0, 3000.0]),
            smoothing_function=smoothing_function, strategy=FittingStrategy.STANDARD,
            read_noise_space=np.array([1.0]), peak_qy_space=np.array([1.0]),
            config=config, overwrite=True,
        )
        assert any(tmp_path.glob("*rawresults.h5"))


# ======================================================================
# test_simulation_method -- remaining continuation/overwrite gaps
# ======================================================================

class TestSimulationMethodGaps:
    def test_config_none_default(self, sim, camera_parameters, wavelength_and_qys, smoothing_function, tmp_path, monkeypatch):
        import pyS3M.simulation.multicolour as mc
        # SimulationConfig()'s real default is n_bootstrap=100_000 -- far too
        # slow for a unit test. Patch the "config is None" branch's factory so
        # it still exercises the default-construction code path, just with a
        # tiny bootstrap count.
        monkeypatch.setattr(mc, "SimulationConfig", lambda **kw: SimulationConfig(n_bootstrap=2, **kw))
        wl, _ = wavelength_and_qys
        sim.test_simulation_method(
            dye=DYE, filters=FILTERS, wavelength=wl, camera_parameters=camera_parameters,
            save_folder=str(tmp_path), n_photon_space=np.array([2000.0]),
            smoothing_function=smoothing_function, strategy=FittingStrategy.STANDARD,
        )

    def test_overwrite_true_deletes_existing_results(self, sim, camera_parameters, wavelength_and_qys, smoothing_function, tmp_path):
        wl, _ = wavelength_and_qys
        config = SimulationConfig(
            n_bootstrap=2, background_photons=5.0, save_raw_results=True,
            save_summary_csvs=False, verbose=False, use_stochastic_photons=False,
        )
        common = dict(
            dye=DYE, filters=FILTERS, wavelength=wl, camera_parameters=camera_parameters,
            save_folder=str(tmp_path), n_photon_space=np.array([2000.0]),
            smoothing_function=smoothing_function, strategy=FittingStrategy.STANDARD,
            config=config,
        )
        sim.test_simulation_method(**common, overwrite=True)
        # Second call, overwrite=True again -> hits the "delete existing file" branch.
        sim.test_simulation_method(**common, overwrite=True)

    def test_overwrite_false_corrupt_existing_h5(self, sim, camera_parameters, wavelength_and_qys, smoothing_function, tmp_path, monkeypatch):
        wl, _ = wavelength_and_qys
        config = SimulationConfig(
            n_bootstrap=2, background_photons=5.0, save_raw_results=True,
            save_summary_csvs=False, verbose=False, use_stochastic_photons=False,
        )
        common = dict(
            dye=DYE, filters=FILTERS, wavelength=wl, camera_parameters=camera_parameters,
            save_folder=str(tmp_path), n_photon_space=np.array([2000.0]),
            smoothing_function=smoothing_function, strategy=FittingStrategy.STANDARD,
            config=config,
        )
        sim.test_simulation_method(**common, overwrite=True)

        def raiser(*args, **kwargs):
            raise ValueError("forced corrupt read")

        monkeypatch.setattr(sim.io, "read_h5_database", raiser)
        sim.test_simulation_method(**common, overwrite=False)

    def test_overwrite_false_skips_mid_loop_and_processes_new_level(self, sim, camera_parameters, wavelength_and_qys, smoothing_function, tmp_path):
        wl, _ = wavelength_and_qys
        config1 = SimulationConfig(
            n_bootstrap=2, background_photons=5.0, save_raw_results=True,
            save_summary_csvs=False, verbose=False, use_stochastic_photons=False,
        )
        sim.test_simulation_method(
            dye=DYE, filters=FILTERS, wavelength=wl, camera_parameters=camera_parameters,
            save_folder=str(tmp_path), n_photon_space=np.array([2000.0]),
            smoothing_function=smoothing_function, strategy=FittingStrategy.STANDARD,
            config=config1, overwrite=True,
        )
        # Now request 2 photon levels with overwrite=False: level 0 is already
        # completed (skip branch), level 1 is new (processed + appended).
        sim.test_simulation_method(
            dye=DYE, filters=FILTERS, wavelength=wl, camera_parameters=camera_parameters,
            save_folder=str(tmp_path), n_photon_space=np.array([2000.0, 3000.0]),
            smoothing_function=smoothing_function, strategy=FittingStrategy.STANDARD,
            config=config1, overwrite=False,
        )

    def test_overwrite_false_groundtruth_bootstrap_mismatch(self, sim, camera_parameters, wavelength_and_qys, smoothing_function, tmp_path):
        wl, _ = wavelength_and_qys
        config1 = SimulationConfig(
            n_bootstrap=2, background_photons=5.0, save_raw_results=False,
            save_summary_csvs=False, verbose=False, use_stochastic_photons=False,
        )
        sim.test_simulation_method(
            dye=DYE, filters=FILTERS, wavelength=wl, camera_parameters=camera_parameters,
            save_folder=str(tmp_path), n_photon_space=np.array([2000.0]),
            smoothing_function=smoothing_function, strategy=FittingStrategy.STANDARD,
            config=config1, overwrite=True,
        )
        # Different n_bootstrap -> saved groundtruth length mismatch -> treated as fresh.
        config2 = SimulationConfig(
            n_bootstrap=3, background_photons=5.0, save_raw_results=False,
            save_summary_csvs=False, verbose=False, use_stochastic_photons=False,
        )
        sim.test_simulation_method(
            dye=DYE, filters=FILTERS, wavelength=wl, camera_parameters=camera_parameters,
            save_folder=str(tmp_path), n_photon_space=np.array([2000.0]),
            smoothing_function=smoothing_function, strategy=FittingStrategy.STANDARD,
            config=config2, overwrite=False,
        )

    def test_subtractx0y0(self, sim, camera_parameters, wavelength_and_qys, smoothing_function, tmp_path):
        wl, _ = wavelength_and_qys
        config = SimulationConfig(
            n_bootstrap=2, background_photons=5.0, save_raw_results=True,
            save_summary_csvs=False, verbose=False, use_stochastic_photons=False,
            subtractx0y0=True,
        )
        sim.test_simulation_method(
            dye=DYE, filters=FILTERS, wavelength=wl, camera_parameters=camera_parameters,
            save_folder=str(tmp_path), n_photon_space=np.array([2000.0]),
            smoothing_function=smoothing_function, strategy=FittingStrategy.STANDARD,
            config=config,
        )


# ======================================================================
# Full-chip calibration sampling (per-bootstrap real-sensor noise crops)
# ======================================================================

from pyS3M.simulation.multicolour import (  # noqa: E402
    sample_chip_patches_batch,
    sanitise_calibration_maps,
)


def _chip_maps(shape=(30, 40), seed=0):
    rng = np.random.default_rng(seed)
    return {
        "gain": rng.uniform(0.9, 1.1, shape),
        "offset": np.full(shape, 100.0),
        "variance": rng.uniform(0.5, 1.5, shape),
        "rqe": np.ones(shape),
    }


class TestChipSamplingHelpers:
    def test_crops_share_one_location_across_maps(self):
        chip = {"a": np.arange(30 * 40).reshape(30, 40), "b": -np.arange(30 * 40).reshape(30, 40)}
        crops, r0, c0 = sample_chip_patches_batch(chip, (5, 6), 50, np.random.default_rng(1))
        assert crops["a"].shape == (50, 5, 6)
        np.testing.assert_array_equal(crops["a"], -crops["b"])
        for k in range(50):
            np.testing.assert_array_equal(crops["a"][k], chip["a"][r0[k]:r0[k] + 5, c0[k]:c0[k] + 6])

    def test_offsets_reach_last_valid_position_and_stay_in_bounds(self):
        chip = {"a": np.zeros((6, 7))}
        _, r0, c0 = sample_chip_patches_batch(chip, (5, 5), 2000, np.random.default_rng(2))
        assert r0.min() == 0 and r0.max() == 1
        assert c0.min() == 0 and c0.max() == 2

    def test_given_offsets_reproduce_draw(self):
        chip = _chip_maps()
        crops1, r0, c0 = sample_chip_patches_batch(chip, (4, 4), 10, np.random.default_rng(3))
        crops2, _, _ = sample_chip_patches_batch(chip, (4, 4), 10, None, offsets=(r0, c0))
        np.testing.assert_array_equal(crops1["gain"], crops2["gain"])

    def test_patch_larger_than_chip_raises(self):
        with pytest.raises(ValueError, match="larger than chip"):
            sample_chip_patches_batch({"a": np.zeros((4, 4))}, (5, 5), 1, np.random.default_rng())

    def test_sanitise_replaces_only_invalid_pixels(self):
        chip = _chip_maps()
        chip["gain"][0, 0] = -20.0
        chip["rqe"][1, 1] = 0.0
        chip["variance"][2, 2] = np.nan
        chip["variance"][3, 3] = 44_000.0  # hot pixel: real, must be kept
        out, n = sanitise_calibration_maps(chip)
        assert n == 3
        assert out["gain"][0, 0] > 0 and out["rqe"][1, 1] > 0 and np.isfinite(out["variance"][2, 2])
        assert out["variance"][3, 3] == pytest.approx(44_000.0)
        assert out["gain"].dtype == np.float32

    def test_sanitise_shape_and_key_errors(self):
        with pytest.raises(ValueError, match="missing"):
            sanitise_calibration_maps({"gain": np.ones((3, 3))})
        bad = _chip_maps()
        bad["rqe"] = np.ones((5, 5))
        with pytest.raises(ValueError, match="share one 2D shape"):
            sanitise_calibration_maps(bad)


class TestPerFrameCalibrationGeneration:
    @pytest.mark.parametrize("vectorized", [True, False])
    def test_hot_offset_pixel_lands_in_its_own_frame(
        self, sim, camera_parameters, wavelength_and_qys, smoothing_function, vectorized
    ):
        wl, _ = wavelength_and_qys
        x0y0, n_photons = _minimal_x0y0_photons()
        cp = dict(camera_parameters)
        n = 2
        for k in ("gain", "rqe"):
            cp[k] = np.stack([camera_parameters[k]] * n)
        cp["variance"] = np.stack([camera_parameters["variance"]] * n)
        cp["offset"] = np.stack([camera_parameters["offset"]] * n)
        cp["offset"][1, 0, 0] = 5000.0  # hot offset only in frame 1
        bayer, _, _ = sim.gen_camera_image_stack(
            cp, wl, 660.0, np.array([0.2, 0.3, 0.5]), n_photons, x0y0,
            smoothing_function=smoothing_function, background_photons=5.0,
            use_vectorized_photoelectrons=vectorized,
        )
        assert bayer.shape == (2, 8, 8)
        assert bayer[1, 0, 0] >= 5000
        assert bayer[0, 0, 0] < 1000


class TestSimulationMethodFullChip:
    def _config(self, **kw):
        base = dict(
            n_bootstrap=3, background_photons=5.0, save_raw_results=True,
            save_summary_csvs=False, verbose=False, use_stochastic_photons=False,
            n_unit_cells=4, chip_sampling_seed=7,
        )
        base.update(kw)
        return SimulationConfig(**base)

    def _cp(self, sim, pixel_QYs):
        cp = _camera_params_dict(size=8, mosaic_unit=sim.mosaic_unit, pixel_QYs=pixel_QYs)
        cp["full_chip_calibration"] = _chip_maps()
        return cp

    def test_runs_and_saves_crop_offsets(self, sim, wavelength_and_qys, smoothing_function, tmp_path):
        wl, pixel_QYs = wavelength_and_qys
        sim.test_simulation_method(
            dye=DYE, filters=FILTERS, wavelength=wl, camera_parameters=self._cp(sim, pixel_QYs),
            save_folder=str(tmp_path), n_photon_space=np.array([2000.0, 4000.0]),
            smoothing_function=smoothing_function, strategy=FittingStrategy.STANDARD,
            config=self._config(),
        )
        offsets = pd.read_csv(next(tmp_path.glob("*chipcrop_offsets.csv")))
        assert len(offsets) == 3
        assert offsets["row0"].between(0, 30 - 8).all() and offsets["col0"].between(0, 40 - 8).all()
        raw = pd.read_hdf(next(tmp_path.glob("*rawresults.h5")))
        assert set(raw["photon_level"]) == {0, 1}

    def test_continuation_reuses_saved_offsets(self, sim, wavelength_and_qys, smoothing_function, tmp_path):
        wl, pixel_QYs = wavelength_and_qys
        common = dict(
            dye=DYE, filters=FILTERS, wavelength=wl, camera_parameters=self._cp(sim, pixel_QYs),
            save_folder=str(tmp_path), n_photon_space=np.array([2000.0]),
            smoothing_function=smoothing_function, strategy=FittingStrategy.STANDARD,
        )
        sim.test_simulation_method(**common, config=self._config(save_raw_results=False), overwrite=True)
        path = next(tmp_path.glob("*chipcrop_offsets.csv"))
        first = pd.read_csv(path)
        sim.test_simulation_method(
            **common, config=self._config(save_raw_results=False, chip_sampling_seed=999),
            overwrite=False,
        )
        pd.testing.assert_frame_equal(first, pd.read_csv(path))

    def test_unsupported_strategy_raises(self, sim, wavelength_and_qys, smoothing_function, tmp_path):
        wl, pixel_QYs = wavelength_and_qys
        with pytest.raises(SimulationValidationError, match="only supported"):
            sim.test_simulation_method(
                dye=DYE, filters=FILTERS, wavelength=wl, camera_parameters=self._cp(sim, pixel_QYs),
                save_folder=str(tmp_path), n_photon_space=np.array([2000.0]),
                smoothing_function=smoothing_function, strategy=FittingStrategy.DEMOSAIC,
                config=self._config(save_raw_results=False),
            )


class TestBackgroundDefinition:
    """background_photons = mean photons sensed per pixel (QE = 1, dye-independent)."""

    def _bg_pe(self, sim, camera_parameters, wl, dpe, colour, n=4000, n_photons=0):
        x0y0, _ = _minimal_x0y0_photons(n_bootstrap=n)
        return sim.gen_camera_image_stack(
            camera_parameters, wl, 600.0, np.asarray(dpe), {"dye": np.full(n, n_photons)},
            x0y0, background_photons=5.0, background_colour=colour,
            return_photoelectrons_stack=True,
        )

    @pytest.mark.parametrize("dpe", [[0.12, 0.61, 0.09], [0.04, 0.14, 0.49]])
    def test_each_pixel_senses_background_photons_for_any_dye(
        self, sim, camera_parameters, wavelength_and_qys, dpe
    ):
        wl, _ = wavelength_and_qys
        pe = self._bg_pe(sim, camera_parameters, wl, dpe, [1, 1, 1])
        for c in "BGR":
            assert pe[:, camera_parameters["masks"][c]].mean() == pytest.approx(5.0, rel=0.03)

    def test_colour_scaled_by_max(self, sim, camera_parameters, wavelength_and_qys):
        wl, _ = wavelength_and_qys
        pe = self._bg_pe(sim, camera_parameters, wl, [0.1, 0.5, 0.3], [1, 2, 1])  # B, G, R
        m = camera_parameters["masks"]
        assert pe[:, m["G"]].mean() == pytest.approx(5.0, rel=0.03)
        assert pe[:, m["B"]].mean() == pytest.approx(2.5, rel=0.03)
        assert pe[:, m["R"]].mean() == pytest.approx(2.5, rel=0.03)

    def test_bad_background_colour_raises(self, sim, camera_parameters, wavelength_and_qys):
        wl, _ = wavelength_and_qys
        with pytest.raises(SimulationValidationError, match="background_colour"):
            self._bg_pe(sim, camera_parameters, wl, [0.1, 0.5, 0.3], [1, 1], n=2)
