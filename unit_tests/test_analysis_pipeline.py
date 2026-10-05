#!/usr/bin/env python3
"""
Full coverage tests for pyS3M.AnalysisPipeline — the packaged public-facing API.

Every example notebook under notebooks/analyses/ exercises this class end-to-end
and is verified by eye, but until this file nothing *asserted* on it in pytest.
Strategy for keeping this fast (see claude/TODO.md PRIORITY 1):

- One real, fixture-based end-to-end fit (test_tiffs/single_FOV_smlm/, ~20s) is run
  once per test session and reused by every test that needs real localisations
  (mirrors notebooks/analyses/01's own parameters) -- this is the one test whose
  failure would mean the *real* pipeline is broken, not just this orchestration
  layer.
- Pure orchestration branches (bad mode/method, calibration-missing, callback
  wiring, undrift passthrough, calibrate() success/failure) are tested with mocks
  or tiny synthetic data -- they exercise AnalysisPipeline's own logic without
  paying for a second real fit or a real flat-field calibration run.
- load_localisations[_per_fov] are tested entirely against tiny synthetic HDF5
  files built in tmp_path -- no image data needed at all.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import pyS3M.CalibrationFunctions as CalibrationFunctions
from pyS3M.AnalysisPipeline import AnalysisPipeline, FittingConfig
from pyS3M.Constants import AnalysisConfig, FilteringCriteria
from pyS3M.clustering import ClusteringConfig

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "test_tiffs" / "single_FOV_smlm"
CAL_DIR = PROJECT_ROOT / "Camera_Calibrations" / "Ximea_Camera"


# ======================================================================
# Shared expensive fixture: one real fit, reused across the whole module
# ======================================================================

@pytest.fixture(scope="module")
def fitted_pipeline(tmp_path_factory):
    """Real end-to-end fit on a copy of the small single_FOV_smlm fixture.

    Copies into a tmp dir first so the HDF5 output fit() writes alongside the
    TIFF doesn't land in (and pollute) the repo's test_tiffs/ tree.
    """
    assert DATA_DIR.is_dir(), f"Bundled fixture missing: {DATA_DIR}"
    assert CAL_DIR.is_dir(), f"Bundled calibration missing: {CAL_DIR}"

    work_dir = tmp_path_factory.mktemp("analysis_pipeline_fit")
    for f in DATA_DIR.iterdir():
        if f.is_file():
            shutil.copy(f, work_dir / f.name)

    calls: list[tuple] = []
    cfg = AnalysisConfig(
        display=False,
        progress_callback=lambda frac, msg: calls.append(("progress", frac, msg)),
        logging_callback=lambda msg: calls.append(("log", msg)),
    )
    pipe = AnalysisPipeline(camera="ximea", config=cfg)
    pipe.load_calibration(CAL_DIR)

    fc = FittingConfig(peak_wavelength=0.685, pfa=1e-3)
    fit_return = pipe.fit(work_dir, mode="smlm", fitting_config=fc)
    locs = pipe.load_localisations(work_dir)

    return {
        "pipe": pipe,
        "work_dir": work_dir,
        "locs": locs,
        "fit_return": fit_return,
        "callback_calls": calls,
    }


# ======================================================================
# __init__ / camera + pixel_size + config resolution
# ======================================================================

class TestInit:
    def test_defaults_resolve_pixel_size_and_config_from_camera(self):
        pipe = AnalysisPipeline(camera="ximea")
        assert pipe.camera == "ximea"
        assert pipe.pixel_size == pytest.approx(0.069)
        assert isinstance(pipe.config, AnalysisConfig)
        assert pipe.gain_map is None
        assert pipe.offset_map is None
        assert pipe.rqe is None
        assert pipe.read_noise is None
        assert pipe.variance is None

    def test_explicit_pixel_size_overrides_camera_default(self):
        pipe = AnalysisPipeline(camera="zwo", pixel_size=0.123)
        assert pipe.pixel_size == 0.123

    def test_explicit_config_is_stored_as_is(self):
        cfg = AnalysisConfig(display=False, dpi=72)
        pipe = AnalysisPipeline(config=cfg)
        assert pipe.config is cfg


# ======================================================================
# Lazy sub-function properties
# ======================================================================

class TestLazyProperties:
    def test_sr_property_creates_and_caches(self):
        pipe = AnalysisPipeline(camera="ximea")
        sr1 = pipe.sr
        assert sr1 is not None
        assert pipe.sr is sr1  # cached, not rebuilt

    def test_sm_property_creates_and_caches(self):
        pipe = AnalysisPipeline(camera="ximea")
        sm1 = pipe.sm
        assert sm1 is not None
        assert pipe.sm is sm1

    def test_dcf_property_creates_and_caches(self):
        pipe = AnalysisPipeline(camera="ximea")
        dcf1 = pipe.dcf
        assert dcf1 is not None
        assert pipe.dcf is dcf1

    def test_nile_red_property_creates_and_caches(self):
        pipe = AnalysisPipeline(camera="ximea")
        nr1 = pipe.nile_red
        assert nr1 is not None
        assert pipe.nile_red is nr1


# ======================================================================
# Calibration
# ======================================================================

class TestLoadCalibration:
    def test_load_calibration_success(self):
        pipe = AnalysisPipeline(camera="ximea")
        pipe.load_calibration(CAL_DIR)
        assert pipe.gain_map is not None
        assert pipe.offset_map is not None
        assert pipe.variance is not None
        assert pipe.read_noise is not None
        assert pipe.rqe is not None
        assert pipe.gain_map.dtype == np.float32

    def test_load_calibration_accepts_str_path(self):
        pipe = AnalysisPipeline(camera="ximea")
        pipe.load_calibration(str(CAL_DIR))
        assert pipe.gain_map is not None

    def test_load_calibration_invokes_logging_callback(self):
        messages = []
        cfg = AnalysisConfig(logging_callback=messages.append)
        pipe = AnalysisPipeline(camera="ximea", config=cfg)
        pipe.load_calibration(CAL_DIR)
        assert any("Calibration loaded" in m for m in messages)

    def test_load_calibration_missing_file_raises(self, tmp_path):
        # Empty dir -> gain.tif (checked first) is missing.
        pipe = AnalysisPipeline(camera="ximea")
        with pytest.raises(FileNotFoundError, match="gain.tif"):
            pipe.load_calibration(tmp_path)


class TestCalibrate:
    def test_calibrate_success_stores_maps_and_logs(self, monkeypatch, tmp_path):
        fake_maps = tuple(np.full((4, 4), i, dtype=np.float32) for i in range(5))

        def fake_calibrate_multicolour_camera(self, directory, imtype=".tif", mode="rgb"):
            return fake_maps

        monkeypatch.setattr(
            CalibrationFunctions.Calibration_Functions,
            "calibrate_multicolour_camera",
            fake_calibrate_multicolour_camera,
        )
        messages = []
        cfg = AnalysisConfig(logging_callback=messages.append)
        pipe = AnalysisPipeline(camera="ximea", config=cfg)
        pipe.calibrate(tmp_path)

        offset, variance, gain, read_noise, rqe = fake_maps
        np.testing.assert_array_equal(pipe.offset_map, offset)
        np.testing.assert_array_equal(pipe.variance, variance)
        np.testing.assert_array_equal(pipe.gain_map, gain)
        np.testing.assert_array_equal(pipe.read_noise, read_noise)
        np.testing.assert_array_equal(pipe.rqe, rqe)
        assert any("Calibration computed" in m for m in messages)

    def test_calibrate_none_result_raises_runtime_error(self, monkeypatch, tmp_path):
        def fake_returns_none(self, directory, imtype=".tif", mode="rgb"):
            return None

        monkeypatch.setattr(
            CalibrationFunctions.Calibration_Functions,
            "calibrate_multicolour_camera",
            fake_returns_none,
        )
        pipe = AnalysisPipeline(camera="ximea")
        with pytest.raises(RuntimeError, match="returned None"):
            pipe.calibrate(tmp_path)


# ======================================================================
# make_smoothing_function
# ======================================================================

class TestMakeSmoothingFunction:
    def test_returns_namespace_with_expected_attrs(self):
        pipe = AnalysisPipeline(camera="ximea")
        sf = pipe.make_smoothing_function(sigma=2.5)
        assert callable(sf.smoothing_function)
        assert sf.args == {"sigma": 2.5}
        assert sf.data_arg == "image"
        assert sf.extent == 2.5

    def test_default_sigma(self):
        pipe = AnalysisPipeline(camera="ximea")
        sf = pipe.make_smoothing_function()
        assert sf.args == {"sigma": 1.5}


# ======================================================================
# fit() — calibration guard, mode validation, real success path
# ======================================================================

class TestFit:
    def test_fit_without_calibration_raises(self, tmp_path):
        pipe = AnalysisPipeline(camera="ximea")
        with pytest.raises(RuntimeError, match="Calibration not loaded"):
            pipe.fit(tmp_path, mode="smlm")

    def test_fit_unknown_mode_raises(self):
        pipe = AnalysisPipeline(camera="ximea")
        pipe.load_calibration(CAL_DIR)
        with pytest.raises(ValueError, match="Unknown mode"):
            pipe.fit(Path("."), mode="not_a_real_mode")

    def test_fit_dispatches_and_forwards_kwargs_without_real_fitting(self):
        """Exercise the dict-building / dispatch / progress_callback lines
        cheaply by mocking the underlying fit_* method -- SR_Functions'
        own correctness is covered separately (Tier 2)."""
        progress_events = []
        cfg = AnalysisConfig(
            progress_callback=lambda frac, msg: progress_events.append((frac, msg)),
        )
        pipe = AnalysisPipeline(camera="ximea", config=cfg)
        pipe.load_calibration(CAL_DIR)
        pipe._sr = MagicMock()
        pipe._sr.fit_imaging_data = MagicMock(return_value=None)

        fc = FittingConfig(pfa=5e-4)
        pipe.fit(Path("some_folder"), mode="imaging", fitting_config=fc, extra_flag=True)

        pipe._sr.fit_imaging_data.assert_called_once()
        args, kwargs = pipe._sr.fit_imaging_data.call_args
        assert args[0] == Path("some_folder")
        assert kwargs["pfa"] == 5e-4
        assert kwargs["extra_flag"] is True
        assert kwargs["gain_map"] is pipe.gain_map
        assert progress_events[0] == (0.0, "Starting imaging fit")
        assert progress_events[-1] == (1.0, "imaging fit complete")

    def test_fit_default_fitting_config_used_when_none_given(self):
        pipe = AnalysisPipeline(camera="ximea")
        pipe.load_calibration(CAL_DIR)
        pipe._sr = MagicMock()
        pipe._sr.fit_SM_data = MagicMock(return_value=None)

        pipe.fit(Path("x"), mode="smlm")

        _, kwargs = pipe._sr.fit_SM_data.call_args
        default = FittingConfig()
        assert kwargs["pfa"] == default.pfa
        assert kwargs["ROI_size"] == default.ROI_size
        assert kwargs["fit_min_sigma"] is None and kwargs["fit_max_sigma"] is None

    def test_fit_forwards_fit_sigma_bounds(self):
        pipe = AnalysisPipeline(camera="ximea")
        pipe.load_calibration(CAL_DIR)
        pipe._sr = MagicMock()
        pipe._sr.fit_SM_data = MagicMock(return_value=None)

        pipe.fit(Path("x"), mode="smlm", fitting_config=FittingConfig(fit_min_sigma=0.7, fit_max_sigma=3.6))

        _, kwargs = pipe._sr.fit_SM_data.call_args
        assert kwargs["fit_min_sigma"] == 0.7 and kwargs["fit_max_sigma"] == 3.6

    def test_fit_real_end_to_end_smlm(self, fitted_pipeline):
        """Mirrors notebooks/analyses/01: real fit, real localisations."""
        locs = fitted_pipeline["locs"]
        assert len(locs) > 0
        assert "frame" in locs.columns
        assert "xc" in locs.columns and "yc" in locs.columns

        # fit() returns load_localisations_per_fov(image_folder)
        fit_return = fitted_pipeline["fit_return"]
        assert isinstance(fit_return, list)
        assert len(fit_return) == 1
        fov_df, tif_path = fit_return[0]
        assert len(fov_df) == len(locs)
        assert tif_path is not None and Path(tif_path).is_file()

        # progress_callback and logging_callback (set on this fixture's config)
        # both fired during the real run.
        kinds = {c[0] for c in fitted_pipeline["callback_calls"]}
        assert "progress" in kinds
        assert "log" in kinds


# ======================================================================
# load_localisations
# ======================================================================

class TestLoadLocalisations:
    def _write_h5(self, path: Path, frames, key="data"):
        df = pd.DataFrame({"frame": frames, "xc": np.arange(len(frames), dtype=float)})
        df.to_hdf(path, key=key, mode="w")
        return df

    def test_no_matching_files_returns_empty(self, tmp_path):
        pipe = AnalysisPipeline(camera="ximea")
        out = pipe.load_localisations(tmp_path)
        assert isinstance(out, pd.DataFrame)
        assert out.empty

    def test_files_present_but_unreadable_returns_empty(self, tmp_path):
        self._write_h5(tmp_path / "a.h5", [0, 1, 2], key="not_data")
        pipe = AnalysisPipeline(camera="ximea")
        out = pipe.load_localisations(tmp_path)
        assert out.empty

    def test_concatenates_readable_files_and_skips_unreadable(self, tmp_path):
        self._write_h5(tmp_path / "a.h5", [0, 1, 2])
        self._write_h5(tmp_path / "b.h5", [3, 4], key="not_data")
        self._write_h5(tmp_path / "c.h5", [5, 6])
        pipe = AnalysisPipeline(camera="ximea")
        out = pipe.load_localisations(tmp_path)
        assert len(out) == 5  # 3 from a.h5 + 2 from c.h5, b.h5 skipped
        assert sorted(out["frame"].tolist()) == [0, 1, 2, 5, 6]

    def test_start_frame_filters_rows(self, tmp_path):
        self._write_h5(tmp_path / "a.h5", [0, 1, 2, 3, 4])
        pipe = AnalysisPipeline(camera="ximea")
        out = pipe.load_localisations(tmp_path, start_frame=2)
        assert sorted(out["frame"].tolist()) == [2, 3, 4]

    def test_start_frame_zero_does_not_filter(self, tmp_path):
        self._write_h5(tmp_path / "a.h5", [0, 1, 2])
        pipe = AnalysisPipeline(camera="ximea")
        out = pipe.load_localisations(tmp_path, start_frame=0)
        assert len(out) == 3


# ======================================================================
# load_localisations_per_fov
# ======================================================================

class TestLoadLocalisationsPerFov:
    def _write_h5(self, path: Path, key="data"):
        df = pd.DataFrame({"frame": [0, 1], "xc": [1.0, 2.0]})
        df.to_hdf(path, key=key, mode="w")

    def test_tif_matched_via_tif_glob(self, tmp_path):
        self._write_h5(tmp_path / "a_MMStack_Default.h5")
        (tmp_path / "a_MMStack_Default.ome.tif").touch()
        pipe = AnalysisPipeline(camera="ximea")
        result = pipe.load_localisations_per_fov(tmp_path)
        assert len(result) == 1
        df, tif = result[0]
        assert len(df) == 2
        assert tif is not None and tif.endswith("a_MMStack_Default.ome.tif")

    def test_tif_matched_via_tiff_fallback_glob(self, tmp_path):
        self._write_h5(tmp_path / "b_MMStack_Default.h5")
        (tmp_path / "b_MMStack_Default.ome.tiff").touch()  # no plain .tif present
        pipe = AnalysisPipeline(camera="ximea")
        result = pipe.load_localisations_per_fov(tmp_path)
        assert len(result) == 1
        df, tif = result[0]
        assert tif is not None and tif.endswith(".tiff")

    def test_no_matching_tif_gives_none(self, tmp_path):
        self._write_h5(tmp_path / "c_MMStack_Default.h5")
        pipe = AnalysisPipeline(camera="ximea")
        result = pipe.load_localisations_per_fov(tmp_path)
        assert len(result) == 1
        df, tif = result[0]
        assert tif is None

    def test_unreadable_h5_is_skipped(self, tmp_path):
        self._write_h5(tmp_path / "d_MMStack_Default.h5", key="not_data")
        pipe = AnalysisPipeline(camera="ximea")
        result = pipe.load_localisations_per_fov(tmp_path)
        assert result == []

    def test_multiple_fovs_alphabetical_order(self, tmp_path):
        self._write_h5(tmp_path / "fov1.h5")
        self._write_h5(tmp_path / "fov2.h5")
        (tmp_path / "fov1.tif").touch()
        (tmp_path / "fov2.tif").touch()
        pipe = AnalysisPipeline(camera="ximea")
        result = pipe.load_localisations_per_fov(tmp_path)
        assert len(result) == 2
        assert result[0][1].endswith("fov1.tif")
        assert result[1][1].endswith("fov2.tif")


# ======================================================================
# filter_and_cluster
# ======================================================================

class TestFilterAndCluster:
    def test_unknown_clustering_method_raises(self):
        pipe = AnalysisPipeline(camera="ximea")
        cc = ClusteringConfig(clustering_method="not_a_real_method")
        with pytest.raises(ValueError, match="Unknown clustering method"):
            pipe.filter_and_cluster(pd.DataFrame(), clustering_config=cc)

    @pytest.mark.parametrize("method", ["HDBSCAN", "DBSCAN", "LINKED", "hdbscan"])
    def test_dispatches_to_configured_method(self, fitted_pipeline, method):
        pipe = fitted_pipeline["pipe"]
        locs = fitted_pipeline["locs"]
        cc = ClusteringConfig(clustering_method=method, min_cluster_size=2)
        sm_db, sf_db = pipe.filter_and_cluster(locs, clustering_config=cc)
        assert isinstance(sm_db, pd.DataFrame)
        assert isinstance(sf_db, pd.DataFrame)

    def test_default_criteria_and_config_used_when_none_given(self, fitted_pipeline):
        pipe = fitted_pipeline["pipe"]
        locs = fitted_pipeline["locs"]
        sm_db, sf_db = pipe.filter_and_cluster(locs)
        assert isinstance(sm_db, pd.DataFrame)
        assert isinstance(sf_db, pd.DataFrame)

    def test_explicit_criteria_forwarded(self, fitted_pipeline):
        pipe = fitted_pipeline["pipe"]
        locs = fitted_pipeline["locs"]
        filt = FilteringCriteria(min_photons=0)
        cc = ClusteringConfig(min_cluster_size=2)
        sm_db, sf_db = pipe.filter_and_cluster(locs, criteria=filt, clustering_config=cc)
        assert isinstance(sm_db, pd.DataFrame)


# ======================================================================
# undrift — pure passthrough to dcf.undrift
# ======================================================================

class TestUndrift:
    def test_forwards_args_and_returns_result(self):
        pipe = AnalysisPipeline(camera="ximea")
        fake_dcf = MagicMock()
        fake_dcf.undrift.return_value = ("corrected_locs", "drift_result")
        pipe._dcf = fake_dcf  # bypass real (heavier) construction

        locs = np.recarray((0,), dtype=[("x", float)])
        info = [{"foo": "bar"}]
        result = pipe.undrift(locs, info, method="fiducial", roi_r=5)

        assert result == ("corrected_locs", "drift_result")
        fake_dcf.undrift.assert_called_once_with(locs, info, method="fiducial", roi_r=5)

    def test_default_method_is_auto(self):
        pipe = AnalysisPipeline(camera="ximea")
        fake_dcf = MagicMock()
        fake_dcf.undrift.return_value = (None, None)
        pipe._dcf = fake_dcf

        pipe.undrift(np.recarray((0,), dtype=[("x", float)]), [])

        _, kwargs = fake_dcf.undrift.call_args
        assert kwargs["method"] == "auto"
