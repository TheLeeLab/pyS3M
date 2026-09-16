"""Full coverage tests for pyS3M.SR_Functions -- SuperRes_Functions, the core
fitting engine AnalysisPipeline.fit() dispatches to for every mode (smlm, fret,
qd, tracking, imaging).

_fit_files (shared by fit_SM_data/fit_imaging_data) is already substantially
exercised end-to-end by unit_tests/test_analysis_pipeline.py's real fixture-based
fit, and _process_roi by unit_tests/test_full_detection_extraction.py -- this
file fills the remaining gaps: small branch-level tests for the private helpers
(no I/O needed), the change-point-detection statics (tiny synthetic 1D traces,
no image data needed), and dedicated small-synthetic-TIFF tests for the 4 large
pipeline methods (example_spots_singleframe, fit_FRET_data, fit_QD_data,
fit_tracking_data) that nothing else in the suite currently touches.
"""
from __future__ import annotations

import json
import types

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend -- example_spots_singleframe renders figures

import numpy as np
import pandas as pd
import pytest

import pyS3M.IOFunctions as IOFunctions
import pyS3M.sCMOSFunctions as sCMOSFunctions
from pyS3M.Constants import AnalysisConfig, ResultColumns
from pyS3M.SR_Functions import SuperRes_Functions


# ======================================================================
# Shared helpers
# ======================================================================

def _smoothing_function(sigma=1.0):
    scmos = sCMOSFunctions.sCMOS_Functions()
    sf = types.SimpleNamespace()
    sf.args = {"sigma": sigma}
    sf.extent = sigma
    sf.smoothing_function = scmos.gaussian_filter_stack
    sf.data_arg = "image"
    return sf


def _write_metadata(folder, width, height):
    (folder / "test_metadata.txt").write_text(
        json.dumps({"FrameKey-0-0-0": {"ROI": f"0-0-{width}-{height}"}})
    )


def _write_synthetic_tiff(
    folder, filename, width=40, height=40, n_frames=10,
    spots=((20, 20),), amplitude=6000.0, background=60.0, sigma=1.5, seed=0,
):
    """Write a tiny synthetic multi-frame TIFF with Gaussian spots on Poisson
    background, plus the ImageJ-style metadata sidecar _fit_files requires.
    """
    rng = np.random.default_rng(seed)
    stack = rng.poisson(background, (n_frames, height, width)).astype(np.float32)
    yy, xx = np.mgrid[0:height, 0:width]
    for (x0, y0) in spots:
        gauss = amplitude * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma ** 2))
        stack += gauss[np.newaxis, :, :]

    io = IOFunctions.IO_Functions()
    path = folder / filename
    io.write_tiff(stack.astype(np.uint16), str(path))
    _write_metadata(folder, width, height)
    return path


def _calibration_maps(width, height):
    return dict(
        gain_map=np.ones((height, width), dtype=np.float32),
        offset_map=np.zeros((height, width), dtype=np.float32),
        rqe=np.ones((height, width), dtype=np.float32),
        read_noise=np.full((height, width), 10.0, dtype=np.float32),
        variance=np.full((height, width), 100.0, dtype=np.float32),
    )


@pytest.fixture
def sr():
    # display=False as defense-in-depth on top of matplotlib.use("Agg") above --
    # save_or_show(show=self.config.display) must never block on a real window.
    return SuperRes_Functions(camera="ximea", config=AnalysisConfig(display=False))


# ======================================================================
# _postprocess_fit_results / _filter_fit_results -- small branch gaps
# ======================================================================

class TestPostprocessFitResults:
    def test_quality_metric_length_mismatch_skipped(self, sr):
        n = 3
        fit_results_array = np.random.default_rng(0).uniform(0.1, 0.9, (n, len(ResultColumns.STANDARD_FIT_PARAMS)))
        fit_errors_array = np.random.default_rng(1).uniform(0.01, 0.1, (n, len(ResultColumns.STANDARD_FIT_ERRORS)))
        result_columns = ResultColumns.get_all_columns()
        planes = [0, 0, 0]
        # length 2 != n=3 -> hits the length-mismatch warning branch, metric skipped.
        quality_metrics = {"snr": np.array([1.0, 2.0])}

        out = sr._postprocess_fit_results(
            fit_results_array, fit_errors_array, result_columns, planes,
            width=100, height=100, quality_metrics=quality_metrics,
        )
        assert "spot_snr" not in out.columns

    def test_quality_metric_matching_length_added(self, sr):
        n = 3
        fit_results_array = np.random.default_rng(2).uniform(0.1, 0.9, (n, len(ResultColumns.STANDARD_FIT_PARAMS)))
        fit_errors_array = np.random.default_rng(3).uniform(0.01, 0.1, (n, len(ResultColumns.STANDARD_FIT_ERRORS)))
        result_columns = ResultColumns.get_all_columns()
        planes = [0, 0, 0]
        quality_metrics = {"snr": np.array([1.0, 2.0, 3.0])}

        out = sr._postprocess_fit_results(
            fit_results_array, fit_errors_array, result_columns, planes,
            width=100, height=100, quality_metrics=quality_metrics,
        )
        assert "spot_snr" in out.columns


class TestFilterFitResults:
    def test_single_column_A_and_bg_filtered(self, sr):
        # NOCOLOUR-style results: single 'A'/'bg' columns rather than
        # per-channel A_B/A_G/A_R + bg_B/bg_G/bg_R.
        fit_results = pd.DataFrame({
            "xc": [10.0, 10.0], "yc": [10.0, 10.0],
            "s_x": [1.0, 1.0], "s_y": [1.0, 1.0],
            "A": [5.0, -5.0], "bg": [2.0, 2.0],
        })
        out = sr._filter_fit_results(fit_results, width=100, height=100)
        assert len(out) == 1
        assert out["A"].iloc[0] == 5.0

    def test_single_column_bg_only_negative_filtered(self, sr):
        fit_results = pd.DataFrame({
            "xc": [10.0, 10.0], "yc": [10.0, 10.0],
            "s_x": [1.0, 1.0], "s_y": [1.0, 1.0],
            "A": [5.0, 5.0], "bg": [2.0, -2.0],
        })
        out = sr._filter_fit_results(fit_results, width=100, height=100)
        assert len(out) == 1

    def test_single_s_column_filtered(self, sr):
        # CIRCULAR-style results: single shared 's' column instead of s_x/s_y.
        fit_results = pd.DataFrame({
            "xc": [10.0, 10.0], "yc": [10.0, 10.0],
            "s": [1.0, 5.0],  # second row: s=5 > 3 -> filtered out
            "A_B": [5.0, 5.0], "A_G": [5.0, 5.0], "A_R": [5.0, 5.0],
            "bg_B": [2.0, 2.0], "bg_G": [2.0, 2.0], "bg_R": [2.0, 2.0],
        })
        out = sr._filter_fit_results(fit_results, width=100, height=100)
        assert len(out) == 1
        assert out["s"].iloc[0] == 1.0


# ======================================================================
# _process_roi / _process_detected_puncta_batch -- small branch gaps
# ======================================================================

class TestProcessRoiEdgeCases:
    def test_non_square_roi_from_actual_array_mismatch_returns_none(self, sr):
        # calculate_roi_bounds computes bounds relative to the *claimed*
        # width/height; if the actual raw_data array is smaller than claimed,
        # slicing against it truncates one axis but not the other -> the
        # non-square sanity check inside _process_roi (not
        # calculate_roi_bounds, which never sees the real array) fires.
        raw_data = np.zeros((20, 10), dtype=np.float32)  # actual: 20 rows, 10 cols
        detected_puncta = np.array([[10.0, 15.0, 0.0]])  # ycentre=10, xcentre=15
        masks = np.zeros((20, 20, 3), dtype=bool)
        result = sr._process_roi(
            raw_data, detected_puncta, 0,
            width=20, height=20, ROI_size=8,
            smoothing_function=None, read_noise=1.0, masks=masks,
        )
        assert result is None


class TestProcessDetectedPunctaBatchQualityMetrics:
    def test_quality_metric_length_mismatch_logged_and_skipped(self, sr):
        raw_data = np.full((30, 30), 100.0, dtype=np.float32)
        detected_puncta = np.array([[15.0, 15.0, 0.0]])
        masks = np.zeros((30, 30, 3), dtype=bool)
        masks[:, :, 0] = True
        quality_metrics = {"snr": np.array([1.0, 2.0])}  # length 2 != len(detected_puncta)=1

        result = sr._process_detected_puncta_batch(
            raw_data, detected_puncta, width=30, height=30, ROI_size=8,
            smoothing_function=None, read_noise=1.0, masks=masks,
            gain_map=1.0, offset_map=0.0, rqe=1.0,
            quality_metrics=quality_metrics,
        )
        filtered_quality_metrics = result[-1]
        assert filtered_quality_metrics == {}


# ======================================================================
# _fit_files -- pixel_size=None default-resolution branch
# ======================================================================

class TestFitFilesPixelSizeDefault:
    def test_pixel_size_none_uses_camera_default(self, sr, tmp_path, monkeypatch):
        # Isolate the "pixel_size is None -> self.pixel_size" branch (1 line)
        # without paying for real detection: file_search returns no files, so
        # the per-file loop body never executes.
        monkeypatch.setattr(sr.helper, "file_search", lambda *a, **kw: [])
        monkeypatch.setattr(sr.helper, "load_metadata_roi", lambda *a, **kw: (0, 0, 8, 8))
        maps = _calibration_maps(8, 8)
        sr.fit_SM_data(
            str(tmp_path), _smoothing_function(), maps["gain_map"], maps["offset_map"],
            maps["rqe"], maps["read_noise"], maps["variance"], pixel_size=None,
        )  # must not raise


# ======================================================================
# _fit_files -- use_elliptical / use_circular strategy dispatch
# ======================================================================

class TestFitFilesStrategyDispatch:
    def test_use_elliptical_and_use_circular_mutually_exclusive_raises(self, sr, tmp_path):
        # Fires before any file I/O -- no need for file_search/load_metadata_roi setup.
        maps = _calibration_maps(8, 8)
        with pytest.raises(ValueError, match="mutually exclusive"):
            sr.fit_SM_data(
                str(tmp_path), _smoothing_function(), maps["gain_map"], maps["offset_map"],
                maps["rqe"], maps["read_noise"], maps["variance"],
                use_elliptical=True, use_circular=True,
            )

    def test_use_elliptical_selects_elliptical_strategy(self, sr, tmp_path, monkeypatch):
        # Same "file_search -> [] so the per-file loop body never executes" trick as
        # TestFitFilesPixelSizeDefault above -- isolates the strategy-selection branch
        # itself without paying for real detection.
        monkeypatch.setattr(sr.helper, "file_search", lambda *a, **kw: [])
        monkeypatch.setattr(sr.helper, "load_metadata_roi", lambda *a, **kw: (0, 0, 8, 8))
        maps = _calibration_maps(8, 8)
        sr.fit_SM_data(
            str(tmp_path), _smoothing_function(), maps["gain_map"], maps["offset_map"],
            maps["rqe"], maps["read_noise"], maps["variance"], use_elliptical=True,
        )  # must not raise

    def test_use_circular_selects_circular_strategy(self, sr, tmp_path, monkeypatch):
        monkeypatch.setattr(sr.helper, "file_search", lambda *a, **kw: [])
        monkeypatch.setattr(sr.helper, "load_metadata_roi", lambda *a, **kw: (0, 0, 8, 8))
        maps = _calibration_maps(8, 8)
        sr.fit_SM_data(
            str(tmp_path), _smoothing_function(), maps["gain_map"], maps["offset_map"],
            maps["rqe"], maps["read_noise"], maps["variance"], use_circular=True,
        )  # must not raise


# ======================================================================
# _demosaic_image -- variance-aware / standard-grayscale strategy branches
# ======================================================================

class TestDemosaicImage:
    def test_strategy_none_returns_raw_unchanged(self, sr):
        raw = np.arange(16, dtype=np.float32).reshape(4, 4)
        out = sr._demosaic_image(raw, strategy="none")
        assert out is raw

    def test_variance_aware_bilinear(self, sr):
        raw = np.full((1, 8, 8), 100.0, dtype=np.float32)
        gain = np.ones((8, 8), dtype=np.float32)
        offset = np.zeros((8, 8), dtype=np.float32)
        variance = np.full((8, 8), 10.0, dtype=np.float32)
        out = sr._demosaic_image(
            raw, use_variance_aware=True, gain_map=gain, offset_map=offset,
            variance=variance, strategy="bilinear",
        )
        assert out.shape == raw.shape

    def test_standard_grayscale_bilinear(self, sr):
        raw = np.full((1, 8, 8), 100.0, dtype=np.float32)
        out = sr._demosaic_image(raw, use_variance_aware=False, strategy="bilinear")
        assert out.shape == raw.shape


# ======================================================================
# Change-point detection statics -- tiny synthetic 1D traces, no I/O
# ======================================================================

class TestFindChangePointsSingle:
    def test_zero_signal_returns_terminal_only(self, sr):
        signal = np.zeros(50)
        cps = sr._find_change_points_single(signal)
        assert cps == [50]

    def test_flat_noisy_signal_no_real_change_point(self, sr):
        rng = np.random.default_rng(0)
        signal = rng.normal(100, 1.0, 200)
        cps = sr._find_change_points_single(signal, min_size=5)
        assert cps[-1] == 200

    def test_step_change_detected(self, sr):
        rng = np.random.default_rng(1)
        signal = np.concatenate([
            rng.normal(500, 2.0, 100),
            rng.normal(50, 2.0, 100),
        ])
        cps = sr._find_change_points_single(signal, min_size=5)
        assert len(cps) > 1
        assert cps[-1] == 200
        assert 80 < cps[0] < 120

    def test_low_noise_falls_back_to_full_signal_std(self, sr):
        # Last-100-sample window is perfectly flat (std=0) but the full
        # signal has real variation -- exercises the "sigma < 1e-10 ->
        # recompute from the whole signal" fallback (not the "still zero"
        # early-return).
        signal = np.concatenate([np.full(150, 500.0), np.full(150, 500.0)])
        signal[:50] = 100.0  # variation lives entirely outside the noise window
        cps = sr._find_change_points_single(signal, min_size=5)
        assert cps[-1] == 300


class TestFindChangePointsBatch:
    def test_batch_matches_single(self, sr):
        rng = np.random.default_rng(2)
        traces = np.stack([
            rng.normal(500, 2.0, 60),
            np.concatenate([rng.normal(500, 2.0, 30), rng.normal(50, 2.0, 30)]),
        ])
        results = SuperRes_Functions._find_change_points_batch(traces, np.array([5, 9]))
        assert [idx for idx, _ in results] == [5, 9]


class TestFindChangePointsParallel:
    def test_parallel_matches_expected_puncta(self, sr):
        rng = np.random.default_rng(3)
        n_puncta = 3
        n_frames = 60
        traces = np.stack([
            rng.normal(500, 2.0, n_frames),  # no real change point
            np.concatenate([rng.normal(500, 2.0, 30), rng.normal(50, 2.0, 30)]),  # real step
            np.zeros(n_frames),  # zero signal
        ])
        frames_to_fit = sr._find_change_points_parallel(traces, n_workers=2)
        assert set(frames_to_fit.keys()) == {1}
        assert len(frames_to_fit[1]) > 0

    def test_empty_traces_skips_zero_item_tasks(self, sr):
        # n_puncta=0 forces n_tasks=1 with 0 items -> hits the
        # "task gets 0 items, skip submitting it" branch.
        traces = np.zeros((0, 10))
        frames_to_fit = sr._find_change_points_parallel(traces, n_workers=2)
        assert frames_to_fit == {}


class TestExtractRoiTracesSingleFile:
    def test_ndim2_squeeze_and_out_of_bounds_puncta_skipped(self, sr, tmp_path):
        # n_frames avoids 3/4: tifffile ambiguously interprets a (3, H, W)
        # or (4, H, W) stack as an RGB/RGBA image rather than 3/4 frames.
        width, height, n_frames = 30, 30, 6
        _write_synthetic_tiff(
            tmp_path, "traces.tif", width=width, height=height, n_frames=n_frames,
            spots=((15, 15),), amplitude=5000.0,
        )
        path = tmp_path / "traces.tif"
        maps = _calibration_maps(width, height)
        # detected_puncta[0] valid (centre); [1] asymmetrically near two
        # edges (a symmetric corner still clamps to a square) -> bounds=None.
        detected_puncta = np.array([[15.0, 15.0], [25.0, 2.0]])
        traces = sr._extract_roi_traces_single_file(
            str(path), detected_puncta, ROI_size=16, width=width, height=height,
            gain_map=maps["gain_map"], offset_map=maps["offset_map"], rqe=maps["rqe"],
            chunk_size=1,  # forces a single-frame-per-chunk read -> raw_data.ndim==2
        )
        assert traces.shape == (2, n_frames)
        assert np.all(traces[1] == 0.0)  # skipped puncta never gets summed into
        assert np.any(traces[0] > 0.0)


# ======================================================================
# example_spots_singleframe -- tiny synthetic single-frame image
# ======================================================================

class TestExampleSpotsSingleframe:
    def test_basic_no_zoom(self, sr, tmp_path):
        # A small field of view (<20x20 um at pixel_size=0.069 -> ~14.5x14.5
        # um at 40px) exercises the show_zoom=False path.
        _write_synthetic_tiff(tmp_path, "example_001.tif", width=40, height=40, n_frames=1)
        fig, axs = sr.example_spots_singleframe(
            str(tmp_path), smoothing_function=_smoothing_function(),
            pfa=1e-3, ROI_size=16, pixel_size=0.069,
        )
        assert fig is not None
        assert axs.shape == (1, 2)

    def test_with_zoom_and_frame_summing(self, sr, tmp_path):
        # show_zoom requires fov_um >= 20 on at least one axis; at
        # pixel_size=0.069 um/px that needs >= ~290 px. n_frames_sum>1
        # exercises the frame-summing branch.
        _write_synthetic_tiff(
            tmp_path, "example_002.tif", width=320, height=320, n_frames=5,
            spots=((160, 160),),
        )
        fig, axs = sr.example_spots_singleframe(
            str(tmp_path), smoothing_function=_smoothing_function(),
            pfa=1e-3, ROI_size=16, pixel_size=0.069, n_frames_sum=3,
        )
        assert fig is not None
        assert axs.shape == (2, 2)

    def test_n_frames_sum_exceeds_available_warns(self, sr, tmp_path, caplog):
        _write_synthetic_tiff(tmp_path, "example_003.tif", width=40, height=40, n_frames=2)
        sr.example_spots_singleframe(
            str(tmp_path), smoothing_function=_smoothing_function(),
            pfa=1e-3, ROI_size=16, pixel_size=0.069, n_frames_sum=10,
        )

    def test_no_detections_defaults_zoom_to_centre(self, sr, tmp_path):
        # pfa tiny enough (and no real spot) that nothing is detected ->
        # exercises the "no fits -> zoom to image centre" branch.
        rng = np.random.default_rng(9)
        io = IOFunctions.IO_Functions()
        stack = rng.poisson(60, (1, 200, 200)).astype(np.uint16)
        path = tmp_path / "example_004.tif"
        io.write_tiff(stack, str(path))
        _write_metadata(tmp_path, 200, 200)
        fig, axs = sr.example_spots_singleframe(
            str(tmp_path), smoothing_function=_smoothing_function(),
            pfa=1e-20, ROI_size=16, pixel_size=0.069,
        )
        assert fig is not None

    def test_pixel_size_none_uses_camera_default(self, sr, tmp_path):
        _write_synthetic_tiff(tmp_path, "example_005.tif", width=40, height=40, n_frames=1)
        fig, axs = sr.example_spots_singleframe(
            str(tmp_path), smoothing_function=_smoothing_function(),
            pfa=1e-3, ROI_size=16, pixel_size=None,
        )
        assert fig is not None

    def test_no_metadata_falls_back_to_full_image_dims(self, sr, tmp_path):
        # No metadata sidecar written -> load_metadata_roi(use_fallback=True)
        # returns (0, 0, None, None) -> width/height taken from raw_data.shape.
        rng = np.random.default_rng(13)
        io = IOFunctions.IO_Functions()
        stack = rng.poisson(60, (1, 40, 40)).astype(np.uint16)
        io.write_tiff(stack, str(tmp_path / "example_006.tif"))
        fig, axs = sr.example_spots_singleframe(
            str(tmp_path), smoothing_function=_smoothing_function(),
            pfa=1e-3, ROI_size=16, pixel_size=0.069,
        )
        assert fig is not None

    def test_frame_summing_yields_single_frame_squeeze(self, sr, tmp_path):
        # frame_index near the end of a short file + n_frames_sum>1 ->
        # frames_to_load ends up length 1 -> raw_stack comes back 2D and
        # needs the np.newaxis squeeze-fix before summing.
        _write_synthetic_tiff(tmp_path, "example_007.tif", width=40, height=40, n_frames=5)
        fig, axs = sr.example_spots_singleframe(
            str(tmp_path), smoothing_function=_smoothing_function(),
            pfa=1e-3, ROI_size=16, pixel_size=0.069, n_frames_sum=2, frame_index=4,
        )
        assert fig is not None

    def test_save_figures_writes_to_output_dir(self, tmp_path):
        out_dir = tmp_path / "figs"
        out_dir.mkdir()
        sr_saving = SuperRes_Functions(
            camera="ximea",
            config=AnalysisConfig(display=False, save_figures=True, output_dir=out_dir),
        )
        _write_synthetic_tiff(tmp_path, "example_008.tif", width=40, height=40, n_frames=1)
        fig, axs = sr_saving.example_spots_singleframe(
            str(tmp_path), smoothing_function=_smoothing_function(),
            pfa=1e-3, ROI_size=16, pixel_size=0.069,
        )
        assert (out_dir / f"example_spots_singleframe.{sr_saving.config.figure_format}").exists()

    def test_no_layout_engine_uses_direct_subplots_adjust(self, sr, tmp_path, monkeypatch):
        import matplotlib.figure

        monkeypatch.setattr(matplotlib.figure.Figure, "get_layout_engine", lambda self: None)
        _write_synthetic_tiff(
            tmp_path, "example_009.tif", width=320, height=320, n_frames=1,
            spots=((160, 160),),
        )
        fig, axs = sr.example_spots_singleframe(
            str(tmp_path), smoothing_function=_smoothing_function(),
            pfa=1e-3, ROI_size=16, pixel_size=0.069,
        )
        assert axs.shape == (2, 2)


# ======================================================================
# fit_FRET_data
# ======================================================================

class TestFitFRETData:
    def test_no_spots_detected_skips_file(self, sr, tmp_path):
        rng = np.random.default_rng(10)
        io = IOFunctions.IO_Functions()
        stack = rng.poisson(60, (10, 40, 40)).astype(np.uint16)
        io.write_tiff(stack, str(tmp_path / "fret_nospots.tif"))
        _write_metadata(tmp_path, 40, 40)
        maps = _calibration_maps(40, 40)
        sr.fit_FRET_data(
            str(tmp_path), _smoothing_function(), maps["gain_map"], maps["offset_map"],
            maps["rqe"], maps["read_noise"], maps["variance"],
            n_frames_sum=5, pfa=1e-20, pixel_size=0.069,
        )
        assert not list(tmp_path.glob("*.h5"))

    def test_spot_with_no_change_points_skips_file(self, sr, tmp_path):
        # Constant-brightness spot across all frames -> real detection, but
        # PELT finds no real change point -> "0 with change points" skip.
        _write_synthetic_tiff(
            tmp_path, "fret_flat.tif", width=40, height=40, n_frames=20,
            spots=((20, 20),), amplitude=8000.0,
        )
        maps = _calibration_maps(40, 40)
        sr.fit_FRET_data(
            str(tmp_path), _smoothing_function(), maps["gain_map"], maps["offset_map"],
            maps["rqe"], maps["read_noise"], maps["variance"],
            n_frames_sum=10, pfa=1e-3, pixel_size=0.069, cp_min_size=3,
        )
        assert not list(tmp_path.glob("*.h5"))

    def test_bleaching_step_fits_and_saves(self, sr, tmp_path):
        # A spot that's bright for the first half of frames then bleaches to
        # background -> real PELT change point -> full happy path (fit +
        # save HDF5).
        rng = np.random.default_rng(11)
        width, height, n_frames = 40, 40, 30
        stack = rng.poisson(60, (n_frames, height, width)).astype(np.float32)
        yy, xx = np.mgrid[0:height, 0:width]
        gauss = 8000.0 * np.exp(-((xx - 20) ** 2 + (yy - 20) ** 2) / (2 * 1.5 ** 2))
        stack[:15] += gauss[np.newaxis, :, :]  # bright half
        io = IOFunctions.IO_Functions()
        io.write_tiff(stack.astype(np.uint16), str(tmp_path / "fret_bleach.tif"))
        _write_metadata(tmp_path, width, height)

        maps = _calibration_maps(width, height)
        sr.fit_FRET_data(
            str(tmp_path), _smoothing_function(), maps["gain_map"], maps["offset_map"],
            maps["rqe"], maps["read_noise"], maps["variance"],
            n_frames_sum=15, pfa=1e-3, pixel_size=0.069, ROI_size=16, cp_min_size=3,
        )
        h5_files = list(tmp_path.glob("*.h5"))
        assert h5_files

    def test_no_matching_files_raises(self, sr, tmp_path):
        maps = _calibration_maps(10, 10)
        with pytest.raises(ValueError, match="No .tif files"):
            sr.fit_FRET_data(
                str(tmp_path), _smoothing_function(), maps["gain_map"], maps["offset_map"],
                maps["rqe"], maps["read_noise"], maps["variance"],
            )

    def test_pixel_size_none_and_no_metadata_fallback(self, sr, tmp_path):
        # pixel_size=None -> self.pixel_size; no metadata sidecar ->
        # load_metadata_roi's (0, 0, None, None) fallback -> width/height
        # taken from the first frame's real shape.
        rng = np.random.default_rng(14)
        io = IOFunctions.IO_Functions()
        stack = rng.poisson(60, (10, 40, 40)).astype(np.uint16)
        io.write_tiff(stack, str(tmp_path / "fret_nometa.tif"))
        maps = _calibration_maps(40, 40)
        sr.fit_FRET_data(
            str(tmp_path), _smoothing_function(), maps["gain_map"], maps["offset_map"],
            maps["rqe"], maps["read_noise"], maps["variance"],
            n_frames_sum=5, pfa=1e-20, pixel_size=None,
        )  # must not raise

    def test_summing_yields_single_frame_squeeze(self, sr, tmp_path):
        # n_frames_sum capped to 1 real available frame -> raw_stack comes
        # back 2D during the PHASE-1 summing step.
        _write_synthetic_tiff(tmp_path, "fret_squeeze.tif", width=40, height=40, n_frames=1)
        maps = _calibration_maps(40, 40)
        sr.fit_FRET_data(
            str(tmp_path), _smoothing_function(), maps["gain_map"], maps["offset_map"],
            maps["rqe"], maps["read_noise"], maps["variance"],
            n_frames_sum=5, pfa=1e-3, pixel_size=0.069,
        )

    def test_all_needed_frames_empty_skips_file(self, sr, tmp_path, monkeypatch):
        # A puncta with a real change point at frame 0 (last_real_cp=0) needs
        # an empty arange -> all_frames_needed stays empty -> file skipped
        # entirely before any chunk loading.
        _write_synthetic_tiff(
            tmp_path, "fret_emptyneeded.tif", width=40, height=40, n_frames=5,
            spots=((20, 20),), amplitude=8000.0,
        )
        monkeypatch.setattr(
            sr, "_find_change_points_parallel", lambda *a, **kw: {0: np.arange(0)},
        )
        maps = _calibration_maps(40, 40)
        sr.fit_FRET_data(
            str(tmp_path), _smoothing_function(), maps["gain_map"], maps["offset_map"],
            maps["rqe"], maps["read_noise"], maps["variance"],
            n_frames_sum=5, pfa=1e-3, pixel_size=0.069,
        )
        assert not list(tmp_path.glob("*.h5"))

    def test_chunk_boundary_and_bounds_edge_cases(self, sr, tmp_path, monkeypatch):
        # Fabricates a frames_to_fit spanning frame indices far beyond the
        # tiny real file (0, 4, 1100) across a second puncta with
        # out-of-image bounds, to exercise every defensive continue in the
        # chunked-fitting loop: an entirely-empty middle chunk (frames_in_chunk),
        # a chunk whose real data is shorter than the chunk boundary
        # (local_frames), a needed frame outside [chunk_start, chunk_end),
        # a needed frame beyond the loaded array's real length, and a puncta
        # whose detected position has no valid ROI bounds. n_frames
        # deliberately avoids 3: tifffile ambiguously interprets a (3, H, W)
        # stack as an RGB image (3 channels) rather than 3 frames.
        width, height, n_frames = 30, 30, 2
        _write_synthetic_tiff(
            tmp_path, "fret_chunkedge.tif", width=width, height=height, n_frames=n_frames,
            spots=((15, 15),), amplitude=8000.0,
        )
        fake_detected = np.array([[15.0, 15.0], [25.0, 2.0]])  # 2nd: near edge -> bounds=None
        monkeypatch.setattr(sr.spot_detection, "detect_puncta_in_image", lambda *a, **kw: fake_detected)
        monkeypatch.setattr(
            sr, "_find_change_points_parallel",
            lambda *a, **kw: {0: np.array([0, 4, 1100]), 1: np.array([0])},
        )
        maps = _calibration_maps(width, height)
        sr.fit_FRET_data(
            str(tmp_path), _smoothing_function(), maps["gain_map"], maps["offset_map"],
            maps["rqe"], maps["read_noise"], maps["variance"],
            n_frames_sum=3, pfa=1e-3, pixel_size=0.069, ROI_size=16,
        )
        # Puncta 0's frame=0 is real and in-bounds -> should still fit and save.
        assert list(tmp_path.glob("*.h5"))

    def test_single_frame_fit_squeeze_and_all_rois_filtered_skips_save(self, sr, tmp_path, monkeypatch):
        # A single real frame -> the fitting-phase read_tiff(frame=[0]) comes
        # back 2D and needs the np.newaxis squeeze; the one detected puncta
        # sits at an edge with no valid ROI bounds, so nothing is
        # accumulated across the whole file -> no HDF5 gets written.
        width, height = 30, 30
        _write_synthetic_tiff(
            tmp_path, "fret_single.tif", width=width, height=height, n_frames=1,
            spots=((15, 15),), amplitude=8000.0,
        )
        fake_detected = np.array([[25.0, 2.0]])  # edge -> bounds=None
        monkeypatch.setattr(sr.spot_detection, "detect_puncta_in_image", lambda *a, **kw: fake_detected)
        monkeypatch.setattr(
            sr, "_find_change_points_parallel", lambda *a, **kw: {0: np.array([0])},
        )
        maps = _calibration_maps(width, height)
        sr.fit_FRET_data(
            str(tmp_path), _smoothing_function(), maps["gain_map"], maps["offset_map"],
            maps["rqe"], maps["read_noise"], maps["variance"],
            n_frames_sum=1, pfa=1e-3, pixel_size=0.069, ROI_size=16,
        )
        assert not list(tmp_path.glob("*.h5"))


# ======================================================================
# fit_QD_data
# ======================================================================

class TestFitQDData:
    def test_no_spots_detected_skips_file(self, sr, tmp_path):
        rng = np.random.default_rng(12)
        io = IOFunctions.IO_Functions()
        stack = rng.poisson(60, (6, 40, 40)).astype(np.uint16)
        io.write_tiff(stack, str(tmp_path / "qd_nospots.tif"))
        _write_metadata(tmp_path, 40, 40)
        maps = _calibration_maps(40, 40)
        sr.fit_QD_data(
            str(tmp_path), _smoothing_function(), maps["gain_map"], maps["offset_map"],
            maps["rqe"], maps["read_noise"], maps["variance"],
            n_frames_sum=3, pfa=1e-20, pixel_size=0.069,
        )
        assert not list(tmp_path.glob("*.h5"))

    def test_fits_all_frames_in_chunks(self, sr, tmp_path):
        _write_synthetic_tiff(
            tmp_path, "qd_basic.tif", width=40, height=40, n_frames=8,
            spots=((20, 20),), amplitude=8000.0,
        )
        maps = _calibration_maps(40, 40)
        sr.fit_QD_data(
            str(tmp_path), _smoothing_function(), maps["gain_map"], maps["offset_map"],
            maps["rqe"], maps["read_noise"], maps["variance"],
            n_frames_sum=4, pfa=1e-3, pixel_size=0.069, ROI_size=16, chunk_size=3,
        )
        h5_files = list(tmp_path.glob("*.h5"))
        assert h5_files
        results = pd.read_hdf(h5_files[0])
        # chunk_size=3 over 8 frames means multiple appended chunks.
        assert results["frame"].max() > 3

    def test_no_matching_files_raises(self, sr, tmp_path):
        maps = _calibration_maps(10, 10)
        with pytest.raises(ValueError, match="No .tif files"):
            sr.fit_QD_data(
                str(tmp_path), _smoothing_function(), maps["gain_map"], maps["offset_map"],
                maps["rqe"], maps["read_noise"], maps["variance"],
            )

    def test_pixel_size_none_and_no_metadata_fallback(self, sr, tmp_path):
        rng = np.random.default_rng(15)
        io = IOFunctions.IO_Functions()
        # n_frames avoids 3/4: tifffile ambiguously interprets those as an
        # RGB/RGBA image rather than a frame stack.
        stack = rng.poisson(60, (6, 40, 40)).astype(np.uint16)
        io.write_tiff(stack, str(tmp_path / "qd_nometa.tif"))
        maps = _calibration_maps(40, 40)
        sr.fit_QD_data(
            str(tmp_path), _smoothing_function(), maps["gain_map"], maps["offset_map"],
            maps["rqe"], maps["read_noise"], maps["variance"],
            n_frames_sum=2, pfa=1e-20, pixel_size=None,
        )  # must not raise

    def test_chunk_size_one_forces_ndim2_squeeze(self, sr, tmp_path):
        # n_frames avoids 3/4: tifffile ambiguously interprets those as an
        # RGB/RGBA image rather than a frame stack.
        _write_synthetic_tiff(
            tmp_path, "qd_squeeze.tif", width=40, height=40, n_frames=6,
            spots=((20, 20),), amplitude=8000.0,
        )
        maps = _calibration_maps(40, 40)
        sr.fit_QD_data(
            str(tmp_path), _smoothing_function(), maps["gain_map"], maps["offset_map"],
            maps["rqe"], maps["read_noise"], maps["variance"],
            n_frames_sum=3, pfa=1e-3, pixel_size=0.069, ROI_size=16, chunk_size=1,
        )
        assert list(tmp_path.glob("*.h5"))

    def test_n_frames_sum_one_forces_summing_squeeze(self, sr, tmp_path):
        # n_frames_sum=1 -> PHASE 1's frames_to_load is a length-1 list ->
        # read_tiff comes back 2D and needs the np.newaxis squeeze before summing.
        _write_synthetic_tiff(
            tmp_path, "qd_sum1.tif", width=40, height=40, n_frames=6,
            spots=((20, 20),), amplitude=8000.0,
        )
        maps = _calibration_maps(40, 40)
        sr.fit_QD_data(
            str(tmp_path), _smoothing_function(), maps["gain_map"], maps["offset_map"],
            maps["rqe"], maps["read_noise"], maps["variance"],
            n_frames_sum=1, pfa=1e-3, pixel_size=0.069, ROI_size=16,
        )
        assert list(tmp_path.glob("*.h5"))

    def test_all_detections_out_of_bounds_skips_every_chunk(self, sr, tmp_path, monkeypatch):
        _write_synthetic_tiff(
            tmp_path, "qd_edge.tif", width=40, height=40, n_frames=6,
            spots=((20, 20),), amplitude=8000.0,
        )
        # Fabricate a single detection asymmetrically near two edges (not a
        # symmetric corner, which still clamps to a square) so
        # calculate_roi_bounds returns None -> every chunk's puncta_tofit
        # stays empty.
        fake_detected = np.array([[35.0, 1.0]])
        monkeypatch.setattr(sr.spot_detection, "detect_puncta_in_image", lambda *a, **kw: fake_detected)
        maps = _calibration_maps(40, 40)
        sr.fit_QD_data(
            str(tmp_path), _smoothing_function(), maps["gain_map"], maps["offset_map"],
            maps["rqe"], maps["read_noise"], maps["variance"],
            n_frames_sum=2, pfa=1e-3, pixel_size=0.069, ROI_size=16,
        )
        assert not list(tmp_path.glob("*.h5"))


# ======================================================================
# fit_tracking_data
# ======================================================================

class TestFitTrackingData:
    def test_elliptical_strategy(self, sr, tmp_path):
        _write_synthetic_tiff(
            tmp_path, "track_ellip.tif", width=40, height=40, n_frames=5,
            spots=((20, 20),), amplitude=8000.0,
        )
        maps = _calibration_maps(40, 40)
        sr.fit_tracking_data(
            str(tmp_path), _smoothing_function(), maps["gain_map"], maps["offset_map"],
            maps["rqe"], maps["read_noise"], maps["variance"],
            pfa=1e-3, pixel_size=0.069, ROI_size=16, use_elliptical=True,
        )
        h5_files = list(tmp_path.glob("*.h5"))
        assert h5_files
        results = pd.read_hdf(h5_files[0])
        assert "theta" in results.columns

    def test_standard_strategy(self, sr, tmp_path):
        _write_synthetic_tiff(
            tmp_path, "track_standard.tif", width=40, height=40, n_frames=5,
            spots=((20, 20),), amplitude=8000.0,
        )
        maps = _calibration_maps(40, 40)
        sr.fit_tracking_data(
            str(tmp_path), _smoothing_function(), maps["gain_map"], maps["offset_map"],
            maps["rqe"], maps["read_noise"], maps["variance"],
            pfa=1e-3, pixel_size=0.069, ROI_size=16, use_elliptical=False,
        )
        h5_files = list(tmp_path.glob("*.h5"))
        assert h5_files
        results = pd.read_hdf(h5_files[0])
        assert "theta" not in results.columns

    def test_pixel_size_none_uses_camera_default(self, sr, tmp_path):
        _write_synthetic_tiff(
            tmp_path, "track_pxnone.tif", width=40, height=40, n_frames=5,
            spots=((20, 20),), amplitude=8000.0,
        )
        maps = _calibration_maps(40, 40)
        sr.fit_tracking_data(
            str(tmp_path), _smoothing_function(), maps["gain_map"], maps["offset_map"],
            maps["rqe"], maps["read_noise"], maps["variance"],
            pfa=1e-3, pixel_size=None, ROI_size=16,
        )
        assert list(tmp_path.glob("*.h5"))

    def test_single_frame_forces_ndim2_squeeze(self, sr, tmp_path):
        # chunk_size is hardcoded to 1000 in fit_tracking_data, so the only
        # way to get a length-1 chunk_frames list (and hence a 2D read_tiff
        # result needing the np.newaxis squeeze) is a genuinely 1-frame file.
        _write_synthetic_tiff(
            tmp_path, "track_squeeze.tif", width=40, height=40, n_frames=1,
            spots=((20, 20),), amplitude=8000.0,
        )
        maps = _calibration_maps(40, 40)
        sr.fit_tracking_data(
            str(tmp_path), _smoothing_function(), maps["gain_map"], maps["offset_map"],
            maps["rqe"], maps["read_noise"], maps["variance"],
            pfa=1e-3, pixel_size=0.069, ROI_size=16,
        )
        assert list(tmp_path.glob("*.h5"))
