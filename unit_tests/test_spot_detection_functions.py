"""Full coverage tests for pyS3M.SpotDetectionFunctions -- CA-CFAR-based spot
detection (Hekrdla et al. 2025) with matched filtering, quality-metric
extraction, and parallel multi-frame dispatch.

Complements the existing `test_real_spot_detection.py`/`test_full_detection_
extraction.py`/`test_quality_metrics_real_data.py` (which already exercise the
end-to-end detect_puncta_in_image happy path on larger realistic images) with
small hand-built images for the lower-level CFAR/kernel machinery and the
gaps those integration-style tests don't reach (return_quality branches,
dependency-injection defaults, parallel dispatch, the standalone
multiprocessing worker's exception fallback).

A larger pixel_size (0.15 um, vs. the 0.069 um camera default) is used
throughout to keep the matched-filter/CFAR kernel small enough for compact
test images without changing any detection logic.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyS3M.SpotDetectionFunctions import (
    SpotDetection_Functions,
    KernelCache,
    _detect_puncta_in_images_standalone,
)

PIXEL_SIZE = 0.15


@pytest.fixture
def sd():
    return SpotDetection_Functions(pixel_size=PIXEL_SIZE)


def _synthetic_image(size=32, spots=((10, 10), (22, 20)), amplitude=800.0,
                      background=15.0, sigma=1.3, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    image = np.full((size, size), background, dtype=np.float64)
    for (sx, sy) in spots:
        image += amplitude * np.exp(-((x - sx) ** 2 + (y - sy) ** 2) / (2 * sigma**2))
    image += rng.normal(0, 2.0, (size, size))
    return image.astype(np.float32)


# ======================================================================
# __init__ / dependency injection
# ======================================================================

class TestInit:
    def test_default_construction(self):
        sd_default = SpotDetection_Functions()
        assert sd_default.pixel_size > 0
        assert isinstance(sd_default.kernel_cache, KernelCache)

    def test_zwo_camera(self):
        sd_zwo = SpotDetection_Functions(camera="zwo")
        assert sd_zwo.pixel_size > 0

    def test_explicit_pixel_size_overrides_camera(self):
        sd_custom = SpotDetection_Functions(pixel_size=0.5)
        assert sd_custom.pixel_size == pytest.approx(0.5)

    def test_injected_dependencies_are_used(self):
        import types
        fake_psf = types.SimpleNamespace(sigma_PSF=lambda wl, na: 0.1)
        fake_scmos = types.SimpleNamespace()
        fake_helper = types.SimpleNamespace()
        sd_injected = SpotDetection_Functions(
            psf_functions=fake_psf, scmos_functions=fake_scmos, helper_functions=fake_helper,
        )
        assert sd_injected.psf is fake_psf
        assert sd_injected.scmos is fake_scmos
        assert sd_injected.helper is fake_helper


# ======================================================================
# KernelCache
# ======================================================================

class TestKernelCache:
    def test_caches_and_reuses(self):
        cache = KernelCache(max_size=2)
        calls = {"n": 0}

        def compute(x):
            calls["n"] += 1
            return x * 2

        assert cache.get_kernel("a", compute, 3) == 6
        assert cache.get_kernel("a", compute, 3) == 6
        assert calls["n"] == 1

    def test_evicts_oldest_when_full(self):
        cache = KernelCache(max_size=2)
        cache.get_kernel("a", lambda: 1)
        cache.get_kernel("b", lambda: 2)
        cache.get_kernel("c", lambda: 3)
        assert len(cache.cache) == 2


# ======================================================================
# gauss2d / _gauss2d_core
# ======================================================================

class TestGauss2d:
    def test_peak_at_centre(self, sd):
        peak = sd.gauss2d(5.0, 5.0, 5.0, 5.0, 1.3, 100.0)
        off_centre = sd.gauss2d(8.0, 5.0, 5.0, 5.0, 1.3, 100.0)
        assert peak > off_centre

    def test_jit_core_py_func(self, sd):
        jit_val = sd._gauss2d_core(5.0, 5.0, 5.0, 5.0, 1.3, 100.0)
        py_val = sd._gauss2d_core.py_func(5.0, 5.0, 5.0, 5.0, 1.3, 100.0)
        assert jit_val == pytest.approx(py_val)


# ======================================================================
# get_single_spot / vectorized gaussian / vectorized generic
# ======================================================================

class TestGetSingleSpot:
    def test_default_gaussian_path(self, sd):
        out = sd.get_single_spot(x0=8, y0=8, psf_fun=None, sigma=1.3, a=1.0, size=16)
        assert out.shape == (16, 16)
        assert out.sum() > 0

    def test_explicit_gauss2d_path(self, sd):
        out = sd.get_single_spot(x0=8, y0=8, psf_fun=sd.gauss2d, sigma=1.3, a=1.0, size=16)
        assert out.shape == (16, 16)

    def test_generic_psf_fallback(self, sd):
        def custom_psf(x, y, x0, y0, sigma, a):
            return a * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))

        out = sd.get_single_spot(x0=8, y0=8, psf_fun=custom_psf, sigma=1.3, a=1.0, size=16)
        assert out.shape == (16, 16)
        assert out.sum() > 0


class TestComputeMf:
    def test_via_get_mf_uses_cache(self, sd):
        w1 = sd.get_mf(sd.gauss2d, 1.3, 4)
        w2 = sd.get_mf(sd.gauss2d, 1.3, 4)
        np.testing.assert_array_equal(w1, w2)
        assert len(sd.kernel_cache.cache) == 1


# ======================================================================
# filter_image / get_square_annulus / isf_threshold
# ======================================================================

class TestFilterImage:
    def test_convolves(self, sd):
        image = np.ones((10, 10), dtype=np.float32)
        w = np.ones((3, 3), dtype=np.float32) / 9.0
        out = sd.filter_image(image, w)
        assert out.shape == (10, 10)
        np.testing.assert_allclose(out, 1.0, atol=1e-5)


class TestGetSquareAnnulus:
    def test_shape_and_hole(self, sd):
        kernel = sd.get_square_annulus(guard_interval=2, reference_interval=2)
        assert kernel.shape == (9, 9)
        centre = kernel.shape[0] // 2
        assert kernel[centre, centre] == 0


class TestIsfThreshold:
    def test_matches_scipy(self, sd):
        from scipy.stats import norm
        val = sd.isf_threshold(0.001, mu=10.0, sigma=2.0)
        assert val == pytest.approx(norm.isf(0.001, loc=10.0, scale=2.0))


# ======================================================================
# CA-CFAR family
# ======================================================================

class TestCacfarFamily:
    def test_background_mean_and_std_estimates(self, sd):
        image = _synthetic_image()
        kernel = sd.get_square_annulus(2, 2)
        mean_est = sd.cacfar_background_mean_estimate(image, kernel)
        std_est = sd.cacfar_background_std_estimate(image, mean_est, kernel)
        assert mean_est.shape == image.shape
        assert np.all(std_est >= 0)

    def test_segmentation_and_cacfar(self, sd):
        image = _synthetic_image()
        kernel = sd.get_square_annulus(2, 2)
        seg = sd.cacfar_segmentation(image, pfa=1e-3, kernel=kernel)
        assert seg.dtype == bool
        mask = sd.cacfar(image, pfa=1e-3, local_max_range=3, kernel=kernel)
        assert mask.shape == image.shape
        assert mask.sum() <= seg.sum()


# ======================================================================
# neigborhood / is_local_max / get_local_max_points / remove_nonlocal_maxima
# ======================================================================

class TestNeighborhoodAndLocalMax:
    def test_neigborhood_clips_to_bounds(self, sd):
        T = np.arange(25).reshape(5, 5)
        sub = sd.neigborhood(T, np.array([0, 0]), r=2)
        assert sub.shape == (3, 3)

    def test_is_local_max_true_at_peak(self, sd):
        T = np.zeros((9, 9))
        T[4, 4] = 100.0
        assert sd.is_local_max(T, np.array([4, 4]), r=3)

    def test_is_local_max_false_off_peak(self, sd):
        T = np.zeros((9, 9))
        T[4, 4] = 100.0
        assert not sd.is_local_max(T, np.array([2, 2]), r=3)

    def test_get_local_max_points_filters(self, sd):
        T = np.zeros((9, 9))
        T[4, 4] = 100.0
        points = np.array([[4, 4], [2, 2]])
        out = sd.get_local_max_points(T, points, local_max_range=3)
        assert list(out[0]) == [4, 4]

    def test_remove_nonlocal_maxima(self, sd):
        T = _synthetic_image(size=24, spots=((12, 12),))
        seg = np.zeros((24, 24), dtype=bool)
        seg[10:15, 10:15] = True
        mask = sd.remove_nonlocal_maxima(seg, T, local_max_range=3)
        assert mask.shape == (24, 24)


# ======================================================================
# mask2points / points2mask
# ======================================================================

class TestMaskPointsRoundtrip:
    def test_roundtrip(self, sd):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[3, 4] = 1
        mask[7, 8] = 1
        points = sd.mask2points(mask)
        assert sorted(map(tuple, points)) == [(3, 4), (7, 8)]
        rebuilt = sd.points2mask(points, size=(10, 10))
        np.testing.assert_array_equal(rebuilt, mask)

    def test_points2mask_int_size(self, sd):
        points = np.array([[2, 2]])
        out = sd.points2mask(points, size=5)
        assert out.shape == (5, 5)

    def test_points2mask_empty_points(self, sd):
        out = sd.points2mask(np.array([]), size=(6, 6))
        np.testing.assert_allclose(out, 0.0)

    def test_points2mask_out_of_bounds_points_dropped(self, sd):
        points = np.array([[2, 2], [-1, 0], [100, 100]])
        out = sd.points2mask(points, size=(6, 6))
        assert out.sum() == 1
        assert out[2, 2] == 1


# ======================================================================
# get_detection_points
# ======================================================================

class TestGetDetectionPoints:
    def test_with_kernel(self, sd):
        image = _synthetic_image()
        kernel = sd.get_square_annulus(2, 2)
        points = sd.get_detection_points(image, sd.cacfar, pfa=1e-3, local_max_range=3, kernel=kernel)
        assert points.ndim == 2

    def test_without_kernel(self, sd):
        def detector_no_kernel(T, pfa, local_max_range):
            return T > (np.mean(T) + 3 * np.std(T))

        image = _synthetic_image()
        points = sd.get_detection_points(image, detector_no_kernel, pfa=1e-3, local_max_range=3)
        assert points.ndim == 2


# ======================================================================
# intensity_pixel_indices / spot_quality_metrics
# ======================================================================

class TestSpotQualityMetrics:
    def test_keys_and_lengths(self, sd):
        image = _synthetic_image(size=32, spots=((16, 16),))
        detected = np.array([[16, 16], [2, 2]])
        quality = sd.spot_quality_metrics(
            image, detected, guard_interval=2, reference_interval=2,
        )
        assert set(quality.keys()) == {
            "background", "background_std", "mean_inner_intensity", "snr",
        }
        assert len(quality["background"]) == 2
        # the spot is brighter than its annulus; the empty corner is not
        assert quality["snr"][0] > quality["snr"][1]

    def test_detection_keeps_every_cfar_maximum(self, sd):
        # no post-CFAR threshold: with and without metrics give the same spots
        image = _synthetic_image(size=48, spots=((16, 16), (32, 30)))
        pts = sd.detect_puncta_in_image(image, pfa=1e-3, pixel_size=0.069)
        pts_q, q = sd.detect_puncta_in_image(image, pfa=1e-3, pixel_size=0.069, return_quality=True)
        np.testing.assert_array_equal(pts, pts_q)
        assert set(q) == {"matched_filter_response", "background", "background_std",
                          "mean_inner_intensity", "snr"}
        assert all(len(v) == len(pts) for v in q.values())


# ======================================================================
# detect_puncta_in_image
# ======================================================================

class TestDetectPunctaInImage:
    def test_basic_detection_no_quality(self, sd):
        image = _synthetic_image(size=40, spots=((15, 15), (28, 25)), amplitude=1500.0)
        points = sd.detect_puncta_in_image(image, wavelength=0.6, NA=1.49)
        assert points.ndim == 2
        assert points.shape[1] == 2

    def test_detection_with_quality(self, sd):
        image = _synthetic_image(size=40, spots=((15, 15), (28, 25)), amplitude=1500.0)
        points, quality = sd.detect_puncta_in_image(
            image, wavelength=0.6, NA=1.49, return_quality=True,
        )
        assert points.ndim == 2
        if len(points) > 0:
            assert "snr" in quality
            assert len(quality["snr"]) == len(points)

    def test_with_variance_map(self, sd):
        image = _synthetic_image(size=40, spots=((15, 15),), amplitude=1500.0)
        variance = np.full(image.shape, 4.0, dtype=np.float32)
        points = sd.detect_puncta_in_image(image, variance=variance, wavelength=0.6, NA=1.49)
        assert points.ndim == 2


# ======================================================================
# detect_puncta_in_images
# ======================================================================

class TestDetectPunctaInImages:
    def test_stack_no_quality(self, sd):
        stack = np.stack([
            _synthetic_image(size=32, spots=((16, 16),), amplitude=1500.0, seed=i)
            for i in range(3)
        ])
        detected = sd.detect_puncta_in_images(stack, start_frame=0, wavelength=0.6, NA=1.49)
        assert len(detected) == 3
        for frame_points in detected:
            assert frame_points.shape[1] == 3  # x, y, frame

    def test_stack_with_quality(self, sd):
        stack = np.stack([
            _synthetic_image(size=32, spots=((16, 16),), amplitude=1500.0, seed=i)
            for i in range(3)
        ])
        detected, quality = sd.detect_puncta_in_images(
            stack, start_frame=5, wavelength=0.6, NA=1.49, return_quality=True,
        )
        assert len(detected) == 3
        assert isinstance(quality, dict)


# ======================================================================
# spots_from_futures / spots_and_quality_from_futures
# ======================================================================

class _FakeFuture:
    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result


class TestSpotsFromFutures:
    def test_spots_from_futures(self, sd):
        fs = [
            _FakeFuture([np.array([[1, 2, 0]]), np.array([[3, 4, 1]])]),
            _FakeFuture([np.array([[5, 6, 2]])]),
        ]
        out = sd.spots_from_futures(fs)
        assert out.shape == (3, 3)

    def test_spots_and_quality_from_futures_tuple_results(self, sd):
        q1 = {"snr": np.array([5.0])}
        q2 = {"snr": np.array([6.0])}
        fs = [
            _FakeFuture(([np.array([[1, 2, 0]])], q1)),
            _FakeFuture(([np.array([[3, 4, 1]])], q2)),
        ]
        points, quality = sd.spots_and_quality_from_futures(fs)
        assert points.shape == (2, 3)
        assert quality["snr"].shape == (2,)

    def test_spots_and_quality_from_futures_backward_compat_plain_array(self, sd):
        fs = [_FakeFuture([np.array([[1, 2, 0]])])]
        points, quality = sd.spots_and_quality_from_futures(fs)
        assert points.shape == (1, 3)
        assert quality == {}


# ======================================================================
# detect_puncta_in_stack_parallel
# ======================================================================

class TestDetectPunctaInStackParallel:
    def test_parallel_no_quality(self, sd):
        stack = np.stack([
            _synthetic_image(size=32, spots=((16, 16),), amplitude=1500.0, seed=i)
            for i in range(2)
        ])
        out = sd.detect_puncta_in_stack_parallel(stack, wavelength=0.6, NA=1.49)
        assert out.ndim == 2
        assert out.shape[1] == 3

    def test_parallel_with_quality(self, sd):
        stack = np.stack([
            _synthetic_image(size=32, spots=((16, 16),), amplitude=1500.0, seed=i)
            for i in range(2)
        ])
        out, quality = sd.detect_puncta_in_stack_parallel(
            stack, wavelength=0.6, NA=1.49, return_quality=True,
        )
        assert out.ndim == 2
        assert isinstance(quality, dict)


# ======================================================================
# _detect_puncta_in_images_standalone
# ======================================================================

class TestDetectPunctaInImagesStandalone:
    def test_success(self):
        stack = np.stack([
            _synthetic_image(size=32, spots=((16, 16),), amplitude=1500.0, seed=i)
            for i in range(2)
        ])
        result = _detect_puncta_in_images_standalone(
            stack, start_frame=0, wavelength=0.6, NA=1.49, pixel_size=PIXEL_SIZE,
        )
        assert len(result) == 2

    def test_exception_returns_empty_no_quality(self, monkeypatch):
        import pyS3M.SpotDetectionFunctions as SpotDetectionFunctions

        def _raise(self, *a, **kw):
            raise RuntimeError("forced failure")

        monkeypatch.setattr(
            SpotDetectionFunctions.SpotDetection_Functions, "detect_puncta_in_images", _raise,
        )
        result = _detect_puncta_in_images_standalone(
            np.zeros((1, 8, 8), dtype=np.float32), start_frame=0,
        )
        assert result.shape == (0, 3)

    def test_exception_returns_empty_with_quality(self, monkeypatch):
        import pyS3M.SpotDetectionFunctions as SpotDetectionFunctions

        def _raise(self, *a, **kw):
            raise RuntimeError("forced failure")

        monkeypatch.setattr(
            SpotDetectionFunctions.SpotDetection_Functions, "detect_puncta_in_images", _raise,
        )
        points, quality = _detect_puncta_in_images_standalone(
            np.zeros((1, 8, 8), dtype=np.float32), start_frame=0, return_quality=True,
        )
        assert points.shape == (0, 3)
        assert quality == {}

    def test_sys_path_insertion_branch(self, monkeypatch):
        import sys
        import pyS3M.SpotDetectionFunctions as SpotDetectionFunctions
        from pathlib import Path

        _dir = str(Path(SpotDetectionFunctions.__file__).parent)
        pruned_path = [p for p in sys.path if p != _dir]
        monkeypatch.setattr(sys, "path", pruned_path)
        assert _dir not in sys.path

        stack = np.stack([_synthetic_image(size=32, spots=((16, 16),), amplitude=1500.0)])
        result = _detect_puncta_in_images_standalone(
            stack, start_frame=0, wavelength=0.6, NA=1.49, pixel_size=PIXEL_SIZE,
        )
        assert len(result) == 1
        assert _dir in sys.path
