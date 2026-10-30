"""Full coverage tests for pyS3M.gaussoptfuncs -- the numba-JIT Gaussian PSF
model/chi/initial-guess functions ImageAnalysisFunctions.py's leastsq fitting
dispatches to (see ImageAnalysisFunctions.py's `gaussoptfuncs.*` call sites).

All functions here are `@numba.jit(nopython=True)` -- once JIT-compiled they
run as machine code that bypasses Python's trace hooks, so coverage.py cannot
see line hits inside them no matter how many times they're called normally.
Each function exposes the original, uncompiled Python implementation via
`.py_func`; tests call both the normal (JIT) path, to verify real runtime
behaviour, and the `.py_func` path, purely so coverage.py can see the body
executed -- same pattern as unit_tests/test_localise.py and
unit_tests/test_render.py.

Deliberately tiny arrays throughout (size=4-6) -- these are branch-coverage
unit tests on pure numeric functions, not accuracy benchmarks.
"""
from __future__ import annotations

import numpy as np
import pytest

import pyS3M.gaussoptfuncs as gf


SIZE = 6


def _x(size=SIZE):
    return np.arange(size, dtype=np.float64)


def _bayer_masks(size=SIZE, n_ch=3):
    """3 (or n_ch) disjoint boolean channel masks covering the whole grid."""
    idx = np.arange(size * size).reshape(size, size) % n_ch
    masks = np.zeros((size, size, n_ch), dtype=np.bool_)
    for c in range(n_ch):
        masks[:, :, c] = idx == c
    return masks


def _positive_blob(size=SIZE, x0=3.0, y0=3.0, sigma=1.2, amplitude=100.0, background=5.0):
    """A small positive Gaussian blob -- used as smoothed/raw data everywhere
    a genuinely positive-sum, non-degenerate image is needed (centre-of-mass,
    sigma, and theta estimators all divide by the total intensity)."""
    x = _x(size)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    return background + amplitude * np.exp(
        -0.5 * (((xx - x0) / sigma) ** 2 + ((yy - y0) / sigma) ** 2)
    )


# ======================================================================
# gaussian_unscaled_model
# ======================================================================

class TestGaussianUnscaledModel:
    def test_peak_at_centre(self):
        x = _x()
        arr = np.zeros((SIZE, SIZE))
        out = gf.gaussian_unscaled_model(arr, x, SIZE, 3.0, 3.0, 1.0, 1.0)
        assert out[3, 3] == pytest.approx(out.max())
        assert np.all(out >= 0)

    def test_py_func_matches_jit(self):
        x = _x()
        arr = np.zeros((SIZE, SIZE))
        jit_out = gf.gaussian_unscaled_model(arr.copy(), x, SIZE, 2.0, 4.0, 1.0, 1.5)
        py_out = gf.gaussian_unscaled_model.py_func(arr.copy(), x, SIZE, 2.0, 4.0, 1.0, 1.5)
        np.testing.assert_allclose(jit_out, py_out)


# ======================================================================
# Model functions (WLS_*_model_nobounds)
# ======================================================================

class TestWLSNocolourModel:
    def test_basic_and_py_func(self):
        x = _x()
        data = np.zeros((SIZE, SIZE))  # unused by the model itself
        params = np.array([3.0, 3.0, 1.0, 1.0, 1.0, 2.0])  # [x0,y0,sx,sy,sqrt(bg),sqrt(A)]
        gauss_2d = np.zeros((SIZE, SIZE))
        jit_out = gf.WLS_nocolour_model_nobounds(params, data, x, gauss_2d.copy())
        py_out = gf.WLS_nocolour_model_nobounds.py_func(params, data, x, gauss_2d.copy())
        np.testing.assert_allclose(jit_out, py_out)


class TestWLSModelNobounds:
    def test_basic_and_py_func(self):
        x = _x()
        masks = _bayer_masks()
        # [x0,y0,sy,sx, bg0,bg1,bg2, A0,A1,A2]
        params = np.array([3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        gauss_2d = np.zeros((SIZE, SIZE))
        jit_out = gf.WLS_model_nobounds(params, masks, x, gauss_2d.copy())
        py_out = gf.WLS_model_nobounds.py_func(params, masks, x, gauss_2d.copy())
        np.testing.assert_allclose(jit_out, py_out)


class TestWLSModelElliptical:
    def test_basic_and_py_func(self):
        x = _x()
        masks = _bayer_masks()
        # [x0,y0,sx,sy,theta, bg0,bg1,bg2, A0,A1,A2]
        params = np.array([3.0, 3.0, 1.2, 0.8, 0.4, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        gauss_2d = np.zeros((SIZE, SIZE))
        jit_out = gf.WLS_model_elliptical_nobounds(params, masks, x, gauss_2d.copy())
        py_out = gf.WLS_model_elliptical_nobounds.py_func(params, masks, x, gauss_2d.copy())
        np.testing.assert_allclose(jit_out, py_out)


class TestGaussianUnscaledModelElliptical:
    def test_basic_and_py_func(self):
        x = _x()
        arr = np.zeros((SIZE, SIZE))
        jit_out = gf.gaussian_unscaled_model_elliptical(arr.copy(), x, SIZE, 3.0, 3.0, 1.2, 0.8, 0.4)
        py_out = gf.gaussian_unscaled_model_elliptical.py_func(arr.copy(), x, SIZE, 3.0, 3.0, 1.2, 0.8, 0.4)
        np.testing.assert_allclose(jit_out, py_out)
        assert jit_out.max() > 0


# ======================================================================
# Chi functions (WLS_chi_*)
# ======================================================================

class TestWLSChiNobounds:
    def test_basic_and_py_func(self):
        x = _x()
        masks = _bayer_masks()
        params = np.array([3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        data = gf.WLS_model_nobounds(params, masks, x, np.zeros((SIZE, SIZE)))
        weights = np.ones((SIZE, SIZE))
        ravelsize = SIZE * SIZE
        jit_out = gf.WLS_chi_nobounds(params, data, masks, weights, SIZE, ravelsize)
        py_out = gf.WLS_chi_nobounds.py_func(params, data, masks, weights, SIZE, ravelsize)
        np.testing.assert_allclose(jit_out, py_out)
        # Chi against its own generating model should be ~0.
        np.testing.assert_allclose(jit_out, 0.0, atol=1e-6)


class TestWLSChiNocolourNobounds:
    def test_basic_and_py_func(self):
        x = _x()
        params = np.array([3.0, 3.0, 1.0, 1.0, 1.0, 2.0])
        data = np.zeros((SIZE, SIZE))
        model = gf.WLS_nocolour_model_nobounds(params, data, x, np.zeros((SIZE, SIZE)))
        weights = np.ones((SIZE, SIZE))
        ravelsize = SIZE * SIZE
        jit_out = gf.WLS_chi_nocolour_nobounds(params, model, weights, SIZE, ravelsize)
        py_out = gf.WLS_chi_nocolour_nobounds.py_func(params, model, weights, SIZE, ravelsize)
        np.testing.assert_allclose(jit_out, py_out)
        np.testing.assert_allclose(jit_out, 0.0, atol=1e-6)


class TestWLSChiElliptical:
    def test_basic_and_py_func(self):
        x = _x()
        masks = _bayer_masks()
        params = np.array([3.0, 3.0, 1.2, 0.8, 0.4, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        model = gf.WLS_model_elliptical_nobounds(params, masks, x, np.zeros((SIZE, SIZE)))
        weights = np.ones((SIZE, SIZE))
        ravelsize = SIZE * SIZE
        jit_out = gf.WLS_chi_elliptical_nobounds(params, model, masks, weights, SIZE, ravelsize)
        py_out = gf.WLS_chi_elliptical_nobounds.py_func(params, model, masks, weights, SIZE, ravelsize)
        np.testing.assert_allclose(jit_out, py_out)
        np.testing.assert_allclose(jit_out, 0.0, atol=1e-6)


class TestWLSChiCircular:
    def test_basic_and_py_func(self):
        # params: [x, y, s, bg_0, bg_1, bg_2, A_0, A_1, A_2] -- one shared width,
        # duplicated into both the sigma_y and sigma_x slots before delegating to
        # the same WLS_model_nobounds STANDARD_DATA uses (see WLS_chi_circular_nobounds's
        # docstring). Build the generating model from that expanded [x,y,s,s,bg...,A...]
        # vector so chi against its own generating model is ~0, matching every other
        # WLS_chi_* test above.
        x = _x()
        masks = _bayer_masks()
        params = np.array([3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        full_params = np.array([3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        data = gf.WLS_model_nobounds(full_params, masks, x, np.zeros((SIZE, SIZE)))
        weights = np.ones((SIZE, SIZE))
        ravelsize = SIZE * SIZE
        jit_out = gf.WLS_chi_circular_nobounds(params, data, masks, weights, SIZE, ravelsize)
        py_out = gf.WLS_chi_circular_nobounds.py_func(params, data, masks, weights, SIZE, ravelsize)
        np.testing.assert_allclose(jit_out, py_out)
        np.testing.assert_allclose(jit_out, 0.0, atol=1e-6)


# ======================================================================
# Initial-guess helpers (_sum_and_centre_of_mass / _initial_sigma / _initial_theta)
# ======================================================================

class TestSumAndCentreOfMass:
    def test_recovers_known_centroid_and_py_func(self):
        data = _positive_blob(x0=4.0, y0=2.0)
        jit_out = gf._sum_and_centre_of_mass(data, SIZE)
        py_out = gf._sum_and_centre_of_mass.py_func(data, SIZE)
        np.testing.assert_allclose(jit_out, py_out)
        A, x_ig, y_ig = jit_out
        assert A > 0
        assert x_ig == pytest.approx(4.0, abs=0.5)
        assert y_ig == pytest.approx(2.0, abs=0.5)


class TestInitialSigma:
    def test_basic_and_py_func(self):
        data = _positive_blob(sigma=1.2)
        A, x_ig, y_ig = gf._sum_and_centre_of_mass(data, SIZE)
        jit_out = gf._initial_sigma(data, x_ig, y_ig, A, SIZE)
        py_out = gf._initial_sigma.py_func(data, x_ig, y_ig, A, SIZE)
        np.testing.assert_allclose(jit_out, py_out)
        assert all(s > 0 for s in jit_out)

    @pytest.mark.parametrize("fn", [gf._initial_sigma, gf._initial_sigma.py_func])
    def test_off_centre_blob_is_symmetric(self, fn):
        # Each axis' moment must be taken about its own centroid: for a round blob far off the
        # diagonal (row 10, column 5 of 16) the two widths agree. Pairing them the wrong way
        # round inflates both by the centroid separation.
        size = 16
        data = _positive_blob(size=size, x0=10.0, y0=5.0, background=0.0)
        A, x_ig, y_ig = gf._sum_and_centre_of_mass(data, size)
        s_row, s_col = fn(data, x_ig, y_ig, A, size)
        assert s_row == pytest.approx(s_col, rel=1e-6)
        assert s_row == pytest.approx(0.5 * 1.2, rel=1e-3)  # 0.5 x the blob's sigma


class TestInitialTheta:
    def test_basic_and_py_func(self):
        data = _positive_blob()
        A, x_ig, y_ig = gf._sum_and_centre_of_mass(data, SIZE)
        jit_out = gf._initial_theta(data, x_ig, y_ig, SIZE)
        py_out = gf._initial_theta.py_func(data, x_ig, y_ig, SIZE)
        assert jit_out == pytest.approx(py_out)
        assert -np.pi / 2 <= jit_out <= np.pi / 2

    def test_zero_data_skips_accumulation_branch(self):
        # All-zero data means the inner `if v > 0.0` body and the outer
        # `if A > 0.0` normalisation are both skipped -> theta falls back to
        # atan2(0, 0) == 0.0. Exercises the function with no positive pixels.
        data = np.zeros((SIZE, SIZE))
        jit_out = gf._initial_theta(data, 3.0, 3.0, SIZE)
        py_out = gf._initial_theta.py_func(data, 3.0, 3.0, SIZE)
        assert jit_out == py_out == 0.0


# ======================================================================
# initial_guess / initial_guess_elliptical
# ======================================================================

class TestInitialGuess:
    def test_basic_and_py_func(self):
        smoothed = _positive_blob(x0=4.0, y0=2.0)
        raw = _positive_blob(x0=4.0, y0=2.0, background=3.0)
        masks = _bayer_masks()
        jit_out = gf.initial_guess(smoothed, raw, masks)
        py_out = gf.initial_guess.py_func(smoothed, raw, masks)
        np.testing.assert_allclose(jit_out, py_out)
        assert jit_out.shape == (10,)  # 4 + 2*3
        assert np.all(np.isfinite(jit_out))

    def test_single_channel_mask(self):
        # n_ch=1 exercises the same loop body with a different channel count.
        smoothed = _positive_blob()
        raw = _positive_blob(background=3.0)
        masks = _bayer_masks(n_ch=1)
        out = gf.initial_guess(smoothed, raw, masks)
        assert out.shape == (6,)  # 4 + 2*1

    @pytest.mark.parametrize("fn", [gf.initial_guess, gf.initial_guess.py_func])
    def test_zero_minimum_background_guess_is_floored(self, fn):
        # A channel whose minimum pixel is 0 (common at a few background photons per pixel)
        # must not get a background guess of exactly 0: the model squares it, so 0 has a zero
        # gradient and stalls the fit (equal colour amplitudes). See gf._BG_GUESS_FLOOR.
        smoothed = _positive_blob()
        raw = _positive_blob(background=3.0)
        raw[0, 0:3] = 0.0  # one zero in each of the 3 channels (_bayer_masks cycles mod 3)
        out = fn(smoothed, raw, _bayer_masks())
        np.testing.assert_allclose(out[4:7], np.sqrt(gf._BG_GUESS_FLOOR))

    @pytest.mark.parametrize("fn", [gf.initial_guess, gf.initial_guess.py_func])
    def test_background_guess_above_floor_is_channel_minimum(self, fn):
        # The floor only acts below it: a channel minimum >= _BG_GUESS_FLOOR is used as is.
        smoothed = _positive_blob()
        raw = _positive_blob(background=3.0)
        masks = _bayer_masks()
        out = fn(smoothed, raw, masks)
        expected = [np.sqrt(raw[masks[:, :, c]].min()) for c in range(3)]
        assert min(expected) ** 2 >= gf._BG_GUESS_FLOOR
        np.testing.assert_allclose(out[4:7], expected)

    @pytest.mark.parametrize("fn", [gf.initial_guess, gf.initial_guess.py_func])
    def test_amplitude_guess_is_per_channel(self, fn):
        # Data built from the model with distinct colour amplitudes and the guess's own
        # position/width: each channel's amplitude guess recovers its own amplitude.
        masks = _bayer_masks()
        smoothed = _positive_blob(background=0.0)
        amps = np.array([50.0, 400.0, 1500.0])
        bgs = np.array([2.0, 3.0, 4.0])
        guess = fn(smoothed, smoothed, masks)
        params = guess.copy()
        params[4:7] = np.sqrt(bgs)
        params[7:10] = np.sqrt(amps)
        raw = gf.WLS_model_nobounds(params, masks, _x(SIZE), np.zeros((SIZE, SIZE)))
        out = fn(smoothed, raw, masks)
        np.testing.assert_allclose(out[7:10] ** 2, amps, rtol=1e-6)

    @pytest.mark.parametrize("fn", [gf.initial_guess, gf.initial_guess.py_func])
    def test_empty_channel_amplitude_guess_is_floored(self, fn):
        # A channel with no signal above its background must not get a 0 (or negative)
        # amplitude guess: in sqrt space that has a zero gradient. See gf._AMP_GUESS_FLOOR.
        masks = _bayer_masks()
        smoothed = _positive_blob()
        raw = _positive_blob(background=3.0)
        raw[masks[:, :, 0]] = 3.0  # channel 0 is flat background only
        out = fn(smoothed, raw, masks)
        assert out[7] == pytest.approx(np.sqrt(gf._AMP_GUESS_FLOOR))
        assert np.all(out[8:10] ** 2 > gf._AMP_GUESS_FLOOR)


class TestInitialGuessElliptical:
    def test_basic_and_py_func(self):
        smoothed = _positive_blob(x0=4.0, y0=2.0)
        raw = _positive_blob(x0=4.0, y0=2.0, background=3.0)
        masks = _bayer_masks()
        jit_out = gf.initial_guess_elliptical(smoothed, raw, masks)
        py_out = gf.initial_guess_elliptical.py_func(smoothed, raw, masks)
        np.testing.assert_allclose(jit_out, py_out)
        assert len(jit_out) == 11
        assert all(np.isfinite(v) for v in jit_out)

    @pytest.mark.parametrize("fn", [gf.initial_guess_elliptical, gf.initial_guess_elliptical.py_func])
    def test_zero_minimum_background_guess_is_floored(self, fn):
        # Same floor as initial_guess: a zero channel minimum must not give a 0 background guess.
        smoothed = _positive_blob()
        raw = _positive_blob(background=3.0)
        raw[0, 0:3] = 0.0  # one zero in each of the 3 channels
        out = fn(smoothed, raw, _bayer_masks())
        np.testing.assert_allclose(out[5:8], np.sqrt(gf._BG_GUESS_FLOOR))


# ======================================================================
# compute_A_median
# ======================================================================

class TestComputeAMedian:
    def test_odd_length_and_py_func(self):
        # 5x5 -> 25 elements (odd) -> the single-median-element branch.
        data = _positive_blob(size=5)
        jit_out = gf.compute_A_median(data)
        py_out = gf.compute_A_median.py_func(data)
        assert jit_out == pytest.approx(py_out)
        assert jit_out > 0  # real spot: sum(above median) - sum(below) > 0

    def test_even_length_and_py_func(self):
        # 4x4 -> 16 elements (even) -> the averaged-pair-median branch.
        data = _positive_blob(size=4)
        jit_out = gf.compute_A_median(data)
        py_out = gf.compute_A_median.py_func(data)
        assert jit_out == pytest.approx(py_out)

    def test_flat_data_near_zero(self):
        data = np.full((SIZE, SIZE), 5.0)
        out = gf.compute_A_median(data)
        assert out == pytest.approx(0.0, abs=1e-9)
