"""Full coverage tests for pyS3M.ImageAnalysisFunctions -- the strategy-pattern
Gaussian fitting engine SR_Functions.py/simulation/multicolour.py dispatch to.

Deliberately tiny synthetic data throughout (8x8 ROIs) built directly from the
same gaussoptfuncs model functions the fitter itself uses, so real `leastsq`
calls converge cleanly and quickly without any file/fixture I/O. Failure and
exception branches (leastsq non-converged success codes, malformed inputs)
are reached via `monkeypatch` on `scipy.optimize.leastsq` / `gaussoptfuncs`,
since realistic small data essentially never fails to converge on its own.
"""
from __future__ import annotations

import numpy as np
import pytest

import pyS3M.gaussoptfuncs as gaussoptfuncs
import pyS3M.ImageAnalysisFunctions as iaf
from pyS3M.ImageAnalysisFunctions import (
    FittingStrategy,
    FittingConstants,
    FittingParameters,
    FittingValidationError,
    FittingResultProcessor,
    StandardFittingProcessor,
    StandardIGFittingProcessor,
    StandardIterFittingProcessor,
    StandardDataFittingProcessor,
    EllipticalFittingProcessor,
    CircularFittingProcessor,
    NoColourFittingProcessor,
    JustColourFittingProcessor,
    RawColourFittingProcessor,
    PosthenColourFittingProcessor,
    Image_Analysis_Functions,
    _fit_puncta_method_standalone,
)


SIZE = 8


def _x(size=SIZE):
    return np.arange(size, dtype=np.float32)


def _bayer_masks(size=SIZE, n_ch=3):
    idx = np.arange(size * size).reshape(size, size) % n_ch
    masks = np.zeros((size, size, n_ch), dtype=np.bool_)
    for c in range(n_ch):
        masks[:, :, c] = idx == c
    return masks


def _synthetic_punctum(size=SIZE, x0=4.0, y0=4.0, sigma=1.3, amp=400.0, bg=20.0, masks=None):
    """Build a punctum from the real WLS_model_nobounds so a leastsq fit
    started near truth converges fast and cleanly (no real-noise randomness)."""
    if masks is None:
        masks = _bayer_masks(size)
    n_ch = masks.shape[-1]
    params = np.zeros(4 + 2 * n_ch)
    params[0], params[1] = x0, y0
    params[2] = params[3] = sigma
    params[4:4 + n_ch] = np.sqrt(bg)
    params[4 + n_ch:4 + 2 * n_ch] = np.sqrt(amp)
    x = _x(size)
    gauss_2d = np.zeros((size, size), dtype=np.float32)
    punctum = gaussoptfuncs.WLS_model_nobounds(params, masks, x, gauss_2d).astype(np.float32)
    return punctum, masks


def _weights(size=SIZE):
    return np.ones((size, size), dtype=np.float32)


class _FakeLeastsqResult:
    """Minimal stand-in for scipy.optimize.leastsq's full_output=True return."""


def _fail_leastsq(*args, **kwargs):
    # 5 is a genuine scipy failure code (max fev / dnorm no longer decreasing).
    n = len(kwargs["x0"] if "x0" in kwargs else args[1])
    return np.full(n, np.nan), None, {}, "forced failure", 5


# ======================================================================
# FittingParameters
# ======================================================================

class TestFittingParameters:
    def test_valid_construction(self):
        punctum, masks = _synthetic_punctum()
        params = FittingParameters(
            puncta=[punctum], smoothed_puncta=[punctum], weights=[_weights()],
            relative_coords=[[0.0, 0.0]], planes=[0], strategy=FittingStrategy.NOCOLOUR,
        )
        assert params.masks is None

    def test_mismatched_array_lengths_raises(self):
        punctum, masks = _synthetic_punctum()
        with pytest.raises(FittingValidationError, match="same length as puncta"):
            FittingParameters(
                puncta=[punctum], smoothed_puncta=[punctum, punctum], weights=[_weights()],
                relative_coords=[[0.0, 0.0]], planes=[0], strategy=FittingStrategy.NOCOLOUR,
            )

    def test_colour_strategy_without_masks_raises(self):
        punctum, masks = _synthetic_punctum()
        with pytest.raises(FittingValidationError, match="requires masks"):
            FittingParameters(
                puncta=[punctum], smoothed_puncta=[punctum], weights=[_weights()],
                relative_coords=[[0.0, 0.0]], planes=[0], strategy=FittingStrategy.STANDARD,
            )

    def test_masks_length_mismatch_raises(self):
        punctum, masks = _synthetic_punctum()
        with pytest.raises(FittingValidationError, match="Masks array must have same length"):
            FittingParameters(
                puncta=[punctum], smoothed_puncta=[punctum], weights=[_weights()],
                relative_coords=[[0.0, 0.0]], planes=[0], strategy=FittingStrategy.STANDARD,
                masks=[masks, masks],
            )


# ======================================================================
# FittingResultProcessor.calculate_errors
# ======================================================================

class TestCalculateErrors:
    def test_pcov_none(self):
        out = FittingResultProcessor.calculate_errors(None, FittingStrategy.NOCOLOUR)
        assert out == [np.nan] * FittingConstants.PARAM_DIMENSIONS[FittingStrategy.NOCOLOUR]["error"]

    def test_pcov_scalar(self):
        out = FittingResultProcessor.calculate_errors(np.inf, FittingStrategy.NOCOLOUR)
        assert all(np.isnan(v) for v in out)

    def test_pcov_0d_array(self):
        out = FittingResultProcessor.calculate_errors(np.array(np.inf), FittingStrategy.NOCOLOUR)
        assert all(np.isnan(v) for v in out)

    def test_pcov_contains_inf(self):
        pcov = np.eye(3)
        pcov[0, 0] = np.inf
        out = FittingResultProcessor.calculate_errors(pcov, FittingStrategy.JUSTCOLOUR)
        assert all(np.isnan(v) for v in out)

    def test_normal_pcov(self):
        pcov = np.diag([4.0, 9.0])
        out = FittingResultProcessor.calculate_errors(pcov, FittingStrategy.JUSTCOLOUR)
        assert out == pytest.approx([2.0, 3.0])

    def test_non_positive_diagonal_becomes_nan(self):
        pcov = np.diag([4.0, -1.0])
        out = FittingResultProcessor.calculate_errors(pcov, FittingStrategy.JUSTCOLOUR)
        assert out[0] == pytest.approx(2.0)
        assert np.isnan(out[1])

    def test_undersized_pcov_padded(self):
        # Padding is driven by pcov.shape[0], not the strategy's nominal size --
        # only a non-square (more rows than diagonal entries) pcov triggers it.
        pcov = np.zeros((3, 1))
        pcov[0, 0] = 4.0
        out = FittingResultProcessor.calculate_errors(pcov, FittingStrategy.NOCOLOUR)
        assert len(out) == 3
        assert out[0] == pytest.approx(2.0)
        assert all(np.isnan(v) for v in out[1:])

    def test_exception_path_returns_nan(self):
        # A 3-D array makes np.diag raise ValueError internally.
        pcov = np.ones((2, 2, 2))
        out = FittingResultProcessor.calculate_errors(pcov, FittingStrategy.NOCOLOUR)
        assert all(np.isnan(v) for v in out)

    def test_pfit_none_preserves_sqrt_space_error(self):
        # Backward-compat: omitting pfit returns sqrt(diag(pcov)) unchanged --
        # the pre-delta-method-fix behaviour, still correct for callers (e.g.
        # POSTHENCOLOUR) that don't square pfit before using it.
        pcov = np.diag([4.0, 9.0])
        out = FittingResultProcessor.calculate_errors(pcov, FittingStrategy.JUSTCOLOUR)
        assert out == pytest.approx([2.0, 3.0])

    def test_pfit_applies_jacobian_correction(self):
        # JUSTCOLOUR: the whole 2-param vector is sqrt-space (A, bg). A parameter
        # fitted as sqrt(A)=5.0 with sqrt-space sigma=2.0 has real-space sigma
        # sigma_A = 2*|sqrt(A)|*sigma_sqrt(A) = 2*5*2 = 20 (delta method for x^2).
        pcov = np.diag([4.0, 9.0])
        pfit = np.array([5.0, 3.0])
        out = FittingResultProcessor.calculate_errors(pcov, FittingStrategy.JUSTCOLOUR, pfit)
        assert out == pytest.approx([2 * 5.0 * 2.0, 2 * 3.0 * 3.0])

    def test_pfit_only_corrects_sqrt_space_range(self):
        # ELLIPTICAL: not STANDARD-like, so only the plain Jacobian correction
        # applies (no ratio-propagation branch) -- theta (index 4) and position
        # (0:4) are untouched, bg/A (5:11) get 2*|sqrt(V)|*sigma_sqrt(V).
        pfit = np.array([1.0, 2.0, 1.3, 1.3, 0.4, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0])
        pcov = np.diag(np.full(11, 4.0))  # sqrt-space sigma = 2.0 everywhere
        out = FittingResultProcessor.calculate_errors(pcov, FittingStrategy.ELLIPTICAL, pfit)
        assert out[:5] == pytest.approx([2.0, 2.0, 2.0, 2.0, 2.0])
        assert out[5:8] == pytest.approx([2 * 1.0 * 2.0] * 3)
        assert out[8:11] == pytest.approx([2 * 5.0 * 2.0] * 3)

    def test_standard_data_bg_amp_ratio_propagated(self):
        # STANDARD_DATA IS standard-like: bg/A (4:10) get the full ratio-
        # propagation (ImageAnalysisFunctions._propagate_ratio_errors), not
        # just the plain Jacobian correction -- position (0:4) still untouched.
        # Equal raw values (bg=[1,1,1]^2=1 each, A=[5,5,5]^2=25 each) and a
        # diagonal-only sqrt-space covariance (sigma=2 everywhere): hand
        # cross-checked against _propagate_ratio_errors directly.
        pfit = np.array([1.0, 2.0, 1.3, 1.3, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0])
        pcov = np.diag(np.full(10, 4.0))
        out = FittingResultProcessor.calculate_errors(pcov, FittingStrategy.STANDARD_DATA, pfit)
        assert out[:4] == pytest.approx([2.0, 2.0, 2.0, 2.0])
        assert out[4:7] == pytest.approx([1.08866211] * 3, rel=1e-6)
        assert out[7:10] == pytest.approx([0.21773242] * 3, rel=1e-6)

    def test_standard_data_undersized_pcov_skips_ratio_propagation(self):
        # pcov smaller than the bg/A group ranges it needs (degenerate-fit-style
        # mismatch, as already handled for the plain Jacobian correction) --
        # both groups fall below the 2-element minimum for a ratio to even be
        # defined, so the loop's `continue` guard is exercised without error.
        pfit = np.array([1.0, 2.0, 1.3, 1.3, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0])
        pcov = np.diag(np.full(5, 4.0))
        out = FittingResultProcessor.calculate_errors(pcov, FittingStrategy.STANDARD_DATA, pfit)
        assert out[:4] == pytest.approx([2.0, 2.0, 2.0, 2.0])

    def test_pfit_shorter_than_pcov_clips_safely(self):
        # Degenerate-fit-style length mismatch between pfit and pcov: the
        # correction must clip to the shared length rather than raise.
        pcov = np.diag([4.0, 9.0, 16.0])
        pfit = np.array([5.0])  # only covers index 0
        out = FittingResultProcessor.calculate_errors(pcov, FittingStrategy.JUSTCOLOUR, pfit)
        assert out[0] == pytest.approx(2 * 5.0 * 2.0)
        assert out[1] == pytest.approx(3.0)   # beyond pfit -> uncorrected
        assert out[2] == pytest.approx(4.0)   # beyond pfit -> uncorrected


class TestSqrtSpaceSlice:
    def test_standard_like(self):
        assert FittingResultProcessor._sqrt_space_slice(FittingStrategy.STANDARD, 10) == slice(4, 10)
        assert FittingResultProcessor._sqrt_space_slice(FittingStrategy.STANDARD_ITER, 10) == slice(4, 10)
        assert FittingResultProcessor._sqrt_space_slice(FittingStrategy.STANDARD_DATA, 10) == slice(4, 10)

    def test_elliptical(self):
        assert FittingResultProcessor._sqrt_space_slice(FittingStrategy.ELLIPTICAL, 11) == slice(5, 11)

    def test_circular(self):
        assert FittingResultProcessor._sqrt_space_slice(FittingStrategy.CIRCULAR, 9) == slice(3, 9)

    def test_nocolour_long_enough(self):
        assert FittingResultProcessor._sqrt_space_slice(FittingStrategy.NOCOLOUR, 6) == slice(4, 6)

    def test_nocolour_too_short(self):
        assert FittingResultProcessor._sqrt_space_slice(FittingStrategy.NOCOLOUR, 4) == slice(0, 0)

    def test_justcolour(self):
        assert FittingResultProcessor._sqrt_space_slice(FittingStrategy.JUSTCOLOUR, 2) == slice(0, 2)

    def test_rawcolour(self):
        assert FittingResultProcessor._sqrt_space_slice(FittingStrategy.RAWCOLOUR, 6) == slice(0, 6)

    def test_unhandled_strategy_returns_empty(self):
        assert FittingResultProcessor._sqrt_space_slice(FittingStrategy.POSTHENCOLOUR, 10) == slice(0, 0)


class TestBgAmpSlices:
    def test_standard_like_splits_at_midpoint(self):
        bg, amp = FittingResultProcessor._bg_amp_slices(FittingStrategy.STANDARD_DATA, 10)
        assert bg == slice(4, 7)
        assert amp == slice(7, 10)

    def test_elliptical_splits_at_midpoint(self):
        bg, amp = FittingResultProcessor._bg_amp_slices(FittingStrategy.ELLIPTICAL, 11)
        assert bg == slice(5, 8)
        assert amp == slice(8, 11)

    def test_circular_splits_at_midpoint(self):
        bg, amp = FittingResultProcessor._bg_amp_slices(FittingStrategy.CIRCULAR, 9)
        assert bg == slice(3, 6)
        assert amp == slice(6, 9)

    def test_rawcolour_splits_at_midpoint(self):
        bg, amp = FittingResultProcessor._bg_amp_slices(FittingStrategy.RAWCOLOUR, 6)
        assert bg == slice(0, 3)
        assert amp == slice(3, 6)

    def test_single_channel_pair_returns_empty(self):
        # NOCOLOUR/JUSTCOLOUR: one bg + one A value, n=2 -- no ratio to
        # propagate (a lone channel's fraction is always exactly 1).
        bg, amp = FittingResultProcessor._bg_amp_slices(FittingStrategy.NOCOLOUR, 6)
        assert bg == slice(0, 0)
        assert amp == slice(0, 0)

    def test_unhandled_strategy_returns_empty(self):
        bg, amp = FittingResultProcessor._bg_amp_slices(FittingStrategy.POSTHENCOLOUR, 10)
        assert bg == slice(0, 0)
        assert amp == slice(0, 0)


class TestPropagateRatioErrors:
    def test_symmetric_zero_cross_covariance(self):
        # Two equal channels, independent (no off-diagonal): hand-derived via
        # the delta method, grad_0 = [(T-V0)/T^2, -V0/T^2] = [0.25, -0.25] for
        # V=[1,1], T=2 -> Var(p0) = 0.25^2*4*1 + 0.25^2*4*1 = 0.5.
        out = FittingResultProcessor._propagate_ratio_errors(
            np.array([1.0, 1.0]), np.eye(2),
        )
        np.testing.assert_allclose(out, [np.sqrt(0.5), np.sqrt(0.5)])

    def test_full_covariance_differs_from_diagonal_only(self):
        # Regression guard: the off-diagonal terms must actually be used, not
        # silently dropped back to an independent-channels approximation.
        raw = np.array([4.0, 2.0, 2.0])
        sqrtcov = np.array([
            [0.01, -0.005, -0.004],
            [-0.005, 0.02, -0.003],
            [-0.004, -0.003, 0.015],
        ])
        full = FittingResultProcessor._propagate_ratio_errors(raw, sqrtcov)
        diag_only = FittingResultProcessor._propagate_ratio_errors(raw, np.diag(np.diag(sqrtcov)))
        np.testing.assert_allclose(full, [0.04823635, 0.04542445, 0.04077367], rtol=1e-6)
        assert not np.allclose(full, diag_only)

    def test_nonpositive_total_returns_nan(self):
        out = FittingResultProcessor._propagate_ratio_errors(
            np.array([0.0, 0.0]), np.eye(2),
        )
        assert np.all(np.isnan(out))

    def test_non_finite_covariance_returns_nan(self):
        cov = np.array([[1.0, np.inf], [np.inf, 1.0]])
        out = FittingResultProcessor._propagate_ratio_errors(np.array([1.0, 1.0]), cov)
        assert np.all(np.isnan(out))

    def test_empty_input(self):
        out = FittingResultProcessor._propagate_ratio_errors(np.array([]), np.zeros((0, 0)))
        assert out.shape == (0,)


class TestCalculateReducedChisquared:
    def test_basic(self):
        residuals = np.array([1.0, 2.0, 3.0])
        out = FittingResultProcessor.calculate_reduced_chisquared(residuals, 10, 4)
        assert out == pytest.approx((1 + 4 + 9) / 6)


class TestProcessCovariance:
    def test_enough_dof_and_real_pcov(self):
        pcov = np.eye(2)
        out = FittingResultProcessor.process_covariance(pcov, 2.0, 10, 4)
        np.testing.assert_allclose(out, pcov * 2.0)

    def test_not_enough_dof_returns_inf(self):
        pcov = np.eye(2)
        out = FittingResultProcessor.process_covariance(pcov, 2.0, 4, 4)
        assert out == np.inf

    def test_pcov_none_returns_inf(self):
        out = FittingResultProcessor.process_covariance(None, 2.0, 10, 4)
        assert out == np.inf


class TestComputeAmplitudeSnr:
    def test_pcov_not_ndarray_returns_zero(self):
        pfit = np.array([1, 1, 1, 1, 1, 1, 1, 2, 2, 2], dtype=float)
        assert FittingResultProcessor._compute_amplitude_snr(pfit, np.inf) == 0.0

    def test_nonpositive_variance_returns_zero(self):
        pfit = np.array([1, 1, 1, 1, 1, 1, 1, 2, 2, 2], dtype=float)
        pcov = np.diag([1, 1, 1, 1, 1, 1, 1, -1, 1, 1])
        assert FittingResultProcessor._compute_amplitude_snr(pfit, pcov) == 0.0

    def test_normal(self):
        pfit = np.array([1, 1, 1, 1, 1, 1, 1, 2, 2, 2], dtype=float)
        pcov = np.diag(np.ones(10) * 0.01)
        out = FittingResultProcessor._compute_amplitude_snr(pfit, pcov)
        assert out > 0


class TestComputeAmplitudeSnrElliptical:
    def test_pcov_not_ndarray_returns_zero(self):
        pfit = np.zeros(11)
        assert FittingResultProcessor._compute_amplitude_snr_elliptical(pfit, np.inf) == 0.0

    def test_nonpositive_variance_returns_zero(self):
        pfit = np.array([1, 1, 1, 1, 0, 1, 1, 1, 2, 2, 2], dtype=float)
        pcov = np.diag(np.concatenate([np.ones(8), [-1, 1, 1]]))
        assert FittingResultProcessor._compute_amplitude_snr_elliptical(pfit, pcov) == 0.0

    def test_normal(self):
        pfit = np.array([1, 1, 1, 1, 0, 1, 1, 1, 2, 2, 2], dtype=float)
        pcov = np.diag(np.ones(11) * 0.01)
        assert FittingResultProcessor._compute_amplitude_snr_elliptical(pfit, pcov) > 0


class TestComputeAmplitudeSnrCircular:
    # pfit layout: [x, y, s, bg_0, bg_1, bg_2, A_0, A_1, A_2] -- amplitudes start
    # at index 3 + n_ch = 6 (one earlier than _compute_amplitude_snr's 4 + n_ch,
    # since there's only one width parameter instead of sy/sx).
    def test_pcov_not_ndarray_returns_zero(self):
        pfit = np.zeros(9)
        assert FittingResultProcessor._compute_amplitude_snr_circular(pfit, np.inf) == 0.0

    def test_nonpositive_variance_returns_zero(self):
        pfit = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2], dtype=float)
        pcov = np.diag(np.concatenate([np.ones(6), [-1, 1, 1]]))
        assert FittingResultProcessor._compute_amplitude_snr_circular(pfit, pcov) == 0.0

    def test_normal(self):
        pfit = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2], dtype=float)
        pcov = np.diag(np.ones(9) * 0.01)
        assert FittingResultProcessor._compute_amplitude_snr_circular(pfit, pcov) > 0


# ======================================================================
# FittingResultProcessor.process_fit_results
# ======================================================================

class TestProcessFitResults:
    def test_pfit_none(self):
        pfit, err = FittingResultProcessor.process_fit_results(
            None, None, SIZE, [0.0, 0.0], FittingStrategy.NOCOLOUR,
        )
        assert np.all(np.isnan(pfit))
        assert len(pfit) == FittingConstants.PARAM_DIMENSIONS[FittingStrategy.NOCOLOUR]["fit"]

    def test_pfit_empty(self):
        pfit, err = FittingResultProcessor.process_fit_results(
            np.array([]), None, SIZE, [0.0, 0.0], FittingStrategy.JUSTCOLOUR,
        )
        assert np.all(np.isnan(pfit))

    def test_standard_like_normal_path(self):
        # bg=[4.5,4.5,4.5] -> squared [20.25]*3, sum=60.75=background_photons.
        # A=[20,20,20] -> squared [400]*3, sum=1200=photons. bg/A values in
        # `out` come back normalised to fractions (1/3 each, equal inputs).
        pfit = np.array([4.0, 4.0, 1.3, 1.3, 4.5, 4.5, 4.5, 20.0, 20.0, 20.0])
        n = len(pfit)
        pcov = np.diag(np.ones(n) * 1e-4)
        out, err = FittingResultProcessor.process_fit_results(
            pfit, pcov, SIZE, [1.0, 2.0], FittingStrategy.STANDARD, chisqr=1.0,
        )
        assert not np.any(np.isnan(out))
        assert out[0] == pytest.approx(5.0)  # x + relative_coords[0]
        assert out[1] == pytest.approx(6.0)
        assert out[4:7] == pytest.approx([1 / 3] * 3)  # bg_B/G/R fractions
        assert out[7:10] == pytest.approx([1 / 3] * 3)  # A_B/G/R fractions
        assert out[-3] == pytest.approx(1.0)   # chisqr
        assert out[-2] == pytest.approx(1200.0)  # photons (raw A total)
        assert out[-1] == pytest.approx(60.75)   # background_photons (raw bg total)

    def test_standard_like_position_out_of_bounds(self):
        pfit = np.array([-1.0, 4.0, 1.3, 1.3, 4.5, 4.5, 4.5, 20.0, 20.0, 20.0])
        pcov = np.diag(np.ones(len(pfit)) * 1e-4)
        out, err = FittingResultProcessor.process_fit_results(
            pfit, pcov, SIZE, [0.0, 0.0], FittingStrategy.STANDARD,
        )
        assert np.all(np.isnan(out))

    def test_standard_like_low_amplitude_snr_rejected(self):
        pfit = np.array([4.0, 4.0, 1.3, 1.3, 4.5, 4.5, 4.5, 20.0, 20.0, 20.0])
        pcov = np.diag(np.ones(len(pfit)) * 1e6)  # huge variance -> low SNR
        out, err = FittingResultProcessor.process_fit_results(
            pfit, pcov, SIZE, [0.0, 0.0], FittingStrategy.STANDARD,
        )
        assert np.all(np.isnan(out))

    def test_standard_like_degenerate_zero_background_rejected(self):
        # bg sqrt-space values all 0 -> squared bg all 0 -> background_photons
        # total <= 0 -- degenerate, same treatment as the position/SNR gates.
        pfit = np.array([4.0, 4.0, 1.3, 1.3, 0.0, 0.0, 0.0, 20.0, 20.0, 20.0])
        pcov = np.diag(np.ones(len(pfit)) * 1e-4)
        out, err = FittingResultProcessor.process_fit_results(
            pfit, pcov, SIZE, [0.0, 0.0], FittingStrategy.STANDARD, chisqr=1.0,
        )
        assert np.all(np.isnan(out))

    def test_standard_like_error_jacobian_corrected(self):
        # End-to-end: process_fit_results must pass pfit/pcov through to
        # calculate_errors so amplitude/background errors come back fully
        # ratio-propagated (real-space, fractional units matching `out`'s
        # now-normalised values), not raw sqrt-space sigma. Equal-valued
        # groups + diagonal-only pcov: hand cross-checked against
        # _propagate_ratio_errors directly (same numbers as
        # TestCalculateErrors.test_standard_data_bg_amp_ratio_propagated,
        # scaled for this test's sigma=0.01 vs that test's sigma=2.0).
        pfit = np.array([4.0, 4.0, 1.3, 1.3, 4.5, 4.5, 4.5, 20.0, 20.0, 20.0])
        pcov = np.diag(np.full(len(pfit), 0.01 ** 2))  # sqrt-space sigma = 0.01
        out, err = FittingResultProcessor.process_fit_results(
            pfit, pcov, SIZE, [0.0, 0.0], FittingStrategy.STANDARD, chisqr=1.0,
        )
        assert not np.any(np.isnan(out))
        assert err[4:7] == pytest.approx([0.001209624564337372] * 3, rel=1e-9)
        assert err[7:10] == pytest.approx([0.0002721655269759087] * 3, rel=1e-9)

    def test_standard_like_relative_coords_none_skips_offset(self):
        pfit = np.array([4.0, 4.0, 1.3, 1.3, 4.5, 4.5, 4.5, 20.0, 20.0, 20.0])
        pcov = np.diag(np.ones(len(pfit)) * 1e-4)
        out, err = FittingResultProcessor.process_fit_results(
            pfit, pcov, SIZE, None, FittingStrategy.STANDARD,
        )
        assert out[0] == pytest.approx(4.0)

    def test_elliptical_normal_path(self):
        pfit = np.array([4.0, 4.0, 1.3, 1.3, 0.4, 4.5, 4.5, 4.5, 20.0, 20.0, 20.0])
        pcov = np.diag(np.ones(len(pfit)) * 1e-4)
        out, err = FittingResultProcessor.process_fit_results(
            pfit, pcov, SIZE, [0.0, 0.0], FittingStrategy.ELLIPTICAL,
        )
        assert not np.any(np.isnan(out))
        assert out[4] == pytest.approx(0.4)  # theta not squared

    def test_elliptical_position_out_of_bounds(self):
        pfit = np.array([-1.0, 4.0, 1.3, 1.3, 0.4, 4.5, 4.5, 4.5, 20.0, 20.0, 20.0])
        pcov = np.diag(np.ones(len(pfit)) * 1e-4)
        out, err = FittingResultProcessor.process_fit_results(
            pfit, pcov, SIZE, [0.0, 0.0], FittingStrategy.ELLIPTICAL,
        )
        assert np.all(np.isnan(out))

    def test_elliptical_low_amplitude_snr_rejected(self):
        pfit = np.array([4.0, 4.0, 1.3, 1.3, 0.4, 4.5, 4.5, 4.5, 20.0, 20.0, 20.0])
        pcov = np.diag(np.ones(len(pfit)) * 1e6)
        out, err = FittingResultProcessor.process_fit_results(
            pfit, pcov, SIZE, [0.0, 0.0], FittingStrategy.ELLIPTICAL,
        )
        assert np.all(np.isnan(out))

    def test_circular_normal_path(self):
        # [x, y, s, bg_0, bg_1, bg_2, A_0, A_1, A_2] -- one fewer leading param
        # than STANDARD (no separate sx/sy), so bg/A squaring starts one index
        # earlier too (pfit_processed[3:3+2*n_ch] vs STANDARD's [4:4+2*n_ch]).
        pfit = np.array([4.0, 4.0, 1.3, 4.5, 4.5, 4.5, 20.0, 20.0, 20.0])
        pcov = np.diag(np.ones(len(pfit)) * 1e-4)
        out, err = FittingResultProcessor.process_fit_results(
            pfit, pcov, SIZE, [1.0, 2.0], FittingStrategy.CIRCULAR, chisqr=1.0,
        )
        assert not np.any(np.isnan(out))
        assert out[0] == pytest.approx(5.0)  # x + relative_coords[0]
        assert out[1] == pytest.approx(6.0)
        assert out[-1] == pytest.approx(1.0)  # chisqr appended
        assert out[3] == pytest.approx(4.5 ** 2)  # bg squared
        assert out[6] == pytest.approx(20.0 ** 2)  # amplitude squared

    def test_circular_position_out_of_bounds(self):
        # CIRCULAR's own [:3] position gate (x, y, s) -- one narrower than
        # position_strategies' shared [:4] (x, y, sx, sy) gate above.
        pfit = np.array([-1.0, 4.0, 1.3, 4.5, 4.5, 4.5, 20.0, 20.0, 20.0])
        pcov = np.diag(np.ones(len(pfit)) * 1e-4)
        out, err = FittingResultProcessor.process_fit_results(
            pfit, pcov, SIZE, [0.0, 0.0], FittingStrategy.CIRCULAR,
        )
        assert np.all(np.isnan(out))

    def test_circular_low_amplitude_snr_rejected(self):
        pfit = np.array([4.0, 4.0, 1.3, 4.5, 4.5, 4.5, 20.0, 20.0, 20.0])
        pcov = np.diag(np.ones(len(pfit)) * 1e6)
        out, err = FittingResultProcessor.process_fit_results(
            pfit, pcov, SIZE, [0.0, 0.0], FittingStrategy.CIRCULAR,
        )
        assert np.all(np.isnan(out))

    def test_nocolour_normal_path_squares_bg_and_amplitude(self):
        pfit = np.array([4.0, 4.0, 1.3, 1.3, 4.5, 20.0])
        out, err = FittingResultProcessor.process_fit_results(
            pfit, None, SIZE, [0.0, 0.0], FittingStrategy.NOCOLOUR,
        )
        assert out[4] == pytest.approx(4.5 ** 2)
        assert out[5] == pytest.approx(20.0 ** 2)

    def test_justcolour_normal_path_squares_both(self):
        pfit = np.array([3.0, 2.0])
        out, err = FittingResultProcessor.process_fit_results(
            pfit, None, SIZE, [0.0, 0.0], FittingStrategy.JUSTCOLOUR,
        )
        assert out[0] == pytest.approx(9.0)
        assert out[1] == pytest.approx(4.0)

    def test_rawcolour_normal_path_squares_all(self):
        pfit = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        out, err = FittingResultProcessor.process_fit_results(
            pfit, None, SIZE, [0.0, 0.0], FittingStrategy.RAWCOLOUR,
        )
        np.testing.assert_allclose(out[:6], [1.0, 1.0, 1.0, 4.0, 4.0, 4.0])


class TestFittingProcessorAbstractBody:
    def test_base_method_body_is_a_noop(self):
        # ABC prevents direct instantiation, but the abstract method's own body
        # (a bare `pass`) is reachable via an explicit unbound call through a
        # concrete subclass instance -- covers the stub itself.
        punctum, masks = _synthetic_punctum()
        instance = NoColourFittingProcessor()
        result = iaf.FittingProcessor.fit_single_punctum(
            instance, punctum, punctum, _weights(), [0.0, 0.0], masks=masks,
        )
        assert result is None


# ======================================================================
# StandardFittingProcessor / StandardIGFittingProcessor
# ======================================================================

class TestStandardFittingProcessor:
    def test_masks_none_raises(self):
        punctum, masks = _synthetic_punctum()
        with pytest.raises(FittingValidationError, match="requires masks"):
            StandardFittingProcessor().fit_single_punctum(punctum, punctum, _weights(), [0.0, 0.0], masks=None)

    def test_all_nonpositive_smoothed_returns_nan(self):
        punctum, masks = _synthetic_punctum()
        smoothed = np.zeros((SIZE, SIZE), dtype=np.float32)
        pfit, err = StandardFittingProcessor().fit_single_punctum(punctum, smoothed, _weights(), [0.0, 0.0], masks=masks)
        assert np.all(np.isnan(pfit))

    def test_normal_fit_recovers_known_position(self):
        punctum, masks = _synthetic_punctum(x0=4.0, y0=4.0)
        pfit, err = StandardFittingProcessor().fit_single_punctum(
            punctum, punctum, _weights(), [10.0, 20.0], masks=masks,
        )
        assert pfit[0] == pytest.approx(14.0, abs=1.0)
        assert pfit[1] == pytest.approx(24.0, abs=1.0)

    def test_leastsq_non_convergence_returns_nan(self, monkeypatch):
        punctum, masks = _synthetic_punctum()
        monkeypatch.setattr(iaf, "leastsq", _fail_leastsq)
        pfit, err = StandardFittingProcessor().fit_single_punctum(punctum, punctum, _weights(), [0.0, 0.0], masks=masks)
        assert np.all(np.isnan(pfit))

    def test_exception_path_returns_nan(self, monkeypatch):
        punctum, masks = _synthetic_punctum()

        def _raise(*a, **kw):
            raise RuntimeError("forced leastsq failure")

        monkeypatch.setattr(iaf, "leastsq", _raise)
        pfit, err = StandardFittingProcessor().fit_single_punctum(punctum, punctum, _weights(), [0.0, 0.0], masks=masks)
        assert np.all(np.isnan(pfit))


class TestStandardIGFittingProcessor:
    def test_masks_none_raises(self):
        punctum, masks = _synthetic_punctum()
        with pytest.raises(FittingValidationError, match="requires masks"):
            StandardIGFittingProcessor().fit_single_punctum(punctum, punctum, _weights(), (4, 4, 1.3, 1.3, 20, 400), masks=None)

    def test_all_nonpositive_smoothed_returns_nan(self):
        punctum, masks = _synthetic_punctum()
        smoothed = np.zeros((SIZE, SIZE), dtype=np.float32)
        pfit, err = StandardIGFittingProcessor().fit_single_punctum(
            punctum, smoothed, _weights(), (4, 4, 1.3, 1.3, 20, 400), masks=masks,
        )
        assert np.all(np.isnan(pfit))

    def test_normal_fit_seeded_from_demosaiced_result(self):
        punctum, masks = _synthetic_punctum(x0=4.0, y0=4.0)
        pfit, err = StandardIGFittingProcessor().fit_single_punctum(
            punctum, punctum, _weights(), (4.0, 4.0, 1.3, 1.3, 20.0, 400.0), masks=masks,
        )
        assert pfit[0] == pytest.approx(4.0, abs=1.0)
        assert pfit[1] == pytest.approx(4.0, abs=1.0)


# ======================================================================
# StandardIterFittingProcessor / StandardDataFittingProcessor
# ======================================================================

class TestStandardIterFittingProcessor:
    def test_masks_none_raises(self):
        punctum, masks = _synthetic_punctum()
        with pytest.raises(FittingValidationError, match="requires masks"):
            StandardIterFittingProcessor().fit_single_punctum(punctum, punctum, _weights(), [0.0, 0.0], masks=None)

    def test_all_nonpositive_smoothed_returns_nan(self):
        punctum, masks = _synthetic_punctum()
        smoothed = np.zeros((SIZE, SIZE), dtype=np.float32)
        pfit, err = StandardIterFittingProcessor().fit_single_punctum(punctum, smoothed, _weights(), [0.0, 0.0], masks=masks)
        assert np.all(np.isnan(pfit))

    def test_normal_fit_all_three_stages(self):
        punctum, masks = _synthetic_punctum(x0=4.0, y0=4.0)
        pfit, err = StandardIterFittingProcessor(readnoise=1.5).fit_single_punctum(
            punctum, punctum, _weights(), [0.0, 0.0], masks=masks,
        )
        assert pfit[0] == pytest.approx(4.0, abs=1.0)

    def test_stage1_failure_returns_nan(self, monkeypatch):
        punctum, masks = _synthetic_punctum()
        proc = StandardIterFittingProcessor()
        monkeypatch.setattr(proc, "_leastsq_step", lambda *a, **kw: (a[0], None, 5))
        pfit, err = proc.fit_single_punctum(punctum, punctum, _weights(), [0.0, 0.0], masks=masks)
        assert np.all(np.isnan(pfit))

    def test_stage2_failure_returns_nan(self, monkeypatch):
        punctum, masks = _synthetic_punctum()
        proc = StandardIterFittingProcessor()
        real_step = proc._leastsq_step
        calls = {"n": 0}

        def _step(x0, data, masks_, weights_, size):
            calls["n"] += 1
            if calls["n"] == 1:
                return real_step(x0, data, masks_, weights_, size)
            return x0, None, 5

        monkeypatch.setattr(proc, "_leastsq_step", _step)
        pfit, err = proc.fit_single_punctum(punctum, punctum, _weights(), [0.0, 0.0], masks=masks)
        assert np.all(np.isnan(pfit))

    def test_stage3_failure_returns_nan(self, monkeypatch):
        punctum, masks = _synthetic_punctum()
        proc = StandardIterFittingProcessor()
        real_step = proc._leastsq_step
        calls = {"n": 0}

        def _step(x0, data, masks_, weights_, size):
            calls["n"] += 1
            if calls["n"] <= 2:
                return real_step(x0, data, masks_, weights_, size)
            return x0, None, 5

        monkeypatch.setattr(proc, "_leastsq_step", _step)
        pfit, err = proc.fit_single_punctum(punctum, punctum, _weights(), [0.0, 0.0], masks=masks)
        assert np.all(np.isnan(pfit))

    def test_exception_path_returns_nan(self, monkeypatch):
        punctum, masks = _synthetic_punctum()
        proc = StandardIterFittingProcessor()

        def _raise(*a, **kw):
            raise RuntimeError("forced failure")

        monkeypatch.setattr(proc, "_leastsq_step", _raise)
        pfit, err = proc.fit_single_punctum(punctum, punctum, _weights(), [0.0, 0.0], masks=masks)
        assert np.all(np.isnan(pfit))


class TestStandardDataFittingProcessor:
    def test_masks_none_raises(self):
        punctum, masks = _synthetic_punctum()
        with pytest.raises(FittingValidationError, match="requires masks"):
            StandardDataFittingProcessor().fit_single_punctum(punctum, punctum, _weights(), [0.0, 0.0], masks=None)

    def test_all_nonpositive_smoothed_returns_nan(self):
        punctum, masks = _synthetic_punctum()
        smoothed = np.zeros((SIZE, SIZE), dtype=np.float32)
        pfit, err = StandardDataFittingProcessor().fit_single_punctum(punctum, smoothed, _weights(), [0.0, 0.0], masks=masks)
        assert np.all(np.isnan(pfit))

    def test_normal_fit_all_three_stages(self):
        punctum, masks = _synthetic_punctum(x0=4.0, y0=4.0)
        pfit, err = StandardDataFittingProcessor(readnoise=1.5).fit_single_punctum(
            punctum, punctum, _weights(), [0.0, 0.0], masks=masks,
        )
        assert pfit[0] == pytest.approx(4.0, abs=1.0)

    def test_stage1_failure_returns_nan(self, monkeypatch):
        punctum, masks = _synthetic_punctum()
        proc = StandardDataFittingProcessor()
        monkeypatch.setattr(proc, "_leastsq_step", lambda *a, **kw: (a[0], None, 5))
        pfit, err = proc.fit_single_punctum(punctum, punctum, _weights(), [0.0, 0.0], masks=masks)
        assert np.all(np.isnan(pfit))

    def test_stage2_failure_returns_nan(self, monkeypatch):
        punctum, masks = _synthetic_punctum()
        proc = StandardDataFittingProcessor()
        real_step = proc._leastsq_step
        calls = {"n": 0}

        def _step(x0, data, masks_, weights_, size):
            calls["n"] += 1
            if calls["n"] == 1:
                return real_step(x0, data, masks_, weights_, size)
            return x0, None, 5

        monkeypatch.setattr(proc, "_leastsq_step", _step)
        pfit, err = proc.fit_single_punctum(punctum, punctum, _weights(), [0.0, 0.0], masks=masks)
        assert np.all(np.isnan(pfit))

    def test_stage3_failure_returns_nan(self, monkeypatch):
        punctum, masks = _synthetic_punctum()
        proc = StandardDataFittingProcessor()
        real_step = proc._leastsq_step
        calls = {"n": 0}

        def _step(x0, data, masks_, weights_, size):
            calls["n"] += 1
            if calls["n"] <= 2:
                return real_step(x0, data, masks_, weights_, size)
            return x0, None, 5

        monkeypatch.setattr(proc, "_leastsq_step", _step)
        pfit, err = proc.fit_single_punctum(punctum, punctum, _weights(), [0.0, 0.0], masks=masks)
        assert np.all(np.isnan(pfit))

    def test_exception_path_returns_nan(self, monkeypatch):
        punctum, masks = _synthetic_punctum()
        proc = StandardDataFittingProcessor()

        def _raise(*a, **kw):
            raise RuntimeError("forced failure")

        monkeypatch.setattr(proc, "_leastsq_step", _raise)
        pfit, err = proc.fit_single_punctum(punctum, punctum, _weights(), [0.0, 0.0], masks=masks)
        assert np.all(np.isnan(pfit))


# ======================================================================
# EllipticalFittingProcessor
# ======================================================================

class TestEllipticalFittingProcessor:
    def test_masks_none_raises(self):
        punctum, masks = _synthetic_punctum()
        with pytest.raises(FittingValidationError, match="requires masks"):
            EllipticalFittingProcessor().fit_single_punctum(punctum, punctum, _weights(), [0.0, 0.0], masks=None)

    def test_all_nonpositive_smoothed_returns_nan(self):
        punctum, masks = _synthetic_punctum()
        smoothed = np.zeros((SIZE, SIZE), dtype=np.float32)
        pfit, err = EllipticalFittingProcessor().fit_single_punctum(punctum, smoothed, _weights(), [0.0, 0.0], masks=masks)
        assert np.all(np.isnan(pfit))

    def test_normal_fit(self):
        punctum, masks = _synthetic_punctum(x0=4.0, y0=4.0)
        pfit, err = EllipticalFittingProcessor().fit_single_punctum(punctum, punctum, _weights(), [0.0, 0.0], masks=masks)
        # 11 model params + appended chisqr; PARAM_DIMENSIONS["fit"]=13 includes
        # the plane index appended later by Image_Analysis_Functions.fit_puncta_method.
        assert len(pfit) == 12

    def test_leastsq_non_convergence_returns_nan(self, monkeypatch):
        punctum, masks = _synthetic_punctum()
        monkeypatch.setattr(iaf, "leastsq", _fail_leastsq)
        pfit, err = EllipticalFittingProcessor().fit_single_punctum(punctum, punctum, _weights(), [0.0, 0.0], masks=masks)
        assert np.all(np.isnan(pfit))

    def test_exception_path_returns_nan(self, monkeypatch):
        punctum, masks = _synthetic_punctum()

        def _raise(*a, **kw):
            raise RuntimeError("forced failure")

        monkeypatch.setattr(iaf, "leastsq", _raise)
        pfit, err = EllipticalFittingProcessor().fit_single_punctum(punctum, punctum, _weights(), [0.0, 0.0], masks=masks)
        assert np.all(np.isnan(pfit))


# ======================================================================
# CircularFittingProcessor
# ======================================================================

class TestCircularFittingProcessor:
    def test_masks_none_raises(self):
        punctum, masks = _synthetic_punctum()
        with pytest.raises(FittingValidationError, match="requires masks"):
            CircularFittingProcessor().fit_single_punctum(punctum, punctum, _weights(), [0.0, 0.0], masks=None)

    def test_all_nonpositive_smoothed_returns_nan(self):
        punctum, masks = _synthetic_punctum()
        smoothed = np.zeros((SIZE, SIZE), dtype=np.float32)
        pfit, err = CircularFittingProcessor().fit_single_punctum(punctum, smoothed, _weights(), [0.0, 0.0], masks=masks)
        assert np.all(np.isnan(pfit))

    def test_normal_fit_all_three_stages(self):
        # _synthetic_punctum builds sx == sy by construction (params[2]=params[3]=sigma),
        # i.e. a genuinely circular PSF -- exactly what CircularFittingProcessor assumes.
        punctum, masks = _synthetic_punctum(x0=4.0, y0=4.0)
        pfit, err = CircularFittingProcessor().fit_single_punctum(punctum, punctum, _weights(), [0.0, 0.0], masks=masks)
        # 9 model params ([x,y,s,bg x3,A x3]) + appended chisqr; PARAM_DIMENSIONS["fit"]=11
        # includes the plane index appended later by Image_Analysis_Functions.fit_puncta_method
        # (same +1-beyond-fit_single_punctum convention as EllipticalFittingProcessor above).
        assert len(pfit) == 10
        assert not np.any(np.isnan(pfit))
        assert pfit[0] == pytest.approx(4.0, abs=1.0)
        assert pfit[1] == pytest.approx(4.0, abs=1.0)

    def test_stage1_failure_returns_nan(self, monkeypatch):
        punctum, masks = _synthetic_punctum()
        proc = CircularFittingProcessor()
        monkeypatch.setattr(proc, "_leastsq_step", lambda *a, **kw: (a[0], None, 5))
        pfit, err = proc.fit_single_punctum(punctum, punctum, _weights(), [0.0, 0.0], masks=masks)
        assert np.all(np.isnan(pfit))

    def test_stage2_failure_returns_nan(self, monkeypatch):
        punctum, masks = _synthetic_punctum()
        proc = CircularFittingProcessor()
        real_step = proc._leastsq_step
        calls = {"n": 0}

        def _step(x0, data, masks_, weights_, size):
            calls["n"] += 1
            if calls["n"] == 1:
                return real_step(x0, data, masks_, weights_, size)
            return x0, None, 5

        monkeypatch.setattr(proc, "_leastsq_step", _step)
        pfit, err = proc.fit_single_punctum(punctum, punctum, _weights(), [0.0, 0.0], masks=masks)
        assert np.all(np.isnan(pfit))

    def test_stage3_failure_returns_nan(self, monkeypatch):
        punctum, masks = _synthetic_punctum()
        proc = CircularFittingProcessor()
        real_step = proc._leastsq_step
        calls = {"n": 0}

        def _step(x0, data, masks_, weights_, size):
            calls["n"] += 1
            if calls["n"] <= 2:
                return real_step(x0, data, masks_, weights_, size)
            return x0, None, 5

        monkeypatch.setattr(proc, "_leastsq_step", _step)
        pfit, err = proc.fit_single_punctum(punctum, punctum, _weights(), [0.0, 0.0], masks=masks)
        assert np.all(np.isnan(pfit))

    def test_exception_path_returns_nan(self, monkeypatch):
        punctum, masks = _synthetic_punctum()
        proc = CircularFittingProcessor()

        def _raise(*a, **kw):
            raise RuntimeError("forced failure")

        monkeypatch.setattr(proc, "_leastsq_step", _raise)
        pfit, err = proc.fit_single_punctum(punctum, punctum, _weights(), [0.0, 0.0], masks=masks)
        assert np.all(np.isnan(pfit))


# ======================================================================
# NoColourFittingProcessor
# ======================================================================

class TestNoColourFittingProcessor:
    def test_normal_fit_recovers_known_position(self):
        punctum, masks = _synthetic_punctum(x0=4.0, y0=4.0)
        pfit, err = NoColourFittingProcessor().fit_single_punctum(punctum, punctum, _weights(), [0.0, 0.0])
        assert pfit[0] == pytest.approx(4.0, abs=1.5)

    def test_leastsq_non_convergence_returns_nan(self, monkeypatch):
        punctum, masks = _synthetic_punctum()
        monkeypatch.setattr(iaf, "leastsq", _fail_leastsq)
        pfit, err = NoColourFittingProcessor().fit_single_punctum(punctum, punctum, _weights(), [0.0, 0.0])
        assert np.all(np.isnan(pfit))

    def test_exception_path_returns_nan(self, monkeypatch):
        punctum, masks = _synthetic_punctum()

        def _raise(*a, **kw):
            raise RuntimeError("forced failure")

        monkeypatch.setattr(iaf, "leastsq", _raise)
        pfit, err = NoColourFittingProcessor().fit_single_punctum(punctum, punctum, _weights(), [0.0, 0.0])
        assert np.all(np.isnan(pfit))


# ======================================================================
# JustColourFittingProcessor
# ======================================================================

class TestJustColourFittingProcessor:
    def test_masks_none_raises(self):
        punctum, masks = _synthetic_punctum()
        with pytest.raises(FittingValidationError, match="requires masks"):
            JustColourFittingProcessor().fit_single_punctum(punctum, punctum, _weights(), [4.0, 4.0, 1.3, 1.3], masks=None)

    def test_normal_fit(self):
        punctum, masks = _synthetic_punctum(x0=4.0, y0=4.0)
        pfit, err = JustColourFittingProcessor().fit_single_punctum(
            punctum, punctum, _weights(), [4.0, 4.0, 1.3, 1.3], masks=masks,
        )
        # 2 model params + appended chisqr; PARAM_DIMENSIONS["fit"]=4 includes
        # the plane index appended later by Image_Analysis_Functions.fit_puncta_method.
        assert len(pfit) == 3

    def test_leastsq_non_convergence_returns_nan(self, monkeypatch):
        punctum, masks = _synthetic_punctum()
        monkeypatch.setattr(iaf, "leastsq", _fail_leastsq)
        pfit, err = JustColourFittingProcessor().fit_single_punctum(
            punctum, punctum, _weights(), [4.0, 4.0, 1.3, 1.3], masks=masks,
        )
        assert np.all(np.isnan(pfit))

    def test_exception_path_returns_nan(self, monkeypatch):
        punctum, masks = _synthetic_punctum()

        def _raise(*a, **kw):
            raise RuntimeError("forced failure")

        monkeypatch.setattr(iaf, "leastsq", _raise)
        pfit, err = JustColourFittingProcessor().fit_single_punctum(
            punctum, punctum, _weights(), [4.0, 4.0, 1.3, 1.3], masks=masks,
        )
        assert np.all(np.isnan(pfit))


# ======================================================================
# RawColourFittingProcessor
# ======================================================================

class TestRawColourFittingProcessor:
    def test_masks_none_raises(self):
        punctum, masks = _synthetic_punctum()
        with pytest.raises(FittingValidationError, match="requires masks"):
            RawColourFittingProcessor().fit_single_punctum(punctum, punctum, _weights(), [4.0, 4.0, 1.3, 1.3], masks=None)

    def test_normal_fit(self):
        punctum, masks = _synthetic_punctum(x0=4.0, y0=4.0)
        pfit, err = RawColourFittingProcessor().fit_single_punctum(
            punctum, punctum, _weights(), [4.0, 4.0, 1.3, 1.3], masks=masks,
        )
        # 6 model params + appended chisqr; PARAM_DIMENSIONS["fit"]=8 includes
        # the plane index appended later by Image_Analysis_Functions.fit_puncta_method.
        assert len(pfit) == 7

    def test_leastsq_non_convergence_returns_nan(self, monkeypatch):
        punctum, masks = _synthetic_punctum()
        monkeypatch.setattr(iaf, "leastsq", _fail_leastsq)
        pfit, err = RawColourFittingProcessor().fit_single_punctum(
            punctum, punctum, _weights(), [4.0, 4.0, 1.3, 1.3], masks=masks,
        )
        assert np.all(np.isnan(pfit))

    def test_exception_path_returns_nan(self, monkeypatch):
        punctum, masks = _synthetic_punctum()

        def _raise(*a, **kw):
            raise RuntimeError("forced failure")

        monkeypatch.setattr(iaf, "leastsq", _raise)
        pfit, err = RawColourFittingProcessor().fit_single_punctum(
            punctum, punctum, _weights(), [4.0, 4.0, 1.3, 1.3], masks=masks,
        )
        assert np.all(np.isnan(pfit))


# ======================================================================
# PosthenColourFittingProcessor
# ======================================================================

class TestPosthenColourFittingProcessor:
    def test_masks_none_raises(self):
        punctum, masks = _synthetic_punctum()
        with pytest.raises(FittingValidationError, match="requires masks"):
            PosthenColourFittingProcessor().fit_single_punctum(punctum, punctum, _weights(), [0.0, 0.0], masks=None)

    def test_normal_fit_default_raw_punctum(self):
        punctum, masks = _synthetic_punctum(x0=4.0, y0=4.0)
        pfit, err = PosthenColourFittingProcessor().fit_single_punctum(
            punctum, punctum, _weights(), [0.0, 0.0], masks=masks,
        )
        assert not np.any(np.isnan(pfit))  # bug fix confirmed: real (non-NaN) result now

    def test_normal_fit_explicit_raw_punctum(self):
        punctum, masks = _synthetic_punctum(x0=4.0, y0=4.0)
        pfit, err = PosthenColourFittingProcessor().fit_single_punctum(
            punctum, punctum, _weights(), [0.0, 0.0], masks=masks, raw_punctum=punctum,
        )
        assert not np.any(np.isnan(pfit))

    def test_colour_leastsq_non_convergence_returns_nan(self, monkeypatch):
        punctum, masks = _synthetic_punctum()
        monkeypatch.setattr(iaf, "leastsq", _fail_leastsq)
        pfit, err = PosthenColourFittingProcessor().fit_single_punctum(
            punctum, punctum, _weights(), [0.0, 0.0], masks=masks,
        )
        assert np.all(np.isnan(pfit))

    def test_exception_path_returns_nan(self, monkeypatch):
        punctum, masks = _synthetic_punctum()

        def _raise(*a, **kw):
            raise RuntimeError("forced failure")

        monkeypatch.setattr(iaf, "leastsq", _raise)
        pfit, err = PosthenColourFittingProcessor().fit_single_punctum(
            punctum, punctum, _weights(), [0.0, 0.0], masks=masks,
        )
        assert np.all(np.isnan(pfit))


# ======================================================================
# Image_Analysis_Functions -- serial + parallel dispatch
# ======================================================================

class TestImageAnalysisFunctionsSerial:
    def test_fit_puncta_method_nocolour(self):
        punctum, masks = _synthetic_punctum(x0=4.0, y0=4.0)
        iaf_obj = Image_Analysis_Functions()
        pfit, perr = iaf_obj.fit_puncta_method(
            [punctum], [punctum], [_weights()], [[0.0, 0.0]], [0], FittingStrategy.NOCOLOUR,
        )
        assert pfit.shape[0] == 1
        assert pfit[0, 0] == pytest.approx(4.0, abs=1.5)

    def test_fit_puncta_method_standard_with_masks_derives_dims(self):
        punctum, masks = _synthetic_punctum(x0=4.0, y0=4.0)
        iaf_obj = Image_Analysis_Functions()
        pfit, perr = iaf_obj.fit_puncta_method(
            [punctum], [punctum], [_weights()], [[0.0, 0.0]], [0],
            FittingStrategy.STANDARD, masks=[masks],
        )
        assert pfit.shape == (1, 4 + 2 * 3 + 4)
        assert perr.shape == (1, 4 + 2 * 3)

    def test_fit_puncta_method_circular_with_masks_derives_dims(self):
        punctum, masks = _synthetic_punctum(x0=4.0, y0=4.0)
        iaf_obj = Image_Analysis_Functions()
        pfit, perr = iaf_obj.fit_puncta_method(
            [punctum], [punctum], [_weights()], [[0.0, 0.0]], [0],
            FittingStrategy.CIRCULAR, masks=[masks],
        )
        assert pfit.shape == (1, 3 + 2 * 3 + 2)
        assert perr.shape == (1, 3 + 2 * 3)

    def test_fit_puncta_method_high_photon_count_uses_float64(self):
        # amp is integrated intensity, not peak height (model normalises by 1/(2*pi*sigma^2)),
        # so amp must be well above the max_value>50000 threshold's raw peak requirement.
        punctum, masks = _synthetic_punctum(x0=4.0, y0=4.0, amp=700000.0)
        iaf_obj = Image_Analysis_Functions()
        pfit, perr = iaf_obj.fit_puncta_method(
            [punctum], [punctum], [_weights()], [[0.0, 0.0]], [0], FittingStrategy.NOCOLOUR,
        )
        assert pfit.dtype == np.float64

    def test_fit_puncta_method_truncates_oversized_fit_params(self, monkeypatch):
        punctum, masks = _synthetic_punctum(x0=4.0, y0=4.0)
        iaf_obj = Image_Analysis_Functions()
        processor = iaf_obj.processors[FittingStrategy.NOCOLOUR]
        monkeypatch.setattr(
            processor, "fit_single_punctum",
            lambda *a, **kw: (np.arange(20, dtype=float), np.arange(6, dtype=float)),
        )
        pfit, perr = iaf_obj.fit_puncta_method(
            [punctum], [punctum], [_weights()], [[0.0, 0.0]], [0], FittingStrategy.NOCOLOUR,
        )
        assert pfit.shape[1] == FittingConstants.PARAM_DIMENSIONS[FittingStrategy.NOCOLOUR]["fit"]

    def test_fit_puncta_method_per_punctum_exception_leaves_nan(self, monkeypatch):
        punctum, masks = _synthetic_punctum(x0=4.0, y0=4.0)
        iaf_obj = Image_Analysis_Functions()
        processor = iaf_obj.processors[FittingStrategy.NOCOLOUR]

        def _raise(*a, **kw):
            raise RuntimeError("forced per-punctum failure")

        monkeypatch.setattr(processor, "fit_single_punctum", _raise)
        pfit, perr = iaf_obj.fit_puncta_method(
            [punctum], [punctum], [_weights()], [[0.0, 0.0]], [0], FittingStrategy.NOCOLOUR,
        )
        assert np.all(np.isnan(pfit))


class TestFitPunctaMethodStandalone:
    def test_standalone_normal(self):
        punctum, masks = _synthetic_punctum(x0=4.0, y0=4.0)
        pfit, perr = _fit_puncta_method_standalone(
            [punctum], [punctum], [_weights()], [[0.0, 0.0]], [0], FittingStrategy.NOCOLOUR,
        )
        assert pfit.shape[0] == 1

    def test_standalone_exception_returns_nan_colour(self, monkeypatch):
        punctum, masks = _synthetic_punctum(x0=4.0, y0=4.0)

        def _raise_init(*a, **kw):
            raise RuntimeError("forced instantiation failure")

        monkeypatch.setattr(iaf, "Image_Analysis_Functions", _raise_init)
        pfit, perr = _fit_puncta_method_standalone(
            [punctum], [punctum], [_weights()], [[0.0, 0.0]], [0],
            FittingStrategy.STANDARD, masks=[masks],
        )
        assert pfit.shape == (1, 4 + 2 * 3 + 4)
        assert np.all(np.isnan(pfit))

    def test_standalone_exception_returns_nan_circular(self, monkeypatch):
        punctum, masks = _synthetic_punctum(x0=4.0, y0=4.0)

        def _raise_init(*a, **kw):
            raise RuntimeError("forced instantiation failure")

        monkeypatch.setattr(iaf, "Image_Analysis_Functions", _raise_init)
        pfit, perr = _fit_puncta_method_standalone(
            [punctum], [punctum], [_weights()], [[0.0, 0.0]], [0],
            FittingStrategy.CIRCULAR, masks=[masks],
        )
        assert pfit.shape == (1, 3 + 2 * 3 + 2)
        assert np.all(np.isnan(pfit))

    def test_standalone_exception_returns_nan_noncolour(self, monkeypatch):
        punctum, masks = _synthetic_punctum(x0=4.0, y0=4.0)

        def _raise_init(*a, **kw):
            raise RuntimeError("forced instantiation failure")

        monkeypatch.setattr(iaf, "Image_Analysis_Functions", _raise_init)
        pfit, perr = _fit_puncta_method_standalone(
            [punctum], [punctum], [_weights()], [[0.0, 0.0]], [0], FittingStrategy.NOCOLOUR,
        )
        assert pfit.shape == (1, FittingConstants.PARAM_DIMENSIONS[FittingStrategy.NOCOLOUR]["fit"])
        assert np.all(np.isnan(pfit))


class TestImageAnalysisFunctionsParallel:
    def test_fit_puncta_parallel_method_normal(self):
        punctum, masks = _synthetic_punctum(x0=4.0, y0=4.0)
        iaf_obj = Image_Analysis_Functions()
        pfit, perr = iaf_obj.fit_puncta_parallel_method(
            [punctum, punctum], [punctum, punctum], [_weights(), _weights()],
            [[0.0, 0.0], [0.0, 0.0]], [0, 1], FittingStrategy.NOCOLOUR,
        )
        assert pfit.shape[0] == 2

    def test_fit_puncta_parallel_method_asynch_returns_futures(self):
        punctum, masks = _synthetic_punctum(x0=4.0, y0=4.0)
        iaf_obj = Image_Analysis_Functions()
        fs = iaf_obj.fit_puncta_parallel_method(
            [punctum], [punctum], [_weights()], [[0.0, 0.0]], [0],
            FittingStrategy.NOCOLOUR, asynch=True,
        )
        assert len(fs) >= 1
        results = iaf_obj.fits_from_futures(fs, FittingStrategy.NOCOLOUR)
        assert results[0].shape[0] == 1

    def test_fits_from_futures_all_failed_returns_empty(self):
        iaf_obj = Image_Analysis_Functions()

        class _FailingFuture:
            def result(self):
                raise RuntimeError("forced future failure")

        pfit, perr = iaf_obj.fits_from_futures([_FailingFuture()], FittingStrategy.NOCOLOUR)
        assert pfit.shape == (0, FittingConstants.PARAM_DIMENSIONS[FittingStrategy.NOCOLOUR]["fit"])
        assert perr.shape == (0, FittingConstants.PARAM_DIMENSIONS[FittingStrategy.NOCOLOUR]["error"])

    def test_fit_puncta_parallel_method_skips_empty_chunks(self, monkeypatch):
        # calculate_parallel_chunks never naturally emits a 0-count task (n_tasks
        # is capped at total_items), so force one to cover the `continue` branch.
        punctum, masks = _synthetic_punctum(x0=4.0, y0=4.0)
        iaf_obj = Image_Analysis_Functions()
        monkeypatch.setattr(
            iaf_obj.helper, "calculate_parallel_chunks",
            lambda *a, **kw: (1, 2, [0, 1], np.array([0, 0])),
        )
        pfit, perr = iaf_obj.fit_puncta_parallel_method(
            [punctum], [punctum], [_weights()], [[0.0, 0.0]], [0], FittingStrategy.NOCOLOUR,
        )
        assert pfit.shape[0] == 1


class TestFitPunctaMethodStandaloneSysPath:
    def test_inserts_src_dir_when_missing_from_sys_path(self, monkeypatch):
        import sys
        from pathlib import Path

        punctum, masks = _synthetic_punctum(x0=4.0, y0=4.0)
        _dir = str(Path(iaf.__file__).parent)
        pruned_path = [p for p in sys.path if p != _dir]
        monkeypatch.setattr(sys, "path", pruned_path)
        assert _dir not in sys.path

        pfit, perr = _fit_puncta_method_standalone(
            [punctum], [punctum], [_weights()], [[0.0, 0.0]], [0], FittingStrategy.NOCOLOUR,
        )
        assert pfit.shape[0] == 1
        assert _dir in sys.path
