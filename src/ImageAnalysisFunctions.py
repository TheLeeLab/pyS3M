#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Image analysis functions for multicolour single-molecule localization microscopy.

This module provides comprehensive functionality for:
- Gaussian fitting with various colour strategies  
- Parallel processing for large datasets
- Weighted least squares optimisation
- Multiple fitting strategies for different analysis needs

The module uses a strategy pattern for handling different types of fitting approaches
(colour, no-colour, just-colour, raw-colour, post-colour) with unified interfaces.

Created on Tue Dec 10 08:59:38 2024
@author: jbeckwith
jsb92, 2024/01/02 - Refactored August 15, 2025
"""

from typing import Optional, List, Tuple, Union, Dict, Any
from enum import Enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from scipy.optimize import leastsq
from pathlib import Path
import sys
from numba import jit
import multiprocessing
from concurrent import futures
from tqdm import tqdm
import pyS3M.ProgressUtils as ProgressUtils
import logging

# Set up module paths
sys.path.append(str(Path(__file__).parent))
import pyS3M.IOFunctions as IOFunctions
import pyS3M.sCMOSFunctions as sCMOSFunctions
import pyS3M.PSFFunctions as PSFFunctions
import pyS3M.gaussoptfuncs as gaussoptfuncs
import pyS3M.HelperFunctions as HelperFunctions
import pyS3M.FitGate as FitGate


def _lm_stuck(pfit: np.ndarray, x0: np.ndarray) -> bool:
    """True if leastsq handed back its starting point unchanged (no step accepted).

    leastsq still reports success (ier 1-4) when no step reduces chi^2, so such fits used to
    reach the database holding their initial guess -- with the old equal-split amplitude
    guess, A_B = A_G = A_R = 1/3, which analyses filtered by value. Rejecting the
    unmoved fit here works whatever the guess is. Apply it to the first LM call only: a
    later stage that starts from a converged fit may legitimately not move.
    """
    return bool(np.array_equal(np.asarray(pfit), np.asarray(x0)))


class FittingStrategy(Enum):
    """Enumeration for different fitting strategies.

    Defines the various approaches to Gaussian fitting with different
    colour channel handling strategies.
    """

    STANDARD = "standard"  # Full colour fitting with all channels
    STANDARD_ITER = "standard_iter"  # STANDARD with 2 IRLS model-weight iterations
    STANDARD_DATA = "standard_data"  # smooth → model → raw-data weights (unbiased final pass)
    CIRCULAR = "circular"  # STANDARD_DATA with a single shared PSF width (no sx/sy split)
    ELLIPTICAL = "elliptical"  # Rotated elliptical Gaussian (11 params; for tracking)
    NOCOLOUR = "nocolour"  # No colour information, intensity only


class FittingConstants:
    """Constants for fitting operations."""

    # Fitting tolerances
    DEFAULT_FTOL = 1e-2
    # xtol is relative to the norm of the whole parameter vector, which the sqrt-space amplitudes
    # dominate (~150 at 50k photons), so 1e-2 accepted steps of ~1.5 in sigma as "converged". At
    # high photon counts that stopped ~5% of vectorial-PSF fits on a stalled wide branch (sigma
    # +20%, chi^2 4-8x the true minimum) and doubled the sigma/fingerprint scatter of the rest.
    # 1e-4 removes both at no measurable cost; 1e-3 already does, smaller values change nothing.
    DEFAULT_XTOL = 1e-4

    # Amplitude SNR threshold (Wald t-statistic, replaces all three hard gates)
    AMPLITUDE_SNR_THRESHOLD = 2.0  # sigma; z = sum(|q_c|)/sqrt(sum(var(q_c))) >= this

    # Parallel processing limits
    MAX_WORKERS = 60  # Python crashes when using >64 cores
    WORKER_RATIO = 0.9
    TASKS_PER_WORKER = 100

    # Array dimensions for different strategies
    PARAM_DIMENSIONS = {
        FittingStrategy.STANDARD: {"fit": 12, "error": 10},
        FittingStrategy.STANDARD_ITER: {"fit": 12, "error": 10},
        FittingStrategy.STANDARD_DATA: {"fit": 12, "error": 10},
        FittingStrategy.CIRCULAR: {"fit": 11, "error": 9},
        FittingStrategy.ELLIPTICAL: {"fit": 13, "error": 11},
        FittingStrategy.NOCOLOUR: {"fit": 8, "error": 6},
    }


@dataclass
class FittingParameters:
    """Configuration parameters for fitting operations.

    Encapsulates all parameters needed for puncta fitting with validation.

    Attributes:
        puncta: List of 2D arrays containing puncta data.
        smoothed_puncta: List of 2D arrays containing smoothed puncta data.
        masks: Optional list of 3D arrays containing colour masks.
        weights: List of 2D arrays containing fitting weights.
        relative_coords: List of relative coordinate offsets.
        planes: List of plane indices for each punctum.
        strategy: Fitting strategy to use.
    """

    puncta: List[np.ndarray]
    smoothed_puncta: List[np.ndarray]
    weights: List[np.ndarray]
    relative_coords: List[List[float]]
    planes: List[int]
    strategy: FittingStrategy
    masks: Optional[List[np.ndarray]] = None

    def __post_init__(self):
        """Validate fitting parameters after initialization."""
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate that all input parameters are consistent.

        Raises:
            FittingValidationError: If parameters are inconsistent or invalid.
        """
        n_puncta = len(self.puncta)

        # Check array length consistency
        if not all(
            len(arr) == n_puncta
            for arr in [
                self.smoothed_puncta,
                self.weights,
                self.relative_coords,
                self.planes,
            ]
        ):
            raise FittingValidationError(
                "All input arrays must have the same length as puncta"
            )

        # Check masks requirement for colour strategies
        colour_strategies = {
            FittingStrategy.STANDARD,
            FittingStrategy.STANDARD_ITER,
            FittingStrategy.STANDARD_DATA,
            FittingStrategy.CIRCULAR,
            FittingStrategy.ELLIPTICAL,
        }

        if self.strategy in colour_strategies and self.masks is None:
            raise FittingValidationError(
                f"Strategy {self.strategy.value} requires masks parameter"
            )

        if self.masks is not None and len(self.masks) != n_puncta:
            raise FittingValidationError("Masks array must have same length as puncta")


class FittingValidationError(Exception):
    """Custom exception for fitting parameter validation errors."""

    pass


class FittingResultProcessor:
    """Handles processing and validation of fitting results."""

    @staticmethod
    def _sqrt_space_slice(strategy: FittingStrategy, length: int) -> slice:
        """Index range of sqrt-space amplitude/background parameters for a strategy.

        leastsq fits amplitude/background as sqrt(value) to enforce positivity;
        process_fit_results squares this same range to recover the real-space
        values it stores (e.g. A_R, bg_R). calculate_errors needs the identical
        range to delta-method-propagate their errors into those same real-space
        units — this is the single source of truth both use, so the two can't
        drift out of sync.
        """
        standard_like = {
            FittingStrategy.STANDARD, FittingStrategy.STANDARD_ITER, FittingStrategy.STANDARD_DATA,
        }
        if strategy in standard_like:
            n_ch = (length - 4) // 2
            return slice(4, 4 + 2 * n_ch)
        if strategy == FittingStrategy.ELLIPTICAL:
            return slice(5, 11)
        if strategy == FittingStrategy.CIRCULAR:
            n_ch_circ = (length - 3) // 2
            return slice(3, 3 + 2 * n_ch_circ)
        if strategy == FittingStrategy.NOCOLOUR:
            return slice(4, 6) if length >= 6 else slice(0, 0)
        return slice(0, 0)

    @staticmethod
    def calculate_errors(
        pcov: np.ndarray, strategy: FittingStrategy, pfit: Optional[np.ndarray] = None,
    ) -> List[float]:
        """Calculate parameter errors from covariance matrix.

        Args:
            pcov: Parameter covariance matrix from fitting.
            strategy: Fitting strategy to determine expected error array size.
            pfit: Raw (sqrt-space, pre-squaring) fitted parameters, same order
                as pcov's rows/columns. When given, the sqrt-space amplitude/
                background entries (see _sqrt_space_slice) are delta-method
                propagated into real-space errors via sigma_A = 2*|sqrt(A)|*
                sigma_sqrt(A) — matching process_fit_results squaring those
                same parameters for storage. Omitting it returns errors in
                whatever space pcov's diagonal is in (sqrt-space for those
                entries).

        Returns:
            List of parameter errors (standard deviations).
        """
        # Default expected size from strategy table (3-channel baseline)
        default_size = FittingConstants.PARAM_DIMENSIONS[strategy]["error"]

        if pcov is None:
            return [np.nan] * default_size

        # Handle scalar pcov (e.g., np.inf from failed fits)
        if np.isscalar(pcov) or pcov.ndim == 0:
            return [np.nan] * default_size

        # Handle inf values in matrix
        if np.any(np.isinf(pcov)):
            return [np.nan] * default_size

        # Use actual pcov size — handles n_ch != 3 patterns without hardcoding
        expected_size = pcov.shape[0]

        try:
            # Vectorized error calculation - more efficient than loops
            diagonal = np.diag(pcov)
            # Use NaN (not 0) for non-positive diagonals so that downstream filters
            # (xc_err < threshold) reject these fits rather than treating them as
            # perfectly precise.  0.0 was the old value and caused eps=0 in DBSCAN
            # when chi_sqr is very small (bright spots → pcov*chisqr→0).
            errors = np.where(diagonal > 0, np.sqrt(diagonal), np.nan)

            if pfit is not None:
                sqrt_slice = FittingResultProcessor._sqrt_space_slice(strategy, len(pfit))
                stop = min(sqrt_slice.stop, len(errors), len(pfit))
                start = min(sqrt_slice.start, stop)
                fixed = slice(start, stop)
                errors[fixed] = errors[fixed] * (2.0 * np.abs(pfit[fixed]))

            error_list = errors.tolist()

            # Pad if pcov was undersized (degenerate fit)
            if len(error_list) < expected_size:
                error_list.extend([np.nan] * (expected_size - len(error_list)))

            return error_list
        except (IndexError, ValueError):
            return [np.nan] * expected_size

    @staticmethod
    def fraction_errors(
        pcov: np.ndarray, strategy: FittingStrategy, pfit: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Errors of the normalised amplitude and background fractions, from the full covariance.

        The fitted parameters are q_k = sqrt(A_k), so with S = sum_k A_k the fraction
        f_c = A_c / S has dF_c/dq_k = 2 q_k (delta_ck - f_c) / S, and
        sigma_f = sqrt(diag(J C J^T)) with C the (chi^2-scaled) covariance of the q block.
        Amplitudes and backgrounds are each normalised over their own channels. Unlike
        sigma_A / S, this includes the uncertainty of S and every covariance between
        channels.

        Args:
            pcov: Parameter covariance matrix (already chi_sqr-scaled), in pfit's order.
            strategy: Fitting strategy (sets where the sqrt-space parameters are).
            pfit: Raw (sqrt-space, pre-squaring) fitted parameters.

        Returns:
            The sqrt-space slice's errors (backgrounds then amplitudes, in pfit's order) as
            fraction errors, or None where there is nothing to normalise (fewer than two
            channels, no usable covariance).
        """
        if pcov is None or np.isscalar(pcov) or np.ndim(pcov) != 2 or not np.all(np.isfinite(pcov)):
            return None
        sl = FittingResultProcessor._sqrt_space_slice(strategy, len(pfit))
        n_ch = (sl.stop - sl.start) // 2
        if n_ch < 2 or sl.stop > pcov.shape[0]:
            return None
        out = np.full(2 * n_ch, np.nan)
        for half in range(2):   # backgrounds, then amplitudes
            idx = np.arange(sl.start + half * n_ch, sl.start + (half + 1) * n_ch)
            q = np.asarray(pfit, dtype=np.float64)[idx]
            a = np.square(q)
            total = a.sum()
            if not np.isfinite(total) or total <= 0:
                continue
            f = a / total
            jac = (np.eye(n_ch) - f[:, None]) * (2.0 * q[None, :] / total)
            var = np.einsum("ij,jk,ik->i", jac, pcov[np.ix_(idx, idx)], jac)
            out[half * n_ch:(half + 1) * n_ch] = np.where(var > 0, np.sqrt(np.maximum(var, 0.0)), np.nan)
        return out

    @staticmethod
    def calculate_reduced_chisquared(
        residuals: np.ndarray, n_data_points: int, n_parameters: int
    ) -> float:
        """Calculate reduced chi-squared from residuals.

        Args:
            residuals: Array of residuals from chi function
            n_data_points: Number of data points used in fit
            n_parameters: Number of fitted parameters

        Returns:
            Reduced chi-squared value

        Notes:
            Reduced chi-squared = sum(residuals^2) / (n_data - n_params)
            This is the standard formula for goodness-of-fit.
        """
        return np.sum(np.square(residuals)) / (n_data_points - n_parameters)

    @staticmethod
    def process_covariance(
        pcov: np.ndarray,
        chisqr: float,
        n_data_points: int,
        n_parameters: int,
    ) -> np.ndarray:
        """Process covariance matrix by scaling with chi-squared.

        Args:
            pcov: Raw covariance matrix from curve_fit
            chisqr: Chi-squared value
            n_data_points: Number of data points used in fit
            n_parameters: Number of fitted parameters

        Returns:
            Processed covariance matrix (either scaled by chi-squared or np.inf)

        Notes:
            The covariance matrix should be scaled by the chi-squared value
            to account for goodness of fit. If there aren't enough degrees of
            freedom or pcov is None, returns np.inf to signal invalid fit.
        """
        if (n_data_points > n_parameters) and pcov is not None:
            return pcov * chisqr
        else:
            return np.inf

    @staticmethod
    def _compute_amplitude_snr(pfit: np.ndarray, pcov: np.ndarray) -> float:
        """Wald amplitude SNR for all amplitude parameters (sqrt-space), n-channel adaptive.

        pfit layout: [x, y, sy, sx, bg_0,...,bg_{n-1}, A_0,...,A_{n-1}]
        Amplitude indices start at 4 + n_ch where n_ch = (len(pfit) - 4) // 2.

        pcov must already be chi_sqr-scaled (as returned by process_covariance).
        High chi_sqr inflates pcov → reduces z, making the statistic automatically
        conservative for poor fits without a separate chi_sqr gate.

        Returns 0.0 when pcov is unavailable (degenerate fit → reject).
        """
        if not isinstance(pcov, np.ndarray):
            return 0.0
        n_ch = (len(pfit) - 4) // 2
        amp_start = 4 + n_ch
        variances = np.diag(pcov)[amp_start:amp_start + n_ch]
        if np.any(variances <= 0):
            return 0.0
        return float(np.sum(np.abs(pfit[amp_start:amp_start + n_ch])) / np.sqrt(np.sum(variances)))

    @staticmethod
    def _compute_amplitude_snr_elliptical(pfit: np.ndarray, pcov: np.ndarray) -> float:
        """Wald amplitude SNR for the elliptical model (amplitudes at indices 8–10).

        Same logic as _compute_amplitude_snr but offset by 1 due to the extra
        theta parameter at index 4 in the 11-parameter elliptical vector.
        """
        if not isinstance(pcov, np.ndarray):
            return 0.0
        variances = np.diag(pcov)[8:11]
        if np.any(variances <= 0):
            return 0.0
        return float(np.sum(np.abs(pfit[8:11])) / np.sqrt(np.sum(variances)))

    @staticmethod
    def _compute_amplitude_snr_circular(pfit: np.ndarray, pcov: np.ndarray) -> float:
        """Wald amplitude SNR for the circular model, n-channel adaptive.

        pfit layout: [x, y, s, bg_0,...,bg_{n-1}, A_0,...,A_{n-1}]
        Amplitude indices start at 3 + n_ch (one earlier than _compute_amplitude_snr's
        4 + n_ch, since there is only one width parameter instead of sy/sx).
        """
        if not isinstance(pcov, np.ndarray):
            return 0.0
        n_ch = (len(pfit) - 3) // 2
        amp_start = 3 + n_ch
        variances = np.diag(pcov)[amp_start:amp_start + n_ch]
        if np.any(variances <= 0):
            return 0.0
        return float(np.sum(np.abs(pfit[amp_start:amp_start + n_ch])) / np.sqrt(np.sum(variances)))

    @staticmethod
    def delta_chi_squared(
        residuals: np.ndarray, data: np.ndarray, masks: np.ndarray, weights: np.ndarray
    ) -> float:
        """Likelihood-ratio statistic: weighted chi^2 of a background-only model minus that
        of the fit, under the fit's own final weights.

        The background-only model is one constant per colour channel, constrained >= 0 like
        the fit's backgrounds (so the two models are nested); its best value per channel is
        the weighted mean of that channel's pixels.

        Args:
            residuals: The fit's chi vector, sqrt(w) * (data - model).
            data: Punctum data (photoelectrons).
            masks: (H, W, n_channels) boolean colour masks.
            weights: The fit's final per-pixel weights.

        Returns:
            float: chi^2(background only) - chi^2(fit). Larger = stronger evidence of an emitter.
        """
        d = np.asarray(data, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64)
        chi2_fit = float(np.sum(np.square(np.asarray(residuals, dtype=np.float64))))
        chi2_bg = 0.0
        for c in range(masks.shape[-1]):
            sel = np.asarray(masks[:, :, c], dtype=bool)
            if not sel.any():
                continue
            ws, ds = w[sel], d[sel]
            mu = max(float(np.sum(ws * ds) / np.sum(ws)), 0.0)
            chi2_bg += float(np.sum(ws * np.square(ds - mu)))
        return chi2_bg - chi2_fit

    @staticmethod
    def _xy_permutation(strategy: FittingStrategy, length: int) -> np.ndarray:
        """Index order taking a model's [row, col, ...] parameters to the output's [x, y, ...].

        STANDARD-like and NOCOLOUR: [row, col, s_row, s_col, ...] -> [col, row, s_col, s_row, ...];
        ELLIPTICAL: the same, theta stays at index 4; CIRCULAR: [row, col, s, ...] ->
        [col, row, s, ...].
        """
        perm = np.arange(length)
        if length < 2:
            return perm
        if strategy in (
            FittingStrategy.STANDARD, FittingStrategy.STANDARD_ITER, FittingStrategy.STANDARD_DATA,
            FittingStrategy.ELLIPTICAL, FittingStrategy.NOCOLOUR,
        ) and length >= 4:
            perm[:4] = [1, 0, 3, 2]
        elif strategy == FittingStrategy.CIRCULAR:
            perm[:2] = [1, 0]
        return perm

    @staticmethod
    def process_fit_results(
        pfit: np.ndarray,
        pcov: np.ndarray,
        size: int,
        relative_coords: List[float],
        strategy: FittingStrategy,
        chisqr: float = 1.0,
        delta_chi2: Optional[float] = None,
        gate: Any = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Process raw fitting results into standardized format.

        Args:
            pfit: Raw fitting parameters.
            pcov: Parameter covariance matrix (already chi_sqr-scaled).
            relative_coords: Coordinate offsets to add.
            size: Image size in fitting.
            strategy: Fitting strategy used.
            chisqr: Chi-squared value from fitting (default 1.0).
            delta_chi2: Drop in weighted chi^2 from a background-only model to this fit
                (see delta_chi_squared), used by FitGate.DeltaChi2Gate.
            gate: Fit-acceptance gate (FitGate): a fit it rejects comes back as NaN.
                None keeps every fit that passes the position/width check.

        Returns:
            Tuple of (processed_parameters, parameter_errors).
        """
        if pfit is None or not hasattr(pfit, "__len__") or len(pfit) == 0:
            dims = FittingConstants.PARAM_DIMENSIONS[strategy]
            return (np.full(dims["fit"], np.nan), np.full(dims["error"], np.nan))

        # Chi-squared is now passed as parameter from the fitting processor

        # Axis convention (see Constants.ResultColumns): xc is the image column and yc the row,
        # in full-frame pixels, so xc runs along the width and yc along the height. The models in
        # gaussoptfuncs work on the ROI array instead, in [row, col] order: pfit[0] is the row
        # centre and pfit[1] the column centre, then the row width before the column width.
        # _xy_permutation reorders pfit into the output's [x, y, ...] order; the errors (in pcov's
        # order, i.e. pfit's) take the same permutation below, and relative_coords is the ROI's
        # (column, row) origin, added to (xc, yc).
        _standard_like = {FittingStrategy.STANDARD, FittingStrategy.STANDARD_ITER, FittingStrategy.STANDARD_DATA}
        _gated_strategies = _standard_like | {FittingStrategy.ELLIPTICAL, FittingStrategy.CIRCULAR}

        n_ch = (len(pfit) - 4) // 2   # standard-like strategies only
        perm = FittingResultProcessor._xy_permutation(strategy, len(pfit))
        pfit_processed = np.asarray(pfit, dtype=np.float64)[perm]
        if strategy == FittingStrategy.ELLIPTICAL:
            # theta rotates the model's (row, col) frame; in the output's (x, y) frame (y down the
            # image) the s_x axis lies at -theta from +x towards +y.
            pfit_processed[4] = -pfit_processed[4]

        # Position gate only applies to strategies where first params are x, y, sx, sy
        position_strategies = {
            FittingStrategy.STANDARD, FittingStrategy.STANDARD_ITER, FittingStrategy.STANDARD_DATA,
            FittingStrategy.ELLIPTICAL,
            FittingStrategy.NOCOLOUR,
        }
        if strategy in position_strategies:
            if np.any(pfit_processed[:4] < 0) | np.any(pfit_processed[:4] > size):
                return (
                    np.full(len(pfit_processed), np.nan),
                    np.full(len(pfit_processed), np.nan),
                )

            # Add relative coordinates to position parameters (first two elements)
            if (
                relative_coords is not None
                and hasattr(relative_coords, "__len__")
                and len(relative_coords) >= 2
            ):
                pfit_processed[:2] += relative_coords[:2]
        elif strategy == FittingStrategy.CIRCULAR:
            # Only x, y, s lead the vector (one fewer than the sx/sy strategies above),
            # so this needs its own [:3] gate rather than reusing position_strategies' [:4].
            if np.any(pfit_processed[:3] < 0) | np.any(pfit_processed[:3] > size):
                return (
                    np.full(len(pfit_processed), np.nan),
                    np.full(len(pfit_processed), np.nan),
                )
            if (
                relative_coords is not None
                and hasattr(relative_coords, "__len__")
                and len(relative_coords) >= 2
            ):
                pfit_processed[:2] += relative_coords[:2]

        # Stage 2: fit gate -- is this a real emitter, or noise? (see FitGate). Runs on the
        # raw sqrt-space pfit, before squaring.
        if gate is not None and strategy in _gated_strategies:
            if strategy == FittingStrategy.ELLIPTICAL:
                n_ch_gate = (len(pfit) - 5) // 2
            elif strategy == FittingStrategy.CIRCULAR:
                n_ch_gate = (len(pfit) - 3) // 2
            else:
                n_ch_gate = (len(pfit) - 4) // 2
            background = FitGate.fitted_background(pfit, strategy, n_ch_gate)
            if not gate.accept(pfit, pcov, strategy, delta_chi2=delta_chi2, background=background):
                return (
                    np.full(len(pfit_processed), np.nan),
                    np.full(len(pfit_processed), np.nan),
                )

        # Square amplitude and background parameters for storage
        # leastsq returns optimised square-root values, but we store squared values as photon counts
        if strategy in _standard_like:
            pfit_processed[4:4 + 2 * n_ch] = np.square(pfit_processed[4:4 + 2 * n_ch])
        elif strategy == FittingStrategy.ELLIPTICAL:
            # theta is at index 4 — do NOT square it; bg/A are at indices 5:11
            pfit_processed[5:11] = np.square(pfit_processed[5:11])
        elif strategy == FittingStrategy.CIRCULAR:
            # x, y, s lead (index 0:3); bg/A start one index earlier than _standard_like
            n_ch_circ = (len(pfit_processed) - 3) // 2
            pfit_processed[3:3 + 2 * n_ch_circ] = np.square(pfit_processed[3:3 + 2 * n_ch_circ])
        elif strategy == FittingStrategy.NOCOLOUR:
            if len(pfit_processed) >= 6:
                pfit_processed[4:6] = np.square(pfit_processed[4:6])

        # Append chi-squared
        pfit_final = np.append(pfit_processed, chisqr)

        # Calculate errors — pfit (raw, sqrt-space, pre-squaring) is required so
        # sqrt-space amplitude/background errors get delta-method propagated into
        # the same real-space units as the squared values stored above.
        errors = np.asarray(FittingResultProcessor.calculate_errors(pcov, strategy, pfit))
        # Colour and background errors are those of the normalised fractions (A_c / sum A,
        # bg_c / sum bg), from the full covariance: what A_*_err and bg_*_err report
        # downstream, where only the values are divided by their totals.
        frac = FittingResultProcessor.fraction_errors(pcov, strategy, pfit)
        if frac is not None:
            sl = FittingResultProcessor._sqrt_space_slice(strategy, len(pfit))
            errors = errors.astype(np.float64)
            errors[sl] = frac
        if len(errors) == len(perm):
            errors = errors[perm]

        return pfit_final, errors


class FittingProcessor(ABC):
    """Abstract base class for fitting strategy processors."""

    @abstractmethod
    def fit_single_punctum(
        self,
        punctum: np.ndarray,
        smoothed_punctum: np.ndarray,
        weights: np.ndarray,
        relative_coords: List[float],
        masks: Optional[np.ndarray] = None,
        readnoise_map: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit a single punctum with this strategy.

        Args:
            punctum: 2D array containing punctum data.
            smoothed_punctum: 2D array containing smoothed punctum data.
            weights: 2D array containing fitting weights.
            relative_coords: Relative coordinate offsets.
            masks: Optional 3D array containing colour masks.
            readnoise_map: Optional 2D array, the punctum's own per-pixel read noise (e-),
                cropped from the camera's read-noise map. Strategies that re-weight
                between passes use it per pixel; without it they use the processor's
                scalar ``readnoise``.

        Returns:
            Tuple of (fit_parameters, parameter_errors).
        """
        pass

    readnoise: float = 1.5  # scalar fallback (e-); processors that re-weight set their own
    gate: Any = None        # fit-acceptance gate (FitGate); set by Image_Analysis_Functions

    def _readnoise_sq(self, readnoise_map: Optional[np.ndarray]) -> Union[np.ndarray, np.float32]:
        """Per-pixel read-noise variance (e-^2) for weighting: the punctum's own map when
        given (a CMOS camera's read noise varies per pixel), else the scalar read noise."""
        if readnoise_map is not None:
            return np.square(np.asarray(readnoise_map, dtype=np.float32))
        return np.float32(float(self.readnoise) ** 2)


class StandardFittingProcessor(FittingProcessor):
    """Processor for standard colour fitting strategy."""

    def fit_single_punctum(
        self,
        punctum: np.ndarray,
        smoothed_punctum: np.ndarray,
        weights: np.ndarray,
        relative_coords: List[float],
        masks: Optional[np.ndarray] = None,
        readnoise_map: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit single punctum using standard colour fitting.

        Uses weighted least squares with colour channel information.

        Args:
            punctum: 2D array containing punctum data.
            smoothed_punctum: 2D array containing smoothed punctum data.
            weights: 2D array containing fitting weights.
            relative_coords: Relative coordinate offsets.
            masks: 3D array containing colour masks (required).

        Returns:
            Tuple of (fit_parameters, parameter_errors).

        Raises:
            FittingValidationError: If masks are not provided.
        """
        if masks is None:
            raise FittingValidationError("Standard fitting requires masks")

        # Stage 1: fast pre-filter — skip leastsq on entirely non-positive ROIs
        if np.max(smoothed_punctum) <= 0:
            n_ch = masks.shape[-1] if masks is not None else 3
            return (np.full(4 + 2 * n_ch + 2, np.nan), np.full(4 + 2 * n_ch, np.nan))

        # Get initial guess from smoothed and raw data
        initial_guess = self._generate_initial_guess(smoothed_punctum, punctum, masks)

        # Perform weighted least squares fit
        return self._perform_wls_fit(
            punctum, initial_guess, masks, weights, relative_coords,
        )

    def _generate_initial_guess(
        self, smoothed_punctum: np.ndarray, raw_punctum: np.ndarray, masks: np.ndarray
    ) -> np.ndarray:
        """Generate initial parameter guess for fitting using gaussoptfuncs.

        Args:
            smoothed_punctum: Smoothed punctum data.
            raw_punctum: Raw punctum data.
            masks: Colour masks.

        Returns:
            Initial parameter guess array.
        """
        # Use the proper initial_guess function from gaussoptfuncs
        # Returns: [x, y, sy, sx, bg_B, bg_G, bg_R, A_B, A_G, A_R]
        return gaussoptfuncs.initial_guess(smoothed_punctum, raw_punctum, masks)

    def _perform_wls_fit(
        self,
        data: np.ndarray,
        initial_guess: np.ndarray,
        masks: np.ndarray,
        weights: np.ndarray,
        relative_coords: List[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform weighted least squares fitting.

        Args:
            data: Data to fit.
            initial_guess: Initial parameter guess.
            masks: Colour masks.
            weights: Fitting weights.
            relative_coords: Coordinate offsets.

        Returns:
            Tuple of (fit_parameters, parameter_errors).
        """
        size = int(data.shape[0])
        ravelsize = int(np.prod(data.shape))

        try:
            pfit, pcov, infodict, errmsg, success = leastsq(
                gaussoptfuncs.WLS_chi_nobounds,
                x0=initial_guess,
                args=(data, masks, weights, size, ravelsize),
                full_output=True,
                ftol=FittingConstants.DEFAULT_FTOL,
                xtol=FittingConstants.DEFAULT_XTOL,
            )

            if success not in np.array([1, 2, 3, 4]) or _lm_stuck(pfit, initial_guess):
                n_ch = (len(initial_guess) - 4) // 2
                return (np.full(4 + 2 * n_ch + 2, np.nan), np.full(4 + 2 * n_ch, np.nan))

            # Calculate chi-squared
            residuals = gaussoptfuncs.WLS_chi_nobounds(
                pfit, data, masks, weights, size, ravelsize
            )
            chisqr = FittingResultProcessor.calculate_reduced_chisquared(
                residuals, len(data.ravel()), len(initial_guess)
            )

            # Process covariance matrix
            pcov = FittingResultProcessor.process_covariance(
                pcov, chisqr, len(data.ravel()), len(initial_guess)
            )

            delta_chi2 = FittingResultProcessor.delta_chi_squared(residuals, data, masks, weights)
            return FittingResultProcessor.process_fit_results(
                pfit, pcov, size, relative_coords, FittingStrategy.STANDARD, chisqr,
                delta_chi2=delta_chi2, gate=self.gate,
            )

        except Exception as e:
            import traceback

            logging.warning(f"Fitting failed: {e}")
            logging.warning(
                f"Full traceback:\n{''.join(traceback.format_tb(e.__traceback__))}"
            )
            logging.warning(
                f"Data shapes - data: {data.shape}, weights: {weights.shape}, masks: {masks.shape}"
            )
            logging.warning(f"Initial guess: {initial_guess}")
            logging.warning(
                f"Data dtype: {data.dtype}, min: {data.min():.2f}, max: {data.max():.2f}"
            )
            n_ch = (len(initial_guess) - 4) // 2
            return (np.full(4 + 2 * n_ch + 2, np.nan), np.full(4 + 2 * n_ch, np.nan))


class StandardIterFittingProcessor(StandardFittingProcessor):
    """STANDARD fitting with two IRLS model-weight update iterations.

    Workflow:
    - Stage 1: Fit with smoothing-based weights (same as STANDARD — basin finding).
    - Stage 2: Recompute weights from Stage 1 model, refit from warm start.
    - Stage 3: Recompute weights from Stage 2 model, refit from warm start.

    The model-based weight formula matches production IOFunctions.generate_weights:
        w = 1 / (max(model_pe, 0) + 1 + readnoise_e²)

    The smoothing-based Stage 1 weights over-estimate variance at PSF flank pixels
    (Gaussian smoothing spreads signal outward), causing chi² < 1 at high photon
    counts and a systematic upward bias in fitted sigma.  Replacing them with
    model-derived weights removes this bias and converges chi² closer to 1.

    Args:
        readnoise: Camera read noise in electrons (default 1.5 e-).
    """

    def __init__(self, readnoise: float = 1.5):
        self.readnoise = readnoise

    def _model_based_weights(
        self, pfit: np.ndarray, masks: np.ndarray, size: int, rn2=None
    ) -> np.ndarray:
        """Compute per-pixel weights from the current model estimate."""
        x_arr = np.arange(size, dtype=np.float32)
        buf = np.zeros((size, size), dtype=np.float32)
        model = gaussoptfuncs.WLS_model_nobounds(
            pfit.astype(np.float32), masks, x_arr, buf
        )
        e = np.maximum(model, 0).astype(np.float32) + 1.0 + (self._readnoise_sq(None) if rn2 is None else rn2)
        return (1.0 / e).astype(np.float32)

    def _leastsq_step(
        self,
        x0: np.ndarray,
        data: np.ndarray,
        masks: np.ndarray,
        weights: np.ndarray,
        size: int,
    ):
        """Single Levenberg-Marquardt step; returns (pfit, pcov, ok)."""
        pfit, pcov, _, _, ok = leastsq(
            gaussoptfuncs.WLS_chi_nobounds,
            x0=x0,
            args=(data, masks, weights, size, size * size),
            full_output=True,
            ftol=FittingConstants.DEFAULT_FTOL,
            xtol=FittingConstants.DEFAULT_XTOL,
        )
        return pfit, pcov, ok

    def fit_single_punctum(
        self,
        punctum: np.ndarray,
        smoothed_punctum: np.ndarray,
        weights: np.ndarray,
        relative_coords,
        masks: Optional[np.ndarray] = None,
        readnoise_map: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if masks is None:
            raise FittingValidationError("Standard-ITER fitting requires masks")

        n_ch = masks.shape[-1]
        nan_result = (np.full(4 + 2 * n_ch + 2, np.nan), np.full(4 + 2 * n_ch, np.nan))

        if np.max(smoothed_punctum) <= 0:
            return nan_result

        size = int(punctum.shape[0])
        rn2 = self._readnoise_sq(readnoise_map)
        ravelsize = size * size

        try:
            ig = self._generate_initial_guess(smoothed_punctum, punctum, masks)

            # Stage 1: smoothing weights (passed in from SR_Functions / simulation)
            pfit1, _, ok1 = self._leastsq_step(ig, punctum, masks, weights, size)
            if ok1 not in (1, 2, 3, 4) or _lm_stuck(pfit1, ig):
                return nan_result

            # Stage 2: model weights from Stage 1 fit
            w2 = self._model_based_weights(pfit1, masks, size, rn2)
            pfit2, _, ok2 = self._leastsq_step(pfit1, punctum, masks, w2, size)
            if ok2 not in (1, 2, 3, 4):
                return nan_result

            # Stage 3: model weights from Stage 2 fit (final)
            w3 = self._model_based_weights(pfit2, masks, size, rn2)
            pfit3, pcov3, ok3 = self._leastsq_step(pfit2, punctum, masks, w3, size)
            if ok3 not in (1, 2, 3, 4):
                return nan_result

            # Chi² and covariance from Stage 3 weights
            residuals = gaussoptfuncs.WLS_chi_nobounds(
                pfit3.astype(np.float32), punctum, masks, w3, size, ravelsize
            )
            chisqr = FittingResultProcessor.calculate_reduced_chisquared(
                residuals, ravelsize, len(ig)
            )
            pcov3 = FittingResultProcessor.process_covariance(
                pcov3, chisqr, ravelsize, len(ig)
            )

            delta_chi2 = FittingResultProcessor.delta_chi_squared(residuals, punctum, masks, w3)
            return FittingResultProcessor.process_fit_results(
                pfit3, pcov3, size, relative_coords, FittingStrategy.STANDARD_ITER, chisqr,
                delta_chi2=delta_chi2, gate=self.gate,
            )

        except Exception as e:
            import traceback
            logging.warning(f"STANDARD_ITER fitting failed: {e}")
            logging.warning(
                f"Full traceback:\n{''.join(traceback.format_tb(e.__traceback__))}"
            )
            return nan_result


class StandardDataFittingProcessor(StandardIterFittingProcessor):
    """STANDARD fitting with raw-data weights in the final pass (S4 strategy).

    Workflow:
    - Stage 1: Fit with smoothing-based weights (basin finding).
    - Stage 2: Recompute weights from Stage 1 model, refit (warm start).
    - Stage 3: Recompute weights from raw observed data, refit (warm start).

    Stage 3 weight formula:
        w = 1 / (max(data_pe, 0) + 1 + readnoise²)

    Because E[1/(data + 1 + rn²)] ≈ 1/(true_signal + 1 + rn²), the weight
    denominator is decoupled from the current amplitude estimate, breaking the
    double-inflation coupling that biases STANDARD_ITER amplitudes.

    Args:
        readnoise: Camera read noise in electrons (default 1.5 e-).
    """

    def _raw_data_weights(self, punctum: np.ndarray, rn2=None) -> np.ndarray:
        """Compute per-pixel weights from the observed raw data.

        Args:
            punctum: Raw photoelectron ROI, shape (H, W), float32.

        Returns:
            float32 weight array, shape (H, W).
        """
        e = np.maximum(punctum, 0).astype(np.float32) + 1.0 + (self._readnoise_sq(None) if rn2 is None else rn2)
        return (1.0 / e).astype(np.float32)

    def fit_single_punctum(
        self,
        punctum: np.ndarray,
        smoothed_punctum: np.ndarray,
        weights: np.ndarray,
        relative_coords,
        masks: Optional[np.ndarray] = None,
        readnoise_map: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if masks is None:
            raise FittingValidationError("Standard-DATA fitting requires masks")

        n_ch = masks.shape[-1]
        nan_result = (np.full(4 + 2 * n_ch + 2, np.nan), np.full(4 + 2 * n_ch, np.nan))

        if np.max(smoothed_punctum) <= 0:
            return nan_result

        size = int(punctum.shape[0])
        rn2 = self._readnoise_sq(readnoise_map)
        ravelsize = size * size

        try:
            ig = self._generate_initial_guess(smoothed_punctum, punctum, masks)

            # Stage 1: smoothing weights (passed in from SR_Functions / simulation)
            pfit1, _, ok1 = self._leastsq_step(ig, punctum, masks, weights, size)
            if ok1 not in (1, 2, 3, 4) or _lm_stuck(pfit1, ig):
                return nan_result

            # Stage 2: model weights from Stage 1 fit
            w2 = self._model_based_weights(pfit1, masks, size, rn2)
            pfit2, _, ok2 = self._leastsq_step(pfit1, punctum, masks, w2, size)
            if ok2 not in (1, 2, 3, 4):
                return nan_result

            # Stage 3: raw-data weights (unbiased final pass)
            w3 = self._raw_data_weights(punctum, rn2)
            pfit3, pcov3, ok3 = self._leastsq_step(pfit2, punctum, masks, w3, size)
            if ok3 not in (1, 2, 3, 4):
                return nan_result

            # Chi² and covariance from Stage 3 weights
            residuals = gaussoptfuncs.WLS_chi_nobounds(
                pfit3.astype(np.float32), punctum, masks, w3, size, ravelsize
            )
            chisqr = FittingResultProcessor.calculate_reduced_chisquared(
                residuals, ravelsize, len(ig)
            )
            pcov3 = FittingResultProcessor.process_covariance(
                pcov3, chisqr, ravelsize, len(ig)
            )

            delta_chi2 = FittingResultProcessor.delta_chi_squared(residuals, punctum, masks, w3)
            return FittingResultProcessor.process_fit_results(
                pfit3, pcov3, size, relative_coords, FittingStrategy.STANDARD_DATA, chisqr,
                delta_chi2=delta_chi2, gate=self.gate,
            )

        except Exception as e:
            import traceback
            logging.warning(f"STANDARD_DATA fitting failed: {e}")
            logging.warning(
                f"Full traceback:\n{''.join(traceback.format_tb(e.__traceback__))}"
            )
            return nan_result


class CircularFittingProcessor(StandardDataFittingProcessor):
    """STANDARD_DATA fitting constrained to a single, shared PSF width.

    Identical three-stage algorithm (smooth -> model -> raw-data weights) and the same
    underlying coloured-Gaussian model as StandardDataFittingProcessor -- the only
    difference is that the optimiser sees one width parameter ``s`` instead of
    independent ``s_y``/``s_x``, which get duplicated from ``s`` before every model
    evaluation (WLS_chi_circular_nobounds / gaussoptfuncs.WLS_model_nobounds). Useful for
    testing whether letting sigma_x/sigma_y float independently is absorbing a small,
    real optical anisotropy into the width fit rather than leaving it as position error --
    if the PSF is genuinely close to circular, constraining it here removes two degrees of
    freedom the optimiser doesn't need, at the cost of not being able to represent any real
    ellipticity at all.

    Args:
        readnoise: Camera read noise in electrons (default 1.5 e-).
    """

    def _generate_initial_guess(
        self, smoothed_punctum: np.ndarray, raw_punctum: np.ndarray, masks: np.ndarray
    ) -> np.ndarray:
        """[x, y, sy, sx, bg..., A...] from gaussoptfuncs.initial_guess, collapsed to
        [x, y, s, bg..., A...] by averaging the sy/sx guess into one shared width.
        """
        ig = gaussoptfuncs.initial_guess(smoothed_punctum, raw_punctum, masks)
        s_mean = 0.5 * (ig[2] + ig[3])
        return np.concatenate([[ig[0], ig[1], s_mean], ig[4:]])

    def _model_based_weights(
        self, pfit: np.ndarray, masks: np.ndarray, size: int, rn2=None
    ) -> np.ndarray:
        """Same as StandardIterFittingProcessor._model_based_weights, but pfit carries
        one shared width -- expand to [x, y, s, s, bg..., A...] before evaluating the
        (unchanged) coloured-Gaussian model.
        """
        full_pfit = np.concatenate([pfit[:2], [pfit[2]], pfit[2:]]).astype(np.float32)
        x_arr = np.arange(size, dtype=np.float32)
        buf = np.zeros((size, size), dtype=np.float32)
        model = gaussoptfuncs.WLS_model_nobounds(full_pfit, masks, x_arr, buf)
        e = np.maximum(model, 0).astype(np.float32) + 1.0 + (self._readnoise_sq(None) if rn2 is None else rn2)
        return (1.0 / e).astype(np.float32)

    def _leastsq_step(
        self,
        x0: np.ndarray,
        data: np.ndarray,
        masks: np.ndarray,
        weights: np.ndarray,
        size: int,
    ):
        """Single Levenberg-Marquardt step against the circular chi function."""
        pfit, pcov, _, _, ok = leastsq(
            gaussoptfuncs.WLS_chi_circular_nobounds,
            x0=x0,
            args=(data, masks, weights, size, size * size),
            full_output=True,
            ftol=FittingConstants.DEFAULT_FTOL,
            xtol=FittingConstants.DEFAULT_XTOL,
        )
        return pfit, pcov, ok

    def fit_single_punctum(
        self,
        punctum: np.ndarray,
        smoothed_punctum: np.ndarray,
        weights: np.ndarray,
        relative_coords,
        masks: Optional[np.ndarray] = None,
        readnoise_map: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if masks is None:
            raise FittingValidationError("Circular fitting requires masks")

        n_ch = masks.shape[-1]
        nan_result = (np.full(3 + 2 * n_ch + 2, np.nan), np.full(3 + 2 * n_ch, np.nan))

        if np.max(smoothed_punctum) <= 0:
            return nan_result

        size = int(punctum.shape[0])
        rn2 = self._readnoise_sq(readnoise_map)
        ravelsize = size * size

        try:
            ig = self._generate_initial_guess(smoothed_punctum, punctum, masks)

            # Stage 1: smoothing weights (passed in from SR_Functions / simulation)
            pfit1, _, ok1 = self._leastsq_step(ig, punctum, masks, weights, size)
            if ok1 not in (1, 2, 3, 4) or _lm_stuck(pfit1, ig):
                return nan_result

            # Stage 2: model weights from Stage 1 fit
            w2 = self._model_based_weights(pfit1, masks, size, rn2)
            pfit2, _, ok2 = self._leastsq_step(pfit1, punctum, masks, w2, size)
            if ok2 not in (1, 2, 3, 4):
                return nan_result

            # Stage 3: raw-data weights (unbiased final pass)
            w3 = self._raw_data_weights(punctum, rn2)
            pfit3, pcov3, ok3 = self._leastsq_step(pfit2, punctum, masks, w3, size)
            if ok3 not in (1, 2, 3, 4):
                return nan_result

            # Chi² and covariance from Stage 3 weights
            residuals = gaussoptfuncs.WLS_chi_circular_nobounds(
                pfit3.astype(np.float32), punctum, masks, w3, size, ravelsize
            )
            chisqr = FittingResultProcessor.calculate_reduced_chisquared(
                residuals, ravelsize, len(ig)
            )
            pcov3 = FittingResultProcessor.process_covariance(
                pcov3, chisqr, ravelsize, len(ig)
            )

            delta_chi2 = FittingResultProcessor.delta_chi_squared(residuals, punctum, masks, w3)
            return FittingResultProcessor.process_fit_results(
                pfit3, pcov3, size, relative_coords, FittingStrategy.CIRCULAR, chisqr,
                delta_chi2=delta_chi2, gate=self.gate,
            )

        except Exception as e:
            import traceback
            logging.warning(f"CIRCULAR fitting failed: {e}")
            logging.warning(
                f"Full traceback:\n{''.join(traceback.format_tb(e.__traceback__))}"
            )
            return nan_result


class EllipticalFittingProcessor(FittingProcessor):
    """Processor for rotated elliptical Gaussian fitting.

    Uses an 11-parameter model:
        [x0, y0, sigma_x, sigma_y, theta,
         sqrt(bg_B), sqrt(bg_G), sqrt(bg_R),
         sqrt(A_B),  sqrt(A_G),  sqrt(A_R)]

    Designed for tracking data where motion blur elongates the PSF.
    The rotation angle theta (radians) is estimated from image second moments
    and freely optimised by the fitter.
    """

    def fit_single_punctum(
        self,
        punctum: np.ndarray,
        smoothed_punctum: np.ndarray,
        weights: np.ndarray,
        relative_coords: List[float],
        masks: Optional[np.ndarray] = None,
        readnoise_map: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if masks is None:
            raise FittingValidationError("Elliptical fitting requires masks")

        if np.max(smoothed_punctum) <= 0:
            dims = FittingConstants.PARAM_DIMENSIONS[FittingStrategy.ELLIPTICAL]
            return (np.full(dims["fit"], np.nan), np.full(dims["error"], np.nan))

        ig = gaussoptfuncs.initial_guess_elliptical(
            smoothed_punctum, punctum, masks
        )
        return self._perform_elliptical_fit(
            punctum, ig, masks, weights, relative_coords
        )

    def _perform_elliptical_fit(
        self,
        data: np.ndarray,
        initial_guess: np.ndarray,
        masks: np.ndarray,
        weights: np.ndarray,
        relative_coords: List[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        size = int(data.shape[0])
        ravelsize = size * size
        dims = FittingConstants.PARAM_DIMENSIONS[FittingStrategy.ELLIPTICAL]

        try:
            pfit, pcov, _, _, success = leastsq(
                gaussoptfuncs.WLS_chi_elliptical_nobounds,
                x0=initial_guess,
                args=(data, masks, weights, size, ravelsize),
                full_output=True,
                ftol=FittingConstants.DEFAULT_FTOL,
                xtol=FittingConstants.DEFAULT_XTOL,
            )

            if success not in np.array([1, 2, 3, 4]) or _lm_stuck(pfit, initial_guess):
                return (np.full(dims["fit"], np.nan), np.full(dims["error"], np.nan))

            residuals = gaussoptfuncs.WLS_chi_elliptical_nobounds(
                pfit, data, masks, weights, size, ravelsize
            )
            chisqr = FittingResultProcessor.calculate_reduced_chisquared(
                residuals, ravelsize, len(initial_guess)
            )
            pcov = FittingResultProcessor.process_covariance(
                pcov, chisqr, ravelsize, len(initial_guess)
            )

            delta_chi2 = FittingResultProcessor.delta_chi_squared(residuals, data, masks, weights)
            return FittingResultProcessor.process_fit_results(
                pfit, pcov, size, relative_coords, FittingStrategy.ELLIPTICAL, chisqr,
                delta_chi2=delta_chi2, gate=self.gate,
            )

        except Exception as e:
            import traceback
            logging.warning(f"Elliptical fitting failed: {e}")
            logging.warning(
                f"Full traceback:\n{''.join(traceback.format_tb(e.__traceback__))}"
            )
            return (np.full(dims["fit"], np.nan), np.full(dims["error"], np.nan))


class NoColourFittingProcessor(FittingProcessor):
    """Processor for no-colour fitting strategy."""

    def fit_single_punctum(
        self,
        punctum: np.ndarray,
        smoothed_punctum: np.ndarray,
        weights: np.ndarray,
        relative_coords: List[float],
        masks: Optional[np.ndarray] = None,
        readnoise_map: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit single punctum using no-colour strategy.

        Fits intensity information only, ignoring colour channels.

        Args:
            punctum: 2D array containing punctum data.
            smoothed_punctum: 2D array containing smoothed punctum data.
            weights: 2D array containing fitting weights.
            relative_coords: Relative coordinate offsets.
            masks: Not used for this strategy.

        Returns:
            Tuple of (fit_parameters, parameter_errors).
        """
        # Get initial guess (simplified for no-colour)
        initial_guess = self._generate_initial_guess(smoothed_punctum)

        # Perform fitting without colour information
        return self._perform_nocolour_fit(
            punctum, initial_guess, weights, relative_coords
        )

    def _generate_initial_guess(self, smoothed_punctum: np.ndarray) -> np.ndarray:
        """Generate initial guess for no-colour fitting.

        Args:
            smoothed_punctum: Smoothed punctum data.

        Returns:
            Initial parameter guess (8 parameters for no-colour).
        """
        centre = np.array(smoothed_punctum.shape) // 2
        max_val = np.max(smoothed_punctum)

        # 6-parameter guess for no-colour: [x, y, sx, sy, bg, A]
        # Note: WLS_nocolour_model_nobounds expects params[4]=background, params[5]=amplitude
        initial_guess = np.array(
            [
                centre[1],  # params[0] = x centre
                centre[0],  # params[1] = y centre
                1.0,  # params[2] = sigma_x
                1.0,  # params[3] = sigma_y
                np.min(smoothed_punctum),  # params[4] = background
                max_val,  # params[5] = amplitude
            ]
        )

        return initial_guess

    def _perform_nocolour_fit(
        self,
        data: np.ndarray,
        initial_guess: np.ndarray,
        weights: np.ndarray,
        relative_coords: List[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform no-colour fitting.

        Args:
            data: Data to fit.
            initial_guess: Initial parameter guess.
            weights: Fitting weights.
            relative_coords: Coordinate offsets.

        Returns:
            Tuple of (fit_parameters, parameter_errors).
        """
        size = int(data.shape[0])
        ravelsize = int(np.prod(data.shape))

        try:
            # Perform no-colour fitting using weighted least squares optimization
            pfit, pcov, infodict, errmsg, success = leastsq(
                gaussoptfuncs.WLS_chi_nocolour_nobounds,
                x0=initial_guess,
                args=(data, weights, size, ravelsize),
                full_output=True,
                ftol=FittingConstants.DEFAULT_FTOL,
                xtol=FittingConstants.DEFAULT_XTOL,
            )

            if success not in np.array([1, 2, 3, 4]):
                dims = FittingConstants.PARAM_DIMENSIONS[FittingStrategy.NOCOLOUR]
                return (np.full(dims["fit"], np.nan), np.full(dims["error"], np.nan))

            # Calculate chi-squared
            residuals = gaussoptfuncs.WLS_chi_nocolour_nobounds(
                pfit, data, weights, size, ravelsize
            )
            chisqr = FittingResultProcessor.calculate_reduced_chisquared(
                residuals, len(data.ravel()), len(initial_guess)
            )

            # Process covariance matrix
            pcov = FittingResultProcessor.process_covariance(
                pcov, chisqr, len(data.ravel()), len(initial_guess)
            )

            return FittingResultProcessor.process_fit_results(
                pfit,
                pcov,
                int(data.shape[0]),
                relative_coords,
                FittingStrategy.NOCOLOUR,
                chisqr,
            )

        except Exception as e:
            logging.warning(f"No-colour fitting failed: {e}")
            dims = FittingConstants.PARAM_DIMENSIONS[FittingStrategy.NOCOLOUR]
            return (np.full(dims["fit"], np.nan), np.full(dims["error"], np.nan))


class Image_Analysis_Functions:
    """A class for image analysis and Gaussian fitting in multicolour SMLM.

    This class provides comprehensive functionality for:
    - Multiple fitting strategies (colour, no-colour, just-colour, etc.)
    - Parallel processing for large datasets
    - Weighted least squares optimisation
    - Unified interface for all fitting approaches

    The class uses a strategy pattern to handle different fitting approaches
    while providing a consistent API and optimised performance.

    Example::

        analyzer = Image_Analysis_Functions()

        # Standard colour fitting
        fit_params, errors = analyzer.fit_puncta_method(
            puncta_data, smoothed_data, masks, weights, coords, planes,
            strategy=FittingStrategy.STANDARD
        )

        # Parallel processing
        fit_params, errors = analyzer.fit_puncta_parallel_method(
            puncta_data, smoothed_data, masks, weights, coords, planes,
            strategy=FittingStrategy.NOCOLOUR
        )
    """

    def __init__(self, helper_functions=None, readnoise: float = 1.5,
                 fit_gate: Any = "delta_chi2", noise_rate: float = 0.01):
        """Initialize the Image_Analysis_Functions class.

        Sets up strategy processors and loads required dependencies.

        Args:
            helper_functions: Helper functions instance (default: creates new instance)
            readnoise: Camera read noise in electrons, used by STANDARD_ITER for
                model-based weight updates (default 1.5 e-).
            fit_gate: Fit-acceptance gate for the colour Gaussian strategies (STANDARD,
                STANDARD_ITER, STANDARD_DATA, CIRCULAR, ELLIPTICAL): "delta_chi2" (default,
                a likelihood-ratio test calibrated to noise_rate), "amplitude_snr" (the
                previous Wald test), None (no gate), or a FitGate gate object.
            noise_rate: For "delta_chi2", the fraction of pure-noise ROIs that should pass
                (default 0.01).
        """
        # Initialize strategy processors
        self.processors = {
            FittingStrategy.STANDARD: StandardFittingProcessor(),
            FittingStrategy.STANDARD_ITER: StandardIterFittingProcessor(readnoise=readnoise),
            FittingStrategy.STANDARD_DATA: StandardDataFittingProcessor(readnoise=readnoise),
            FittingStrategy.CIRCULAR: CircularFittingProcessor(readnoise=readnoise),
            FittingStrategy.ELLIPTICAL: EllipticalFittingProcessor(),
            FittingStrategy.NOCOLOUR: NoColourFittingProcessor(),
        }

        self._skip_gate_prepare = False   # set in worker processes, which receive a prepared gate
        self.set_gate(fit_gate, noise_rate)

        # Initialize dependencies
        self.io = IOFunctions.IO_Functions()
        self.scmos = sCMOSFunctions.sCMOS_Functions()
        self.psf_f = PSFFunctions.PSF_Functions()
        self.helper = (
            helper_functions
            if helper_functions is not None
            else HelperFunctions.Helper_Functions()
        )

    # Processor attributes a worker process must inherit (see _fit_puncta_method_standalone)
    _WORKER_STATE_ATTRS = ("readnoise", "gate")

    _GATED_STRATEGIES = (
        FittingStrategy.STANDARD, FittingStrategy.STANDARD_ITER, FittingStrategy.STANDARD_DATA,
        FittingStrategy.CIRCULAR, FittingStrategy.ELLIPTICAL,
    )

    def set_gate(self, fit_gate: Any = "delta_chi2", noise_rate: float = 0.01) -> None:
        """Set the fit-acceptance gate of every gated strategy (see __init__)."""
        self.gate = FitGate.make_gate(fit_gate, noise_rate)
        for strategy in self._GATED_STRATEGIES:
            self.processors[strategy].gate = self.gate

    def _prepare_gate(self, strategy: FittingStrategy, puncta, masks, readnoise_maps) -> None:
        """Load (or calibrate, once, and cache) the delta-chi^2 gate's thresholds for this
        strategy, ROI size and read noise before fitting."""
        gate = getattr(self.processors[strategy], "gate", None)
        if (self._skip_gate_prepare or not isinstance(gate, FitGate.DeltaChi2Gate)
                or len(puncta) == 0 or masks is None):
            return
        if readnoise_maps is not None and len(readnoise_maps) > 0:
            readnoise = float(np.median(np.concatenate([np.ravel(r) for r in readnoise_maps])))
        else:
            readnoise = float(getattr(self.processors[strategy], "readnoise", 1.5))
        gate.prepare(strategy, int(np.shape(puncta[0])[0]), readnoise, np.asarray(masks[0]))

    def _processor_state(self, strategy: FittingStrategy) -> dict:
        """Settings of this strategy's processor that worker processes must copy."""
        processor = self.processors[strategy]
        return {name: getattr(processor, name) for name in self._WORKER_STATE_ATTRS
                if name in vars(processor)}

    def fit_puncta_method(
        self,
        puncta: List[np.ndarray],
        smoothed_puncta: List[np.ndarray],
        weights: List[np.ndarray],
        relative_coords: List[List[float]],
        planes: List[int],
        strategy: FittingStrategy,
        masks: Optional[List[np.ndarray]] = None,
        readnoise_maps: Optional[List[np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit puncta using specified strategy.

        Unified method that replaces the 19 duplicate fitting functions from
        the original implementation. Uses strategy pattern to handle different
        fitting approaches efficiently.

        Args:
            puncta: List of 2D arrays containing puncta data.
            smoothed_puncta: List of 2D arrays containing smoothed puncta data.
            weights: List of 2D arrays containing fitting weights.
            relative_coords: List of relative coordinate offsets for each punctum.
            planes: List of plane indices for each punctum.
            strategy: Fitting strategy to use (STANDARD, NOCOLOUR, etc.).
            masks: Optional list of 3D arrays containing colour masks.
                  Required for colour-based strategies.

        Returns:
            Tuple containing:
                - pfit_leastsq: Array of shape (n_puncta, n_params) with fit parameters
                - perr_leastsq: Array of shape (n_puncta, n_errors) with parameter errors

        Raises:
            FittingValidationError: If input parameters are invalid or inconsistent.

        Example::

            # Standard colour fitting
            fit_params, errors = analyzer.fit_puncta_method(
                puncta, smoothed, weights, coords, planes,
                strategy=FittingStrategy.STANDARD, masks=colour_masks
            )

            # No-colour fitting
            fit_params, errors = analyzer.fit_puncta_method(
                puncta, smoothed, weights, coords, planes,
                strategy=FittingStrategy.NOCOLOUR
            )
        """
        self._prepare_gate(strategy, puncta, masks, readnoise_maps)

        # Create and validate parameters
        params = FittingParameters(
            puncta=puncta,
            smoothed_puncta=smoothed_puncta,
            weights=weights,
            relative_coords=relative_coords,
            planes=planes,
            strategy=strategy,
            masks=masks,
        )

        # Get appropriate processor
        processor = self.processors[strategy]

        # Get array dimensions for this strategy
        dims = FittingConstants.PARAM_DIMENSIONS[strategy]

        # For colour strategies, derive actual fit/error dimensions from channel count in masks.
        # This makes the allocations correct for cameras with more than 3 colour channels.
        _colour_strategies = {
            FittingStrategy.STANDARD,
            FittingStrategy.STANDARD_ITER, FittingStrategy.STANDARD_DATA,
        }
        if strategy in _colour_strategies and masks is not None and len(masks) > 0:
            n_ch = masks[0].shape[-1]
            fit_dim = 4 + 2 * n_ch + 2   # [x,y,sx,sy, bg×n_ch, A×n_ch, chi, frame]
            err_dim = 4 + 2 * n_ch        # [xe,ye,sxe,sye, bg_err×n_ch, A_err×n_ch]
        elif strategy == FittingStrategy.CIRCULAR and masks is not None and len(masks) > 0:
            n_ch = masks[0].shape[-1]
            fit_dim = 3 + 2 * n_ch + 2   # [x,y,s, bg×n_ch, A×n_ch, chi, frame]
            err_dim = 3 + 2 * n_ch        # [xe,ye,se, bg_err×n_ch, A_err×n_ch]
        else:
            fit_dim = dims["fit"]
            err_dim = dims["error"]

        # Auto-detect precision requirements based on data range
        max_value = np.max([np.max(p) for p in puncta])
        precision_dtype = np.float64 if max_value > 50000 else np.float32

        # Pre-allocate result arrays with appropriate precision
        n_puncta = len(puncta)
        pfit_leastsq = np.empty((n_puncta, fit_dim), dtype=precision_dtype)
        perr_leastsq = np.empty((n_puncta, err_dim), dtype=precision_dtype)

        # Initialize with NaN
        pfit_leastsq.fill(np.nan)
        perr_leastsq.fill(np.nan)

        # Process each punctum
        for i, punctum in enumerate(puncta):
            try:
                # Get masks for this punctum if provided
                punctum_masks = masks[i] if masks is not None else None

                # Fit single punctum
                fit_params, fit_errors = processor.fit_single_punctum(
                    punctum=punctum,
                    smoothed_punctum=smoothed_puncta[i],
                    weights=weights[i],
                    relative_coords=relative_coords[i],
                    masks=punctum_masks,
                    readnoise_map=readnoise_maps[i] if readnoise_maps is not None else None,
                )

                # fit_params: [x,y,sx,sy, bg×n_ch, A×n_ch, chi_sqr]  (fit_dim-1 elements)
                # Final row:  [x,y,sx,sy, bg×n_ch, A×n_ch, chi_sqr, plane_idx]  (fit_dim elements)
                fit_param_count = len(fit_params)
                max_fit_params = fit_dim - 1  # Leave last position for plane index

                if fit_param_count <= max_fit_params:
                    pfit_leastsq[i, :fit_param_count] = fit_params
                else:
                    pfit_leastsq[i, :max_fit_params] = fit_params[:max_fit_params]

                pfit_leastsq[i, -1] = planes[i]
                perr_leastsq[i, :] = fit_errors[:err_dim]

            except Exception as e:
                logging.warning(f"Failed to fit punctum {i}: {e}")
                # Results already initialized with NaN
                continue

        return pfit_leastsq, perr_leastsq

    def fit_puncta_parallel_method(
        self,
        puncta: List[np.ndarray],
        smoothed_puncta: List[np.ndarray],
        weights: List[np.ndarray],
        relative_coords: List[List[float]],
        planes: List[int],
        strategy: FittingStrategy,
        masks: Optional[List[np.ndarray]] = None,
        asynch: bool = False,
        readnoise_maps: Optional[List[np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit puncta in parallel using specified strategy.

        Parallel version of fit_puncta_method that distributes work across
        multiple CPU cores for improved performance on large datasets.

        Args:
            puncta: List of 2D arrays containing puncta data.
            smoothed_puncta: List of 2D arrays containing smoothed puncta data.
            weights: List of 2D arrays containing fitting weights.
            relative_coords: List of relative coordinate offsets for each punctum.
            planes: List of plane indices for each punctum.
            strategy: Fitting strategy to use.
            masks: Optional list of 3D arrays containing colour masks.
            asynch: If True, return futures for asynchronous processing.

        Returns:
            Tuple containing fit parameters and errors arrays.
            If asynch=True, returns futures that can be processed later.

        Example::

            # Parallel processing with optimal batch size
            fit_params, errors = analyzer.fit_puncta_parallel_method(
                large_puncta_list, smoothed_list, weights_list,
                coords_list, planes_list,
                strategy=FittingStrategy.STANDARD, masks=masks_list
            )
        """
        # Calculate optimal parallelization parameters
        n_puncta = len(puncta)
        n_workers, n_tasks, puncta_per_task, start_indices = (
            self.helper.calculate_parallel_chunks(
                n_puncta,
                max_workers=FittingConstants.MAX_WORKERS,
                worker_ratio=FittingConstants.WORKER_RATIO,
                tasks_per_worker=FittingConstants.TASKS_PER_WORKER,
            )
        )

        self._prepare_gate(strategy, puncta, masks, readnoise_maps)

        # Worker processes build their own fitter, so copy this processor's settings
        # (scalar read noise, fit gate) across rather than falling back to defaults
        processor_state = self._processor_state(strategy)

        # Submit tasks to process pool
        fs = []
        with futures.ProcessPoolExecutor(n_workers) as executor:
            for i, n_puncta_task in zip(start_indices, puncta_per_task):
                if n_puncta_task == 0:
                    continue

                # Slice data for this task
                task_puncta = puncta[i : i + n_puncta_task]
                task_smoothed = smoothed_puncta[i : i + n_puncta_task]
                task_weights = weights[i : i + n_puncta_task]
                task_coords = relative_coords[i : i + n_puncta_task]
                task_planes = planes[i : i + n_puncta_task]
                task_masks = masks[i : i + n_puncta_task] if masks is not None else None
                task_readnoise = (
                    readnoise_maps[i : i + n_puncta_task] if readnoise_maps is not None else None
                )

                # Submit task
                fs.append(
                    executor.submit(
                        _fit_puncta_method_standalone,
                        task_puncta,
                        task_smoothed,
                        task_weights,
                        task_coords,
                        task_planes,
                        strategy,
                        task_masks,
                        task_readnoise,
                        processor_state,
                    )
                )

        if asynch:
            return fs

        # Wait for completion and combine results
        return self.fits_from_futures(fs, strategy)

    def fits_from_futures(
        self, fs: List[futures.Future], strategy: FittingStrategy
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Combine results from parallel fitting futures.

        Args:
            fs: List of futures from parallel fitting operations.
            strategy: Fitting strategy used (determines array dimensions).

        Returns:
            Tuple of combined (fit_parameters, parameter_errors) arrays.
        """
        # Get dimensions for this strategy
        dims = FittingConstants.PARAM_DIMENSIONS[strategy]

        # Collect all results
        all_fits = []
        all_errors = []

        with ProgressUtils.fitting_progress_bar(
            total=len(fs), desc="Collecting fitting results"
        ) as pbar:
            for f in fs:
                try:
                    fit_params, fit_errors = f.result()
                    all_fits.append(fit_params)
                    all_errors.append(fit_errors)
                except Exception as e:
                    logging.warning(f"Future failed: {e}")
                finally:
                    pbar.update(1)

        # Concatenate results
        if all_fits:
            combined_fits = np.vstack(all_fits)
            combined_errors = np.vstack(all_errors)
        else:
            # No successful results - use float64 for compatibility with high photon counts
            combined_fits = np.empty((0, dims["fit"]), dtype=np.float64)
            combined_errors = np.empty((0, dims["error"]), dtype=np.float64)

        return combined_fits, combined_errors


# Module-level standalone functions for multiprocessing (pickleable)
def _fit_puncta_method_standalone(
    puncta: List[np.ndarray],
    smoothed_puncta: List[np.ndarray],
    weights: List[np.ndarray],
    relative_coords: List[List[float]],
    planes: List[int],
    strategy: FittingStrategy,
    masks: Optional[List[np.ndarray]] = None,
    readnoise_maps: Optional[List[np.ndarray]] = None,
    processor_state: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Standalone version of fit_puncta_method for multiprocessing.

    This function creates a temporary instance to perform fitting
    since bound methods cannot be pickled for multiprocessing.
    """
    # Import here to ensure all dependencies are available in worker process
    # Add src to path if needed (for worker processes)
    _dir = str(Path(__file__).parent)
    if _dir not in sys.path:
        sys.path.insert(0, _dir)

    try:
        # Create instance with proper error handling
        fitter = Image_Analysis_Functions()
        # Settings of the parent process's processor (scalar read noise, prepared fit gate)
        for name, value in (processor_state or {}).items():
            setattr(fitter.processors[strategy], name, value)
        fitter._skip_gate_prepare = "gate" in (processor_state or {})
        return fitter.fit_puncta_method(
            puncta=puncta,
            smoothed_puncta=smoothed_puncta,
            weights=weights,
            relative_coords=relative_coords,
            planes=planes,
            strategy=strategy,
            masks=masks,
            readnoise_maps=readnoise_maps,
        )
    except Exception:
        # Return empty arrays if fitting fails to prevent crash
        dims = FittingConstants.PARAM_DIMENSIONS[strategy]
        n_puncta = len(puncta)
        _colour_strategies = {
            FittingStrategy.STANDARD,
            FittingStrategy.STANDARD_ITER, FittingStrategy.STANDARD_DATA,
        }
        if strategy in _colour_strategies and masks is not None and len(masks) > 0:
            n_ch = masks[0].shape[-1]
            fit_dim = 4 + 2 * n_ch + 2
            err_dim = 4 + 2 * n_ch
        elif strategy == FittingStrategy.CIRCULAR and masks is not None and len(masks) > 0:
            n_ch = masks[0].shape[-1]
            fit_dim = 3 + 2 * n_ch + 2
            err_dim = 3 + 2 * n_ch
        else:
            fit_dim = dims["fit"]
            err_dim = dims["error"]
        empty_fits = np.full((n_puncta, fit_dim), np.nan, dtype=np.float64)
        empty_errors = np.full((n_puncta, err_dim), np.nan, dtype=np.float64)
        return empty_fits, empty_errors
