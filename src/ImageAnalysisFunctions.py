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


class FittingStrategy(Enum):
    """Enumeration for different fitting strategies.

    Defines the various approaches to Gaussian fitting with different
    colour channel handling strategies.
    """

    STANDARD = "standard"  # Full colour fitting with all channels
    STANDARD_IG = "standard_ig"  # Full fit on raw Bayer, seeded from demosaiced fit
    STANDARD_ITER = "standard_iter"  # STANDARD with 2 IRLS model-weight iterations
    STANDARD_DATA = "standard_data"  # smooth → model → raw-data weights (unbiased final pass)
    CIRCULAR = "circular"  # STANDARD_DATA with a single shared PSF width (no sx/sy split)
    ELLIPTICAL = "elliptical"  # Rotated elliptical Gaussian (11 params; for tracking)
    NOCOLOUR = "nocolour"  # No colour information, intensity only
    JUSTCOLOUR = "justcolour"  # Colour channels only, no intensity
    RAWCOLOUR = "rawcolour"  # Raw colour data without processing
    POSTHENCOLOUR = "posthencolour"  # Post-processing colour enhancement


class FittingConstants:
    """Constants for fitting operations."""

    # Fitting tolerances
    DEFAULT_FTOL = 1e-2
    DEFAULT_XTOL = 1e-2

    # Amplitude SNR threshold (Wald t-statistic, replaces all three hard gates)
    AMPLITUDE_SNR_THRESHOLD = 2.0  # sigma; z = sum(|q_c|)/sqrt(sum(var(q_c))) >= this

    # Parallel processing limits
    MAX_WORKERS = 60  # Python crashes when using >64 cores
    WORKER_RATIO = 0.9
    TASKS_PER_WORKER = 100

    # Array dimensions for different strategies
    PARAM_DIMENSIONS = {
        FittingStrategy.STANDARD: {"fit": 12, "error": 10},
        FittingStrategy.STANDARD_IG: {"fit": 12, "error": 10},
        FittingStrategy.STANDARD_ITER: {"fit": 12, "error": 10},
        FittingStrategy.STANDARD_DATA: {"fit": 12, "error": 10},
        FittingStrategy.CIRCULAR: {"fit": 11, "error": 9},
        FittingStrategy.ELLIPTICAL: {"fit": 13, "error": 11},
        FittingStrategy.NOCOLOUR: {"fit": 8, "error": 6},
        FittingStrategy.JUSTCOLOUR: {"fit": 4, "error": 2},
        FittingStrategy.RAWCOLOUR: {"fit": 8, "error": 6},
        FittingStrategy.POSTHENCOLOUR: {"fit": 10, "error": 8},
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
            FittingStrategy.JUSTCOLOUR,
            FittingStrategy.RAWCOLOUR,
            FittingStrategy.POSTHENCOLOUR,
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
    def calculate_errors(pcov: np.ndarray, strategy: FittingStrategy) -> List[float]:
        """Calculate parameter errors from covariance matrix.

        Args:
            pcov: Parameter covariance matrix from fitting.
            strategy: Fitting strategy to determine expected error array size.

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
            error_list = errors.tolist()

            # Pad if pcov was undersized (degenerate fit)
            if len(error_list) < expected_size:
                error_list.extend([np.nan] * (expected_size - len(error_list)))

            return error_list
        except (IndexError, ValueError):
            return [np.nan] * expected_size

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
    def process_fit_results(
        pfit: np.ndarray,
        pcov: np.ndarray,
        size: int,
        relative_coords: List[float],
        strategy: FittingStrategy,
        chisqr: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Process raw fitting results into standardized format.

        Args:
            pfit: Raw fitting parameters.
            pcov: Parameter covariance matrix (already chi_sqr-scaled).
            relative_coords: Coordinate offsets to add.
            size: Image size in fitting.
            strategy: Fitting strategy used.
            chisqr: Chi-squared value from fitting (default 1.0).

        Returns:
            Tuple of (processed_parameters, parameter_errors).
        """
        if pfit is None or not hasattr(pfit, "__len__") or len(pfit) == 0:
            dims = FittingConstants.PARAM_DIMENSIONS[strategy]
            return (np.full(dims["fit"], np.nan), np.full(dims["error"], np.nan))

        # Chi-squared is now passed as parameter from the fitting processor

        # Process parameters based on strategy
        # For STANDARD: pfit has [x, y, sy, sx, bg_B, bg_G, bg_R, A_B, A_G, A_R] (10 parameters from gaussoptfuncs)
        # But output needs [x, y, sx, sy, bg_B, bg_G, bg_R, A_B, A_G, A_R] (sx/sy order corrected)

        _standard_like = {FittingStrategy.STANDARD, FittingStrategy.STANDARD_ITER, FittingStrategy.STANDARD_DATA}

        if strategy in _standard_like:
            # Dynamic n_ch: pfit layout is [x, y, sy, sx, bg_0,...,bg_{n-1}, A_0,...,A_{n-1}]
            # Output layout:               [x, y, sx, sy, bg_0,...,bg_{n-1}, A_0,...,A_{n-1}]
            n_ch = (len(pfit) - 4) // 2
            pfit_processed = np.concatenate([
                np.array([pfit[0], pfit[1], pfit[3], pfit[2]]),  # x, y, sx, sy (swap indices 2,3)
                pfit[4:4 + n_ch],           # bg channels
                pfit[4 + n_ch:4 + 2 * n_ch],  # A channels
            ])
        elif strategy == FittingStrategy.ELLIPTICAL:
            # pfit: [x0, y0, sigma_x, sigma_y, theta, √bg_B, √bg_G, √bg_R, √A_B, √A_G, √A_R]
            # output: [xc, yc, s_x, s_y, theta, bg_B, bg_G, bg_R, A_B, A_G, A_R]
            pfit_processed = np.array(
                [
                    pfit[0],
                    pfit[1],
                    pfit[3],
                    pfit[2],   # x, y, s_x (col sigma), s_y (row sigma)
                    pfit[4],   # theta
                    pfit[5],
                    pfit[6],
                    pfit[7],   # bg_B, bg_G, bg_R
                    pfit[8],
                    pfit[9],
                    pfit[10],  # A_B, A_G, A_R
                ]
            )
        elif strategy == FittingStrategy.CIRCULAR:
            # pfit: [x, y, s, bg_0,...,bg_{n-1}, A_0,...,A_{n-1}] -- no sx/sy swap needed,
            # there is only one width parameter, so this is used as-is.
            pfit_processed = pfit.copy()
        else:
            # For other strategies, use as-is for now
            pfit_processed = pfit.copy()

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

        # Stage 2: Amplitude SNR gate (Wald t-statistic, replaces MIN_PHOTON + MAX_CHI_SQUARED)
        # Uses sqrt-space pfit[7:10] and chi_sqr-scaled pcov — must run BEFORE squaring.
        # High chi_sqr inflates pcov → reduces z (automatically conservative for poor fits).
        if strategy in _standard_like:
            amplitude_snr = FittingResultProcessor._compute_amplitude_snr(pfit, pcov)
            if amplitude_snr < FittingConstants.AMPLITUDE_SNR_THRESHOLD:
                return (
                    np.full(len(pfit_processed), np.nan),
                    np.full(len(pfit_processed), np.nan),
                )
        elif strategy == FittingStrategy.ELLIPTICAL:
            # Amplitudes are at indices 8:11 in the 11-param elliptical vector
            amplitude_snr = FittingResultProcessor._compute_amplitude_snr_elliptical(pfit, pcov)
            if amplitude_snr < FittingConstants.AMPLITUDE_SNR_THRESHOLD:
                return (
                    np.full(len(pfit_processed), np.nan),
                    np.full(len(pfit_processed), np.nan),
                )
        elif strategy == FittingStrategy.CIRCULAR:
            amplitude_snr = FittingResultProcessor._compute_amplitude_snr_circular(pfit, pcov)
            if amplitude_snr < FittingConstants.AMPLITUDE_SNR_THRESHOLD:
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
        elif strategy == FittingStrategy.JUSTCOLOUR:
            # params[0]=sqrt(A), params[1]=sqrt(b) — square both
            pfit_processed[0:2] = np.square(pfit_processed[0:2])
        elif strategy == FittingStrategy.RAWCOLOUR:
            # params[0:3]=sqrt(bg_B/G/R), params[3:6]=sqrt(A_B/G/R) — square all
            pfit_processed[0:6] = np.square(pfit_processed[0:6])

        # Append chi-squared
        pfit_final = np.append(pfit_processed, chisqr)

        # Calculate errors
        errors = FittingResultProcessor.calculate_errors(pcov, strategy)

        return pfit_final, np.array(errors)


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
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit a single punctum with this strategy.

        Args:
            punctum: 2D array containing punctum data.
            smoothed_punctum: 2D array containing smoothed punctum data.
            weights: 2D array containing fitting weights.
            relative_coords: Relative coordinate offsets.
            masks: Optional 3D array containing colour masks.

        Returns:
            Tuple of (fit_parameters, parameter_errors).
        """
        pass


class StandardFittingProcessor(FittingProcessor):
    """Processor for standard colour fitting strategy."""

    def fit_single_punctum(
        self,
        punctum: np.ndarray,
        smoothed_punctum: np.ndarray,
        weights: np.ndarray,
        relative_coords: List[float],
        masks: Optional[np.ndarray] = None,
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

            if success not in np.array([1, 2, 3, 4]):
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

            return FittingResultProcessor.process_fit_results(
                pfit, pcov, size, relative_coords, FittingStrategy.STANDARD, chisqr,
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


class StandardIGFittingProcessor(StandardFittingProcessor):
    """Full STANDARD fit on raw Bayer data, seeded from a prior demosaiced fit.

    The six-element ``relative_coords`` tuple carries
    ``(xc, yc, s_x, s_y, b, A)`` from a NOCOLOUR fit on the demosaiced image.
    These replace the image-derived initial guess so the optimiser starts from
    an accurate position rather than the centre-of-mass estimate, improving
    both colour accuracy and localisation precision.

    WLS_model_nobounds parameter layout:
        [x, y, sigma_y, sigma_x, sqrt(bg_B), sqrt(bg_G), sqrt(bg_R),
         sqrt(A_B), sqrt(A_G), sqrt(A_R)]

    Note sigma_y precedes sigma_x — opposite of the NOCOLOUR convention —
    so the seed swaps them at indices 2/3.
    """

    def fit_single_punctum(
        self,
        punctum: np.ndarray,
        smoothed_punctum: np.ndarray,
        weights: np.ndarray,
        relative_coords,          # (xc, yc, s_x, s_y, b, A) from demosaiced fit
        masks: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if masks is None:
            raise FittingValidationError("Standard-IG fitting requires masks")

        n_ch = masks.shape[-1]

        if np.max(smoothed_punctum) <= 0:
            return (np.full(4 + 2 * n_ch + 2, np.nan), np.full(4 + 2 * n_ch, np.nan))

        xc, yc, s_x, s_y, b, A = (float(v) for v in relative_coords)
        b_ch = max(b / n_ch, 1e-6)
        A_ch = max(A / n_ch, 1e-6)

        # sigma_y at index 2, sigma_x at index 3 (WLS_model_nobounds convention)
        initial_guess = np.concatenate([
            [xc, yc, s_y, s_x],
            np.full(n_ch, np.sqrt(b_ch)),   # bg channels
            np.full(n_ch, np.sqrt(A_ch)),   # A channels
        ])

        # relative_coords=(0,0): xc/yc in the seed are already local ROI coords
        return self._perform_wls_fit(punctum, initial_guess, masks, weights, (0.0, 0.0))


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
        self, pfit: np.ndarray, masks: np.ndarray, size: int
    ) -> np.ndarray:
        """Compute per-pixel weights from the current model estimate."""
        x_arr = np.arange(size, dtype=np.float32)
        buf = np.zeros((size, size), dtype=np.float32)
        model = gaussoptfuncs.WLS_model_nobounds(
            pfit.astype(np.float32), masks, x_arr, buf
        )
        e = np.maximum(model, 0).astype(np.float32) + 1.0 + float(self.readnoise) ** 2
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
    ) -> Tuple[np.ndarray, np.ndarray]:
        if masks is None:
            raise FittingValidationError("Standard-ITER fitting requires masks")

        n_ch = masks.shape[-1]
        nan_result = (np.full(4 + 2 * n_ch + 2, np.nan), np.full(4 + 2 * n_ch, np.nan))

        if np.max(smoothed_punctum) <= 0:
            return nan_result

        size = int(punctum.shape[0])
        ravelsize = size * size

        try:
            ig = self._generate_initial_guess(smoothed_punctum, punctum, masks)

            # Stage 1: smoothing weights (passed in from SR_Functions / simulation)
            pfit1, _, ok1 = self._leastsq_step(ig, punctum, masks, weights, size)
            if ok1 not in (1, 2, 3, 4):
                return nan_result

            # Stage 2: model weights from Stage 1 fit
            w2 = self._model_based_weights(pfit1, masks, size)
            pfit2, _, ok2 = self._leastsq_step(pfit1, punctum, masks, w2, size)
            if ok2 not in (1, 2, 3, 4):
                return nan_result

            # Stage 3: model weights from Stage 2 fit (final)
            w3 = self._model_based_weights(pfit2, masks, size)
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

            return FittingResultProcessor.process_fit_results(
                pfit3, pcov3, size, relative_coords, FittingStrategy.STANDARD_ITER, chisqr
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

    def _raw_data_weights(self, punctum: np.ndarray) -> np.ndarray:
        """Compute per-pixel weights from the observed raw data.

        Args:
            punctum: Raw photoelectron ROI, shape (H, W), float32.

        Returns:
            float32 weight array, shape (H, W).
        """
        e = np.maximum(punctum, 0).astype(np.float32) + 1.0 + float(self.readnoise) ** 2
        return (1.0 / e).astype(np.float32)

    def fit_single_punctum(
        self,
        punctum: np.ndarray,
        smoothed_punctum: np.ndarray,
        weights: np.ndarray,
        relative_coords,
        masks: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if masks is None:
            raise FittingValidationError("Standard-DATA fitting requires masks")

        n_ch = masks.shape[-1]
        nan_result = (np.full(4 + 2 * n_ch + 2, np.nan), np.full(4 + 2 * n_ch, np.nan))

        if np.max(smoothed_punctum) <= 0:
            return nan_result

        size = int(punctum.shape[0])
        ravelsize = size * size

        try:
            ig = self._generate_initial_guess(smoothed_punctum, punctum, masks)

            # Stage 1: smoothing weights (passed in from SR_Functions / simulation)
            pfit1, _, ok1 = self._leastsq_step(ig, punctum, masks, weights, size)
            if ok1 not in (1, 2, 3, 4):
                return nan_result

            # Stage 2: model weights from Stage 1 fit
            w2 = self._model_based_weights(pfit1, masks, size)
            pfit2, _, ok2 = self._leastsq_step(pfit1, punctum, masks, w2, size)
            if ok2 not in (1, 2, 3, 4):
                return nan_result

            # Stage 3: raw-data weights (unbiased final pass)
            w3 = self._raw_data_weights(punctum)
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

            return FittingResultProcessor.process_fit_results(
                pfit3, pcov3, size, relative_coords, FittingStrategy.STANDARD_DATA, chisqr
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
        self, pfit: np.ndarray, masks: np.ndarray, size: int
    ) -> np.ndarray:
        """Same as StandardIterFittingProcessor._model_based_weights, but pfit carries
        one shared width -- expand to [x, y, s, s, bg..., A...] before evaluating the
        (unchanged) coloured-Gaussian model.
        """
        full_pfit = np.concatenate([pfit[:2], [pfit[2]], pfit[2:]]).astype(np.float32)
        x_arr = np.arange(size, dtype=np.float32)
        buf = np.zeros((size, size), dtype=np.float32)
        model = gaussoptfuncs.WLS_model_nobounds(full_pfit, masks, x_arr, buf)
        e = np.maximum(model, 0).astype(np.float32) + 1.0 + float(self.readnoise) ** 2
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
    ) -> Tuple[np.ndarray, np.ndarray]:
        if masks is None:
            raise FittingValidationError("Circular fitting requires masks")

        n_ch = masks.shape[-1]
        nan_result = (np.full(3 + 2 * n_ch + 2, np.nan), np.full(3 + 2 * n_ch, np.nan))

        if np.max(smoothed_punctum) <= 0:
            return nan_result

        size = int(punctum.shape[0])
        ravelsize = size * size

        try:
            ig = self._generate_initial_guess(smoothed_punctum, punctum, masks)

            # Stage 1: smoothing weights (passed in from SR_Functions / simulation)
            pfit1, _, ok1 = self._leastsq_step(ig, punctum, masks, weights, size)
            if ok1 not in (1, 2, 3, 4):
                return nan_result

            # Stage 2: model weights from Stage 1 fit
            w2 = self._model_based_weights(pfit1, masks, size)
            pfit2, _, ok2 = self._leastsq_step(pfit1, punctum, masks, w2, size)
            if ok2 not in (1, 2, 3, 4):
                return nan_result

            # Stage 3: raw-data weights (unbiased final pass)
            w3 = self._raw_data_weights(punctum)
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

            return FittingResultProcessor.process_fit_results(
                pfit3, pcov3, size, relative_coords, FittingStrategy.CIRCULAR, chisqr
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

            if success not in np.array([1, 2, 3, 4]):
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

            return FittingResultProcessor.process_fit_results(
                pfit, pcov, size, relative_coords, FittingStrategy.ELLIPTICAL, chisqr
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


class JustColourFittingProcessor(FittingProcessor):
    """Processor for just-colour fitting strategy."""

    def fit_single_punctum(
        self,
        punctum: np.ndarray,
        smoothed_punctum: np.ndarray,
        weights: np.ndarray,
        relative_coords: List[float],
        masks: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit single punctum using just-colour strategy.

        Args:
            punctum: 2D array containing punctum data.
            smoothed_punctum: 2D array containing smoothed punctum data.
            weights: 2D array containing fitting weights.
            relative_coords: Relative coordinate offsets.
            masks: 3D array containing colour masks (required).

        Returns:
            Tuple of (fit_parameters, parameter_errors).
        """
        if masks is None:
            raise FittingValidationError("Just-colour fitting requires masks")

        # Implementation similar to other processors but for just-colour strategy
        initial_guess = self._generate_initial_guess(smoothed_punctum, masks)
        return self._perform_justcolour_fit(
            punctum, initial_guess, masks, weights, relative_coords
        )

    def _generate_initial_guess(
        self, smoothed_punctum: np.ndarray, masks: np.ndarray
    ) -> np.ndarray:
        """Generate initial guess for just-colour fitting.

        Position and shape are fixed via locparams; only amplitude and
        background are free parameters (in sqrt-space, as the model uses
        params[0]**2 and params[1]**2).
        """
        max_val = np.max(smoothed_punctum)
        min_val = np.min(smoothed_punctum)
        return np.array(
            [
                np.sqrt(np.abs(max_val)),   # params[0] = sqrt(A)
                np.sqrt(np.abs(min_val)),   # params[1] = sqrt(b)
            ]
        )

    def _perform_justcolour_fit(
        self,
        data: np.ndarray,
        initial_guess: np.ndarray,
        masks: np.ndarray,
        weights: np.ndarray,
        relative_coords: List[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform just-colour fitting."""
        size = int(data.shape[0])
        locparams = relative_coords  # For justcolour fitting

        try:
            pfit, pcov, infodict, errmsg, success = leastsq(
                gaussoptfuncs.WLS_chi_justcolour_nobounds,
                x0=initial_guess,
                args=(data, weights, size, locparams),
                full_output=True,
                ftol=FittingConstants.DEFAULT_FTOL,
                xtol=FittingConstants.DEFAULT_XTOL,
            )

            if success not in np.array([1, 2, 3, 4]):
                dims = FittingConstants.PARAM_DIMENSIONS[FittingStrategy.JUSTCOLOUR]
                return (np.full(dims["fit"], np.nan), np.full(dims["error"], np.nan))

            # Calculate chi-squared
            residuals = gaussoptfuncs.WLS_chi_justcolour_nobounds(
                pfit, data, weights, size, locparams
            )
            chisqr = FittingResultProcessor.calculate_reduced_chisquared(
                residuals, len(data.ravel()), len(initial_guess)
            )

            # Process covariance matrix
            pcov = FittingResultProcessor.process_covariance(
                pcov, chisqr, len(data.ravel()), len(initial_guess)
            )

            return FittingResultProcessor.process_fit_results(
                pfit, pcov, size, relative_coords, FittingStrategy.JUSTCOLOUR, chisqr
            )

        except Exception as e:
            logging.warning(f"Just-colour fitting failed: {e}")
            dims = FittingConstants.PARAM_DIMENSIONS[FittingStrategy.JUSTCOLOUR]
            return (np.full(dims["fit"], np.nan), np.full(dims["error"], np.nan))


class RawColourFittingProcessor(FittingProcessor):
    """Processor for raw-colour fitting strategy."""

    def fit_single_punctum(
        self,
        punctum: np.ndarray,
        smoothed_punctum: np.ndarray,
        weights: np.ndarray,
        relative_coords: List[float],
        masks: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit single punctum using raw-colour strategy."""
        if masks is None:
            raise FittingValidationError("Raw-colour fitting requires masks")

        initial_guess = self._generate_initial_guess(smoothed_punctum, masks)
        return self._perform_rawcolour_fit(
            punctum, initial_guess, masks, weights, relative_coords
        )

    def _generate_initial_guess(
        self, smoothed_punctum: np.ndarray, masks: np.ndarray
    ) -> np.ndarray:
        """Generate initial guess for raw-colour fitting.

        Position and shape are fixed via locparams; only per-channel
        backgrounds and amplitudes are free (in sqrt-space).  Layout must
        match the negative-index access in WLS_rawcolour_model_nobounds:
          params[-6+i] -> bg_B, bg_G, bg_R  (indices 0, 1, 2)
          params[-3+i] -> A_B,  A_G,  A_R   (indices 3, 4, 5)
        """
        max_val = np.max(smoothed_punctum)
        min_val = np.min(smoothed_punctum)
        return np.array(
            [
                np.sqrt(np.abs(min_val)),        # params[0] = sqrt(bg_B)
                np.sqrt(np.abs(min_val)),        # params[1] = sqrt(bg_G)
                np.sqrt(np.abs(min_val)),        # params[2] = sqrt(bg_R)
                np.sqrt(np.abs(max_val * 0.33)), # params[3] = sqrt(A_B)
                np.sqrt(np.abs(max_val * 0.33)), # params[4] = sqrt(A_G)
                np.sqrt(np.abs(max_val * 0.33)), # params[5] = sqrt(A_R)
            ]
        )

    def _perform_rawcolour_fit(
        self,
        data: np.ndarray,
        initial_guess: np.ndarray,
        masks: np.ndarray,
        weights: np.ndarray,
        relative_coords: List[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform raw-colour fitting."""
        size = int(data.shape[0])
        ravelsize = int(np.prod(data.shape))
        locparams = relative_coords  # For rawcolour fitting

        try:
            pfit, pcov, infodict, errmsg, success = leastsq(
                gaussoptfuncs.WLS_rawcolour_chi_nobounds,
                x0=initial_guess,
                args=(data, masks, weights, size, ravelsize, locparams),
                full_output=True,
                ftol=FittingConstants.DEFAULT_FTOL,
                xtol=FittingConstants.DEFAULT_XTOL,
            )

            if success not in np.array([1, 2, 3, 4]):
                dims = FittingConstants.PARAM_DIMENSIONS[FittingStrategy.RAWCOLOUR]
                return (np.full(dims["fit"], np.nan), np.full(dims["error"], np.nan))

            # Calculate chi-squared
            residuals = gaussoptfuncs.WLS_rawcolour_chi_nobounds(
                pfit, data, masks, weights, size, ravelsize, locparams
            )
            chisqr = FittingResultProcessor.calculate_reduced_chisquared(
                residuals, len(data.ravel()), len(initial_guess)
            )

            # Process covariance matrix
            pcov = FittingResultProcessor.process_covariance(
                pcov, chisqr, len(data.ravel()), len(initial_guess)
            )

            return FittingResultProcessor.process_fit_results(
                pfit, pcov, size, relative_coords, FittingStrategy.RAWCOLOUR, chisqr
            )

        except Exception as e:
            logging.warning(f"Raw-colour fitting failed: {e}")
            dims = FittingConstants.PARAM_DIMENSIONS[FittingStrategy.RAWCOLOUR]
            return (np.full(dims["fit"], np.nan), np.full(dims["error"], np.nan))


class PosthenColourFittingProcessor(FittingProcessor):
    """Processor for post-colour enhancement fitting strategy."""

    def fit_single_punctum(
        self,
        punctum: np.ndarray,
        smoothed_punctum: np.ndarray,
        weights: np.ndarray,
        relative_coords: List[float],
        masks: Optional[np.ndarray] = None,
        raw_punctum: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit single punctum using post-colour enhancement strategy.

        The posthencolour strategy performs a two-stage fit:
        1. First fits position using no-colour (greyscale) data
        2. Then fits colour using raw data with fixed positions
        """
        if masks is None:
            raise FittingValidationError("Post-colour fitting requires masks")

        # Use raw_punctum if provided, otherwise use punctum
        if raw_punctum is None:
            raw_punctum = punctum

        return self._perform_posthencolour_fit(
            punctum, raw_punctum, smoothed_punctum, weights, masks, relative_coords
        )

    def _perform_posthencolour_fit(
        self,
        grayscale_punctum: np.ndarray,
        raw_punctum: np.ndarray,
        smoothed_punctum: np.ndarray,
        weights: np.ndarray,
        masks: np.ndarray,
        relative_coords: List[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform post-colour enhancement fitting.

        This implements the two-stage fitting approach from the original:
        1. First fit position parameters using greyscale data (no-colour)
        2. Then fit colour parameters using raw data with fixed positions
        """
        try:
            # Generate initial guess from smoothed data
            initial_guess = np.empty(10, dtype=np.float32)
            initial_guess[:] = gaussoptfuncs.initial_guess(
                smoothed_punctum, raw_punctum, masks
            )

            # Split into position and colour components
            initial_guess_position = initial_guess[:6]
            initial_guess_position[-1] = initial_guess[-1] * 3
            initial_guess_position[-2] = initial_guess_position[-2] * 3
            initial_guess_colour = initial_guess[6:]

            # Stage 1: Fit position using no-colour (greyscale) fitting
            nocolour_processor = NoColourFittingProcessor()
            pfit_pos_leastsq, perr_pos_leastsq = (
                nocolour_processor._perform_nocolour_fit(
                    grayscale_punctum, initial_guess_position, weights, relative_coords
                )
            )

            # Extract location parameters for colour fitting
            locparams = np.array(
                [
                    pfit_pos_leastsq[0],
                    pfit_pos_leastsq[1],
                    pfit_pos_leastsq[2],
                    pfit_pos_leastsq[3],
                    pfit_pos_leastsq[4],
                    pfit_pos_leastsq[5],
                ]
            )

            # Stage 2: Fit colour using raw data with fixed positions
            rawcolour_processor = RawColourFittingProcessor()

            # Create a mock WLS_fit_rawcolour_nobounds call
            size = int(raw_punctum.shape[0])
            ravelsize = int(np.prod(raw_punctum.shape))

            pfit_colour, pcov_colour, infodict, errmsg, success = leastsq(
                gaussoptfuncs.WLS_rawcolour_chi_nobounds,
                x0=initial_guess_colour,
                args=(raw_punctum, masks, weights, size, ravelsize, locparams),
                full_output=True,
                ftol=FittingConstants.DEFAULT_FTOL,
                xtol=FittingConstants.DEFAULT_XTOL,
            )

            if success not in np.array([1, 2, 3, 4]):
                dims = FittingConstants.PARAM_DIMENSIONS[FittingStrategy.POSTHENCOLOUR]
                return (np.full(dims["fit"], np.nan), np.full(dims["error"], np.nan))

            # Calculate chi-squared for colour component
            residuals_colour = gaussoptfuncs.WLS_rawcolour_chi_nobounds(
                pfit_colour,
                raw_punctum,
                masks,
                weights,
                size,
                ravelsize,
                locparams,
            )
            chisqr_colour = FittingResultProcessor.calculate_reduced_chisquared(
                residuals_colour, len(raw_punctum.ravel()), len(initial_guess_colour)
            )

            # Process covariance matrix
            pcov_colour = FittingResultProcessor.process_covariance(
                pcov_colour,
                chisqr_colour,
                len(raw_punctum.ravel()),
                len(initial_guess_colour),
            )

            # Calculate errors for colour component
            perr_colour_leastsq = FittingResultProcessor.calculate_errors(
                pcov_colour, FittingStrategy.RAWCOLOUR
            )

            # Combine position and colour results
            pfit_leastsq = np.concatenate(
                [pfit_pos_leastsq[:-1], pfit_colour]
            )  # Exclude position chi-squared
            perr_leastsq = np.concatenate(
                [perr_pos_leastsq[:-1], perr_colour_leastsq]
            )  # Exclude position error

            return pfit_leastsq, np.array(perr_leastsq)

        except Exception as e:
            logging.warning(f"Post-colour fitting failed: {e}")
            dims = FittingConstants.PARAM_DIMENSIONS[FittingStrategy.POSTHENCOLOUR]
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

    def __init__(self, helper_functions=None, readnoise: float = 1.5):
        """Initialize the Image_Analysis_Functions class.

        Sets up strategy processors and loads required dependencies.

        Args:
            helper_functions: Helper functions instance (default: creates new instance)
            readnoise: Camera read noise in electrons, used by STANDARD_ITER for
                model-based weight updates (default 1.5 e-).
        """
        # Initialize strategy processors
        self.processors = {
            FittingStrategy.STANDARD: StandardFittingProcessor(),
            FittingStrategy.STANDARD_IG: StandardIGFittingProcessor(),
            FittingStrategy.STANDARD_ITER: StandardIterFittingProcessor(readnoise=readnoise),
            FittingStrategy.STANDARD_DATA: StandardDataFittingProcessor(readnoise=readnoise),
            FittingStrategy.CIRCULAR: CircularFittingProcessor(readnoise=readnoise),
            FittingStrategy.ELLIPTICAL: EllipticalFittingProcessor(),
            FittingStrategy.NOCOLOUR: NoColourFittingProcessor(),
            FittingStrategy.JUSTCOLOUR: JustColourFittingProcessor(),
            FittingStrategy.RAWCOLOUR: RawColourFittingProcessor(),
            FittingStrategy.POSTHENCOLOUR: PosthenColourFittingProcessor(),
        }

        # Initialize dependencies
        self.io = IOFunctions.IO_Functions()
        self.scmos = sCMOSFunctions.sCMOS_Functions()
        self.psf_f = PSFFunctions.PSF_Functions()
        self.helper = (
            helper_functions
            if helper_functions is not None
            else HelperFunctions.Helper_Functions()
        )

    def fit_puncta_method(
        self,
        puncta: List[np.ndarray],
        smoothed_puncta: List[np.ndarray],
        weights: List[np.ndarray],
        relative_coords: List[List[float]],
        planes: List[int],
        strategy: FittingStrategy,
        masks: Optional[List[np.ndarray]] = None,
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
            FittingStrategy.STANDARD, FittingStrategy.STANDARD_IG,
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
        return fitter.fit_puncta_method(
            puncta=puncta,
            smoothed_puncta=smoothed_puncta,
            weights=weights,
            relative_coords=relative_coords,
            planes=planes,
            strategy=strategy,
            masks=masks,
        )
    except Exception:
        # Return empty arrays if fitting fails to prevent crash
        dims = FittingConstants.PARAM_DIMENSIONS[strategy]
        n_puncta = len(puncta)
        _colour_strategies = {
            FittingStrategy.STANDARD, FittingStrategy.STANDARD_IG,
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
