#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global Constants for pyS3M

This module contains all magic numbers and constants used throughout the codebase
to improve maintainability and avoid hard-coded values scattered across files.

Created on August 29, 2025
@author: Claude Code
"""

# Standard logging for constants module
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray

# Shared array type alias used throughout the package
ImageArray = NDArray[np.float32]

from pyS3M.LoggingFramework import setup_logger
from pyS3M.CameraDefaults import CAMERAS as _CAMERAS

logger = setup_logger(__name__, console_output=False)
logger.info("Constants module loaded with standardised values")

# Camera and Image Processing Constants
DEFAULT_PIXEL_SIZE = 3.45  # μm - standard pixel size for many sCMOS cameras
DEFAULT_SMOOTHING_SIZE = 10  # pixels - size for uniform filtering
DEFAULT_N_BINS_FALLBACK = 10  # fallback number of bins for histograms

# Time Constants (seconds)
SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600

# Default Camera Parameters
DEFAULT_CAMERA_OFFSET = 100.0  # ADU
DEFAULT_CAMERA_VARIANCE = 8.0  # ADU²
DEFAULT_N_PHOTONS = 1000  # photons per molecule
DEFAULT_N_FRAMES = 100  # frames per experiment

# Numerical Constants
SMALL_NUMBER = 1e-100  # small number to avoid division by zero
KERNEL_WIDTH_MULTIPLIER = 10  # multiplier for blur kernel width calculation

# File I/O Constants
DEFAULT_ENCODING = "utf-8"
TIFF_FLOAT_BIT_DEPTH = float

# Error Handling Constants
DEFAULT_TIMEOUT_SECONDS = 30
MAX_RETRY_ATTEMPTS = 3


class CalibrationConstants:
    """Constants specific to camera calibration routines."""

    PIXEL_SIZE = DEFAULT_PIXEL_SIZE
    SMOOTHING_SIZE = DEFAULT_SMOOTHING_SIZE
    TIME_DISPLAY_THRESHOLD_MINUTES = SECONDS_PER_MINUTE
    TIME_DISPLAY_THRESHOLD_HOURS = SECONDS_PER_HOUR


class ProcessingConstants:
    """Constants for image and data processing."""

    N_BINS_FALLBACK = DEFAULT_N_BINS_FALLBACK
    SMALL_EPSILON = SMALL_NUMBER
    KERNEL_MULTIPLIER = KERNEL_WIDTH_MULTIPLIER


class DefaultParameters:
    """Default parameter values for various functions."""

    CAMERA_OFFSET = DEFAULT_CAMERA_OFFSET
    CAMERA_VARIANCE = DEFAULT_CAMERA_VARIANCE
    N_PHOTONS = DEFAULT_N_PHOTONS
    N_FRAMES = DEFAULT_N_FRAMES


class DriftConstants:
    """Constants for drift correction algorithms.

    Pixel sizes are derived from CameraDefaults so they stay in sync
    if camera specs are updated there.

    AIM distances are stored in nm; divide by pixel_size_nm to get camera pixels.
    """

    XIMEA_PIXEL_SIZE_NM: float = _CAMERAS["ximea"].pixel_size * 1e3  # 69.0 nm
    ZWO_PIXEL_SIZE_NM: float = _CAMERAS["zwo"].pixel_size * 1e3      # 71.5 nm

    DEFAULT_SEGMENTATION_FRAMES: int = 100
    FIDUCIAL_BOX_SIZE_NM: float = 900.0          # nm — detection box for fiducials
    AIM_INTERSECT_DISTANCE_NM: float = 20.0      # nm — AIM intersection distance
    AIM_ROI_RADIUS_NM: float = 60.0             # nm — AIM search region radius


class FilteringConstants:
    """Constants for single-molecule quality filtering.

    Sigma bounds are in nm; divide by pixel_size_nm to get camera pixels.
    """

    MAX_COLOUR_ERROR: float = 0.15              # fractional amplitude error
    MAX_LOCALISATION_ERROR_PX: float = 1.0      # pixels
    MIN_SIGMA_NM: float = 75.0                  # nm — minimum PSF sigma
    MAX_SIGMA_NM: float = 160.0                 # nm — maximum PSF sigma
    MAX_SIGMA_ERROR_NM: float = 40.0            # nm — maximum fitted sigma error
    MIN_PHOTONS: int = 500                      # photon count lower bound


class ResultColumns:
    """Column names for localization fit results.

    Defines standard column names for fitting results and their error estimates
    to ensure consistency across all fitting workflows.
    """

    # Standard fit parameter columns
    STANDARD_FIT_PARAMS = [
        "xc",  # X center coordinate
        "yc",  # Y center coordinate
        "s_x",  # X sigma (width)
        "s_y",  # Y sigma (width)
        "bg_B",  # Background (Blue channel)
        "bg_G",  # Background (Green channel)
        "bg_R",  # Background (Red channel)
        "A_B",  # Amplitude (Blue channel)
        "A_G",  # Amplitude (Green channel)
        "A_R",  # Amplitude (Red channel)
        "chi_sqr",  # Chi-squared goodness of fit
        "photons",  # Raw total amplitude (A_B+A_G+A_R before normalisation)
        "background_photons",  # Raw total background (bg_B+bg_G+bg_R before normalisation)
        "frame",  # Frame number
    ]

    # Standard fit error columns
    STANDARD_FIT_ERRORS = [
        "xc_err",
        "yc_err",
        "s_x_err",
        "s_y_err",
        "bg_B_err",
        "bg_G_err",
        "bg_R_err",
        "A_B_err",
        "A_G_err",
        "A_R_err",
    ]

    # Elliptical fit parameter columns (adds theta to STANDARD layout)
    ELLIPTICAL_FIT_PARAMS = [
        "xc",     # X center coordinate
        "yc",     # Y center coordinate
        "s_x",    # X sigma (width, rotated)
        "s_y",    # Y sigma (width, rotated)
        "theta",  # Rotation angle (radians)
        "bg_B",   # Background (Blue channel)
        "bg_G",   # Background (Green channel)
        "bg_R",   # Background (Red channel)
        "A_B",    # Amplitude (Blue channel)
        "A_G",    # Amplitude (Green channel)
        "A_R",    # Amplitude (Red channel)
        "chi_sqr",  # Chi-squared goodness of fit
        "frame",    # Frame number
    ]

    # Elliptical fit error columns
    ELLIPTICAL_FIT_ERRORS = [
        "xc_err",
        "yc_err",
        "s_x_err",
        "s_y_err",
        "theta_err",
        "bg_B_err",
        "bg_G_err",
        "bg_R_err",
        "A_B_err",
        "A_G_err",
        "A_R_err",
    ]

    # Circular fit parameter columns (STANDARD layout with s_x/s_y collapsed to one s)
    CIRCULAR_FIT_PARAMS = [
        "xc",     # X center coordinate
        "yc",     # Y center coordinate
        "s",      # Shared PSF sigma (width), sx and sy constrained equal
        "bg_B",   # Background (Blue channel)
        "bg_G",   # Background (Green channel)
        "bg_R",   # Background (Red channel)
        "A_B",    # Amplitude (Blue channel)
        "A_G",    # Amplitude (Green channel)
        "A_R",    # Amplitude (Red channel)
        "chi_sqr",  # Chi-squared goodness of fit
        "frame",    # Frame number
    ]

    # Circular fit error columns
    CIRCULAR_FIT_ERRORS = [
        "xc_err",
        "yc_err",
        "s_err",
        "bg_B_err",
        "bg_G_err",
        "bg_R_err",
        "A_B_err",
        "A_G_err",
        "A_R_err",
    ]

    @classmethod
    def get_all_columns(cls) -> list[str]:
        """Get all column names (parameters + errors).

        Returns:
            list: Combined list of all parameter and error column names
        """
        return cls.STANDARD_FIT_PARAMS + cls.STANDARD_FIT_ERRORS

    @classmethod
    def get_elliptical_columns(cls) -> list[str]:
        """Get all column names for elliptical fitting (parameters + errors).

        Returns:
            list: Combined list of elliptical parameter and error column names
        """
        return cls.ELLIPTICAL_FIT_PARAMS + cls.ELLIPTICAL_FIT_ERRORS

    @classmethod
    def get_circular_columns(cls) -> list[str]:
        """Get all column names for circular (single-sigma) fitting (parameters + errors).

        Returns:
            list: Combined list of circular parameter and error column names
        """
        return cls.CIRCULAR_FIT_PARAMS + cls.CIRCULAR_FIT_ERRORS


@dataclass
class FilteringCriteria:
    """Quality-filtering thresholds for single-molecule localisation data.

    Groups the 8 parameters that are re-passed identically on every call to
    ``filter_quality_localisations`` and ``extract_single_molecules_*``.
    Pass a single ``FilteringCriteria`` instance instead of the individual
    keyword arguments to reduce call-site verbosity.

    ``None`` fields are resolved at filter time using ``FilteringConstants``
    divided by the camera pixel size (same logic as the individual-param path).

    Example::

        filt = FilteringCriteria(min_photons=1000, max_colour_error=0.10)
        sm_db, sf_db = SM_E.extract_single_molecules_HDBSCAN(data, criteria=filt)
    """

    chi_val: Optional[float] = None     # None → no chi-squared filter applied
    max_localisation_error: float = FilteringConstants.MAX_LOCALISATION_ERROR_PX
    max_colour_error: float = FilteringConstants.MAX_COLOUR_ERROR
    min_sigma: Optional[float] = None   # px; None → MIN_SIGMA_NM / pixel_size_nm
    max_sigma: Optional[float] = None   # px; None → MAX_SIGMA_NM / pixel_size_nm
    max_sigma_error: Optional[float] = None  # px; None → MAX_SIGMA_ERROR_NM / pixel_size_nm
    min_photons: int = FilteringConstants.MIN_PHOTONS
    max_photons: Optional[float] = None


@dataclass
class AnalysisConfig:
    """Controls I/O and display behaviour for analysis functions.

    Pass to any analysis function to decouple figure display from computation.
    GUI code can set ``display=False`` and supply callbacks; headless scripts
    can set ``save_figures=True`` with an ``output_dir`` to write all outputs
    to disk without opening any windows.

    Example::

        cfg = AnalysisConfig(display=False, save_figures=True,
                             output_dir=Path('results/'), dpi=300)
        sr.fit_SM_data(..., config=cfg)

    Attributes:
        output_dir: Directory for saved figures/data.  ``None`` means current
            working directory (figures are only saved when ``save_figures`` is
            True).
        display: Show interactive figure windows.  Set ``False`` for
            GUI/server/headless runs.
        save_figures: Write figures to ``output_dir`` automatically.
        figure_format: File extension for saved figures (``'svg'``, ``'pdf'``,
            ``'png'``, …).
        dpi: Resolution used when saving raster figures.
        progress_callback: Optional callable ``(fraction: float, msg: str)``
            invoked during long operations so a GUI can update a progress bar.
        logging_callback: Optional callable ``(msg: str)`` that receives log
            messages instead of (or in addition to) the standard logger.
    """

    output_dir: Optional[Path] = None
    display: bool = True
    save_figures: bool = False
    figure_format: str = "svg"
    dpi: int = 300
    progress_callback: Optional[Callable[[float, str], None]] = None
    logging_callback: Optional[Callable[[str], None]] = None
