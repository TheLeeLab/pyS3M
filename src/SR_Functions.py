# -*- coding: utf-8 -*-
"""
This class contains functions pertaining to analysis of images,
relating to the bayerSMLM concept.
jsb92, 2024/01/02
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import gc
import multiprocessing
import sys
from concurrent import futures

import numpy as np
import pandas as pd
from numpy.typing import NDArray

import ruptures as rpt

sys.path.append(str(Path(__file__).parent))
import pyS3M.IOFunctions as IOFunctions
import pyS3M.HelperFunctions as HelperFunctions
import pyS3M.MaskFunctions as MaskFunctions
import pyS3M.ImageAnalysisFunctions as ImageAnalysisFunctions
from pyS3M.ImageAnalysisFunctions import FittingStrategy, FittingConstants
import pyS3M.SpotDetectionFunctions as SpotDetectionFunctions
from pyS3M.PlottingBase import PublicationPlotter
import pyS3M.sCMOSFunctions as sCMOSFunctions
from pyS3M.Constants import ResultColumns, AnalysisConfig
import logging
logger = logging.getLogger(__name__)


class SuperRes_Functions:
    """Super-resolution microscopy analysis functions.

    Provides functionality for super-resolution image reconstruction,
    localization processing, and analysis for Bayer filter SMLM systems.
    """

    def __init__(
        self,
        camera: str = "ximea",
        mosaic_unit: NDArray | None = None,
        pixel_size: float | None = None,
        io_functions: Any | None = None,
        helper_functions: Any | None = None,
        mask_functions: Any | None = None,
        image_analysis_functions: Any | None = None,
        spot_detection_functions: Any | None = None,
        plotter: Any | None = None,
        scmos: Any | None = None,
        config: AnalysisConfig | None = None,
    ) -> None:
        """Initialize SuperRes_Functions class.

        Args:
            camera: Camera model name used to set default ``pixel_size`` and
                ``mosaic_unit``.  Currently ``"ximea"`` (69 nm, BGGR) or
                ``"zwo"`` (78 nm, RGGB).  Overridden by explicit kwargs.
            mosaic_unit: Bayer mosaic pattern array.  If ``None``, taken from
                *camera* defaults.
            pixel_size: Physical pixel size in µm.  If ``None``, taken from
                *camera* defaults.
            io_functions: IO functions instance (default: creates new instance)
            helper_functions: Helper functions instance (default: creates new instance)
            mask_functions: Mask functions instance (default: creates new instance)
            image_analysis_functions: Image analysis functions instance (default: creates new instance)
            spot_detection_functions: Spot detection functions instance (default: creates new instance)
            plotter: Plotter instance (default: creates new instance)
            config: :class:`~Constants.AnalysisConfig` controlling display and
                I/O behaviour.  Defaults to ``AnalysisConfig()`` (interactive,
                no auto-save).
        """
        import pyS3M.CameraDefaults as CameraDefaults
        cam_cfg = CameraDefaults.get_camera_config(camera)
        self.pixel_size = pixel_size if pixel_size is not None else cam_cfg.pixel_size
        self.mosaic_unit = mosaic_unit if mosaic_unit is not None else cam_cfg.mosaic_unit

        # Dependency injection with sensible defaults
        self.io = (
            io_functions if io_functions is not None else IOFunctions.IO_Functions()
        )
        self.helper = (
            helper_functions
            if helper_functions is not None
            else HelperFunctions.Helper_Functions()
        )
        self.mask = (
            mask_functions
            if mask_functions is not None
            else MaskFunctions.Mask_Functions()
        )
        self.image_analysis = (
            image_analysis_functions
            if image_analysis_functions is not None
            else ImageAnalysisFunctions.Image_Analysis_Functions()
        )
        self.spot_detection = (
            spot_detection_functions
            if spot_detection_functions is not None
            else SpotDetectionFunctions.SpotDetection_Functions()
        )
        self.plotter = plotter if plotter is not None else PublicationPlotter()
        self.scmos = scmos if scmos is not None else sCMOSFunctions.sCMOS_Functions()
        self.config = config if config is not None else AnalysisConfig()

    def _postprocess_fit_results(
        self,
        fit_results_array: NDArray[np.float64],
        fit_errors_array: NDArray[np.float64],
        result_columns: list[str],
        planes: list[int] | NDArray[np.int_],
        width: int,
        height: int,
        quality_metrics: dict[str, NDArray] | None = None,
    ) -> pd.DataFrame:
        """Post-process fitting results into filtered DataFrame.

        Args:
            fit_results_array (np.ndarray): Raw fit results from parallel fitting
            fit_errors_array (np.ndarray): Raw fit errors from parallel fitting
            result_columns (list): Column names for DataFrame
            planes (list): Frame numbers for each punctum
            width (int): ROI width for filtering
            height (int): ROI height for filtering
            quality_metrics (dict): Optional dict of quality metric arrays per detection

        Returns:
            pd.DataFrame: Filtered and sorted fit results with quality metrics (if provided)
        """
        # Stack results and errors
        fit_tosave = np.hstack([fit_results_array, fit_errors_array])
        fit_results = pd.DataFrame(fit_tosave, columns=result_columns)

        # NEW: Add quality metrics BEFORE filtering (so they're filtered together with fits)
        if quality_metrics is not None and len(quality_metrics) > 0:
            # Add each quality metric as a column (with 'spot_' prefix to avoid name conflicts)
            for key, values in quality_metrics.items():
                if len(values) == len(fit_results):
                    fit_results[f'spot_{key}'] = values
                else:
                    logger.info(f"    WARNING: Skipping quality metric '{key}' due to length mismatch: {len(values)} != {len(fit_results)}")

        # Fix frame numbers: replace with offset plane values for continuous numbering
        if len(planes) == len(fit_results):
            fit_results["frame"] = planes

        # Sort by frame for consistent ordering in saved files
        fit_results = fit_results.sort_values("frame").reset_index(drop=True)

        # Apply filtering - this will automatically filter both fits AND quality metrics
        # Failed fits (NaN values) will be removed along with their quality metrics
        fit_results = self._filter_fit_results(fit_results, width, height)

        return fit_results

    def _filter_fit_results(self, fit_results: pd.DataFrame, width: int, height: int) -> pd.DataFrame:
        """Filter localization results based on physical and quality constraints.

        Applies multiple quality filters in a single pass for optimal performance:
        - Removes NaN values
        - Filters coordinates to be within image bounds
        - Filters sigma values to reasonable PSF range (0-3 pixels) -- independent
          s_x/s_y for STANDARD/ELLIPTICAL-style results, or a single shared "s" for
          CIRCULAR-style results, depending on which are present.
        - Ensures positive amplitudes and backgrounds -- per-channel (A_B/A_G/A_R,
          bg_B/bg_G/bg_R) for STANDARD/ELLIPTICAL-style results, or single-column
          (A, bg) for NOCOLOUR-style results, depending on which are present.

        Args:
            fit_results: Structured array of localization results
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            Filtered results with index reset
        """
        # Combine all filters into a single boolean mask for efficient filtering
        mask = (
            fit_results.notna().all(axis=1)
            & (fit_results["xc"] > 0)
            & (fit_results["xc"] < width)
            & (fit_results["yc"] > 0)
            & (fit_results["yc"] < height)
        )

        if all(c in fit_results.columns for c in ("s_x", "s_y")):
            mask &= (
                (fit_results["s_x"] > 0) & (fit_results["s_x"] < 3)
                & (fit_results["s_y"] > 0) & (fit_results["s_y"] < 3)
            )
        elif "s" in fit_results.columns:
            mask &= (fit_results["s"] > 0) & (fit_results["s"] < 3)

        if all(c in fit_results.columns for c in ("A_B", "A_G", "A_R")):
            mask &= (
                (fit_results["A_B"] > 0)
                & (fit_results["A_G"] > 0)
                & (fit_results["A_R"] > 0)
            )
        elif "A" in fit_results.columns:
            mask &= fit_results["A"] > 0

        if all(c in fit_results.columns for c in ("bg_B", "bg_G", "bg_R")):
            mask &= (
                (fit_results["bg_B"] > 0)
                & (fit_results["bg_G"] > 0)
                & (fit_results["bg_R"] > 0)
            )
        elif "bg" in fit_results.columns:
            mask &= fit_results["bg"] > 0

        return fit_results[mask].reset_index(drop=True)

    def _process_roi(
        self,
        raw_data,
        detected_puncta,
        i,
        width,
        height,
        ROI_size,
        smoothing_function,
        read_noise,
        masks,
        gain_map=1.0,
        offset_map=0.0,
        rqe=1.0,
        frame_offset=0,
        is_multi_frame=False,
    ):
        """
        Process a single detected ROI to extract photoelectron data, smoothed data, and weights.

        Args:
            raw_data (np.ndarray): Full raw camera image data in ADU (used for extraction)
            detected_puncta (np.ndarray): Array of detected puncta coordinates
            i (int): Index of current puncta to process
            width (int): Image width
            height (int): Image height
            ROI_size (int): Size of ROI to extract
            smoothing_function: Function for smoothing data
            read_noise: Read noise map or scalar
            masks (np.ndarray): Color masks
            gain_map (matrix or float): Gain map for ADU to photoelectron conversion
            offset_map (matrix or float): Offset map for ADU to photoelectron conversion
            rqe (matrix or float): Relative quantum efficiency map
            frame_offset (int): Frame offset for plane labeling
            is_multi_frame (bool): Whether data has multiple frames

        Returns:
            tuple or None: (photoelectron_roi, smoothed_roi, weights_roi, mask_roi, coords, plane)
                          Returns None if ROI is invalid (not square)
        """
        # detected_puncta stores [row, col, frame] from np.where()
        # row = y, col = x (confirmed by test_real_spot_detection.py)
        ycentre = detected_puncta[i, 0]  # First index is row (y)
        xcentre = detected_puncta[i, 1]  # Second index is col (x)
        frame = int(detected_puncta[i, 2]) if is_multi_frame else 0

        # Calculate ROI boundaries using helper function
        bounds = self.helper.calculate_roi_bounds(
            xcentre, ycentre, ROI_size, width, height
        )
        if bounds is None:
            return None
        xmin, xmax, ymin, ymax = bounds

        # Extract raw ROI for fitting (note: arrays are [row, col] = [y, x])
        if is_multi_frame:
            raw_roi = (
                raw_data[frame, ymin:ymax, xmin:xmax]
                if len(raw_data.shape) > 2
                else raw_data[ymin:ymax, xmin:xmax]
            )
        else:
            raw_roi = raw_data[ymin:ymax, xmin:xmax]

        # Verify ROI is actually square (sanity check)
        if raw_roi.shape[0] != raw_roi.shape[1]:
            import logging

            expected_size = xmax - xmin
            logging.warning(
                f"Non-square ROI extracted: {raw_roi.shape}, expected {expected_size}x{expected_size}"
            )
            logging.warning(
                f"  Boundaries: xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax}"
            )
            logging.warning(f"  Image dims: width={width}, height={height}")
            logging.warning(f"  Center: ({xcentre}, {ycentre}), ROI_size={ROI_size}")
            return None

        # Extract camera parameter ROIs for conversion (arrays are [y, x])
        gain_roi = (
            gain_map[ymin:ymax, xmin:xmax]
            if not isinstance(gain_map, (int, float))
            else gain_map
        )
        offset_roi = (
            offset_map[ymin:ymax, xmin:xmax]
            if not isinstance(offset_map, (int, float))
            else offset_map
        )
        rqe_roi = (
            rqe[ymin:ymax, xmin:xmax] if not isinstance(rqe, (int, float)) else rqe
        )

        # Convert raw ROI to photoelectrons
        photoelectron_roi = self.io.convert_to_photoelectrons(
            raw_roi, gain_map=gain_roi, offset_map=offset_roi, rqe=rqe_roi
        )

        # Extract read_noise ROI for weights calculation (arrays are [y, x])
        read_noise_roi = (
            read_noise[ymin:ymax, xmin:xmax]
            if not isinstance(read_noise, (int, float))
            else read_noise
        )

        # Generate smoothed and weights only for this ROI
        smoothed_roi = self.io.apply_smoothing(
            photoelectron_roi, smoothing_function, dtype="float32"
        )

        weights_roi = self.io.generate_weights(
            smoothed_roi, read_noise=read_noise_roi, dtype="float32"
        )

        # Extract mask ROI (note: masks are indexed as [row, col] = [y, x])
        mask_roi = masks[ymin:ymax, xmin:xmax, :]

        # Return processed data and metadata
        coords = (xmin, ymin)
        plane = frame + frame_offset

        return photoelectron_roi, smoothed_roi, weights_roi, mask_roi, coords, plane

    def _process_detected_puncta_batch(
        self,
        raw_data,
        detected_puncta,
        width,
        height,
        ROI_size,
        smoothing_function,
        read_noise,
        masks,
        gain_map=None,
        offset_map=None,
        rqe=None,
        frame_offset=0,
        is_multi_frame=False,
        quality_metrics=None,
    ):
        """Process a batch of detected puncta into fitting-ready ROIs.

        Consolidates the ROI processing loop pattern used across multiple methods.
        Iterates through detected puncta, processes each ROI, and accumulates results
        for batch fitting.

        Args:
            raw_data (np.ndarray): Raw image data for detection (2D or 3D)
            detected_puncta (list): List of detected punctum locations from spot detection
            width (int): Image width in pixels
            height (int): Image height in pixels
            ROI_size (int): Size of ROI box to extract around each punctum
            smoothing_function (callable): Function for smoothing photoelectron data
            read_noise (float or np.ndarray): Read noise map or scalar value
            masks (np.ndarray): Bayer mask array (width, height, 3)
            gain_map (float or np.ndarray, optional): Camera gain map or scalar
            offset_map (float or np.ndarray, optional): Camera offset map or scalar
            rqe (float or np.ndarray, optional): Relative QE map or scalar
            frame_offset (int, optional): Frame number offset for multi-file processing (default: 0)
            is_multi_frame (bool, optional): Whether processing multi-frame data (default: False)
            quality_metrics (dict, optional): Dict of quality metric arrays from spot detection.
                Will be filtered to match ROIs that pass processing (not too close to edges, etc.)

        Returns:
            tuple: (puncta_tofit, smoothed_puncta_tofit, masks_tofit, weights_tofit,
                   relative_coords, planes, filtered_quality_metrics)
                - puncta_tofit: List of photoelectron ROIs ready for fitting
                - smoothed_puncta_tofit: List of smoothed ROIs
                - masks_tofit: List of Bayer mask ROIs
                - weights_tofit: List of weight ROIs for fitting
                - relative_coords: List of (x, y) coordinates for each ROI
                - planes: List of frame numbers for each ROI
                - filtered_quality_metrics: Quality metrics filtered to match processed ROIs
                  (None if quality_metrics was None)

        Example:
            >>> # Single frame processing
            >>> results = self._process_detected_puncta_batch(
            ...     raw_data, detected_puncta, width, height, ROI_size,
            ...     smoothing_function, read_noise, masks
            ... )
            >>> puncta, smoothed, masks_roi, weights, coords, frames, qm = results

        """
        puncta_tofit = []
        smoothed_puncta_tofit = []
        masks_tofit = []
        weights_tofit = []
        relative_coords = []
        planes = []
        valid_indices = []  # Track which indices were successfully processed

        for i in np.arange(len(detected_puncta)):
            result = self._process_roi(
                raw_data,
                detected_puncta,
                i,
                width,
                height,
                ROI_size,
                smoothing_function,
                read_noise,
                masks,
                gain_map=gain_map,
                offset_map=offset_map,
                rqe=rqe,
                frame_offset=frame_offset,
                is_multi_frame=is_multi_frame,
            )

            if result is None:
                continue

            photoelectron_roi, smoothed_roi, weights_roi, mask_roi, coords, plane = (
                result
            )

            puncta_tofit.append(photoelectron_roi)
            smoothed_puncta_tofit.append(smoothed_roi)
            masks_tofit.append(mask_roi)
            weights_tofit.append(weights_roi)
            relative_coords.append(coords)
            planes.append(plane)
            valid_indices.append(i)  # Track that this index was successfully processed

        # Filter quality metrics to match successfully processed ROIs
        filtered_quality_metrics = None
        if quality_metrics is not None and len(quality_metrics) > 0:
            filtered_quality_metrics = {}
            # Convert valid_indices to numpy array for proper indexing
            # (dtype=int matters when valid_indices is empty — an empty list
            # would otherwise default to float64, which can't be used for
            # fancy indexing below)
            valid_indices_array = np.array(valid_indices, dtype=int)
            for key, values in quality_metrics.items():
                if len(values) == len(detected_puncta):
                    # Filter to only include metrics for successfully processed ROIs
                    filtered_quality_metrics[key] = values[valid_indices_array]
                else:
                    import logging
                    logging.warning(
                        f"Quality metric '{key}' length ({len(values)}) doesn't match "
                        f"detected_puncta length ({len(detected_puncta)}). Skipping."
                    )

        return (
            puncta_tofit,
            smoothed_puncta_tofit,
            masks_tofit,
            weights_tofit,
            relative_coords,
            planes,
            filtered_quality_metrics,
        )

    def example_spots_singleframe(
        self,
        image_folder: Path | str,
        image_type: str = ".tif",
        smoothing_function: Any | None = None,
        gain_map: NDArray[np.float32] | None = None,
        offset_map: NDArray[np.float32] | None = None,
        rqe: NDArray[np.float32] | None = None,
        read_noise: NDArray[np.float32] | None = None,
        variance: NDArray[np.float32] | None = None,
        pfa: float = 1e-3,
        mf_factor: float = 3.0,
        local_factor: float = 3.0,
        ROI_size: int = 16,
        peak_wavelength: float = 0.638,
        NA: float = 1.49,
        pixel_size: float | None = None,
        s: int = 5,
        sigma: float = 1.5,
        fraction_true: float = 0.0,
        use_variance_aware_demosaic: bool = True,
        frame_index: int = 0,
        n_frames_sum: int = 1,
    ) -> tuple[Any, Any]:
        """Example spot detection and fitting on a single frame with visualization.

        Demonstrates the complete workflow: spot detection, ROI extraction, fitting,
        and visualization with zoom insets on highest density region.

        Args:
            image_folder (str): Path to folder containing image files
            image_type (str): Image file extension (default: ".tif")
            smoothing_function: Function to smooth data
            gain_map (np.ndarray): 2D gain map
            offset_map (np.ndarray): 2D offset map
            rqe (np.ndarray): 2D RQE map
            read_noise (np.ndarray): 2D read noise map
            variance (np.ndarray): 2D variance map
            pfa (float): Probability of false alarm (default: 1e-3)
            mf_factor (float): Matched filter factor (default: 3.0)
            local_factor (float): Local threshold factor (default: 3.0)
            ROI_size (int): ROI extraction size (default: 12)
            peak_wavelength (float): PSF peak wavelength in µm (default: 0.638)
            NA (float): Numerical aperture (default: 1.49)
            pixel_size (float): Pixel size in µm (default: 0.069)
            s (int): Scatter plot marker size (default: 5)
            sigma (float): Gaussian sigma for detection (default: 1.5)
            fraction_true (float): Expected fraction of true spots (default: 0.0)
            use_variance_aware_demosaic (bool): Use variance-aware demosaicing (default: True)
            frame_index (int): Which frame to start from (default: 0)
            n_frames_sum (int): Number of frames to sum for spot detection (default: 1).
                When > 1, frames from frame_index to frame_index + n_frames_sum - 1 are summed
                before demosaicing. This improves SNR for weak signals. Note: variance is
                scaled accordingly (summed variance = n * single_frame_variance).

        Returns:
            tuple: (fig, axs) Matplotlib figure and axes with 2x2 subplot showing:
                - [0,0]: Detected spots on processed image (full field)
                - [0,1]: Fitted spots on raw image (full field)
                - [1,0]: Detected spots zoomed to highest density region
                - [1,1]: Fitted spots zoomed to highest density region
        """
        image_files = self.helper.file_search(image_folder, image_type, "")

        # Load ROI from metadata (with fallback to full image if not found)
        start_x, start_y, width, height = self.helper.load_metadata_roi(
            image_folder, self.io, use_fallback=True
        )

        if pixel_size is None:
            pixel_size = self.pixel_size
        file = image_files[0]
        puncta_tofit = []
        smoothed_puncta_tofit = []
        masks_tofit = []
        weights_tofit = []
        relative_coords = []

        # Load raw data for the requested frame(s)
        if n_frames_sum > 1:
            # Sum multiple frames for improved SNR
            total_frames = self.io.get_num_pages_in_TIF(file)
            end_frame = min(frame_index + n_frames_sum, total_frames)
            actual_frames_summed = end_frame - frame_index

            if actual_frames_summed < n_frames_sum:
                logger.warning(f"Warning: Only {actual_frames_summed} frames available from frame {frame_index}, requested {n_frames_sum}")

            logger.info(f"Summing {actual_frames_summed} frames (frames {frame_index} to {end_frame - 1}) for spot detection")

            # Load all frames at once and sum
            frames_to_load = list(range(frame_index, end_frame))
            raw_stack = self.io.read_tiff(file, dtype="float32", frame=frames_to_load)
            if raw_stack.ndim == 2:
                raw_stack = raw_stack[np.newaxis, :, :]
            raw_data = np.sum(raw_stack, axis=0)
            del raw_stack
        else:
            actual_frames_summed = 1
            raw_data = self.io.read_tiff(
                file,
                dtype="float32",
                frame=frame_index,
            )

        # Update width/height if they weren't set from metadata
        if width is None or height is None:
            height, width = raw_data.shape

        # Create default calibration maps if not provided (use full image size first)
        full_height, full_width = raw_data.shape
        if gain_map is None:
            gain_map = np.ones((full_height, full_width), dtype=np.float32)
        if offset_map is None:
            offset_map = np.zeros((full_height, full_width), dtype=np.float32)
        if rqe is None:
            rqe = np.ones((full_height, full_width), dtype=np.float32)
        if read_noise is None:
            read_noise = (
                np.ones((full_height, full_width), dtype=np.float32) * 1.6
            )  # Typical value
        if variance is None:
            variance = read_noise**2

        # Create masks for ROI
        masks = self.mask.get_stacked_masks(
            start_x, start_y, width, height, self.mosaic_unit
        )

        # Crop calibration maps to ROI
        cropped_maps = self.helper.crop_calibration_maps(
            {
                "gain_map": gain_map,
                "offset_map": offset_map,
                "read_noise": read_noise,
                "rqe": rqe,
                "variance": variance,
            },
            start_x,
            start_y,
            width,
            height,
        )
        gain_map = cropped_maps["gain_map"]
        offset_map = cropped_maps["offset_map"]
        read_noise = cropped_maps["read_noise"]
        rqe = cropped_maps["rqe"]
        variance = cropped_maps["variance"]

        # Scale calibration maps for summed frames
        # For summed data: offset scales linearly, variance scales linearly
        # (variance of sum of N iid variables = N * single variance)
        if actual_frames_summed > 1:
            offset_map = offset_map * actual_frames_summed
            variance = variance * actual_frames_summed

        # Demosaic the raw Bayer image for detection (default: skip demosaicing,
        # see _demosaic_image)
        image_to_analyse = self._demosaic_image(
            raw_data,
            use_variance_aware=use_variance_aware_demosaic,
            gain_map=gain_map,
            offset_map=offset_map,
            variance=variance,
        )

        detected_puncta = self.spot_detection.detect_puncta_in_image(
            image_to_analyse,
            pfa=pfa,
            variance=variance,
            wavelength=peak_wavelength,
            pixel_size=pixel_size,
            NA=NA,
            mf_factor=mf_factor,
            local_factor=local_factor,
            sigma=sigma,
            fraction_true=fraction_true,
        )

        # Extract detected ROIs and generate smoothed/weights only for ROIs (most memory efficient)
        (
            puncta_tofit,
            smoothed_puncta_tofit,
            masks_tofit,
            weights_tofit,
            relative_coords,
            _,
            _,  # filtered_quality_metrics (None - not using quality metrics in this method)
        ) = self._process_detected_puncta_batch(
            raw_data,
            detected_puncta,
            width,
            height,
            ROI_size,
            smoothing_function,
            read_noise,
            masks,
            gain_map=gain_map,
            offset_map=offset_map,
            rqe=rqe,
            frame_offset=0,
            is_multi_frame=False,
        )
        gc.collect()

        raw_image_for_fitting = self.io.convert_to_photoelectrons(
            raw_data, gain_map=gain_map, offset_map=offset_map, rqe=rqe
        )

        fit_results, _ = self.image_analysis.fit_puncta_parallel_method(
            puncta_tofit,
            smoothed_puncta_tofit,
            weights_tofit,
            relative_coords,
            list(np.zeros(len(puncta_tofit), dtype=int)),
            FittingStrategy.STANDARD_DATA,
            masks=masks_tofit,
        )

        columns = [
            "xc",
            "yc",
            "s_x",
            "s_y",
            "bg_B",
            "bg_G",
            "bg_R",
            "A_B",
            "A_G",
            "A_R",
            "chi_sqr",
            "frame",
        ]
        fit_results = pd.DataFrame(fit_results, columns=columns)
        # Create figure using PlottingBase for cleaner code
        import matplotlib.patches as patches
        from pyS3M.PlottingBase import PublicationPlotter

        plotter = PublicationPlotter()

        # A separate zoom-in row only adds information for a field of view
        # large enough that "zoomed in" actually means something; below
        # ~20x20 um the full-field panels already show essentially the
        # zoomed-in detail, so the zoom row is redundant clutter.
        fov_width_um = width * pixel_size
        fov_height_um = height * pixel_size
        show_zoom = not (fov_width_um < 20.0 and fov_height_um < 20.0)

        def _adaptive_scalebar_um(field_um: float, target_fraction: float = 0.3) -> float:
            """Pick a "nice" round scale-bar length proportional to the
            field it's drawn over, rather than a fixed value that can
            dwarf (or barely register against) a small/large field."""
            nice_values = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
            target = field_um * target_fraction
            candidates = [v for v in nice_values if v <= target]
            return candidates[-1] if candidates else nice_values[0]

        def _scalebar_label(length_um: float) -> str:
            return f"{length_um:g} μm"

        fig, axs = plotter.two_column_plot(
            nrows=2 if show_zoom else 1, ncols=2, height=8 if show_zoom else 4,
        )
        axs = np.atleast_2d(axs)

        full_field_um = min(fov_width_um, fov_height_um)
        full_field_bar_um = _adaptive_scalebar_um(full_field_um)

        # Calculate percentiles for consistent display
        vmin_processed = np.percentile(image_to_analyse, 1)
        vmax_processed = np.percentile(image_to_analyse, 99.8)

        # Plot the photoelectron image that was actually fitted
        vmin_raw = np.percentile(raw_image_for_fitting, 1)
        vmax_raw = np.percentile(raw_image_for_fitting, 99.8)

        # Find highest density region for zoom
        x_fit = fit_results["xc"].to_numpy()
        y_fit = fit_results["yc"].to_numpy()
        valid_mask = ~np.isnan(x_fit) & ~np.isnan(y_fit)
        x_valid = x_fit[valid_mask]
        y_valid = y_fit[valid_mask]

        if len(x_valid) > 0:
            density_hist, x_edges, y_edges = np.histogram2d(
                x_valid, y_valid, bins=50
            )
            max_density_idx = np.unravel_index(
                np.argmax(density_hist), density_hist.shape
            )
            # Center of zoom region
            center_x = (
                x_edges[max_density_idx[0]] + x_edges[max_density_idx[0] + 1]
            ) / 2
            center_y = (
                y_edges[max_density_idx[1]] + y_edges[max_density_idx[1] + 1]
            ) / 2
            # Zoom window (100x100 pixels)
            zoom_size = 50
            min_x, max_x = center_x - zoom_size, center_x + zoom_size
            min_y, max_y = center_y - zoom_size, center_y + zoom_size
        else:
            # Default zoom to center if no fits
            min_x, max_x = width // 2 - 50, width // 2 + 50
            min_y, max_y = height // 2 - 50, height // 2 + 50

        # Top row: Full field views
        # [0,0] Detected spots on processed image
        im = plotter.create_image_plot(
            axs[0, 0],
            image_to_analyse,
            vmin=vmin_processed,
            vmax=vmax_processed,
            cmap="gray",
        )
        # detected_puncta stores [row, col] = [y, x], but scatter needs (x, y)
        axs[0, 0].scatter(
            detected_puncta[:, 1],
            detected_puncta[:, 0],
            s=s,
            c="red",
            marker="o",
            alpha=0.25,
        )
        plotter.setup_axis(
            axs[0, 0],
            title="Detected Spots (Full Field)",
            grid=False,
            equal_aspect=True,
        )
        axs[0, 0].set_xticks([])
        axs[0, 0].set_yticks([])
        plotter.add_scalebar(
            axs[0, 0],
            pixelsize=pixel_size * 1000,
            length_nm=full_field_bar_um * 1000,
            label=_scalebar_label(full_field_bar_um),
            color="white",
        )

        # Add zoom rectangle
        if show_zoom:
            rect_00 = patches.Rectangle(
                (min_x, min_y),
                max_x - min_x,
                max_y - min_y,
                linewidth=1,
                edgecolor="cyan",
                facecolor="none",
            )
            axs[0, 0].add_patch(rect_00)

        # [0,1] Fitted spots on raw image
        im = plotter.create_image_plot(
            axs[0, 1], raw_image_for_fitting, vmin=vmin_raw, vmax=vmax_raw, cmap="gray"
        )
        axs[0, 1].scatter(x_fit, y_fit, s=s, c="lime", marker="o", alpha=0.25)
        plotter.setup_axis(
            axs[0, 1],
            title="Fitted Spots (Full Field)",
            grid=False,
            equal_aspect=True,
        )
        axs[0, 1].set_xticks([])
        axs[0, 1].set_yticks([])
        plotter.add_scalebar(
            axs[0, 1],
            pixelsize=pixel_size * 1000,
            length_nm=full_field_bar_um * 1000,
            label=_scalebar_label(full_field_bar_um),
            color="white",
        )

        # Add zoom rectangle
        if show_zoom:
            rect_01 = patches.Rectangle(
                (min_x, min_y),
                max_x - min_x,
                max_y - min_y,
                linewidth=1,
                edgecolor="cyan",
                facecolor="none",
            )
            axs[0, 1].add_patch(rect_01)

        # Bottom row: Zoomed views — only when show_zoom (below ~20x20 um
        # the full-field panels above are already the zoomed-in view)
        if show_zoom:
            zoom_field_um = (max_x - min_x) * pixel_size
            zoom_bar_um = _adaptive_scalebar_um(zoom_field_um)

            # [1,0] Detected spots zoomed
            im = plotter.create_image_plot(
                axs[1, 0],
                image_to_analyse,
                vmin=vmin_processed,
                vmax=vmax_processed,
                cmap="gray",
            )
            # detected_puncta stores [row, col] = [y, x], but scatter needs (x, y)
            axs[1, 0].scatter(
                detected_puncta[:, 1],
                detected_puncta[:, 0],
                s=s * 5,
                c="red",
                marker="o",
                alpha=0.25,
            )
            axs[1, 0].set_xlim(min_x, max_x)
            axs[1, 0].set_ylim(min_y, max_y)
            plotter.setup_axis(
                axs[1, 0],
                title="Detected Spots (Zoom)",
                grid=False,
                equal_aspect=True,
            )
            axs[1, 0].set_xticks([])
            axs[1, 0].set_yticks([])
            plotter.add_scalebar(
                axs[1, 0],
                pixelsize=pixel_size * 1000,
                length_nm=zoom_bar_um * 1000,
                label=_scalebar_label(zoom_bar_um),
                color="white",
            )

            # [1,1] Fitted spots zoomed
            im = plotter.create_image_plot(
                axs[1, 1], raw_image_for_fitting, vmin=vmin_raw, vmax=vmax_raw, cmap="gray"
            )
            axs[1, 1].scatter(x_fit, y_fit, s=s * 5, c="lime", marker="o", alpha=0.25)
            axs[1, 1].set_xlim(min_x, max_x)
            axs[1, 1].set_ylim(min_y, max_y)
            plotter.setup_axis(
                axs[1, 1],
                title="Fitted Spots (Zoom)",
                grid=False,
                equal_aspect=True,
            )
            axs[1, 1].set_xticks([])
            axs[1, 1].set_yticks([])
            plotter.add_scalebar(
                axs[1, 1],
                pixelsize=pixel_size * 1000,
                length_nm=zoom_bar_um * 1000,
                label=_scalebar_label(zoom_bar_um),
                color="white",
            )

            # Store drag config; handler is wired up by the GUI after the Qt canvas is created
            fig._zoom_drag_data = ([rect_00, rect_01], [axs[1, 0], axs[1, 1]], image_to_analyse.shape)

            engine = fig.get_layout_engine()
            if engine is not None and hasattr(engine, 'set'):
                # constrained/compressed — ask the engine to widen the row gap
                engine.set(h_pad=0.4, hspace=0.15)
            else:
                # no layout engine — direct grid spacing is safe
                fig.subplots_adjust(hspace=0.45, top=0.92)

        # Clean up
        del raw_data
        gc.collect()

        # Save and/or display according to AnalysisConfig
        save_path = None
        if self.config.save_figures and self.config.output_dir is not None:
            stem = "example_spots_singleframe"
            save_path = str(
                self.config.output_dir / f"{stem}.{self.config.figure_format}"
            )
        self.plotter.save_or_show(
            fig, save_path=save_path, show=self.config.display, dpi=self.config.dpi
        )

        return fig, axs

    def fit_FRET_data(
        self,
        image_folder: str,
        smoothing_function,
        gain_map: np.ndarray,
        offset_map: np.ndarray,
        rqe: np.ndarray,
        read_noise: np.ndarray,
        variance: np.ndarray,
        # Spot detection parameters
        n_frames_sum: int = 50,
        pfa: float = 1e-3,
        mf_factor: float = 3.0,
        local_factor: float = 3.0,
        sigma: float = 1.5,
        fraction_true: float = 0.0,
        # Change point detection parameters
        cp_model: str = "l2",
        cp_min_size: int = 5,
        cp_penalty_factor: float = 1.0,
        # Fitting parameters
        ROI_size: int = 16,
        peak_wavelength: float = 0.638,
        NA: float = 1.49,
        pixel_size: float = None,
        image_type: str = ".tif",
        use_variance_aware_demosaic: bool = True,
    ) -> None:
        """Complete FRET analysis pipeline with change point detection.

        Processes each image file independently:
        1. Sum first n_frames_sum frames for improved SNR spot detection
        2. Detect spots on summed (demosaiced) image
        3. Extract time traces (sum of ROI photoelectrons) for each spot
        4. Run PELT change point detection on traces
        5. Filter: keep only spots with >1 change point (real signal)
        6. Fit remaining spots at all frames up to final change point
        7. Save results to HDF5 database (one per input file)

        Args:
            image_folder: Path to folder containing image files
            smoothing_function: Function namespace with smoothing parameters
            gain_map: 2D gain calibration map
            offset_map: 2D offset calibration map
            rqe: 2D relative quantum efficiency map
            read_noise: 2D read noise map
            variance: 2D variance map
            n_frames_sum: Number of frames to sum for spot detection (default: 50)
            pfa: Probability of false alarm for detection (default: 1e-3)
            mf_factor: Matched filter factor (default: 3.0)
            local_factor: Local threshold factor (default: 3.0)
            sigma: Gaussian sigma for detection (default: 1.5)
            fraction_true: Expected fraction of true spots (default: 0.0)
            cp_model: Ruptures model for change point detection ("l2", "rbf", "normal")
            cp_min_size: Minimum segment size for change points (default: 5)
            cp_penalty_factor: Multiplier for automatic penalty (default: 1.0)
            ROI_size: Size of ROI for fitting (default: 16)
            peak_wavelength: PSF peak wavelength in um (default: 0.638)
            NA: Numerical aperture (default: 1.49)
            pixel_size: Pixel size in um (default: 0.069)
            image_type: Image file extension (default: ".tif")
            use_variance_aware_demosaic: Use variance-aware demosaicing (default: True)

        Saves:
            For each input file, saves {filename}.h5 with columns:
            puncta_id, frame, xc, yc, s_x, s_y, bg_B, bg_G, bg_R, A_B, A_G, A_R, chi_sqr
        """
        from tqdm import tqdm
        if pixel_size is None:
            pixel_size = self.pixel_size


        # Find image files
        image_files = self.helper.file_search(image_folder, image_type, "")
        if not image_files:
            raise ValueError(f"No {image_type} files found in {image_folder}")

        # Load ROI from metadata
        start_x, start_y, width, height = self.helper.load_metadata_roi(
            image_folder, self.io, use_fallback=True
        )

        # Get first file to determine dimensions if needed
        first_frame = self.io.read_tiff(image_files[0], dtype="float32", frame=0)
        full_height, full_width = first_frame.shape

        if width is None or height is None:
            height, width = full_height, full_width

        # Create masks for ROI (get_stacked_masks returns a 3D array)
        masks_stacked = self.mask.get_stacked_masks(
            start_x, start_y, width, height, self.mosaic_unit
        )

        # Crop calibration maps to ROI
        cropped_maps = self.helper.crop_calibration_maps(
            {
                "gain_map": gain_map,
                "offset_map": offset_map,
                "read_noise": read_noise,
                "rqe": rqe,
                "variance": variance,
            },
            start_x,
            start_y,
            width,
            height,
        )
        gain_map_crop = cropped_maps["gain_map"]
        offset_map_crop = cropped_maps["offset_map"]
        read_noise_crop = cropped_maps["read_noise"]
        rqe_crop = cropped_maps["rqe"]
        variance_crop = cropped_maps["variance"]

        # Process each file independently
        for FOVn, file in enumerate(image_files):
            logger.info(f"\nFile {FOVn+1}/{len(image_files)}: {Path(file).name}")

            fit_savename = file.split(".")[0] + ".h5"
            total_frames = self.io.get_num_pages_in_TIF(file)

            # PHASE 1: Spot detection on summed frames

            actual_frames_to_sum = min(n_frames_sum, total_frames)
            frames_to_load = list(range(actual_frames_to_sum))
            raw_stack = self.io.read_tiff(file, dtype="float32", frame=frames_to_load)
            if raw_stack.ndim == 2:
                raw_stack = raw_stack[np.newaxis, :, :]
            summed_data = np.sum(raw_stack, axis=0)
            del raw_stack

            # Scale variance for summed frames
            variance_summed = variance_crop * actual_frames_to_sum
            offset_summed = offset_map_crop * actual_frames_to_sum

            # Demosaic the summed image (default: skip demosaicing, see _demosaic_image)
            image_to_analyse = self._demosaic_image(
                summed_data,
                use_variance_aware=use_variance_aware_demosaic,
                gain_map=gain_map_crop,
                offset_map=offset_summed,
                variance=variance_summed,
            )

            # Detect spots
            detected_puncta = self.spot_detection.detect_puncta_in_image(
                image_to_analyse,
                pfa=pfa,
                variance=variance_summed,
                wavelength=peak_wavelength,
                pixel_size=pixel_size,
                NA=NA,
                mf_factor=mf_factor,
                local_factor=local_factor,
                sigma=sigma,
                fraction_true=fraction_true,
            )

            n_detected = len(detected_puncta)

            if n_detected == 0:
                logger.info(f"  No spots detected, skipping")
                del summed_data, image_to_analyse, variance_summed, offset_summed
                gc.collect()
                continue

            del summed_data, image_to_analyse, variance_summed, offset_summed
            gc.collect()

            # PHASE 2: Extract time traces for each spot
            traces = self._extract_roi_traces_single_file(
                file,
                detected_puncta,
                ROI_size,
                width,
                height,
                gain_map_crop,
                offset_map_crop,
                rqe_crop,
            )

            # PHASE 3: Change point detection
            frames_to_fit = self._find_change_points_parallel(
                traces,
                model=cp_model,
                min_size=cp_min_size,
                penalty_factor=cp_penalty_factor,
            )

            del traces
            gc.collect()

            if len(frames_to_fit) == 0:
                logger.info(f"  {n_detected} spots, 0 with change points, skipping")
                continue

            # PHASE 4: Fit spots at all frames up to change point
            total_rois = sum(len(frames) for frames in frames_to_fit.values())
            logger.info(f"  {n_detected} spots, {len(frames_to_fit)} with CPs, {total_rois} ROIs to fit")

            # Accumulate ROIs for fitting
            puncta_tofit = []
            smoothed_puncta_tofit = []
            masks_tofit = []
            weights_tofit = []
            relative_coords = []
            puncta_ids = []
            frame_indices = []

            # Pre-compute ROI bounds for puncta that passed change point filter
            puncta_bounds = {}
            for puncta_idx in frames_to_fit.keys():
                ycentre = int(detected_puncta[puncta_idx, 0])
                xcentre = int(detected_puncta[puncta_idx, 1])
                bounds = self.helper.calculate_roi_bounds(xcentre, ycentre, ROI_size, width, height)
                puncta_bounds[puncta_idx] = bounds

            # Get all frames needed
            all_frames_needed = set()
            for puncta_idx, frames_array in frames_to_fit.items():
                all_frames_needed.update(frames_array.tolist())
            all_frames_needed = sorted(all_frames_needed)

            if len(all_frames_needed) == 0:
                continue

            max_frame = max(all_frames_needed)

            # Process in chunks
            chunk_size = 500
            for chunk_start in tqdm(range(0, max_frame + 1, chunk_size), desc="Loading frames"):
                chunk_end = min(chunk_start + chunk_size, max_frame + 1)

                # Find frames in this chunk that are actually needed
                frames_in_chunk = [f for f in range(chunk_start, chunk_end) if f in all_frames_needed]
                if not frames_in_chunk:
                    continue

                # Load the chunk
                local_frames = list(range(chunk_start, min(chunk_end, total_frames)))
                if not local_frames:
                    continue

                raw_data = self.io.read_tiff(file, dtype="float32", frame=local_frames)
                if raw_data.ndim == 2:
                    raw_data = raw_data[np.newaxis, :, :]

                # Convert to photoelectrons
                photoelectrons = self.io.convert_to_photoelectrons(
                    raw_data, gain_map=gain_map_crop, offset_map=offset_map_crop, rqe=rqe_crop
                )

                # Apply smoothing (copy args to avoid mutating the shared namespace)
                _smargs = dict(smoothing_function.args)
                _smargs[smoothing_function.data_arg] = photoelectrons
                smoothed = smoothing_function.smoothing_function(**_smargs)

                # Compute weights
                weights_data = 1.0 / (read_noise_crop**2 + np.maximum(photoelectrons, 0) / gain_map_crop)

                # Extract ROIs for each puncta at frames in this chunk
                for puncta_idx, frames_array in frames_to_fit.items():
                    bounds = puncta_bounds[puncta_idx]
                    if bounds is None:
                        continue
                    xmin, xmax, ymin, ymax = bounds

                    for frame in frames_array:
                        if frame < chunk_start or frame >= chunk_end:
                            continue

                        local_idx = frame - chunk_start
                        if local_idx < 0 or local_idx >= photoelectrons.shape[0]:
                            continue

                        puncta_tofit.append(photoelectrons[local_idx, ymin:ymax, xmin:xmax].copy())
                        smoothed_puncta_tofit.append(smoothed[local_idx, ymin:ymax, xmin:xmax].copy())
                        masks_tofit.append(masks_stacked[ymin:ymax, xmin:xmax, :].copy())
                        weights_tofit.append(weights_data[local_idx, ymin:ymax, xmin:xmax].copy())
                        relative_coords.append((xmin, ymin))
                        puncta_ids.append(puncta_idx)
                        frame_indices.append(frame)

                del raw_data, photoelectrons, smoothed, weights_data
                gc.collect()

            logger.info(f"  Accumulated {len(puncta_tofit)} ROIs for fitting")

            if len(puncta_tofit) == 0:
                continue

            fit_results, fit_errors = self.image_analysis.fit_puncta_parallel_method(
                puncta_tofit,
                smoothed_puncta_tofit,
                weights_tofit,
                relative_coords,
                list(range(len(puncta_tofit))),
                FittingStrategy.STANDARD_DATA,
                masks=masks_tofit,
            )

            # Build result DataFrame
            result_columns = [
                "xc", "yc", "s_x", "s_y",
                "bg_B", "bg_G", "bg_R",
                "A_B", "A_G", "A_R",
                "chi_sqr",
            ]
            error_columns = [f"{col}_err" for col in result_columns[:-1]]

            fit_df = pd.DataFrame(fit_results, columns=result_columns + ["_dummy"])
            fit_df = fit_df.drop(columns=["_dummy"])

            if fit_errors is not None and len(fit_errors) > 0:
                err_df = pd.DataFrame(fit_errors, columns=error_columns)
                fit_df = pd.concat([fit_df, err_df], axis=1)

            # Add puncta_id and frame columns
            fit_df["puncta_id"] = puncta_ids
            fit_df["frame"] = frame_indices

            # Filter results
            fit_df = self._filter_fit_results(fit_df, width, height)

            # Save to HDF5 database
            self.io.write_h5_database(fit_df, fit_savename, append=False)
            logger.info(f"  Saved {len(fit_df)} fits to {Path(fit_savename).name}")

            # Cleanup
            del (
                puncta_tofit, smoothed_puncta_tofit, masks_tofit, weights_tofit,
                relative_coords, puncta_ids, frame_indices, fit_df, detected_puncta
            )
            gc.collect()

        return

    def fit_QD_data(
        self,
        image_folder: str,
        smoothing_function,
        gain_map: np.ndarray,
        offset_map: np.ndarray,
        rqe: np.ndarray,
        read_noise: np.ndarray,
        variance: np.ndarray,
        # Spot detection parameters
        n_frames_sum: int = 50,
        pfa: float = 1e-3,
        mf_factor: float = 3.0,
        local_factor: float = 3.0,
        sigma: float = 1.5,
        fraction_true: float = 0.0,
        # Fitting parameters
        ROI_size: int = 16,
        peak_wavelength: float = 0.638,
        NA: float = 1.49,
        pixel_size: float = None,
        image_type: str = ".tif",
        use_variance_aware_demosaic: bool = True,
        chunk_size: int = 500,
    ) -> None:
        """QDot analysis pipeline: detect spots on summed frames, fit every frame.

        Unlike fit_FRET_data, no change-point detection is applied.  Every frame
        is fitted at every detected spot location so that the full photon-count
        time series is available for downstream blinking / spectral analysis.

        Processes each image file independently:
        1. Sum first n_frames_sum frames for improved SNR spot detection
        2. Detect spots on summed (demosaiced) image
        3. Fit ALL frames at ALL detected spot locations (in memory-bounded chunks)
        4. Save results to HDF5 database (one per input file, chunks appended)

        Args:
            image_folder: Path to folder containing image files
            smoothing_function: Function namespace with smoothing parameters
            gain_map: 2D gain calibration map
            offset_map: 2D offset calibration map
            rqe: 2D relative quantum efficiency map
            read_noise: 2D read noise map
            variance: 2D variance map
            n_frames_sum: Number of frames to sum for spot detection (default: 50)
            pfa: Probability of false alarm for detection (default: 1e-3)
            mf_factor: Matched filter factor (default: 3.0)
            local_factor: Local threshold factor (default: 3.0)
            sigma: Gaussian sigma for detection (default: 1.5)
            fraction_true: Expected fraction of true spots (default: 0.0)
            ROI_size: Size of ROI for fitting (default: 16)
            peak_wavelength: PSF peak wavelength in um (default: 0.638)
            NA: Numerical aperture (default: 1.49)
            pixel_size: Pixel size in um (default: 0.069)
            image_type: Image file extension (default: ".tif")
            use_variance_aware_demosaic: Use variance-aware demosaicing (default: True)
            chunk_size: Number of frames to load and fit at once (default: 500)

        Saves:
            For each input file, saves {filename}.h5 with columns:
            puncta_id, frame, xc, yc, s_x, s_y, bg_B, bg_G, bg_R, A_B, A_G, A_R, chi_sqr
        """
        from tqdm import tqdm
        if pixel_size is None:
            pixel_size = self.pixel_size


        # Find image files
        image_files = self.helper.file_search(image_folder, image_type, "")
        if not image_files:
            raise ValueError(f"No {image_type} files found in {image_folder}")

        # Load ROI from metadata
        start_x, start_y, width, height = self.helper.load_metadata_roi(
            image_folder, self.io, use_fallback=True
        )

        # Get first file to determine dimensions if needed
        first_frame = self.io.read_tiff(image_files[0], dtype="float32", frame=0)
        full_height, full_width = first_frame.shape

        if width is None or height is None:
            height, width = full_height, full_width

        # Create masks for ROI
        masks_stacked = self.mask.get_stacked_masks(
            start_x, start_y, width, height, self.mosaic_unit
        )

        # Crop calibration maps to ROI
        cropped_maps = self.helper.crop_calibration_maps(
            {
                "gain_map": gain_map,
                "offset_map": offset_map,
                "read_noise": read_noise,
                "rqe": rqe,
                "variance": variance,
            },
            start_x,
            start_y,
            width,
            height,
        )
        gain_map_crop = cropped_maps["gain_map"]
        offset_map_crop = cropped_maps["offset_map"]
        read_noise_crop = cropped_maps["read_noise"]
        rqe_crop = cropped_maps["rqe"]
        variance_crop = cropped_maps["variance"]

        result_columns = [
            "xc", "yc", "s_x", "s_y",
            "bg_B", "bg_G", "bg_R",
            "A_B", "A_G", "A_R",
            "chi_sqr",
        ]
        error_columns = [f"{col}_err" for col in result_columns[:-1]]

        # Process each file independently
        for FOVn, file in enumerate(image_files):
            logger.info(f"\nProcessing file {FOVn+1}/{len(image_files)}: {Path(file).name}")

            fit_savename = file.split(".")[0] + ".h5"
            total_frames = self.io.get_num_pages_in_TIF(file)

            # PHASE 1: Spot detection on summed frames
            actual_frames_to_sum = min(n_frames_sum, total_frames)
            frames_to_load = list(range(actual_frames_to_sum))
            raw_stack = self.io.read_tiff(file, dtype="float32", frame=frames_to_load)
            if raw_stack.ndim == 2:
                raw_stack = raw_stack[np.newaxis, :, :]
            summed_data = np.sum(raw_stack, axis=0)
            del raw_stack

            variance_summed = variance_crop * actual_frames_to_sum
            offset_summed = offset_map_crop * actual_frames_to_sum

            image_to_analyse = self._demosaic_image(
                summed_data,
                use_variance_aware=use_variance_aware_demosaic,
                gain_map=gain_map_crop,
                offset_map=offset_summed,
                variance=variance_summed,
            )

            detected_puncta = self.spot_detection.detect_puncta_in_image(
                image_to_analyse,
                pfa=pfa,
                variance=variance_summed,
                wavelength=peak_wavelength,
                pixel_size=pixel_size,
                NA=NA,
                mf_factor=mf_factor,
                local_factor=local_factor,
                sigma=sigma,
                fraction_true=fraction_true,
            )

            n_detected = len(detected_puncta)

            del summed_data, image_to_analyse, variance_summed, offset_summed
            gc.collect()

            if n_detected == 0:
                logger.info(f"  No spots detected, skipping")
                continue

            logger.info(f"  {n_detected} spots detected; fitting all {total_frames} frames")

            # Pre-compute ROI bounds for all detected spots
            puncta_bounds = {}
            for puncta_idx in range(n_detected):
                ycentre = int(detected_puncta[puncta_idx, 0])
                xcentre = int(detected_puncta[puncta_idx, 1])
                bounds = self.helper.calculate_roi_bounds(xcentre, ycentre, ROI_size, width, height)
                puncta_bounds[puncta_idx] = bounds

            # PHASE 2: Fit every frame at every detected spot location
            first_save = True
            for chunk_start in tqdm(range(0, total_frames, chunk_size), desc="Fitting frames"):
                chunk_end = min(chunk_start + chunk_size, total_frames)
                chunk_frames = list(range(chunk_start, chunk_end))

                # Load chunk
                raw_data = self.io.read_tiff(file, dtype="float32", frame=chunk_frames)
                if raw_data.ndim == 2:
                    raw_data = raw_data[np.newaxis, :, :]

                # Convert to photoelectrons
                photoelectrons = self.io.convert_to_photoelectrons(
                    raw_data, gain_map=gain_map_crop, offset_map=offset_map_crop, rqe=rqe_crop
                )

                # Apply smoothing (copy args to avoid mutating the shared namespace)
                _smargs = dict(smoothing_function.args)
                _smargs[smoothing_function.data_arg] = photoelectrons
                smoothed = smoothing_function.smoothing_function(**_smargs)

                # Compute weights
                weights_data = 1.0 / (read_noise_crop**2 + np.maximum(photoelectrons, 0) / gain_map_crop)

                # Build ROI lists for this chunk
                puncta_tofit = []
                smoothed_puncta_tofit = []
                masks_tofit = []
                weights_tofit = []
                relative_coords = []
                puncta_ids = []
                frame_indices = []

                for local_idx, frame in enumerate(chunk_frames):
                    for puncta_idx in range(n_detected):
                        bounds = puncta_bounds[puncta_idx]
                        if bounds is None:
                            continue
                        xmin, xmax, ymin, ymax = bounds

                        puncta_tofit.append(photoelectrons[local_idx, ymin:ymax, xmin:xmax].copy())
                        smoothed_puncta_tofit.append(smoothed[local_idx, ymin:ymax, xmin:xmax].copy())
                        masks_tofit.append(masks_stacked[ymin:ymax, xmin:xmax, :].copy())
                        weights_tofit.append(weights_data[local_idx, ymin:ymax, xmin:xmax].copy())
                        relative_coords.append((xmin, ymin))
                        puncta_ids.append(puncta_idx)
                        frame_indices.append(frame)

                del raw_data, photoelectrons, smoothed, weights_data
                gc.collect()

                if len(puncta_tofit) == 0:
                    continue

                fit_results, fit_errors = self.image_analysis.fit_puncta_parallel_method(
                    puncta_tofit,
                    smoothed_puncta_tofit,
                    weights_tofit,
                    relative_coords,
                    list(range(len(puncta_tofit))),
                    FittingStrategy.STANDARD_DATA,
                    masks=masks_tofit,
                )

                # Build result DataFrame
                fit_df = pd.DataFrame(fit_results, columns=result_columns + ["_dummy"])
                fit_df = fit_df.drop(columns=["_dummy"])

                if fit_errors is not None and len(fit_errors) > 0:
                    err_df = pd.DataFrame(fit_errors, columns=error_columns)
                    fit_df = pd.concat([fit_df, err_df], axis=1)

                fit_df["puncta_id"] = puncta_ids
                fit_df["frame"] = frame_indices

                # Filter results
                fit_df = self._filter_fit_results(fit_df, width, height)

                # Append to HDF5 (first chunk creates the file, subsequent chunks append)
                self.io.write_h5_database(fit_df, fit_savename, append=(not first_save))
                first_save = False

                del (
                    puncta_tofit, smoothed_puncta_tofit, masks_tofit, weights_tofit,
                    relative_coords, puncta_ids, frame_indices, fit_df
                )
                gc.collect()

            logger.info(f"  Saved fits to {Path(fit_savename).name}")

        return

    def _extract_roi_traces_single_file(
        self,
        file: str,
        detected_puncta: np.ndarray,
        ROI_size: int,
        width: int,
        height: int,
        gain_map: np.ndarray,
        offset_map: np.ndarray,
        rqe: np.ndarray,
        chunk_size: int = 500,
    ) -> np.ndarray:
        """Extract time traces for each detected puncta from a single file.

        Args:
            file: Path to image file
            detected_puncta: Array of shape (n_puncta, 2) with [row, col] coordinates
            ROI_size: Size of ROI to extract around each puncta
            width, height: Image dimensions
            gain_map, offset_map, rqe: Calibration maps (cropped to ROI)
            chunk_size: Number of frames to load at once

        Returns:
            Array of shape (n_puncta, n_frames) containing summed photoelectrons
            in each ROI at each frame.
        """
        total_frames = self.io.get_num_pages_in_TIF(file)
        n_puncta = len(detected_puncta)

        # Pre-compute ROI bounds for each puncta
        roi_bounds = []
        for i in range(n_puncta):
            ycentre = int(detected_puncta[i, 0])
            xcentre = int(detected_puncta[i, 1])
            bounds = self.helper.calculate_roi_bounds(xcentre, ycentre, ROI_size, width, height)
            roi_bounds.append(bounds)

        # Initialize output array
        traces = np.zeros((n_puncta, total_frames), dtype=np.float32)

        # Process in chunks
        for chunk_start in range(0, total_frames, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_frames)
            chunk_frames = list(range(chunk_start, chunk_end))
            n_chunk = len(chunk_frames)

            # Load chunk
            raw_data = self.io.read_tiff(file, dtype="float32", frame=chunk_frames)
            if raw_data.ndim == 2:
                raw_data = raw_data[np.newaxis, :, :]

            # Convert to photoelectrons
            photoelectrons = self.io.convert_to_photoelectrons(
                raw_data, gain_map=gain_map, offset_map=offset_map, rqe=rqe
            )

            # Extract ROI sums for each puncta
            for puncta_idx, bounds in enumerate(roi_bounds):
                if bounds is None:
                    continue
                xmin, xmax, ymin, ymax = bounds
                roi_data = photoelectrons[:, ymin:ymax, xmin:xmax]
                roi_sums = np.sum(roi_data, axis=(1, 2))
                traces[puncta_idx, chunk_start:chunk_start + n_chunk] = roi_sums

            del raw_data, photoelectrons
            gc.collect()

        return traces

    def fit_SM_data(
        self,
        image_folder: Path | str,
        smoothing_function: Any,
        gain_map: NDArray[np.float32],
        offset_map: NDArray[np.float32],
        rqe: NDArray[np.float32],
        read_noise: NDArray[np.float32],
        variance: NDArray[np.float32],
        pfa: float = 1e-3,
        ROI_size: int = 16,
        peak_wavelength: float = 0.638,
        NA: float = 1.49,
        pixel_size: float | None = None,
        sigma: float = 1.5,
        fraction_true: float = 0.0,
        image_type: str = ".tif",
        use_variance_aware_demosaic: bool = True,
        use_elliptical: bool = False,
        use_circular: bool = False,
        combined_output: bool = False,
    ) -> None:
        """Single-molecule data fitting function.

        Analyzes single-molecule localization microscopy data by detecting puncta
        and fitting them with Gaussian models to extract precise positions and photon counts.

        Args:
            image_folder (str): Path to folder containing image files
            smoothing_function (type): function to smooth data
            gain_map (np.2darray): 2darray of gain map
            offset_map (np.2darray): 2darray of offset map
            rqe (np.2darray): 2d array of RQE
            read_noise (np.2darray): 2d array of read noise
            masks (dict): dict of colour masks
            peak_wavelength (float): peak wavelength of PSF
            sigma (float): sigma parameter for spot detection (default: 1.5)
            fraction_true (float): fraction of true spots expected (default: 0.0)
            image_type (str): image string end
            use_variance_aware_demosaic (bool): Whether to use variance-aware demosaicing for spot detection.
                If True (default), uses gain, offset, and variance maps to create robust photoelectron
                images that suppress hot pixels. If False, uses standard grayscale demosaicing.
            use_elliptical (bool): Use the 11-parameter rotated elliptical Gaussian model
                (FittingStrategy.ELLIPTICAL, adds xc,yc,s_x,s_y,theta) instead of the default
                axis-aligned STANDARD_DATA model. Default False (unchanged behaviour). See
                fit_tracking_data for the same option on the tracking-data pipeline.
            use_circular (bool): Use FittingStrategy.CIRCULAR (STANDARD_DATA with s_x/s_y
                constrained to one shared width, "s") instead of STANDARD_DATA. Mutually
                exclusive with use_elliptical.
            combined_output (bool): If True, treat every TIFF in image_folder as one
                continuous field of view: frame numbers accumulate across files and all
                results are appended to a single image_folder/Localisations.h5, matching
                fit_imaging_data's behaviour. Use for a FOV split across multiple TIFFs
                (e.g. Micro-Manager's automatic multi-part split once a recording exceeds
                its per-file size limit). If False (default), each TIFF is its own FOV and
                gets its own HDF5 file alongside it — use when image_folder genuinely holds
                multiple separate FOVs.

        Returns:
            bayer_image (np.ndarray): colour images imaged through the bayer filter supplied
        """
        return self._fit_files(
            image_folder, smoothing_function, gain_map, offset_map, rqe, read_noise, variance,
            pfa=pfa, ROI_size=ROI_size, peak_wavelength=peak_wavelength, NA=NA,
            pixel_size=pixel_size, sigma=sigma, fraction_true=fraction_true,
            image_type=image_type, use_variance_aware_demosaic=use_variance_aware_demosaic,
            use_elliptical=use_elliptical, use_circular=use_circular,
            accumulate_frame_numbers=combined_output, combined_output=combined_output,
        )

    def _fit_files(
        self,
        image_folder: Path | str,
        smoothing_function: Any,
        gain_map: NDArray[np.float32],
        offset_map: NDArray[np.float32],
        rqe: NDArray[np.float32],
        read_noise: NDArray[np.float32],
        variance: NDArray[np.float32],
        pfa: float = 1e-3,
        ROI_size: int = 16,
        peak_wavelength: float = 0.638,
        NA: float = 1.49,
        pixel_size: float | None = None,
        sigma: float = 1.5,
        fraction_true: float = 0.0,
        image_type: str = ".tif",
        use_variance_aware_demosaic: bool = True,
        use_elliptical: bool = False,
        use_circular: bool = False,
        accumulate_frame_numbers: bool = False,
        combined_output: bool = False,
    ) -> None:
        """Unified file-fitting pipeline shared by fit_SM_data and fit_imaging_data.

        Args:
            accumulate_frame_numbers: If True, frame indices accumulate across files
                (imaging mode). If False, each file resets to frame 0 (SM mode).
            combined_output: If True, all files append to one shared HDF5
                (Localisations.h5 in image_folder). If False, each file gets its
                own HDF5 alongside the TIFF.
            use_elliptical: Use FittingStrategy.ELLIPTICAL (rotated Gaussian, adds theta)
                instead of STANDARD_DATA. See fit_tracking_data for the same option.
            use_circular: Use FittingStrategy.CIRCULAR (STANDARD_DATA with s_x/s_y
                constrained to one shared width) instead of STANDARD_DATA. Mutually
                exclusive with use_elliptical.
            All other args: see fit_SM_data / fit_imaging_data.
        """
        if use_elliptical and use_circular:
            raise ValueError("use_elliptical and use_circular are mutually exclusive")

        if use_elliptical:
            strategy = FittingStrategy.ELLIPTICAL
            result_params = ResultColumns.get_elliptical_columns()
        elif use_circular:
            strategy = FittingStrategy.CIRCULAR
            result_params = ResultColumns.get_circular_columns()
        else:
            strategy = FittingStrategy.STANDARD_DATA
            result_params = ResultColumns.get_all_columns()
        if pixel_size is None:
            pixel_size = self.pixel_size

        image_files = self.helper.file_search(image_folder, image_type, "")
        start_x, start_y, width, height = self.helper.load_metadata_roi(
            image_folder, self.io, use_fallback=False
        )

        masks = self.mask.get_stacked_masks(
            start_x, start_y, width, height, self.mosaic_unit
        )
        cropped_maps = self.helper.crop_calibration_maps(
            {
                "gain_map": gain_map,
                "offset_map": offset_map,
                "read_noise": read_noise,
                "rqe": rqe,
                "variance": variance,
            },
            start_x,
            start_y,
            width,
            height,
        )
        gain_map = cropped_maps["gain_map"]
        offset_map = cropped_maps["offset_map"]
        read_noise = cropped_maps["read_noise"]
        rqe = cropped_maps["rqe"]
        variance = cropped_maps["variance"]

        if combined_output:
            fit_savename = Path(image_folder) / "Localisations.h5"

        total_frames = 0
        for FOVn, file in enumerate(image_files):
            if not combined_output:
                fit_savename = file.split(".")[0] + ".h5"

            file_frames = self.io.get_num_pages_in_TIF(file)

            chunk_size = 1000
            all_puncta_tofit = []
            all_smoothed_puncta_tofit = []
            all_masks_tofit = []
            all_weights_tofit = []
            all_relative_coords = []
            all_planes = []
            all_quality_metrics = []

            logger.info(f"Processing file {FOVn+1}/{len(image_files)}: {file_frames} frames in chunks of {chunk_size}")

            for chunk_start in range(0, file_frames, chunk_size):
                chunk_end = min(chunk_start + chunk_size, file_frames)
                chunk_frames = list(range(chunk_start, chunk_end))

                logger.info(f"  Processing chunk: frames {chunk_start}-{chunk_end-1}")

                raw_data = self.io.read_tiff(file, dtype="float32", frame=chunk_frames)
                if raw_data.ndim == 2:
                    raw_data = raw_data[np.newaxis, :, :]

                image_to_analyse = self._demosaic_image(
                    raw_data,
                    use_variance_aware=use_variance_aware_demosaic,
                    gain_map=gain_map,
                    offset_map=offset_map,
                    variance=variance,
                )

                detected_puncta, quality_metrics = self.spot_detection.detect_puncta_in_stack_parallel(
                    image_to_analyse,
                    pfa=pfa,
                    variance=variance,
                    wavelength=peak_wavelength,
                    pixel_size=pixel_size,
                    NA=NA,
                    sigma=sigma,
                    fraction_true=fraction_true,
                    return_quality=True,
                )

                frame_offset = (total_frames if accumulate_frame_numbers else 0) + chunk_start

                (
                    chunk_puncta,
                    chunk_smoothed,
                    chunk_masks,
                    chunk_weights,
                    chunk_coords,
                    chunk_planes,
                    filtered_quality_metrics,
                ) = self._process_detected_puncta_batch(
                    raw_data,
                    detected_puncta,
                    width,
                    height,
                    ROI_size,
                    smoothing_function,
                    read_noise,
                    masks,
                    gain_map=gain_map,
                    offset_map=offset_map,
                    rqe=rqe,
                    frame_offset=frame_offset,
                    is_multi_frame=True,
                    quality_metrics=quality_metrics,
                )

                all_puncta_tofit.extend(chunk_puncta)
                all_smoothed_puncta_tofit.extend(chunk_smoothed)
                all_masks_tofit.extend(chunk_masks)
                all_weights_tofit.extend(chunk_weights)
                all_relative_coords.extend(chunk_coords)
                all_planes.extend(chunk_planes)
                if filtered_quality_metrics is not None:
                    all_quality_metrics.append(filtered_quality_metrics)

                del raw_data, detected_puncta, quality_metrics, image_to_analyse
                gc.collect()

            logger.info(f"  Found {len(all_puncta_tofit)} puncta across all chunks")

            combined_quality_metrics = {}
            if all_quality_metrics:
                for quality_dict in all_quality_metrics:
                    if quality_dict:
                        for key in quality_dict.keys():
                            combined_quality_metrics[key] = []
                        break
                for quality_dict in all_quality_metrics:
                    if quality_dict:
                        for key in combined_quality_metrics.keys():
                            if key in quality_dict:
                                combined_quality_metrics[key].append(quality_dict[key])
                for key in combined_quality_metrics.keys():
                    if combined_quality_metrics[key]:
                        combined_quality_metrics[key] = np.concatenate(combined_quality_metrics[key])

            fit_results_array, fit_errors_array = self.image_analysis.fit_puncta_parallel_method(
                all_puncta_tofit,
                all_smoothed_puncta_tofit,
                all_weights_tofit,
                all_relative_coords,
                all_planes,
                strategy,
                masks=all_masks_tofit,
            )

            fit_results = self._postprocess_fit_results(
                fit_results_array,
                fit_errors_array,
                result_params,
                all_planes,
                width,
                height,
                quality_metrics=combined_quality_metrics,
            )

            append = combined_output and FOVn > 0
            self.io.write_h5_database(fit_results, fit_savename, append=append)

            if accumulate_frame_numbers:
                total_frames += file_frames

            del (
                fit_results_array,
                fit_results,
                fit_errors_array,
                all_puncta_tofit,
                all_smoothed_puncta_tofit,
                all_masks_tofit,
                all_weights_tofit,
                all_relative_coords,
                all_planes,
            )
            gc.collect()

    def fit_tracking_data(
        self,
        image_folder: Path | str,
        smoothing_function: Any,
        gain_map: NDArray[np.float32],
        offset_map: NDArray[np.float32],
        rqe: NDArray[np.float32],
        read_noise: NDArray[np.float32],
        variance: NDArray[np.float32],
        pfa: float = 1e-3,
        ROI_size: int = 16,
        peak_wavelength: float = 0.638,
        NA: float = 1.49,
        pixel_size: float | None = None,
        sigma: float = 1.5,
        fraction_true: float = 0.0,
        image_type: str = ".tif",
        use_variance_aware_demosaic: bool = True,
        use_elliptical: bool = True,
    ) -> None:
        """Single-molecule tracking data fitting function.

        Identical pipeline to fit_SM_data but fits each localisation with an
        11-parameter rotated elliptical Gaussian model (when use_elliptical=True)
        to account for motion-blur-induced PSF elongation in tracking data.

        The fitted angle ``theta`` (radians) reports the orientation of the major
        axis of the PSF; its magnitude reflects the instantaneous direction of
        motion during the camera exposure.

        Args:
            image_folder (str): Path to folder containing image files.
            smoothing_function (callable): Function to smooth data.
            gain_map (np.ndarray): 2D gain map.
            offset_map (np.ndarray): 2D offset map.
            rqe (np.ndarray): 2D relative quantum efficiency map.
            read_noise (np.ndarray): 2D read noise map.
            variance (np.ndarray): 2D variance map.
            pfa (float): Probability of false alarm for spot detection (default 1e-3).
            ROI_size (int): ROI size in pixels (default 16).
            peak_wavelength (float): Peak emission wavelength in µm (default 0.638).
            NA (float): Numerical aperture (default 1.49).
            pixel_size (float): Pixel size in µm (default 0.069).
            sigma (float): Spot detection sigma parameter (default 1.5).
            fraction_true (float): Expected fraction of true spots (default 0.0).
            image_type (str): Image file extension (default ".tif").
            use_variance_aware_demosaic (bool): Use variance-aware demosaicing (default True).
            use_elliptical (bool): Use rotated elliptical Gaussian model (default True).
                Set to False to fall back to the isotropic STANDARD strategy.

        Returns:
            None. Results written to HDF5 file alongside each input TIFF.
            Output columns when use_elliptical=True:
                xc, yc, s_x, s_y, theta, bg_B, bg_G, bg_R, A_B, A_G, A_R,
                chi_sqr, frame  [+ per-parameter errors]
        """
        from pyS3M.ImageAnalysisFunctions import FittingStrategy

        if use_elliptical:
            strategy = FittingStrategy.ELLIPTICAL
            result_params = ResultColumns.get_elliptical_columns()
        else:
            strategy = FittingStrategy.STANDARD_DATA
            result_params = ResultColumns.get_all_columns()

        if pixel_size is None:
            pixel_size = self.pixel_size

        image_files = self.helper.file_search(image_folder, image_type, "")
        start_x, start_y, width, height = self.helper.load_metadata_roi(
            image_folder, self.io, use_fallback=False
        )

        masks = self.mask.get_stacked_masks(
            start_x, start_y, width, height, self.mosaic_unit
        )
        cropped_maps = self.helper.crop_calibration_maps(
            {
                "gain_map": gain_map,
                "offset_map": offset_map,
                "read_noise": read_noise,
                "rqe": rqe,
                "variance": variance,
            },
            start_x,
            start_y,
            width,
            height,
        )
        gain_map = cropped_maps["gain_map"]
        offset_map = cropped_maps["offset_map"]
        read_noise = cropped_maps["read_noise"]
        rqe = cropped_maps["rqe"]
        variance = cropped_maps["variance"]

        for FOVn, file in enumerate(image_files):
            fit_savename = file.split(".")[0] + ".h5"

            total_frames = self.io.get_num_pages_in_TIF(file)

            chunk_size = 1000
            all_puncta_tofit = []
            all_smoothed_puncta_tofit = []
            all_masks_tofit = []
            all_weights_tofit = []
            all_relative_coords = []
            all_planes = []
            all_quality_metrics = []

            logger.info(f"Processing file {FOVn+1}/{len(image_files)}: {total_frames} frames " f"in chunks of {chunk_size} (strategy: {strategy.value})")

            for chunk_start in range(0, total_frames, chunk_size):
                chunk_end = min(chunk_start + chunk_size, total_frames)
                chunk_frames = list(range(chunk_start, chunk_end))

                logger.info(f"  Processing chunk: frames {chunk_start}-{chunk_end-1}")

                raw_data = self.io.read_tiff(file, dtype="float32", frame=chunk_frames)

                if raw_data.ndim == 2:
                    raw_data = raw_data[np.newaxis, :, :]

                image_to_analyse = self._demosaic_image(
                    raw_data,
                    use_variance_aware=use_variance_aware_demosaic,
                    gain_map=gain_map,
                    offset_map=offset_map,
                    variance=variance,
                )

                detected_puncta, quality_metrics = (
                    self.spot_detection.detect_puncta_in_stack_parallel(
                        image_to_analyse,
                        pfa=pfa,
                        variance=variance,
                        wavelength=peak_wavelength,
                        pixel_size=pixel_size,
                        NA=NA,
                        sigma=sigma,
                        fraction_true=fraction_true,
                        return_quality=True,
                    )
                )

                (
                    chunk_puncta,
                    chunk_smoothed,
                    chunk_masks,
                    chunk_weights,
                    chunk_coords,
                    chunk_planes,
                    filtered_quality_metrics,
                ) = self._process_detected_puncta_batch(
                    raw_data,
                    detected_puncta,
                    width,
                    height,
                    ROI_size,
                    smoothing_function,
                    read_noise,
                    masks,
                    gain_map=gain_map,
                    offset_map=offset_map,
                    rqe=rqe,
                    frame_offset=chunk_start,
                    is_multi_frame=True,
                    quality_metrics=quality_metrics,
                )

                all_puncta_tofit.extend(chunk_puncta)
                all_smoothed_puncta_tofit.extend(chunk_smoothed)
                all_masks_tofit.extend(chunk_masks)
                all_weights_tofit.extend(chunk_weights)
                all_relative_coords.extend(chunk_coords)
                all_planes.extend(chunk_planes)
                if filtered_quality_metrics is not None:
                    all_quality_metrics.append(filtered_quality_metrics)

                del raw_data, detected_puncta, quality_metrics, image_to_analyse
                gc.collect()

            logger.info(f"  Found {len(all_puncta_tofit)} puncta across all chunks")

            combined_quality_metrics = {}
            if len(all_quality_metrics) > 0:
                for quality_dict in all_quality_metrics:
                    if len(quality_dict) > 0:
                        for key in quality_dict.keys():
                            combined_quality_metrics[key] = []
                        break
                for quality_dict in all_quality_metrics:
                    if len(quality_dict) > 0:
                        for key in combined_quality_metrics.keys():
                            if key in quality_dict:
                                combined_quality_metrics[key].append(quality_dict[key])
                for key in combined_quality_metrics.keys():
                    if len(combined_quality_metrics[key]) > 0:
                        combined_quality_metrics[key] = np.concatenate(
                            combined_quality_metrics[key]
                        )

            fit_results_array, fit_errors_array = (
                self.image_analysis.fit_puncta_parallel_method(
                    all_puncta_tofit,
                    all_smoothed_puncta_tofit,
                    all_weights_tofit,
                    all_relative_coords,
                    all_planes,
                    strategy,
                    masks=all_masks_tofit,
                )
            )

            fit_results = self._postprocess_fit_results(
                fit_results_array,
                fit_errors_array,
                result_params,
                all_planes,
                width,
                height,
                quality_metrics=combined_quality_metrics,
            )

            self.io.write_h5_database(fit_results, fit_savename, append=False)
            del (
                fit_results_array,
                fit_results,
                fit_errors_array,
                all_puncta_tofit,
                all_smoothed_puncta_tofit,
                all_masks_tofit,
                all_weights_tofit,
                all_relative_coords,
                all_planes,
            )
            gc.collect()
        return

    def _demosaic_image(
        self,
        raw_data: np.ndarray,
        use_variance_aware: bool = True,
        gain_map: np.ndarray = None,
        offset_map: np.ndarray = None,
        variance: np.ndarray = None,
        strategy: str = 'none',
    ) -> np.ndarray:
        """Demosaic Bayer pattern image using specified method (or skip demosaicing).

        Args:
            raw_data: Bayer pattern image (2D or 3D array)
            use_variance_aware: If True, use variance-aware demosaicing with calibration maps.
                              If False, use standard grayscale demosaicing. Ignored when
                              strategy='none'. (default: True)
            gain_map: Gain calibration map (required if use_variance_aware=True)
            offset_map: Offset calibration map (required if use_variance_aware=True)
            variance: Variance map (required if use_variance_aware=True)
            strategy: Demosaicing algorithm, passed through to sCMOSFunctions
                (see sCMOSFunctions.DEMOSAIC_STRATEGIES), or 'none' (default) to skip
                demosaicing entirely and run detection directly on the raw Bayer image,
                untouched. Emitter-density tests showed the raw (non-demosaiced) image
                gives better detection performance across densities than any demosaic
                strategy — see claude/LOG.md. Other options: 'bilinear' (fastest
                demosaic), 'malvar', 'ddfapd', 'menon2007'.

        Returns:
            Image for spot detection (same shape as input): either raw_data unchanged
            (strategy='none') or a demosaiced grayscale image.

        Notes:
            - strategy='none' skips demosaicing (and any calibration conversion)
              entirely and returns raw_data as-is. Downstream, detect_puncta_in_image
              whitens by dividing by the raw (ADU²) variance calibration map — that
              division only yields a correct, unit-variance statistic for the matched
              filter if the image is in the SAME raw ADU units the variance map was
              calibrated against. Converting to photoelectrons first (dividing by a
              per-pixel gain map) before that whitening step would reintroduce a
              per-pixel gain-dependent bias the whitening is meant to remove — so
              'none' must NOT apply any gain/offset/rqe correction. Photoelectron
              conversion still happens later, but only at the fitting stage (via
              io.convert_to_photoelectrons on the raw frame), entirely separate from
              detection.
            - Variance-aware demosaicing uses calibration maps to suppress hot pixels
              and create robust photoelectron images
            - Standard demosaicing uses simple Bayer-to-grayscale conversion
        """
        if strategy == 'none':
            return raw_data
        if use_variance_aware:
            # Use variance-aware demosaicing for robust spot detection
            return self.scmos.variance_aware_demosaic(
                raw_data,
                variance_map=variance,
                offset_map=offset_map,
                gain=gain_map,
                grayscale=True,
                strategy=strategy,
            )
        else:
            # Use standard grayscale demosaicing
            return self.scmos.bayer_demosaic_stack_grayscale(raw_data, strategy=strategy)

    @staticmethod
    def _find_change_points_single(
        signal: np.ndarray,
        model: str = "l2",
        min_size: int = 5,
        penalty_factor: float = 1.0,
    ) -> list:
        """Run PELT change point detection on a single trace.

        Args:
            signal: 1D time trace of intensities
            model: ruptures model ("l2", "rbf", "normal")
            min_size: minimum segment size between change points
            penalty_factor: multiplier for automatic penalty calculation

        Returns:
            List of change point indices. Last element is always len(signal).
            If only 1 element returned (terminal only), no real change points found.
        """
        n = len(signal)
        # Estimate noise from last 100 frames (typically bleached region)
        noise_window = min(100, n // 2)
        sigma = np.nanstd(signal[-noise_window:]) if noise_window > 10 else np.nanstd(signal)

        # Handle edge case of zero/very low noise
        if sigma < 1e-10:
            sigma = np.nanstd(signal)
        if sigma < 1e-10:
            return [n]  # No valid signal, return terminal only

        algo = rpt.Pelt(model=model, min_size=min_size, jump=1).fit(signal)
        penalty = n * sigma**2 * penalty_factor
        change_points = algo.predict(pen=penalty)

        return change_points

    @staticmethod
    def _find_change_points_batch(
        traces: np.ndarray,
        indices: np.ndarray,
        model: str = "l2",
        min_size: int = 5,
        penalty_factor: float = 1.0,
    ) -> list:
        """Process a batch of traces for change point detection.

        Args:
            traces: 2D array of shape (n_traces_in_batch, n_frames)
            indices: original puncta indices for this batch
            model, min_size, penalty_factor: passed to _find_change_points_single

        Returns:
            List of (puncta_index, change_points) tuples
        """
        results = []
        for i, idx in enumerate(indices):
            signal = traces[i]
            cps = SuperRes_Functions._find_change_points_single(
                signal, model, min_size, penalty_factor
            )
            results.append((idx, cps))
        return results

    def _find_change_points_parallel(
        self,
        traces: np.ndarray,
        model: str = "l2",
        min_size: int = 5,
        penalty_factor: float = 1.0,
        n_workers: int = None,
    ) -> dict:
        """Run change point detection on all traces in parallel.

        Args:
            traces: Shape (n_puncta, n_frames)
            model: ruptures model ("l2", "rbf", "normal")
            min_size: minimum segment size between change points
            penalty_factor: multiplier for automatic penalty (pen = n * sigma^2 * factor)
            n_workers: number of parallel workers (default: 90% of CPUs)

        Returns:
            Dict mapping puncta_index -> array of frame indices to fit.
            Only includes puncta with >1 change point (i.e., real signal detected).
            Frames to fit are from 0 to the second-to-last change point (before bleaching).
        """
        from tqdm import tqdm

        n_puncta = traces.shape[0]
        n_frames = traces.shape[1]

        if n_workers is None:
            n_workers = min(60, max(1, int(0.9 * multiprocessing.cpu_count())))

        # Determine task distribution
        n_tasks = max(1, min(n_puncta, 100 * n_workers))
        puncta_per_task = [(n_puncta // n_tasks + (1 if i < n_puncta % n_tasks else 0))
                          for i in range(n_tasks)]

        # Build index ranges for each task
        start_indices = np.cumsum([0] + puncta_per_task[:-1])

        # Submit tasks. _find_change_points_batch/_single are @staticmethod
        # (no `self`) precisely so this submits a plain, cheaply-picklable
        # function -- submitting a bound method would pickle the whole
        # SuperRes_Functions instance (calibration arrays and all) once per
        # task. That's a no-op cost under fork (Linux/Mac; child processes
        # share memory, nothing is actually pickled), but under Windows'
        # spawn-only multiprocessing every task really does re-pickle it,
        # which reliably crashed every worker ("terminated abruptly") there.
        frames_to_fit = {}
        with futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            fs = []
            for start_idx, n_in_task in zip(start_indices, puncta_per_task):
                if n_in_task == 0:
                    continue
                end_idx = start_idx + n_in_task
                task_traces = traces[start_idx:end_idx]
                task_indices = np.arange(start_idx, end_idx)
                fs.append(
                    executor.submit(
                        self._find_change_points_batch,
                        task_traces,
                        task_indices,
                        model,
                        min_size,
                        penalty_factor,
                    )
                )

            # Collect results with progress bar
            with tqdm(desc="Finding change points", total=len(fs), unit="batch") as pbar:
                for f in futures.as_completed(fs):
                    batch_results = f.result()
                    for puncta_idx, cps in batch_results:
                        # Only keep puncta with real change points (not just terminal)
                        if len(cps) > 1:
                            # Fit frames from 0 to the last real change point (before bleaching)
                            last_real_cp = cps[-2]
                            frames_to_fit[puncta_idx] = np.arange(last_real_cp)
                    pbar.update(1)

        return frames_to_fit

    def fit_imaging_data(
        self,
        image_folder: Path | str,
        smoothing_function: Any,
        gain_map: NDArray[np.float32],
        offset_map: NDArray[np.float32],
        rqe: NDArray[np.float32],
        read_noise: NDArray[np.float32],
        variance: NDArray[np.float32],
        pfa: float = 1e-3,
        ROI_size: int = 20,
        peak_wavelength: float = 0.638,
        NA: float = 1.49,
        pixel_size: float | None = None,
        sigma: float = 1.5,
        fraction_true: float = 0.0,
        image_type: str = ".tif",
        use_variance_aware_demosaic: bool = True,
        use_elliptical: bool = False,
        use_circular: bool = False,
    ) -> None:
        """Cross-file imaging data fitting function.

        Analyzes imaging data across multiple files by detecting puncta and fitting them
        with Gaussian models, maintaining frame numbering consistency across files.

        Args:
            image_folder (str): Path to folder containing image files
            smoothing_function (type): function to smooth data
            gain_map (np.2darray): 2darray of gain map
            offset_map (np.2darray): 2darray of offset map
            rqe (np.2darray): 2d array of RQE
            read_noise (np.2darray): 2d array of read noise
            variance (np.2darray): 2d array of variance
            pfa (float): Probability of false alarm for spot detection (default: 1e-3)
            ROI_size (int): Size of ROI for fitting (default: 12)
            peak_wavelength (float): peak wavelength of PSF (default: 0.638)
            NA (float): Numerical aperture (default: 1.49)
            pixel_size (float): Pixel size in microns (default: 0.069)
            sigma (float): sigma parameter for spot detection (default: 1.5)
            fraction_true (float): fraction of true spots expected (default: 0.0)
            image_type (str): image file extension (default: ".tif")
            use_variance_aware_demosaic (bool): Whether to use variance-aware demosaicing for spot detection.
                If True (default), uses gain, offset, and variance maps to create robust photoelectron
                images that suppress hot pixels. If False, uses standard grayscale demosaicing.
            use_elliptical (bool): Use the 11-parameter rotated elliptical Gaussian model
                (FittingStrategy.ELLIPTICAL, adds xc,yc,s_x,s_y,theta) instead of the default
                axis-aligned STANDARD_DATA model. Default False (unchanged behaviour).
            use_circular (bool): Use FittingStrategy.CIRCULAR (STANDARD_DATA with s_x/s_y
                constrained to one shared width, "s") instead of STANDARD_DATA. Mutually
                exclusive with use_elliptical.
        Returns:
            None: Writes results to HDF5 file:
                - image_folder/Localisations.h5
        """
        return self._fit_files(
            image_folder, smoothing_function, gain_map, offset_map, rqe, read_noise, variance,
            pfa=pfa, ROI_size=ROI_size, peak_wavelength=peak_wavelength, NA=NA,
            pixel_size=pixel_size, sigma=sigma, fraction_true=fraction_true,
            image_type=image_type, use_variance_aware_demosaic=use_variance_aware_demosaic,
            use_elliptical=use_elliptical, use_circular=use_circular,
            accumulate_frame_numbers=True, combined_output=True,
        )
