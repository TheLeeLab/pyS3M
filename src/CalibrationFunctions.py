#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 16:27:38 2024

@author: jbeckwith
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np
import scipy
import scipy.ndimage
from numpy.typing import NDArray

sys.path.append(str(Path(__file__).parent))
import pyS3M.IOFunctions as IOFunctions
from pyS3M.Constants import CalibrationConstants
import pyS3M.MaskFunctions as MaskFunctions
import pyS3M.HelperFunctions as HelperFunctions
import logging
logger = logging.getLogger(__name__)



class Calibration_Functions:
    """Camera calibration routines for sCMOS cameras and Bayer filter patterns.

    Provides functionality for calibrating camera parameters, processing
    gain/offset/variance maps, and handling Bayer pattern configurations.
    """

    def __init__(
        self,
        camera: str = "ximea",
        mosaic_unit: NDArray | None = None,
        high_memory: bool = False,
        chunk_size: int = 50,
        io_functions: object | None = None,
        mask_functions: object | None = None,
        helper_functions: object | None = None,
    ) -> None:
        """Initialize Calibration_Functions class.

        Args:
            camera: Camera model name used to set default ``mosaic_unit``.
                Currently ``"ximea"`` (BGGR) or ``"zwo"`` (RGGB).
                Overridden by an explicit *mosaic_unit* kwarg.
            mosaic_unit: Custom Bayer mosaic pattern.  If ``None``, taken
                from *camera* defaults.
            high_memory: Whether to use high-memory processing mode.
            chunk_size: Number of frames to read per I/O call in low-memory mode.
                        Larger values are faster but use more RAM (default: 50).
            io_functions: IO functions instance (default: creates new instance)
            mask_functions: Mask functions instance (default: creates new instance)
            helper_functions: Helper functions instance (default: creates new instance)
        """
        import pyS3M.CameraDefaults as CameraDefaults
        config = CameraDefaults.get_camera_config(camera)
        self.high_memory = high_memory
        self.chunk_size = chunk_size
        self.mosaic_unit = mosaic_unit if mosaic_unit is not None else config.mosaic_unit

        # Dependency injection with sensible defaults
        self.io = (
            io_functions
            if io_functions is not None
            else IOFunctions.IO_Functions(camera=camera, mosaic_unit=self.mosaic_unit)
        )
        self.Mask = (
            mask_functions
            if mask_functions is not None
            else MaskFunctions.Mask_Functions(camera=camera, mosaic_unit=self.mosaic_unit)
        )
        self.helper = (
            helper_functions
            if helper_functions is not None
            else HelperFunctions.Helper_Functions()
        )

    def filesearch(self, directory: Path | str, string1: str, string2: str) -> list[str]:
        files = [p.name for p in Path(directory).iterdir()]
        files = np.sort([x for x in files if string1 in x])
        files = np.sort([x for x in files if string2 in x])
        return files

    def _discover_intensity_strings(self, folder: Path | str, imtype: str) -> np.ndarray:
        """Sorted ``"Intensity_<value>"`` strings found among *imtype* files in
        *folder* — shared by both the per-colour (RGB) and single-folder (NIR)
        branches of :meth:`calibrate_multicolour_camera`."""
        files = np.unique([p.name for p in Path(folder).iterdir()])
        files = np.sort([x for x in files if imtype in x])
        intensity_strings = np.sort(
            np.unique([x.split("Intensity_")[1].split("_")[0] for x in files])
        )
        return np.asarray(
            np.full(len(intensity_strings), "Intensity_")
            + np.asarray(intensity_strings, dtype="object"),
            dtype=str,
        )

    def calibrate_multicolour_camera(
        self, directory: Path | str, imtype: str = ".tif", mode: str = "rgb"
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]] | None:
        """
        Calibrates multicolour camera.

        Args:
            directory (string): Folder containing tifs
            imtype (string): image type to read in
            mode (string): ``"rgb"`` (default) expects one flat-field
                subfolder per Bayer colour (named exactly matching
                ``np.unique(self.mosaic_unit)``, e.g. "R"/"G"/"B"), each
                illuminated separately and masked into its own Bayer
                positions. ``"nir"`` expects exactly one flat-field
                subfolder (any name) alongside ``dark`` — the Ximea/ZWO
                Bayer filters' R/G/B transmission spectra converge above
                ~750 nm, so a single >750 nm flat-field acquisition
                illuminates every pixel identically regardless of its
                Bayer colour, and is applied uniformly (no per-colour
                masking) rather than requiring separate per-colour data.

        Returns:
            offset (np.2darray): offset matrix
            variance (np.2darray): variance matrix
            gain (np.2darray): gain matrix
            read_noise (np.2darray): read noise matrix
            r_QE (np.2darray): relative QE matrix
        """
        directories = [p.name for p in Path(directory).iterdir()]
        dark_directory = np.sort([x for x in directories if "dark" in x])
        try:
            if len(dark_directory) != 1:
                raise Exception("Incorrect number of dark folders; should be 1")
        except Exception as error:
            logger.warning("Caught this error: " + repr(error))
            return
        dark_directory = str(dark_directory[0])

        # calculate dark offset and variance (single pass)
        offset, variance = self.calculate_offset_and_variance(
            Path(directory) / dark_directory, "dark"
        )

        if mode == "nir":
            nir_directories = [x for x in directories if x != dark_directory]
            try:
                if len(nir_directories) != 1:
                    raise Exception(
                        "Incorrect number of NIR folders; should be exactly 1 "
                        "(Bayer colours respond identically above ~750 nm, so "
                        "only one flat-field folder is needed)"
                    )
            except Exception as error:
                logger.warning("Caught this error: " + repr(error))
                return
            nir_directory = str(nir_directories[0])

            intensity_strings = self._discover_intensity_strings(
                Path(directory) / nir_directory, imtype
            )
            n_powers = len(intensity_strings)

            offset_intensities = np.zeros([offset.shape[0], offset.shape[1], n_powers])
            variance_intensities = np.zeros(
                [variance.shape[0], variance.shape[1], n_powers]
            )
            for i, intensity in enumerate(intensity_strings):
                # Every pixel gets the same source data — Bayer position stops
                # mattering above 750 nm, so no colour masking is applied.
                off_i, var_i = self.calculate_offset_and_variance(
                    Path(directory) / nir_directory, intensity
                )
                offset_intensities[:, :, i] = off_i
                variance_intensities[:, :, i] = var_i
        else:
            colours = np.unique(self.mosaic_unit)
            colour_directories = np.sort([x for x in directories if x in colours])

            n_powermatrix = np.zeros(len(colour_directories))
            for i, colour in enumerate(colour_directories):
                intensity_strings = self._discover_intensity_strings(
                    Path(directory) / colour, imtype
                )
                n_powermatrix[i] = len(intensity_strings)

            try:
                if n_powermatrix.ptp() != 0:
                    raise Exception(
                        "Incorrect number of intensity values for each colour; should be equal"
                    )
            except Exception as error:
                logger.warning("Caught this error: " + repr(error))
                return

            n_powers = int(np.unique(n_powermatrix))
            offset_intensities = np.zeros([offset.shape[0], offset.shape[1], n_powers])
            variance_intensities = np.zeros(
                [variance.shape[0], variance.shape[1], n_powers]
            )

            masks = self.Mask.get_masks(
                mosaic_unit=self.mosaic_unit, size_x=offset.shape[0], size_y=offset.shape[1]
            )

            for i, intensity in enumerate(intensity_strings):
                for colour in colour_directories:
                    # Single-pass: compute offset and variance for this colour/intensity
                    # together so each channel uses its own correct mean (not a
                    # partially-accumulated cross-channel offset as the old code did).
                    off_i, var_i = self.calculate_offset_and_variance(
                        Path(directory) / colour, intensity
                    )
                    offset_intensities[:, :, i] += np.multiply(off_i, masks[str(colour)])
                    variance_intensities[:, :, i] += np.multiply(var_i, masks[str(colour)])

        A = np.subtract(variance_intensities, variance[:, :, np.newaxis])
        B = np.subtract(offset_intensities, offset[:, :, np.newaxis])

        A_filepath = Path(directory) / "A.tif"
        B_filepath = Path(directory) / "B.tif"
        self.io.write_tiff(
            np.swapaxes(np.swapaxes(A, -1, 0), -1, -2),
            A_filepath,
            bit=float,
            pixel_size=CalibrationConstants.PIXEL_SIZE,
        )
        self.io.write_tiff(
            np.swapaxes(np.swapaxes(B, -1, 0), -1, -2),
            B_filepath,
            bit=float,
            pixel_size=CalibrationConstants.PIXEL_SIZE,
        )

        # Vectorised OLS: gain[i,j] = sum(A*B) / sum(B*B) per pixel.
        # Bi is a 1-D vector so pinv(Bi@Bi.T) = 1/||Bi||^2, reducing the
        # original per-pixel pinv loop to a single pair of einsum calls.
        denom = np.einsum('ijk,ijk->ij', B, B)
        numer = np.einsum('ijk,ijk->ij', A, B)
        gain = np.where(denom > 0, numer / denom, np.nan)

        readnoise = np.divide(np.sqrt(variance), gain)
        rqe = self.calculate_rqe(offset_intensities[:, :, -1], offset, gain)

        offset_file_path = Path(directory) / "offset.tif"
        gain_file_path = Path(directory) / "gain.tif"
        variance_file_path = Path(directory) / "variance.tif"
        readnoise_file_path = Path(directory) / "readnoise.tif"
        rqe_file_path = Path(directory) / "rqe.tif"

        # write tiffs
        self.io.write_tiff(offset, offset_file_path, bit=float, pixel_size=3.45)
        self.io.write_tiff(variance, variance_file_path, bit=float, pixel_size=3.45)
        self.io.write_tiff(gain, gain_file_path, bit=float, pixel_size=3.45)
        self.io.write_tiff(readnoise, readnoise_file_path, bit=float, pixel_size=3.45)
        self.io.write_tiff(rqe, rqe_file_path, bit=float, pixel_size=3.45)

        logger.debug("The average offset is {:.3f} +- {:.3f} ADU counts".format( np.nanmean(offset), np.nanstd(offset) ))
        logger.debug("The average variance is {:.3f} +- {:.3f} ADU^2 counts".format( np.nanmean(variance), np.nanstd(variance) ))
        logger.debug("The average gain is {:.3f} +- {:.3f} ADU counts/photoelectron".format( np.nanmean(gain), np.nanstd(gain) ))
        logger.debug("The average read noise is {:.3f} +- {:.3f} photoelectrons".format( np.nanmean(readnoise), np.nanstd(readnoise) ))
        logger.debug("The median read noise is {:.3f} photoelectrons".format( np.nanmedian(readnoise) ))
        logger.debug("The RMS read noise is {:.3f} photoelectrons".format( np.sqrt(np.nanmean(np.square(readnoise))) ))
        return offset, variance, gain, readnoise, rqe

    def calculate_rqe(self, intensity_image: NDArray[np.float32], offset: NDArray[np.float32], gain: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Calibrates relative QE. Given an intensity image, an offset and a gain, calculate a relative QE.

        Args:
            intensity_image (np.2darray): Intensity image
            offset (np.2darray): 2d matrix of offset
            gain (np.2darray): 2d matrix of gain

        Returns:
            rqe (np.2darray): rqe matrix
        """
        corrected_image = np.divide(np.subtract(intensity_image, offset), gain)
        smoothed_image = scipy.ndimage.uniform_filter(
            corrected_image, size=CalibrationConstants.SMOOTHING_SIZE, mode="nearest"
        )
        rqe = np.divide(corrected_image, smoothed_image)
        return rqe

    def _process_calibration_files(
        self,
        directory,
        intensity_string,
        filelist,
        accumulator,
        operation_name,
        process_single_frame_fn,
        process_multi_frame_fn,
    ):
        """
        Generic file processing loop for calibration calculations.

        Args:
            directory (string): Folder containing tifs
            intensity_string (string): Intensity string for display
            filelist (list): List of filenames to process
            accumulator (np.ndarray): Array to accumulate results into
            operation_name (string): Name of operation for progress display (e.g., "offset", "variance")
            process_single_frame_fn (callable): Function to process single frame images
            process_multi_frame_fn (callable): Function to process multi-frame images

        Returns:
            tuple: (accumulator, framesCounter) - updated accumulator and total frame count
        """
        framesCounter = 0
        start = time.time()

        for i, file in enumerate(filelist):
            if self.high_memory == True:
                image = self.io.read_tiff(Path(directory) / file)
                if len(image.shape) == 2:
                    n_frames = 1
                else:
                    # read_tiff returns frame-first (N, H, W); process_multi_frame_fn
                    # expects (H, W, N) -- same contract the low-memory chunked path
                    # below transposes to, so match it here too.
                    image = np.transpose(image, (1, 2, 0))
                    n_frames = image.shape[-1]
                if n_frames == 1:
                    accumulator = process_single_frame_fn(accumulator, image)
                else:
                    accumulator = process_multi_frame_fn(accumulator, image)
            else:
                path = Path(directory) / file
                n_frames = self.io.get_n_frames(path)
                for chunk_start in range(0, n_frames, self.chunk_size):
                    chunk_end = min(chunk_start + self.chunk_size, n_frames)
                    chunk = self.io.read_tiff(
                        path, frame=list(range(chunk_start, chunk_end))
                    )
                    if chunk.ndim == 2:
                        chunk = chunk[np.newaxis, ...]  # (1, H, W)
                    # process_multi_frame_fn expects (H, W, N); transpose accordingly
                    chunk = chunk.transpose(1, 2, 0)
                    accumulator = process_multi_frame_fn(accumulator, chunk)

            framesCounter = framesCounter + n_frames
            elapsed = time.time() - start
            elapsed_display, timestring = self.helper.format_elapsed_time(elapsed)

            if Path(directory).name == intensity_string:
                logger.debug(f"Analysed {operation_name} of " + intensity_string + " image {}/{}    Time elapsed: {:.3f} {}".format( i + 1, len(filelist), elapsed_display, timestring ))
            else:
                logger.debug(f"Analysed {operation_name} of " + Path(directory).name + " " + intensity_string + " image {}/{}    Time elapsed: {:.3f} {}".format( i + 1, len(filelist), elapsed_display, timestring ))

        return accumulator, framesCounter

    def calculate_offset(self, directory: Path | str, intensity_string: str, imtype: str = ".tif") -> NDArray[np.float32]:
        """
        Calibrates offset. Given a directory, looks for a particular intensity
        string and loads these images. Then gets an offset.

        Args:
            directory (string): Folder containing tifs
            intensity_string (string): Intensity string
            imtype (string): image type to read in

        Returns:
            offset (np.2darray): offset matrix
        """
        filelist = self.filesearch(directory, imtype, intensity_string)

        frame0_shape = self.io.read_tiff(Path(directory) / filelist[0], 0).shape
        width = frame0_shape[0]
        height = frame0_shape[1]
        offset = np.zeros([width, height])

        if Path(directory).name == intensity_string:
            logger.debug("Starting offset analysis of " + intensity_string)
        else:
            logger.debug("Starting offset analysis of " + Path(directory).name + " " + intensity_string)

        # Define processing functions for offset calculation
        def process_single(acc, frame):
            return np.add(acc, frame)

        def process_multi(acc, image):
            return np.add(acc, np.sum(image, axis=-1))

        # Process all files
        offset, framesCounter = self._process_calibration_files(
            directory,
            intensity_string,
            filelist,
            offset,
            "offset",
            process_single,
            process_multi,
        )

        offset = offset / framesCounter
        return offset

    def calculate_offset_and_variance(self, directory: Path | str, intensity_string: str, imtype: str = ".tif") -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Compute offset (mean) and variance in a single pass over calibration files.

        Accumulates sum(X) and sum(X²) simultaneously, then computes:
            offset   = sum(X)  / N
            variance = sum(X²) / N  −  offset²

        This halves file I/O compared to calling calculate_offset and
        calculate_variance separately, and avoids the bug where variance
        was computed using a partially-filled cross-channel offset map.

        sum(X²) is accumulated in float64 to avoid precision loss when
        squaring 16-bit ADU values in float32.

        Args:
            directory (string): Folder containing tifs
            intensity_string (string): Intensity string
            imtype (string): image type to read in

        Returns:
            offset   (np.2darray, float32): mean pixel value (ADU)
            variance (np.2darray, float32): pixel variance  (ADU²)
        """
        filelist = self.filesearch(directory, imtype, intensity_string)

        frame0_shape = self.io.read_tiff(Path(directory) / filelist[0], 0).shape
        sum_frames = np.zeros(frame0_shape, dtype=np.float64)
        sum_sq_frames = np.zeros(frame0_shape, dtype=np.float64)
        framesCounter = 0

        dir_label = Path(directory).name
        display_label = (
            intensity_string
            if dir_label == intensity_string
            else f"{dir_label} {intensity_string}"
        )
        logger.debug(f"Starting offset+variance analysis of {display_label}")

        start_t = time.time()
        for file_i, file in enumerate(filelist):
            path = Path(directory) / file
            n_file_frames = self.io.get_n_frames(path)

            for chunk_start in range(0, n_file_frames, self.chunk_size):
                chunk_end = min(chunk_start + self.chunk_size, n_file_frames)
                chunk = self.io.read_tiff(
                    path, frame=list(range(chunk_start, chunk_end))
                )
                if chunk.ndim == 2:
                    chunk = chunk[np.newaxis, ...]  # (1, H, W)
                # chunk shape: (N, H, W); sum over frame axis
                sum_frames += np.sum(chunk, axis=0)
                sum_sq_frames += np.sum(chunk.astype(np.float64) ** 2, axis=0)
                framesCounter += chunk.shape[0]

            elapsed = time.time() - start_t
            elapsed_display, timestring = self.helper.format_elapsed_time(elapsed)
            logger.debug(f"Analysed offset+variance of {display_label} " f"image {file_i + 1}/{len(filelist)}    " f"Time elapsed: {elapsed_display:.3f} {timestring}")

        mean = sum_frames / framesCounter
        variance = sum_sq_frames / framesCounter - mean ** 2
        return mean.astype(np.float32), variance.astype(np.float32)

    # calculate_variance removed 2026-04-10 — superseded by calculate_offset_and_variance()
    # which computes both in a single file I/O pass. Backed up in claude/backup/old_calibration.py.
