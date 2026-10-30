# -*- coding: utf-8 -*-
"""
This class contains functions pertaining to IO of files for pyS3M.
@author: jbeckwith
jsb92, 2024/01/02
"""
from __future__ import annotations

import json
from copy import copy
from pathlib import Path
from typing import Any, Optional
import sys
import logging

import numpy as np
import pandas as pd
import polars as pl
import tifffile
from numpy.typing import NDArray
from tifffile import imread, imwrite

logger = logging.getLogger(__name__)


sys.path.append(str(Path(__file__).parent))


class IO_Functions:
    """File I/O operations for microscopy data and analysis results.

    Provides functionality for reading and writing various file formats
    used in single-molecule localization microscopy analysis.
    """

    def __init__(self, camera: str = "ximea", mosaic_unit: NDArray | None = None) -> None:
        """Initialize IO_Functions class.

        Args:
            camera: Camera model name; sets the mosaic used to convert fitted amplitudes
                to detected photoelectrons. Overridden by an explicit *mosaic_unit*.
            mosaic_unit: Mosaic pattern array. If ``None``, taken from *camera* defaults.
        """
        from pyS3M.MaskFunctions import Mask_Functions
        self.pixel_fractions = Mask_Functions(camera=camera, mosaic_unit=mosaic_unit).pixel_fractions()

    def _mask_weighted_sum(self, df, prefix):
        """``sum_p fraction_p * df[prefix + p]`` over the mosaic's pixel types."""
        return sum(f * df[f"{prefix}{p}"] for p, f in self.pixel_fractions.items())

    def add_photoelectron_columns(self, df):
        """Add detected photoelectrons to a database with normalised amplitudes.

        ``photons`` (``A_B + A_G + A_R``) adds up what each pixel type would collect if it
        covered the whole sensor, so it is not the signal detected. Weighting each
        amplitude by its pixel type's share of the sensor (:meth:`Mask_Functions.pixel_fractions`)
        gives:

        * ``photoelectrons`` -- detected photoelectrons of the localisation,
          ``photons * sum_p fraction_p * A_p``;
        * ``background_photoelectrons`` -- mean background per pixel (photoelectrons),
          ``background_photons * sum_p fraction_p * bg_p``.

        Works on any database written by :meth:`write_h5_database`, so existing results need
        no refit.

        Args:
            df (pd.DataFrame): Database with ``photons`` and normalised ``A_*`` (and/or
                ``background_photons`` and normalised ``bg_*``) columns.

        Returns:
            pd.DataFrame: Copy of *df* with the columns added.
        """
        df = df.copy()
        amplitudes = [f"A_{p}" for p in self.pixel_fractions]
        backgrounds = [f"bg_{p}" for p in self.pixel_fractions]
        if all(col in df.columns for col in ["photons", *amplitudes]):
            df["photoelectrons"] = df["photons"] * self._mask_weighted_sum(df, "A_")
        if all(col in df.columns for col in ["background_photons", *backgrounds]):
            df["background_photoelectrons"] = df["background_photons"] * self._mask_weighted_sum(df, "bg_")
        return df

    def _normalize_color_channels(self, df, total_col, color_cols):
        """Normalize color channel values by their total.

        Args:
            df (pd.DataFrame): Input dataframe
            total_col (str): Name of column containing total values
            color_cols (list): List of column names to normalize (e.g., ['A_B', 'A_G', 'A_R'])

        Returns:
            pd.DataFrame: Dataframe with normalized columns

        Notes:
            This method normalizes values by dividing by the total, avoiding
            division by zero using a mask. The errors are left as they are: the fitter
            already reports them as errors of the normalised fractions, propagated from
            its full covariance (FittingResultProcessor.fraction_errors).
        """
        # Avoid division by zero
        mask = df[total_col] > 0

        # Normalize color channel values
        for col in color_cols:
            df.loc[mask, col] = df.loc[mask, col] / df.loc[mask, total_col]

        return df

    def read_h5_database(self, filepath: Path | str, key: str = "data") -> pd.DataFrame:
        """Read localisation database from HDF5 file.

        Args:
            filepath (str): Path to HDF5 file.
            key (str): HDF5 key to read (default: "data").

        Returns:
            pd.DataFrame: Localisation data.
        """
        return pd.read_hdf(filepath, key=key)

    def write_h5_database(self, df: pd.DataFrame, filepath: Path | str, append: bool = False, normalise_photons: bool = True, verbose: bool = True, key: str = "data") -> None:
        """Write localisation DataFrame to HDF5 with optional photon normalisation and frame sorting.

        Args:
            df (pd.DataFrame): Data to write.
            filepath (str): Destination HDF5 path.
            append (bool): If True, append to existing file and re-sort by frame.
            normalise_photons (bool): If True, add/normalise photon columns.
            verbose (bool): Print progress messages.
            key (str): HDF5 key to write under (default: "data", the convention
                :meth:`read_h5_database` and callers that glob a folder for result
                files expect). Non-localisation companion files written alongside
                real results in the same folder -- e.g. simulation ground truth --
                should use a different key so they're skipped (via the existing
                ``KeyError`` handling) rather than misread as fit results.
        """
        if df.shape[0] > 0:
            # first, remove any rows that are all NaN
            df = df.dropna(axis=0, how="all")
            # Convert frame column to int32 only when present
            # (some DataFrames, e.g. single-molecule summaries, have no frame column)
            if "frame" in df.columns:
                df["frame"] = pd.to_numeric(df["frame"], errors="coerce").astype("int32")
            # Add photon columns if amplitude columns are present
            if "photons" not in df.columns:
                df = self._add_photon_columns(df, normalise=normalise_photons)

            if append and Path(filepath).is_file():
                # Check schema compatibility before appending
                df = self._ensure_hdf5_compatibility(df, filepath, key=key)
                df.to_hdf(filepath, key=key, append=True, mode="r+", format="table", index=False)

                # Re-read entire file, sort by frame, and rewrite
                # This ensures proper frame ordering for visualization
                if verbose:
                    logger.info(f"Sorting appended HDF5 file by frame: {Path(filepath).name}")
                with pd.HDFStore(filepath, mode="r+") as store:
                    if key in store:
                        # Read all data
                        full_df = store[key]
                        if verbose:
                            logger.info(f"  Read {len(full_df):,} total localizations")

                        # Sort by frame with stable sort to preserve order within frames
                        sorted_df = full_df.sort_values(
                            by="frame", kind="mergesort", ignore_index=True
                        ).reset_index(drop=True)
                        if verbose:
                            logger.info(f"  Sorted by frame (range: {sorted_df['frame'].min()}-{sorted_df['frame'].max()})")

                        # Remove old data and write sorted data back
                        store.remove(key)
                        store.put(key, sorted_df, format="table", index=False)
                        if verbose:
                            logger.info(f"  Rewritten sorted data to {Path(filepath).name}")
            else:
                df.to_hdf(filepath, key=key, format="table", index=False)

    # Keep private alias for any external callers not yet migrated
    _write_h5_database = write_h5_database

    def _add_photon_columns(self, df, normalise=True):
        """
        Automatically add photon columns and optionally normalise amplitude/background data.

        Also adds the detected ``photoelectrons`` and ``background_photoelectrons`` (per
        pixel), weighting each pixel type by its share of the mosaic (see
        :meth:`add_photoelectron_columns`).

        Args:
            df (pd.DataFrame): Input dataframe
            normalise (bool): Whether to normalise A_B/A_G/A_R and bg_B/bg_G/bg_R columns

        Returns:
            pd.DataFrame: Dataframe with photon columns added and optionally normalised
        """
        df = df.copy()  # Avoid modifying original dataframe

        # Add total photons column and normalise A_B, A_G, A_R if they exist
        if all(col in df.columns for col in ["A_B", "A_G", "A_R"]):
            df["photons"] = df["A_B"] + df["A_G"] + df["A_R"]
            df["photoelectrons"] = self._mask_weighted_sum(df, "A_")

            if normalise:
                df = self._normalize_color_channels(
                    df,
                    total_col="photons",
                    color_cols=["A_B", "A_G", "A_R"],
                )

        # Add background photons column and normalise bg_B, bg_G, bg_R if they exist
        if all(col in df.columns for col in ["bg_B", "bg_G", "bg_R"]):
            df["background_photons"] = df["bg_B"] + df["bg_G"] + df["bg_R"]
            df["background_photoelectrons"] = self._mask_weighted_sum(df, "bg_")

            if normalise:
                df = self._normalize_color_channels(
                    df,
                    total_col="background_photons",
                    color_cols=["bg_B", "bg_G", "bg_R"],
                )

        return df

    def sort_h5_by_frame(self, filepath: Path | str, backup: bool = True) -> None:
        """Sort existing HDF5 file by frame number.

        Useful for fixing files where frame numbers are not in order due to
        multiple FOVs being appended without frame offset.

        Args:
            filepath: Path to HDF5 file to sort
            backup: Whether to create backup before sorting (recommended)
        """
        import shutil

        backup_path = None
        if backup:
            backup_path = filepath.replace(".h5", "_backup.h5")
            shutil.copy2(filepath, backup_path)
            logger.info(f"Created backup: {backup_path}")

        # Read, sort, and rewrite the data
        df_sorted = None
        try:
            with pd.HDFStore(filepath, mode="r") as store:
                if "data" in store:
                    df = store["data"]
                    logger.info(f"Read {len(df):,} localizations")

                    # Sort by frame using mergesort for stability
                    df_sorted = df.sort_values(
                        by="frame", kind="mergesort", ignore_index=True
                    ).reset_index(drop=True)
                    logger.info(f"Sorted data by frame (range: {df_sorted['frame'].min()}-{df_sorted['frame'].max()})")

            # Write sorted data back
            if df_sorted is not None:
                df_sorted.to_hdf(filepath, key="data", format="table", mode="w", index=False)
                logger.info(f"Sorted HDF5 file saved: {filepath}")

        except Exception as e:
            logger.warning(f"Error sorting HDF5 file {filepath}: {e}")
            if backup and backup_path:
                logger.info(f"Restore from backup if needed: {backup_path}")

    def _ensure_hdf5_compatibility(self, df, filepath, key="data"):
        """
        Ensure new DataFrame is compatible with existing HDF5 table schema.

        This method reads the existing HDF5 table schema and adjusts the new DataFrame's
        dtypes to match, preventing the "invalid combination of values_axes" error.

        Args:
            df (pd.DataFrame): New DataFrame to append
            filepath (str): Path to existing HDF5 file
            key (str): HDF5 key to check against (default: "data").

        Returns:
            pd.DataFrame: DataFrame with compatible dtypes
        """
        import pandas as pd

        try:
            # Read just the first row to get schema information
            existing_df = pd.read_hdf(filepath, key=key, stop=1)

            if len(existing_df) == 0:
                # Empty table - return original df
                return df

            # Get column overlap
            common_columns = set(df.columns) & set(existing_df.columns)

            if not common_columns:
                # No common columns - return original
                return df

            # Create copy to avoid modifying original
            df_compatible = df.copy()

            # Adjust dtypes for common columns
            for col in common_columns:
                existing_dtype = existing_df[col].dtype
                new_dtype = df_compatible[col].dtype

                if existing_dtype != new_dtype:
                    try:
                        # Special handling for frame column - always prefer int32
                        if col == "frame":
                            if existing_dtype.kind == "i" and new_dtype.kind == "i":
                                # Both integers - prefer wider int32 for frame numbers
                                target_dtype = (
                                    "int32"
                                    if existing_dtype == "int16"
                                    else existing_dtype
                                )
                                df_compatible[col] = df_compatible[col].astype(
                                    target_dtype
                                )
                                if target_dtype != existing_dtype:
                                    logger.info(f"Note: Frame column upgrading from {existing_dtype} to {target_dtype} for large frame number support")
                            else:
                                df_compatible[col] = df_compatible[col].astype(
                                    existing_dtype
                                )
                        # Try to convert to existing dtype
                        elif pd.api.types.is_numeric_dtype(
                            existing_dtype
                        ) and pd.api.types.is_numeric_dtype(new_dtype):
                            # For numeric types, convert carefully
                            if existing_dtype.kind == "i" and new_dtype.kind == "i":
                                # Both integers - use the existing dtype
                                df_compatible[col] = df_compatible[col].astype(
                                    existing_dtype
                                )
                            elif existing_dtype.kind in [
                                "i",
                                "u",
                            ] and new_dtype.kind in ["f"]:
                                # New is float, existing is int - keep as int if possible
                                if (
                                    df_compatible[col].dtype == "float64"
                                    and not df_compatible[col].isna().any()
                                ):
                                    # Check if all values are integers
                                    if (
                                        df_compatible[col]
                                        == df_compatible[col].astype(int)
                                    ).all():
                                        df_compatible[col] = df_compatible[col].astype(
                                            existing_dtype
                                        )
                                    else:
                                        # Can't convert to int - widen existing type to float
                                        logger.warning(f"Warning: Column '{col}' dtype mismatch ({existing_dtype} vs {new_dtype}). " f"Values contain decimals, keeping as {new_dtype}")
                                else:
                                    df_compatible[col] = df_compatible[col].astype(
                                        existing_dtype
                                    )
                            else:
                                # Other numeric conversions
                                df_compatible[col] = df_compatible[col].astype(
                                    existing_dtype
                                )
                        else:
                            # Non-numeric - try direct conversion
                            df_compatible[col] = df_compatible[col].astype(
                                existing_dtype
                            )

                    except (ValueError, TypeError) as e:
                        logger.warning(f"Warning: Could not convert column '{col}' from {new_dtype} to {existing_dtype}: {e}")
                        logger.info(f"Keeping original dtype {new_dtype}")
                        # Keep original dtype if conversion fails

            return df_compatible

        except Exception as e:
            logger.warning(f"Warning: Could not check HDF5 compatibility: {e}")
            logger.info("Proceeding with original DataFrame")
            return df

    def read_json(self, filename: Path | str, encoding: str = "ISO-8859-1") -> dict[str, Any]:
        """
        read data from a JSON file.

        Args:
            filename (str): The name of the JSON file to load.

        Returns:
            data (dict): The loaded JSON data.
        """
        try:
            with open(filename, "r", encoding=encoding) as file:
                data = json.load(file)
        except (UnicodeDecodeError, LookupError) as e:
            with open(filename, "r") as file:
                data = json.load(file)
        except json.JSONDecodeError as e:
            # Try to salvage partial JSON data
            try:
                with open(filename, "r", encoding=encoding) as file:
                    content = file.read()

                # Try to find and parse up to the error position
                if e.pos and e.pos > 0:
                    # Find the last complete object before the error
                    partial = content[:e.pos]
                    # Try to close incomplete structures
                    brace_count = partial.count('{') - partial.count('}')
                    bracket_count = partial.count('[') - partial.count(']')

                    if brace_count > 0 or bracket_count > 0:
                        partial += ']' * bracket_count + '}' * brace_count
                        try:
                            data = json.loads(partial)
                            return data
                        except json.JSONDecodeError:
                            pass

                # If that doesn't work, raise the original error with helpful context
                raise json.JSONDecodeError(
                    f"Cannot parse JSON file. {e.msg} at line {e.lineno}, col {e.colno}. "
                    f"File may be corrupted or incomplete. Position: {e.pos}/{len(content)} bytes",
                    e.doc, e.pos
                ) from e
            except Exception as read_err:
                raise json.JSONDecodeError(
                    f"Cannot parse JSON file: {e.msg}. File may be corrupted.",
                    e.doc, e.pos
                ) from e
        return data

    def read_json_streaming_first_framekey(self, filename: Path | str, encoding: str = "ISO-8859-1") -> dict[str, Any]:
        """
        Stream-parse a large ImageJ JSON file to find the first FrameKey entry without loading the entire file.

        This is much more memory-efficient for large metadata files (>100MB) that contain
        per-frame information, as it stops parsing once the first FrameKey is found.

        Args:
            filename (str): Path to the ImageJ JSON metadata file
            encoding (str): File encoding, defaults to ISO-8859-1 for ImageJ compatibility

        Returns:
            dict: Dictionary containing the first FrameKey and its metadata
                  Format: {framekey_name: {metadata_dict}}

        Raises:
            ValueError: If no FrameKey is found in the file
            json.JSONDecodeError: If the JSON is malformed before finding a FrameKey
        """
        import re

        with open(filename, "r", encoding=encoding) as f:
            buffer = ""
            framekey_pattern = re.compile(r'"(FrameKey-\d+-\d+-\d+)"\s*:\s*\{')
            framekey_name = None
            brace_depth = 0
            framekey_value_start = -1
            found_start = False

            # Read file in chunks
            chunk_size = 8192
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break

                buffer += chunk

                # Look for FrameKey pattern if we haven't found one yet
                if not found_start:
                    match = framekey_pattern.search(buffer)
                    if match:
                        framekey_name = match.group(1)
                        found_start = True
                        # The opening brace position is at match.end() - 1
                        framekey_value_start = match.end() - 1
                        brace_depth = 1  # We've seen the opening brace

                if found_start:
                    # Count braces from where we left off
                    start_pos = max(0, len(buffer) - len(chunk) - 1)
                    for i in range(start_pos, len(buffer)):
                        char = buffer[i]
                        if i > framekey_value_start:  # Only count after the opening brace
                            if char == '{':
                                brace_depth += 1
                            elif char == '}':
                                brace_depth -= 1
                                if brace_depth == 0:
                                    # Found complete FrameKey entry
                                    framekey_end = i + 1
                                    framekey_json = buffer[framekey_value_start:framekey_end].strip()

                                    try:
                                        framekey_data = json.loads(framekey_json)
                                        return {framekey_name: framekey_data}
                                    except json.JSONDecodeError as e:
                                        raise json.JSONDecodeError(
                                            f"Error parsing FrameKey '{framekey_name}': {e.msg}",
                                            e.doc, e.pos
                                        ) from e

                # Don't truncate buffer if we're actively parsing
                # Only truncate if we haven't found the start yet
                if not found_start and len(buffer) > chunk_size * 2:
                    buffer = buffer[-chunk_size:]

        raise ValueError("No FrameKey found in JSON file")

    def get_num_pages_in_TIF(self, filename: Path | str) -> int:
        """
        Get the number of frames in a TIFF file without loading the entire file.

        Args:
            filename (str): The name of the tif file to load.

        Returns:
            n_pages (int): number of frames in TIFF file.
        """
        with tifffile.TiffFile(
            filename, is_ome=False, is_mmstack=False, is_imagej=False
        ) as tif:
            return len(tif.pages)

    def metadata_reader_imageJ(self, filename: Path | str, return_exposure: bool = False) -> tuple[int, int, int, int] | tuple[int, int, int, int, float]:
        """
        Loads metadata from an imageJ json file.
        NB ImageJ starts its ROIs at (0,0), like Python

        Uses streaming parser for large files (>10MB) to avoid loading entire file into memory.

        Args:
            filename (str): The name of the json file to load.
            return_exposure (bool): If True, also return exposure time. Default False.

        Returns:
            x_coord (int): starting x_coord pixel.
            y_coord (int): starting y_coord pixel.
            width (int): width
            height (int): height
            exposure_ms (float): exposure time in ms (only if return_exposure=True)
        """
        # Check file size to decide on parsing strategy
        file_size = Path(filename).stat().st_size

        if file_size > 10 * 1024 * 1024:  # > 10MB
            # Use streaming parser for large files
            data = self.read_json_streaming_first_framekey(filename)
            key = list(data.keys())[0]  # Already have the first FrameKey
        else:
            # Use standard parser for small files
            data = self.read_json(filename)
            key = np.sort([x for x in data.keys() if "FrameKey" in x])[0]

        metadatadict = data[key]
        ROI = metadatadict["ROI"].split("-")
        # ROI format from ImageJ/MicroManager is: x-y-width-height, matching
        # MMCore.getROI(label, x, y, xSize, ySize)'s own parameter order
        # (https://javadoc.scijava.org/Micro-Manager-Core/mmcorej/CMMCore.html).
        x_coord = int(ROI[0])  # left (column start)
        y_coord = int(ROI[1])  # top (row start)
        width = int(ROI[2])  # extent in x-direction (columns)
        height = int(ROI[3])  # extent in y-direction (rows)

        if return_exposure:
            exposure_ms = float(metadatadict.get("Exposure-ms", 0.0))
            return x_coord, y_coord, width, height, exposure_ms

        return x_coord, y_coord, width, height

    def metadata_nframes_reader_imageJ(self, filename: Path | str) -> int:
        """
        Loads metadata from an imageJ json file.
        NB ImageJ starts its ROIs at (0,0), like Python

        Args:
            filename (str): The name of the json file to load.

        Returns:
            n_frames (int): intended n_frames.
        """
        data = self.read_json(filename)
        n_frames = int(data["Summary"]["IntendedDimensions"]["time"])
        return n_frames

    def metadata_reader_Thorlabs(self, filename: Path | str) -> tuple[int, int, int, int]:
        """
        Loads metadata from a json file.

        Args:
            filename (str): The name of the json file to load.

        Returns:
            x_coord (int): starting x_coord pixel.
            y_coord (int): starting y_coord pixel.
            width (int): width
            height (int): height
        """
        data = self.read_json(filename)
        x_coord = int(data["ROIOriginX_pixels"])
        y_coord = int(data["ROIOriginY_pixels"])
        width = int(data["ROIWidth_pixels"])
        height = int(data["ROIHeight_pixels"])
        return x_coord, y_coord, width, height

    def _read_tiff_robust(self, file_path, frames, dtype="float32"):
        """
        Robust frame-by-frame TIFF reader that skips corrupted frames.

        This method attempts to read each frame individually, skipping any frames
        that fail to load due to corruption or incomplete data.

        Args:
            file_path (str): Path to the TIFF file
            frames (list or None): List of frame indices to read, or None for all frames
            dtype (str): Data type for output array

        Returns:
            numpy.ndarray: Array of successfully loaded frames

        Raises:
            RuntimeError: If no frames can be loaded successfully
        """
        logger.info(f"Opening TIFF with robust frame-by-frame reader...")

        with tifffile.TiffFile(file_path) as tif:
            n_frames = len(tif.pages)

            # Determine which frames to attempt
            if frames is None:
                frame_indices = list(range(n_frames))
            else:
                frame_indices = list(frames)

            logger.info(f"Total frames in file: {n_frames}")
            logger.info(f"Attempting to load {len(frame_indices)} frame(s)")

            # Try to read first good frame to get shape
            first_frame = None
            first_idx = None
            for idx in frame_indices:
                try:
                    first_frame = np.asarray(tif.pages[idx].asarray(), dtype=dtype)
                    first_idx = idx
                    break
                except Exception as e:
                    logger.warning(f"  Frame {idx}: FAILED ({type(e).__name__})")
                    continue

            if first_frame is None:
                raise RuntimeError(f"Could not load any frames from {file_path}")

            # Pre-allocate array
            shape = (len(frame_indices),) + first_frame.shape
            images = np.zeros(shape, dtype=dtype)

            successful_frames = []
            failed_frames = []

            # Load all frames
            for i, idx in enumerate(frame_indices):
                try:
                    if idx == first_idx:
                        # Already loaded
                        images[i] = first_frame
                    else:
                        # Read directly from page without keyframe
                        images[i] = np.asarray(tif.pages[idx].asarray(), dtype=dtype)
                    successful_frames.append(idx)
                except Exception as e:
                    # Frame is corrupted, fill with zeros
                    logger.info(f"  Frame {idx}: FAILED ({type(e).__name__}), filling with zeros")
                    failed_frames.append(idx)
                    images[i] = np.zeros(first_frame.shape, dtype=dtype)

            logger.info(f"Successfully loaded: {len(successful_frames)}/{len(frame_indices)} frames")
            if failed_frames:
                logger.warning(f"Failed frames (filled with zeros): {failed_frames}")

            # Return single frame if only one requested
            if len(frame_indices) == 1:
                return images[0]

            return images

    def read_hyperstack(self, file_path: Path | str, dtype: str = "float32") -> tuple[NDArray[np.float32], str | None]:
        """Read an ImageJ hyperstack TIFF preserving TZCYX dimensions.

        Unlike read_tiff (which disables ImageJ metadata parsing), this reads
        with ImageJ metadata so the returned array keeps its hyperstack shape
        (e.g. T, Z, C, Y, X).

        Args:
            file_path (str): Path to the TIFF file.
            dtype (str): Output dtype (default: "float32").

        Returns:
            numpy.ndarray: Array with shape matching the ImageJ hyperstack axes.
            str or None: Axes string (e.g. 'TZCYX') if available.
        """
        with tifffile.TiffFile(file_path) as tif:
            data = tif.asarray()
            axes = tif.series[0].axes if tif.series else None
            return data.astype(dtype), axes

    def read_tiff(self, file_path: Path | str, frame: int | list[int] | None = None, dtype: str = "float32", memmap: bool = True) -> NDArray[np.float32]:
        """
        Read a TIFF file using the tifffile library with memory mapping support.

        Args:
            file_path (str): The path to the TIFF file to be read.
            frame (int): if not None, loads a single frame
            dtype (str): Data type for output array. Default "float32" for 50% memory reduction.
            memmap (bool): Use memory mapping for large files. Default True.

        Returns:
            image (numpy.ndarray): The image data from the TIFF file.
        """
        try:
            if isinstance(frame, type(None)):
                # Read entire TIFF stack
                if memmap:
                    with tifffile.TiffFile(
                        file_path, is_ome=False, is_mmstack=False, is_imagej=False
                    ) as tif:
                        image = tif.asarray(out="memmap").astype(dtype)
                else:
                    image = np.asarray(
                        imread(
                            file_path, is_ome=False, is_mmstack=False, is_imagej=False
                        ),
                        dtype=dtype,
                    )
            else:
                # Read specific frame(s)
                if hasattr(frame, "__len__"):
                    # Multiple frames
                    if memmap:
                        with tifffile.TiffFile(
                            file_path, is_ome=False, is_mmstack=False, is_imagej=False
                        ) as tif:
                            image = tif.asarray(key=frame, out="memmap").astype(dtype)
                    else:
                        image = np.asarray(
                            imread(
                                file_path,
                                key=frame,
                                is_ome=False,
                                is_mmstack=False,
                                is_imagej=False,
                            ),
                            dtype=dtype,
                        )
                else:
                    # Single frame
                    if memmap:
                        with tifffile.TiffFile(
                            file_path, is_ome=False, is_mmstack=False, is_imagej=False
                        ) as tif:
                            image = tif.asarray(key=int(frame), out="memmap").astype(
                                dtype
                            )
                    else:
                        image = np.asarray(
                            imread(
                                file_path,
                                key=int(frame),
                                is_ome=False,
                                is_mmstack=False,
                                is_imagej=False,
                            ),
                            dtype=dtype,
                        )
        except IndexError:
            raise  # out-of-range frame is an intentional EOF signal — don't recover
        except Exception as e:
            # Fallback to non-memmap if memory mapping fails
            logger.info(f"Memory mapping failed for {file_path}, falling back to standard loading: {e}")
            if isinstance(frame, type(None)):
                # Try standard loading for all frames
                try:
                    image = np.asarray(
                        imread(file_path, is_ome=False, is_mmstack=False, is_imagej=False),
                        dtype=dtype,
                    )
                except Exception as e2:
                    logger.warning(f"Standard loading also failed: {e2}")
                    logger.info("Attempting frame-by-frame recovery...")
                    image = self._read_tiff_robust(file_path, None, dtype)
            else:
                if hasattr(frame, "__len__"):
                    # Try standard loading for frame list
                    try:
                        image = np.asarray(
                            imread(
                                file_path,
                                key=frame,
                                is_ome=False,
                                is_mmstack=False,
                                is_imagej=False,
                            ),
                            dtype=dtype,
                        )
                    except Exception as e2:
                        logger.warning(f"Standard loading of frame list failed: {e2}")
                        logger.info("Attempting frame-by-frame recovery...")
                        image = self._read_tiff_robust(file_path, frame, dtype)
                else:
                    # Try standard loading for single frame
                    try:
                        image = np.asarray(
                            imread(
                                file_path,
                                key=int(frame),
                                is_ome=False,
                                is_mmstack=False,
                                is_imagej=False,
                            ),
                            dtype=dtype,
                        )
                    except IndexError:
                        raise  # out-of-range frame is an intentional EOF signal — don't recover
                    except Exception as e2:
                        logger.warning(f"Standard loading of frame {frame} failed: {e2}")
                        logger.info("Attempting frame-by-frame recovery...")
                        image = self._read_tiff_robust(file_path, [int(frame)], dtype)
        return image

    def get_n_frames(self, file_path: Path | str) -> int:
        """Return the number of frames in a TIFF file without loading pixel data.

        Args:
            file_path (str): Path to the TIFF file.

        Returns:
            int: Number of frames (pages).
        """
        with tifffile.TiffFile(
            file_path, is_ome=False, is_mmstack=False, is_imagej=False
        ) as tif:
            return len(tif.pages)

    def read_tiff_tophotoelectrons(
        self,
        file_path: Path | str,
        dtype: str = "double",
        gain_map: float | NDArray[np.float32] = 1.0,
        offset_map: float | NDArray[np.float32] = 0.0,
        rqe: float | NDArray[np.float32] = 1.0,
        frame: int | list[int] | None = None,
    ) -> NDArray:
        """
        Read a TIFF file using the skimage library.
        Use camera parameters to convert output to photoelectrons
        This uses the formula (eqn 3) of Lin et al (Lin, R., Clowsley, A. H.,
        Jayasinghe, I. D., Baddeley, D. & Soeller, C.
        Opt. Express, 25, 11701–11716 (2017))

        Args:
            file_path (str): The path to the TIFF file to be read.
            smoothing_function (type): function, args to smooth data
            dtype (str): data type to read out
            gain_map (matrix, or float): gain map. Assumes units of ADU/photoelectrons
            offset_map (matrix, or float): offset map. Assumes units of ADU
            rqe (matrix, or float): relative quantum yield map.
            frame (int, optional): if not None, loads a single frame

        Returns:
            raw_data (numpy.ndarray): The raw image data from the TIFF file.
            data (numpy.ndarray): The photoelectron data from the TIFF file.
            smoothed_data (np.ndarray): Smoothed data for use in initial guesses etc.
            weights_map (np.ndarray): Weights for fitting of data.
        """
        # Use skimage's imread function to read the TIFF file
        # specifying the 'tifffile' plugin explicitly
        data = self.read_tiff(file_path, dtype=dtype, frame=frame)
        if type(gain_map) is not float:

            if data.shape[-2:] != gain_map.shape:
                logger.info("Gain and offset map not compatible with image dimensions. Defaulting to gain of 1 and offset of 0.")
                gain_map = 1.0
                offset_map = 0.0

        rqe_is_array = type(rqe) is not float
        if type(gain_map) is not float:
            if len(data.shape) > 2:
                rqe_term = rqe[np.newaxis, :, :] if rqe_is_array else rqe
                photoelectron_data = np.divide(
                    np.divide(
                        np.subtract(data, offset_map[np.newaxis, :, :]),
                        gain_map[np.newaxis, :, :],
                    ),
                    rqe_term,
                )
            else:
                photoelectron_data = np.divide(
                    np.divide(np.subtract(data, offset_map), gain_map), rqe
                )
        else:
            photoelectron_data = np.divide(
                np.divide(np.subtract(data, offset_map), gain_map), rqe
            )
        return photoelectron_data

    def convert_to_photoelectrons(
        self,
        raw_data: NDArray,
        gain_map: float | NDArray[np.float32] = 1.0,
        offset_map: float | NDArray[np.float32] = 0.0,
        rqe: float | NDArray[np.float32] = 1.0,
    ) -> NDArray:
        """
        Convert raw ADU data to photoelectrons using camera parameters.
        Memory-efficient version that processes data in-place when possible.

        Args:
            raw_data (np.ndarray): Raw camera data in ADU
            gain_map (matrix or float): Gain map. Units: ADU/photoelectrons
            offset_map (matrix or float): Offset map. Units: ADU
            rqe (matrix or float): Relative quantum efficiency map

        Returns:
            np.ndarray: Photoelectron data
        """
        # Check if calibration maps are arrays
        gain_is_array = not isinstance(gain_map, (int, float))
        offset_is_array = not isinstance(offset_map, (int, float))
        rqe_is_array = not isinstance(rqe, (int, float))

        # Validate array shapes if any are arrays
        if gain_is_array and raw_data.shape[-2:] != gain_map.shape:
            logger.info("Gain and offset map not compatible with image dimensions. " "Defaulting to gain of 1 and offset of 0.")
            gain_map = 1.0
            offset_map = 0.0
            gain_is_array = False
            offset_is_array = False

        # Determine if we need 3D broadcasting (for stacks)
        is_3d = len(raw_data.shape) > 2

        # Build the computation step by step with proper broadcasting
        # Step 1: Subtract offset
        if offset_is_array and is_3d:
            result = np.subtract(raw_data, offset_map[np.newaxis, :, :])
        else:
            result = np.subtract(raw_data, offset_map)

        # Step 2: Divide by gain
        if gain_is_array and is_3d:
            result = np.divide(result, gain_map[np.newaxis, :, :])
        else:
            result = np.divide(result, gain_map)

        # Step 3: Divide by RQE
        if rqe_is_array and is_3d:
            result = np.divide(result, rqe[np.newaxis, :, :])
        else:
            result = np.divide(result, rqe)

        return result

    def apply_smoothing(self, data: NDArray, smoothing_function: Any, dtype: str = "double") -> NDArray:
        """
        Apply smoothing function to data.

        Args:
            data (np.ndarray): Input data to smooth
            smoothing_function: Smoothing function object with args and data_arg
            dtype (str): Output data type

        Returns:
            np.ndarray: Smoothed data
        """
        # If no smoothing function provided, return data as-is
        if smoothing_function is None:
            return data.astype(dtype)

        smoothed_data = data.copy()

        smoothing_args = dict(smoothing_function.args)  # copy — do not mutate the original
        smoothing_args[smoothing_function.data_arg] = smoothed_data
        smoothed_data = smoothing_function.smoothing_function(**smoothing_args)

        return smoothed_data.astype(dtype)

    def generate_weights(
        self, smoothed_data: NDArray, read_noise: float | NDArray[np.float32] = 1.0, hot_pixel_threshold: float = 20, dtype: str = "double"
    ) -> NDArray[np.float64]:
        """
        Generate weights map for fitting from smoothed photoelectron data.

        Args:
            smoothed_data (np.ndarray): Smoothed photoelectron data
            read_noise (matrix or float): Read noise map of the camera
            hot_pixel_threshold (float): Threshold for hot pixel detection
            dtype (str): Output data type

        Returns:
            np.ndarray: Weights map for fitting
        """
        error_data = smoothed_data.copy()
        error_data[error_data < 0] = 0
        error_data = error_data + 1

        if not isinstance(read_noise, (int, float)):
            if len(smoothed_data.shape) > 2:
                error_map = np.add(error_data, np.square(read_noise[np.newaxis, :, :]))
            else:
                error_map = np.add(error_data, np.square(read_noise))
        else:
            error_map = np.add(error_data, np.square(read_noise))

        weights_map = np.power(error_map, -1)

        if not isinstance(read_noise, (int, float)):
            hot_pixels = read_noise > hot_pixel_threshold
            if len(smoothed_data.shape) > 2:
                hot_pixels = np.tile(hot_pixels, (smoothed_data.shape[0], 1, 1))
            weights_map[hot_pixels] = 1e-8

        return weights_map.astype(dtype)

    def write_tiff(self, volume: NDArray, file_path: Path | str, bit: str | type = "double", pixel_size: float = 0.069, photometric: str | None = None) -> None:
        """
        Write a TIFF file using tifffile.

        Args:
            volume (numpy.ndarray): The volume data to be saved as a TIFF file.
            file_path (str): The path where the TIFF file will be saved.
            bit (str or dtype): Bit-depth for the saved TIFF file (default is "double").
            pixel_size (float): Pixel size in microns (default is 0.069).
            photometric (str, optional): Photometric interpretation. Use 'rgb' for
                RGB images (shape should be [..., H, W, 3]). Default is None
                (tifffile default, typically 'minisblack' for grayscale).

        Notes:
            For RGB images, set photometric='rgb' to ensure ImageJ recognizes
            the color channels correctly. The volume should have shape (H, W, 3)
            for single frames or (frames, H, W, 3) for stacks.
        """
        volume = np.asarray(volume, dtype=bit)

        pixel_unit = int(1e6 / pixel_size)
        resolution = (pixel_unit / 1e6, pixel_unit / 1e6)  # pixels per micron

        if photometric == 'rgb':
            # For RGB images, use ImageJ-compatible format
            # Shape should be (T, H, W, 3) for stacks or (H, W, 3) for single frame
            imwrite(
                file_path,
                volume,
                imagej=True,
                photometric='rgb',
                resolution=resolution,
                metadata={'unit': 'um'},
            )
        else:
            # Original behavior for grayscale
            xamount = str(volume.shape[-2])
            yamount = str(volume.shape[-1])
            description = "ImageJ=1.54f\nunit=micron\nmin=" + xamount + "\nmax=" + yamount

            extra_tags = [
                ("ImageDescription", "s", 1, description, True),
                ("XResolution", "i", 2, (pixel_unit, 1000000), True),
                ("YResolution", "i", 2, (pixel_unit, 1000000), True),
                ("ResolutionUnit", "i", 1, True),
            ]

            imwrite(
                file_path,
                volume,
                extratags=extra_tags,
            )

    def save_simulation_results(
        self,
        save_folder: Path | str,
        starting_flag: str,
        default_params: NDArray,
        n_photon_space: NDArray[np.float64],
        fit_RMSE_mean: NDArray[np.float64],
        fit_RMSE_std: NDArray[np.float64],
        pixel_size: float,
        NA: float,
        background_photons: float,
        fit_function_name: Any,
        smoothing_function_name: str,
        smoothing_function_extent: float,
        dye: str,
    ) -> None:
        """
        Saves simulation analysis.

        Args:
            save_folder (string): Folder to save data to.
            starting_flat (string): Starting flag of data saving.
            default_params (array): Array of default parameters to be saved.
            n_photon_space (np.1darray): 1d array of different photon values to save.
            fit_RMSE_mean (np.2darray): 2d array of fit RMSEs to save
            fit_RMSE_std (np.2darray): 2d array of fit stds to save
            pixel_size (float): pixel size of simulations
            NA (float): NA of simulations
            background_photons (float): background of simulations
            fit_function_name (object): fit function name to save
            smoothing_function_name (str): smoothing function name to save
            smoothing_function_extent (int): default extent of smoothing function
            dye (string): dye string
        """
        parameters_to_save = list(
            np.concatenate([np.array(["n_photons"]), default_params])
        )
        means = pl.DataFrame(
            data=np.vstack([n_photon_space, fit_RMSE_mean]).T,
            schema=parameters_to_save,
        )
        stds = pl.DataFrame(
            data=np.vstack([n_photon_space, fit_RMSE_std]).T,
            schema=parameters_to_save,
        )
        if int(pixel_size) == pixel_size:
            px_save = str(int(pixel_size))
        else:
            px_save = str(np.around(pixel_size, 1)).replace(".", "p")
        if int(NA) == NA:
            NA_save = str(int(NA))
        else:
            NA_save = str(np.around(NA, 2)).replace(".", "p")
        if int(background_photons) == background_photons:
            b_save = str(int(background_photons))
        else:
            b_save = str(np.around(background_photons, 2)).replace(".", "p")
        if int(smoothing_function_extent) == smoothing_function_extent:
            sf_e_save = str(int(smoothing_function_extent)).zfill(6)
        else:
            sf_e_save = (
                str(np.around(smoothing_function_extent, 2)).replace(".", "p").zfill(6)
            )

        dyestr = dye.replace("/", "-")
        means.write_csv(
            Path(save_folder) / (
                starting_flag
                + str(fit_function_name)
                + "_smoothingfunction_"
                + str(smoothing_function_name)
                + "_smoothingextent_"
                + sf_e_save
                + "_"
                + dyestr
                + "_"
                + px_save
                + "_pixelsize_"
                + NA_save
                + "_NA_"
                + b_save
                + "_background_RMSE_mean_bootstrapping.csv"
            )
        )
        stds.write_csv(
            Path(save_folder) / (
                starting_flag
                + str(fit_function_name)
                + "_smoothingfunction_"
                + str(smoothing_function_name)
                + "_smoothingextent_"
                + sf_e_save
                + "_"
                + dyestr
                + "_"
                + px_save
                + "_pixelsize_"
                + NA_save
                + "_NA_"
                + b_save
                + "_background_RMSE_std_bootstrapping.csv"
            )
        )
        return
