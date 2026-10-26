#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 11:50:17 2024

@author: jbeckwith
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import sys

import numpy as np
from numpy.typing import NDArray

sys.path.append(str(Path(__file__).parent))


class Mask_Functions:
    """Spatial masking operations for Bayer filter pattern analysis.

    Provides functionality for creating and optimising Bayer-type patterns,
    spatial filtering, and mask operations for multicolour SMLM.
    """

    def __init__(self, camera: str = "ximea", mosaic_unit: NDArray | None = None) -> None:
        """Initialize Mask_Functions class.

        Args:
            camera: Camera model name used to set default ``mosaic_unit``.
                Currently ``"ximea"`` (BGGR) or ``"zwo"`` (RGGB).
                Overridden by an explicit *mosaic_unit* kwarg.
            mosaic_unit: Bayer mosaic pattern array.  If ``None``, taken
                from *camera* defaults.
        """
        import pyS3M.CameraDefaults as CameraDefaults
        config = CameraDefaults.get_camera_config(camera)
        self.mosaic_unit = mosaic_unit if mosaic_unit is not None else config.mosaic_unit

    def pixel_fractions(self, mosaic_unit: NDArray | None = None) -> dict[str, float]:
        """Fraction of the sensor area covered by each pixel type.

        A localisation's fitted amplitude ``A_p`` is the signal pixel type *p* would collect
        if it covered the whole plane, so the signal the sensor detects is
        ``sum_p fraction_p * A_p`` (for a Bayer mosaic, ``A_B/4 + A_G/2 + A_R/4``).

        Args:
            mosaic_unit: Mosaic pattern. If ``None``, uses the instance default.

        Returns:
            Fraction of pixels of each type, keyed in the order of :meth:`get_masks` (and so
            of the fitted amplitudes); the fractions sum to 1.
        """
        if mosaic_unit is None:
            mosaic_unit = self.mosaic_unit
        colours, counts = np.unique(mosaic_unit, return_counts=True)
        return {str(c): n / mosaic_unit.size for c, n in zip(colours, counts)}

    def get_ROI_mask(
        self,
        ROI_x_start: int,
        ROI_y_start: int,
        width: int,
        height: int,
        mosaic_unit: NDArray | None = None,
    ) -> dict[Any, NDArray[np.bool_]]:
        """
        Generates a mask and then reshapes based on ROI.

        Args:
            ROI_x_start (int): An integer saying where the ROI started (x)
            ROI_y_start (int): An integer saying where the ROI started (y)
            width (int): An integer saying how big the image is in the x direction.
            height (int): An integer saying how big the image is in the y direction.

        Returns:
            masks (dict): A dictionary containing the assigned masks.
        """
        if mosaic_unit is None:
            mosaic_unit = self.mosaic_unit
        # Note: get_masks uses (size_x, size_y) but internally treats size_x as height (rows)
        # and size_y as width (columns) due to legacy naming convention
        size_x = ROI_y_start + height  # size_x is actually rows
        size_y = ROI_x_start + width  # size_y is actually columns
        masks = self.get_masks(size_x, size_y, mosaic_unit)
        for colour in masks:
            # Numpy indexing is [row, col] = [y, x]
            masks[colour] = masks[colour][ROI_y_start:, ROI_x_start:]
        return masks

    def get_stacked_masks(
        self, ROI_x_start: int, ROI_y_start: int, width: int, height: int, mosaic_unit: NDArray | None = None
    ) -> NDArray[np.bool_]:
        """Get ROI masks and stack into 3D array for fitting.

        Convenience method that combines get_ROI_mask() with np.dstack() to create
        a 3D mask array suitable for multi-channel fitting operations.

        Args:
            ROI_x_start (int): Starting x coordinate (column) of ROI
            ROI_y_start (int): Starting y coordinate (row) of ROI
            width (int): Width of ROI (pixels)
            height (int): Height of ROI (pixels)
            mosaic_unit (np.ndarray, optional): Mosaic pattern. If None, uses default RGGB pattern.

        Returns:
            np.ndarray: 3D array of shape (height, width, n_channels) containing stacked masks
        """
        if mosaic_unit is None:
            mosaic_unit = self.mosaic_unit

        masks = self.get_ROI_mask(
            ROI_x_start=ROI_x_start,
            ROI_y_start=ROI_y_start,
            width=width,
            height=height,
            mosaic_unit=mosaic_unit,
        )
        return np.dstack([masks[x] for x in masks.keys()])

    def get_masks(self, size_x: int, size_y: int, mosaic_unit: NDArray | None = None) -> dict[Any, NDArray[np.bool_]]:
        """
        Assigns the appropriate masks based on the mosaic unit values.

        Args:
            size_x (int): An integer saying how big the image is in the x direction.
            size_y (int): An integer saying how big the image is in the y direction.
            mosaic_unit (np.2darray): unit of mosaic on the camera

        Returns:
            masks (dict): A dictionary containing the assigned masks.
        """
        if mosaic_unit is None:
            mosaic_unit = self.mosaic_unit
        masks = {}
        default_unit = np.zeros_like(mosaic_unit)
        colours = np.unique(mosaic_unit)

        if not size_x % 2:
            repeat_size_x = int(size_x / 2)
        else:
            repeat_size_x = int(size_x + 1 / 2)

        if not size_y % 2:
            repeat_size_y = int(size_y / 2)
        else:
            repeat_size_y = int(size_y + 1 / 2)

        if mosaic_unit.shape != (size_x, size_y):
            for colour in colours:
                x, y = np.where(mosaic_unit == colour)
                mask = np.zeros_like(default_unit, dtype=bool)
                mask[x, y] = True
                masks[colour] = np.tile(mask, (repeat_size_x, repeat_size_y))
                masks[colour] = masks[colour][:size_x, :size_y]
        else:
            for colour in colours:
                x, y = np.where(mosaic_unit == colour)
                mask = np.zeros_like(default_unit, dtype=bool)
                mask[x, y] = True
                masks[colour] = mask
                masks[colour] = masks[colour][:size_x, :size_y]
        return masks
