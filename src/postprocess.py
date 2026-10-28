"""
    postprocess
    ~~~~~~~~~~~~~~~~~~~~

    Data analysis of localisation lists

    :original authors: Joerg Schnitzbauer, Maximilian Thomas Strauss, 2015-2018

    Updated by jsb92, 2026/02/21
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional
import sys
import numpy as np
import numba
from numpy.typing import NDArray

from scipy import interpolate
from scipy.special import iv

from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import itertools
from collections import OrderedDict

sys.path.append(str(Path(__file__).parent))
from pyS3M.ImportManager import get_module, is_available
from pyS3M.Constants import AnalysisConfig
import pyS3M.lib as lib
import pyS3M.render as render
from threading import Thread
import pyS3M.ProgressUtils as ProgressUtils
from numpy.lib.recfunctions import stack_arrays
import logging
logger = logging.getLogger(__name__)


plt = get_module("matplotlib.pyplot")


def get_index_blocks(locs: np.recarray, width: float, height: float, size: float, callback: Callable | None = None) -> tuple[np.recarray, float, NDArray[np.uint32], NDArray[np.uint32], NDArray[np.uint32], NDArray[np.uint32], int, int]:
    # Sort locs by indices
    x_index = np.uint32(locs.xc / size)
    y_index = np.uint32(locs.yc / size)
    sort_indices = np.lexsort([x_index, y_index])
    locs = locs[sort_indices]
    x_index = x_index[sort_indices]
    y_index = y_index[sort_indices]
    # Allocate block info arrays
    n_blocks_y, n_blocks_x = index_blocks_shape(width, height, size)
    block_starts = np.zeros((n_blocks_y, n_blocks_x), dtype=np.uint32)
    block_ends = np.zeros((n_blocks_y, n_blocks_x), dtype=np.uint32)
    K, L = block_starts.shape
    # Fill in block starts and ends
    thread = Thread(
        target=_fill_index_blocks,
        args=(block_starts, block_ends, x_index, y_index),
    )
    thread.start()
    thread.join()
    return locs, size, x_index, y_index, block_starts, block_ends, K, L


def index_blocks_shape(width: float, height: float, size: float) -> tuple[int, int]:
    """Returns the shape of the index grid, given the movie and grid sizes"""
    n_blocks_x = int(np.ceil(width / size))
    n_blocks_y = int(np.ceil(height / size))
    return n_blocks_y, n_blocks_x


def get_block_locs_at(x: float, y: float, index_blocks: tuple) -> np.recarray:
    locs, size, _, _, block_starts, block_ends, K, L = index_blocks
    x_index = np.uint32(x / size)
    y_index = np.uint32(y / size)
    indices = []
    for k in range(y_index - 1, y_index + 2):
        if 0 <= k < K:
            for l in range(x_index - 1, x_index + 2):
                if 0 <= l < L:
                    indices.append(list(range(block_starts[k, l], block_ends[k, l])))
    indices = list(itertools.chain(*indices))
    return locs[indices]


@numba.jit(nopython=True, nogil=True)
def _fill_index_blocks(block_starts, block_ends, x_index, y_index):
    Y, X = block_starts.shape
    N = len(x_index)
    k = 0
    for i in range(Y):
        for j in range(X):
            k = _fill_index_block(
                block_starts, block_ends, N, x_index, y_index, i, j, k
            )


@numba.jit(nopython=True, nogil=True)
def _fill_index_block(block_starts, block_ends, N, x_index, y_index, i, j, k):
    block_starts[i, j] = k
    while k < N and y_index[k] == i and x_index[k] == j:
        k += 1
    block_ends[i, j] = k
    return k


def picked_locs(
    locs: np.recarray,
    width: float,
    height: float,
    picks: list,
    pick_shape: str,
    pick_size: float | None = None,
    add_group: bool = True,
    callback: Callable | None = None,
    parallel: bool = False,
) -> np.recarray:
    """Finds picked localisations.

    Parameters
    ----------
    locs : np.recarray
        Localization list.
    info : list of dicts
        Metadata of the localisations list.
    picks : list
        List of picks.
    pick_shape : {'Circle', 'Rectangle', 'Polygon'}
        Shape of the pick.
    pick_size : float (default=None)
        Size of the pick. Radius for the circles, width for the
        rectangles, None for the polygons.
    add_group : boolean (default=True)
        True if group id should be added to locs. Each pick will be
        assigned a different id.
    callback : function (default=None)
        Function to display progress. If "console", tqdm is used to
        display the progress. If None, no progress is displayed.
    parallel : bool (default=False)
        Whether to use parallel processing for Rectangle and Circle picks.
        Uses chunk-based processing - picks are distributed across
        worker processes in chunks, not one job per pick.
        Only beneficial for 8+ picks. Supports both Circle and Rectangle shapes.

    Returns
    -------
    picked_locs : list of np.recarrays
        List of np.recarrays, each containing locs from one pick.
    """

    # Use parallel processing for picks if requested
    if parallel and len(picks) >= 8:
        if pick_shape == "Rectangle":
            return _parallel_picked_locs_rectangle(
                locs, width, height, picks, pick_size, add_group, callback
            )
        elif pick_shape == "Circle":
            return _parallel_picked_locs_circle(
                locs, width, height, picks, pick_size, add_group, callback
            )

    if len(picks):
        picked_locs = []
        progress_bar_context = None
        progress = None
        if callback == "console":
            progress_bar_context = ProgressUtils.analysis_progress_bar(
                total=len(picks), desc="Picking locs"
            )
            progress = progress_bar_context.__enter__()

        if pick_shape == "Circle":
            index_blocks = get_index_blocks(locs, width, height, pick_size)
            for i, pick in enumerate(picks):
                x, y = pick
                block_locs = get_block_locs_at(x, y, index_blocks)
                group_locs = lib.locs_at(x, y, block_locs, pick_size)
                if add_group:
                    group = i * np.ones(len(group_locs), dtype=np.int32)
                    group_locs = lib.append_to_rec(group_locs, group, "group")
                group_locs.sort(kind="mergesort", order="frame")
                picked_locs.append(group_locs)

                if callback == "console":
                    progress.update(1)
                elif callback is not None:
                    callback(i + 1)

        elif pick_shape == "Rectangle":
            for i, pick in enumerate(picks):
                (xs, ys), (xe, ye) = pick
                X, Y = lib.get_pick_rectangle_corners(xs, ys, xe, ye, pick_size)
                x_min = min(X)
                x_max = max(X)
                y_min = min(Y)
                y_max = max(Y)
                group_locs = locs[locs.xc > x_min]
                group_locs = group_locs[group_locs.xc < x_max]
                group_locs = group_locs[group_locs.yc > y_min]
                group_locs = group_locs[group_locs.yc < y_max]
                group_locs = lib.locs_in_rectangle(group_locs, X, Y)
                # store rotated coordinates in x_rot and y_rot
                angle = 0.5 * np.pi - np.arctan2((ye - ys), (xe - xs))
                x_shifted = group_locs.xc - xs
                y_shifted = group_locs.yc - ys
                x_pick_rot = x_shifted * np.cos(angle) - y_shifted * np.sin(angle)
                y_pick_rot = x_shifted * np.sin(angle) + y_shifted * np.cos(angle)
                group_locs = lib.append_to_rec(group_locs, x_pick_rot, "x_pick_rot")
                group_locs = lib.append_to_rec(group_locs, y_pick_rot, "y_pick_rot")
                if add_group:
                    group = i * np.ones(len(group_locs), dtype=np.int32)
                    group_locs = lib.append_to_rec(group_locs, group, "group")
                group_locs.sort(kind="mergesort", order="frame")
                picked_locs.append(group_locs)

                if callback == "console":
                    progress.update(1)
                elif callback is not None:
                    callback(i + 1)

        elif pick_shape == "Polygon":
            for i, pick in enumerate(picks):
                X, Y = lib.get_pick_polygon_corners(pick)
                if X is None:
                    if callback == "console":
                        progress.update(1)
                    elif callback is not None:
                        callback(i + 1)
                    continue
                group_locs = locs[locs.xc > min(X)]
                group_locs = group_locs[group_locs.xc < max(X)]
                group_locs = group_locs[group_locs.yc > min(Y)]
                group_locs = group_locs[group_locs.yc < max(Y)]
                group_locs = lib.locs_in_polygon(group_locs, X, Y)
                if add_group:
                    group = i * np.ones(len(group_locs), dtype=np.int32)
                    group_locs = lib.append_to_rec(group_locs, group, "group")
                group_locs.sort(kind="mergesort", order="frame")
                picked_locs.append(group_locs)

                if callback == "console":
                    progress.update(1)
                elif callback is not None:
                    callback(i + 1)

        else:
            raise ValueError(
                "Invalid pick shape. Please choose from 'Circle', 'Rectangle', "
                "'Polygon'."
            )

        # Clean up progress bar
        if progress_bar_context is not None:
            progress_bar_context.__exit__(None, None, None)

        return picked_locs


@numba.jit(nopython=True)
def get_link_groups(locs, d_max, max_dark_time, group):
    """Assumes that locs are sorted by frame"""
    frame = locs.frame
    x = locs.xc
    y = locs.yc
    N = len(x)
    link_group = -np.ones(N, dtype=np.int32)
    current_link_group = -1
    for i in range(N):
        if link_group[i] == -1:  # loc has no group yet
            current_link_group += 1
            link_group[i] = current_link_group
            current_index = i
            next_loc_index_in_group = _get_next_loc_index_in_link_group(
                current_index,
                link_group,
                N,
                frame,
                x,
                y,
                d_max,
                max_dark_time,
                group,
            )
            while next_loc_index_in_group != -1:
                link_group[next_loc_index_in_group] = current_link_group
                current_index = next_loc_index_in_group
                next_loc_index_in_group = _get_next_loc_index_in_link_group(
                    current_index,
                    link_group,
                    N,
                    frame,
                    x,
                    y,
                    d_max,
                    max_dark_time,
                    group,
                )
    return link_group


@numba.jit(nopython=True)
def _get_next_loc_index_in_link_group(
    current_index, link_group, N, frame, x, y, d_max, max_dark_time, group
):
    current_frame = frame[current_index]
    current_x = x[current_index]
    current_y = y[current_index]
    current_group = group[current_index]
    min_frame = current_frame + 1
    for min_index in range(current_index + 1, N):
        if frame[min_index] >= min_frame:
            break
    max_frame = current_frame + max_dark_time + 1
    for max_index in range(min_index, N):
        if frame[max_index] > max_frame:
            break
    else:
        max_index = N
    d_max_2 = d_max**2
    for j in range(min_index, max_index):
        if group[j] == current_group:
            if link_group[j] == -1:
                dx2 = (current_x - x[j]) ** 2
                if dx2 <= d_max_2:
                    dy2 = (current_y - y[j]) ** 2
                    if dy2 <= d_max_2:
                        if dx2 + dy2 <= d_max_2:
                            return j
    return -1


def undrift_from_picked(picked_locs: list[np.recarray], n_frames: int) -> np.recarray:
    """Finds drift from picked localisations. Note that unlike other
    undrifting functions, this function does not return undrifted
    localisations but only drift."""

    drift_x = _undrift_from_picked_coordinate(picked_locs, n_frames, "xc")
    drift_y = _undrift_from_picked_coordinate(picked_locs, n_frames, "yc")

    # A rec array to store the applied drift
    drift = (drift_x, drift_y)
    drift = np.rec.array(drift, dtype=[("xc", "f"), ("yc", "f")])
    return drift


def _undrift_from_picked_coordinate(picked_locs, n_frames, coordinate):
    """Calculates drift in a given coordinate.

    Parameters
    ----------
    picked_locs : list
        List of np.recarrays with locs for each pick.
    n_frames : number of frames
    coordinate : {"x", "y", "z"}
        Spatial coordinate where drift is to be found.

    Returns
    -------
    drift_mean : np.array
        Average drift across picks for all frames
    """

    n_picks = len(picked_locs)

    # Drift per pick per frame
    drift = np.empty((n_picks, n_frames))
    drift.fill(np.nan)

    # Remove center of mass offset
    for i, locs in enumerate(picked_locs):
        coordinates = getattr(locs, coordinate)
        drift[i, locs.frame.astype(int)] = coordinates - np.mean(coordinates)

    # Mean drift over picks
    drift_mean = np.nanmean(drift, 0)
    # Square deviation of each pick's drift to mean drift along frames
    sd = (drift - drift_mean) ** 2
    # Mean of square deviation for each pick
    msd = np.nanmean(sd, 1)
    # New mean drift over picks
    # where each pick is weighted according to its msd
    nan_mask = np.isnan(drift)
    drift = np.ma.MaskedArray(drift, mask=nan_mask)
    drift_mean = np.ma.average(drift, axis=0, weights=1 / msd)
    drift_mean = drift_mean.filled(np.nan)

    # Linear interpolation for frames without localisations
    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    nans, nonzero = nan_helper(drift_mean)
    drift_mean[nans] = np.interp(nonzero(nans), nonzero(~nans), drift_mean[~nans])
    return drift_mean


def _parallel_picked_locs_rectangle(
    locs, width, height, picks, pick_size, add_group=True, callback=None
):
    """
    Parallelised version of rectangle picking for picked_locs.

    Uses ThreadPoolExecutor with shared memory for efficient parallel processing.
    Much faster than ProcessPoolExecutor due to reduced serialization overhead.

    Args:
        locs: Localisation data
        width: Image width
        height: Image height
        picks: List of rectangle picks in format [((xs,ys), (xe,ye)), ...]
        pick_size: Size of the pick region
        add_group: Whether to add group IDs to localisations
        callback: Progress callback function

    Returns:
        List of localisation arrays, one per pick
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import multiprocessing as mp
    import math

    if len(picks) == 0:
        return []

    # Only use parallel processing if we have enough picks to make it worthwhile
    if len(picks) < 8:
        return _serial_picked_locs_rectangle(
            locs, width, height, picks, pick_size, add_group, callback
        )

    try:
        # Calculate optimal thread count - use more threads since they're lightweight
        n_threads = min(mp.cpu_count() * 2, len(picks), 16)  # Up to 16 threads

        # Set up progress tracking
        progress_bar_context = None
        progress = None
        total_completed = 0

        if callback == "console":
            import pyS3M.ProgressUtils as ProgressUtils

            progress_bar_context = ProgressUtils.analysis_progress_bar(
                total=len(picks), desc="Picking locs (parallel threads)"
            )
            progress = progress_bar_context.__enter__()

        try:
            # Process individual picks with ThreadPool (no chunking needed)
            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                # Submit individual pick processing jobs
                future_to_index = {}
                for i, pick in enumerate(picks):
                    future = executor.submit(
                        _process_single_rectangle_pick,
                        locs,
                        i,
                        pick,
                        pick_size,
                        add_group,
                    )
                    future_to_index[future] = i

                # Collect results as they complete (more efficient than waiting sequentially)
                picked_locs = [None] * len(picks)

                for future in as_completed(future_to_index):
                    try:
                        original_idx, result = future.result()
                        picked_locs[original_idx] = result

                        total_completed += 1

                        # Update progress
                        if callback == "console" and progress:
                            progress.update(1)
                        elif callback is not None:
                            callback(total_completed)

                    except Exception as e:
                        # Handle individual pick failures gracefully
                        original_idx = future_to_index[future]
                        logger.warning(f"Warning: Processing pick {original_idx} failed with error: {e}")
                        picked_locs[original_idx] = np.array([], dtype=locs.dtype).view(
                            np.recarray
                        )

                        if callback == "console" and progress:
                            progress.update(1)
                        elif callback is not None:
                            callback(total_completed)

            # Fill any remaining None positions (shouldn't happen, but safety check)
            for i in range(len(picked_locs)):
                if picked_locs[i] is None:
                    picked_locs[i] = np.array([], dtype=locs.dtype).view(np.recarray)

            return picked_locs

        finally:
            # Cleanup progress bar
            if progress_bar_context:
                progress_bar_context.__exit__(None, None, None)

    except Exception as e:
        logger.warning(f"Parallel processing failed ({e}), falling back to serial processing")
        # Fall back to serial processing
        return _serial_picked_locs_rectangle(
            locs, width, height, picks, pick_size, add_group, callback
        )


def _process_single_rectangle_pick(locs, original_idx, pick, pick_size, add_group):
    """
    Process a single rectangle pick (efficient thread-based processing).

    Args:
        locs: Localisation data (shared in memory across threads)
        original_idx: Original index of this pick
        pick: Rectangle pick in format ((xs,ys), (xe,ye))
        pick_size: Size of the pick region
        add_group: Whether to add group IDs

    Returns:
        Tuple of (original_index, filtered_localisations)
    """
    try:
        # Import required modules
        import pyS3M.lib as lib
        import numpy as np

        # Process the single pick
        (xs, ys), (xe, ye) = pick

        # Get rectangle corners
        X, Y = lib.get_pick_rectangle_corners(xs, ys, xe, ye, pick_size)
        x_min, x_max = min(X), max(X)
        y_min, y_max = min(Y), max(Y)

        # Filter localisations by bounding box (fast pre-filter)
        group_locs = locs[
            (locs.xc > x_min)
            & (locs.xc < x_max)
            & (locs.yc > y_min)
            & (locs.yc < y_max)
        ]

        # Apply precise rectangle filtering
        group_locs = lib.locs_in_rectangle(group_locs, X, Y)

        # Add rotated coordinates
        angle = 0.5 * np.pi - np.arctan2((ye - ys), (xe - xs))
        x_shifted = group_locs.xc - xs
        y_shifted = group_locs.yc - ys
        x_pick_rot = x_shifted * np.cos(angle) - y_shifted * np.sin(angle)
        y_pick_rot = x_shifted * np.sin(angle) + y_shifted * np.cos(angle)

        group_locs = lib.append_to_rec(group_locs, x_pick_rot, "x_pick_rot")
        group_locs = lib.append_to_rec(group_locs, y_pick_rot, "y_pick_rot")

        # Add group ID if requested (use original index for group ID)
        if add_group:
            group = original_idx * np.ones(len(group_locs), dtype=np.int32)
            group_locs = lib.append_to_rec(group_locs, group, "group")

        # Sort by frame
        group_locs.sort(kind="mergesort", order="frame")

        return (original_idx, group_locs)

    except Exception as e:
        # Return empty result on error
        return (original_idx, np.array([], dtype=locs.dtype).view(np.recarray))


def _parallel_picked_locs_circle(
    locs, width, height, picks, pick_size, add_group=True, callback=None
):
    """
    Parallelised version of circle picking for picked_locs.

    Uses ThreadPoolExecutor with shared memory for efficient parallel processing.
    Much faster than ProcessPoolExecutor due to reduced serialization overhead.

    Args:
        locs: Localisation data
        width: Image width
        height: Image height
        picks: List of circle picks in format [(x, y), ...]
        pick_size: Radius of the pick circles
        add_group: Whether to add group IDs to localisations
        callback: Progress callback function

    Returns:
        List of localisation arrays, one per pick
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import multiprocessing as mp
    import math

    if len(picks) == 0:
        return []

    # Only use parallel processing if we have enough picks to make it worthwhile
    if len(picks) < 8:
        return _serial_picked_locs_circle(
            locs, width, height, picks, pick_size, add_group, callback
        )

    try:
        # Calculate optimal thread count - use more threads since they're lightweight
        n_threads = min(mp.cpu_count() * 2, len(picks), 16)  # Up to 16 threads

        # Set up progress tracking
        progress_bar_context = None
        progress = None
        total_completed = 0

        if callback == "console":
            import pyS3M.ProgressUtils as ProgressUtils

            progress_bar_context = ProgressUtils.analysis_progress_bar(
                total=len(picks), desc="Picking locs (parallel threads)"
            )
            progress = progress_bar_context.__enter__()

        try:
            # Process individual picks with ThreadPool (no chunking needed)
            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                # Submit individual pick processing jobs
                future_to_index = {}
                for i, pick in enumerate(picks):
                    future = executor.submit(
                        _process_single_circle_pick,
                        locs,
                        width,
                        height,
                        i,
                        pick,
                        pick_size,
                        add_group,
                    )
                    future_to_index[future] = i

                # Collect results as they complete (more efficient than waiting sequentially)
                picked_locs = [None] * len(picks)

                for future in as_completed(future_to_index):
                    try:
                        original_idx, result = future.result()
                        picked_locs[original_idx] = result

                        total_completed += 1

                        # Update progress
                        if callback == "console" and progress:
                            progress.update(1)
                        elif callback is not None:
                            callback(total_completed)

                    except Exception as e:
                        # Handle individual pick failures gracefully
                        original_idx = future_to_index[future]
                        logger.warning(f"Warning: Processing circle pick {original_idx} failed with error: {e}")
                        picked_locs[original_idx] = np.array([], dtype=locs.dtype).view(
                            np.recarray
                        )

                        if callback == "console" and progress:
                            progress.update(1)
                        elif callback is not None:
                            callback(total_completed)

            # Fill any remaining None positions (shouldn't happen, but safety check)
            for i in range(len(picked_locs)):
                if picked_locs[i] is None:
                    picked_locs[i] = np.array([], dtype=locs.dtype).view(np.recarray)

            return picked_locs

        finally:
            # Cleanup progress bar
            if progress_bar_context:
                progress_bar_context.__exit__(None, None, None)

    except Exception as e:
        logger.warning(f"Parallel processing failed ({e}), falling back to serial processing")
        # Fall back to serial processing
        return _serial_picked_locs_circle(
            locs, width, height, picks, pick_size, add_group, callback
        )


def _process_single_circle_pick(
    locs, width, height, original_idx, pick, pick_size, add_group
):
    """
    Process a single circle pick (efficient thread-based processing).

    Args:
        locs: Localisation data (shared in memory across threads)
        width: Image width
        height: Image height
        original_idx: Original index of this pick
        pick: Circle pick coordinate (x, y)
        pick_size: Radius of the pick circle
        add_group: Whether to add group IDs

    Returns:
        Tuple of (original_index, filtered_localisations)
    """
    try:
        # Import required modules
        import pyS3M.lib as lib
        import numpy as np

        # Process the single pick
        x, y = pick

        # Use spatial index for efficient filtering
        index_blocks = get_index_blocks(locs, width, height, pick_size)
        block_locs = get_block_locs_at(x, y, index_blocks)
        group_locs = lib.locs_at(x, y, block_locs, pick_size)

        # Add group ID if requested (use original index for group ID)
        if add_group:
            group = original_idx * np.ones(len(group_locs), dtype=np.int32)
            group_locs = lib.append_to_rec(group_locs, group, "group")

        # Sort by frame
        group_locs.sort(kind="mergesort", order="frame")

        return (original_idx, group_locs)

    except Exception as e:
        # Return empty result on error
        return (original_idx, np.array([], dtype=locs.dtype).view(np.recarray))


def _serial_picked_locs_circle(
    locs, width, height, picks, pick_size, add_group=True, callback=None
):
    """
    Serial version of circle picking (fallback for parallel processing).

    Args:
        locs: Localisation data
        width: Image width
        height: Image height
        picks: List of circle picks in format [(x, y), ...]
        pick_size: Radius of pick circles
        add_group: Whether to add group IDs
        callback: Progress callback function

    Returns:
        List of localisation arrays, one per pick
    """
    try:
        import pyS3M.lib as lib
    except ImportError:
        raise ImportError("lib module required for circle picking")

    picked_locs = []

    # Set up progress tracking
    progress_bar_context = None
    progress = None
    if callback == "console":
        import pyS3M.ProgressUtils as ProgressUtils

        progress_bar_context = ProgressUtils.analysis_progress_bar(
            total=len(picks), desc="Picking locs (serial circles)"
        )
        progress = progress_bar_context.__enter__()

    try:
        # Create index blocks for efficient spatial queries
        index_blocks = get_index_blocks(locs, width, height, pick_size)

        for i, pick in enumerate(picks):
            try:
                x, y = pick

                # Get localisations in spatial vicinity
                block_locs = get_block_locs_at(x, y, index_blocks)

                # Apply circular filtering
                group_locs = lib.locs_at(x, y, block_locs, pick_size)

                if add_group:
                    group = i * np.ones(len(group_locs), dtype=np.int32)
                    group_locs = lib.append_to_rec(group_locs, group, "group")

                group_locs.sort(kind="mergesort", order="frame")
                picked_locs.append(group_locs)

                # Update progress
                if callback == "console" and progress:
                    progress.update(1)
                elif callback is not None:
                    callback(i + 1)

            except Exception as e:
                logger.warning(f"Warning: Processing pick {i} failed: {e}")
                picked_locs.append(np.array([], dtype=locs.dtype).view(np.recarray))

                # Still update progress on error
                if callback == "console" and progress:
                    progress.update(1)
                elif callback is not None:
                    callback(i + 1)

    finally:
        # Cleanup progress bar
        if progress_bar_context:
            progress_bar_context.__exit__(None, None, None)

    return picked_locs


def _serial_picked_locs_rectangle(
    locs, width, height, picks, pick_size, add_group=True, callback=None
):
    """
    Serial version of rectangle picking (fallback for parallel processing).

    Args:
        locs: Localisation data
        width: Image width
        height: Image height
        picks: List of rectangle picks
        pick_size: Size of pick region
        add_group: Whether to add group IDs
        callback: Progress callback function

    Returns:
        List of localisation arrays, one per pick
    """
    try:
        import pyS3M.lib as lib
    except ImportError:
        raise ImportError("lib module required for rectangle picking")

    picked_locs = []

    # Set up progress tracking
    progress_bar_context = None
    progress = None
    if callback == "console":
        import pyS3M.ProgressUtils as ProgressUtils

        progress_bar_context = ProgressUtils.analysis_progress_bar(
            total=len(picks), desc="Picking locs (serial fallback)"
        )
        progress = progress_bar_context.__enter__()

    try:
        for i, pick in enumerate(picks):
            try:
                (xs, ys), (xe, ye) = pick
                X, Y = lib.get_pick_rectangle_corners(xs, ys, xe, ye, pick_size)
                x_min, x_max = min(X), max(X)
                y_min, y_max = min(Y), max(Y)

                # Filter localisations
                group_locs = locs[
                    (locs.xc > x_min)
                    & (locs.xc < x_max)
                    & (locs.yc > y_min)
                    & (locs.yc < y_max)
                ]
                group_locs = lib.locs_in_rectangle(group_locs, X, Y)

                # Add rotated coordinates
                angle = 0.5 * np.pi - np.arctan2((ye - ys), (xe - xs))
                x_shifted = group_locs.xc - xs
                y_shifted = group_locs.yc - ys
                x_pick_rot = x_shifted * np.cos(angle) - y_shifted * np.sin(angle)
                y_pick_rot = x_shifted * np.sin(angle) + y_shifted * np.cos(angle)

                group_locs = lib.append_to_rec(group_locs, x_pick_rot, "x_pick_rot")
                group_locs = lib.append_to_rec(group_locs, y_pick_rot, "y_pick_rot")

                if add_group:
                    group = i * np.ones(len(group_locs), dtype=np.int32)
                    group_locs = lib.append_to_rec(group_locs, group, "group")

                group_locs.sort(kind="mergesort", order="frame")
                picked_locs.append(group_locs)

                # Update progress
                if callback == "console" and progress:
                    progress.update(1)
                elif callback is not None:
                    callback(i + 1)

            except Exception as e:
                logger.warning(f"Warning: Processing pick {i} failed: {e}")
                picked_locs.append(np.array([], dtype=locs.dtype).view(np.recarray))

                # Still update progress on error
                if callback == "console" and progress:
                    progress.update(1)
                elif callback is not None:
                    callback(i + 1)

    finally:
        # Cleanup progress bar
        if progress_bar_context:
            progress_bar_context.__exit__(None, None, None)

    return picked_locs


def segment_locs_by_rendered_image(
    locs: pd.DataFrame | np.recarray,
    width: float,
    height: float,
    oversampling: int = 8,
    pixel_size_nm: float = 69.0,
    min_area_nm2: float = 100.0,
    min_localisations: int = 100,
    threshold_method: str = "otsu",
    blur_method: str | None = "smooth",
    callback: Callable | str | None = None,
    verbose: bool = False,
    config: AnalysisConfig = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Memory-efficient aggregate detection using image-based segmentation.

    This function replaces memory-intensive DBSCAN clustering with a more efficient
    approach:
    1. Render localisations to a super-resolved image
    2. Apply thresholding (Otsu or Li) to detect objects
    3. Filter objects by area threshold
    4. Extract localisations within valid regions

    This approach is ~10-100x more memory efficient than DBSCAN for large datasets.

    Parameters
    ----------
    locs : pd.DataFrame or np.recarray
        Localization data with at least 'xc', 'yc' columns
    width : float
        Image width in pixels (camera pixels, not super-res)
    height : float
        Image height in pixels (camera pixels, not super-res)
    oversampling : int, optional
        Super-resolution oversampling factor (default: 8)
    pixel_size_nm : float, optional
        Size of one camera pixel in nanometers (default: 69.0 nm)
        Used to convert areas from pixels to physical units.
    min_area_nm2 : float, optional
        Minimum area in square nanometers for valid aggregates (default: 100.0 nm²)
    min_localisations : int, optional
        Minimum number of localisations per aggregate (default: 100)
    threshold_method : str, optional
        Thresholding method: 'otsu', 'li', or 'percentile' (default: 'otsu')
    blur_method : str, optional
        Blur method for rendering: 'smooth', 'gaussian', or None (default: 'smooth')
    callback : callable or str, optional
        Progress callback. If 'console', uses tqdm progress bar.
    verbose : bool, optional
        If True, display diagnostic plots showing the rendered image,
        binary mask, and labeled regions (default: False)

    Returns
    -------
    aggregate_locs : pd.DataFrame or np.recarray
        Localizations within valid aggregates with added columns:
        - 'aggregate_id': Unique ID for each aggregate
        - 'aggregate_area_nm2': Area of the aggregate in nm²
    per_aggregate_stats : pd.DataFrame or np.recarray
        Per-aggregate statistics with one row per aggregate

    Examples
    --------
    >>> # Basic usage
    >>> agg_locs, stats = segment_locs_by_rendered_image(
    ...     locs, width=2048, height=2048, min_area_nm2=100.0
    ... )
    >>>
    >>> # With custom threshold and progress bar
    >>> agg_locs, stats = segment_locs_by_rendered_image(
    ...     locs, width=2048, height=2048,
    ...     threshold_method='li',
    ...     min_localisations=50,
    ...     min_area_nm2=50.0,
    ...     callback='console'
    ... )
    """
    try:
        # Import required modules
        from skimage import filters, measure
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            f"Required module not found: {e}. "
            "Please install scikit-image: pip install scikit-image"
        )

    # Convert to pandas DataFrame if numpy recarray
    if isinstance(locs, np.recarray):
        locs_df = pd.DataFrame(locs)
    else:
        locs_df = locs.copy()

    # Validate required columns
    if "xc" not in locs_df.columns or "yc" not in locs_df.columns:
        raise ValueError("Localizations must have 'xc' and 'yc' columns")

    # Step 1: Render super-resolved image
    if callback == "console" or callback is not None:
        logger.info("Step 1/5: Rendering super-resolved image...")
    if config is not None and config.progress_callback:
        config.progress_callback(0.0, "Step 1/5: Rendering super-resolved image...")
    if config is not None and config.logging_callback:
        config.logging_callback("Step 1/5: Rendering super-resolved image...")

    info = [{
    "Width": width,         # Image width in pixels
    "Height": height,        # Image height in pixels
    "Frames": np.max(locs_df['frame']),      # Total number of frames in the movie
    "Pixelsize": pixel_size_nm,        # pixel size always in nm
    }]

    n_locs, rendered_image = render.render(
        locs=locs_df.to_records(index=False) if isinstance(locs_df, pd.DataFrame) else locs_df,
        oversampling=oversampling,
        info=info,
        blur_method=blur_method,
    )

    # Step 2: Apply thresholding
    if callback == "console" or callback is not None:
        logger.info(f"Step 2/5: Applying {threshold_method} thresholding...")
    if config is not None and config.progress_callback:
        config.progress_callback(0.2, f"Step 2/5: Applying {threshold_method} thresholding...")
    if config is not None and config.logging_callback:
        config.logging_callback(f"Step 2/5: Applying {threshold_method} thresholding...")

    if threshold_method == "otsu":
        threshold = filters.threshold_otsu(rendered_image)
    elif threshold_method == "li":
        threshold = filters.threshold_li(rendered_image)
    elif threshold_method == "percentile":
        threshold = np.percentile(rendered_image[rendered_image > 0], 95)
    else:
        raise ValueError(
            f"Unknown threshold_method: {threshold_method}. "
            "Choose from 'otsu', 'li', or 'percentile'"
        )

    binary_image = rendered_image > threshold

    # Step 3: Label connected components
    if callback == "console" or callback is not None:
        logger.info("Step 3/5: Detecting connected regions...")
    if config is not None and config.progress_callback:
        config.progress_callback(0.4, "Step 3/5: Detecting connected regions...")
    if config is not None and config.logging_callback:
        config.logging_callback("Step 3/5: Detecting connected regions...")

    label_image = measure.label(binary_image)
    regions = measure.regionprops(label_image)

    # Step 4: Filter by area and count localisations
    if callback == "console" or callback is not None:
        logger.info("Step 4/5: Filtering aggregates by size and localisation count...")
    if config is not None and config.progress_callback:
        config.progress_callback(0.6, "Step 4/5: Filtering aggregates by size and localisation count...")
    if config is not None and config.logging_callback:
        config.logging_callback("Step 4/5: Filtering aggregates by size and localisation count...")

    valid_regions = []
    # Calculate super-resolved pixel size
    # If oversampling=8 and camera pixel=69nm, then super-res pixel = 69/8 = 8.625nm
    superres_pixel_size_nm = pixel_size_nm / oversampling

    # Convert min_area from nm² to super-resolved pixels²
    min_area_pixels = min_area_nm2 / (superres_pixel_size_nm ** 2)

    for region in regions:
        # Check area threshold
        if region.area >= min_area_pixels:
            # Create mask for this region
            minr, minc, maxr, maxc = region.bbox

            # Convert to camera pixel coordinates
            x_min = minc / oversampling
            x_max = maxc / oversampling
            y_min = minr / oversampling
            y_max = maxr / oversampling

            # Find localisations in bounding box
            in_bbox = (
                (locs_df["xc"] >= x_min)
                & (locs_df["xc"] <= x_max)
                & (locs_df["yc"] >= y_min)
                & (locs_df["yc"] <= y_max)
            )
            bbox_locs = locs_df[in_bbox]

            # Further filter by actual region mask
            x_sr = ((bbox_locs["xc"] - x_min) * oversampling).astype(int)
            y_sr = ((bbox_locs["yc"] - y_min) * oversampling).astype(int)

            # Clamp coordinates to mask bounds
            x_sr = np.clip(x_sr, 0, maxc - minc - 1)
            y_sr = np.clip(y_sr, 0, maxr - minr - 1)

            # Check which localisations are inside the region
            region_mask = region.image
            in_region = region_mask[y_sr, x_sr]

            n_locs_in_region = np.sum(in_region)

            # Check localisation count threshold
            if n_locs_in_region >= min_localisations:
                valid_regions.append({
                    "region": region,
                    "label": region.label,
                    "area_nm2": region.area * (superres_pixel_size_nm ** 2),
                    "n_localisations": n_locs_in_region,
                })

    # Display diagnostic plots if verbose
    if verbose:
        try:
            import matplotlib.pyplot as plt
            from pyS3M.PlottingBase import AnalysisPlotter

            plotter = AnalysisPlotter()

            # Create a 3-panel figure
            fig, axs = plotter.two_column_plot(
                nrows=3, ncols=1, height_ratios=[1, 1, 1]
            )

            # Panel 1: Rendered super-resolved image
            im1 = plotter.create_image_plot(
                axs[0],
                rendered_image,   # [y, x], as render.render returns it
                cmap="inferno",
                origin="lower",
            )
            plotter.add_colorbar(im1, axs[0], label="Localisations")
            plotter.add_scalebar(axs[0], pixelsize=superres_pixel_size_nm, length_nm=1000, location="lower right")
            axs[0].set_title(f"Step 1: Rendered Image ({len(locs_df)} locs)", fontsize=10)

            # Panel 2: Binary thresholded image
            im2 = plotter.create_image_plot(
                axs[1],
                binary_image.astype(float),  # Convert boolean to float for plotting
                cmap="gray",
                vmin=0,
                vmax=1,
                origin="lower",
            )
            plotter.add_scalebar(axs[1], pixelsize=superres_pixel_size_nm, length_nm=1000, location="lower right")
            axs[1].set_title(
                f"Step 2: Binary Mask (threshold={threshold:.2f}, {threshold_method})",
                fontsize=10
            )

            # Panel 3: Labeled regions with valid aggregates highlighted
            # Create a color image showing valid vs invalid regions
            display_image = np.zeros_like(label_image, dtype=float)
            valid_labels = [r["label"] for r in valid_regions]

            for region in regions:
                if region.label in valid_labels:
                    display_image[label_image == region.label] = 2  # Valid aggregates
                else:
                    display_image[label_image == region.label] = 1  # Rejected regions

            im3 = plotter.create_image_plot(
                axs[2],
                display_image,
                cmap="Set1",
                vmin=0,
                vmax=3,
                origin="lower",
            )
            plotter.add_colorbar(im3, axs[2], label="Region type (1=rejected, 2=valid)")
            plotter.add_scalebar(axs[2], pixelsize=superres_pixel_size_nm, length_nm=1000, location="lower right")
            axs[2].set_title(
                f"Step 3: Labeled Regions ({len(valid_regions)}/{len(regions)} valid)",
                fontsize=10
            )

            plotter.save_or_show(fig, show=True)

        except Exception as e:
            logger.warning(f"Warning: Could not display verbose plots: {e}")
            import traceback
            traceback.print_exc()

    if len(valid_regions) == 0:
        # Provide diagnostic information
        if len(regions) > 0:
            region_areas_nm2 = [r.area * (superres_pixel_size_nm ** 2) for r in regions]
            logger.warning(f"Warning: No valid aggregates found (tried {len(regions)} regions).\n" f"  Threshold value: {threshold:.4f} ({threshold_method} method)\n" f"  Min area threshold: {min_area_nm2:.1f} nm² ({min_area_pixels:.1f} pixels)\n" f"  Min localisations: {min_localisations}\n" f"  Region areas found: min={min(region_areas_nm2):.1f} nm², " f"max={max(region_areas_nm2):.1f} nm², mean={np.mean(region_areas_nm2):.1f} nm²\n" f"Suggestions:\n" f"  - Try threshold_method='li' or 'percentile' (current: '{threshold_method}')\n" f"  - Try reducing min_area_nm2 (current: {min_area_nm2:.1f} nm²)\n" f"  - Try reducing min_localisations (current: {min_localisations})\n" f"  - Use verbose=True to visualize the thresholding")
        else:
            logger.warning(f"Warning: No regions detected after thresholding.\n" f"  Threshold value: {threshold:.4f} ({threshold_method} method)\n" f"  Image stats: min={rendered_image.min():.4f}, max={rendered_image.max():.4f}, " f"mean={rendered_image.mean():.4f}\n" f"Suggestions:\n" f"  - Try threshold_method='li' or 'percentile' (current: '{threshold_method}')\n" f"  - Use verbose=True to visualize the image and threshold")
        # Return empty results
        empty_locs = locs_df.iloc[:0].copy()
        empty_locs["aggregate_id"] = pd.Series(dtype=int)
        empty_locs["aggregate_area_nm2"] = pd.Series(dtype=float)
        empty_stats = pd.DataFrame(columns=["aggregate_id", "area_nm2", "n_localisations"])
        return empty_locs, empty_stats

    if callback == "console" or callback is not None:
        logger.info(f"Found {len(valid_regions)} valid aggregates " f"(from {len(regions)} total regions)")

    # Step 5: Extract localisations and compute statistics
    if callback == "console" or callback is not None:
        logger.info("Step 5/5: Extracting localisations and computing statistics...")
    if config is not None and config.progress_callback:
        config.progress_callback(0.8, "Step 5/5: Extracting localisations and computing statistics...")
    if config is not None and config.logging_callback:
        config.logging_callback("Step 5/5: Extracting localisations and computing statistics...")

    aggregate_locs_list = []
    per_aggregate_stats_list = []

    # Set up progress bar for aggregates
    progress_bar_context = None
    progress = None
    if callback == "console":
        progress_bar_context = ProgressUtils.analysis_progress_bar(
            total=len(valid_regions), desc="Processing aggregates"
        )
        progress = progress_bar_context.__enter__()

    try:
        for i, region_info in enumerate(valid_regions):
            region = region_info["region"]
            minr, minc, maxr, maxc = region.bbox

            # Convert to camera pixel coordinates
            x_min = minc / oversampling
            x_max = maxc / oversampling
            y_min = minr / oversampling
            y_max = maxr / oversampling

            # Find localisations in bounding box
            in_bbox = (
                (locs_df["xc"] >= x_min)
                & (locs_df["xc"] <= x_max)
                & (locs_df["yc"] >= y_min)
                & (locs_df["yc"] <= y_max)
            )
            bbox_locs = locs_df[in_bbox].copy()

            # Further filter by actual region mask
            x_sr = ((bbox_locs["xc"] - x_min) * oversampling).astype(int)
            y_sr = ((bbox_locs["yc"] - y_min) * oversampling).astype(int)

            # Clamp coordinates to mask bounds
            x_sr = np.clip(x_sr, 0, maxc - minc - 1)
            y_sr = np.clip(y_sr, 0, maxr - minr - 1)

            # Check which localisations are inside the region
            region_mask = region.image
            in_region = region_mask[y_sr, x_sr]

            # Get localisations in this aggregate
            aggregate_locs = bbox_locs[in_region].copy()

            # Add aggregate ID and area
            aggregate_locs["aggregate_id"] = i
            aggregate_locs["aggregate_area_nm2"] = region_info["area_nm2"]

            aggregate_locs_list.append(aggregate_locs)

            # Compute per-aggregate statistics
            stats = {
                "aggregate_id": i,
                "area_nm2": region_info["area_nm2"],
                "n_localisations": len(aggregate_locs),
            }

            # Add mean values for numerical columns
            # Use the original column names (without _mean suffix) for compatibility
            # with downstream analysis functions (e.g., Nile Red functions)
            for col in aggregate_locs.columns:
                # Skip columns we don't want to average
                if col in ["aggregate_id", "aggregate_area_nm2", "frame"]:
                    continue

                # Skip error columns (they'll be handled separately)
                if col.endswith("_err"):
                    continue

                if np.issubdtype(aggregate_locs[col].dtype, np.number):
                    # Sum photons instead of averaging
                    if col == "photons":
                        stats[col] = aggregate_locs[col].sum()
                    # Use weighted mean if error column exists
                    elif f"{col}_err" in aggregate_locs.columns:
                        # Weight by 1/error (not 1/error²) as per NileRedFunctions convention
                        errors = aggregate_locs[f"{col}_err"]
                        weights = 1.0 / errors
                        stats[col] = np.average(aggregate_locs[col], weights=weights)

                        # Propagate error: σ_mean = 1 / sqrt(Σ(1/σ_i²))
                        stats[f"{col}_err"] = 1.0 / np.sqrt(np.sum(1.0 / errors**2))
                    else:
                        # Simple mean for columns without errors
                        stats[col] = aggregate_locs[col].mean()

            per_aggregate_stats_list.append(stats)

            # Update progress
            if callback == "console" and progress:
                progress.update(1)
            elif callable(callback):
                callback(i + 1)

    finally:
        if progress_bar_context:
            progress_bar_context.__exit__(None, None, None)

    # Combine results
    aggregate_locs_combined = pd.concat(aggregate_locs_list, ignore_index=True)
    per_aggregate_stats = pd.DataFrame(per_aggregate_stats_list)

    # Add frame column to per_aggregate_stats for compatibility with IO functions
    # Since this is aggregate-level data, we use frame=0 as a placeholder
    if "frame" not in per_aggregate_stats.columns:
        per_aggregate_stats["frame"] = 0

    _done_msg = (f"✓ Complete! Extracted {len(aggregate_locs_combined)} localisations "
                 f"in {len(valid_regions)} aggregates")
    if callback == "console" or callback is not None:
        logger.info(_done_msg)
    if config is not None and config.progress_callback:
        config.progress_callback(1.0, _done_msg)
    if config is not None and config.logging_callback:
        config.logging_callback(_done_msg)

    return aggregate_locs_combined, per_aggregate_stats


def remove_fiducials(
    aggregate_locs: pd.DataFrame,
    per_aggregate_stats: pd.DataFrame,
    n_frames: int,
    A_R_threshold: float | tuple[float, str] | None = None,
    A_G_threshold: float | tuple[float, str] | None = None,
    density_threshold: float = 0.6,
    require_all: bool = False,
    verbose: bool = False,
    config: AnalysisConfig = None,
) -> tuple[pd.DataFrame, pd.DataFrame, NDArray[np.bool_]]:
    """
    Remove fiducial markers from aggregate data based on spectral and density criteria.

    Fiducial markers (e.g. gold nanoparticles) can be identified by two properties:

    1. **Spectral signature**: their mean A_R and/or A_G values fall outside
       the range expected for the fluorescent probe (e.g. Nile Red).
    2. **High density**: fiducials are typically emitting in >60% of frames,
       producing many more localisations per aggregate than transient binders.

    The density criterion uses ``n_localisations / n_frames`` per aggregate.
    Aggregates exceeding the density threshold are flagged as fiducials.

    Parameters
    ----------
    aggregate_locs : pd.DataFrame
        Localisation data with an ``aggregate_id`` column, as returned by
        ``segment_locs_by_rendered_image``.
    per_aggregate_stats : pd.DataFrame
        Per-aggregate statistics with columns including ``aggregate_id``,
        ``n_localisations``, and optionally ``A_R``, ``A_G``.
    n_frames : int
        Total number of frames in the acquisition. Used to compute
        the localisation density per aggregate.
    A_R_threshold : float, tuple, or None, optional
        Threshold for mean A_R. Can be specified as:
        - ``float``: flag aggregates with A_R **above** this value (default direction).
        - ``(value, 'above')``: flag aggregates with A_R >= value.
        - ``(value, 'below')``: flag aggregates with A_R <= value.
        If None, A_R is not used for filtering.
    A_G_threshold : float, tuple, or None, optional
        Threshold for mean A_G. Same format as ``A_R_threshold``.
        If None, A_G is not used for filtering.
    density_threshold : float, optional
        Fraction of frames an aggregate must be localised in to be
        considered a fiducial (default: 0.6, i.e. present in >60% of frames).
    require_all : bool, optional
        If True, an aggregate must satisfy **all** active criteria to be
        removed. If False (default), satisfying **any** single criterion
        is sufficient for removal.
    verbose : bool, optional
        If True, print summary of removed fiducials (default: False).

    Returns
    -------
    filtered_locs : pd.DataFrame
        Localisations with fiducial aggregates removed.
    filtered_stats : pd.DataFrame
        Per-aggregate statistics with fiducial aggregates removed.
    fiducial_mask : np.ndarray of bool
        Boolean mask over ``per_aggregate_stats`` rows: True for fiducials.

    Examples
    --------
    >>> # Remove fiducials with A_R > 0.5 or density > 60%
    >>> filt_locs, filt_stats, mask = remove_fiducials(
    ...     agg_locs, stats, n_frames=10000,
    ...     A_R_threshold=0.5, density_threshold=0.6,
    ... )
    >>>
    >>> # Remove fiducials with A_G above 0.3 AND A_R below 0.4
    >>> filt_locs, filt_stats, mask = remove_fiducials(
    ...     agg_locs, stats, n_frames=10000,
    ...     A_R_threshold=(0.4, 'below'),
    ...     A_G_threshold=(0.3, 'above'),
    ...     density_threshold=0.6,
    ...     require_all=True,
    ... )
    """
    import pandas as pd

    def _parse_threshold(threshold):
        """Parse a threshold into (value, direction).

        Accepts a float (defaults to 'above') or a (value, direction) tuple.
        """
        if isinstance(threshold, (list, tuple)):
            if len(threshold) != 2:
                raise ValueError(
                    f"Threshold tuple must be (value, 'above'|'below'), "
                    f"got length {len(threshold)}"
                )
            value, direction = threshold
            direction = direction.lower()
            if direction not in ("above", "below"):
                raise ValueError(
                    f"Threshold direction must be 'above' or 'below', "
                    f"got '{direction}'"
                )
            return float(value), direction
        return float(threshold), "above"

    def _apply_threshold(values, value, direction):
        """Return boolean mask: True where the criterion is met."""
        if direction == "above":
            return values >= value
        else:
            return values <= value

    # Determine the aggregate ID column name (handle rename from notebook)
    if "aggregate_id" in per_aggregate_stats.columns:
        id_col = "aggregate_id"
    elif "cluster_id" in per_aggregate_stats.columns:
        id_col = "cluster_id"
    else:
        raise ValueError(
            "per_aggregate_stats must contain 'aggregate_id' or 'cluster_id' column"
        )

    # Similarly for n_localisations
    if "n_localisations" in per_aggregate_stats.columns:
        nloc_col = "n_localisations"
    elif "n_locs" in per_aggregate_stats.columns:
        nloc_col = "n_locs"
    else:
        raise ValueError(
            "per_aggregate_stats must contain 'n_localisations' column"
        )

    n_aggregates = len(per_aggregate_stats)

    # Build individual criterion masks (True = flagged as fiducial)
    criteria = []

    # Density criterion
    if density_threshold is not None:
        density = per_aggregate_stats[nloc_col].values / n_frames
        is_dense = density >= density_threshold
        criteria.append(("density >= {:.2f}".format(density_threshold), is_dense))

    # Spectral criteria
    for col, threshold in [("A_R", A_R_threshold), ("A_G", A_G_threshold)]:
        if threshold is not None:
            if col not in per_aggregate_stats.columns:
                raise ValueError(
                    f"{col} threshold specified but '{col}' column not found in "
                    "per_aggregate_stats. Ensure segment_locs_by_rendered_image "
                    f"input data contains {col} values."
                )
            value, direction = _parse_threshold(threshold)
            op = ">=" if direction == "above" else "<="
            mask = _apply_threshold(per_aggregate_stats[col].values, value, direction)
            criteria.append((f"{col} {op} {value}", mask))

    if len(criteria) == 0:
        raise ValueError(
            "At least one criterion must be active. "
            "Set A_R_threshold, A_G_threshold, and/or density_threshold."
        )

    # Combine criteria
    masks = np.column_stack([m for _, m in criteria])
    if require_all:
        fiducial_mask = np.all(masks, axis=1)
    else:
        fiducial_mask = np.any(masks, axis=1)

    n_fiducials = np.sum(fiducial_mask)

    if verbose or (config is not None and config.logging_callback):
        combine_str = "ALL" if require_all else "ANY"
        summary_lines = [
            f"Fiducial removal summary:",
            f"  Total aggregates: {n_aggregates}",
            *[f"  Flagged by {name}: {np.sum(m)}" for name, m in criteria],
            f"  Combined ({combine_str}): {n_fiducials} fiducials",
            f"  Remaining: {n_aggregates - n_fiducials} aggregates",
        ]
        if verbose:
            for line in summary_lines:
                logger.info(line)
        if config is not None and config.logging_callback:
            config.logging_callback("\n".join(summary_lines))

    # Filter out fiducial aggregates
    fiducial_ids = set(per_aggregate_stats.loc[fiducial_mask, id_col].values)
    keep_stats = ~fiducial_mask
    filtered_stats = per_aggregate_stats[keep_stats].reset_index(drop=True)

    # Determine the matching column in aggregate_locs
    if id_col in aggregate_locs.columns:
        locs_id_col = id_col
    elif "aggregate_id" in aggregate_locs.columns:
        locs_id_col = "aggregate_id"
    elif "cluster_id" in aggregate_locs.columns:
        locs_id_col = "cluster_id"
    else:
        raise ValueError(
            "aggregate_locs must contain 'aggregate_id' or 'cluster_id' column"
        )

    keep_locs = ~aggregate_locs[locs_id_col].isin(fiducial_ids)
    filtered_locs = aggregate_locs[keep_locs].reset_index(drop=True)

    if verbose or (config is not None and config.logging_callback):
        n_removed_locs = len(aggregate_locs) - len(filtered_locs)
        msg = (f"  Localisations removed: {n_removed_locs} "
               f"({100 * n_removed_locs / len(aggregate_locs):.1f}%)")
        if verbose:
            logger.info(msg)
        if config is not None and config.logging_callback:
            config.logging_callback(msg)

    return filtered_locs, filtered_stats, fiducial_mask
