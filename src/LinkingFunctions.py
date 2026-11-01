# -*- coding: utf-8 -*-
"""
LinkingFunctions
~~~~~~~~~~~~~~~~

Post-hoc linking of single-molecule localisations across frames.

Repeated detections of the same emitter (blinking) that appear at the same
position within ``r_max`` pixels and within ``max_dark_time`` dark frames are
collapsed into a single improved localisation:

* position (xc, yc)          — inverse-variance weighted mean  (1/σ² weights)
* errors  (xc_err, yc_err)   — combined as 1/√(Σ 1/σ²)
* photons, photoelectrons    — summed
* bg_B, bg_G, bg_R           — summed  (total background accumulated)
* A_B, A_G, A_R, s_x, s_y   — inverse-variance weighted mean using paired _err columns;
                               falls back to simple mean if _err absent
* A_B_err, A_G_err, A_R_err,
  s_x_err, s_y_err           — combined as 1/√(Σ 1/σ²)
* chi_sqr                    — simple mean

:original authors: Joerg Schnitzbauer, Maximilian Thomas Strauss, 2015-2018
                   (from Picasso / Eva_Wong_Code linking_functions.py)
:adapted for pyS3M: jsb92, 2026-03-11
"""

import numpy as np
import numba
import pandas as pd
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level numba JIT helpers (operate on plain 1-D numpy arrays)
# ---------------------------------------------------------------------------

@numba.jit(nopython=True)
def _link_group_count(link_group, n_locs, n_groups):
    result = np.zeros(n_groups, dtype=np.uint32)
    for i in range(n_locs):
        result[link_group[i]] += 1
    return result


@numba.jit(nopython=True)
def _link_group_sum(column, link_group, n_locs, n_groups):
    result = np.zeros(n_groups, dtype=column.dtype)
    for i in range(n_locs):
        result[link_group[i]] += column[i]
    return result


@numba.jit(nopython=True)
def _link_group_mean(column, link_group, n_locs, n_groups, n_locs_per_group):
    group_sum = _link_group_sum(column, link_group, n_locs, n_groups)
    result = np.empty(n_groups, dtype=np.float32)
    for i in range(n_groups):
        result[i] = group_sum[i] / n_locs_per_group[i]
    return result


@numba.jit(nopython=True)
def _link_group_weighted_mean(column, weights, link_group, n_locs, n_groups, n_locs_per_group):
    sum_weights = _link_group_sum(weights, link_group, n_locs, n_groups)
    weighted_col = column * weights
    numerator = _link_group_sum(weighted_col, link_group, n_locs, n_groups)
    result = np.empty(n_groups, dtype=np.float32)
    for i in range(n_groups):
        result[i] = numerator[i] / sum_weights[i] if sum_weights[i] > 0 else 0.0
    return result, sum_weights


@numba.jit(nopython=True)
def _link_group_min_max(column, link_group, n_locs, n_groups):
    min_ = np.empty(n_groups, dtype=column.dtype)
    max_ = np.empty(n_groups, dtype=column.dtype)
    min_[:] = column.max()
    max_[:] = column.min()
    for i in range(n_locs):
        i_ = link_group[i]
        v = column[i]
        if v < min_[i_]:
            min_[i_] = v
        if v > max_[i_]:
            max_[i_] = v
    return min_, max_


# ---------------------------------------------------------------------------
# Core greedy-chain linking (numba JIT, operates on structured recarray)
# ---------------------------------------------------------------------------

@numba.jit(nopython=True)
def _get_next_loc_index(current_index, link_group, N, frame, x, y,
                        d_max, max_dark_time, group):
    current_frame = frame[current_index]
    current_x = x[current_index]
    current_y = y[current_index]
    current_group = group[current_index]
    min_frame = current_frame + 1
    min_index = current_index + 1
    while min_index < N and frame[min_index] < min_frame:
        min_index += 1
    max_frame = current_frame + max_dark_time + 1
    max_index = min_index
    while max_index < N and frame[max_index] <= max_frame:
        max_index += 1
    d_max_2 = d_max ** 2
    for j in range(min_index, max_index):
        if group[j] == current_group and link_group[j] == -1:
            dx2 = (current_x - x[j]) ** 2
            if dx2 <= d_max_2:
                dy2 = (current_y - y[j]) ** 2
                if dx2 + dy2 <= d_max_2:
                    return j
    return -1


@numba.jit(nopython=True)
def _get_link_groups(frame, x, y, group, d_max, max_dark_time):
    """Assign each localisation to a link group.

    Localisations must already be sorted by frame.  The greedy chain
    algorithm starts a new group for each unlinked localisation and
    extends it forward until no neighbour is found within the radius and
    dark-time window.

    Returns
    -------
    link_group : np.ndarray of int32, shape (N,)
    """
    N = len(frame)
    link_group = -np.ones(N, dtype=np.int32)
    current_link_group = -1
    for i in range(N):
        if link_group[i] == -1:
            current_link_group += 1
            link_group[i] = current_link_group
            current_index = i
            next_index = _get_next_loc_index(
                current_index, link_group, N, frame, x, y,
                d_max, max_dark_time, group,
            )
            while next_index != -1:
                link_group[next_index] = current_link_group
                current_index = next_index
                next_index = _get_next_loc_index(
                    current_index, link_group, N, frame, x, y,
                    d_max, max_dark_time, group,
                )
    return link_group


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def link_localisations(
    df: pd.DataFrame,
    n_frames: int,
    r_max: float = 0.05,
    max_dark_time: int = 1,
    remove_ambiguous_lengths: bool = True,
) -> pd.DataFrame:
    """Link Bayer-SMLM localisations from the same emitter across frames.

    Repeated detections of the same blinking emitter that fall within
    ``r_max`` pixels and ``max_dark_time`` dark frames of each other are
    collapsed into a single row with improved statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Localisation table as produced by the SR pipeline.  Must contain
        ``frame``, ``xc``, ``yc``, ``xc_err``, ``yc_err``.  The following
        optional columns are aggregated if present:
        ``photons``, ``photoelectrons`` — summed
        ``bg_B``, ``bg_G``, ``bg_R`` — summed
        ``A_B``,  ``A_G``,  ``A_R``  — averaged (spectral fractions)
        ``s_x``, ``s_y``         — averaged
        ``chi_sqr``              — averaged
    n_frames : int
        Total number of frames in the acquisition (used to remove events
        whose start/end coincides with the first/last frame when
        ``remove_ambiguous_lengths=True``).
    r_max : float
        Maximum linking radius in **pixels**.  A good starting value is
        2–3× the median ``xc_err``.
    max_dark_time : int
        Maximum number of consecutive *dark* frames over which to bridge a
        gap.  ``1`` means only consecutive frames are linked.
    remove_ambiguous_lengths : bool
        If True (default), discard linked events whose first frame is 0 or
        whose last frame is ``n_frames - 1``, because their true on-time is
        unknown.

    Returns
    -------
    linked : pd.DataFrame
        Reduced localisation table with the same columns as ``df`` plus:
        ``n``            — number of raw localisations merged
        ``len``          — frame span (last_frame - first_frame + 1)
        ``photon_rate``  — photons / n  (if photons present)
    """
    if len(df) == 0:
        return df.copy()

    df = df.sort_values("frame").reset_index(drop=True)
    n = len(df)

    frame = df["frame"].to_numpy(dtype=np.int32)
    x = df["xc"].to_numpy(dtype=np.float32)
    y = df["yc"].to_numpy(dtype=np.float32)
    group = np.zeros(n, dtype=np.int32)

    link_group = _get_link_groups(frame, x, y, group, np.float32(r_max), int(max_dark_time))

    n_groups = int(link_group.max()) + 1
    n_ = _link_group_count(link_group, n, n_groups)

    first_frame, last_frame = _link_group_min_max(frame, link_group, n, n_groups)

    columns: OrderedDict = OrderedDict()
    columns["frame"] = first_frame

    # Position: inverse-variance weighted mean
    xc_err = np.abs(df["xc_err"].to_numpy(dtype=np.float32))
    yc_err = np.abs(df["yc_err"].to_numpy(dtype=np.float32))
    # Guard against zero errors
    xc_err = np.where(xc_err > 0, xc_err, np.nanmedian(xc_err[xc_err > 0])).astype(np.float32)
    yc_err = np.where(yc_err > 0, yc_err, np.nanmedian(yc_err[yc_err > 0])).astype(np.float32)

    wx = (1.0 / xc_err ** 2).astype(np.float32)
    wy = (1.0 / yc_err ** 2).astype(np.float32)

    columns["xc"], sum_wx = _link_group_weighted_mean(x, wx, link_group, n, n_groups, n_)
    columns["yc"], sum_wy = _link_group_weighted_mean(y, wy, link_group, n, n_groups, n_)
    columns["xc_err"] = np.sqrt(1.0 / sum_wx).astype(np.float32)
    columns["yc_err"] = np.sqrt(1.0 / sum_wy).astype(np.float32)

    # Summed columns (accumulated over the on-time)
    for col in ("photons", "photoelectrons", "bg_B", "bg_G", "bg_R"):
        if col in df.columns:
            columns[col] = _link_group_sum(
                df[col].to_numpy(dtype=np.float32), link_group, n, n_groups
            )

    # Inverse-variance weighted average for columns that have a paired _err.
    # A more precise single-frame detection gets a higher weight.
    # Falls back to a simple mean when the error column is absent.
    for col in ("A_B", "A_G", "A_R", "s_x", "s_y"):
        err_col = f"{col}_err"
        if col not in df.columns:
            continue
        vals = df[col].to_numpy(dtype=np.float32)
        if err_col in df.columns:
            errs = np.abs(df[err_col].to_numpy(dtype=np.float32))
            med_err = float(np.nanmedian(errs[errs > 0])) if (errs > 0).any() else 1.0
            errs = np.where(errs > 0, errs, med_err).astype(np.float32)
            w = (1.0 / errs ** 2).astype(np.float32)
            columns[col], sum_w = _link_group_weighted_mean(vals, w, link_group, n, n_groups, n_)
            columns[err_col] = np.sqrt(1.0 / sum_w).astype(np.float32)
        else:
            columns[col] = _link_group_mean(vals, link_group, n, n_groups, n_)

    # chi_sqr: simple mean (no error column for this one)
    if "chi_sqr" in df.columns:
        columns["chi_sqr"] = _link_group_mean(
            df["chi_sqr"].to_numpy(dtype=np.float32), link_group, n, n_groups, n_
        )

    # Pass-through: any remaining numeric columns not yet handled are averaged
    # across the linked group so no data is silently discarded.
    handled = {
        "frame", "xc", "yc", "xc_err", "yc_err",
        "photons", "photoelectrons", "bg_B", "bg_G", "bg_R",
        "A_B", "A_G", "A_R", "A_B_err", "A_G_err", "A_R_err",
        "s_x", "s_y", "s_x_err", "s_y_err",
        "chi_sqr",
    }
    for col in df.columns:
        if col in handled or col in columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        columns[col] = _link_group_mean(
            df[col].to_numpy(dtype=np.float32), link_group, n, n_groups, n_
        )

    columns["len"] = (last_frame - first_frame + 1).astype(np.uint32)
    columns["n"] = n_

    if "photons" in columns:
        columns["photon_rate"] = (columns["photons"] / n_).astype(np.float32)

    linked = pd.DataFrame(columns)

    if remove_ambiguous_lengths:
        valid = (first_frame > 0) & (last_frame < n_frames - 1)
        linked = linked[valid].reset_index(drop=True)

    return linked


def joint_spectral_spatial_cluster(
    df: pd.DataFrame,
    spatial_cols: list = None,
    spectral_cols: list = None,
    spatial_err_cols: list = None,
    spectral_err_cols: list = None,
    d_threshold: float = 2.0,
    min_cluster_size: int = 3,
) -> pd.DataFrame:
    """Cluster temporally-linked blink events in joint (x, y, A_R, A_G) space.

    Each pair of blink events is considered to originate from the same physical
    molecule if their joint Mahalanobis distance in position-and-spectrum space
    is below ``d_threshold``:

        d² = (Δx/σx)² + (Δy/σy)² + (ΔA_R/σAR)² + (ΔA_G/σAG)²

    where σ for each dimension is the per-pair combined uncertainty:
        σ_combined² = σ_i² + σ_j²

    This means two blink events at the same spatial position but from different
    dye species (large ΔA_R / σ_AR_combined) will *not* cluster together, even
    though their spatial distance is zero.

    Parameters
    ----------
    df : pd.DataFrame
        Temporally-linked localisation table.  Must contain column ``'n'``
        (number of raw frames merged by ``link_localisations``).
    spatial_cols : list
        Column names for spatial coordinates. Default ['xc', 'yc'].
    spectral_cols : list
        Column names for spectral features. Default ['A_R', 'A_G'].
    spatial_err_cols : list
        Column names for spatial uncertainties. Default ['xc_err', 'yc_err'].
    spectral_err_cols : list
        Column names for spectral uncertainties. Default ['A_R_err', 'A_G_err'].
    d_threshold : float
        Mahalanobis distance gate (in combined-σ units). Default 2.0
        (~1 σ per dimension on average for a 4-D metric).
    min_cluster_size : int
        Clusters with fewer members than this are treated as isolated (label −1).

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with an added column ``'joint_cluster_id'``.
        Isolated blink events have ``joint_cluster_id == -1``.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    from sklearn.neighbors import KDTree

    if spatial_cols is None:
        spatial_cols = ['xc', 'yc']
    if spectral_cols is None:
        spectral_cols = ['A_R', 'A_G']
    if spatial_err_cols is None:
        spatial_err_cols = ['xc_err', 'yc_err']
    if spectral_err_cols is None:
        spectral_err_cols = ['A_R_err', 'A_G_err']

    if 'n' not in df.columns:
        raise ValueError(
            "Input must be temporally linked (column 'n' missing). "
            "Run link_localisations() first."
        )

    all_feat_cols = spatial_cols + spectral_cols
    all_err_cols = spatial_err_cols + spectral_err_cols
    for col in all_feat_cols + all_err_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame.")

    n = len(df)
    feat = df[all_feat_cols].to_numpy(dtype=np.float64)
    err = np.maximum(df[all_err_cols].to_numpy(dtype=np.float64), 1e-12)

    # Spatial KD-tree pre-filter: search radius is conservative upper bound.
    # Any pair that could pass the full 4D test must have spatial distance
    # < d_threshold × sqrt(σx_i² + σx_j²) ≤ d_threshold × sqrt(2) × max(σx).
    # Using sqrt(2) × d_threshold × median(σx) is fast and catches >99% of pairs.
    xy = feat[:, :2]
    median_spatial_err = np.median(err[:, 0])
    r_spatial = d_threshold * np.sqrt(2.0) * median_spatial_err

    logger.debug(f"joint_spectral_spatial_cluster: n={n}, r_spatial={r_spatial:.4f} px")

    tree = KDTree(xy)
    neighbours = tree.query_radius(xy, r=r_spatial)

    # Evaluate full 4D metric for spatial candidate pairs
    row_list, col_list = [], []
    for i, nbrs in enumerate(neighbours):
        for j in nbrs:
            if j <= i:
                continue
            combined_var = err[i] ** 2 + err[j] ** 2   # (4,) array
            delta = feat[i] - feat[j]                   # (4,) array
            d2 = float(np.sum(delta ** 2 / combined_var))
            if d2 < d_threshold ** 2:
                row_list.append(i)
                col_list.append(j)

    if not row_list:
        logger.debug("No pairs within threshold — all blink events isolated.")
        out = df.copy()
        out['joint_cluster_id'] = np.int32(-1)
        return out

    # Symmetric adjacency matrix → connected components (single-linkage clusters)
    rows_arr = np.array(row_list + col_list, dtype=np.int32)
    cols_arr = np.array(col_list + row_list, dtype=np.int32)
    data_arr = np.ones(len(rows_arr), dtype=np.float32)
    adj = csr_matrix((data_arr, (rows_arr, cols_arr)), shape=(n, n))

    _, labels = connected_components(adj, directed=False, return_labels=True)

    # Drop clusters below min_cluster_size → label them −1
    sizes = np.bincount(labels)
    valid_components = np.where(sizes >= min_cluster_size)[0]
    remap = np.full(sizes.shape[0], -1, dtype=np.int32)
    for new_id, old_id in enumerate(valid_components):
        remap[old_id] = new_id
    cluster_ids = remap[labels]

    n_clusters = len(valid_components)
    n_isolated = int((cluster_ids == -1).sum())
    logger.debug(
        f"  {n_clusters} clusters (≥{min_cluster_size} members), "
        f"{n_isolated} isolated blink events"
    )

    out = df.copy()
    out['joint_cluster_id'] = cluster_ids
    return out
