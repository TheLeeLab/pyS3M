# -*- coding: utf-8 -*-
"""
FRCFunctions
~~~~~~~~~~~~

Fourier Ring Correlation (FRC) resolution estimation for SMLM data.

Implements the FIRE (Fourier Image REsolution) method from:
  Nieuwenhuizen et al., Nature Methods 10, 557-562 (2013).

Ported from the MATLAB distribution supplied with that paper.
All function names and algorithms match the MATLAB originals.

Public API
----------
bin_localisations(positions, nx, ny, zoom)   — bin x,y coords → SR image
frc(im1, im2)                                — FRC curve from two images
frc_to_resolution(frc_curve, sz)             — 1/7-threshold resolution
fire(positions, nx, ..., rng=None)           — single-image FIRE value

:source: Nieuwenhuizen et al. MATLAB distribution
:ported: jsb92, 2026-03-11
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from typing import Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _radial_sum(image: np.ndarray) -> np.ndarray:
    """Sum pixel values in each radial frequency ring.

    Replaces DIPimage ``radialsum()``.  Ring index *k* is assigned by
    rounding the Euclidean distance from the image centre.  The image
    must have DC at the centre (i.e. after ``fftshift``).

    Parameters
    ----------
    image : 2-D real array.

    Returns
    -------
    result : 1-D float64 array, length = round(r_max) + 1.
    """
    ny, nx = image.shape
    cy, cx = ny // 2, nx // 2
    y, x = np.indices(image.shape)
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_int = np.round(r).astype(int)
    n_bins = int(r_int.max()) + 1
    result = np.bincount(r_int.ravel(), weights=np.asarray(image).ravel(),
                         minlength=n_bins)
    return result


def _tukey_window(ny: int, nx: int) -> np.ndarray:
    """2-D separable Tukey (cosine-tapered) edge mask.

    Matches ``frc.m``: nfac=8, so the outer 1/8 of each axis is tapered
    from 1→0 with a raised cosine; the inner 6/8 is flat at 1.
    For square images this is applied identically to both axes and the
    2-D mask is the outer product.
    """
    nfac = 8

    def _mask_1d(n: int) -> np.ndarray:
        c = np.arange(n) - n // 2          # centred pixel indices
        x = c / n                           # normalised ≈ (-0.5, 0.5)
        m = 0.5 - 0.5 * np.cos(np.pi * nfac * x)
        # inner (nfac-2)/nfac fraction = 1
        m[np.abs(x) < (nfac - 2) / (nfac * 2)] = 1.0
        return m

    mx = _mask_1d(nx)
    my = _mask_1d(ny)
    return (my[:, np.newaxis] * mx[np.newaxis, :]).astype(np.float32)


def _intersect(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Find x-values where curve a(x) crosses curve b(x).

    Uses linear interpolation between grid points.
    Matches MATLAB ``isect.m``.
    """
    diff = a - b
    valid = np.isfinite(diff)
    xv, dv = x[valid], diff[valid]

    sign_changes = np.where(np.diff(np.sign(dv)) != 0)[0]
    x_cross = []
    for i in sign_changes:
        d0, d1 = dv[i], dv[i + 1]
        if d1 != d0:
            xi = xv[i] - d0 * (xv[i + 1] - xv[i]) / (d1 - d0)
            x_cross.append(float(xi))

    # Exact grid-point intersections
    x_exact = xv[dv == 0].tolist()

    out = np.array(x_cross + x_exact)
    out = out[np.isfinite(out)]
    out = np.unique(out)          # remove duplicates (sign-change + exact-zero)
    return out


def _first_decreasing_crossing(q: np.ndarray,
                                frc_smooth: np.ndarray,
                                isects: np.ndarray,
                                sz: int) -> float:
    """Return the first intersection where the smoothed FRC is decreasing."""
    for qi in isects:
        idx = min(1 + int(np.floor(sz * qi)), len(frc_smooth) - 2)
        if frc_smooth[idx + 1] < frc_smooth[idx]:
            return float(qi)
    return np.nan


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def bin_localisations(
    positions: np.ndarray,
    nx: int,
    ny: int,
    zoom: float = 1.0,
) -> np.ndarray:
    """Bin 2-D localisation coordinates into a super-resolution image.

    Replicates MATLAB ``binlocalizations()``.  Positions are shifted by
    +0.5 and scaled by *zoom* before histogramming.

    Parameters
    ----------
    positions : (N, 2+) array, columns [x, y, ...] in camera pixels.
    nx, ny    : output image size in super-resolution pixels.
    zoom      : SR magnification (SR pixels per camera pixel).

    Returns
    -------
    image : (ny, nx) float32 array — count per SR pixel.
    """
    pos = np.asarray(positions[:, :2], dtype=np.float32)
    pos = (pos + 0.5) * float(zoom)

    keep = ((pos[:, 0] >= 0) & (pos[:, 0] <= nx) &
            (pos[:, 1] >= 0) & (pos[:, 1] <= ny))
    pos = pos[keep]

    image, _, _ = np.histogram2d(
        pos[:, 1], pos[:, 0],           # rows=y, cols=x  →  (ny, nx)
        bins=[ny, nx],
        range=[[0, ny], [0, nx]],
    )
    return image.astype(np.float32)


def frc(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """Compute the Fourier Ring Correlation curve from two 2-D images.

    Replicates MATLAB ``frc()``.  A 2-D Tukey window is applied before
    the FFT to suppress edge artefacts.  Non-square images are zero-padded
    to a square.

    Parameters
    ----------
    im1, im2 : 2-D arrays of the same shape.

    Returns
    -------
    frc_curve : 1-D float64 array — correlation per radial ring.
                Ring *k* ↔ spatial frequency *k / N* where *N* is the
                (padded) image size.
    """
    if im1.shape != im2.shape:
        raise ValueError(f"Image shapes differ: {im1.shape} vs {im2.shape}")

    ny, nx = im1.shape

    # Apply Tukey window
    mask = _tukey_window(ny, nx)
    a1 = im1 * mask
    a2 = im2 * mask

    # Zero-pad to square
    if ny != nx:
        sz = max(ny, nx)
        b1 = np.zeros((sz, sz), dtype=np.float32)
        b2 = np.zeros((sz, sz), dtype=np.float32)
        b1[:ny, :nx] = a1
        b2[:ny, :nx] = a2
        a1, a2 = b1, b2

    # FFT — DC at centre, matching DIPimage ft()
    F1 = np.fft.fftshift(np.fft.fft2(a1))
    F2 = np.fft.fftshift(np.fft.fft2(a2))

    num  = _radial_sum(np.real(F1 * np.conj(F2)))
    d1   = _radial_sum(np.abs(F1) ** 2)
    d2   = _radial_sum(np.abs(F2) ** 2)

    denom = np.sqrt(np.abs(d1 * d2))
    with np.errstate(invalid='ignore', divide='ignore'):
        curve = np.where(denom > 0, num / denom, 0.0)

    return curve


def frc_to_resolution(
    frc_curve: np.ndarray,
    sz: int,
) -> Tuple[float, float, float]:
    """Extract resolution from an FRC curve using the 1/7 threshold.

    Replicates MATLAB ``frctoresolution()``.  The curve is smoothed with
    a Savitzky-Golay filter (equivalent to MATLAB's loess with the same
    span) before finding the first decreasing crossing of 1/7.

    Parameters
    ----------
    frc_curve : 1-D array as returned by :func:`frc`.
    sz        : side length (pixels) of the *zoomed* SR images used to
                compute the curve.  Used to set the frequency axis.

    Returns
    -------
    resolution      : SR pixels at the 1/7 crossing.  NaN if not found.
    resolution_high : upper bound (better / smaller value).
    resolution_low  : lower bound (worse / larger value).
    """
    frc_in = np.array(frc_curve, dtype=np.float64).ravel()
    frc_in = np.clip(frc_in, -1.0, 1.0)

    # Pixels per ring — must match frc_curve length
    nr = _radial_sum(np.ones((sz, sz)))
    if len(nr) != len(frc_in):
        raise ValueError(
            f"FRC length ({len(frc_in)}) does not match sz={sz} "
            f"(need {len(nr)} rings)."
        )

    # Effective number of correlated pixels (from the noise floor)
    frc_tmp = np.where(np.isfinite(frc_in), frc_in, 0.0)
    neg_idx = np.where(frc_in < 0)[0]
    if len(neg_idx) > 0:
        n1 = int(neg_idx[0])
        ne_inv = float(np.mean(frc_tmp[n1:] ** 2 * nr[n1:]))
    else:
        ne_inv = 1.0

    # Smooth (Savitzky-Golay ≈ MATLAB loess with span ceil(sz/20))
    sspan = max(5, int(np.ceil(sz / 20)))
    if sspan % 2 == 0:
        sspan += 1

    try:
        frc_smooth = savgol_filter(frc_in, sspan, polyorder=2)
    except Exception:
        frc_smooth = gaussian_filter1d(frc_in, sigma=0.9)

    q = np.arange(len(frc_in), dtype=np.float64) / sz
    threshold = np.full_like(q, 1.0 / 7.0)

    isects = _intersect(q, frc_smooth, threshold)
    isects = isects[isects < 0.5]          # below Nyquist only

    if len(isects) == 0:
        return np.nan, 0.0, 2.0 * sz      # unresolved

    q_cross = _first_decreasing_crossing(q, frc_smooth, isects, sz)
    if not np.isfinite(q_cross):
        return np.nan, 0.0, 2.0 * sz

    resolution = 1.0 / q_cross

    # --- Uncertainty via quadratic fit in smoothing window ---
    isect_ind = min(1 + int(np.floor(sz * q_cross)), len(frc_in) - 2)

    frc_var = ((1 + 2 * frc_smooth - frc_smooth ** 2) *
               (1 - frc_smooth) ** 2 * ne_inv /
               np.maximum(nr, 1))

    lo = max(0, int(np.ceil(isect_ind - sspan / 2)))
    hi = min(len(frc_in), int(np.floor(isect_ind + sspan / 2)) + 1)
    idx = np.arange(lo, hi)

    try:
        S = np.diag(frc_var[idx])
        X = np.column_stack([np.ones(len(idx)), q[idx], q[idx] ** 2])
        XtX_inv = np.linalg.inv(X.T @ X)
        C = XtX_inv @ X.T @ S @ X @ XtX_inv
        v = np.array([1.0, resolution, resolution ** 2])
        frc_newvar = float(ne_inv * (1.0 / v) @ C @ (1.0 / v))
        if not np.isfinite(frc_newvar) or frc_newvar < 0:
            frc_newvar = float(frc_var[isect_ind])
    except np.linalg.LinAlgError:
        frc_newvar = float(frc_var[isect_ind])

    sigma_frc = float(np.sqrt(max(frc_newvar, 0.0)))

    frc_high = frc_smooth + sigma_frc
    frc_low  = frc_smooth - sigma_frc

    isects_h = _intersect(q, frc_high, threshold)
    isects_h = isects_h[isects_h < 0.5]
    q_h = _first_decreasing_crossing(q, frc_high, isects_h, sz)
    resolution_high = 1.0 / q_h if np.isfinite(q_h) else 0.0

    isects_l = _intersect(q, frc_low, threshold)
    isects_l = isects_l[isects_l < 0.5]
    q_l = _first_decreasing_crossing(q, frc_low, isects_l, sz)
    resolution_low = 1.0 / q_l if np.isfinite(q_l) else 2.0 * sz

    return resolution, resolution_high, resolution_low


def _postofrc(
    positions: np.ndarray,
    sz: int,
    zoom: float,
    n_blocks: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Single temporal-split FRC curve.  Internal use only."""
    n_blocks = max(2, n_blocks)

    if positions.shape[1] >= 3:
        frames = positions[:, 2]
        max_t = frames.max()
        if max_t > 0:
            blocks = np.ceil(frames / max_t * n_blocks).astype(int)
        else:
            blocks = np.ones(len(positions), dtype=int)
        blocks = np.clip(blocks, 1, n_blocks)
    else:
        n_blocks = min(n_blocks, len(positions))
        N = len(positions)
        blocks = (np.arange(N) * n_blocks // N) + 1

    perm = rng.permutation(n_blocks) + 1             # shuffled 1..n_blocks
    half = perm[:int(np.ceil(n_blocks / 2))]
    mask = np.isin(blocks, half)

    im1 = bin_localisations(positions[mask],  sz, sz, zoom)
    im2 = bin_localisations(positions[~mask], sz, sz, zoom)
    return frc(im1, im2)


def fire(
    positions: Union[np.ndarray, pd.DataFrame],
    nx: int,
    ny: Optional[int] = None,
    zoom: float = 1.0,
    n_blocks: int = 50,
    reps: int = 20,
    pixel_size_nm: float = 1.0,
    rng: Optional[Union[int, np.random.Generator]] = None,
) -> Tuple[float, np.ndarray, float, float]:
    """Compute FIRE from a single localisation dataset.

    Replicates MATLAB ``postoresolution()``.  The dataset is split into
    *n_blocks* temporal blocks, randomly assigned to two halves, binned
    into SR images, and FRC is computed.  This is repeated *reps* times
    and results are averaged.

    Parameters
    ----------
    positions    : (N, 2) or (N, 3) array [x, y] or [x, y, frame] in
                   camera pixels.  Also accepts a DataFrame with columns
                   ``xc``, ``yc`` and optionally ``frame``.
    nx           : field width in camera pixels.
    ny           : field height in camera pixels (default = nx).
    zoom         : SR magnification (SR pixels per camera pixel).
    n_blocks     : number of temporal blocks for the half-split (default 50).
    reps         : independent repeats to average over (default 20).
    pixel_size_nm: camera pixel size in nm — used to convert output to nm.
    rng          : seed or ``np.random.Generator`` for the temporal
                   half-split.  Default (None) draws fresh entropy each
                   call, matching prior behaviour; pass an int or
                   Generator for reproducible splits.

    Returns
    -------
    resolution_nm    : FIRE value in nm.
    frc_mean         : averaged FRC curve (1-D array, length ≈ sz*√2/2).
    resolution_hi_nm : upper uncertainty bound in nm (better / smaller).
    resolution_lo_nm : lower uncertainty bound in nm (worse / larger).
    """
    if isinstance(positions, pd.DataFrame):
        cols = ['xc', 'yc']
        if 'frame' in positions.columns:
            cols.append('frame')
        positions = positions[cols].to_numpy(dtype=np.float64)

    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] < 2:
        raise ValueError("positions must be (N, 2) or (N, 3).")

    if ny is None:
        ny = nx

    sz   = max(int(nx * zoom), int(ny * zoom))
    n_blocks = max(2, min(n_blocks, len(positions)))
    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

    frc_accum: Optional[np.ndarray] = None
    res_list: list = []

    for _ in range(reps):
        curve = _postofrc(positions, sz, zoom, n_blocks, rng)
        res, _, _ = frc_to_resolution(curve, sz)
        if np.isfinite(res):
            res_list.append(res)
        if frc_accum is None:
            frc_accum = curve.copy()
        else:
            n = min(len(frc_accum), len(curve))
            frc_accum[:n] += curve[:n]

    frc_mean = frc_accum / reps if frc_accum is not None else np.array([])

    if not res_list:
        return np.nan, frc_mean, np.nan, np.nan

    resolution = float(np.mean(res_list))
    std_res    = float(np.std(res_list))

    sr_px_nm = pixel_size_nm / zoom
    return (
        resolution           * sr_px_nm,
        frc_mean,
        (resolution - std_res) * sr_px_nm,   # hi = tighter
        (resolution + std_res) * sr_px_nm,   # lo = looser
    )
