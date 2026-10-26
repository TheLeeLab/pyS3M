"""TetraSpeck bead precision (Figure 2d): acquisitions, settings and the per-bead analysis.

Each bead is imaged for 200 frames at one laser and power. Its spread about its own mean gives the
precision of one localisation (position) and of one frame's colour (relative pixel signal), as in
the SI note "Calibrating Precision":

* ``sigma_xy`` -- mean of the standard deviations of x and y about the bead's mean position (nm);
* ``prec_X`` -- standard deviation / mean of ``A_X`` (X = B, G, R; the fraction of the signal on
  pixel 1, 2 or 3);
* ``photons`` -- mean of ``A_B + A_G + A_R`` per frame. This is NOT the detected photoelectrons:
  each A_X is what pixel X would collect over the whole sensor, so the sum overcounts the
  detected photoelectrons (``photoelectrons`` = A_B/4 + A_G/2 + A_R/4) 2-4x. The published
  Fig. 2d axis used it; switching the axis to ``photoelectrons`` is pending.
  ``photons_X = photons * mean(A_X)``.

The Bayer mosaic has twice as many pixel-2 (green) pixels as pixel-1 or pixel-3 ones; the
published plot scales the green colour precision by sqrt(2) to put it on a per-pixel footing
with the other two (``GREEN_SCALE``).
"""

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit

BEAD_ROOT = "2026_Multicolour_Paper/Ximea/TetraspeckBeads"
LASERS = ["488", "561", "638"]
POWERS = ["1mW", "2mW", "3mW", "4mW", "6mW", "8mW", "10mW"]
LASER_COLOURS = {"488": "#1f77b4", "561": "#2ca02c", "638": "#d62728"}
CHANNELS = ["B", "G", "R"]   # pixels 1, 2, 3
CHANNEL_COLOURS = {"B": "blue", "G": "darkgreen", "R": "red"}
GREEN_SCALE = np.sqrt(2)
PIXEL_NM = 69.0


def remote_folder(laser, power):
    return f"{BEAD_ROOT}/Tetraspeck_{laser}_excitation/{power}_excitation"


# PSF model wavelength: the TetraSpeck dye each laser excites mainly (Invitrogen T7279 nominal
# emission peaks: 515 nm "green", 580 nm "orange", 680 nm "dark red").
PEAK_UM = {"488": 0.515, "561": 0.58, "638": 0.68}

# As for the single dyes (_single_dyes.FIT_PARAMS); the PSF wavelength is set per laser.
FIT_PARAMS = dict(pfa=1e-3, sigma=1.5, fraction_true=0.05, ROI_size=12, NA=1.49, pixel_size=0.069)

# AIM drift correction as in the paper's Methods: 10-frame segments, 20 nm intersection
# distance, 60 nm search radius (in pixels).
DRIFT_PARAMS = dict(segmentation=10, intersect_d=20 / PIXEL_NM, roi_r=60 / PIXEL_NM)

# Quality cuts as in the Methods (chi2 < 3, localisation error < 1 px, > 50 detected photoelectrons), except:
# * no cut on the colour error: a cut on a frame's own fitted error would select frames on the
#   very quantity whose spread is being measured;
# * PSF sigma 75-250 nm instead of the single-dye 75-160 nm: the beads' fitted sigma is ~145 nm
#   at 561 nm and ~170 nm at 638 nm, so the 160 nm cut drops 40 % and 90 % of their frames.
CLUSTER_PARAMS = dict(clustering_method="HDBSCAN", min_cluster_size=10, chi_val=3,
                      max_localisation_error=1.0, max_colour_error=np.inf,
                      min_sigma=75 / PIXEL_NM, max_sigma=250 / PIXEL_NM,
                      min_photons=0, max_photons=1e7, min_photoelectrons=50)

# A bead must be found in at least this many of the 200 frames. Clusters with fewer are mostly
# out-of-focus or transient spots; HDBSCAN also returns some of those as spread-out clusters.
MIN_FRAMES = 50

# Published-plot cleaning: positions 0.01-20 nm, colour precision 0.5-50 %, < 2e4 photons per
# frame, both for the scatter and for the fits.
LOC_RANGE_NM = (0.01, 20.0)
COLOUR_RANGE_PCT = (0.5, 50.0)
MAX_PHOTONS = 2e4

# Typical photons per localisation (detected photoelectrons), 25th-75th percentiles of the
# literature values collected for the paper's Fig. 2d shading.
FLUORESCENT_PROTEINS = [686, 550, 262, 1000, 1000, 1000, 675, 725, 313]
ORGANIC_DYES = [1341, 1193, 1231, 1493, 6241, 4884, 11022, 4968, 19714, 5202, 5873, 4433, 6337,
                2753, 2073, 2811, 2969, 3879, 23195, 11600, 2429, 1338, 7801, 17090, 23327,
                18448, 8273, 12024, 10399, 12731, 14440, 10348]


def bead_precision(sf_db, min_frames=MIN_FRAMES, pixel_nm=PIXEL_NM):
    """Per-bead position and colour precision from one acquisition's single-frame localisations.

    Beads are grouped by ``(fov_name, molecular_index)``. Besides the published ``sigma_xy``,
    ``sigma_xy_cm`` removes residual stage motion: from each localisation's offset it subtracts
    that frame's mean offset of the *other* beads in the same field (leave-one-out, so a bead's
    own noise is not subtracted from itself), using only reference beads whose own ``sigma_xy``
    is within ``LOC_RANGE_NM``. A frame needs at least 2 other reference beads; a bead needs
    ``min_frames`` such frames, otherwise ``sigma_xy_cm`` is NaN.

    Args:
        sf_db: Single-frame database from ``cluster_sample`` (xc, yc in px; A_B/G/R; photons).
        min_frames: Minimum number of localisations per bead.
        pixel_nm: Pixel size (nm).

    Returns:
        pd.DataFrame: One row per bead: ``fov_name``, ``bead``, ``n``, ``photons``,
        ``sigma_xy``, ``sigma_xy_cm`` (nm), and for X in B, G, R: ``A_X`` (mean),
        ``prec_X`` (std / mean, fraction) and ``photons_X``.
    """
    d = sf_db.copy()
    d["bead"] = d["molecular_index"]
    keys = ["fov_name", "bead"]
    d = d[d.groupby(keys)["xc"].transform("size") >= min_frames]
    if d.empty:
        return pd.DataFrame()

    for c in ("xc", "yc"):
        d["d" + c] = (d[c] - d.groupby(keys)[c].transform("mean")) * pixel_nm
    # leave-one-out common mode within (field, frame), from reference beads only: those whose
    # own spread is in LOC_RANGE_NM (a smeared-out cluster would add its scatter to every bead)
    spread = (d.groupby(keys)["dxc"].transform("std") + d.groupby(keys)["dyc"].transform("std")) / 2
    ref = spread.between(*LOC_RANGE_NM, inclusive="neither")
    grp = d.assign(**{c + "_ref": d[c].where(ref, 0.0) for c in ("dxc", "dyc")},
                   n_ref=ref.astype(int)).groupby(["fov_name", "frame"])
    n_others = grp["n_ref"].transform("sum") - ref
    for c in ("dxc", "dyc"):
        others = (grp[c + "_ref"].transform("sum") - d[c].where(ref, 0.0)) / n_others.where(n_others > 0)
        d[c + "_cm"] = (d[c] - others).where(n_others >= 2)

    g = d.groupby(keys)
    out = pd.DataFrame({
        "n": g.size(),
        "photons": g["photons"].mean(),
        "sigma_xy": (g["dxc"].std() + g["dyc"].std()) / 2,
        "sigma_xy_cm": ((g["dxc_cm"].std() + g["dyc_cm"].std()) / 2)
                       .where(g["dxc_cm"].count() >= min_frames),
    })
    for ch in CHANNELS:
        a = g["A_" + ch]
        out["A_" + ch] = a.mean()
        out["prec_" + ch] = a.std() / a.mean()
        out["photons_" + ch] = out["photons"] * out["A_" + ch]
    return out.reset_index()


def colour_long(beads):
    """One row per bead and pixel: ``photons`` on that pixel, ``prec_pct`` (green x sqrt 2)."""
    rows = []
    for ch in CHANNELS:
        scale = GREEN_SCALE if ch == "G" else 1.0
        rows.append(beads.assign(channel=ch, photons_px=beads["photons_" + ch],
                                 prec_pct=100 * scale * beads["prec_" + ch]))
    return pd.concat(rows, ignore_index=True)


def clean_loc(beads, column="sigma_xy"):
    m = beads[column].between(*LOC_RANGE_NM, inclusive="neither") & (beads["photons"] < MAX_PHOTONS)
    return beads[m]


def clean_colour(long):
    m = (long["prec_pct"].between(*COLOUR_RANGE_PCT, inclusive="neither")
         & (long["photons_px"] < MAX_PHOTONS))
    return long[m]


def power_law(x, a, c):
    return a * x ** -0.5 + c


def fit_power_law(x, y, n_boot=1000, seed=0):
    """Fit ``a / sqrt(N) + c`` as the published plot did (``curve_fit``, ``sigma = 1 / N``).

    Returns ``(a, c)``, and their 95 % bootstrap intervals over beads (``(2, 2)`` array, rows
    a and c) when ``n_boot > 0``.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    p, _ = curve_fit(power_law, x, y, sigma=1 / x, p0=(np.median(y) * np.sqrt(np.median(x)), 0))
    if n_boot <= 0:
        return p, None
    rng = np.random.default_rng(seed)
    boot = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)   # covariance unused; intervals come from resampling
        for _ in range(n_boot):
            i = rng.integers(0, len(x), len(x))
            try:
                boot.append(curve_fit(power_law, x[i], y[i], sigma=1 / x[i], p0=p)[0])
            except RuntimeError:
                continue
    return p, np.percentile(np.array(boot), [2.5, 97.5], axis=0).T


def binned_medians(x, y, edges, min_count=5, n_boot=1000, seed=0):
    """Median of ``y`` in bins of ``x`` with a 95 % bootstrap interval; bins with too few points
    are NaN."""
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        v = y[(x >= lo) & (x < hi)]
        if len(v) < min_count:
            rows.append((np.sqrt(lo * hi), len(v), np.nan, np.nan, np.nan))
            continue
        b = np.median(rng.choice(v, (n_boot, len(v))), axis=1)
        rows.append((np.sqrt(lo * hi), len(v), np.median(v), *np.percentile(b, [2.5, 97.5])))
    return pd.DataFrame(rows, columns=["photons", "n", "median", "lo", "hi"])


def quartiles(values):
    return np.percentile(values, [25, 75])
