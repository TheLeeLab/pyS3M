"""Shared code for FigureSI_Nanorulers (GATTA-PAINT 80R / 80Y nanorulers, Figs. S76 and S77).

Pipeline, per ruler slide:

1. fit     -- ``SuperRes_Functions.fit_imaging_data`` writes ``Localisations.h5`` beside the TIFFs.
              :func:`fit_or_reuse` keeps an existing fit (and only then skips the download).
2. cuts    -- precision, PSF width, colour and photoelectron cuts (:func:`quality_cuts`).
3. undrift -- AIM (no fiducials on these slides), then linking of repeat detections of one
              binding event (:func:`undrift_and_link`, cached with its settings).
4. render  -- Gaussian renders of the linked localisations (no clustering or fiducial removal):
              a whole-field overview to choose the regions, then panels a) and b).

Coordinates: ``xc`` is the image column and ``yc`` the row, in camera pixels
(``Constants.ResultColumns``). Distances are in nm.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import ConnectionPatch, Rectangle

from pyS3M import IOFunctions
from pyS3M import render as _render
from pyS3M.DriftCorrectionFunctions import Drift_Correction_Functions
from pyS3M.LinkingFunctions import link_localisations

LOC_NAME = "Localisations.h5"
FIT_RECORD = "fit_record.json"
AXIS_FIX = datetime(2026, 10, 28, 12, 21)   # fits before this carry the x/y ROI-origin bug

MM = 1 / 25.4


# --------------------------------------------------------------------------- fitting
def fit_or_reuse(folder, record, fit, refit=False):
    """Use ``folder/Localisations.h5`` if it was made with these settings, else call ``fit()``.

    A fit with a matching ``fit_record.json`` is reused. A fit without a record (made before
    this notebook existed) is reused as long as it postdates the x/y fix. ``fit`` is a
    zero-argument callable that downloads the data and writes ``Localisations.h5``.

    Returns:
        Path of ``Localisations.h5``.
    """
    folder = Path(folder)
    loc, rec = folder / LOC_NAME, folder / FIT_RECORD
    record = json.loads(json.dumps(record))   # tuples -> lists, as read back
    if loc.exists() and not refit:
        made = datetime.fromtimestamp(loc.stat().st_mtime)
        previous = json.loads(rec.read_text()) if rec.exists() else None
        if previous == record or (previous is None and made > AXIS_FIX):
            print(f"Reusing {loc} (fitted {made:%Y-%m-%d %H:%M}"
                  + (", no fit record)" if previous is None else ", same settings)"))
            return loc
        if previous is None:
            raise RuntimeError(f"{loc} was fitted {made:%Y-%m-%d}, before the x/y fix: refit "
                               "(REFIT = True) or move it out")
    if loc.exists():
        loc.unlink()   # the fitter appends; never fit on top of an old file
    fit()
    rec.write_text(json.dumps(record, indent=1))
    return loc


def acquisition_info(folder, io=None):
    """(width, height, n_frames) from the MicroManager metadata beside the fit."""
    io = io or IOFunctions.IO_Functions()
    meta = sorted(Path(folder).glob("*metadata.txt"))
    if not meta:
        raise FileNotFoundError(f"no *metadata.txt in {folder}")
    _, _, width, height = io.metadata_reader_imageJ(str(meta[0]))
    return int(width), int(height), int(io.metadata_nframes_reader_imageJ(str(meta[0])))


# --------------------------------------------------------------------------- cuts, drift, linking
def quality_cuts(locs, pixel_nm, max_err_nm=40.0, sigma_nm=(80.0, 175.0),
                 min_colour=None, dim_pe=180.0, dim_mfr=25.0):
    """The cuts of the original nanoruler analysis, in nm.

    Keeps 0 < xc_err, yc_err, s_x_err, s_y_err < ``max_err_nm``; ``sigma_nm[0]`` < s_x, s_y <
    ``sigma_nm[1]``; each colour fraction above ``min_colour[c]``; and drops dim, weak
    detections (photoelectrons < ``dim_pe`` *and* matched-filter response < ``dim_mfr``).
    Returns the kept rows and a table of how many rows each cut removed (in order).
    """
    e = max_err_nm / pixel_nm
    lo, hi = sigma_nm[0] / pixel_nm, sigma_nm[1] / pixel_nm
    cuts = [
        ("precision", (locs["xc_err"] > 0) & (locs["yc_err"] > 0)
         & (locs["xc_err"] < e) & (locs["yc_err"] < e)),
        ("PSF width error", (locs["s_x_err"] < e) & (locs["s_y_err"] < e)),
        ("PSF width", locs["s_x"].between(lo, hi, inclusive="neither")
         & locs["s_y"].between(lo, hi, inclusive="neither")),
    ]
    for c, v in (min_colour or {}).items():
        cuts.append((f"{c} > {v}", locs[c] > v))
    cuts.append(("dim", ~((locs["photoelectrons"] < dim_pe)
                          & (locs["spot_matched_filter_response"] < dim_mfr))))
    keep = pd.Series(True, index=locs.index)
    rows = []
    for name, m in cuts:
        new = keep & m
        rows.append(dict(cut=name, removed=int(keep.sum() - new.sum()), left=int(new.sum())))
        keep = new
    return locs[keep].reset_index(drop=True), pd.DataFrame(rows)


def undrift_and_link(locs, width, height, n_frames, pixel_nm, out_file, settings,
                     redo=False):
    """AIM drift correction then linking within the median precision, cached in ``out_file``.

    ``settings``: dict with segmentation, intersect_nm, roi_nm, max_dark_time (and anything
    else that should invalidate the cache, e.g. the cuts). Returns (linked, drift or None).
    """
    out_file = Path(out_file)
    rec = out_file.with_suffix(".json")
    record = json.loads(json.dumps({"n_locs": int(len(locs)), **settings}))   # tuples -> lists, as read back
    if (not redo and out_file.exists() and rec.exists()
            and json.loads(rec.read_text()) == record):
        print(f"Reusing {out_file}")
        return pd.read_hdf(out_file), None
    info = [{"Width": width, "Height": height, "Frames": n_frames, "Pixelsize": pixel_nm}]
    corrected, drift = Drift_Correction_Functions().undrift(
        locs=locs.to_records(index=False), info=info, method="aim",
        segmentation=settings["segmentation"], intersect_d=settings["intersect_nm"] / pixel_nm,
        roi_r=settings["roi_nm"] / pixel_nm)
    corrected = pd.DataFrame(corrected)
    r = 0.5 * (corrected["xc_err"].median() + corrected["yc_err"].median())
    linked = link_localisations(corrected, n_frames=n_frames, r_max=r,
                                max_dark_time=settings["max_dark_time"])
    out_file.parent.mkdir(parents=True, exist_ok=True)
    IOFunctions.IO_Functions().write_h5_database(df=linked, filepath=str(out_file),
                                                normalise_photons=False, verbose=False)
    rec.write_text(json.dumps(record, indent=1))
    return linked, drift


def plot_drift(drift, pixel_nm, ax=None):
    ax = ax or plt.subplots(figsize=(4, 2.5))[1]
    ax.plot(np.asarray(drift.drift_x) * pixel_nm, label="x")
    ax.plot(np.asarray(drift.drift_y) * pixel_nm, label="y")
    ax.set(xlabel="frame", ylabel="drift / nm")
    ax.legend(frameon=False)
    ax.grid(True, ls="--", lw=0.6, alpha=0.6, color="grey")
    return ax


KEEP_COLUMNS = ["frame", "xc", "yc", "xc_err", "yc_err", "photoelectrons", "A_B", "A_G", "A_R",
                "len", "n"]


def slim(locs):
    """Only the columns the renders (and later analysis) use; the linked tables run to ~10 M rows."""
    return locs[[c for c in KEEP_COLUMNS if c in locs]].copy()


# --------------------------------------------------------------------------- rendering
def render_region(locs, x0, y0, size, oversampling, min_blur_nm=0.0, pixel_nm=69.0):
    """Gaussian render (each event blurred by its own precision) of a square region.

    ``x0``, ``y0`` (column, row) and ``size`` in camera pixels. Returns the image
    (rows = y), ``size * oversampling`` square.
    """
    sub = locs[(locs["xc"] >= x0 - 1) & (locs["xc"] < x0 + size + 1)
               & (locs["yc"] >= y0 - 1) & (locs["yc"] < y0 + size + 1)]
    viewport = ((y0, x0), (y0 + size, x0 + size))
    return _render.render(locs=sub.to_records(index=False), oversampling=oversampling,
                          viewport=viewport, blur_method="gaussian",
                          min_blur_width=min_blur_nm / pixel_nm)[1]


def colour_limits(img, pct=(0.0, 99.9)):
    """(vmin, vmax) of an image's colour scale from (low, high) percentiles of its pixels."""
    lo, hi = pct
    return float(np.percentile(img, lo)), float(np.percentile(img, hi))


def plot_overview(locs, width, height, oversampling=2, grid_px=50, clim=(0.0, 99.5)):
    """Whole field of view with a camera-pixel grid, to choose ``VIEWS``.

    ``clim``: (low, high) percentiles of the image setting the colour scale.
    Returns (fig, ax, (vmin, vmax) used)."""
    img = render_region(locs, 0, 0, max(width, height), oversampling)
    lims = colour_limits(img, clim)
    fig, ax = plt.subplots(figsize=(10, 10))
    ext = (0, max(width, height), 0, max(width, height))
    ax.imshow(img, origin="lower", cmap="hot", vmin=lims[0], vmax=lims[1], extent=ext,
              interpolation="none")
    ticks = np.arange(0, max(width, height) + 1, grid_px)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(labelsize=7)
    ax.grid(color="white", lw=0.3, alpha=0.4)
    ax.set(xlabel="x (column) / camera px", ylabel="y (row) / camera px", xlim=(0, width),
           ylim=(0, height))
    return fig, ax, lims


def plot_views(fov_img, zoom_img, fov, zoom, oversampling_fov, oversampling_zoom, pixel_nm,
               fov_bar_nm=1000, zoom_bar_nm=80, clim_fov=(0.0, 99.9), clim_zoom=(0.0, 99.9),
               grid_px=None):
    """Panels a) field of view with the zoom box and b) the zoom-in, connected as in S76/S77.

    ``fov``/``zoom``: dicts x0, y0, size (camera px). ``clim_fov``/``clim_zoom``: (low, high)
    percentiles of each panel's own image setting its colour scale.
    ``grid_px``: draw a camera-pixel grid on panel a (for choosing the zoom; leave None for the
    figure). Returns (fig, (ax_a, ax_b), {"fov": (vmin, vmax), "zoom": (vmin, vmax)} used).
    """
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(167.8 * MM, 80 * MM),
                                     gridspec_kw=dict(wspace=0.08))
    used = {}
    for ax, img, os_, bar, clim, key in (
            (ax_a, fov_img, oversampling_fov, fov_bar_nm, clim_fov, "fov"),
            (ax_b, zoom_img, oversampling_zoom, zoom_bar_nm, clim_zoom, "zoom")):
        used[key] = colour_limits(img, clim)
        ax.imshow(img, origin="lower", cmap="hot", vmin=used[key][0], vmax=used[key][1],
                  interpolation="none")
        ax.set_axis_off()
        _scalebar(ax, bar / pixel_nm * os_, f"{bar / 1000:g} µm" if bar >= 1000 else f"{bar:g} nm",
                  img.shape[0])
    if grid_px:
        ax_a.set_axis_on()
        t = np.arange(0, fov["size"] + 1, grid_px)
        ax_a.set_xticks(t * oversampling_fov, [f"{fov['x0'] + v:g}" for v in t])
        ax_a.set_yticks(t * oversampling_fov, [f"{fov['y0'] + v:g}" for v in t])
        ax_a.tick_params(labelsize=6)
        ax_a.grid(color="white", lw=0.3, alpha=0.5)
    bx = (zoom["x0"] - fov["x0"]) * oversampling_fov
    by = (zoom["y0"] - fov["y0"]) * oversampling_fov
    bs = zoom["size"] * oversampling_fov
    ax_a.add_patch(Rectangle((bx, by), bs, bs, fill=False, ec="white", lw=1))
    nb = zoom_img.shape[0]
    for (xa, ya), (xb, yb) in (((bx + bs, by + bs), (0, nb)), ((bx + bs, by), (0, 0))):
        fig.add_artist(ConnectionPatch((xa, ya), (xb, yb), "data", "data", axesA=ax_a,
                                       axesB=ax_b, color="white", lw=1))
    for ax, letter in ((ax_a, "a)"), (ax_b, "b)")):
        ax.text(0.015, 0.985, letter, transform=ax.transAxes, color="white", fontsize=9,
                fontweight="bold", va="top", ha="left")
    return fig, (ax_a, ax_b), used


def _scalebar(ax, length_px, label, img_px, colour="white", fs=7):
    pad = 0.05 * img_px
    y = pad
    ax.plot([img_px - pad - length_px, img_px - pad], [y, y], color=colour, lw=2.5,
            solid_capstyle="butt")
    ax.text(img_px - pad - length_px / 2, y + 0.025 * img_px, label, color=colour,
            ha="center", va="bottom", fontsize=fs)
