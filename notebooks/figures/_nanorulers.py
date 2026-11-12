"""Shared code for FigureSI_Nanorulers (GATTA-PAINT 80R / 80Y nanorulers, Figs. S76 and S77).

Pipeline, per ruler slide:

1. fit     -- ``SuperRes_Functions.fit_imaging_data`` writes ``Localisations.h5`` beside the TIFFs.
              :func:`fit_or_reuse` keeps an existing fit (and only then skips the download).
2. cuts    -- precision, PSF width, colour and photoelectron cuts (:func:`quality_cuts`).
3. undrift -- AIM (no fiducials on these slides), then linking of repeat detections of one
              binding event (:func:`undrift_and_link`, cached with its settings).
4. render  -- Gaussian renders of the linked localisations (no clustering or fiducial removal):
              a whole-field overview to choose the regions, then panels a) and b).
5. rulers  -- docking sites by DBSCAN of the linked events (:func:`find_sites`; each site's
              centre is the mean of its events, the maximum-likelihood centre of a Gaussian
              blob), sites grouped into rulers, and only rulers of exactly three sites in a line
              kept (:func:`find_rulers`). Centre-to-centre distances between neighbouring sites
              (design 80 nm) and the two end sites (160 nm) give panels c)-f)
              (:func:`distance_stats`, :func:`plot_distances`).

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
from sklearn.cluster import DBSCAN

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


# --------------------------------------------------------------------------- ruler distances
def find_sites(locs, pixel_nm, n_frames, eps_nm=10.0, min_events=50, cell_nm=5.0):
    """Docking sites: DBSCAN of the linked events, one row per site.

    Events are binned on a ``cell_nm`` grid and DBSCAN runs on the occupied cells weighted by
    their counts (``sample_weight``), so ``min_events`` is a number of events within ``eps_nm``;
    this keeps memory bounded for millions of events. Per site: number of events ``n``, centre
    ``x``, ``y`` (nm; mean of its events), centre uncertainty ``ex``, ``ey`` (SD / sqrt(n)),
    major/minor SD of the event cloud ``smaj``, ``smin`` and the fraction of frames it is on.
    """
    xy = locs[["xc", "yc"]].to_numpy(float) * pixel_nm
    cells, inv, cnt = np.unique(np.floor(xy / cell_nm).astype(np.int64), axis=0,
                                return_inverse=True, return_counts=True)
    lab = DBSCAN(eps=eps_nm, min_samples=min_events).fit_predict(
        (cells + 0.5) * cell_nm, sample_weight=cnt)[inv.ravel()]
    keep = lab >= 0
    lab, p = lab[keep], xy[keep]
    k = lab.max() + 1
    n = np.bincount(lab, minlength=k).astype(float)
    mx, my = np.bincount(lab, p[:, 0], k) / n, np.bincount(lab, p[:, 1], k) / n
    dx, dy = p[:, 0] - mx[lab], p[:, 1] - my[lab]
    sxx, syy = np.bincount(lab, dx * dx, k) / n, np.bincount(lab, dy * dy, k) / n
    sxy = np.bincount(lab, dx * dy, k) / n
    half_tr = (sxx + syy) / 2
    root = np.sqrt(np.maximum(half_tr ** 2 - (sxx * syy - sxy ** 2), 0))
    on = np.bincount(lab, locs["len"].to_numpy()[keep], k) / n_frames
    return pd.DataFrame(dict(n=n, x=mx, y=my, ex=np.sqrt(sxx / n), ey=np.sqrt(syy / n),
                             smaj=np.sqrt(half_tr + root),
                             smin=np.sqrt(np.maximum(half_tr - root, 0)), on=on))


def find_rulers(sites, min_events=30, max_on=0.2, max_sd_nm=40.0, link_nm=120.0):
    """Group sites into rulers; one row per ruler of exactly three sites.

    Sites with fewer than ``min_events``, on in more than ``max_on`` of frames (several
    overlapping structures, or sticking) or broader than ``max_sd_nm`` are dropped. The rest are
    grouped by single linkage at ``link_nm`` (between the 80 nm spacing and the typical distance
    between rulers). In each three-site group the sites are ordered along the group's principal
    axis; ``angle`` is the angle at the middle site (180 deg = a straight line) and ``perp`` the
    distance of the middle site from the line through the end sites. ``d1``, ``d2`` (neighbours)
    and ``e2e`` (end to end) are centre-to-centre distances (nm), ``e_*`` their uncertainties
    from the site-centre uncertainties. Returns (rulers, summary dict).
    """
    good = sites[(sites.n >= min_events) & (sites.on <= max_on) & (sites.smaj <= max_sd_nm)]
    groups = DBSCAN(eps=link_nm, min_samples=1).fit_predict(good[["x", "y"]].to_numpy())
    sizes = np.bincount(groups)
    rows = []
    for g in np.flatnonzero(sizes == 3):
        sel = groups == g
        pts, err = good[["x", "y"]].to_numpy()[sel], good[["ex", "ey"]].to_numpy()[sel]
        c = pts - pts.mean(0)
        order = np.argsort(c @ np.linalg.svd(c)[2][0])
        pts, err = pts[order], err[order]
        u, w = pts[0] - pts[1], pts[2] - pts[1]
        angle = np.degrees(np.arccos(np.clip(u @ w / np.linalg.norm(u) / np.linalg.norm(w),
                                             -1, 1)))
        end = pts[2] - pts[0]
        u_end = end / np.linalg.norm(end)
        perp = abs(u_end[0] * (pts[1] - pts[0])[1] - u_end[1] * (pts[1] - pts[0])[0])

        def dist(a, b):
            d = pts[b] - pts[a]
            r = np.linalg.norm(d)
            return r, np.sqrt((((d / r) * err[a]) ** 2).sum() + (((d / r) * err[b]) ** 2).sum())

        (d1, e1), (d2, e2), (d3, e3) = dist(0, 1), dist(1, 2), dist(0, 2)
        rows.append(dict(x=pts[1, 0], y=pts[1, 1], angle=angle, perp=perp, d1=d1, d2=d2, e2e=d3,
                         e_d1=e1, e_d2=e2, e_e2e=e3,
                         n_min=good.n.to_numpy()[sel].min()))
    summary = dict(sites=len(sites), sites_kept=len(good),
                   group_sizes={int(k): int(v) for k, v in
                                zip(*np.unique(sizes, return_counts=True))},
                   site_centre_err_nm=float(np.median(np.r_[good.ex, good.ey])),
                   site_sd_nm=float(np.median(np.r_[good.smaj, good.smin])))
    return pd.DataFrame(rows), summary


def distance_stats(rulers, design_nm=80.0, tolerance_nm=5.0, n_boot=2000, seed=0):
    """Accuracy and spread of the neighbour (design ``design_nm``) and end-to-end (twice that)
    distances. The 95% CI of the median bias is a bootstrap over rulers (a ruler's two
    neighbour distances are resampled together). Returns a DataFrame, one column per distance.
    """
    rng = np.random.default_rng(seed)
    boot = rng.integers(0, len(rulers), (n_boot, len(rulers)))
    out = {}
    for key, cols, design in (("neighbour", ["d1", "d2"], design_nm),
                              ("end-to-end", ["e2e"], 2 * design_nm)):
        per_ruler = rulers[cols].to_numpy()
        v = per_ruler.ravel()
        err = v - design
        med = np.median(v)
        ci = np.percentile([np.median(per_ruler[b].ravel()) for b in boot], [2.5, 97.5]) - design
        expected = np.median(rulers[["e_" + c for c in cols]].to_numpy())
        out[key] = {"design / nm": design, "n distances": len(v), "median / nm": med,
                    "bias / nm": med - design, "bias 95% CI low / nm": ci[0],
                    "bias 95% CI high / nm": ci[1], "relative bias / %": 100 * (med / design - 1),
                    "mean / nm": v.mean(), "SD / nm": v.std(ddof=1),
                    "MAD-based SD / nm": 1.4826 * np.median(abs(v - med)),
                    "RMSE / nm": np.sqrt(np.mean(err ** 2)),
                    f"within ±{tolerance_nm:g} nm / %": 100 * np.mean(abs(err) <= tolerance_nm),
                    "SD expected from site centres / nm": expected}
    return pd.DataFrame(out)


def plot_distances(rulers, stats, design_nm=80.0, tolerance_nm=5.0, half_range_nm=25.0,
                   bin_nm=1.0, info=None):
    """Panels c) neighbour and d) end-to-end distance histograms (design value, ± tolerance band
    and median), e) cumulative fraction of distances within a given error of design, and f) the
    statistics table. ``info``: (label, value) lines printed under the table."""
    fig, axs = plt.subplots(2, 2, figsize=(167.8 * MM, 125 * MM),
                            layout="none",
                            gridspec_kw=dict(left=0.09, right=0.98, bottom=0.09, top=0.94,
                                             wspace=0.3, hspace=0.42))
    (ax_c, ax_d), (ax_e, ax_f) = axs
    within = f"within ±{tolerance_nm:g} nm / %"
    colours = {"neighbour": "tab:blue", "end-to-end": "tab:orange"}
    for ax, key, cols in ((ax_c, "neighbour", ["d1", "d2"]), (ax_d, "end-to-end", ["e2e"])):
        st = stats[key]
        design = st["design / nm"]
        v = rulers[cols].to_numpy().ravel()
        ax.hist(v, bins=np.arange(design - half_range_nm, design + half_range_nm + bin_nm, bin_nm),
                color="grey")
        ax.axvspan(design - tolerance_nm, design + tolerance_nm, color="tab:green", alpha=0.15,
                   lw=0, label=f"design ±{tolerance_nm:g} nm")
        ax.axvline(design, color="k", ls="--", lw=1.2, label=f"design {design:g} nm")
        ax.axvline(st["median / nm"], color="red", lw=1.2,
                   label=f"median {st['median / nm']:.1f} nm")
        ax.set(xlim=(design - half_range_nm, design + half_range_nm), ylabel="distances",
               xlabel=f"{key} distance / nm")
        ax.legend(fontsize=6, loc="upper left", title=f"{st[within]:.0f}% within ±{tolerance_nm:g} nm",
                  title_fontsize=6)
    for key, cols in (("neighbour", ["d1", "d2"]), ("end-to-end", ["e2e"])):
        e = np.sort(abs(rulers[cols].to_numpy().ravel() - stats[key]["design / nm"]))
        ax_e.plot(e, np.arange(1, len(e) + 1) / len(e), color=colours[key], lw=1.5,
                  label=f"{key} ({stats[key][within]:.0f}%)")
    ax_e.axvline(tolerance_nm, color="k", ls="--", lw=1.2)
    ax_e.set(xlim=(0, half_range_nm), ylim=(0, 1.01), xlabel="|distance − design| / nm",
             ylabel="cumulative fraction")
    ax_e.legend(fontsize=6, loc="lower right", title=f"within ±{tolerance_nm:g} nm",
                title_fontsize=6)
    for ax in (ax_c, ax_d, ax_e):
        ax.tick_params(labelsize=7)
        ax.xaxis.label.set_size(8)
        ax.yaxis.label.set_size(8)
        ax.grid(True, ls="--", lw=0.6, alpha=0.6, color="grey")
        ax.set_axisbelow(True)

    def row(label, key, fmt):
        return [label] + [fmt(stats[c][key]) for c in stats.columns]
    ci = {c: f"{stats[c]['bias / nm']:+.1f} [{stats[c]['bias 95% CI low / nm']:+.1f}, "
             f"{stats[c]['bias 95% CI high / nm']:+.1f}]" for c in stats.columns}
    cells = [row("distances", "n distances", lambda x: f"{x:,.0f}"),
             row("median / nm", "median / nm", lambda x: f"{x:.1f}"),
             ["bias / nm [95% CI]"] + [ci[c] for c in stats.columns],
             row("relative bias / %", "relative bias / %", lambda x: f"{x:+.1f}"),
             row("SD / nm", "SD / nm", lambda x: f"{x:.1f}"),
             row("MAD-based SD / nm", "MAD-based SD / nm", lambda x: f"{x:.1f}"),
             row("RMSE / nm", "RMSE / nm", lambda x: f"{x:.1f}"),
             row(f"within ±{tolerance_nm:g} nm / %", within, lambda x: f"{x:.0f}"),
             row("SD from site centres / nm", "SD expected from site centres / nm",
                 lambda x: f"{x:.2f}")]
    ax_f.set_axis_off()
    table = ax_f.table(cellText=cells,
                       colLabels=[""] + [f"{c} ({stats[c]['design / nm']:g} nm)"
                                         for c in stats.columns],
                       cellLoc="center", colWidths=[0.44, 0.28, 0.28],
                       bbox=[0.0, 0.2 if info else 0.0, 1.0, 0.8 if info else 1.0])
    table.auto_set_font_size(False)
    table.set_fontsize(6)
    for (r, c), cell in table.get_celld().items():
        cell.set_linewidth(0.4)
        if c == 0:
            cell.set_text_props(ha="left")
            cell.PAD = 0.03
    if info:
        ax_f.text(0.0, 0.15, "\n".join(f"{label}: {value}" for label, value in info),
                  transform=ax_f.transAxes, fontsize=6, va="top", ha="left")
    for ax, letter in ((ax_c, "c)"), (ax_d, "d)"), (ax_e, "e)"), (ax_f, "f)")):
        ax.text(-0.16 if ax is not ax_f else -0.08, 1.04, letter, transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="bottom", ha="left")
    return fig, axs
