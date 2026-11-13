"""Shared code for FigureSI_Nanoruler_BayerPhase: no localisation, distance or colour bias with the place
of a molecule on the Bayer pattern, from the drift of the GATTA-PAINT 80Y / 80R nanorulers.

Each acquisition drifts by several camera pixels over its 60,000 frames, so every ruler is imaged at many
places on the 2 x 2 Bayer cell. The translation is AIM's drift estimate from the whole field, not from the
structure measured, so it cannot absorb a per-site bias. Per event: raw camera position = drift-corrected
position + drift at its frame; place in the cell = (raw position + 0.5) mod 2 px (pixel centres at
integers, so 0-1 px is the even pixel and 1-2 px the odd one, in ROI coordinates).

Coordinates: ``xc`` is the image column and ``yc`` the row, in camera pixels. Distances are in nm.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

from pyS3M.DriftCorrectionFunctions import Drift_Correction_Functions

TINT = {"B": "#5b7bd6", "G": "#58b05a", "R": "#d9534f"}   # filter tints of the mosaic drawings
CELL_BINS = np.linspace(0, 2, 5)                           # 0.5 px bins over the 2 x 2 px cell


# --------------------------------------------------------------------------- colours
def iwanthue(n, hue=(0, 360), chroma=(30, 80), lightness=(35, 70), n_samples=40000, seed=0):
    """iwanthue's k-means mode (medialab.github.io/iwanthue; no Python package): sample the CIELab colours
    inside the hue / chroma / lightness limits that sRGB can show, cluster them into ``n`` groups, and
    return the members nearest the group centres, sorted by hue."""
    from skimage.color import lab2rgb, rgb2lab
    from sklearn.cluster import KMeans
    rng = np.random.default_rng(seed)
    lab = np.column_stack([rng.uniform(*lightness, n_samples), rng.uniform(-128, 128, (n_samples, 2))])
    h = np.degrees(np.arctan2(lab[:, 2], lab[:, 1])) % 360
    c = np.hypot(lab[:, 1], lab[:, 2])
    lab = lab[(c >= chroma[0]) & (c <= chroma[1]) & (h >= hue[0]) & (h <= hue[1])]
    rgb = lab2rgb(lab[None])[0]
    lab = lab[np.abs(rgb2lab(rgb[None])[0] - lab).max(axis=1) < 0.5]   # in the sRGB gamut
    centres = KMeans(n, n_init=10, random_state=seed).fit(lab).cluster_centers_
    pick = lab[np.argmin(((lab[:, None] - centres[None]) ** 2).sum(-1), axis=0)]
    pick = pick[np.argsort(np.degrees(np.arctan2(pick[:, 2], pick[:, 1])) % 360)]
    return [tuple(np.clip(c, 0, 1)) for c in lab2rgb(pick[None])[0]]


# --------------------------------------------------------------------------- drift, rulers, events
def drift_trace(kept, width, height, n_frames, pixel_nm, segmentation, intersect_nm, roi_nm):
    """AIM drift (px per frame) of the quality-cut localisations, as FigureSI_Nanorulers computes it
    before linking (its linked table keeps only corrected positions). Checks corrected = raw - drift."""
    corrected, drift = Drift_Correction_Functions().undrift(
        locs=kept.to_records(index=False),
        info=[{"Width": width, "Height": height, "Frames": n_frames, "Pixelsize": pixel_nm}],
        method="aim", segmentation=segmentation, intersect_d=intersect_nm / pixel_nm, roi_r=roi_nm / pixel_nm)
    dx, dy = np.asarray(drift.drift_x), np.asarray(drift.drift_y)
    f = kept["frame"].to_numpy().astype(int)
    err = np.abs(pd.DataFrame(corrected)["xc"].to_numpy() - (kept["xc"].to_numpy() - dx[f])).max()
    return pd.DataFrame(dict(frame=np.arange(len(dx)), drift_x_px=dx, drift_y_px=dy)), err


def ruler_sites(sites, min_angle_deg=170.0, min_events=30, max_on=0.2, max_sd_nm=40.0, link_nm=120.0):
    """The three sites of every straight ruler (the selection of ``_nanorulers.find_rulers``), one row
    per site: ``ruler``, ``pos`` (0-2 along the ruler), centre ``x``, ``y`` (nm) and SD ``sd``."""
    good = sites[(sites.n >= min_events) & (sites.on <= max_on) & (sites.smaj <= max_sd_nm)].reset_index(drop=True)
    group = DBSCAN(eps=link_nm, min_samples=1).fit_predict(good[["x", "y"]].to_numpy())
    rows = []
    for g in np.flatnonzero(np.bincount(group) == 3):
        sel = np.flatnonzero(group == g)
        pts = good.loc[sel, ["x", "y"]].to_numpy()
        c = pts - pts.mean(0)
        order = np.argsort(c @ np.linalg.svd(c)[2][0])
        pts = pts[order]
        u, w = pts[0] - pts[1], pts[2] - pts[1]
        angle = np.degrees(np.arccos(np.clip(u @ w / np.linalg.norm(u) / np.linalg.norm(w), -1, 1)))
        if angle < min_angle_deg:
            continue
        for pos, i in enumerate(sel[order]):
            rows.append(dict(ruler=g, pos=pos, x=good.x[i], y=good.y[i], sd=good.smaj[i]))
    return pd.DataFrame(rows)


def place(raw):
    """Position inside the 2 x 2 px Bayer cell, 0-2 px (0-1: even pixel, 1-2: odd pixel)."""
    return np.mod(np.asarray(raw) + 0.5, 2.0)


def ruler_events(linked, rsites, drift, pixel_nm, assign_nm, window):
    """Linked events within ``assign_nm`` of a straight-ruler site, with their raw camera position
    (``raw_x``, ``raw_y``), place in the Bayer cell (``u``, ``v``), offset from the site centre (``dx``,
    ``dy``, nm), relative pixel 3 QE minus the site mean (``dAR``), time window (``win``) and the same
    three referenced to the mean of all sites in that window (``dx_w``, ``dy_w``, ``dAR_w``)."""
    dist, idx = cKDTree(rsites[["x", "y"]].to_numpy()).query(linked[["xc", "yc"]].to_numpy() * pixel_nm,
                                                             distance_upper_bound=assign_nm)
    on = np.isfinite(dist)
    e = linked[on].copy()
    e["site"] = idx[on]
    e["ruler"] = rsites.ruler.to_numpy()[e.site]
    e["pos"] = rsites.pos.to_numpy()[e.site]
    f = e.frame.to_numpy().astype(int)
    e["raw_x"] = e.xc + drift.drift_x_px.to_numpy()[f]   # corrected = raw - drift
    e["raw_y"] = e.yc + drift.drift_y_px.to_numpy()[f]
    e["u"], e["v"] = place(e.raw_x), place(e.raw_y)
    e["dx"] = e.xc * pixel_nm - rsites.x.to_numpy()[e.site]
    e["dy"] = e.yc * pixel_nm - rsites.y.to_numpy()[e.site]
    e["dAR"] = e.A_R - e.groupby("site").A_R.transform("mean")
    e["win"] = (e.frame // window).astype(int)
    for c in ("dx", "dy", "dAR"):
        e[c + "_w"] = e[c] - e.groupby("win")[c].transform("mean")
    return e


def window_distances(e, pixel_nm, min_window_events):
    """Per ruler and window: the site centres (mean of the window's events, sites with at least
    ``min_window_events``), and the two neighbour distances minus that ruler's mean over windows, at the
    middle site's place in the cell. Returns (pivot of centres per ruler-window, distance table)."""
    cent = e.groupby(["site", "win"]).agg(ruler=("ruler", "first"), pos=("pos", "first"), x=("xc", "mean"),
                                          y=("yc", "mean"), n=("xc", "size"), rx=("raw_x", "mean"),
                                          ry=("raw_y", "mean")).reset_index()
    cent = cent[cent.n >= min_window_events]
    cent[["x", "y"]] *= pixel_nm
    piv = cent.pivot_table(index=["ruler", "win"], columns="pos", values=["x", "y", "rx", "ry", "n"]).dropna()
    parts = []
    for a, b in ((0, 1), (1, 2)):
        d = np.hypot(piv["x"][b] - piv["x"][a], piv["y"][b] - piv["y"][a])
        parts.append(pd.DataFrame(dict(dd=(d - d.groupby(level="ruler").transform("mean")).to_numpy(),
                                       u=place(piv["rx"][1].to_numpy()), v=place(piv["ry"][1].to_numpy()))))
    return piv, pd.concat(parts, ignore_index=True)


def cell_map(values, u, v, bins=CELL_BINS):
    """Mean and SEM of ``values`` in bins of the Bayer cell (rows: ``v``, columns: ``u``)."""
    n = len(bins) - 1
    iu = np.clip(np.digitize(u, bins) - 1, 0, n - 1)
    iv = np.clip(np.digitize(v, bins) - 1, 0, n - 1)
    g = pd.DataFrame(dict(x=np.asarray(values), k=iv * n + iu)).groupby("k").x
    m = g.mean().reindex(range(n * n)).to_numpy().reshape(n, n)
    s = (g.std() / np.sqrt(g.size())).reindex(range(n * n)).to_numpy().reshape(n, n)
    return m, s


# --------------------------------------------------------------------------- drawing
def draw_mosaic(ax, unit, x0, x1, y0, y1, scale=1.0, alpha=0.28, lw=0.4):
    """Each camera pixel in [x0, x1] x [y0, y1] (ROI px) tinted by its colour filter, ``unit`` the 2 x 2
    filters at the ROI origin (rows, columns); drawn in units of ``scale``."""
    for r in range(int(np.floor(y0 + 0.5)), int(np.ceil(y1 + 0.5))):
        for c in range(int(np.floor(x0 + 0.5)), int(np.ceil(x1 + 0.5))):
            ax.add_patch(Rectangle(((c - 0.5) * scale, (r - 0.5) * scale), scale, scale,
                                   facecolor=TINT[unit[r % 2, c % 2]], alpha=alpha, edgecolor="white", lw=lw,
                                   zorder=0))


def cell_axes(ax, unit):
    """A 2 x 2 px Bayer cell map: each quarter outlined in its filter's colour, no ticks."""
    for r in range(2):
        for c in range(2):
            ax.add_patch(Rectangle((c, r), 1, 1, fill=False, edgecolor=TINT[unit[r, c]], lw=2.0, zorder=5))
    ax.set_xlim(0, 2); ax.set_ylim(2, 0); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
