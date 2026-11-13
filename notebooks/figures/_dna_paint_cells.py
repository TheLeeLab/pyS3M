"""Shared code for the two-colour DNA-PAINT cell SI notebooks (Cy3B / ATTO 594 on the ZWO camera,
CF640R / ATTO 655 on the Ximea): quality cuts, drift correction + linking + fiducial removal, the
channel classification used for the close-red pair, FIRE, and the render / FRC figures.

Coordinates: ``xc`` is the image column and ``yc`` the row, in camera pixels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from sklearn.mixture import GaussianMixture

from pyS3M import render
from pyS3M.FRCFunctions import fire
from pyS3M.LinkingFunctions import link_localisations

# Text / line colour standing for each render colour map (cyan / hot are drawn black -> colour).
CMAP_TEXT_COLOUR = {"cyan": "#00ffff", "hot": "#ff8c1a"}
CMAP_LINE_COLOUR = {"cyan": "darkcyan", "hot": "darkorange"}


# --------------------------------------------------------------------------- localisations
def quality_cuts(loc, pixel_nm):
    """The cuts of the paper's analysis of both cells: localisation and PSF-width uncertainty < 40 nm,
    PSF width 75-250 nm, reduced chi^2 < 3, colour errors < 1 (the ~1 % of fits with A_R / A_G errors
    of 3 to > 500 link to anything in the joint clustering and bridge neighbouring clusters)."""
    keep = ((loc["xc_err"] > 0) & (loc["yc_err"] > 0)
            & (loc["xc_err"] < 40 / pixel_nm) & (loc["yc_err"] < 40 / pixel_nm)
            & (loc["s_x_err"] < 40 / pixel_nm) & (loc["s_y_err"] < 40 / pixel_nm)
            & loc["s_x"].between(75 / pixel_nm, 250 / pixel_nm, inclusive="neither")
            & loc["s_y"].between(75 / pixel_nm, 250 / pixel_nm, inclusive="neither")
            & (loc["chi_sqr"] < 3) & (loc["A_R_err"] < 1) & (loc["A_G_err"] < 1))
    return loc[keep].reset_index(drop=True)


def drift_link_fiducials(loc, width, height, pixel_nm, drift_corrector, fiducial_detector,
                         fiducial_fraction=0.075):
    """AIM drift correction, linking of repeat detections (within the median localisation error, up
    to 2 dark frames), and removal of fiducials (bright, dense regions of a 1x render holding
    localisations in at least ``fiducial_fraction`` of frames). Returns (linked, fiducial-free
    linked events, drift result)."""
    n_frames = int(loc["frame"].max()) + 1
    info = [{"Width": width, "Height": height, "Frames": int(loc["frame"].max()), "Pixelsize": pixel_nm}]
    corrected, drift = drift_corrector.undrift(locs=loc.to_records(index=False), info=info, method="aim",
                                               segmentation=20, intersect_d=20 / pixel_nm, roi_r=60 / pixel_nm)
    corrected = pd.DataFrame(corrected)
    link_r = 0.5 * (np.median(corrected["xc_err"]) + np.median(corrected["yc_err"]))
    linked = link_localisations(corrected, n_frames=n_frames, r_max=link_r, max_dark_time=2)

    info = [{"Width": width, "Height": height, "Frames": int(linked["frame"].max()), "Pixelsize": pixel_nm}]
    image_1x = render.render(locs=linked.to_records(index=False), info=info, blur_method="smooth",
                             oversampling=1)[1]
    centres, mask, _, _ = drift_corrector.detect_high_density_regions_from_image(
        smoothed_image=image_1x, histogram_bins=1024, threshold_percentile=99.94, pixelsize=pixel_nm,
        title="Fiducial detection")
    puncta, _ = drift_corrector.select_puncta_from_regions(
        locs=linked.to_records(index=False), region_centres=centres, binary_mask=mask, pixelsize=pixel_nm,
        selection_box_size_nm=350.0, min_localisations_per_region=int(fiducial_fraction * info[0]["Frames"]),
        title="Fiducial selection", create_plot=True)
    clean = pd.DataFrame(fiducial_detector.remove_puncta_locs(linked.to_records(index=False), puncta))
    return linked, clean, drift


def name_channels(assigned, dyes, col="channel"):
    """(``channel_0``, ``channel_1``): the assigned events of the lower and the higher mean A_R
    channel, named ``dyes`` in that order (the unmixing's channel numbers are arbitrary).
    Unassigned events (channel < 0) are dropped."""
    ids = [c for c in sorted(assigned[col].unique()) if c >= 0]
    if len(ids) != 2:
        raise ValueError(f"expected two channels, got {ids}")
    mean_ar = {c: assigned.loc[assigned[col] == c, "A_R"].mean() for c in ids}
    out = []
    for dye, c in zip(dyes, sorted(ids, key=mean_ar.get)):
        df = assigned[assigned[col] == c].reset_index(drop=True)
        print(f"{dye:>9} = channel {c} (mean A_R {mean_ar[c]:.3f}): {len(df):,} localisations")
        out.append(df)
    n_un = int((assigned[col] < 0).sum())
    print(f"unassigned: {n_un:,} ({n_un / len(assigned):.1%})")
    return tuple(out)


def classify_from_large_clusters(joint_locs, min_events=20, conf_threshold=0.90, cols=("A_R", "A_G")):
    """Classify the joint spatial-spectral clusters (``joint_cluster_id``) and isolated events of
    ``unmix_channels_joint_cluster`` against the mean colours of its large clusters.

    For a pair as close as CF640R / ATTO 655 the method's own GMM never separates the dyes (most of
    its clusters hold 3-9 events, whose noise swamps the fit), so:
    1. each cluster's colour is the inverse-variance weighted mean of its events (with its error);
    2. a 2-component GMM on the clusters with >= ``min_events`` events gives the two dyes' means,
       covariances and weights (component 0 = lower A_R);
    3. every cluster and isolated event is classified against them, its own measurement variance
       added to each component's covariance, and assigned if its posterior >= ``conf_threshold``.

    Returns the events with ``channel`` (0, 1 or -1 = unassigned) and a dict of the model and the
    per-cluster / per-isolated-event tables.
    """
    c0, c1 = cols
    df = joint_locs.copy()
    w0, w1 = df[f"{c0}_err"] ** -2, df[f"{c1}_err"] ** -2
    df["_w0"], df["_w1"], df["_x0"], df["_x1"] = w0, w1, w0 * df[c0], w1 * df[c1]
    inc = df["joint_cluster_id"] >= 0
    g = df[inc].groupby("joint_cluster_id").agg(n=(c0, "size"), x0=("_x0", "sum"), x1=("_x1", "sum"),
                                                w0=("_w0", "sum"), w1=("_w1", "sum"))
    clusters = pd.DataFrame({"n": g["n"], c0: g["x0"] / g["w0"], c1: g["x1"] / g["w1"],
                             "e0": g["w0"] ** -0.5, "e1": g["w1"] ** -0.5})
    iso = df[~inc]
    isolated = pd.DataFrame({"n": 1, c0: iso[c0], c1: iso[c1], "e0": iso[f"{c0}_err"], "e1": iso[f"{c1}_err"]})

    large = clusters[clusters["n"] >= min_events]
    gmm = GaussianMixture(2, covariance_type="full", n_init=10, random_state=0).fit(large[[c0, c1]])
    order = np.argsort(gmm.means_[:, 0])
    means, covs, weights = gmm.means_[order], gmm.covariances_[order], gmm.weights_[order]

    def posterior(units):
        X = units[[c0, c1]].to_numpy()
        v0, v1 = units["e0"].to_numpy() ** 2, units["e1"].to_numpy() ** 2
        logp = np.empty((len(X), 2))
        for k in range(2):
            d = X - means[k]
            a, b, c = covs[k][0, 0] + v0, covs[k][0, 1], covs[k][1, 1] + v1
            det = a * c - b * b
            quad = (c * d[:, 0] ** 2 - 2 * b * d[:, 0] * d[:, 1] + a * d[:, 1] ** 2) / det
            logp[:, k] = -0.5 * (quad + np.log(det)) + np.log(weights[k])
        logp -= logp.max(axis=1, keepdims=True)
        p = np.exp(logp)
        p /= p.sum(axis=1, keepdims=True)
        return p.argmax(axis=1), p.max(axis=1)

    for units in (clusters, isolated):
        units["channel"], units["confidence"] = posterior(units)
        units.loc[units["confidence"] < conf_threshold, "channel"] = -1

    df["channel"] = -1
    df.loc[inc, "channel"] = df.loc[inc, "joint_cluster_id"].map(clusters["channel"]).to_numpy()
    df.loc[~inc, "channel"] = isolated["channel"].to_numpy()
    df["channel"] = df["channel"].astype(int)
    info = dict(means=means, covariances=covs, weights=weights, n_large=len(large),
                large_events=int(large["n"].sum()), clusters=clusters, isolated=isolated)
    return df.drop(columns=["_w0", "_w1", "_x0", "_x1"]), info


# --------------------------------------------------------------------------- FIRE
def compute_fire(xy, pixel_nm, zoom=8, n_blocks=50):
    """FIRE resolution of one channel's (xc, yc) points (camera px), with the frequency axis
    (cycles per nm), the below-Nyquist mask and the FIRE value and its +-1 sigma band as
    frequencies, for plotting its FRC curve."""
    xy = np.asarray(xy, float)
    shifted = xy - xy.min(axis=0)
    nx, ny = int(np.ptp(xy[:, 0])) + 1, int(np.ptp(xy[:, 1])) + 1
    resolution_nm, frc_mean, res_hi_nm, res_lo_nm = fire(positions=shifted, nx=nx, ny=ny, zoom=zoom,
                                                         n_blocks=n_blocks, pixel_size_nm=pixel_nm)
    q = np.arange(len(frc_mean)) / (max(nx, ny) * zoom)   # cycles per SR pixel
    return dict(resolution_nm=resolution_nm, frc_mean=frc_mean, mask=(q > 0) & (q < 0.6),
                q_nm=q / (pixel_nm / zoom), q_cross=1.0 / resolution_nm, q_hi=1.0 / res_hi_nm,
                q_lo=1.0 / res_lo_nm, res_hi_nm=res_hi_nm, res_lo_nm=res_lo_nm)


# --------------------------------------------------------------------------- figures
def render_channels(channels, width, height, n_frames, pixel_nm, oversampling=8, min_blur_px=4):
    """Gaussian renders of each channel's events, ``oversampling`` SR px per camera px, with the
    paper's minimum blur (``min_blur_width = min_blur_px x pixel size in um``)."""
    info = [{"Width": width, "Height": height, "Frames": n_frames, "Pixelsize": pixel_nm}]
    return [render.render(ch.to_records(index=False), info, oversampling=oversampling, blur_method="gaussian",
                          min_blur_width=min_blur_px * pixel_nm / 1000)[1] for ch in channels]


def label_dyes(ax, dyes, cmaps, fontsize=6):
    """Write the dye names top left on a render, each in its channel's colour."""
    for k, (dye, cmap) in enumerate(zip(dyes, cmaps)):
        ax.text(0.02, 0.98 - 0.075 * k, dye, transform=ax.transAxes, ha="left", va="top", fontsize=fontsize,
                color=CMAP_TEXT_COLOUR.get(cmap, cmap), fontweight="bold",
                bbox=dict(boxstyle="square,pad=0.15", facecolor="black", alpha=0.6, linewidth=0))
    return ax


def plot_render(plotter, img_cyan, img_hot, pixel_nm, oversampling, size_in, dyes):
    """Two-channel overlay as published: ``dyes[0]`` in cyan, ``dyes[1]`` in hot; 1-99.9th
    percentile scale per channel, 10 um scale bar. Drawn at its printed size (``size_in``)."""
    fig, ax = plotter.one_column_plot(height=size_in, width=size_in)
    plotter.multichannel_overlay_plot(
        ax, [img_hot, img_cyan], cmaps=["hot", "cyan"], pixelsize=pixel_nm / oversampling,
        scalebarsize=10000, scalebarlabel="10 μm",
        vmins=[np.percentile(img_hot, 1), np.percentile(img_cyan, 1)],
        vmaxs=[np.percentile(img_hot, 99.9), np.percentile(img_cyan, 99.9)], brightness_boost=[1, 1])
    label_dyes(ax, dyes, ("cyan", "hot"))
    return fig, ax


def plot_frc(plotter, results, dyes, cmaps=("cyan", "hot")):
    """The FRC inset: each channel's mean FRC, its FIRE value (solid) and +-1 sigma band, the 1/7
    threshold, on a log axis labelled in nm. ``results``: two ``compute_fire`` dicts."""
    colours = [CMAP_LINE_COLOUR.get(c, c) for c in cmaps]
    fig, ax = plotter.one_column_plot(width=2.3, height=1.2)
    for r, colour in zip(results, colours):
        ax.plot(r["q_nm"][r["mask"]], r["frc_mean"][r["mask"]], color=colour, lw=1)
    for r, colour in zip(results, colours):
        ax.axvline(r["q_cross"], color=colour, ls="-", lw=1)
        ax.axvspan(r["q_lo"], r["q_hi"], alpha=0.15, color=colour)
    ax.axhline(1 / 7, color="gray", ls="--", lw=1)
    ax.set_xlabel("1 / spatial frequency (nm)", fontsize=8)
    ax.set_ylabel("FRC", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_xscale("log")
    ticks_nm = [500, 100, 20]
    ax.set_xticks([1 / d for d in ticks_nm], [f"{d} nm" for d in ticks_nm])
    ax.set_xlim(1 / 500, 1 / 15)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, ls="--", color="0.85", lw=0.5)
    ax.legend(handles=ax.get_lines()[:2],
              labels=[f"{d} ({r['resolution_nm']:.0f} nm)" for d, r in zip(dyes, results)],
              loc="lower left", bbox_to_anchor=(0, 0.16), fontsize=5, frameon=False, handlelength=1)
    plt.tight_layout()
    return fig, ax
