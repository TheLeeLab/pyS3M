"""Helpers for the SI SBR figures (`SI/FigureSI_SBR.ipynb`), drawn as published (S34-S36).

- `ternary_cases_figure`: S34, colour-simplex scatter of the fits for three (N, SBR) cases (columns) and three
  dyes (rows), 466 x 400 pt. Positions are measured from the published PDF: triangle side 104.5 pt, columns
  158.2 pt and rows 128 pt apart, ticks 7 pt, axis labels and column headers 8 pt, row letters 9 pt bold.
  Axes: top corner = pixel 3 (``A_R``), left = pixel 2 (``A_G``), right = pixel 1 (``A_B``); the pixel labels
  sit along the side that carries their ticks, in the pixel's colour.
- `surface_figure`: S35/S36, precision over (N_photons, SBR) per dye.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator

PT = 1 / 72
PIXEL_COLOURS = {"pixel 1": "#5858df", "pixel 2": "#53b03b", "pixel 3": "#ff2a2a"}   # as Figs. 2e, 3f

# The three cases (columns): (N_photons, SBR, point colour).
CASES = [(5000, 100.0, "#2196F3"), (500, 100.0, "#FF9800"), (5000, 1.1, "#E91E63")]

# S34 layout (pt, from the figure's top-left corner).
TERNARY_FIG_PT = (466.1, 399.7)   # published 464.1; +2 so the last "0.0" is not clipped
SIDE_PT = 104.5
APEX0_PT = (84.0, 29.0)          # apex of the top-left triangle
COL_STEP_PT, ROW_STEP_PT = 158.2, 128.0
HEADER_Y_PT = 2.0                # top of the column headers
ROW_LETTER_X_PT, ROW_LETTER_DY_PT = 12.5, -11.6    # 'a)' left edge; top relative to the apex
DYE_X_PT = 1.5                   # left edge of the rotated dye names
# Pixel-label centres relative to the apex (x right, y down), and their rotations.
PIXEL_LABELS = [("pixel 1", (-0.6, 110.6), 0), ("pixel 2", (-47.2, 32.0), 60), ("pixel 3", (42.2, 31.6), -60)]


def load_case(save_folder, dye, flag, photon_level, n_sub=3000, rng=None):
    """Fitted (A_B, A_G, A_R) of one sweep point, at most ``n_sub`` of them (random subsample).

    Args:
        save_folder: The sweep's raw-results folder.
        dye: Dye name.
        flag: The SBR's file-name flag (``sbr_flag(sbr)``).
        photon_level: Index into the sweep's photon levels.
        n_sub: Maximum number of fits returned.
        rng: ``np.random.Generator`` (default: seeded 0, so the figure is reproducible).

    Returns:
        (n, 3) array of B, G, R fractions, or None if the file is missing.
    """
    from pathlib import Path
    rng = np.random.default_rng(0) if rng is None else rng
    h5 = Path(save_folder) / f"{flag}LM_method_{dye.replace('/', '-')}_rawresults.h5"
    if not h5.exists():
        return None
    fits = pd.read_hdf(str(h5), key="data")
    fits = fits[fits["photon_level"] == photon_level]
    ok = ((fits["s_x"] > 0) & (fits["s_y"] > 0) & (fits["xc"] > 0) & (fits["yc"] > 0)).to_numpy()
    bgr = fits[["A_B", "A_G", "A_R"]].to_numpy()[ok]
    return bgr[rng.choice(len(bgr), min(len(bgr), n_sub), replace=False)]


def _style_ternary(ax):
    ax.set_facecolor("black")
    for axis in (ax.taxis, ax.laxis, ax.raxis):
        axis.set_major_locator(MultipleLocator(0.2))
        axis.set_minor_locator(MultipleLocator(0.1))
        axis.set_tick_params(labelsize=7)
    ax.grid(lw=0.5, alpha=0.25, ls="--", which="both", axis="both")
    ax.set_tlabel("")
    ax.set_llabel("")
    ax.set_rlabel("")


def _header(n_photons, sbr):
    return rf"($N_{{\rm photon}}$={n_photons:,}, SBR={sbr:g})"


def ternary_cases_figure(samples, truths, dyes, cases=CASES):
    """S34: rows = dyes, columns = cases, fitted colours on black ternaries, the true colour as a white +.

    Args:
        samples: ``samples[i_dye][i_case]`` = (n, 3) B, G, R fractions (``load_case``) or None.
        truths: ``truths[i_dye]`` = the dye's true (B, G, R) fractions, or None.
        dyes: Dye names (row labels).
        cases: (N_photons, SBR, colour) per column; N and SBR are written in the column headers.

    Returns:
        The figure.
    """
    import mpltern  # noqa: F401  (registers the "ternary" projection)
    W, H = TERNARY_FIG_PT
    fig = plt.figure(figsize=(W * PT, H * PT))
    fig.set_layout_engine("none")
    fig.patch.set_facecolor("white")
    height = SIDE_PT * np.sqrt(3) / 2

    def fx(x):
        return x / W

    def fy(y):
        return 1 - y / H

    for i_d, dye in enumerate(dyes):
        for i_c, (n_ph, sbr, colour) in enumerate(cases):
            ax_x = APEX0_PT[0] + i_c * COL_STEP_PT
            ax_y = APEX0_PT[1] + i_d * ROW_STEP_PT
            ax = fig.add_axes([fx(ax_x - SIDE_PT / 2), fy(ax_y + height), SIDE_PT / W, height / H],
                              projection="ternary", ternary_sum=1)
            _style_ternary(ax)
            bgr = samples[i_d][i_c]
            if bgr is None:
                ax.text(0.5, 0.3, "missing", color="white", fontsize=7, ha="center", transform=ax.transAxes)
            else:
                # mpltern order is (top, left, right) = (R, G, B)
                ax.scatter(bgr[:, 2], bgr[:, 1], bgr[:, 0], s=0.8, alpha=0.25, color=colour,
                           edgecolors="none", rasterized=True)
            if truths[i_d] is not None:
                b, g, r = truths[i_d]
                ax.scatter([r], [g], [b], s=30, marker="+", color="white", linewidths=0.8, zorder=5)
            for text, (dx, dy), rot in PIXEL_LABELS:
                fig.text(fx(ax_x + dx), fy(ax_y + dy), text, color=PIXEL_COLOURS[text], fontsize=8,
                         rotation=rot, ha="center", va="center", rotation_mode="anchor")
            if i_d == 0:
                fig.text(fx(ax_x), fy(HEADER_Y_PT), _header(n_ph, sbr), fontsize=8, ha="center", va="top")
        apex_y = APEX0_PT[1] + i_d * ROW_STEP_PT
        fig.text(fx(ROW_LETTER_X_PT), fy(apex_y + ROW_LETTER_DY_PT), f"{'abc'[i_d]})", fontsize=9,
                 fontweight="bold", ha="left", va="top")
        fig.text(fx(DYE_X_PT), fy(apex_y + height / 2), dye, fontsize=8, rotation=90, ha="left",
                 va="center")
    return fig


METRIC_LABELS = {
    "sigma_xy": r"$\sigma_{\rm xy}$/nm",
    "sigma_colour": r"$\sigma_{\rm colour}$/relative pixel QE",
}


def surface_figure(sbr_xr, metric, fontsize=7, cmap="viridis"):
    """S35 (``metric="sigma_xy"``) / S36 (``"sigma_colour"``): one panel per dye, N_photons (log x) by SBR
    (log y, high SBR at the top), one colour scale from the minimum to the 99th percentile over all dyes.

    Args:
        sbr_xr: DataArray [dye, sbr, n_photons, metric] (the notebook's ``SBR_XR``).
        metric: ``"sigma_xy"`` or ``"sigma_colour"``.
        fontsize: Base font size (the publication style's).

    Returns:
        The figure.
    """
    data_all = sbr_xr.sel(metric=metric).values
    norm = mcolors.Normalize(vmin=float(np.nanmin(data_all)), vmax=float(np.nanpercentile(data_all, 99)))
    dyes = list(sbr_xr.coords["dye"].values)
    fig, axs = plt.subplots(nrows=1, ncols=len(dyes), figsize=(6.69, 2.4), squeeze=False)
    for i_d, dye in enumerate(dyes):
        ax = axs[0, i_d]
        # rasterized: no seams between the mesh cells in the SVG
        ax.pcolormesh(sbr_xr.coords["n_photons"].values, sbr_xr.coords["sbr"].values,
                      sbr_xr.isel(dye=i_d).sel(metric=metric).values,
                      cmap=cmap, shading="auto", norm=norm, rasterized=True)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$N_{\rm photons}$", fontsize=fontsize + 1)
        ax.set_ylabel("SBR" if i_d == 0 else "", fontsize=fontsize + 1)
        ax.set_title(dye, fontsize=fontsize)   # as published: the dye names head the panels
        ax.tick_params(labelsize=fontsize - 1)
        ax.invert_yaxis()
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs, label=METRIC_LABELS[metric],
                 shrink=0.85, aspect=20, pad=0.02)
    return fig
