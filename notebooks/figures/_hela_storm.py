"""Helpers for the SI dSTORM HeLa figure (`SI/FigureSI_HeLa_STORM.ipynb`): line profiles through
the super-resolved render, their two-Gaussian fits, and the composite figure laid out as the
published SI figure (169.3 x 50.1 mm: a) render with the zoom box, b) zoom with the two lines,
c) the two profiles with each peak's FWHM and the peak separation).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from skimage.draw import line as bresenham_line

FWHM_PER_SIGMA = 2 * np.sqrt(2 * np.log(2))
NAVY = "#000080"
MM = 1 / 25.4

# Published layout, mm from the top-left corner of the figure: (x, y, width, height).
FIG_MM = (169.33, 50.07)
AX_A = (0.09, 0.09, 62.25, 49.87)
AX_B = (63.89, 0.09, 49.89, 49.89)
AX_C = [(118.53, 0.10, 50.71, 20.91), (118.53, 22.94, 50.71, 20.91)]
LABEL_C_X = 114.63
FWHM_ARROW_NM = 205   # width of the FWHM arrows (a label, not to scale), as published


def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def double_gaussian(x, A1, mu1, s1, A2, mu2, s2, offset):
    return gaussian(x, A1, mu1, s1) + gaussian(x, A2, mu2, s2) + offset


def line_profile(zoom, line, oversampling, sr_px_nm, bin_nm):
    """Mean render intensity along a line, in ``bin_nm`` bins of distance from its first end.

    ``line`` is ((row, col), (row, col)) in camera px within the zoom (array rows; drawn with
    ``origin="lower"`` they count up from the bottom).
    Returns the SR-pixel coordinates of the line and the bin centres and means.
    """
    (r0, c0), (r1, c1) = line
    rows, cols = bresenham_line(r0 * oversampling, c0 * oversampling, r1 * oversampling, c1 * oversampling)
    values = zoom[rows, cols]
    dist = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(rows.astype(float)), np.diff(cols.astype(float))))])
    dist *= sr_px_nm
    edges = np.arange(dist[0], dist[-1] + bin_nm, bin_nm)
    sums, edges = np.histogram(dist, bins=edges, weights=values)
    counts, _ = np.histogram(dist, bins=edges)
    means = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    return rows, cols, (edges[:-1] + edges[1:]) / 2, means


def fit_two_peaks(centres, means, bin_nm):
    """Two Gaussians plus an offset fitted to a profile, started at its two most prominent peaks.

    Returns the fit parameters (A1, mu1, s1, A2, mu2, s2, offset) with the peaks in order of
    position, and their standard errors; None if there aren't two peaks or the fit fails.
    """
    peaks, props = find_peaks(means, distance=3, prominence=means.max() * 0.2)
    if len(peaks) < 2:
        return None
    p1, p2 = sorted(peaks[np.argsort(props["prominences"])[-2:]])
    p0 = [means[p1], centres[p1], 2 * bin_nm, means[p2], centres[p2], 2 * bin_nm, 0.0]
    try:
        popt, pcov = curve_fit(double_gaussian, centres, means, p0=p0, bounds=(0, np.inf), maxfev=5000)
    except RuntimeError:
        return None
    return popt, np.sqrt(np.diag(pcov))


def _ax_mm(fig, box):
    x, y, w, h = box
    W, H = FIG_MM
    return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])


def _frame(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.5)
        s.set_color("k")


def _scalebar(ax, length_px, label, bar_px):
    ax.add_artist(AnchoredSizeBar(ax.transData, length_px, label, loc="lower right", pad=0.3, borderpad=0.3,
                                  sep=1.5, color="white", frameon=False, size_vertical=bar_px, label_top=True,
                                  fontproperties={"size": 6}))


def _panel_label(fig, x_mm, y_mm, text, color):
    W, H = FIG_MM
    fig.text(x_mm / W, 1 - y_mm / H, text, color=color, fontsize=10, fontweight="bold", ha="left", va="top")


def _profile_axis(ax, centres, means, fit, bin_nm, label, xlim):
    ax.bar(centres, means, width=bin_nm * 0.85, color="#888888", edgecolor="k", linewidth=0.4, zorder=2)
    ymax = 1.29 * means[(centres >= xlim[0]) & (centres <= xlim[1])].max()
    ax.set_xlim(xlim)
    ax.set_ylim(0, ymax)
    ax.set_xticks(np.arange(400, xlim[1], 200))
    ax.set_yticks([0.355 * ymax, 0.71 * ymax])   # two gridlines, as published (no y scale)
    ax.set_yticklabels([])
    ax.tick_params(labelsize=7, width=0.5, length=2.2, pad=1.5)
    ax.grid(lw=0.5, ls="--", color="gray", zorder=0)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_linewidth(0.5)
    ax.text(0.01, 0.98, label, transform=ax.transAxes, fontsize=8, fontweight="bold", ha="left", va="top")
    if fit is None:
        return None
    popt, perr = fit
    for k in (0, 3):   # each Gaussian (plus the offset) over its peak, +-3 sigma
        mu, sig = popt[k + 1], popt[k + 2]
        x = np.linspace(mu - 3 * sig, mu + 3 * sig, 200)
        ax.plot(x, gaussian(x, *popt[k:k + 3]) + popt[6], "--", color=NAVY, lw=0.75, zorder=3)
    arrow = dict(arrowstyle="<->", color=NAVY, lw=0.5, shrinkA=0, shrinkB=0, mutation_scale=4)
    trans = ax.get_xaxis_transform()
    fwhm = FWHM_PER_SIGMA * popt[[2, 5]]
    for mu, f in zip(popt[[1, 4]], fwhm):   # each peak's FWHM above it (fixed-width arrow, as published)
        ax.annotate("", (mu - FWHM_ARROW_NM / 2, 0.80), (mu + FWHM_ARROW_NM / 2, 0.80), xycoords=trans,
                    arrowprops=arrow)
        ax.text(mu, 0.84, f"{f:.0f} nm", transform=trans, color=NAVY, fontsize=6, ha="center", va="bottom")
    mu1, mu2 = popt[1], popt[4]   # peak separation, between the two centres
    ax.annotate("", (mu1, 0.57), (mu2, 0.57), xycoords=trans, arrowprops=arrow)
    ax.text((mu1 + mu2) / 2, 0.61, f"{mu2 - mu1:.0f} nm", transform=trans, color=NAVY, fontsize=6,
            ha="center", va="bottom")
    return dict(separation=mu2 - mu1, separation_err=np.hypot(perr[1], perr[4]), fwhm=fwhm)


def plot_hela_storm(image, crop_sr, box, oversampling, sr_px_nm, lines, bin_nm, vmax_pct=(99.6, 99.8),
                    xlim=(400, 1500)):
    """The published SI dSTORM figure: a) cropped render with the zoom box, joined to b) the zoom
    with the two numbered lines, c) the line profiles with two-Gaussian fits.

    Args:
        image: full super-resolved render (SR px).
        crop_sr: (row slice, col slice) of ``image`` shown in a.
        box: zoom box dict(x0, y0, w, h) in camera px.
        lines: two ((row, col), (row, col)) lines in camera px within the box.

    Both images are drawn with ``origin="lower"`` (row 0 at the bottom), as in the published
    figure, so the lines' rows count up from the bottom of the zoom.
        bin_nm: profile bin width.
    Returns:
        fig and a list with each line's separation and FWHMs (None if its fit failed).
    """
    W, H = FIG_MM
    fig = plt.figure(figsize=(W * MM, H * MM))
    fig.set_layout_engine("none")   # axes are placed in mm
    vmin = np.percentile(image, 1)
    crop = image[crop_sr]
    zoom = image[box["y0"] * oversampling:(box["y0"] + box["h"]) * oversampling,
                 box["x0"] * oversampling:(box["x0"] + box["w"]) * oversampling]

    # a) render, box, scale bar
    ax_a = _ax_mm(fig, AX_A)
    ax_a.imshow(crop, cmap="hot", vmin=vmin, vmax=np.percentile(image, vmax_pct[0]), origin="lower",
                aspect="auto", interpolation="none")
    _frame(ax_a)
    bx = box["x0"] * oversampling - crop_sr[1].start
    by = box["y0"] * oversampling - crop_sr[0].start
    bw, bh = box["w"] * oversampling, box["h"] * oversampling
    ax_a.add_patch(Rectangle((bx, by), bw, bh, lw=0.75, ec="white", fc="none"))
    _scalebar(ax_a, 10000 / sr_px_nm, "10 μm", crop.shape[0] * 0.5 / AX_A[3])

    # b) zoom, numbered lines, scale bar
    ax_b = _ax_mm(fig, AX_B)
    ax_b.imshow(zoom, cmap="hot", vmin=vmin, vmax=np.percentile(image, vmax_pct[1]), origin="lower",
                aspect="auto", interpolation="none")
    _frame(ax_b)
    profiles = []
    for k, line in enumerate(lines):
        rows, cols, centres, means = line_profile(zoom, line, oversampling, sr_px_nm, bin_nm)
        profiles.append((centres, means))
        ax_b.plot(cols, rows, color="white", lw=0.75, ls="-" if k == 0 else "--", solid_capstyle="butt")
        top = np.argmax(rows)   # number each line beside its upper end (origin lower: rows go up)
        ax_b.text(cols[top] - 3 * oversampling, rows[top] + 2 * oversampling, str(k + 1), color="white",
                  fontsize=6, fontweight="bold", ha="center", va="bottom")
    _scalebar(ax_b, 1000 / sr_px_nm, "1 μm", zoom.shape[0] * 0.6 / AX_B[3])

    # box corners -> b's left corners: white over the render, black across the white gap
    to_fig = fig.transFigure.inverted()
    edge = ax_a.transAxes.transform((1, 0))[0]
    for corner, target in (((bx + bw, by + bh), (0, 1)), ((bx + bw, by), (0, 0))):
        p0 = to_fig.transform(ax_a.transData.transform(corner))
        p1 = to_fig.transform(ax_b.transAxes.transform(target))
        xe = to_fig.transform((edge, 0))[0]
        pe = (xe, p0[1] + (p1[1] - p0[1]) * (xe - p0[0]) / (p1[0] - p0[0]))
        fig.add_artist(Line2D([p0[0], pe[0]], [p0[1], pe[1]], color="white", lw=0.5))
        fig.add_artist(Line2D([pe[0], p1[0]], [pe[1], p1[1]], color="k", lw=0.5))

    # c) profiles
    results = []
    axs_c = [_ax_mm(fig, box_mm) for box_mm in AX_C]
    for k, (ax, (centres, means)) in enumerate(zip(axs_c, profiles)):
        fit = fit_two_peaks(centres, means, bin_nm)
        results.append(_profile_axis(ax, centres, means, fit, bin_nm, str(k + 1), xlim))
    axs_c[0].set_xticklabels([])
    axs_c[1].set_xlabel("distance / nm", fontsize=8, labelpad=1)

    _panel_label(fig, 0.62, 0.62, "a)", "white")
    _panel_label(fig, 64.34, 0.62, "b)", "white")
    _panel_label(fig, LABEL_C_X, 0.62, "c)", "black")
    return fig, results
