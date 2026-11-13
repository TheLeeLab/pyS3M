"""Helpers for the SI ATTO 647N camera figures (Thorlabs CS505CU and ZWO ASI 585MC):
a field of view with single-step photobleaching traces, and 25 PSF zoom-ins with Bayer keys.

Used by ``SI/FigureSI_ATTO647N_Cameras.ipynb``. Everything is in photoelectrons.
"""
import numpy as np
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max

from pyS3M import PlottingBase

TRACE_COLOURS = ("red", "lime", "lightblue")


def photon_transfer_gain(stack, clip_adu=1, n_bins=20, top_percentile=95):
    """Gain (ADU per photoelectron) of each Bayer pixel type, from the temporal mean and variance
    of each pixel over ``stack`` (raw ADU frames of mostly background, e.g. after bleaching).

    For shot-noise-limited pixels, var = gain x (mean - offset) + read-noise variance, so the slope
    of the variance against the mean is the gain. Pixels that ever read <= ``clip_adu`` are left
    out (their noise is clipped at the sensor's zero), as are the brightest (1 - top_percentile) %
    (remaining molecules). A pixel type with no unclipped pixels gives NaN.

    Returns:
        dict {(row % 2, col % 2): (gain, read noise in photoelectrons, fraction of pixels used)}.
    """
    stack = np.asarray(stack, dtype=np.float64)
    mean, var = stack.mean(0), stack.var(0, ddof=1)
    clipped = (stack <= clip_adu).any(0)
    out = {}
    for i in (0, 1):
        for j in (0, 1):
            m, v, c = mean[i::2, j::2].ravel(), var[i::2, j::2].ravel(), clipped[i::2, j::2].ravel()
            ok = ~c & (m < np.percentile(m, top_percentile))
            if ok.sum() < 100 * n_bins:
                out[(i, j)] = (np.nan, np.nan, ok.mean())
                continue
            edges = np.quantile(m[ok], np.linspace(0, 1, n_bins + 1))
            idx = np.clip(np.digitize(m[ok], edges[1:-1]), 0, n_bins - 1)
            xs = np.array([m[ok][idx == k].mean() for k in range(n_bins)])
            ys = np.array([np.median(v[ok][idx == k]) for k in range(n_bins)])
            gain, intercept = np.polyfit(xs, ys, 1)
            out[(i, j)] = (gain, np.sqrt(max(intercept, 0)) / gain, ok.mean())
    return out


def bayer_from_red_dye(crops, sensor_origins):
    """Filter colour of each (row % 2, col % 2) position on the sensor, read off red-dye PSFs.

    ATTO 647N emits at ~670 nm, where R pixels are the most sensitive, G less and B least, so the
    background-subtracted signal summed over each sensor parity ranks R > G, G > B. A Bayer layout
    needs the two G positions on a diagonal; anything else raises.

    Returns:
        (2, 2) mosaic unit (indexed [row % 2, col % 2] of the full sensor) and the signal fractions.
    """
    signal = np.zeros((2, 2))
    for c, (r0, c0) in zip(crops, sensor_origins):
        s = c - np.median(c)
        rows, cols = np.indices(c.shape)
        for i in (0, 1):
            for j in (0, 1):
                signal[i, j] += s[((rows + r0) % 2 == i) & ((cols + c0) % 2 == j)].sum()
    order = np.argsort(signal.ravel())  # ascending: B, G, G, R
    unit = np.empty(4, dtype="<U1")
    unit[order[0]], unit[order[1]], unit[order[2]], unit[order[3]] = "B", "G", "G", "R"
    unit = unit.reshape(2, 2)
    if not (unit[0, 1] == unit[1, 0] == "G" or unit[0, 0] == unit[1, 1] == "G"):
        raise ValueError(f"signal per sensor parity {np.round(signal / signal.sum(), 3).tolist()} "
                         f"is not a Bayer layout ({unit.tolist()})")
    return unit, signal / signal.sum()


def pixel_types(unit, sensor_origin, shape):
    """Bayer filter (0/1/2 = B/G/R) of each pixel of a crop whose top-left pixel sits at
    ``sensor_origin`` (row, col) on the full sensor, for ``PlottingBase.add_bayer_key``."""
    r0, c0 = sensor_origin
    rows, cols = np.indices(shape)
    colour = unit[(rows + r0) % 2, (cols + c0) % 2]
    return np.searchsorted(np.array(["B", "G", "R"]), colour)


def pick_psfs(pe, n, half_roi, sigma=1.5, isolation=0.2, skip_brightest=0.02):
    """Pick ``n`` clear, isolated single PSFs in one photoelectron frame.

    Peaks of the frame smoothed by ``sigma`` px, each at least ``half_roi`` px from the border and
    from any other peak, are scored by their height above the median of their own crop. Peaks with
    a second peak in their crop higher than ``isolation`` x their own height are dropped, and so
    are the brightest ``skip_brightest`` of the rest (likely several molecules or aggregates); the
    ``n`` brightest left are returned, brightest first.

    Returns:
        (n, 2) int array of (row, col) centres.
    """
    sm = gaussian_filter(np.asarray(pe, dtype=np.float64), sigma)
    peaks = peak_local_max(sm, min_distance=2, exclude_border=half_roi + 1)
    heights = []
    for r, c in peaks:
        crop = sm[r - half_roi:r + half_roi, c - half_roi:c + half_roi]
        heights.append(sm[r, c] - np.median(crop))
    heights = np.array(heights)
    keep = []
    for k, (r, c) in enumerate(peaks):
        if heights[k] <= 0:
            continue
        near = (np.abs(peaks[:, 0] - r) < 2 * half_roi) & (np.abs(peaks[:, 1] - c) < 2 * half_roi)
        near[k] = False
        if np.any(heights[near] > isolation * heights[k]):
            continue
        keep.append(k)
    keep = np.array(keep)
    keep = keep[np.argsort(-heights[keep])]
    keep = keep[int(np.ceil(skip_brightest * len(keep))):]
    if len(keep) < n:
        raise ValueError(f"only {len(keep)} isolated PSFs for {n} panels")
    return peaks[keep[:n]]


def box_trace(stack_pe, box, ring=4):
    """Photoelectrons of a molecule per frame: the sum over its ``box`` (col, row, size) minus the
    local background. The background of each Bayer pixel type is the median of that type's pixels
    in a ``ring`` px wide frame around the box (robust to a neighbouring molecule; per type, as the
    background differs between filter colours), times that type's pixel count in the box."""
    col, row, size = box
    stack = np.asarray(stack_pe, dtype=np.float64)
    inner = stack[:, row:row + size, col:col + size]
    outer = stack[:, row - ring:row + size + ring, col - ring:col + size + ring].copy()
    outer[:, ring:ring + size, ring:ring + size] = np.nan
    r_in, c_in = np.indices((size, size)) + np.array([row, col])[:, None, None]
    r_out, c_out = np.indices(outer.shape[1:]) + np.array([row - ring, col - ring])[:, None, None]
    trace = inner.sum((1, 2))
    for i in (0, 1):
        for j in (0, 1):
            sel_out = (r_out % 2 == i) & (c_out % 2 == j)
            n_in = int(((r_in % 2 == i) & (c_in % 2 == j)).sum())
            trace -= np.nanmedian(outer[:, sel_out], axis=1) * n_in
    return trace


def plot_fov_traces(plotter, frame_pe, boxes, traces, exposure_s, pixel_size_nm, xlim=None,
                    scalebar_um=10):
    """a) the field of view (photoelectrons, colour bar) with each molecule's box; b) its traces."""
    fig, axs = plotter.two_column_plot(ncols=2)
    axs[0], _ = plotter.image_plot(
        axs[0], frame_pe, vmin=np.percentile(frame_pe, 1), vmax=np.percentile(frame_pe, 99.5),
        pixelsize=pixel_size_nm, scalebar=True, scalebarsize=scalebar_um * 1e3,
        scalebarlabel=f"{scalebar_um} μm", colorbar=True, colorbar_label="photoelectrons",
    )
    for (col, row, size), colour in zip(boxes, TRACE_COLOURS):
        axs[0].add_patch(Rectangle((col - 0.5, row - 0.5), size, size, lw=0.5, ec=colour, fc="none"))
    for trace, colour in zip(traces, TRACE_COLOURS):
        t = np.arange(len(trace)) * exposure_s
        axs[1].plot(t, trace, color=colour, lw=0.75)
    axs[1].set_xlabel("time / s")
    axs[1].set_ylabel("photoelectrons")
    axs[1].set_xlim(xlim if xlim is not None else (0, max(len(tr) for tr in traces) * exposure_s))
    axs[1].set_ylim(bottom=min(0, min(np.percentile(tr, 1) for tr in traces)))
    for ax, label in zip(axs, "ab"):
        ax.text(-0.1, 1.02, f"{label})", transform=ax.transAxes, fontsize=8, fontweight="bold")
    return fig, axs


def plot_zoom_ins(plotter, crops, types, pixel_size_nm, scalebar_nm=300):
    """5 x 5 grid of photoelectron crops, each with its own colour bar (labelled in the rightmost
    column only) and the 2 x 2 Bayer key of its top-left corner."""
    fig, axs = plotter.two_column_plot(nrows=5, ncols=5, height=6.3)
    n_cols = axs.shape[1]
    for k, (ax, c, pt) in enumerate(zip(axs.flat, crops, types)):
        plotter.image_plot(
            ax, c, cmap="gray", origin="upper",
            vmin=np.percentile(c, 1), vmax=np.percentile(c, 99),
            colorbar=True, colorbar_label="photoelectrons" if k % n_cols == n_cols - 1 else "",
            scalebar=True, pixelsize=pixel_size_nm, scalebarsize=float(scalebar_nm),
            scalebarlabel=f"{scalebar_nm} nm",
        )
        PlottingBase.add_bayer_key(ax, pt, origin="upper")
    for ax in list(axs.flat)[len(crops):]:
        ax.axis("off")
    return fig, axs
