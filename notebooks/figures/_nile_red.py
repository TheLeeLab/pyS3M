"""Shared code for the Nile Red figure notebooks (Figure 5 g-i, k-m): relative pixel 3 QE per pixel.

Across these samples nearly all of the real pixel-to-pixel colour variation is red vs green (blue stays
near 3 %), so one number per analysis pixel, its relative pixel 3 QE f_R = R / (R + G + B), keeps it.
Before any spectral model, it is the photoelectron-weighted colour of the pixel's localisations.
"""

import numpy as np
import pandas as pd

PIXEL3_LABEL = "relative pixel 3 QE"
SCALE_SD = 2.5   # colour scale: population mean +- SCALE_SD standard deviations over pixels


def red_share_pixelated(df, pixel_size_nm=50.0, camera_pixel_size_nm=69.0, min_localisations=10,
                        aggregate_id_column="cluster_id"):
    """Relative pixel 3 QE f_R = R / (R + G + B) of each analysis pixel's summed photoelectrons.

    Each localisation contributes A_c x photoelectrons to channel c. Same grid and fallback as
    ``NileRed_Functions.fit_wavelengths_pixelated``: pixels keyed by (aggregate, ix, iy), and pixels
    with fewer than ``min_localisations`` usable localisations take their aggregate's value. The error
    is the empirical one, sqrt(sum_i w_i^2 (f_R,i - F_R)^2) / sum_i w_i (w_i = photoelectrons), so a
    few localisations with a failed covariance (huge ``A_R_err``) do not dominate it.

    Args:
        df: Localisations with ``A_R``, ``A_G``, ``A_B``, ``photoelectrons``, ``xc``, ``yc`` (camera px)
            and the aggregate column.
        pixel_size_nm: Analysis pixel (sample-plane nm).
        camera_pixel_size_nm: Camera pixel (sample-plane nm).
        min_localisations: Fewest localisations for a pixel to keep its own value.
        aggregate_id_column: Aggregate (or bacterium) id column.

    Returns:
        (df with ``red_share`` and ``red_share_err`` (its pixel's value) added, one row per pixel with
        at least ``min_localisations``: ``red_share``, ``red_share_err``, ``n_locs``, ``photoelectrons``).
    """
    f = df[["A_R", "A_G", "A_B"]].to_numpy(dtype=np.float64)
    pe = df["photoelectrons"].to_numpy(dtype=np.float64)
    valid = np.isfinite(f).all(axis=1) & (f.sum(axis=1) > 0) & np.isfinite(pe) & (pe > 0)
    w = np.where(valid, pe, 0.0)
    f = np.where(valid[:, None], f, 0.0)
    f_R = np.divide(f[:, 0], f.sum(axis=1), out=np.zeros(len(f)), where=valid)
    x_nm = df["xc"].to_numpy() * camera_pixel_size_nm
    y_nm = df["yc"].to_numpy() * camera_pixel_size_nm
    g = pd.DataFrame({
        "agg": df[aggregate_id_column].to_numpy(),
        "ix": np.floor((x_nm - np.floor(x_nm.min() / pixel_size_nm) * pixel_size_nm) / pixel_size_nm).astype(int),
        "iy": np.floor((y_nm - np.floor(y_nm.min() / pixel_size_nm) * pixel_size_nm) / pixel_size_nm).astype(int),
        "wR": w * f[:, 0], "wtot": w * f.sum(axis=1), "w2": w ** 2, "w2f": w ** 2 * f_R, "w2f2": w ** 2 * f_R ** 2,
        "photoelectrons": w, "n_locs": valid.astype(float),
    })
    sums = ["wR", "wtot", "w2", "w2f", "w2f2", "photoelectrons", "n_locs"]

    def share(s):
        F = s["wR"] / s["wtot"]
        var = np.clip(s["w2f2"] - 2 * F * s["w2f"] + F ** 2 * s["w2"], 0, None)
        return F, np.sqrt(var) / s["photoelectrons"]

    pix = g.groupby(["agg", "ix", "iy"])[sums].transform("sum")
    agg = g.groupby("agg")[sums].transform("sum")
    use_pixel = (pix["n_locs"] >= min_localisations).to_numpy()[:, None]
    src = pd.DataFrame(np.where(use_pixel, pix, agg), columns=sums, index=g.index)
    out = df.copy()
    out["red_share"], out["red_share_err"] = share(src)
    pixels = g.groupby(["agg", "ix", "iy"])[sums].sum()
    pixels = pixels[pixels["n_locs"] >= min_localisations]
    pixels["red_share"], pixels["red_share_err"] = share(pixels)
    return out, pixels[["red_share", "red_share_err", "n_locs", "photoelectrons"]].reset_index()


def colour_scale(pixels, n_sd=SCALE_SD):
    """(mean, SD, lower, upper) of the per-pixel relative pixel 3 QE; the scale is mean +- n_sd SD."""
    m, s = pixels["red_share"].mean(), pixels["red_share"].std()
    return m, s, m - n_sd * s, m + n_sd * s


def sized_image(img, height_pt, pixel_nm, scalebar_nm, scalebar_label, scalebar_pt=1.5, scalebar_fontsize=6,
                cbar=None, margin_pt=1.5, panel_label=None, panel_label_fontsize=8):
    """A rendered RGB image drawn at an exact size (pt), as placed in the paper's figure.

    The image (origin lower) is ``height_pt`` tall and as wide as its aspect ratio makes it, with a
    white scale bar ``margin_pt`` from its lower-right corner, the label centred above it. ``cbar``
    (optional) adds a colour bar beside it: dict(c_min, c_max, label, width_pt, gap_pt, tick_fontsize,
    label_fontsize=8, cmap="jet"). ``panel_label`` (optional) is drawn bold, white, top left. Save with ``bbox_inches="tight"``: the sizes are unchanged.

    Returns:
        (fig, ax) with ax in image pixel coordinates.
    """
    import matplotlib.pyplot as plt

    pt = 1 / 72
    h, w = img.shape[:2]
    img_h = height_pt
    img_w = height_pt * w / h
    extra = (cbar["gap_pt"] + cbar["width_pt"] + 40) if cbar else 0
    fig = plt.figure(figsize=((img_w + extra + 2) * pt, (img_h + 2) * pt))
    fig.set_layout_engine("none")
    W, H = fig.get_size_inches() / pt
    ax = fig.add_axes([1 / W, 1 / H, img_w / W, img_h / H])
    ax.imshow(img, origin="lower", interpolation="none", aspect="auto")
    ax.set_axis_off()

    # scale bar in data (image pixel) units, offsets converted from pt
    px_per_pt = w / img_w
    length = scalebar_nm / pixel_nm
    x1 = w - margin_pt * px_per_pt
    y0 = margin_pt * px_per_pt
    ax.add_patch(plt.Rectangle((x1 - length, y0), length, scalebar_pt * px_per_pt, color="white", lw=0))
    # label centred over the bar, unless that would run it past the image's right edge
    label_w = 0.55 * scalebar_fontsize * len(scalebar_label) * px_per_pt   # rough text width
    x_label, ha = (x1 - length / 2, "center") if label_w / 2 <= length / 2 + margin_pt * px_per_pt else (x1, "right")
    ax.text(x_label, y0 + (scalebar_pt + 1.0) * px_per_pt, scalebar_label, color="white",
            fontsize=scalebar_fontsize, ha=ha, va="bottom")
    if panel_label is not None:   # bold white, top left (as the published Figure 5l numbers)
        ax.text(margin_pt * px_per_pt, h - margin_pt * px_per_pt, str(panel_label), color="white",
                fontsize=panel_label_fontsize, fontweight="bold", ha="left", va="top")
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(-0.5, h - 0.5)

    if cbar:
        cax = fig.add_axes([(1 + img_w + cbar["gap_pt"]) / W, 1 / H, cbar["width_pt"] / W, img_h / H])
        sm = plt.cm.ScalarMappable(cmap=cbar.get("cmap", "jet"),
                                   norm=plt.Normalize(vmin=cbar["c_min"], vmax=cbar["c_max"]))
        cb = fig.colorbar(sm, cax=cax)
        cb.ax.tick_params(labelsize=cbar["tick_fontsize"], length=2, width=0.5, pad=1)
        cb.outline.set_linewidth(0.5)
        cb.set_label(cbar["label"], fontsize=cbar.get("label_fontsize", 8), labelpad=2)
    return fig, ax


def render_red_share(render, locs, c_min, c_max, oversampling, mindensperc=1.0, maxdensperc=99.5,
                     min_blur_width=0.0):
    """The relative pixel 3 QE render used for every Figure 5 panel (S. aureus and alpha-syn alike).

    Each localisation is a Gaussian of its own precision (at least ``min_blur_width``); hue = photon-weighted
    ``red_share`` on ``c_min``-``c_max`` (jet). ``render`` sets its own brightness (density, 1-99.9
    percentiles, below 0.1 of that set to black, which also drops their saturation); that brightness is
    then replaced by the density clipped to its ``mindensperc``-``maxdensperc`` percentiles over the whole
    field, so those faintest pixels come back grey. (This is the published S. aureus render.) Crops
    are cut from this image unchanged, so they share its brightness scale.

    Args:
        render: The ``render`` module (passed in so this file needs no src path).
        locs: DataFrame with ``red_share`` (from ``red_share_pixelated``), ``xc``, ``yc``, ``xc_err``, ``yc_err``.
        c_min, c_max: Colour scale (relative pixel 3 QE).
        oversampling: Render pixels per camera pixel.
        min_blur_width: Smallest Gaussian width (camera px); 0 (the S. aureus render) for none.

    Returns:
        (density image, RGB image), both (rows, cols) in render pixels, origin at camera (0, 0).
    """
    from matplotlib import colors as mcolors

    _, grey, colour = render.render(
        locs=locs.to_records(index=False), oversampling=oversampling,
        viewport=((0, 0), (np.max(locs["yc"]), np.max(locs["xc"]))),
        blur_method="gaussian_colour", cparam="red_share", c_min=c_min, c_max=c_max,
        min_blur_width=min_blur_width)
    lo, hi = np.percentile(grey, mindensperc), np.percentile(grey, maxdensperc)
    hsv = mcolors.rgb_to_hsv(colour)
    hsv[..., 2] = np.clip((grey - lo) / (hi - lo), 0, 1)
    return grey, mcolors.hsv_to_rgb(hsv)


def aggregate_red_share(df, aggregate_id_column="cluster_id"):
    """Relative pixel 3 QE of each aggregate: R / (R + G + B) of its summed A_c x photoelectrons
    (the value ``red_share_pixelated`` falls back to). Series indexed by aggregate id."""
    f = df[["A_R", "A_G", "A_B"]].to_numpy(dtype=np.float64)
    pe = df["photoelectrons"].to_numpy(dtype=np.float64)
    valid = np.isfinite(f).all(axis=1) & (f.sum(axis=1) > 0) & np.isfinite(pe) & (pe > 0)
    s = pd.DataFrame({"agg": df[aggregate_id_column].to_numpy(),
                      "wR": np.where(valid, pe * f[:, 0], 0.0),
                      "wtot": np.where(valid, pe * f.sum(axis=1), 0.0)}).groupby("agg")[["wR", "wtot"]].sum()
    return (s["wR"] / s["wtot"]).rename("red_share")


def slot_crop(image, x_px, y_px, oversampling, slot_w_pt, slot_h_pt, pad=1.25, centroid=False):
    """Crop of a render around some localisations with exactly a figure slot's aspect ratio.

    Centred on the localisations' bounding box (or, with ``centroid``, on their mean position), scaled so
    the box (made symmetric about that centre) times ``pad`` just fits the slot; parts outside the
    render are black.

    Args:
        image: Render (rows, cols[, 3]) with ``oversampling`` render px per camera px.
        x_px, y_px: The localisations' camera-px coordinates (``xc``, ``yc``).
        slot_w_pt, slot_h_pt: The slot's size in the figure (pt).

    Returns:
        The crop, (rows, cols[, 3]); draw it with ``sized_image(height_pt=slot_h_pt)``.
    """
    x0, x1 = np.min(x_px) * oversampling, np.max(x_px) * oversampling
    y0, y1 = np.min(y_px) * oversampling, np.max(y_px) * oversampling
    if centroid:
        cx, cy = np.mean(x_px) * oversampling, np.mean(y_px) * oversampling
        hx, hy = max(cx - x0, x1 - cx), max(cy - y0, y1 - cy)
        x0, x1, y0, y1 = cx - hx, cx + hx, cy - hy, cy + hy
    px_per_pt = pad * max((x1 - x0) / slot_w_pt, (y1 - y0) / slot_h_pt, 1e-9)
    w, h = int(round(slot_w_pt * px_per_pt)), int(round(slot_h_pt * px_per_pt))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    c0, r0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    out = np.zeros((h, w) + image.shape[2:], dtype=image.dtype)
    rs, cs = max(0, r0), max(0, c0)
    re_, ce = min(image.shape[0], r0 + h), min(image.shape[1], c0 + w)
    if re_ > rs and ce > cs:
        out[rs - r0:re_ - r0, cs - c0:ce - c0] = image[rs:re_, cs:ce]
    return out


def rebrighten(colour, grey, lo_pct=1.0, hi_pct=99.5, occupied_only=False):
    """``colour`` with its brightness set again from the density ``grey``, clipped to its
    ``lo_pct``-``hi_pct`` percentiles (over all pixels, or only those with any density)."""
    from matplotlib import colors as mcolors

    vals = grey[grey > 0] if occupied_only else grey
    lo, hi = np.percentile(vals, lo_pct), np.percentile(vals, hi_pct)
    hsv = mcolors.rgb_to_hsv(np.clip(colour, 0, 1))
    hsv[..., 2] = np.clip((grey - lo) / (hi - lo), 0, 1)
    return mcolors.hsv_to_rgb(hsv)
