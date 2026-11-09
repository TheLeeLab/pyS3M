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
                cbar=None, margin_pt=1.5):
    """A rendered RGB image drawn at an exact size (pt), as placed in the paper's figure.

    The image (origin lower) is ``height_pt`` tall and as wide as its aspect ratio makes it, with a
    white scale bar ``margin_pt`` from its lower-right corner, the label centred above it. ``cbar``
    (optional) adds a colour bar beside it: dict(c_min, c_max, label, width_pt, gap_pt, tick_fontsize,
    label_fontsize=8, cmap="jet"). Save with ``bbox_inches="tight"``: the sizes are unchanged.

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
