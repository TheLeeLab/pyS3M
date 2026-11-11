"""Dye-pair mixtures: data, settings and analysis for the Figure 3 a-c notebook.

Each mixture on BioStudies (`Ximea/DyeMixtures/`) is two dyes on one slide, a 4 x 4 grid of
fields of view: one TIFF plus one ``*_metadata.txt`` per position. Each is downloaded, fitted,
drift-corrected (AIM) per position and clustered into molecules by ``_multifov``. The settings
come from the analysis of the published panels (B. Clarke, ``dye_discrimination/
Dye_Mixture_Analysis_*``), except the quality cuts, which are the current ones (chi^2 < 3, colour
error < 1, no cut on the background colour errors).

The misidentification rate follows the same analytical approach:
1. the two dyes' mean colours (A_R, A_G) are a 2-component GMM fit to the brightest molecules,
   selected by a per-pair colour box (``REFERENCE_CUTS``);
2. each molecule's colour is re-estimated as its frames accumulate (inverse-variance weighted);
3. in each bin of accumulated photoelectrons, the covariances are fitted with the means fixed
   (Tukey M-estimator), and the misidentification rate is the overlap of the two Gaussians.
"""
import numpy as np
import pandas as pd

import _multifov as mf

PAIR_ROOT = "2026_Multicolour_Paper/Ximea/DyeMixtures"

# Pair -> (acquisition folder, PSF peak wavelength in um, the two dyes in order of increasing A_R).
# ATTO 647N + QD800: the deposition has the same slide twice. `10mW638_DM405488561638_NF_SP_1`
# (no emission filter; filed at the top of DyeMixtures/) is the one the published panel's notebook
# read; `ATTO647N_QDot800/10mW638_DM405488561638_635lp_1` is with a 635 nm long-pass.
PAIRS = {
    "ATTO520_Rho6G": (f"{PAIR_ROOT}/ATTO520_ATTORho6G/20perc514_DM515_515LP_1", 0.55,
                      ("ATTO 520", "Rho6G")),
    "ATTO565_ATTO594": (f"{PAIR_ROOT}/ATTO565_ATTO594/2p5pM_10perc561_568LP_1", 0.60,
                        ("ATTO 565", "ATTO 594")),
    "ATTO565_ATTO633": (f"{PAIR_ROOT}/ATTO565_ATTO633/5pM_dyes_10p_561_20mW638_NF_SP_1", 0.62,
                        ("ATTO 565", "ATTO 633")),
    "ATTO647N_QD800": (f"{PAIR_ROOT}/10mW638_DM405488561638_NF_SP_1", 0.70,
                       ("ATTO 647N", "QD800")),
}

# Spot detection and fitting, as for the single dyes;
# the PSF wavelength is set per pair (PAIRS).
FIT_PARAMS = dict(pfa=1e-3, ROI_size=12, NA=1.49, pixel_size=0.069)
DRIFT_PARAMS = dict(segmentation=5, intersect_d=0.5, roi_r=2.0)

# Molecules: DBSCAN clusters, as in the published panels' analysis (min. cluster size per pair).
# Quality cuts: chi^2 < 3; colour errors (A_R, A_G, A_B) < 1 (the fitter's errors are realistic,
# so the library's 0.15 default would drop well-measured dim molecules); no cut on the background
# colour errors (bg_*_err, the library default): they say nothing about the spot, and with one
# background channel near zero they are huge (median 2-21 here); a cut at 1 removed 27-99.5 %
# of the localisations (all but 0.5 % for ATTO 565 + ATTO 594);
# > 50 detected photoelectrons (Methods; `photons` = A_B + A_G + A_R overcounts them 2-4x);
# PSF sigma 75-250 nm.
_QC = dict(clustering_method="DBSCAN", chi_val=3, max_colour_error=1.0,
           min_sigma=75 / mf.PIXEL_SIZE_NM, max_sigma=250 / mf.PIXEL_SIZE_NM,
           min_photoelectrons=50)
CLUSTER_PARAMS = {
    "ATTO520_Rho6G":   dict(_QC, min_cluster_size=2, max_localisation_error=1.0),
    "ATTO565_ATTO594": dict(_QC, min_cluster_size=5, max_localisation_error=0.5),
    "ATTO565_ATTO633": dict(_QC, min_cluster_size=2, max_localisation_error=1.0),
    "ATTO647N_QD800":  dict(_QC, min_cluster_size=5, max_localisation_error=1.0),
}

# Molecules that define the two mean colours: bright and inside a colour box that excludes
# outliers. (min photoelectrons, min A_R, min A_G, max A_G)
# The published cuts were 5,000 / 2,000 / 20,000 / 25,000 `photons` (A_B + A_G + A_R). On the Bayer
# mosaic photoelectrons = photons x 0.25 (1 + A_G), so each is converted with the mean A_G of the
# pair's two reference colours (0.72 / 0.29 / 0.33 / 0.27: x 0.43 / 0.32 / 0.33 / 0.32), to 2 s.f.
REFERENCE_CUTS = {
    "ATTO520_Rho6G":   (2200, 0.175, 0.64, 1.0),
    "ATTO565_ATTO594": (640, 0.55, 0.15, 1.0),
    "ATTO565_ATTO633": (6600, 0.40, 0.15, 1.0),
    "ATTO647N_QD800":  (7900, 0.40, 0.10, 1.0),
}

# Bins of accumulated photoelectrons for the misidentification rate (the same for every pair):
# 100 to 31,600, the paper's range (it showed up to ~25,000 `photons`, ~8,000 photoelectrons).
# Beyond ~30,000 photoelectrons the bins hold only molecules lasting hundreds of frames, whose
# colours spread a little more, and the rate creeps back up from its floor.
PE_BINS = np.logspace(2, 4.5, 21)

PIXEL_SIZE_NM = mf.PIXEL_SIZE_NM


def prepare_acquisition(pair, dest_dir, fast_mode, fitter, calib, smoothing_function,
                        drift_corrector, io, refit=False):
    """Download, fit and drift-correct one pair's acquisition; return its ``*_undrifted_locs.h5``."""
    folder, peak_wavelength, _ = PAIRS[pair]
    return mf.prepare_acquisition(folder, dest_dir, fast_mode, fitter, calib, smoothing_function, io,
                                  dict(FIT_PARAMS, peak_wavelength=peak_wavelength),
                                  drift_corrector=drift_corrector, drift_params=DRIFT_PARAMS,
                                  refit=refit)


def cluster_sample(pair, sm_extractor, files, out_dir, recluster=False):
    """Cluster one pair into molecules (``_multifov.cluster_sample`` with the pair's settings)."""
    return mf.cluster_sample(sm_extractor, files, out_dir, pair, CLUSTER_PARAMS[pair],
                             recluster=recluster)


def reference_molecules(pair, sm_db):
    """The bright, in-box molecules that define the pair's two mean colours."""
    min_pe, min_ar, min_ag, max_ag = REFERENCE_CUTS[pair]
    keep = ((sm_db["photoelectrons"] > min_pe) & (sm_db["A_R"] > min_ar)
            & (sm_db["A_G"] > min_ag) & (sm_db["A_G"] < max_ag))
    return sm_db[keep]


def photoelectron_accumulation(sf_db, io):
    """Each molecule's colour as its frames accumulate, against its accumulated photoelectrons.

    One row per (molecule, frame), frames in time order: ``photoelectrons_accumulated`` is the
    running sum, and ``A_R``/``A_G``/``A_B`` the running inverse-variance weighted means with
    their errors (as ``build_photon_accumulation_database``). Databases with the old ``photons``
    column are upgraded first.
    """
    sf_db = io.add_photoelectron_columns(sf_db)
    df = sf_db.sort_values(["molecular_index", "frame"]).reset_index(drop=True)
    g = df.groupby("molecular_index", sort=False)
    out = pd.DataFrame({"molecular_index": df["molecular_index"],
                        "frames_accumulated": g.cumcount() + 1,
                        "photoelectrons_accumulated": g["photoelectrons"].cumsum()})
    for c in ("A_R", "A_G", "A_B"):
        w = 1.0 / (df[f"{c}_err"] ** 2 + 1e-12)
        wsum = (w).groupby(df["molecular_index"], sort=False).cumsum()
        out[c] = (w * df[c]).groupby(df["molecular_index"], sort=False).cumsum() / wsum
        out[f"{c}_err"] = 1.0 / np.sqrt(wsum)
    return out


# ---------------------------------------------------------------------------------------------
# Figure 3c: misidentification rate against photoelectrons, drawn as in the paper's panel:
# axes 63 x 40 mm; markers s = 10 pt^2 (black circle, grey diamond, black down-triangle, open
# up-triangle); 7 pt ticks, 8 pt axis labels, 6 pt legend; green dashed line at 5 %; dashed grid.
PANEL_STYLE = {   # legend order as in the paper
    "ATTO565_ATTO633": dict(marker="o", facecolor="black", edgecolor="none", label="ATTO 565/633"),
    "ATTO565_ATTO594": dict(marker="D", facecolor="#808080", edgecolor="none", label="ATTO 565/594"),
    "ATTO520_Rho6G":   dict(marker="v", facecolor="black", edgecolor="none", label="ATTO 520/Rho6G"),
    "ATTO647N_QD800":  dict(marker="^", facecolor="white", edgecolor="black", linewidths=0.5,
                            label="ATTO 647N/Qdot 800"),
}
PANEL_AXES_MM = (63.0, 40.0)
PANEL_XLIM = (100, 3e4)      # photoelectrons (the paper's 390-25,000 `photons` is ~130-8,000 pe)
PANEL_YLIM = (0.01, 100)     # %; the Monte Carlo overlap resolves ~0.001 % (1e5 samples)


def error_rate_points(summary, min_molecules=50):
    """Bin centres (geometric) and % misidentification, for bins with enough molecules and a
    non-zero rate (zero cannot be drawn on the log axis)."""
    s = summary[(summary["n_molecules"] >= min_molecules) & (summary["overall_error_rate"] > 0)]
    return np.sqrt(s["photoelectron_bin_min"] * s["photoelectron_bin_max"]).to_numpy(), \
        100 * s["overall_error_rate"].to_numpy()


def error_rate_panel(summaries, pairs=None, min_molecules=50):
    """The Figure 3c panel for ``pairs`` (default: all, in the paper's legend order)."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    pairs = [p for p in PANEL_STYLE if p in (pairs or summaries)]
    mm = 1 / 25.4
    left, bottom, right, top = 10.0, 8.5, 5.0, 1.5            # margins, mm
    w, h = PANEL_AXES_MM
    fig = plt.figure(figsize=((left + w + right) * mm, (bottom + h + top) * mm))
    W, H = left + w + right, bottom + h + top
    ax = fig.add_axes([left / W, bottom / H, w / W, h / H])

    for pair in pairs:
        x, y = error_rate_points(summaries[pair], min_molecules)
        ax.scatter(x, y, s=10, zorder=3, **PANEL_STYLE[pair])
    ax.axhline(5, color="green", ls="--", lw=0.75, zorder=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(PANEL_XLIM)
    ax.set_ylim(PANEL_YLIM)
    comma = FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}")
    ax.xaxis.set_major_formatter(comma)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=20))
        axis.set_minor_formatter(NullFormatter())
    ax.tick_params(which="major", labelsize=7, width=1, length=3.5)
    ax.tick_params(which="minor", width=0.6, length=2)
    for spine in ax.spines.values():
        spine.set_linewidth(1)
    ax.grid(which="major", color="#808080", ls="--", lw=0.5, alpha=0.3)
    ax.grid(which="minor", color="#b0b0b0", ls="--", lw=0.5, alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_xlabel(r"$N_{\mathrm{photoelectrons}}$", fontsize=8, labelpad=1)
    ax.set_ylabel("% dye misidentification", fontsize=8, labelpad=1)
    ax.legend(loc="upper right", fontsize=6, framealpha=0.8, edgecolor="#cccccc",
              handletextpad=0.3, borderpad=0.4, labelspacing=0.3)
    return fig, ax


# ---------------------------------------------------------------------------------------------
# Figure 3b: one frame per pair with a molecule of each dye side by side, circled and named in the
# dye's colour. Each image is 23.3 mm square (its size in the paper), in photoelectrons with a small
# colour bar inside it (as Fig. 2b), a 300 nm scale bar, and the 2 x 2 Bayer key of the pixels in
# its top-left corner, read from the sensor masks at the crop's position.
DYE_COLOURS = {"ATTO 520": "#0000ff", "Rho6G": "#00ffff", "ATTO 565": "#00ff00", "ATTO 594": "#ffa500",
               "ATTO 633": "#ff0000", "ATTO 647N": "#ffc0cb", "QD800": "#ff0000"}
DYE_LABELS = {"Rho6G": "ATTO Rho6G", "QD800": "Qdot 800"}   # names as printed in the paper
EXAMPLE_CROP_PX = 20
EXAMPLE_MM = 23.3
SCALEBAR_NM = 300   # as the Fig. 2b zoom-ins


def assign_dyes(sm_db, means, covariances):
    """Each molecule's dye (0 or 1, the order of ``means``) and its posterior probability, from the
    two Gaussians of the pair's colours in (A_R, A_G) with equal weights."""
    from scipy.stats import multivariate_normal
    X = sm_db[["A_R", "A_G"]].to_numpy()
    like = np.column_stack([multivariate_normal(m, c).pdf(X) for m, c in zip(means, covariances)])
    post = like / np.clip(like.sum(axis=1, keepdims=True), 1e-300, None)
    return post.argmax(axis=1), post.max(axis=1)


def find_example(sf_db, sm_db, labels, confidence, frame_shape, min_confidence=0.99,
                 min_sep_px=5, max_sep_px=9, crop_px=EXAMPLE_CROP_PX):
    """The frame showing one confidently assigned molecule of each dye side by side.

    Candidates: pairs of localisations in the same position and frame, one of each dye (each
    molecule's assignment ``>= min_confidence``), ``min_sep_px``-``max_sep_px`` apart, whose
    ``crop_px`` square crop centred between them lies inside the frame and holds no other
    localisation. The one with the most photoelectrons in its dimmer molecule is returned.

    Args:
        sf_db: Localisations (``molecular_index``, ``fov_index``, ``frame``, ``xc``, ``yc``,
            ``photoelectrons``).
        sm_db: Molecules (``molecular_index``), in the order of ``labels`` / ``confidence``.
        frame_shape: (rows, cols) of the camera ROI.

    Returns:
        dict: ``fov_index``, ``frame``, ``row0``, ``col0`` (the crop's top-left pixel in the ROI) and
        ``mols``, a list of two dicts (``dye`` index, ``xc``, ``yc``, ``photoelectrons``), or None.
    """
    lab = pd.DataFrame({"molecular_index": sm_db["molecular_index"].to_numpy(), "dye": labels,
                        "conf": confidence})
    loc = sf_db.merge(lab, on="molecular_index", how="left")
    best, best_score = None, -np.inf
    half = crop_px / 2
    for (fov, frame), g in loc.groupby(["fov_index", "frame"], sort=False):
        ok = g[g["conf"] >= min_confidence]
        a, b = ok[ok["dye"] == 0], ok[ok["dye"] == 1]
        if a.empty or b.empty:
            continue
        for _, p in a.iterrows():
            d = np.hypot(b["xc"] - p["xc"], b["yc"] - p["yc"])
            for _, q in b[(d >= min_sep_px) & (d <= max_sep_px)].iterrows():
                score = min(p["photoelectrons"], q["photoelectrons"])
                if score <= best_score:
                    continue
                cx, cy = (p["xc"] + q["xc"]) / 2, (p["yc"] + q["yc"]) / 2
                col0, row0 = int(round(cx - half)), int(round(cy - half))
                if row0 < 0 or col0 < 0 or row0 + crop_px > frame_shape[0] or col0 + crop_px > frame_shape[1]:
                    continue
                inside = g["xc"].between(col0 - 1.5, col0 + crop_px + 0.5) & g["yc"].between(row0 - 1.5, row0 + crop_px + 0.5)
                if inside.sum() > 2:
                    continue
                best_score = score
                best = dict(fov_index=int(fov), frame=int(frame), row0=row0, col0=col0,
                            mols=[dict(dye=int(m["dye"]), xc=float(m["xc"]), yc=float(m["yc"]),
                                       photoelectrons=float(m["photoelectrons"])) for m in (p, q)])
    return best


def frame_crop(tif, frame, row0, col0, crop_px, gain, offset, rqe):
    """One frame's ``crop_px`` square crop (top-left at ``row0``, ``col0`` of the ROI) in
    photoelectrons; ``gain``, ``offset``, ``rqe`` are the calibration maps cropped to the ROI."""
    import tifffile
    flags = dict(is_ome=False, is_mmstack=False, is_imagej=False)   # see _single_dyes.open_stack
    try:
        stack = tifffile.memmap(tif, mode="r", **flags)
    except ValueError:
        stack = tifffile.imread(tif, **flags)
    sl = np.s_[row0:row0 + crop_px, col0:col0 + crop_px]
    raw = np.asarray(stack[frame][sl], dtype=np.float32)
    return (raw - offset[sl]) / gain[sl] / rqe[sl]


def example_panel(crop, mols, dyes, masks, sensor_origin, pe_range=None, pixel_nm=PIXEL_SIZE_NM,
                  plotter=None):
    """Figure 3b: one pair's crop with each molecule circled and named in its dye's colour.

    Args:
        crop: Image (photoelectrons), ``EXAMPLE_CROP_PX`` square.
        mols: The two molecules (``dye`` index into ``dyes``, ``xc``, ``yc`` in crop pixels).
        dyes: The pair's two dye names (``PAIRS[pair][2]``).
        masks: The calibration's full-sensor colour masks.
        sensor_origin: Sensor (row, col) of the crop's top-left pixel, for the Bayer key.
        pe_range: Display range (photoelectrons); default the crop's 1st-99.8th percentiles.

    Returns:
        (fig, ax).
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    from pyS3M.PlottingBase import add_bayer_key

    mm = 1 / 25.4
    fig = plt.figure(figsize=(EXAMPLE_MM * mm, EXAMPLE_MM * mm))
    fig.set_layout_engine("none")
    ax = fig.add_axes([0, 0, 1, 1])
    vmin, vmax = pe_range if pe_range is not None else np.percentile(crop, [1, 99.8])
    im = ax.imshow(crop, cmap="gray", vmin=vmin, vmax=vmax, origin="upper", interpolation="nearest")
    ax.set_axis_off()
    n = crop.shape[0]
    # regions kept clear of labels, (x0, y0, x1, y1) in crop px: colour bar (top right), Bayer key
    # (top left) and the scale bar, in whichever bottom corner leaves the labels more room
    bar_px = SCALEBAR_NM / pixel_nm
    fixed = [(0.48 * n, -0.5, n, 0.22 * n), (-0.5, -0.5, 2, 2)]
    corners = {"lower right": (n - 1.5 - bar_px, n - 3, n, n), "lower left": (-0.5, n - 3, bar_px + 1, n)}
    labels = [DYE_LABELS.get(dyes[m["dye"]], dyes[m["dye"]]) for m in mols]
    placed = {loc: _place_labels(mols, labels, n, fixed + [rect]) for loc, rect in corners.items()}
    bar_loc = min(placed, key=lambda loc: placed[loc][0] + (0 if loc == "lower right" else 0.5))
    for m, label, (x, y) in zip(mols, labels, placed[bar_loc][1]):
        colour = DYE_COLOURS[dyes[m["dye"]]]
        ax.add_patch(Circle((m["xc"], m["yc"]), 2.6, fill=False, edgecolor=colour, lw=1.0))
        ax.text(x, y, label, color=colour, fontsize=6, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", facecolor=(0.3, 0.3, 0.3, 0.6), edgecolor="none"))
    ax.add_artist(AnchoredSizeBar(ax.transData, bar_px, "", loc=bar_loc, pad=0.25,
                                  borderpad=0.2, color="white", frameon=False, size_vertical=0.35))
    # photoelectron colour bar inside the image, top right (ticks below, as Fig. 2b)
    cax = ax.inset_axes([0.5, 0.9, 0.36, 0.035])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    lo, hi = np.ceil(vmin / 10) * 10, np.floor(vmax / 10) * 10
    cbar.set_ticks([lo, hi])
    cbar.ax.tick_params(labelsize=5, colors="white", length=1.2, width=0.4, pad=0.8)
    cbar.outline.set_edgecolor("white")
    cbar.outline.set_linewidth(0.4)
    cbar.ax.text(1.05, 0.5, r"pe$^{-}$", transform=cbar.ax.transAxes, color="white", fontsize=5,
                 ha="left", va="center")
    add_bayer_key(ax, masks, row_min=sensor_origin[0], col_min=sensor_origin[1])
    return fig, ax


def _place_labels(mols, labels, n, keep_clear, radius=2.6, char_px=1.15, height_px=2.8):
    """Centres (x, y) in crop px for each molecule's label: above, below, left or right of its
    circle, choosing the combination whose label boxes overlap least (returns (cost, centres)) with the circles, each other,
    the ``keep_clear`` rectangles and the image edge."""
    import itertools

    def box(m, label, side):
        w, h = char_px * len(label) + 0.6, height_px
        gap = radius + 0.5
        x, y = {"above": (m["xc"], m["yc"] - gap - h / 2), "below": (m["xc"], m["yc"] + gap + h / 2),
                "left": (m["xc"] - gap - w / 2, m["yc"]), "right": (m["xc"] + gap + w / 2, m["yc"])}[side]
        if side in ("above", "below"):  # slide along x to stay in the image
            x = min(max(x, w / 2 - 0.5), n - 0.5 - w / 2)
        return x, y, (x - w / 2, y - h / 2, x + w / 2, y + h / 2)

    def overlap(a, b):
        return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))

    def outside(r):
        inner = (-0.5, -0.5, n - 0.5, n - 0.5)
        return (r[2] - r[0]) * (r[3] - r[1]) - overlap(r, inner)

    circles = [(m["xc"] - radius, m["yc"] - radius, m["xc"] + radius, m["yc"] + radius) for m in mols]
    best = None
    for sides in itertools.product(["above", "below", "left", "right"], repeat=len(mols)):
        placed = [box(m, lab, s) for m, lab, s in zip(mols, labels, sides)]
        rects = [p[2] for p in placed]
        cost = sum(10 * outside(r) + sum(overlap(r, c) for c in circles)
                   + 5 * sum(overlap(r, k) for k in keep_clear) for r in rects)
        cost += sum(5 * overlap(a, b) for a, b in itertools.combinations(rects, 2))
        cost += 0.01 * sum(s in ("left", "right") for s in sides)  # prefer above/below on ties
        if best is None or cost < best[0]:
            best = (cost, [(p[0], p[1]) for p in placed])
    return best
