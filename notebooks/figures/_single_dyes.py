"""Single-dye data, settings and photobleaching screening for the Figure 2 notebook.

Twelve dyes on BioStudies (`Ximea/Single_Dye_Experiments/<dye>/`), each a 4 x 4 grid of fields of
view (one TIFF + one ``*_metadata.txt`` per position). Each is fitted per position and clustered
into molecules by ``_multifov`` (no drift correction: the molecules bleach within seconds), with
the settings below, following the analysis of the published figures (B. Clarke, J. Beckwith).
"""
from collections import defaultdict

import numpy as np
import tifffile

import _multifov as mf

DYE_ROOT = "2026_Multicolour_Paper/Ximea/Single_Dye_Experiments"

# Filters in the emission path of each acquisition (Semrock part numbers in the spectra database).
_QUAD_DICHROIC = "semrock-di03-r405-488-561-635-t1-25x36"
_NOTCH = "semrock-nf03-405-488-561-635e"
_SHORTPASS_785 = "semrock-bsp01-785r"
_DICHROIC_514 = "semrock-di03-r514-t1-25x36"
_LONGPASS_515 = "semrock-ff01-515-lp"
_RED_PATH = [_QUAD_DICHROIC, _NOTCH, _SHORTPASS_785]

# label: (spectrum name, deposition folder, emission filters, colour on the Fig. 2e ternary or None)
DYES = {
    "ATTO488":   ("ATTO 488",   "ATTO488/40mW488_488LP_BP520-44_1",
                  [_QUAD_DICHROIC, "semrock-blp01-488r", "semrock-ff01-520-44"], "blueviolet"),
    "ATTO514":   ("ATTO 514",   "ATTO514/30p_515_LP515_BP540_80_1",
                  [_DICHROIC_514, _LONGPASS_515, "semrock-ff01-540-80"], None),
    "ATTO520":   ("ATTO 520",   "ATTO520/30p_515_LP515_BP540_80_1",
                  [_DICHROIC_514, _LONGPASS_515, "semrock-ff01-540-80"], "blue"),
    "ATTORho6G": ("ATTO Rho6G", "ATTORho6G/30p_515_LP515_1",
                  [_DICHROIC_514, _LONGPASS_515], "cyan"),
    "ATTO565":   ("ATTO 565",   "ATTO565/20perc_561_LP561_BP582-64_1",
                  [_QUAD_DICHROIC, "semrock-blp01-561r", "semrock-ff01-582-64"], "lime"),
    "ATTO594":   ("ATTO 594",   "ATTO594/40p_NF_785SP_1_488_561dichro",
                  ["semrock-di03-r488-561-t1-25x36", _NOTCH, _SHORTPASS_785], "yellow"),
    "ATTO620":   ("ATTO 620",   "ATTO620/40per_638_1_635LP_1",
                  [_QUAD_DICHROIC, "semrock-blp01-635r"], None),
    "ATTO633":   ("ATTO 633",   "ATTO633/40both638_NF_785SP_1", _RED_PATH, "red"),
    "ATTO647N":  ("ATTO 647N",  "ATTO647N/40both638_NF_785SP_1", _RED_PATH, None),
    "ATTO655":   ("ATTO 655",   "ATTO655/40both638_NF_785SP_1", _RED_PATH, "magenta"),
    "ATTO700":   ("ATTO 700",   "ATTO700/40mW638_both_NF_785sp_1", _RED_PATH, "whitesmoke"),
    "LD655":     ("LD 655",     "LD655/100nMLD655_PLL_40mW638_TIRF_bothlasers_405488561638di_638LP_1",
                  [_QUAD_DICHROIC, "semrock-blp01-635r"], None),
}

# Spot detection and fitting: the paper's 12 px ROI; the PSF model's wavelength is each dye's
# emission peak (set per dye in the notebook).
FIT_PARAMS = dict(pfa=1e-3, ROI_size=12, NA=1.49, pixel_size=0.069)
# Molecules = HDBSCAN clusters of >= 10 localisations (paper's Methods).
# Quality cuts: > 50 detected photoelectrons (Methods; `photons` = A_B + A_G + A_R overcounts them 2-4x);
# chi^2 < 3; colour errors < 1 (100 %: the fitter's errors are realistic, so the library's 0.15
# default would drop well-measured dim molecules); no cut on the background colour errors (the
# library default: they say nothing about the spot and are often > 1); localisation error < 1 px;
# PSF sigma 75-250 nm (the library default).
CLUSTER_PARAMS = dict(clustering_method="HDBSCAN", min_cluster_size=10, chi_val=3,
                      max_localisation_error=1.0, max_colour_error=1.0,
                      min_photoelectrons=50)

# Single-step photobleaching screen (SI_Single_Dye_Photobleaching).
ROI_SIZE = 12        # px; square trace ROI centred on the molecule
MIN_FRAMES = 25      # minimum detected frames for a candidate
POST_PAD = 50        # frames after the last detection included in the trace
STEP_WIN_SIZE = 5    # StepDetector minimum segment length
STEP_ALPHA = 0.1     # StepDetector false-positive rate per segment
SEED = 42


def emission_peak_um(spectral, dye_spectrum_name, wavelength):
    """Wavelength (um) of the dye's emission maximum."""
    spectrum = np.ravel(spectral.get_dye_or_filter_data(dye_spectrum_name, wavelength=wavelength))
    return float(wavelength[np.argmax(spectrum)]) / 1000


def predicted_fingerprint(spectral, dye, wavelength, pixel_QYs):
    """Predicted relative pixel QE (B, G, R) of a dye: its emission through the filters and
    objective, integrated against each pixel's QE, normalised to sum to 1."""
    spectrum_name, _, filters, _ = DYES[dye]
    transmission = np.prod(spectral.get_dye_or_filter_data(names=filters, wavelength=wavelength,
                                                           dye_or_filter=False), axis=0)
    transmission = transmission * spectral.getobjectiveefficiency(wavelength)
    detected = np.ravel(spectral.get_dye_or_filter_data(spectrum_name, wavelength=wavelength)) * transmission
    efficiency = pixel_QYs @ (detected / detected.sum())
    return efficiency / efficiency.sum()


def open_stack(tif):
    """This position's TIFF stack (frames, rows, cols): memory-mapped if possible, else read.

    The OME / MicroManager metadata are ignored (as in ``IOFunctions.read_tiff``): each
    position's OME-XML describes the whole grid, so tifffile would otherwise read every
    position's file as one (positions, frames, rows, cols) series.
    """
    flags = dict(is_ome=False, is_mmstack=False, is_imagej=False)
    try:
        return tifffile.memmap(tif, mode="r", **flags)
    except ValueError:
        return tifffile.imread(tif, **flags)


class BleachScreen:
    """Find molecules that photobleach in a single step, from raw photoelectron ROI traces.

    For each molecule seen in at least ``MIN_FRAMES`` frames, the 12 x 12 px ROI around it is
    summed (in photoelectrons) from its first detection to ``POST_PAD`` frames after its last;
    a likelihood-ratio change-point detector (``StepDetector``) must find exactly one, downward,
    step. Candidates are ranked by on/off contrast (baseline-subtracted mean before / after).
    """

    def __init__(self, step_detector, io, helper, calib):
        self.detector = step_detector
        self.io, self.helper, self.calib = io, helper, calib

    def _maps_for(self, tif):
        x0, y0, w, h = self.io.metadata_reader_imageJ(mf.metadata_for(tif))
        maps = self.helper.crop_calibration_maps(
            {k: self.calib[k] for k in ("gain_map", "offset_map", "rqe")}, x0, y0, w, h)
        return maps["gain_map"], maps["offset_map"], maps["rqe"]

    def to_pe(self, raw, gain, offset, rqe, y0, y1, x0, x1):
        """Raw ADU (frames or one frame) of an ROI -> photoelectrons."""
        return (np.asarray(raw, dtype=np.float32) - offset[y0:y1, x0:x1]) / gain[y0:y1, x0:x1] / rqe[y0:y1, x0:x1]

    def screen(self, sf_db, files):
        """Screen every candidate in every position.

        Args:
            sf_db: Single-frame database (``molecular_index``, ``fov_index``, ``frame``, ``xc``, ``yc``).
            files: Per-position localisation files, in ``fov_index`` order.

        Returns:
            list[dict]: Single-step bleachers (fov_index, xc, yc, f0, cp, trace, cps, snr_ratio),
            best contrast first.
        """
        mol = (sf_db.groupby("molecular_index")
               .agg(n_frames=("frame", "count"), xc=("xc", "mean"), yc=("yc", "mean"),
                    frame_min=("frame", "min"), frame_max=("frame", "max"),
                    fov_index=("fov_index", "first"))
               .reset_index())
        candidates = mol[mol["n_frames"] >= MIN_FRAMES].sample(frac=1, random_state=SEED)
        by_fov = defaultdict(list)
        for row in candidates.itertuples():
            by_fov[int(row.fov_index)].append(row)

        half = ROI_SIZE // 2
        found = []
        for fov, rows in sorted(by_fov.items()):
            tif = mf.tif_for(files[fov])
            stack = open_stack(tif)
            gain, offset, rqe = self._maps_for(tif)
            n_frames, height, width = stack.shape
            for row in rows:
                f0, f1 = int(row.frame_min), min(n_frames, int(row.frame_max) + POST_PAD + 1)
                xpx, ypx = int(round(row.xc)), int(round(row.yc))
                x0, x1 = max(0, xpx - half), min(width, xpx + half)
                y0, y1 = max(0, ypx - half), min(height, ypx + half)
                trace = self.to_pe(stack[f0:f1, y0:y1, x0:x1], gain, offset, rqe, y0, y1, x0, x1).sum(axis=(1, 2))
                cps = self.detector.detect(trace)
                if len(cps) != 2:   # exactly one internal change point (+ the terminal index)
                    continue
                cp = cps[0]
                if trace[cp:].mean() >= trace[:cp].mean():   # must be a downward (bleaching) step
                    continue
                base = trace - trace.min()
                found.append(dict(fov_index=fov, xc=row.xc, yc=row.yc, f0=f0, cp=cp, trace=trace,
                                  cps=cps, snr_ratio=base[:cp].mean() / max(base[cp:].mean(), 1e-6)))
            del stack
        found.sort(key=lambda c: c["snr_ratio"], reverse=True)
        return found


def pick_spread(candidates, n, min_sep_px=10):
    """Greedily take the n best candidates that are at least ``min_sep_px`` apart."""
    chosen = []
    for c in candidates:
        if all(np.hypot(c["xc"] - s["xc"], c["yc"] - s["yc"]) >= min_sep_px for s in chosen):
            chosen.append(c)
        if len(chosen) == n:
            break
    return chosen


# --- Figure 2e: the ternary plots, styled as in the paper ---------------------------------------
# Axis colours (pixel 1 / 2 / 3) and sizes measured from the published figure: the full triangle's
# side is 59.7 mm, the red-dye zoom's 39.6 mm; dashed grid on a black triangle; the legend (dye
# names in their colours) on a grey box in the full plot; the zoomed region outlined in white.
PIXEL_AXIS_COLOURS = {"pixel 1": "#5858df", "pixel 2": "#53b03b", "pixel 3": "#ff2a2a"}
TERNARY_FULL = dict(limits=(0, 1, 0, 1, 0, 1), side_mm=59.7, step=0.2, size=8, alpha=0.08)
TERNARY_ZOOM = dict(limits=(0.6, 0.8, 0.2, 0.4, 0.0, 0.2), side_mm=39.6, step=0.1, size=15, alpha=0.15,
                    labels=False)


def ternary_panel(shown, limits, side_mm, step, size, alpha, labels=True, legend=False, outline=None):
    """One Fig. 2e ternary plot: top corner = pixel 3 (``A_R``), left = pixel 2 (``A_G``),
    right = pixel 1 (``A_B``), each axis in its pixel's colour.

    Args:
        shown: ``{dye: molecules}``, each with ``A_R``, ``A_G``, ``A_B``; drawn in ``DYES`` colours.
        limits: ``(tmin, tmax, lmin, lmax, rmin, rmax)``.
        side_mm: Triangle side length (mm).
        step: Tick and grid spacing.
        size, alpha: Marker size (pt^2) and opacity.
        labels: Name the axes (the paper's zoom has tick labels only).
        legend: Draw the dye legend (dye names in their colours, two columns).
        outline: Limits of a region to outline in white (the zoom), or None.

    Returns:
        (fig, ax).
    """
    import matplotlib.pyplot as plt
    import mpltern  # noqa: F401  (registers the "ternary" projection)

    mm = 1 / 25.4
    margin = (13, 12, 13, 6)   # left, bottom, right, top (mm): room for the ticks and labels
    w, h = side_mm, side_mm * np.sqrt(3) / 2
    fig = plt.figure(figsize=((w + margin[0] + margin[2]) * mm, (h + margin[1] + margin[3]) * mm))
    W, H = fig.get_size_inches() / mm
    ax = fig.add_axes([margin[0] / W, margin[1] / H, w / W, h / H], projection="ternary", ternary_sum=1)
    ax.set_facecolor("black")
    for dye, db in shown.items():
        ax.scatter(db["A_R"], db["A_G"], db["A_B"], color=DYES[dye][3], marker="o", s=size,
                   alpha=alpha, linewidths=0, rasterized=True)
    ax.set_ternary_lim(*limits)
    for axis, lo, hi in [(ax.taxis, *limits[0:2]), (ax.laxis, *limits[2:4]), (ax.raxis, *limits[4:6])]:
        axis.set_ticks(np.round(np.arange(lo, hi + step / 2, step), 3))
    labelled = [(ax.taxis, ax.set_tlabel, "pixel 3"), (ax.laxis, ax.set_llabel, "pixel 2"),
                (ax.raxis, ax.set_rlabel, "pixel 1")]
    for axis, set_label, pixel in labelled:
        colour = PIXEL_AXIS_COLOURS[pixel]
        if labels:
            set_label(f"{pixel} relative QE", color=colour, fontsize=8)
        axis.set_label_position("tick1")
        axis.set_tick_params(colors=colour, labelsize=7, width=0.75)
    # mpltern names each side after the corner opposite it: tside is the bottom edge, lside the
    # right and rside the left; each is drawn in the colour of the axis whose ticks it carries.
    for spine, pixel in [("tside", "pixel 1"), ("lside", "pixel 3"), ("rside", "pixel 2")]:
        ax.spines[spine].set_color(PIXEL_AXIS_COLOURS[pixel])
        ax.spines[spine].set_linewidth(1.0)
    ax.grid(color="gray", lw=0.5, ls="--", alpha=0.6)
    if outline is not None:
        t0, t1, l0, l1, r0, r1 = outline
        corners = np.array([[t1, l0, r0], [t0, l1, r0], [t0, l0, r1], [t1, l0, r0]])
        ax.plot(corners[:, 0], corners[:, 1], corners[:, 2], color="white", lw=0.75)
    if legend:
        handles = [plt.Line2D([], [], ls="none") for _ in shown]
        leg = ax.legend(handles, [DYES[d][0] for d in shown], ncol=2, loc="lower right",
                        bbox_to_anchor=(0.86, 0.02), fontsize=7, handlelength=0, handletextpad=0,
                        columnspacing=1.0, labelspacing=0.15, frameon=True, fancybox=True,
                        facecolor="#7a7a7a", edgecolor="none", framealpha=1)
        for text, dye in zip(leg.get_texts(), shown):
            text.set_color(DYES[dye][3])
    return fig, ax


# --- Figure 2b, c: a field of view with two single-step bleachers -------------------------------
# As published: the field of view 58.8 mm high (full camera ROI); two insets, each a molecule's
# 10 x 10 px PSF (300 nm scale bar) beside its photoelectron trace, on a pale box edged in the
# dye's colour (the second dashed), joined by lines to a box round the molecule. Inset positions
# (left, bottom, width, height; fractions of the field of view) as measured from the figure.
FOV_HEIGHT_MM = 58.8
FOV_PANELS = {
    "ATTO565": dict(colour="#00ff00", insets=[(0.02, 0.731, 0.961, 0.252), (0.02, 0.270, 0.961, 0.252)]),
    "ATTO633": dict(colour="#ff0000", insets=[(0.094, 0.393, 0.888, 0.252), (0.094, 0.097, 0.888, 0.252)]),
}
ZOOM_PX = 10   # PSF zoom-in side (camera px)


FOV_KEEP_CLEAR = 0.12   # bottom strip (fraction of the height) holding the colour and scale bars


def under_insets(xc, yc, width, height, insets, pad_px=ZOOM_PX):
    """True if a molecule at (xc, yc) camera px would sit under (or within pad_px of) an inset,
    or in the bottom strip with the colour and scale bars."""
    fx = (xc + 0.5) / width
    fy = 1 - (yc + 0.5) / height
    px, py = pad_px / width, pad_px / height
    return fy < FOV_KEEP_CLEAR + py or any(l - px < fx < l + w + px and b - py < fy < b + h + py
                                           for l, b, w, h in insets)


def bleach_fov_panel(fov_img, cands, exposure_s, colour, insets, pixel_nm, plotter, masks=None):
    """Fig. 2b / c: the field of view (with a photoelectron colour bar) and one inset per candidate.

    Args:
        fov_img: Field of view (photoelectrons), rows x cols of the camera ROI.
        cands: Candidates (``xc``, ``yc``, ``trace`` in photoelectrons, ``zoom`` 10 x 10 px, and
            ``zoom_origin``: the sensor (row, col) of the zoom's top-left pixel, for the Bayer key).
        masks: The camera calibration's full-sensor colour masks. If given, each PSF zoom-in gets a
            2 x 2 Bayer key (``PlottingBase.add_bayer_key``) showing the filters at its sensor position.
        exposure_s: Frame time (s).
        colour: Edge colour of the insets and molecule boxes.
        insets: (left, bottom, width, height) of each inset, fractions of the figure.
        pixel_nm: Camera pixel size in the sample (nm).
        plotter: ``PlottingBase.PublicationPlotter`` (image and scale bars).

    Returns:
        (fig, ax) of the field of view.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    mm = 1 / 25.4
    height, width = fov_img.shape
    H = FOV_HEIGHT_MM
    W = H * width / height
    fig = plt.figure(figsize=(W * mm, H * mm))
    ax = fig.add_axes([0, 0, 1, 1])
    _, im = plotter.image_plot(ax, fov_img, cmap="gray", origin="upper", vmin=np.percentile(fov_img, 1),
                               vmax=np.percentile(fov_img, 99.5), scalebar=True, pixelsize=pixel_nm,
                               scalebarsize=10000, scalebarlabel="10 µm")
    # photoelectron colour bar, bottom left (below the insets), white on the image
    cax = ax.inset_axes([0.03, 0.05, 0.3, 0.018])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=6, colors="white", length=1.5, width=0.5, pad=1)
    cbar.outline.set_edgecolor("white")
    cbar.outline.set_linewidth(0.5)
    cbar.ax.text(1.04, 0.5, r"pe$^{-}$", transform=cbar.ax.transAxes, color="white", fontsize=7,
                 va="center", ha="left")
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    to_fig = lambda x, y: fig.transFigure.inverted().transform(ax.transData.transform((x, y)))
    half = ZOOM_PX / 2
    for i, (cand, (l, b, w, h)) in enumerate(zip(cands, insets)):
        ls = "-" if i == 0 else "--"
        # box round the molecule, and lines from it to the inset's near corners
        x, y = round(cand["xc"]), round(cand["yc"])
        ax.add_patch(mpatches.Rectangle((x - half - 0.5, y - half - 0.5), 2 * half, 2 * half,
                                        lw=0.75, edgecolor=colour, facecolor="none", ls=ls))
        (bx0, by0), (bx1, by1) = to_fig(x - half - 0.5, y + half - 0.5), to_fig(x + half - 0.5, y - half - 0.5)
        if by0 > b + h:          # molecule above the inset: join its bottom corners to the inset's top
            pairs = [((bx0, by0), (l, b + h)), ((bx1, by0), (l + w, b + h))]
        else:                    # below: its top corners to the inset's bottom
            pairs = [((bx0, by1), (l, b)), ((bx1, by1), (l + w, b))]
        for (x0, y0), (x1, y1) in pairs:
            fig.add_artist(Line2D([x0, x1], [y0, y1], transform=fig.transFigure, color=colour, lw=0.75, ls=ls,
                                  zorder=0.9))
        # (figure-level artists over the field of view and under the inset axes; the inset boxes
        # cover the lines, as published, where a line to the lower inset passes the upper one)
        fig.add_artist(mpatches.Rectangle((l, b), w, h, transform=fig.transFigure, facecolor=(1, 1, 1, 0.8),
                                          edgecolor=colour, lw=1.0, ls=ls, zorder=1))
        # inset layout in mm (as published: 59 x 14.8 mm; PSF 12.3 mm square; trace beside it)
        iw, ih = w * W, h * H
        to_rect = lambda x0, y0, dx, dy: [l + x0 / W, b + y0 / H, dx / W, dy / H]
        ax_z = fig.add_axes(to_rect(0.8, ih - 0.85 - 12.3, 12.3, 12.3), zorder=2)
        plotter.image_plot(ax_z, cand["zoom"], cmap="gray", origin="upper",
                           vmin=np.percentile(cand["zoom"], 1), vmax=np.percentile(cand["zoom"], 99.9))
        if masks is not None:
            from pyS3M.PlottingBase import add_bayer_key
            row_min, col_min = cand["zoom_origin"]
            add_bayer_key(ax_z, masks, row_min=row_min, col_min=col_min)
        ax_z.add_artist(AnchoredSizeBar(ax_z.transData, 300 / pixel_nm, "300 nm", loc="lower left",
                                        pad=0.15, sep=1, borderpad=0.1, color="white", frameon=False,
                                        size_vertical=0.3, label_top=True, fontproperties={"size": 6.5}))
        ax_t = fig.add_axes(to_rect(20.5, ih - 9.9, iw - 20.5 - 1.1, 9.9 - 0.85), zorder=2)
        trace = cand["trace"] - cand["trace"].min()
        t = np.arange(len(trace)) * exposure_s
        ax_t.plot(t, trace / 1e3, color="black", lw=1.0)
        ax_t.set_xlim(t[0], t[-1])
        ax_t.set_ylim(bottom=0)
        ax_t.set_facecolor("none")
        ax_t.tick_params(labelsize=6, length=2, width=0.5, pad=1)
        for spine in ax_t.spines.values():
            spine.set_linewidth(0.75)
        ax_t.set_ylabel(r"pe$^{-}$/10$^3$", fontsize=6.5, labelpad=1)
        ax_t.set_xlabel("time/s", fontsize=7, labelpad=0)
        ax_t.grid(False)
    return fig, ax
