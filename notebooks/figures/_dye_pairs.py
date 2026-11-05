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

# Spot detection and fitting, as for the single dyes (fraction_true 0.2, not the QDots' 0.05);
# the PSF wavelength is set per pair (PAIRS).
FIT_PARAMS = dict(pfa=1e-3, sigma=1.5, fraction_true=0.2, ROI_size=12, NA=1.49, pixel_size=0.069)
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
           min_photons=0, max_photons=1e6, min_photoelectrons=50)
CLUSTER_PARAMS = {
    "ATTO520_Rho6G":   dict(_QC, min_cluster_size=2, max_localisation_error=1.0),
    "ATTO565_ATTO594": dict(_QC, min_cluster_size=5, max_localisation_error=0.5),
    "ATTO565_ATTO633": dict(_QC, min_cluster_size=2, max_localisation_error=1.0),
    "ATTO647N_QD800":  dict(_QC, min_cluster_size=5, max_localisation_error=1.0),
}

# Molecules that define the two mean colours: bright (`photons`, as in the published analysis)
# and inside a colour box that excludes outliers. (min photons, min A_R, min A_G, max A_G)
REFERENCE_CUTS = {
    "ATTO520_Rho6G":   (5000, 0.175, 0.64, 1.0),
    "ATTO565_ATTO594": (2000, 0.55, 0.15, 1.0),
    "ATTO565_ATTO633": (20000, 0.40, 0.15, 1.0),
    "ATTO647N_QD800":  (25000, 0.40, 0.10, 1.0),
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
    min_photons, min_ar, min_ag, max_ag = REFERENCE_CUTS[pair]
    keep = ((sm_db["photons"] > min_photons) & (sm_db["A_R"] > min_ar)
            & (sm_db["A_G"] > min_ag) & (sm_db["A_G"] < max_ag))
    return sm_db[keep]


def photoelectron_accumulation(sf_db, io):
    """Each molecule's colour as its frames accumulate, against its accumulated photoelectrons.

    One row per (molecule, frame), frames in time order: ``photoelectrons_accumulated`` is the
    running sum, and ``A_R``/``A_G``/``A_B`` the running inverse-variance weighted means with
    their errors (as ``build_photon_accumulation_database``, which accumulates ``photons``).
    ``photons_accumulated`` is set to the photoelectrons too, because that is the column the
    misidentification analysis bins on.
    """
    if "photoelectrons" not in sf_db.columns:
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
    out["photons_accumulated"] = out["photoelectrons_accumulated"]
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
    return np.sqrt(s["photon_bin_min"] * s["photon_bin_max"]).to_numpy(), \
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
