"""QDot data, settings and classifier for the Figure 3 notebook.

Each QDot acquisition on BioStudies (`Ximea/QDots/`) is a 5 x 5 grid of fields of view, one
TIFF (500 frames) plus one ``*_metadata.txt`` per position. Every sample is downloaded, fitted
per position, drift-corrected per position and clustered into molecules by ``_multifov``, with
the settings below (from the analysis of the published figure, B. Clarke).

The mixture's molecules are then sorted into the six QDot species by ``FixedParamsGMM``,
whose means and diagonal covariances come from the single-QDot reference samples.
"""
import numpy as np

import _multifov as mf

QDOT_ROOT = "2026_Multicolour_Paper/Ximea/QDots"

# The six-QDot mixture shown in Figure 3, and the position whose image is shown: a 70 x 100 px
# crop of it (PANEL_CROP, camera px of the ROI) holds every QDot, each confidently assigned.
MIX_FOLDER = f"{QDOT_ROOT}/MixB/1nM525_200pM585_200pM605_50pM655_200pM705_50pM800_514DC_515LP_30p515_1"
MIX_PANEL_POSITION = "Pos002_000"
PANEL_CROP = (335, 405), (405, 505)   # x, y

# Single-QDot references, in the GMM's component order. 525/585/605 use their 30 % 515 nm
# acquisition; 655/705/800 pool the 10 % and 30 % acquisitions (50 positions), as in the paper.
REFERENCES = {
    "QDot800": ("red",        [f"{QDOT_ROOT}/QDot800/1nM_514DC_515LP_10p515_1",
                               f"{QDOT_ROOT}/QDot800/1nM_514DC_515LP_30p515_1"]),
    "QDot705": ("darkorange", [f"{QDOT_ROOT}/QDot700/1nM_514DC_515LP_10p515_1",
                               f"{QDOT_ROOT}/QDot700/1nM_514DC_515LP_30p515_1"]),
    "QDot655": ("gold",       [f"{QDOT_ROOT}/QDot655/1nM_514DC_515LP_10p515_1",
                               f"{QDOT_ROOT}/QDot655/1nM_514DC_515LP_30p515_1"]),
    "QDot605": ("lime",       [f"{QDOT_ROOT}/QDot605/10nM_Qdot525_515LP_515_DC_30p515_1"]),
    "QDot585": ("blue",       [f"{QDOT_ROOT}/QDot585/10nM_Qdot525_515LP_515_DC_30p515_1"]),
    "QDot525": ("#9a589a",    [f"{QDOT_ROOT}/QDot525/10nM_Qdot525_515LP_515_DC_30p515_1"]),
}

# Spot detection and fitting.
FIT_PARAMS = dict(pfa=1e-3, peak_wavelength=0.65, ROI_size=12,
                  NA=1.49, pixel_size=0.069)
DRIFT_PARAMS = dict(segmentation=2, intersect_d=0.5, roi_r=3.0)
# Molecules = HDBSCAN clusters of >= 10 localisations, as stated in the paper's Methods for all
# single-molecule data (Figs. 2, 3). This matches the published reference sets (~1,200 QDot655
# molecules per position, as HDBSCAN(10) gives here) and the MixB set of the same run (906 vs
# 1,093 molecules on Pos001_001, median 48 vs 45 frames). Brendan's QDotSampler instead
# re-clustered the mixture with DBSCAN (min. 2 localisations): pass DBSCAN_MIX_PARAMS to
# cluster_sample to reproduce that.
# Quality cuts, on the mixture and the references alike: > 50 detected photoelectrons (Methods;
# `photons` = A_B + A_G + A_R overcounts them 2-4x); chi^2 < 3; PSF sigma 75-250 nm; colour and
# background errors < 0.05 (each set explicitly: the library cuts the background errors only when
# asked). The colour-error cut matters most: with it the synthetic six-QDot mixture is 96.5 %
# correctly sorted, without it 92.8 % (the old 0.15 cut removed almost nothing).
CLUSTER_PARAMS = dict(clustering_method="HDBSCAN", min_cluster_size=10, chi_val=3,
                      max_localisation_error=1.0, max_colour_error=0.05, max_background_error=0.05,
                      min_sigma=75 / mf.PIXEL_SIZE_NM, max_sigma=250 / mf.PIXEL_SIZE_NM,
                      min_photoelectrons=50)
# DBSCAN eps = epsilon_multiplier x mean localisation error; 1.0 matches QDotSampler's counts
# (4,286 vs 4,677 molecules on Pos001_001); the function's default 0.5 splits single QDots.
DBSCAN_MIX_PARAMS = dict(CLUSTER_PARAMS, clustering_method="DBSCAN", min_cluster_size=2,
                         epsilon_multiplier=1.0)
PIXEL_SIZE_NM = mf.PIXEL_SIZE_NM


def prepare_acquisition(remote_folder, dest_dir, fast_mode, fitter, calib, smoothing_function,
                        drift_corrector, io, position=None, refit=False):
    """Download, fit and drift-correct one QDot acquisition (``_multifov.prepare_acquisition``
    with the QDot fit and drift settings); return its ``*_undrifted_locs.h5`` per position."""
    return mf.prepare_acquisition(remote_folder, dest_dir, fast_mode, fitter, calib,
                                  smoothing_function, io, FIT_PARAMS, drift_corrector=drift_corrector,
                                  drift_params=DRIFT_PARAMS, position=position, refit=refit)


def cluster_sample(sm_extractor, files, out_dir, prefix, params=CLUSTER_PARAMS, recluster=False):
    """Cluster a QDot sample into molecules (``_multifov.cluster_sample``, default
    ``CLUSTER_PARAMS``); returns the single-molecule and single-frame databases."""
    return mf.cluster_sample(sm_extractor, files, out_dir, prefix, params, recluster=recluster)


def reference_components(ref_dbs):
    """Means and diagonal covariances of (A_R, A_G, A_B) for each reference, in dict order."""
    cols = ["A_R", "A_G", "A_B"]
    means = np.array([db[cols].mean().to_numpy() for db in ref_dbs.values()])
    covariances = np.array([np.diag(db[cols].std(ddof=0).to_numpy() ** 2) for db in ref_dbs.values()])
    return means, covariances


class FixedParamsGMM:
    """Gaussian mixture with fixed means and covariances; only the weights are fitted (EM).

    Each point is labelled with its most responsible component. This is the classifier used
    for the paper (there with ``confidence_threshold = 0``, i.e. every molecule is assigned).
    """

    def __init__(self, means, covariances, max_iter=200, tol=1e-6, min_weight=1e-12):
        self.means_ = np.asarray(means, dtype=float)
        self.covariances_ = np.asarray(covariances, dtype=float)
        self.max_iter, self.tol, self.min_weight = max_iter, tol, min_weight
        self.weights_ = np.full(len(self.means_), 1 / len(self.means_))

    def _log_pdf(self, X):
        d = X.shape[1]
        out = np.empty((len(X), len(self.means_)))
        for k, (mu, cov) in enumerate(zip(self.means_, self.covariances_)):
            sign, logdet = np.linalg.slogdet(cov)
            if sign <= 0:
                raise ValueError("Covariance must be positive definite.")
            diff = X - mu
            quad = np.einsum("ij,jk,ik->i", diff, np.linalg.inv(cov), diff)
            out[:, k] = -0.5 * (d * np.log(2 * np.pi) + logdet + quad)
        return out

    def _responsibilities(self, X):
        log_w = self._log_pdf(X) + np.log(np.clip(self.weights_, self.min_weight, None))
        top = log_w.max(axis=1, keepdims=True)
        resp = np.exp(log_w - top)
        total = resp.sum(axis=1, keepdims=True)
        return resp / total, float(np.sum(top + np.log(total)))

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        previous = -np.inf
        for self.n_iter_ in range(1, self.max_iter + 1):
            resp, ll = self._responsibilities(X)
            w = np.clip(resp.mean(axis=0), self.min_weight, None)
            self.weights_ = w / w.sum()
            if abs(ll - previous) < self.tol:
                break
            previous = ll
        self.labels_ = self._responsibilities(X)[0].argmax(axis=1)
        return self

    def predict(self, X):
        return self._responsibilities(np.asarray(X, dtype=float))[0].argmax(axis=1)


# --- Figure 3 f, g at their size in the paper ----------------------------------------------------
# Drawn at the size they have in Figure 3 (placed at 100 %, the fonts are the paper's 7 / 8 pt):
# f's triangle side and g's matrix side, as measured from the figure.
TERNARY_SIDE_MM = 45.3
CONFUSION_SIDE_MM = 37.8
BLUE_SCALE = 6   # f stretches pixel 1 (blue) x 6, then renormalises, to spread the QDots out
PIXEL_AXIS_COLOURS = {"pixel 1": "#5858df", "pixel 2": "#53b03b", "pixel 3": "#ff2a2a"}   # as Fig. 2e


def _mm_figure(plot_w_mm, plot_h_mm, margin_mm, projection=None, **kw):
    """A figure holding one axes of exactly plot_w_mm x plot_h_mm, with margins (left, bottom,
    right, top; mm) for the labels."""
    import matplotlib.pyplot as plt
    mm = 1 / 25.4
    left, bottom, right, top = margin_mm
    W, H = plot_w_mm + left + right, plot_h_mm + bottom + top
    fig = plt.figure(figsize=(W * mm, H * mm))
    fig.set_layout_engine("none")   # keep the axes at their size (the style's constrained layout would not)
    ax = fig.add_axes([left / W, bottom / H, plot_w_mm / W, plot_h_mm / H], projection=projection, **kw)
    return fig, ax


def ternary_panel(mix, side_mm=TERNARY_SIDE_MM, blue_scale=BLUE_SCALE, size=2, alpha_scale=150):
    """Figure 3f: the mixture's molecules on a black ternary, coloured by their assigned QDot.

    Top corner = pixel 3 (``A_R``), left = pixel 2 (``A_G``), right = ``blue_scale`` x pixel 1
    (``A_B``), each axis and its dashed grid in its pixel's colour. Point opacity is
    ``alpha_scale / n`` per QDot, so sparse and abundant QDots show alike.

    Args:
        mix: Molecules with ``A_R``, ``A_G``, ``A_B`` and ``qdot`` (a ``REFERENCES`` key).
        side_mm: Triangle side (mm).

    Returns:
        (fig, ax).
    """
    import mpltern  # noqa: F401  (registers the "ternary" projection)
    fig, ax = _mm_figure(side_mm, side_mm * np.sqrt(3) / 2, (13, 11, 13, 5), projection="ternary",
                         ternary_sum=1)
    ax.set_facecolor("black")
    for name, (colour, _) in REFERENCES.items():
        sub = mix[mix["qdot"] == name]
        if len(sub):
            ax.scatter(sub["A_R"], sub["A_G"], blue_scale * sub["A_B"], color=colour, s=size,
                       alpha=min(1.0, alpha_scale / len(sub)), linewidths=0, rasterized=True)
    ticks = np.round(np.arange(0, 1.01, 0.2), 1)
    axes = [(ax.taxis, ax.set_tlabel, "pixel 3", "pixel 3 relative QE"),
            (ax.laxis, ax.set_llabel, "pixel 2", "pixel 2 relative QE"),
            (ax.raxis, ax.set_rlabel, "pixel 1", f"{blue_scale} × pixel 1 relative QE")]
    for axis, set_label, pixel, label in axes:
        colour = PIXEL_AXIS_COLOURS[pixel]
        axis.set_ticks(ticks)
        set_label(label, color=colour, fontsize=8)
        axis.set_label_position("tick1")
        axis.set_tick_params(colors=colour, labelsize=7, width=0.75)
    # mpltern names each side after the corner opposite it (tside = bottom edge, carrying pixel 1)
    for spine, pixel in [("tside", "pixel 1"), ("lside", "pixel 3"), ("rside", "pixel 2")]:
        ax.spines[spine].set_color(PIXEL_AXIS_COLOURS[pixel])
        ax.spines[spine].set_linewidth(1.0)
    for which, pixel in [("t", "pixel 3"), ("l", "pixel 2"), ("r", "pixel 1")]:
        ax.grid(axis=which, color=PIXEL_AXIS_COLOURS[pixel], lw=0.5, ls="--", alpha=0.8)
    ax.text(0.97, 0.03, rf"$N_\mathrm{{QD}}$ = {len(mix):,}", transform=ax.transAxes, color="white",
            fontsize=7, ha="right", va="bottom")
    return fig, ax


def confusion_panel(cm, names, side_mm=CONFUSION_SIDE_MM):
    """Figure 3g: row-normalised confusion matrix, grey scale, values to 2 decimals, no colour bar.

    Args:
        cm: (n, n) confusion matrix, rows = true QDot, normalised per row.
        names: QDot names in ``cm`` order (``QDot800`` is shown as ``Qdot800``, as in the paper).
        side_mm: Side of the matrix (mm).

    Returns:
        (fig, ax).
    """
    import matplotlib.pyplot as plt
    fig, ax = _mm_figure(side_mm, side_mm, (13, 11, 1, 1))
    labels = [n.replace("QDot", "Qdot") for n in names]
    ax.imshow(cm, cmap=plt.cm.gist_yarg, vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if cm[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", rotation_mode="anchor", fontsize=6.5)
    ax.set_yticks(range(len(labels)), labels, fontsize=6.5)
    ax.tick_params(length=0, pad=1.5)
    ax.set_xlabel("Predicted", fontsize=8, labelpad=1)
    ax.set_ylabel("True", fontsize=8, labelpad=1)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.grid(False)
    return fig, ax
