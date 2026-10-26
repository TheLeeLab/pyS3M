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

# The six-QDot mixture shown in Figure 3, and the position whose image is shown (fov_index 6).
MIX_FOLDER = f"{QDOT_ROOT}/MixB/1nM525_200pM585_200pM605_50pM655_200pM705_50pM800_514DC_515LP_30p515_1"
MIX_PANEL_POSITION = "Pos001_001"

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

# Spot detection and fitting (QDots are bright: fraction_true 0.05, not the 0.2 used for dyes).
FIT_PARAMS = dict(pfa=1e-3, sigma=1.5, fraction_true=0.05, peak_wavelength=0.65, ROI_size=12,
                  NA=1.49, pixel_size=0.069)
DRIFT_PARAMS = dict(segmentation=2, intersect_d=0.5, roi_r=3.0)
# Molecules = HDBSCAN clusters of >= 10 localisations, as stated in the paper's Methods for all
# single-molecule data (Figs. 2, 3). This matches the published reference sets (~1,200 QDot655
# molecules per position, as HDBSCAN(10) gives here) and the MixB set of the same run (906 vs
# 1,093 molecules on Pos001_001, median 48 vs 45 frames). Brendan's QDotSampler instead
# re-clustered the mixture with DBSCAN (min. 2 localisations): pass DBSCAN_MIX_PARAMS to
# cluster_sample to reproduce that.
# Quality cut: > 50 detected photoelectrons (Methods); `photons` (A_B + A_G + A_R) overcounts them 2-4x.
CLUSTER_PARAMS = dict(clustering_method="HDBSCAN", min_cluster_size=10, chi_val=2,
                      max_localisation_error=1.0, min_photons=0, max_photons=1e6, min_photoelectrons=50)
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
