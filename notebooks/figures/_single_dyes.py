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
FIT_PARAMS = dict(pfa=1e-3, sigma=1.5, fraction_true=0.05, ROI_size=12, NA=1.49, pixel_size=0.069)
# Molecules = HDBSCAN clusters of >= 10 localisations (paper's Methods).
CLUSTER_PARAMS = dict(clustering_method="HDBSCAN", min_cluster_size=10, chi_val=2,
                      max_localisation_error=1.0, min_photons=100, max_photons=1e6)

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
    """Memory-map a TIFF stack (frames, rows, cols); falls back to reading it."""
    try:
        return tifffile.memmap(tif)
    except ValueError:
        return tifffile.imread(tif)


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
