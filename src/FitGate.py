"""Fit-acceptance gates: decide whether a fitted punctum is a real emitter or noise.

A gate is asked once per fit, after the fit's position/width sanity check
(:meth:`ImageAnalysisFunctions.FittingResultProcessor.process_fit_results`), and
returns True to keep the fit.

- :class:`DeltaChi2Gate` (the default): a likelihood-ratio test. Delta-chi^2 is the
  drop in weighted chi^2 from a background-only model (one constant per colour
  channel, >= 0) to the fitted emitter model, using the fit's own final weights. It
  does not depend on how the model is parameterised, unlike a Wald test on the
  fitted amplitudes. Its null distribution is not a standard chi^2 (position and
  width are undefined without an emitter, amplitudes are bounded at 0, and the fit
  can stop early), and it depends on the background level, so thresholds are
  calibrated by simulation: pure-noise ROIs are fitted with the run's strategy, ROI
  size and read noise at a range of background levels, and the threshold for a
  requested noise rate is the matching upper quantile, interpolated at each fit's
  own background. Calibrations are small JSON files, cached and reused (see
  :func:`default_cache_dir`).
- :class:`AmplitudeSNRGate`: the previous Wald test on the square-root amplitudes,
  kept to reproduce earlier results. It inflates the variance when a colour channel
  is near zero, so it rejects bright single-colour emitters.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Bump when the calibration procedure changes, so stale cached files are not reused.
CALIBRATION_VERSION = 3


def default_cache_dir() -> Path:
    """Where calibration files are kept.

    ``$PYS3M_FIT_GATE_DIR`` if set; otherwise ``Camera_Calibrations/fit_gate`` in the
    pyS3M source tree (next to the camera calibrations); otherwise
    ``~/.cache/pyS3M/fit_gate``.
    """
    env = os.environ.get("PYS3M_FIT_GATE_DIR")
    if env:
        return Path(env)
    for parent in Path(__file__).resolve().parents:
        if (parent / "Camera_Calibrations").is_dir():
            return parent / "Camera_Calibrations" / "fit_gate"
    return Path.home() / ".cache" / "pyS3M" / "fit_gate"


def _strategy_value(strategy: Any) -> str:
    """Strategy as its string value: compares safely even if FittingStrategy was
    imported twice under different module names."""
    return str(getattr(strategy, "value", strategy))


def fitted_background(pfit: np.ndarray, strategy: Any, n_ch: int) -> float:
    """Mean fitted background (photoelectrons per pixel) over the colour channels,
    from a raw (square-root-space) parameter vector."""
    kind = _strategy_value(strategy)
    if kind == "elliptical":
        start = 5          # [x, y, sx, sy, theta, sqrt(bg)...]
    elif kind == "circular":
        start = 3          # [x, y, s, sqrt(bg)...]
    else:
        start = 4          # [x, y, sy, sx, sqrt(bg)...]
    return float(np.mean(np.square(pfit[start:start + n_ch])))


class AmplitudeSNRGate:
    """Previous gate: Wald z = sum|sqrt(A_c)| / sqrt(sum var(sqrt(A_c))) >= threshold."""

    name = "amplitude_snr"

    def __init__(self, threshold: float = 2.0):
        self.threshold = threshold

    def accept(self, pfit, pcov, strategy, delta_chi2=None, background=None) -> bool:
        from pyS3M.ImageAnalysisFunctions import FittingResultProcessor

        kind = _strategy_value(strategy)
        if kind == "elliptical":
            snr = FittingResultProcessor._compute_amplitude_snr_elliptical(pfit, pcov)
        elif kind == "circular":
            snr = FittingResultProcessor._compute_amplitude_snr_circular(pfit, pcov)
        else:
            snr = FittingResultProcessor._compute_amplitude_snr(pfit, pcov)
        return snr >= self.threshold


class RecordingGate:
    """Keeps every fit and records (delta_chi2, background): used for calibration."""

    name = "recording"

    def __init__(self):
        self.records: list[tuple[float, float]] = []

    def accept(self, pfit, pcov, strategy, delta_chi2=None, background=None) -> bool:
        self.records.append((float(delta_chi2), float(background)))
        return True


class DeltaChi2Gate:
    """Likelihood-ratio gate with thresholds calibrated for a noise rate.

    Args:
        noise_rate: Fraction of pure-noise ROIs that should pass (default 0.01).
            Must lie within RATE_GRID's range.
        cache_dir: Where calibration files are read and written (default
            :func:`default_cache_dir`).
        n_null: Pure-noise ROIs fitted per background level when calibrating
            (default 10000, or ``$PYS3M_FIT_GATE_NNULL``).
        smoothing_sigma: Gaussian smoothing (pixels) used to build the calibration
            ROIs' initial guess and first-pass weights, as the pipeline does.
    """

    name = "delta_chi2"
    BACKGROUND_GRID = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)   # photoelectrons per pixel
    RATE_GRID = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)

    def __init__(
        self,
        noise_rate: float = 0.01,
        cache_dir: Optional[Path | str] = None,
        n_null: Optional[int] = None,
        smoothing_sigma: float = 1.5,
    ):
        if not (self.RATE_GRID[0] <= noise_rate <= self.RATE_GRID[-1]):
            raise ValueError(
                f"noise_rate must be between {self.RATE_GRID[0]} and {self.RATE_GRID[-1]}; got {noise_rate}"
            )
        self.noise_rate = float(noise_rate)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.n_null = int(n_null if n_null is not None else os.environ.get("PYS3M_FIT_GATE_NNULL", 10000))
        self.smoothing_sigma = float(smoothing_sigma)
        self.key: Optional[dict] = None
        self._log_bg: Optional[np.ndarray] = None          # log fitted background per calibration level
        self._thresholds: Optional[np.ndarray] = None      # per calibration level, at self.noise_rate

    # ---------------- calibration ----------------

    def calibration_key(self, strategy: Any, roi_size: int, readnoise: float, n_ch: int) -> dict:
        """Everything a calibration depends on."""
        return {
            "version": CALIBRATION_VERSION,
            "strategy": getattr(strategy, "value", str(strategy)),
            "roi_size": int(roi_size),
            "readnoise_e": round(float(readnoise), 1),
            "n_channels": int(n_ch),
            "smoothing_sigma": round(self.smoothing_sigma, 2),
            "n_null": self.n_null,
        }

    def _cache_path(self, key: dict) -> Path:
        digest = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10]
        name = (f"{key['strategy']}_roi{key['roi_size']}_rn{key['readnoise_e']:.1f}"
                f"_ch{key['n_channels']}_{digest}.json")
        return (self.cache_dir or default_cache_dir()) / name

    def prepare(self, strategy: Any, roi_size: int, readnoise: float, masks: np.ndarray) -> None:
        """Make sure thresholds exist for this strategy/ROI size/read noise: reuse the
        current ones, load a cached calibration, or calibrate (and cache) now."""
        key = self.calibration_key(strategy, roi_size, readnoise, masks.shape[-1])
        if key == self.key and self._thresholds is not None:
            return
        path = self._cache_path(key)
        table = fitted_bg = None
        if path.is_file():
            try:
                saved = json.loads(path.read_text())
                if saved.get("key") == key:
                    table = np.asarray(saved["thresholds"], dtype=float)
                    fitted_bg = np.asarray(saved["fitted_background_pe_per_px"], dtype=float)
            except (OSError, ValueError, KeyError):
                logger.warning("Ignoring unreadable fit-gate calibration %s", path)
        if table is None:
            logger.info("Calibrating the delta-chi^2 fit gate (%s) -- once per setting, then cached", key)
            table, fitted_bg = self._calibrate(strategy, roi_size, readnoise, masks)
            self._save(path, key, table, fitted_bg)
        self.key = key
        self._set_table(table, fitted_bg)

    def _set_table(self, table: np.ndarray, fitted_bg: np.ndarray) -> None:
        """Thresholds at this gate's noise rate (log-rate interpolation), indexed by each
        bin's median *fitted* background -- the quantity the gate is given."""
        log_rates = np.log(self.RATE_GRID)
        thresholds = np.array([np.interp(np.log(self.noise_rate), log_rates, row) for row in table])
        log_bg = np.log(np.maximum(fitted_bg, 1e-6))
        order = np.argsort(log_bg)                     # np.interp needs increasing x
        self._log_bg, self._thresholds = log_bg[order], thresholds[order]

    N_BINS = 16   # fitted-background bins of the calibration table (equal counts)

    def _calibrate(self, strategy: Any, roi_size: int, readnoise: float, masks: np.ndarray):
        """Threshold table from fits to pure-noise ROIs.

        Pure-noise ROIs are simulated at every level of BACKGROUND_GRID and fitted. The
        resulting (delta-chi^2, fitted background) pairs are pooled and split into N_BINS
        equal-count bins of *fitted* background -- the quantity the gate sees -- and each
        bin's threshold for a noise rate is the matching upper quantile within the bin.
        Conditioning on the fitted background matters: a noise fluctuation fitted as an
        emitter takes some of the background into its amplitude, so it has a low fitted
        background. Fits that never reach the gate (failed, or outside the ROI) never pass
        and are not part of the table.

        Returns:
            (table, bin_backgrounds): thresholds (N_BINS x len(RATE_GRID)) and each bin's
            median fitted background (photoelectrons per pixel).
        """
        from scipy.ndimage import gaussian_filter
        from pyS3M.ImageAnalysisFunctions import Image_Analysis_Functions

        rng = np.random.default_rng(12345)
        rn = float(readnoise)
        records = []
        for bg in self.BACKGROUND_GRID:
            shape = (self.n_null, roi_size, roi_size)
            pe = (rng.poisson(bg, size=shape) + rng.normal(0.0, rn, size=shape)).astype(np.float32)
            smoothed = gaussian_filter(pe, sigma=(0, self.smoothing_sigma, self.smoothing_sigma), mode="nearest")
            weights = (1.0 / (np.clip(smoothed, 0, None) + 1.0 + rn ** 2)).astype(np.float32)
            recorder = RecordingGate()
            fitter = Image_Analysis_Functions(readnoise=rn, fit_gate=recorder)
            fitter.fit_puncta_method(
                list(pe), list(smoothed), list(weights), [(0.0, 0.0)] * self.n_null,
                list(range(self.n_null)), strategy=strategy, masks=[masks] * self.n_null,
            )
            records.extend(recorder.records)
        rec = np.array(records, dtype=float).reshape(-1, 2)
        rec = rec[np.isfinite(rec).all(axis=1)]
        rec = rec[np.argsort(rec[:, 1], kind="stable")]           # by fitted background
        bins = np.array_split(rec, self.N_BINS)
        table = np.array([[np.quantile(b[:, 0], 1.0 - rate) for rate in self.RATE_GRID] for b in bins])
        bin_bg = np.array([np.median(b[:, 1]) for b in bins])
        return table, bin_bg

    def _save(self, path: Path, key: dict, table: np.ndarray, fitted_bg: np.ndarray) -> None:
        payload = {
            "key": key,
            "background_grid_pe_per_px": list(self.BACKGROUND_GRID),
            "fitted_background_pe_per_px": fitted_bg.round(5).tolist(),
            "noise_rate_grid": list(self.RATE_GRID),
            "thresholds": table.round(4).tolist(),
            "description": "delta-chi^2 thresholds: rows = fitted-background bins (median in fitted_background_pe_per_px), columns = noise rate",
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(payload, indent=1))
            tmp.replace(path)                          # atomic: concurrent runs never see half a file
        except OSError as err:                         # read-only install: still usable, just not cached
            logger.warning("Could not cache fit-gate calibration to %s: %s", path, err)

    # ---------------- decision ----------------

    def threshold(self, background: float) -> float:
        """Delta-chi^2 threshold at a fitted background (photoelectrons per pixel)."""
        if self._thresholds is None:
            raise RuntimeError("DeltaChi2Gate used before prepare(): no calibration loaded")
        log_bg = np.log(max(float(background), 1e-6))
        return float(np.interp(log_bg, self._log_bg, self._thresholds))

    def accept(self, pfit, pcov, strategy, delta_chi2=None, background=None) -> bool:
        if delta_chi2 is None or not np.isfinite(delta_chi2):
            return False
        return delta_chi2 >= self.threshold(background)


def make_gate(fit_gate: Any, noise_rate: float = 0.01) -> Any:
    """Build a gate from a setting: ``"delta_chi2"`` (default), ``"amplitude_snr"``,
    ``None`` (no gate), or a ready-made gate object."""
    if fit_gate is None or hasattr(fit_gate, "accept"):
        return fit_gate
    if fit_gate == "delta_chi2":
        return DeltaChi2Gate(noise_rate=noise_rate)
    if fit_gate == "amplitude_snr":
        return AmplitudeSNRGate()
    raise ValueError(f"Unknown fit_gate {fit_gate!r}: use 'delta_chi2', 'amplitude_snr' or None")
