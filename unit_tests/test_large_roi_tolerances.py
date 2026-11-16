"""Large ROIs (side > FittingConstants.LARGE_ROI_SIZE) are fitted to convergence.

With ~1,600 pixels in a 40 px box, the default ftol (1e-2, a relative drop in total chi^2)
stopped fits far from the minimum, so the fitted colours depended on the starting guess.
"""
from __future__ import annotations

import numpy as np

from pyS3M import ImageAnalysisFunctions as IAF
from pyS3M.FitGate import DeltaChi2Gate
from pyS3M.ImageAnalysisFunctions import FittingConstants, FittingStrategy


def test_small_rois_keep_the_default_stopping_rule():
    kw = IAF._lm_kwargs(FittingConstants.LARGE_ROI_SIZE, 10)
    assert kw == {"ftol": FittingConstants.DEFAULT_FTOL, "xtol": FittingConstants.DEFAULT_XTOL}


def test_large_rois_get_tight_ftol_and_more_evaluations():
    kw = IAF._lm_kwargs(FittingConstants.LARGE_ROI_SIZE + 1, 10)
    assert kw["ftol"] == FittingConstants.LARGE_ROI_FTOL
    assert kw["xtol"] == FittingConstants.LARGE_ROI_XTOL
    assert kw["maxfev"] == FittingConstants.LARGE_ROI_MAXFEV_FACTOR * 200 * 11


def test_gate_key_names_the_large_roi_settings_only_for_large_rois():
    gate = DeltaChi2Gate()
    small = gate.calibration_key(FittingStrategy.STANDARD_DATA, 16, 2.3, 3)
    large = gate.calibration_key(FittingStrategy.STANDARD_DATA, 40, 2.3, 3)
    assert "large_roi_ftol" not in small
    assert large["large_roi_ftol"] == FittingConstants.LARGE_ROI_FTOL
    assert large["large_roi_xtol"] == FittingConstants.LARGE_ROI_XTOL
    assert large["large_roi_maxfev_factor"] == FittingConstants.LARGE_ROI_MAXFEV_FACTOR


def test_large_roi_fit_reaches_the_same_minimum_from_two_starts():
    """A 3,000 photoelectron spot in a 40 px box (the fitter's own model plus Poisson and read
    noise): two starting guesses converge to the same colours. At ftol 1e-2 / xtol 1e-4 they differ by ~0.008."""
    import pyS3M.gaussoptfuncs as gof

    size = 40
    masks = np.zeros((size, size, 3), bool)
    masks[0::2, 0::2, 0] = True
    masks[1::2, 1::2, 2] = True
    masks[:, :, 1] = ~(masks[:, :, 0] | masks[:, :, 2])
    truth = np.r_[19.6, 20.3, 2.2, 2.2, np.sqrt([0.2, 0.2, 0.2]), np.sqrt(3000 * np.array([0.19, 0.76, 0.05]))]
    mean = gof.WLS_model_nobounds(truth, masks, np.arange(size, dtype=np.float32),
                                  np.zeros((size, size), np.float32)).astype(float)
    rng = np.random.default_rng(0)
    data = rng.poisson(mean) + rng.normal(0, 2.3, mean.shape)
    weights = 1.0 / (np.clip(data, 0, None) + 2.3 ** 2)
    proc = IAF.StandardDataFittingProcessor(readnoise=2.3)
    x0 = np.array([19.0, 21.0, 2.0, 2.0, 0.5, 0.5, 0.5, 20.0, 20.0, 20.0])
    fits = []
    for start in (x0, np.r_[x0[:7], np.sqrt([200.0, 400.0, 50.0])]):
        pfit, _, ok = proc._leastsq_step(start, data, masks, weights, size)
        assert ok in (1, 2, 3, 4)
        a = pfit[7:] ** 2
        fits.append(a / a.sum())
    np.testing.assert_allclose(fits[0], fits[1], atol=0.002)   # the fit SD of each share is ~0.014
