"""The colour-filter mosaic reaches every component that uses it, and unmoved fits are rejected.

A non-Bayer mosaic (RRGB: R 1/2, G 1/4, B 1/4) has different pixel fractions from both
cameras' Bayer patterns, so ``photoelectrons`` written with the wrong mosaic would differ.
"""
from __future__ import annotations

from unittest import mock

import numpy as np
import pandas as pd
import pytest

import pyS3M.ImageAnalysisFunctions as IAF
from pyS3M.AnalysisPipeline import AnalysisPipeline
from pyS3M.CalibrationFunctions import Calibration_Functions
from pyS3M.NileRedFunctions import NileRed_Functions
from pyS3M.SM_extractionfunctions import extract_SMs
from pyS3M.SR_Functions import SuperRes_Functions

RRGB = np.array([["R", "R"], ["G", "B"]])
RRGB_FRACTIONS = {"B": 0.25, "G": 0.25, "R": 0.5}


def _fractions(obj_io):
    return {k: pytest.approx(v) for k, v in obj_io.pixel_fractions.items()}


@pytest.mark.parametrize("camera", ["ximea", "zwo"])
def test_camera_default_mosaic_is_bayer(camera):
    sr = SuperRes_Functions(camera=camera)
    assert sr.io.pixel_fractions == pytest.approx({"B": 0.25, "G": 0.5, "R": 0.25})
    np.testing.assert_array_equal(sr.mask.mosaic_unit, sr.mosaic_unit)


def test_superres_passes_mosaic_to_io_and_masks():
    sr = SuperRes_Functions(camera="zwo", mosaic_unit=RRGB)
    assert sr.io.pixel_fractions == _fractions(sr.io) == RRGB_FRACTIONS
    np.testing.assert_array_equal(sr.mask.mosaic_unit, RRGB)


def test_calibration_passes_mosaic_to_io_and_masks():
    cf = Calibration_Functions(camera="ximea", mosaic_unit=RRGB)
    assert cf.io.pixel_fractions == pytest.approx(RRGB_FRACTIONS)
    np.testing.assert_array_equal(cf.Mask.mosaic_unit, RRGB)


def test_sm_extraction_and_nile_red_pass_mosaic_to_io():
    assert extract_SMs(mosaic_unit=RRGB).io.pixel_fractions == pytest.approx(RRGB_FRACTIONS)
    nr = NileRed_Functions(mosaic_unit=RRGB)
    assert nr.io.pixel_fractions == pytest.approx(RRGB_FRACTIONS)
    np.testing.assert_array_equal(nr.mosaic_unit, RRGB)


def test_pipeline_passes_mosaic_to_sub_functions():
    pipe = AnalysisPipeline(camera="ximea", mosaic_unit=RRGB)
    for sub in (pipe.sr, pipe.sm, pipe.nile_red):
        assert sub.io.pixel_fractions == pytest.approx(RRGB_FRACTIONS)
    np.testing.assert_array_equal(pipe.sr.mosaic_unit, RRGB)


def test_written_photoelectrons_follow_mosaic(tmp_path):
    df = pd.DataFrame({"A_B": [100.0], "A_G": [200.0], "A_R": [300.0],
                       "bg_B": [1.0], "bg_G": [2.0], "bg_R": [3.0], "frame": [0]})
    for col in ["A_B", "A_G", "A_R", "bg_B", "bg_G", "bg_R"]:
        df[f"{col}_err"] = 1.0
    sr = SuperRes_Functions(mosaic_unit=RRGB)
    path = tmp_path / "locs.h5"
    sr.io.write_h5_database(df, path, verbose=False)
    out = sr.io.read_h5_database(path)
    assert out["photoelectrons"].iloc[0] == pytest.approx(0.25 * 100 + 0.25 * 200 + 0.5 * 300)


def test_lm_stuck():
    x0 = np.array([1.0, 2.0, 3.0])
    assert IAF._lm_stuck(x0.copy(), x0)
    assert not IAF._lm_stuck(x0 + 1e-12, x0)


def _masks(size=12):
    m = np.zeros((size, size, 3), bool)
    m[1::2, 1::2, 0] = True
    m[0::2, 1::2, 1] = True
    m[1::2, 0::2, 1] = True
    m[0::2, 0::2, 2] = True
    return m


@pytest.mark.parametrize("processor", [IAF.StandardFittingProcessor,
                                       IAF.StandardDataFittingProcessor,
                                       IAF.CircularFittingProcessor])
def test_fit_that_never_leaves_its_guess_is_rejected(processor):
    """leastsq reports success (ier 1) without moving: the fit must come back NaN."""
    rng = np.random.default_rng(1)
    size = 12
    yy, xx = np.mgrid[:size, :size]
    punctum = (5 + 300 * np.exp(-((xx - 6) ** 2 + (yy - 6) ** 2) / 2.0)
               + rng.normal(0, 1, (size, size))).astype(np.float32)
    weights = np.ones_like(punctum)

    def unmoved(func, x0, args, full_output, ftol, xtol):
        return np.array(x0, dtype=float).copy(), np.eye(len(x0)), {}, "", 1

    with mock.patch.object(IAF, "leastsq", side_effect=unmoved):
        res, _ = processor().fit_single_punctum(punctum, punctum, weights, (0, 0), masks=_masks(size))
    assert np.all(np.isnan(res))

    res, _ = processor().fit_single_punctum(punctum, punctum, weights, (0, 0), masks=_masks(size))
    assert np.isfinite(res[0])
