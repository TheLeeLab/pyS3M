"""Branches the module-level test files do not reach: error paths, fallbacks and the pure-Python
bodies of numba kernels (run through ``.py_func``, which coverage can trace)."""
from __future__ import annotations

import json
import logging
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import pyS3M.FitGate as FG
import pyS3M.simulation.vectorial_sampler as vs
from pyS3M.diffusion import MSD
from pyS3M.ImageAnalysisFunctions import FittingResultProcessor, FittingStrategy
from matplotlib.colors import to_hex

from pyS3M.PlottingBase import BAYER_COLOURS, add_bayer_key
from pyS3M.SpectralFunctions import Spectral_Funcs
from pyS3M.simulation.pattern_source import SimulatedDye


# ---------------------------------------------------------------- FitGate

def test_default_cache_dir_without_env_is_beside_the_calibrations(monkeypatch):
    monkeypatch.delenv("PYS3M_FIT_GATE_DIR", raising=False)
    path = FG.default_cache_dir()
    assert path.name == "fit_gate" and path.parent.name == "Camera_Calibrations"


def test_default_cache_dir_falls_back_to_home_cache(monkeypatch):
    monkeypatch.delenv("PYS3M_FIT_GATE_DIR", raising=False)
    monkeypatch.setattr(Path, "is_dir", lambda self: False)
    assert FG.default_cache_dir() == Path.home() / ".cache" / "pyS3M" / "fit_gate"


def _fake_table():
    table = np.tile(np.linspace(20.0, 5.0, len(FG.DeltaChi2Gate.RATE_GRID)), (FG.DeltaChi2Gate.N_BINS, 1))
    return table, np.geomspace(0.03, 100.0, FG.DeltaChi2Gate.N_BINS)


def test_unreadable_cached_calibration_is_ignored_and_recalibrated(tmp_path, monkeypatch, caplog):
    gate = FG.DeltaChi2Gate(cache_dir=tmp_path, n_null=10)
    masks = np.ones((12, 12, 3), dtype=bool)
    key = gate.calibration_key(FittingStrategy.STANDARD, 12, 1.0, 3)
    path = gate._cache_path(key)
    path.write_text("{not json")
    calls = []
    monkeypatch.setattr(gate, "_calibrate", lambda *a: calls.append(a) or _fake_table())
    with caplog.at_level(logging.WARNING, logger=FG.logger.name):
        gate.prepare(FittingStrategy.STANDARD, 12, 1.0, masks)
    assert "unreadable" in caplog.text and len(calls) == 1
    assert json.loads(path.read_text())["key"] == key       # rewritten with the new calibration
    assert np.isfinite(gate.threshold(1.0))


def test_calibration_that_cannot_be_cached_is_still_used(tmp_path, caplog):
    blocker = tmp_path / "a_file"
    blocker.write_text("")
    gate = FG.DeltaChi2Gate(cache_dir=blocker / "sub")       # its parent is a file: mkdir fails
    table, bg = _fake_table()
    with caplog.at_level(logging.WARNING, logger=FG.logger.name):
        gate._save(blocker / "sub" / "cal.json", {"k": 1}, table, bg)
    assert "Could not cache" in caplog.text


# ---------------------------------------------------------------- ImageAnalysisFunctions

def test_fraction_errors_skip_a_block_with_no_signal():
    # STANDARD, 3 channels: [row, col, s_row, s_col, bg_B, bg_G, bg_R, A_B, A_G, A_R] in sqrt space
    pfit = np.array([5.0, 5.0, 1.5, 1.5, 0.0, 0.0, 0.0, 10.0, 12.0, 14.0])
    out = FittingResultProcessor.fraction_errors(np.eye(10) * 0.01, FittingStrategy.STANDARD, pfit)
    assert np.isnan(out[:3]).all()            # backgrounds all zero: nothing to normalise
    assert np.isfinite(out[3:]).all()


def test_delta_chi_squared_skips_an_empty_colour_channel():
    data = np.full((4, 4), 2.0)
    masks = np.zeros((4, 4, 2), dtype=bool)
    masks[..., 0] = True                       # channel 1 covers no pixel
    dchi2 = FittingResultProcessor.delta_chi_squared(np.zeros(16), data, masks, np.ones((4, 4)))
    assert dchi2 == pytest.approx(0.0)


def test_xy_permutation_of_a_single_parameter_is_identity():
    assert FittingResultProcessor._xy_permutation(FittingStrategy.STANDARD, 1).tolist() == [0]


# ---------------------------------------------------------------- PlottingBase

def test_bayer_key_without_an_image_uses_the_masks_from_their_offset():
    fig, ax = plt.subplots()
    pixel_type = np.tile([[0, 1], [1, 2]], (4, 4))
    add_bayer_key(ax, pixel_type, n=2, row_min=1, col_min=1)
    filled = {to_hex(p.get_facecolor()) for p in ax.patches if p.get_fill()}
    # pixel types under the key, from (row 1, col 1): [[1, 0], [2, 1]] -> G, B, R
    assert filled == {to_hex(BAYER_COLOURS[k]) for k in (0, 1, 2)}
    plt.close(fig)


# ---------------------------------------------------------------- SpectralFunctions

def test_dye_spectrum_not_integrating_to_one_is_renormalised_with_a_warning(monkeypatch):
    S_F = Spectral_Funcs()
    R, G, B, wl = S_F.getpixelefficiency()
    pixel_QYs = np.vstack([B, G, R])
    expected, _ = S_F.get_pixel_fractions_dye_and_filters(["ATTO 647N"], None, wl, pixel_QYs)
    real = S_F.get_spectral_data
    # a malformed database entry: the emission spectrum integrates to 2
    monkeypatch.setattr(S_F, "get_spectral_data", lambda *a, **k: 2.0 * real(*a, **k))
    with pytest.warns(UserWarning, match="does not integrate to 1"):
        fractions, _ = S_F.get_pixel_fractions_dye_and_filters(["ATTO 647N"], None, wl, pixel_QYs)
    assert np.allclose(fractions, expected)


def test_bootstrap_colour_ratios_reject_a_spectrum_without_weight(monkeypatch):
    S_F = Spectral_Funcs()
    wl = np.linspace(500, 700, 21)
    monkeypatch.setattr(S_F, "sample_photons_from_spectrum", lambda s, w, n, rng: np.full(n, 600.0))
    with pytest.raises(ValueError, match="no positive weight"):
        S_F.generate_bootstrap_colour_ratios(np.zeros_like(wl), wl, np.ones((3, len(wl))), 5, 2)


# ---------------------------------------------------------------- pattern_source.SimulatedDye

def test_simulated_dye_shape_check_repr_and_equality():
    wl = np.linspace(500, 700, 5)
    with pytest.raises(ValueError, match="same shape"):
        SimulatedDye(np.ones(4), wl)
    dye = SimulatedDye(np.ones(5), wl, label="test")
    assert repr(dye) == "SimulatedDye('test', 500-700 nm)"
    assert dye == SimulatedDye(np.ones(5), wl, label="test")
    assert dye != SimulatedDye(np.ones(5), wl, label="other") and dye != "test"


# ---------------------------------------------------------------- diffusion.msd

def _walk(n=60, seed=0):
    return np.cumsum(np.random.default_rng(seed).normal(0, 1, (n, 2)), axis=0)


def test_pmin_xm_python_body_both_branches():
    pa, pb = MSD.PMin_XM.py_func(np.inf, 100)
    assert (pa, pb) == MSD.PMin_XM(np.inf, 100)
    assert MSD.PMin_XM.py_func(0.5, 100) == MSD.PMin_XM(0.5, 100)
    assert min(MSD.PMin_XM.py_func(0.0, 20)) >= 2


def test_olsf_stops_at_maxiter(capsys):
    msd = MSD()
    counter = iter(range(3, 1000))
    msd.PMin_XM = lambda x, N: (next(counter), next(counter))   # never repeats: never converges
    D, sigma2 = msd.DSigma2_OLSF(_walk(), 1.0, n_d=2, maxiter=3)
    assert np.isnan(D) and np.isnan(sigma2)
    assert "did not converge in 3 iterations" in capsys.readouterr().out


@pytest.mark.parametrize("which", ["intercept", "slope"])
@pytest.mark.parametrize("quiet", [False, True])
def test_olsf_gives_nan_when_more_points_are_asked_than_the_msd_has(which, quiet, capsys):
    msd = MSD()
    coords = _walk()
    p0 = int(np.floor(coords.size / 10))
    big = 10 * len(coords)
    # PMin_XM's (pa, pb) is appended whole to each list, so its last value is the next fit's
    # size. "intercept": both fits ask for too many points. "slope": the intercept converges at
    # the first call (p0 again), then the slope fit asks for too many.
    answers = iter([(big, big)] * 4) if which == "intercept" else iter([(p0, p0)] + [(big, big)] * 3)
    msd.PMin_XM = lambda x, N: next(answers)
    D, sigma2 = msd.DSigma2_OLSF(coords, 1.0, n_d=2, supress_warning=quiet)
    assert np.isnan(D) and np.isnan(sigma2)
    assert ("did not converge" in capsys.readouterr().out) is not quiet


# ---------------------------------------------------------------- vectorial_sampler

@pytest.fixture(scope="module")
def radial_psf():
    return vs.RadialDipolePSF(NA=1.49, n_medium=1.33, n_immersion=1.515)


def test_pixel_patch_needs_odd_sizes(radial_psf):
    with pytest.raises(ValueError, match="must be odd"):
        radial_psf.pixel_patch(0.6, 0.069, 4)


@pytest.mark.parametrize("xp", [np.array([0.0, 1.0, 3.0, 7.0]), np.array([0.0, 2.0, 4.0, 6.0, 8.0])[:4]],
                         ids=["non-uniform", "uniform"])
def test_interp_rows_matches_np_interp(xp):
    fp = np.array([[0.0, 1.0, 2.0, 3.0], [5.0, 4.0, 2.0, 0.0]])
    x = np.array([0.5, 2.0, 5.0, 6.5])
    rows = np.array([0, 1, 0, 1])
    expected = [np.interp(xi, xp, fp[r]) for xi, r in zip(x, rows)]
    assert np.allclose(vs._interp_rows(x, rows, xp, fp), expected)


@pytest.fixture
def python_kernel(monkeypatch):
    """Run photon_by_photon_photoelectrons through the kernel's Python body (traceable)."""
    monkeypatch.setattr(vs, "_photon_kernel", vs._photon_kernel.py_func)


def _qe(wl):
    return np.vstack([np.full_like(wl, 0.5), np.linspace(0.2, 0.9, len(wl))])


@pytest.mark.parametrize("uniform_grid", [True, False])
def test_photon_kernel_python_body(radial_psf, python_kernel, uniform_grid):
    wl = np.linspace(500, 700, 41) if uniform_grid else np.r_[np.linspace(500, 600, 11), np.linspace(605, 700, 20)]
    spectrum = np.exp(-0.5 * ((wl - 620) / 25) ** 2)
    pixel_type = np.indices((9, 9)).sum(axis=0) % 2
    rng = np.random.default_rng(1)
    counts, landed = vs.photon_by_photon_photoelectrons(
        radial_psf, spectrum, wl, _qe(wl), pixel_type, [40, 40, 40],
        np.array([4.0, 4.0, 30.0]), np.array([4.0, 4.0, 4.0]), 69.0, rng,
        rqe=np.full((9, 9), 0.9), land_in_image=False, frame_index=np.array([0, 1, 1]),
        motion_px=np.array([[2.0, 0.0], [0.0, 0.0], [0.0, 0.0]]), return_landed=True,
    )
    assert counts.shape == landed.shape == (2, 9, 9)
    assert landed[0].sum() > 0 and (counts <= landed).all()


def test_photon_kernel_python_body_monochromatic_with_redraw(radial_psf, python_kernel):
    wl = np.linspace(500, 700, 41)
    spectrum = np.zeros_like(wl)
    spectrum[20] = 1.0
    counts = vs.photon_by_photon_photoelectrons(
        radial_psf, spectrum, wl, _qe(wl), np.zeros((7, 7), dtype=int), 30, np.array([3.0]),
        np.array([3.0]), 69.0, np.random.default_rng(2), poisson=False,
    )
    assert counts.shape == (1, 7, 7) and counts.sum() > 0


def test_photons_that_never_land_raise(python_kernel):
    # A PSF whose every photon lands far outside the image (and half are never collected):
    # with land_in_image each photon is redrawn 100,000 times, then counted as failed.
    far = types.SimpleNamespace(_r_of_u=np.full(11, 1e3), _n_u=10, enclosed=0.5, ref_wavelength_um=0.6)
    wl = np.linspace(500, 700, 41)
    with pytest.raises(ValueError, match="1 photons kept landing outside"):
        vs.photon_by_photon_photoelectrons(
            far, np.ones_like(wl), wl, _qe(wl), np.zeros((5, 5), dtype=int), 1, np.array([2.0]),
            np.array([2.0]), 69.0, np.random.default_rng(3), poisson=False,
        )
