"""pyS3M.diffusion.msd: D and the MSD as in pyDiffusion_LeeLab; sigma^2 per Michalet & Berglund.

Michalet & Berglund, Phys. Rev. E 85, 061916 (2012), Eq. 10:
rho(i dT) = 2d (sigma^2 - 2 R D dT) + 2d D dT i.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from pyS3M.diffusion import MSD

PYDIFFUSION_SRC = Path.home() / "Documents" / "Code" / "pyDiffusion_LeeLab" / "src"


def brownian_tracks(n_tracks, n_frames, D, dT, sigma, R_sixth, rng, n_d=2, substeps=50):
    """Tracks of a 2D Brownian particle, localised with Gaussian error sigma per axis.

    With ``R_sixth`` the reported position is the mean over the whole frame (exposure =
    frame time, R = 1/6); otherwise the instantaneous position at the frame start (R = 0).
    """
    tracks = []
    for _ in range(n_tracks):
        steps = rng.normal(0, np.sqrt(2 * D * dT / substeps), (n_frames * substeps, n_d))
        path = np.cumsum(steps, axis=0).reshape(n_frames, substeps, n_d)
        pos = path.mean(axis=1) if R_sixth else path[:, 0, :]
        tracks.append(pos + rng.normal(0, sigma, pos.shape))
    return tracks


@pytest.fixture(scope="module")
def pydiffusion_msd():
    if not (PYDIFFUSION_SRC / "MSDFunctions.py").exists():
        pytest.skip("pyDiffusion_LeeLab not available")
    sys.path.insert(0, str(PYDIFFUSION_SRC))
    try:
        from MSDFunctions import MSD as OldMSD
    finally:
        sys.path.remove(str(PYDIFFUSION_SRC))
    return OldMSD()


def test_D_and_msd_identical_to_pydiffusion(pydiffusion_msd):
    rng = np.random.default_rng(3)
    new = MSD()
    tracks = brownian_tracks(40, 60, D=1e5, dT=0.05, sigma=40.0, R_sixth=True, rng=rng)
    tracks += [t[:12] for t in brownian_tracks(20, 12, D=1e4, dT=0.05, sigma=60.0, R_sixth=False, rng=rng)]
    for track in tracks:
        np.testing.assert_array_equal(new.msd_fft(track), pydiffusion_msd.msd_fft(track))
        D_new, _ = new.DSigma2_OLSF(track, 0.05, R=1 / 6, n_d=2, min_points=5)
        D_old, _ = pydiffusion_msd.DSigma2_OLSF(track, 0.05, R=1 / 6, n_d=2, min_points=5)
        np.testing.assert_array_equal(D_new, D_old)


@pytest.mark.parametrize("R_sixth", [False, True])
def test_sigma2_recovers_localisation_variance(R_sixth):
    """sigma^2 = 1600 nm^2; the blur term 2 R D dT is 833 nm^2 here, so the old
    ``4 R D dT / (2 n_d)`` correction would read ~1200 nm^2 low by half of it in 2D."""
    rng = np.random.default_rng(7)
    D, dT, sigma = 1e5, 0.05, 40.0   # nm^2/s, s, nm
    R = 1 / 6 if R_sixth else 0.0
    est = np.array([MSD().DSigma2_OLSF(t, dT, R=R, n_d=2) for t in
                    brownian_tracks(300, 200, D, dT, sigma, R_sixth, rng)])
    D_med, sigma2_med = np.nanmedian(est, axis=0)
    assert D_med == pytest.approx(D, rel=0.05)
    assert sigma2_med == pytest.approx(sigma**2, rel=0.10)


def test_too_short_or_wrong_shape_gives_nan():
    msd = MSD()
    assert np.all(np.isnan(msd.DSigma2_OLSF(np.zeros((5, 2)), 0.05, n_d=2, min_points=10)))
    assert np.all(np.isnan(msd.DSigma2_OLSF(np.zeros((50, 3)), 0.05, n_d=2)))


def test_msd_fft_matches_direct_average():
    rng = np.random.default_rng(0)
    r = rng.normal(size=(30, 2)).cumsum(axis=0)
    direct = [np.mean(np.sum((r[k:] - r[:-k]) ** 2, axis=1)) for k in range(1, 30)]
    np.testing.assert_allclose(MSD().msd_fft(r), direct, rtol=1e-10)
