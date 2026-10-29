"""pyS3M.simulation.vectorial_sampler: one radial dipole PSF, rescaled per wavelength, photon by photon.

The fast path rests on two identities of ``DipolePSF`` in focus at the coverslip: the image at any
wavelength is the reference image rescaled, and the isotropic image is circularly symmetric. Both
are checked against the full 2D model, as is the photon sampling and the detection on a mosaic.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyS3M.simulation.defocus_psf import DipolePSF
from pyS3M.simulation.vectorial_sampler import (
    RadialDipolePSF,
    background_photoelectrons,
    camera_adu,
    photon_by_photon_photoelectrons,
    sample_wavelengths_nm,
)

PIXEL_UM = 0.069


@pytest.fixture(scope="module")
def psf():
    return RadialDipolePSF(NA=1.49, n_medium=1.33, n_immersion=1.515)


@pytest.mark.parametrize("lam", [0.45, 0.55, 0.75])
def test_psf_is_one_shape_rescaled_with_wavelength(psf, lam):
    d, lam0 = psf.dipole, psf.ref_wavelength_um
    x = np.linspace(-0.8, 0.8, 81)
    direct = d.image(lam, x, x)
    scaled = d.image(lam0, x * lam0 / lam, x * lam0 / lam) * (lam0 / lam) ** 2
    np.testing.assert_allclose(scaled, direct, rtol=0, atol=1e-12 * direct.max())


def test_isotropic_psf_is_circularly_symmetric(psf):
    d = psf.dipole
    r = np.linspace(0, 1.0, 51)
    on_axis = d.image(0.6, r, np.zeros(1))[0]
    for angle in (0.3, 0.7, np.pi / 4):
        off_axis = np.array([d.image(0.6, np.array([ri * np.cos(angle)]), np.array([ri * np.sin(angle)]))[0, 0]
                             for ri in r])
        assert np.abs(off_axis - on_axis).max() < 1e-4 * on_axis.max()


@pytest.mark.parametrize("lam", [0.5, 0.6, 0.72])
@pytest.mark.parametrize("offset", [(0.0, 0.0), (0.0173, -0.0345)])
def test_radial_patch_matches_full_2d_model(psf, lam, offset):
    full = psf.dipole.pixel_patches(lam, PIXEL_UM, 21, 5, *offset)[0]
    radial = psf.pixel_patch(lam, PIXEL_UM, 21, 5, *offset)
    assert np.abs(radial - full).max() < 1e-4 * full.max()


def test_profile_is_normalised_over_the_image_plane(psf):
    # density integrates to 1 over the whole plane; the tabulated radius holds nearly all of it
    assert 0.98 < psf.enclosed < 1.0
    assert np.all(np.diff(psf.cdf) >= -1e-15)


def test_sampled_radii_scale_with_wavelength(psf):
    rng = np.random.default_rng(0)
    n = 200_000
    r = {}
    for lam in (0.5, 0.7):
        dx, dy, inside = psf.sample_offsets(np.full(n, lam), rng)
        r[lam] = np.nanmedian(np.hypot(dx, dy))
    assert r[0.7] / r[0.5] == pytest.approx(0.7 / 0.5, rel=0.01)


def test_photon_by_photon_image_matches_expected_patch(psf):
    """Mean of many mono-600 nm images = n_photons x pixel-integrated PSF x QE of each pixel type."""
    rng = np.random.default_rng(1)
    H = W = 21
    pixel_type = np.indices((H, W)).sum(axis=0) % 2      # a checkerboard of two pixel types
    wl = np.arange(400.0, 801.0)
    spectrum = (wl == 600).astype(float)
    qe = np.vstack([np.full(len(wl), 0.3), np.full(len(wl), 0.8)])
    n, reps, x0, y0 = 2e4, 400, 10.25, 9.6
    pe = photon_by_photon_photoelectrons(psf, spectrum, wl, qe, pixel_type, n,
                                         np.full(reps, x0), np.full(reps, y0), PIXEL_UM * 1000, rng)
    expected = (n * psf.pixel_patch(0.6, PIXEL_UM, 21, 7, (x0 - 10) * PIXEL_UM, (y0 - 10) * PIXEL_UM)
                * np.where(pixel_type == 1, 0.8, 0.3))
    z = (pe.mean(axis=0) - expected) / np.sqrt(np.maximum(expected, 1e-9) / reps)
    bright = expected > 5
    assert np.mean(z[bright] ** 2) == pytest.approx(1.0, abs=0.3)
    assert np.abs(z[bright]).max() < 5


def test_wavelengths_follow_spectrum():
    rng = np.random.default_rng(2)
    wl = np.arange(500.0, 701.0)
    spectrum = np.exp(-0.5 * ((wl - 600) / 20) ** 2)
    lam = sample_wavelengths_nm(spectrum, wl, 200_000, rng)
    assert lam.mean() == pytest.approx(600, abs=0.2)
    assert lam.std() == pytest.approx(20, rel=0.02)
    assert np.all(sample_wavelengths_nm((wl == 550).astype(float), wl, 10, rng) == 550)


def test_background_and_camera_model():
    rng = np.random.default_rng(3)
    qe = np.array([[0.2, 0.6], [0.6, 0.4]])
    bg = background_photoelectrons(10.0, qe, 50_000, rng)
    np.testing.assert_allclose(bg.mean(axis=0), 10.0 * qe, rtol=0.02)
    pe = np.full((50_000, 2, 2), 7)
    adu = camera_adu(pe, gain=np.full((2, 2), 2.0), offset=np.full((2, 2), 100.0),
                     variance=np.full((2, 2), 9.0), rng=rng)
    assert adu.mean() == pytest.approx(114.0, abs=0.05)
    assert adu.std() == pytest.approx(3.0, rel=0.02)
