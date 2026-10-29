"""Fast photon-by-photon simulation with the vectorial dipole PSF, on any spatially patterned detector.

For a freely rotating dipole in focus at the coverslip (``DipolePSF`` with ``z_dipole_um = 0``) the
back-focal-plane field depends only on the emission angle, not on the wavelength: the Fresnel
coefficients, the aplanatic factor and the dipole projections are all functions of angle, and the
refractive indices are taken as non-dispersive. The image is the Fourier transform of that field
evaluated at k x, k = 2 pi NA / lambda, so the PSF at any wavelength is one shape, rescaled:

    I_lambda(r) = (lambda0 / lambda)^2 I_lambda0(r lambda0 / lambda),

and, for the isotropic emitter, that shape is circularly symmetric. ``RadialDipolePSF`` therefore
computes the PSF once, as a radial profile at a reference wavelength, and draws each photon's
landing position from it at that photon's own wavelength (radius from the profile's inverse CDF,
scaled by lambda / lambda0; azimuth uniform). No wavelength bins and no per-position patches are
needed, and the pixel integration is exact (each photon is binned where it lands).
``photon_by_photon_photoelectrons`` then detects every photon in the pixel it lands in, with that
pixel type's QE at the photon's wavelength, for any mosaic, and ``camera_adu`` adds the camera.

Both identities (scaling and symmetry) are checked against the full 2D ``DipolePSF`` in
``unit_tests/test_vectorial_sampler.py``. They do not hold for an emitter above the coverslip
(the height phase depends on lambda), out of focus, or for a fixed dipole orientation.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .defocus_psf import DipolePSF


class RadialDipolePSF:
    """Radial profile of the isotropic dipole PSF, valid at every wavelength by rescaling.

    Args:
        NA: Numerical aperture of the objective.
        n_medium: Refractive index of the medium the dipole is in (e.g. 1.33, water).
        n_immersion: Refractive index of the immersion medium (e.g. 1.515, oil).
        N_pupil: Pupil grid of the ``DipolePSF`` the profile is computed from.
        supercritical: Collect supercritical-angle light (see ``DipolePSF``).
        ref_wavelength_um: Wavelength the profile is computed at (um).
        r_max_um: Largest radius tabulated at ``ref_wavelength_um`` (um). Photons beyond it (the
            far tail, ``1 - enclosed`` of them) are reported as landing outside the profile.
        n_r: Number of radii tabulated (uniform in r).
    """

    def __init__(
        self,
        NA: float = 1.49,
        n_medium: float = 1.33,
        n_immersion: float = 1.515,
        N_pupil: int = 512,
        supercritical: bool = True,
        ref_wavelength_um: float = 0.6,
        r_max_um: float = 6.0,
        n_r: int = 12001,
    ) -> None:
        self.NA = NA
        self.ref_wavelength_um = ref_wavelength_um
        self.dipole = DipolePSF(NA=NA, n_medium=n_medium, n_immersion=n_immersion,
                                N_pupil=N_pupil, supercritical=supercritical, z_dipole_um=0.0)
        self.r_um = np.linspace(0.0, r_max_um, n_r)
        # Intensity density (per um^2, normalised over the whole image plane) along +x; the
        # isotropic image is circularly symmetric, so this is I(r). Evaluated in chunks: the DFT
        # matrix is (n_points x N_pupil).
        y0 = np.zeros(1)
        self.density_um2 = np.concatenate([
            self.dipole.image(ref_wavelength_um, chunk, y0)[0]
            for chunk in np.array_split(self.r_um, max(1, n_r // 1000))
        ])
        # Enclosed fraction within r: integral of 2 pi r I(r) dr (trapezoid)
        ring = 2.0 * np.pi * self.r_um * self.density_um2
        self.cdf = np.concatenate([[0.0], np.cumsum(0.5 * (ring[1:] + ring[:-1]) * np.diff(self.r_um))])
        self.enclosed = float(self.cdf[-1])

    def density(self, r_um: np.ndarray, wavelength_um: float) -> np.ndarray:
        """PSF intensity density (per um^2) at radius ``r_um`` for ``wavelength_um``, by rescaling."""
        s = self.ref_wavelength_um / wavelength_um
        return s ** 2 * np.interp(np.asarray(r_um) * s, self.r_um, self.density_um2, right=np.nan)

    def sample_offsets(self, wavelengths_um: np.ndarray, rng: np.random.Generator
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Landing positions of photons, relative to the emitter, one per entry of ``wavelengths_um``.

        Returns:
            (dx_um, dy_um, inside): offsets along x (columns) and y (rows), and whether the photon
            landed within the tabulated radius (``r_max_um`` scaled by lambda / lambda0); photons
            outside have NaN offsets.
        """
        wavelengths_um = np.asarray(wavelengths_um, dtype=float)
        u = rng.random(wavelengths_um.shape)
        inside = u < self.enclosed
        r = np.interp(u, self.cdf, self.r_um) * (wavelengths_um / self.ref_wavelength_um)
        r = np.where(inside, r, np.nan)
        phi = rng.random(wavelengths_um.shape) * (2.0 * np.pi)
        return r * np.cos(phi), r * np.sin(phi), inside

    def pixel_patch(self, wavelength_um: float, pixel_um: float, patch_px: int, oversample: int = 5,
                    x0_um: float = 0.0, y0_um: float = 0.0) -> np.ndarray:
        """Pixel-integrated PSF from the radial profile, laid out as ``DipolePSF.pixel_patches``: each
        pixel holds the probability that a collected photon lands in it; the emitter sits (x0, y0)
        from the centre of the patch's central pixel. For checks against the full 2D model.

        Returns:
            shape (patch_px, patch_px).
        """
        if oversample % 2 != 1 or patch_px % 2 != 1:
            raise ValueError("oversample and patch_px must be odd, so the central sub-sample is the patch centre")
        n_fine = patch_px * oversample
        sub = pixel_um / oversample
        coords = (np.arange(n_fine) - n_fine // 2) * sub
        r = np.hypot(coords[None, :] - x0_um, coords[:, None] - y0_um)
        fine = self.density(r, wavelength_um) * sub ** 2
        return fine.reshape(patch_px, oversample, patch_px, oversample).sum(axis=(1, 3))


def sample_wavelengths_nm(spectrum: np.ndarray, wavelength_nm: np.ndarray, shape, rng: np.random.Generator
                          ) -> np.ndarray:
    """Photon wavelengths (nm) drawn from an emission spectrum by inverse CDF (trapezoid CDF, linear
    interpolation; as ``SpectralFunctions.sample_photons_from_spectrum``)."""
    pdf = np.maximum(np.asarray(spectrum, dtype=float), 0.0)
    wavelength_nm = np.asarray(wavelength_nm, dtype=float)
    if np.count_nonzero(pdf) == 1:  # monochromatic
        return np.full(shape, wavelength_nm[np.argmax(pdf)])
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(wavelength_nm))])
    cdf /= cdf[-1]
    return np.interp(rng.random(shape), cdf, wavelength_nm)


def photon_by_photon_photoelectrons(
    psf: RadialDipolePSF,
    spectrum: np.ndarray,
    wavelength_nm: np.ndarray,
    qe_curves: np.ndarray,
    pixel_type: np.ndarray,
    n_photons: float,
    x0_px: np.ndarray,
    y0_px: np.ndarray,
    pixel_nm: float,
    rng: np.random.Generator,
    rqe: Optional[np.ndarray] = None,
    poisson: bool = True,
    land_in_image: bool = True,
) -> np.ndarray:
    """Detected signal photoelectrons per pixel, photon by photon, on any mosaic.

    Each repeat has Poisson(``n_photons``) photons (exactly ``n_photons`` if not ``poisson``); each
    photon gets its own wavelength from ``spectrum``, lands at a position drawn from the PSF at that
    wavelength, and is detected in the pixel it lands in with probability ``QE_type(lambda) * rqe``.

    With ``land_in_image`` (default), ``n_photons`` counts the photons landing in the image, as in
    the production model: a photon drawn outside it is redrawn, wavelength and position, until one
    lands inside. The landed photons then have exactly the physical distribution of photons in the
    image: the PSF's shape inside the image is unchanged, and their spectrum is the emission
    spectrum weighted by each wavelength's fraction landing in the image (slightly bluer, as the PSF
    is narrower at short wavelengths). Otherwise
    ``n_photons`` counts every collected photon and those landing outside the image (~30% for the
    dipole PSF and a 14 x 14 px image) are lost.

    Coordinates follow the fitter's convention: x = column, y = row, pixel centres at integers.

    Args:
        psf: The radial PSF.
        spectrum: Emission spectrum on ``wavelength_nm``.
        wavelength_nm: Wavelength grid (nm) of ``spectrum`` and ``qe_curves``.
        qe_curves: (n_types, n_wavelengths) QE of each pixel type.
        pixel_type: (H, W) integer pixel type of every pixel (index into ``qe_curves``), i.e. the
            mosaic; any pattern.
        n_photons: Mean number of collected photons per repeat.
        x0_px, y0_px: Emitter position per repeat (px; shape (n_repeats,)).
        pixel_nm: Pixel size (nm).
        rng: Random generator.
        rqe: (H, W) relative QE of every pixel (default 1).
        poisson: Draw the photon number per repeat from Poisson(n_photons).
        land_in_image: ``n_photons`` counts photons landing in the image (redraw those outside,
            wavelength and position);
            False: it counts all collected photons, and those outside are lost.

    Returns:
        (n_repeats, H, W) int32 photoelectrons.
    """
    x0_px, y0_px = np.atleast_1d(x0_px).astype(float), np.atleast_1d(y0_px).astype(float)
    n_rep = len(x0_px)
    H, W = pixel_type.shape
    n_emit = rng.poisson(n_photons, n_rep) if poisson else np.full(n_rep, int(round(n_photons)))
    rep = np.repeat(np.arange(n_rep), n_emit)
    lam_nm = sample_wavelengths_nm(spectrum, wavelength_nm, rep.shape, rng)

    def land(sel):
        """Pixel (col, row) and on-chip flag of the photons ``sel``, from fresh PSF draws."""
        dx, dy, inside = psf.sample_offsets(lam_nm[sel] / 1000.0, rng)
        with np.errstate(invalid="ignore"):
            c = np.round(x0_px[rep[sel]] + dx * 1000.0 / pixel_nm)
            r = np.round(y0_px[rep[sel]] + dy * 1000.0 / pixel_nm)
        return c, r, inside & (c >= 0) & (c < W) & (r >= 0) & (r < H)

    col, row, on_chip = land(np.arange(len(rep)))
    for _ in range(10_000):
        if not land_in_image or on_chip.all():
            break
        miss = np.flatnonzero(~on_chip)
        lam_nm[miss] = sample_wavelengths_nm(spectrum, wavelength_nm, len(miss), rng)
        col[miss], row[miss], on_chip[miss] = land(miss)
    else:
        raise ValueError("photons keep landing outside the image: is the emitter far outside it?")
    rep, lam_nm, col, row = rep[on_chip], lam_nm[on_chip], col[on_chip].astype(np.int64), row[on_chip].astype(np.int64)
    ptype = pixel_type[row, col]
    p = np.empty(len(rep))
    for t in range(qe_curves.shape[0]):
        sel = ptype == t
        p[sel] = np.interp(lam_nm[sel], wavelength_nm, qe_curves[t])
    if rqe is not None:
        p *= rqe[row, col]
    hit = rng.random(len(rep)) < p
    counts = np.bincount((rep[hit] * H + row[hit]) * W + col[hit], minlength=n_rep * H * W)
    return counts.reshape(n_rep, H, W).astype(np.int32)


def background_photoelectrons(background_photons: float, background_qe: np.ndarray, n_repeats: int,
                              rng: np.random.Generator, rqe: Optional[np.ndarray] = None) -> np.ndarray:
    """Background photoelectrons, (n_repeats, H, W): Binomial(Poisson(b * rqe), QE_bg) per pixel, with
    ``background_qe`` the (H, W) background QE of each pixel (production's definition,
    ``MultiC_Sim_Funcs.background_pixel_efficiency``)."""
    lam = background_photons * (1.0 if rqe is None else rqe)
    photons = rng.poisson(np.broadcast_to(lam, background_qe.shape), size=(n_repeats,) + background_qe.shape)
    return rng.binomial(photons, background_qe).astype(np.int32)


def camera_adu(photoelectrons: np.ndarray, gain: np.ndarray, offset: np.ndarray, variance: np.ndarray,
               rng: np.random.Generator) -> np.ndarray:
    """ADU images from photoelectrons: Normal(gain * pe + offset, variance), clipped at 0 -- the camera
    model of ``PSF_Functions.photoelectrons_to_image``, vectorised and seeded."""
    mean = gain * photoelectrons + offset
    return np.clip(rng.normal(mean, np.sqrt(variance)), 0.0, None)
