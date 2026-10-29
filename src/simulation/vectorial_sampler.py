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

from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
from numba import njit, prange

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
        # Inverse CDF tabulated uniformly in u, so a radius is a direct lookup rather than a binary
        # search (4x faster); it differs from interpolating the CDF itself by < 0.005 nm
        # (the radii are tabulated every 0.5 nm). The CDF increases strictly, so r(u) is continuous.
        self._n_u = 2 ** 20
        self._r_of_u = np.interp(np.linspace(0.0, self.enclosed, self._n_u + 1), self.cdf, self.r_um)

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
        x = np.minimum(u, self.enclosed) * (self._n_u / self.enclosed)
        i = np.minimum(x.astype(np.int64), self._n_u - 1)
        r = self._r_of_u[i] + (x - i) * (self._r_of_u[i + 1] - self._r_of_u[i])
        r = np.where(inside, r * (wavelengths_um / self.ref_wavelength_um), np.nan)
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


@lru_cache(maxsize=8)
def radial_dipole_psf(NA: float = 1.49, n_medium: float = 1.33, n_immersion: float = 1.515) -> RadialDipolePSF:
    """``RadialDipolePSF`` with the default sampling, built once per (NA, n_medium, n_immersion)."""
    return RadialDipolePSF(NA=NA, n_medium=n_medium, n_immersion=n_immersion)


def _interp_rows(x: np.ndarray, rows: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """``np.interp(x[k], xp, fp[rows[k]])`` for every k; by direct indexing when ``xp`` is uniform."""
    d = np.diff(xp)
    if len(xp) < 2 or not np.allclose(d, d[0]):
        out = np.empty(len(x))
        for t in range(fp.shape[0]):
            sel = rows == t
            out[sel] = np.interp(x[sel], xp, fp[t])
        return out
    pos = np.clip((x - xp[0]) / d[0], 0.0, len(xp) - 1)
    i = np.minimum(pos.astype(np.int64), len(xp) - 2)
    lo = fp[rows, i]
    return lo + (pos - i) * (fp[rows, i + 1] - lo)


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
    n_photons,
    x0_px: np.ndarray,
    y0_px: np.ndarray,
    pixel_nm: float,
    rng: np.random.Generator,
    rqe: Optional[np.ndarray] = None,
    poisson: bool = True,
    land_in_image: bool = True,
    frame_index: Optional[np.ndarray] = None,
    n_frames: Optional[int] = None,
    motion_px: Optional[np.ndarray] = None,
    return_landed: bool = False,
):
    """Detected signal photoelectrons per pixel, photon by photon, on any mosaic.

    Each emitter has Poisson(``n_photons``) photons (exactly ``n_photons`` if not ``poisson``); each
    photon gets its own wavelength from ``spectrum``, lands at a position drawn from the PSF at that
    wavelength, and is detected in the pixel it lands in with probability ``QE_type(lambda) * rqe``.

    With ``land_in_image`` (default), ``n_photons`` counts the photons landing in the image, as in
    the production model: a photon drawn outside it is redrawn, wavelength and position, until one
    lands inside. This applies to emitters centred on the image (at mid-exposure); one centred off
    it (e.g. a blinking emitter moved off the field while dark) keeps only its photons that land in
    the image, so a distant emitter contributes nothing. The landed photons then have exactly the physical distribution of photons in the
    image: the PSF's shape inside the image is unchanged, and their spectrum is the emission
    spectrum weighted by each wavelength's fraction landing in the image (slightly bluer, as the PSF
    is narrower at short wavelengths). Otherwise ``n_photons`` counts every collected photon and
    those landing outside the image (~30% for the dipole PSF and a 14 x 14 px image) are lost.

    With ``motion_px``, an emitter moves at constant velocity during the exposure, centred on
    (x0, y0) at mid-exposure (as ``MultiC_Sim_Funcs._apply_motion_blur``): each photon is emitted at
    a uniformly random time, so the motion blur is exact rather than sampled.

    Coordinates follow the fitter's convention: x = column, y = row, pixel centres at integers.

    Args:
        psf: The radial PSF.
        spectrum: Emission spectrum (of the photons reaching the camera) on ``wavelength_nm``.
        wavelength_nm: Wavelength grid (nm) of ``spectrum`` and ``qe_curves``.
        qe_curves: (n_types, n_wavelengths) QE of each pixel type.
        pixel_type: (H, W) integer pixel type of every pixel (index into ``qe_curves``), i.e. the
            mosaic, any pattern; or (n_frames, H, W), one per frame.
        n_photons: Mean number of photons per emitter, scalar or (n_emitters,).
        x0_px, y0_px: Emitter positions (px; shape (n_emitters,)).
        pixel_nm: Pixel size (nm).
        rng: Random generator.
        rqe: Relative QE of every pixel, (H, W) or (n_frames, H, W) (default 1).
        poisson: Draw each emitter's photon number from Poisson(n_photons).
        land_in_image: ``n_photons`` counts photons landing in the image for emitters centred on
            it (redraw those outside, wavelength and position); False: it counts all collected
            photons, and those outside are lost.
        frame_index: Frame of each emitter (default: one frame per emitter); emitters in the same
            frame are summed.
        n_frames: Number of frames (default: ``frame_index.max() + 1``).
        motion_px: (n_emitters, 2) displacement (dx, dy) of each emitter over the exposure (px).
        return_landed: Also return the photons landing in each pixel, before detection.

    Returns:
        (n_frames, H, W) int32 photoelectrons; with ``return_landed``, also the landed photons.
    """
    x0_px, y0_px = np.atleast_1d(x0_px).astype(float), np.atleast_1d(y0_px).astype(float)
    n_em = len(x0_px)
    frame_index = np.arange(n_em) if frame_index is None else np.asarray(frame_index, dtype=np.int64)
    n_frames = int(frame_index.max()) + 1 if n_frames is None else int(n_frames)
    H, W = pixel_type.shape[-2:]
    n_photons = np.broadcast_to(np.asarray(n_photons, dtype=float), (n_em,))
    n_emit = rng.poisson(n_photons) if poisson else np.round(n_photons).astype(np.int64)

    # Wavelength inverse CDF (trapezoid CDF, as sample_wavelengths_nm); a monochromatic spectrum
    # is a single wavelength
    wavelength_nm = np.asarray(wavelength_nm, dtype=float)
    pdf = np.maximum(np.asarray(spectrum, dtype=float), 0.0)
    mono = np.count_nonzero(pdf) == 1
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(wavelength_nm))])
    cdf = cdf / cdf[-1] if not mono else cdf
    d = np.diff(wavelength_nm)
    uniform = bool(np.allclose(d, d[0]))
    pixel_type = np.ascontiguousarray(pixel_type, dtype=np.int64)
    pixel_type = pixel_type if pixel_type.ndim == 3 else pixel_type[np.newaxis]
    rqe = np.ones((1, 1, 1)) if rqe is None else np.ascontiguousarray(rqe, dtype=float)
    rqe = rqe if rqe.ndim == 3 else rqe[np.newaxis]
    motion = np.zeros((n_em, 2)) if motion_px is None else np.ascontiguousarray(motion_px, dtype=float)
    centred = (np.abs(x0_px - (W - 1) / 2) <= W / 2) & (np.abs(y0_px - (H - 1) / 2) <= H / 2)

    counts = np.zeros((n_frames, H, W), dtype=np.int32)
    landed = np.zeros((n_frames, H, W), dtype=np.int32)
    # Emitters grouped by frame: frames run in parallel, each with its own seeded random stream,
    # so the result does not depend on the number of threads
    order = np.argsort(frame_index, kind="stable")
    frame_start = np.concatenate([[0], np.cumsum(np.bincount(frame_index, minlength=n_frames))])
    failed = _photon_kernel(
        int(rng.integers(0, 2**31 - 1)), frame_start, n_emit[order].astype(np.int64), x0_px[order], y0_px[order],
        motion[order], motion_px is not None, (centred & land_in_image)[order],
        cdf, wavelength_nm, mono, float(wavelength_nm[np.argmax(pdf)]),
        psf._r_of_u, psf._n_u, psf.enclosed, psf.ref_wavelength_um, 1000.0 / pixel_nm,
        np.ascontiguousarray(qe_curves, dtype=float), uniform,
        pixel_type, rqe, rqe.shape[1] > 1, counts, landed,
    )
    if failed:
        raise ValueError(f"{failed} photons kept landing outside the image")
    return (counts, landed) if return_landed else counts


@njit(cache=True, parallel=True)
def _photon_kernel(seed, frame_start, n_emit, x0, y0, motion, has_motion, redraw,
                   cdf, wl, mono, mono_wl, r_of_u, n_u, enclosed, ref_wl_um, px_per_um,
                   qe_curves, uniform, pixel_type, rqe, has_rqe, counts, landed):
    """Every photon of ``photon_by_photon_photoelectrons``, in one pass: wavelength, landing pixel
    (redrawn, wavelength and position, while it misses the image if ``redraw``), detection.

    Emitters are sorted by frame, frame f's being ``frame_start[f]:frame_start[f + 1]``. Frames run
    in parallel, each seeding its own random stream (numba keeps one generator per thread).

    Returns:
        The number of photons that missed the image 100,000 times in a row (0 normally).
    """
    n_frames, H, W = counts.shape
    n_wl = wl.shape[0]
    failed = np.zeros(n_frames, dtype=np.int64)
    for f in prange(n_frames):
        np.random.seed((seed * 1_000_003 + f * 7919) % 4_294_967_296)
        fp = np.int64(min(f, pixel_type.shape[0] - 1))  # per-frame or shared mosaic
        fr = np.int64(min(f, rqe.shape[0] - 1))
        for e in range(frame_start[f], frame_start[f + 1]):
            for _ in range(n_emit[e]):
                c = -1
                r_ = -1
                lam = mono_wl
                for attempt in range(100_000):
                    # Wavelength: inverse CDF, exactly as np.interp(u, cdf, wl)
                    if not mono:
                        u = np.random.random()
                        j = min(max(np.searchsorted(cdf, u, side="right") - 1, 0), n_wl - 2)
                        span = cdf[j + 1] - cdf[j]
                        lam = wl[j] + (wl[j + 1] - wl[j]) * ((u - cdf[j]) / span if span > 0 else 0.0)
                    # Radius from the tabulated inverse CDF, rescaled to this wavelength; uniform azimuth
                    u = np.random.random()
                    if u < enclosed:
                        x = u * (n_u / enclosed)
                        i = min(int(x), n_u - 1)
                        rad = (r_of_u[i] + (x - i) * (r_of_u[i + 1] - r_of_u[i])) * (lam / 1000.0 / ref_wl_um)
                        phi = 2.0 * np.pi * np.random.random()
                        cx = x0[e]
                        cy = y0[e]
                        if has_motion:
                            t = np.random.random() - 0.5
                            cx += t * motion[e, 0]
                            cy += t * motion[e, 1]
                        c = int(np.rint(cx + rad * np.cos(phi) * px_per_um))
                        r_ = int(np.rint(cy + rad * np.sin(phi) * px_per_um))
                    else:
                        c = -1
                    if (0 <= c < W and 0 <= r_ < H) or not redraw[e]:
                        break
                else:
                    failed[f] += 1
                    continue
                if not (0 <= c < W and 0 <= r_ < H):
                    continue
                landed[f, r_, c] += 1
                # Detection with this pixel type's QE at this wavelength (as np.interp), times rqe
                row = qe_curves[pixel_type[fp, r_, c]]
                if uniform:
                    pos = min(max((lam - wl[0]) / (wl[1] - wl[0]), 0.0), n_wl - 1.0)
                    k = min(int(pos), n_wl - 2)
                else:
                    k = min(max(np.searchsorted(wl, lam, side="right") - 1, 0), n_wl - 2)
                    pos = k + min(max((lam - wl[k]) / (wl[k + 1] - wl[k]), 0.0), 1.0)
                q = row[k] + (pos - k) * (row[k + 1] - row[k])
                if has_rqe:
                    q *= rqe[fr, r_, c]
                if np.random.random() < q:
                    counts[f, r_, c] += 1
    return failed.sum()


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
