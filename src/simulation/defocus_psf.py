"""Vectorial Debye PSF model for defocus simulation.

Port of simulation-phasemask-psf-main (MATLAB) to Python/NumPy.
Implements the vectorial Debye diffraction integral with:
  - Exact defocus phase (not paraxial approximation)
  - Depth-induced spherical aberration (Gibson-Lanni model)
  - Richards-Wolf apodization
  - Incoherent polychromatic spectral integration

At NA > n_sample (e.g. NA 1.49, n_water 1.33), only the propagating
aperture (rho <= n_sample/NA) contributes to far-field SMLM PSFs.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))


class VectorialPSF:
    """Vectorial Debye PSF for high-NA oil-immersion objectives.

    Computes defocused PSFs using the Fourier-optics propagation:
        E_img = fftshift(fft2(ifftshift(pad(A * e^{iW}))))
        I = |E_img|^2

    where A is the Richards-Wolf apodization and W is the total wavefront
    (defocus + depth-induced spherical aberration + lateral tip/tilt).

    Args:
        NA: Numerical aperture of the objective.
        n_medium: Refractive index of the sample medium (e.g. 1.33 for water).
        n_immersion: Refractive index of the immersion medium (e.g. 1.515 for oil).
        pix_obj_um: Object-space pixel size in µm (e.g. 0.069 for 69 nm).
            Matches pyS3M's ``pixel_size`` convention (stored in nm; divide by
            1000 to get µm).
        psf_size: Output PSF patch side length in pixels (odd preferred).
        N_pupil: Pupil grid size (N_pupil × N_pupil). Higher = more accurate
            but slower. 256 is sufficient for 1 nm phase accuracy at NA 1.49.
    """

    def __init__(
        self,
        NA: float,
        n_medium: float,
        n_immersion: float,
        pix_obj_um: float,
        psf_size: int = 21,
        N_pupil: int = 256,
    ) -> None:
        self.NA = NA
        self.n_medium = n_medium
        self.n_immersion = n_immersion
        self.pix_obj_um = pix_obj_um
        self.psf_size = psf_size
        self.N_pupil = N_pupil

        self._build_pupil_grid()

    # ------------------------------------------------------------------
    # Pupil grid (built once, reused for all wavelengths / z-planes)
    # ------------------------------------------------------------------

    def _build_pupil_grid(self) -> None:
        """Construct normalised pupil coordinates and static arrays."""
        # Normalised pupil radius rho in [0, 1]; 1 = pupil edge (sin(theta_imm) = NA/n_imm)
        rho_1d = np.linspace(-1.0, 1.0, self.N_pupil)
        RHO_X, RHO_Y = np.meshgrid(rho_1d, rho_1d)
        self._phi = np.arctan2(RHO_Y, RHO_X)      # (N, N) azimuthal angle
        self._rho = np.hypot(RHO_X, RHO_Y)         # (N, N) radius in [0, sqrt(2)]

        # Propagating aperture: rays that travel as plane waves in the sample.
        # sin(theta_sample) = rho * NA / n_medium; propagating when <= 1.
        # For NA > n_medium (e.g. 1.49 into water), rho_prop < 1.
        rho_prop = min(1.0, self.n_medium / self.NA)
        self._aperture = (self._rho <= rho_prop).astype(np.float64)  # (N, N)

        # sin/cos in immersion medium — always real within full pupil (NA/n_imm < 1)
        sin_imm = np.clip(self._rho * self.NA / self.n_immersion, 0.0, 1.0)
        self._cos_imm = np.sqrt(np.maximum(0.0, 1.0 - sin_imm ** 2))  # (N, N)

        # sin/cos in sample medium — imaginary beyond TIR; clip to propagating aperture
        sin_med = np.clip(self._rho * self.NA / self.n_medium, 0.0, 1.0)
        self._cos_med = np.sqrt(np.maximum(0.0, 1.0 - sin_med ** 2))  # (N, N)

        # Richards-Wolf apodization: A(rho) = 1 / sqrt(cos(theta_imm))
        # Prevents division by zero at the pupil edge
        safe_cos = np.where(self._cos_imm > 1e-6, self._cos_imm, 1e-6)
        self._apodization = (1.0 / np.sqrt(safe_cos)) * self._aperture  # (N, N)

    # ------------------------------------------------------------------
    # Padding
    # ------------------------------------------------------------------

    def _compute_padding(self, wavelength_um: float) -> int:
        """Padding per side so that FFT output matches the object-space pixel.

        After algebraic simplification (f_obj and p_bfp both cancel):
            N_total = wavelength * N_pupil / (pix_obj_um * 2 * NA)

        Returns:
            pad: Pixels of zero-padding on each side of the pupil.
        """
        N_total_exact = wavelength_um * self.N_pupil / (self.pix_obj_um * 2.0 * self.NA)
        N_total = int(np.round(N_total_exact))
        if N_total % 2 != 0:
            N_total += 1
        return max(0, (N_total - self.N_pupil) // 2)

    # ------------------------------------------------------------------
    # Pupil field
    # ------------------------------------------------------------------

    def _pupil_field(
        self,
        z_offsets_um: np.ndarray,
        wavelength_um: float,
        x0_um: float = 0.0,
        y0_um: float = 0.0,
        distance_from_coverslip_um: float = 0.0,
        phase_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Complex pupil field for all z-planes at one wavelength.

        Args:
            z_offsets_um: Axial offsets from focal plane (µm), shape (nz,).
            wavelength_um: Wavelength in µm.
            x0_um, y0_um: Sub-pixel lateral offsets in object space (µm).
            distance_from_coverslip_um: Depth of focal plane in sample (µm),
                used for Gibson-Lanni spherical aberration.
            phase_mask: Optional additional phase in pupil (N_pupil, N_pupil).

        Returns:
            E: complex128, shape (nz, N_pupil, N_pupil).
        """
        k0 = 2.0 * np.pi / wavelength_um

        # Lateral tip/tilt (spatial frequency k_x = NA * rho * cos(phi) / lambda,
        # invariant across media by Snell's law).
        # W_lateral(rho, phi) = k0 * NA * rho * (x0*cos(phi) + y0*sin(phi))
        phase_lateral = (
            k0
            * self.NA
            * self._rho
            * (x0_um * np.cos(self._phi) + y0_um * np.sin(self._phi))
        )  # (N, N)

        # Gibson-Lanni spherical aberration from depth d in sample:
        # W_SA = k0 * d * (n_medium * cos_theta_medium - n_immersion * cos_theta_imm)
        phase_SA = k0 * distance_from_coverslip_um * (
            self.n_medium * self._cos_med - self.n_immersion * self._cos_imm
        )  # (N, N)

        # Defocus phase (vectorised over z):
        # W_defocus(z0) = k0 * z0 * n_medium * cos_theta_medium
        # Shape: (nz, N, N) via broadcasting
        phase_defocus = (
            k0 * self.n_medium * self._cos_med[np.newaxis, :, :]
            * z_offsets_um[:, np.newaxis, np.newaxis]
        )  # (nz, N, N)

        # Static part of the phase (same for all z-planes)
        phase_static = phase_lateral + phase_SA  # (N, N)
        if phase_mask is not None:
            phase_static = phase_static + phase_mask

        # Assemble: E = apodization * exp(i*(phase_static + phase_defocus))
        E = self._apodization[np.newaxis, :, :] * np.exp(
            1j * (phase_static[np.newaxis, :, :] + phase_defocus)
        )  # (nz, N, N)

        return E

    # ------------------------------------------------------------------
    # Propagation
    # ------------------------------------------------------------------

    def _propagate(self, E: np.ndarray, pad: int) -> np.ndarray:
        """Fourier-propagate pupil field to image plane.

        Port of imageFormationParaxial.m.  For an isotropic emitter both
        polarisation components are identical (flat initial field), so
        I = 2|FFT(E)|^2; the factor of 2 cancels under normalisation.

        Args:
            E: complex pupil field, shape (..., N_pupil, N_pupil).
            pad: Zero-padding on each side.

        Returns:
            I: Intensity, shape (..., N_pupil + 2*pad, N_pupil + 2*pad).
        """
        n = E.ndim
        pad_spec = [(0, 0)] * (n - 2) + [(pad, pad), (pad, pad)]
        E_padded = np.pad(E, pad_spec)

        axes = (-2, -1)
        E_img = np.fft.fftshift(
            np.fft.fft2(np.fft.ifftshift(E_padded, axes=axes), axes=axes),
            axes=axes,
        )
        return np.abs(E_img) ** 2

    # ------------------------------------------------------------------
    # Crop
    # ------------------------------------------------------------------

    def _crop_center(self, I: np.ndarray) -> np.ndarray:
        """Crop central psf_size × psf_size region from intensity images.

        Args:
            I: shape (..., H, W).

        Returns:
            shape (..., psf_size, psf_size).
        """
        H, W = I.shape[-2], I.shape[-1]
        cy, cx = H // 2, W // 2
        half = self.psf_size // 2
        return I[
            ...,
            cy - half : cy - half + self.psf_size,
            cx - half : cx - half + self.psf_size,
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_psf_stack(
        self,
        z_offsets_um: np.ndarray,
        wavelengths_um: np.ndarray,
        spectral_weights: Optional[np.ndarray] = None,
        x0_um: float = 0.0,
        y0_um: float = 0.0,
        distance_from_coverslip_um: float = 0.0,
        phase_mask: Optional[np.ndarray] = None,
        normalise: str = "patch",
    ) -> np.ndarray:
        """Compute vectorial defocus PSF stack.

        Loops over wavelengths (each needs its own FFT padding), vectorises
        over z-planes within each wavelength via batched FFT.

        Args:
            z_offsets_um: Axial offsets from focal plane (µm), shape (nz,).
            wavelengths_um: Wavelength grid in µm, shape (n_lambda,).
            spectral_weights: Per-channel spectral weights, shape
                (n_lambda, n_channels), or None for monochromatic output.
                Obtain from :meth:`build_spectral_weights`.
            x0_um: Sub-pixel x offset in object space (µm).
            y0_um: Sub-pixel y offset in object space (µm).
            distance_from_coverslip_um: Depth of focal plane in sample (µm)
                for Gibson-Lanni spherical aberration.  0 = at coverslip.
            phase_mask: Optional engineered phase mask in pupil plane,
                shape (N_pupil, N_pupil).  None = standard (flat) PSF.
            normalise: ``"patch"`` (default) scales each patch to unit sum.
                ``"field"`` divides by the energy in the whole image plane
                instead, so a patch sums to the fraction of the PSF's energy
                that falls inside it (light in the tails beyond the patch is
                not redistributed into it). Monochromatic output only.

        Returns:
            If spectral_weights is None:
                shape (nz, n_lambda, psf_size, psf_size) — raw monochromatic
                PSFs, each normalised as set by ``normalise``.
            If spectral_weights given:
                shape (nz, n_channels, psf_size, psf_size) — spectrally
                integrated PSFs, each normalised to unit sum.
        """
        if normalise not in ("patch", "field"):
            raise ValueError(f"normalise must be 'patch' or 'field', not {normalise!r}")
        if normalise == "field" and spectral_weights is not None:
            raise ValueError("normalise='field' supports monochromatic output only")
        z_offsets_um = np.asarray(z_offsets_um, dtype=np.float64)
        wavelengths_um = np.asarray(wavelengths_um, dtype=np.float64)
        nz = len(z_offsets_um)
        n_lam = len(wavelengths_um)
        ps = self.psf_size

        # Accumulate: (n_lambda, nz, ps, ps)
        psf_mono = np.zeros((n_lam, nz, ps, ps), dtype=np.float64)

        for i_lam, lam in enumerate(wavelengths_um):
            pad = self._compute_padding(lam)
            E = self._pupil_field(
                z_offsets_um, lam, x0_um, y0_um,
                distance_from_coverslip_um, phase_mask,
            )  # (nz, N, N)
            I = self._propagate(E, pad)     # (nz, N_total, N_total)
            I_crop = self._crop_center(I)   # (nz, ps, ps)

            # Normalise each z-plane to unit sum, over the patch or the whole image plane
            norms = (I_crop if normalise == "patch" else I).sum(axis=(-2, -1), keepdims=True)
            norms = np.where(norms > 0, norms, 1.0)
            psf_mono[i_lam] = I_crop / norms

        # Reorder to (nz, n_lambda, ps, ps)
        psf_mono = psf_mono.transpose(1, 0, 2, 3)

        if spectral_weights is None:
            return psf_mono

        # Spectral integration: sum_lambda w_c(lambda) * PSF(lambda)
        # psf_mono:        (nz, n_lambda, ps, ps)
        # spectral_weights: (n_lambda, n_channels)
        # result:           (nz, n_channels, ps, ps)
        spectral_weights = np.asarray(spectral_weights, dtype=np.float64)
        psf_channels = np.einsum("zlyx,lc->zcyx", psf_mono, spectral_weights)

        # Normalise per (z, channel)
        norms = psf_channels.sum(axis=(-2, -1), keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        return psf_channels / norms

    @staticmethod
    def build_spectral_weights(
        spectral_functions,
        dye: str,
        filters: Optional[list],
        wavelengths_nm: np.ndarray,
        pixel_QYs: np.ndarray,
        include_objective: bool = True,
    ) -> np.ndarray:
        """Build per-channel spectral weights w_c(lambda) for PSF integration.

        Computes the unnormalised integrand of the polychromatic PSF:
            w_c(lambda) = S(lambda) * T_filter(lambda) * T_obj(lambda) * QE_c(lambda)

        This is the per-wavelength version of the quantity computed in aggregate
        by SpectralFunctions.get_absolute_pixel_QYs.  Passing these weights to
        compute_psf_stack gives a spectrally correct PSF per Bayer channel.

        Args:
            spectral_functions: Spectral_Funcs instance.
            dye: Dye name in spectral database.
            filters: Filter names to apply, or None.
            wavelengths_nm: Wavelength grid in nm, shape (n_lambda,).
                Must match the grid used for pixel_QYs.
            pixel_QYs: Per-channel quantum efficiencies,
                shape (n_channels, n_lambda), e.g. [B, G, R] ordering.
            include_objective: If True, multiply by objective transmission.

        Returns:
            weights: shape (n_lambda, n_channels), unnormalised.
        """
        import sys as _sys
        # Get SpectralDataType from the same module as the spectral_functions instance.
        # A bare "from SpectralFunctions import ..." gives a different class when the
        # caller used "from src import SpectralFunctions", breaking enum equality checks
        # inside get_spectral_data.  Try the instance's module first, then known paths.
        SpectralDataType = None
        for _mod_name in [
            type(spectral_functions).__module__,
            "src.SpectralFunctions",
            "SpectralFunctions",
        ]:
            _mod = _sys.modules.get(_mod_name)
            if _mod is not None and hasattr(_mod, "SpectralDataType"):
                SpectralDataType = _mod.SpectralDataType
                break
        if SpectralDataType is None:
            from pyS3M.SpectralFunctions import SpectralDataType  # type: ignore[import]

        # Filter transmission (unity if no filters)
        if filters is not None:
            filter_spectra = spectral_functions.get_spectral_data(
                filters, wavelengths_nm, SpectralDataType.FILTER
            )
            filter_tx = np.prod(filter_spectra, axis=0)
        else:
            filter_tx = np.ones(len(wavelengths_nm), dtype=np.float64)

        # Objective transmission
        if include_objective:
            obj_tx = spectral_functions.getobjectiveefficiency(wavelengths_nm)
        else:
            obj_tx = np.ones(len(wavelengths_nm), dtype=np.float64)

        # Dye emission spectrum, normalised to unit integral
        dye_spectrum = spectral_functions.get_spectral_data(
            [dye], wavelengths_nm, SpectralDataType.DYE
        )[0]  # (n_lambda,)
        total = np.trapz(dye_spectrum, x=wavelengths_nm)
        if total > 0:
            dye_spectrum = dye_spectrum / total

        # System weight at each wavelength
        system_weight = dye_spectrum * filter_tx * obj_tx  # (n_lambda,)

        # Per-channel weights: w_c(lambda) = system_weight(lambda) * QE_c(lambda)
        # pixel_QYs: (n_channels, n_lambda) → weights: (n_lambda, n_channels)
        weights = system_weight[:, np.newaxis] * pixel_QYs.T
        return weights
