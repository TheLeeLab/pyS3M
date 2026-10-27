"""High-NA PSF models: the scalar Debye model for defocus simulation, and a vectorial dipole model.

``VectorialPSF``: port of simulation-phasemask-psf-main (MATLAB) to Python/NumPy. Despite its name it
is the *scalar* high-NA approximation (Fazel et al. 2024, Rev. Mod. Phys. 96, 025003, Eq. 63): one
field component with aplanatic apodisation, no dipole projections or Fresnel coefficients, like the
MATLAB original. It implements:
  - Exact defocus phase (not paraxial approximation)
  - Depth-induced spherical aberration (Gibson-Lanni model)
  - Richards-Wolf apodization
  - Incoherent polychromatic spectral integration

At NA > n_sample (e.g. NA 1.49, n_water 1.33), only the propagating
aperture (rho <= n_sample/NA) contributes to far-field SMLM PSFs.

``DipolePSF``: the vectorial PSF of a freely rotating (isotropic) electric dipole in the sample
medium, imaged through the coverslip interface (Fazel et al. 2024, Eqs. 41, 48, 59-62), including
supercritical-angle light for an emitter at the interface. In focus only.
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


class DipolePSF:
    """Vectorial in-focus PSF of a freely rotating electric dipole (Fazel et al. 2024, Sec. III.C-D).

    The emitter sits in the sample medium (``n_medium``, e.g. water) a height ``z_dipole_um`` above
    the coverslip and is imaged by an aplanatic objective (``NA``, ``n_immersion``) focused on it.
    Its back-focal-plane field, sampled uniformly in the lateral wave vector q = n k0 sin(theta), is

        E_pupil = A(theta) * w/w_d * [ t_s (p.e_s) e_s' + t_p (p.e_p) e_p' ] * exp(i k0 z_d (n_d cos th_d - n cos th))

    with A = 1/sqrt(cos theta) the aplanatic factor (Eq. 41, image space n' = 1), t_s / t_p the
    Fresnel transmission coefficients of the sample/immersion interface, e_s / e_p the s and p unit
    vectors on the sample side (Eq. 61) rotated onto x / y in the pupil, and w/w_d =
    n cos(theta) / (n_d cos(theta_d)) the ratio of axial wave vectors. Eq. 59 prints the immersion
    medium's w in the Weyl integral; the dipole's own medium's w_d is needed for energy conservation
    across the interface, and the two agree when n_d = n (as in Fazel's Fig. 18). Beyond the
    critical angle cos(theta_d) is imaginary: that (supercritical-angle) light reaches the glass
    from emitters within ~100 nm of it, and the z term above makes it decay with height. The
    isotropic PSF is the mean of the x, y and z dipole images (Eq. 62), each |Ex|^2 + |Ey|^2.

    The image is evaluated by a matrix DFT on any sample-space grid, so its sampling is exact and
    independent of the pupil grid; the pupil grid is midpoint-sampled (symmetric, spacing
    2 / N_pupil). Validated in unit_tests/test_defocus_psf.py against Fazel Fig. 18, energy
    conservation through the interface, and the dipole's total power from the interface's
    reflection coefficients (supercritical light).

    Args:
        NA: Numerical aperture of the objective.
        n_medium: Refractive index of the medium the dipole is in (e.g. 1.33 for water).
        n_immersion: Refractive index of the immersion medium (e.g. 1.515 for oil).
        N_pupil: Pupil grid size (N_pupil x N_pupil).
        supercritical: Collect supercritical-angle light (sin(theta_d) > 1). False clips the pupil
            at the sample medium's critical angle, as VectorialPSF does.
        z_dipole_um: Emitter height above the coverslip (um), focus on the emitter.
    """

    def __init__(
        self,
        NA: float,
        n_medium: float,
        n_immersion: float,
        N_pupil: int = 512,
        supercritical: bool = True,
        z_dipole_um: float = 0.0,
    ) -> None:
        self.NA = NA
        self.n_medium = n_medium
        self.n_immersion = n_immersion
        self.N_pupil = N_pupil
        self.supercritical = supercritical
        self.z_dipole_um = z_dipole_um

        # Midpoint grid on [-1, 1] in units of the pupil edge (n_immersion sin theta = NA)
        self._u = -1.0 + (np.arange(N_pupil) + 0.5) * 2.0 / N_pupil
        UX, UY = np.meshgrid(self._u, self._u)
        rho = np.hypot(UX, UY)
        phi = np.arctan2(UY, UX)

        sin_imm = rho * NA / n_immersion
        self._cos_imm = np.sqrt(np.clip(1.0 - sin_imm ** 2, 0.0, None))
        self._sin_med = rho * NA / n_medium  # Snell: n_d sin th_d = n sin th
        self._cos_med = np.sqrt((1.0 - self._sin_med ** 2).astype(complex))  # imaginary if supercritical

        aperture = (rho <= 1.0) & (sin_imm < 1.0)
        if not supercritical:
            aperture &= self._sin_med <= 1.0
        self._aperture = aperture

        # Static amplitude: aplanatic factor x w/w_d x Fresnel, per s / p component
        c, cd, n, nd = self._cos_imm, self._cos_med, n_immersion, n_medium
        safe_c = np.where(aperture, c, 1.0)
        aplanatic = np.where(aperture, 1.0 / np.sqrt(safe_c), 0.0)
        if np.isclose(n, nd):
            weyl = np.ones_like(rho, dtype=complex)
            t_s = t_p = np.ones_like(rho, dtype=complex)
        else:
            safe_cd = np.where(np.abs(cd) > 1e-12, cd, 1e-12)
            weyl = (n * c) / (nd * safe_cd)
            t_s = 2 * nd * cd / (nd * cd + n * c)
            t_p = 2 * nd * cd / (n * cd + nd * c)
        amp = np.where(aperture, aplanatic * weyl, 0.0)

        # Pupil fields (Ex, Ey) for unit dipoles along x, y, z (sample-side s / p projections, Eq. 61)
        cos_phi, sin_phi = np.cos(phi), np.sin(phi)
        self._fields = []
        for p in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            E_s = t_s * (-p[0] * sin_phi + p[1] * cos_phi)
            E_p = t_p * (p[0] * cd * cos_phi + p[1] * cd * sin_phi - p[2] * self._sin_med)
            Ex = amp * (E_p * cos_phi - E_s * sin_phi)
            Ey = amp * (E_p * sin_phi + E_s * cos_phi)
            self._fields.append((Ex, Ey))

    def _height_phase(self, wavelength_um: float) -> np.ndarray:
        """exp(i k0 z (n_d cos th_d - n cos th)): emitter height, focus on the emitter."""
        if not self.z_dipole_um:
            return 1.0
        k0 = 2.0 * np.pi / wavelength_um
        return np.exp(1j * k0 * self.z_dipole_um
                      * (self.n_medium * self._cos_med - self.n_immersion * self._cos_imm))

    def _orientations(self, orientation: str) -> list:
        try:
            return {"x": [0], "y": [1], "z": [2], "iso": [0, 1, 2]}[orientation]
        except KeyError:
            raise ValueError(f"orientation must be 'iso', 'x', 'y' or 'z', not {orientation!r}") from None

    def image(
        self,
        wavelength_um: float,
        x_um: np.ndarray,
        y_um: np.ndarray,
        x0_um: float = 0.0,
        y0_um: float = 0.0,
        orientation: str = "iso",
    ) -> np.ndarray:
        """PSF intensity density (per um^2 of sample space, integrating to the collected fraction 1 over
        the whole image plane) at the grid points ``y_um`` (rows) x ``x_um`` (cols), emitter at (x0, y0).

        Returns:
            shape (len(y_um), len(x_um)).
        """
        k = 2.0 * np.pi * self.NA / wavelength_um  # lateral wave vector at the pupil edge
        Fx = np.exp(1j * k * np.outer(np.asarray(x_um, dtype=float) - x0_um, self._u))
        Fy = np.exp(1j * k * np.outer(np.asarray(y_um, dtype=float) - y0_um, self._u))
        h = self._height_phase(wavelength_um)
        idx = self._orientations(orientation)
        intensity = 0.0
        energy = 0.0
        for i in idx:
            for E in self._fields[i]:
                Eh = E * h
                intensity = intensity + np.abs(Fy @ Eh @ Fx.T) ** 2
                energy = energy + np.sum(np.abs(Eh) ** 2)
        # Parseval: the DFT image repeats with period L = 2 pi / (k du), and its integral over one
        # period is L^2 sum |E_pupil|^2.
        period = 2.0 * np.pi / (k * (self._u[1] - self._u[0]))
        return intensity / (period ** 2 * energy)

    def pixel_patches(
        self,
        wavelengths_um: np.ndarray,
        pixel_um: float,
        patch_px: int,
        oversample: int = 5,
        x0_um: float = 0.0,
        y0_um: float = 0.0,
        orientation: str = "iso",
    ) -> np.ndarray:
        """Pixel-integrated PSFs, one per wavelength: each camera pixel holds the probability that a
        collected photon lands in it (patches sum to < 1: the rest lands outside the patch).

        The emitter sits (x0, y0) from the centre of the patch's central pixel; each pixel is
        integrated by ``oversample`` (odd) x ``oversample`` midpoint sub-samples.

        Returns:
            shape (n_wavelengths, patch_px, patch_px).
        """
        if oversample % 2 != 1 or patch_px % 2 != 1:
            raise ValueError("oversample and patch_px must be odd, so the central sub-sample is the patch centre")
        n_fine = patch_px * oversample
        sub = pixel_um / oversample
        coords = (np.arange(n_fine) - n_fine // 2) * sub
        wavelengths_um = np.atleast_1d(np.asarray(wavelengths_um, dtype=float))
        out = np.empty((len(wavelengths_um), patch_px, patch_px))
        for i, lam in enumerate(wavelengths_um):
            fine = self.image(lam, coords, coords, x0_um, y0_um, orientation) * sub ** 2
            out[i] = fine.reshape(patch_px, oversample, patch_px, oversample).sum(axis=(1, 3))
        return out

    def collected_power(self, orientation: str = "iso", wavelength_um: float = 0.6,
                        supercritical_only: bool = False) -> float:
        """Collected power, in units of the dipole's total emission in a homogeneous sample medium
        (P0; for the isotropic emitter, the mean over x, y, z dipoles of unit strength).
        ``supercritical_only`` counts only light beyond the sample medium's critical angle."""
        idx = self._orientations(orientation)
        h = self._height_phase(wavelength_um)
        mask = self._aperture & (self._sin_med > 1.0) if supercritical_only else self._aperture
        du2 = (self._u[1] - self._u[0]) ** 2
        # |E_pupil|^2 dA_pupil = |E0|^2 dOmega in units where the pupil coordinate is n sin th / NA:
        # dA = (n / NA)^2 cos th dOmega, and A^2 = 1 / cos th. Power density in the immersion medium
        # is n |E0|^2; a free dipole of unit strength in the sample medium emits n_d * 8 pi / 3.
        conv = (self.NA / self.n_immersion) ** 2 * self.n_immersion / (self.n_medium * 8.0 * np.pi / 3.0)
        power = sum(np.sum((np.abs(E * h) ** 2)[mask]) for i in idx for E in self._fields[i])
        return float(power * du2 * conv / len(idx))
