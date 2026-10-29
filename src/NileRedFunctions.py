"""
Nile Red Spectral Model Functions

Forward and inverse models for predicting Nile Red emission properties (RGB intensities
and PSF widths) based on spectral parameters. Uses skew-Gaussian emission model fitted
from experimental data to extract central emission wavelength from localisation data.

Leverages SpectralFunctions for wavelength/energy conversions and spectral models,
and PSFFunctions for wavelength-dependent PSF calculations.

Author: jsb92
Date: October 7, 2025
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize_scalar
from typing import Any, Optional
from numpy.typing import NDArray
import pyS3M.SpectralFunctions as SpectralFunctions
import pyS3M.PSFFunctions as PSFFunctions
import pyS3M.IOFunctions as IOFunctions
from pyS3M.Constants import DriftConstants, AnalysisConfig
from pathlib import Path
import logging
logger = logging.getLogger(__name__)



class NileRed_Functions:
    """
    Nile Red spectral analysis and wavelength extraction.

    This class implements forward and inverse models for Nile Red emission:
    - Forward: central_wavelength → (R, G, B, σ_x, σ_y)
    - Inverse: (R, G, B, σ_x, σ_y) → central_wavelength

    Default spectral parameters from notebook 20251007_NileRedOptimiser:
    - sigma = 0.1630104 eV (Gaussian width in energy space)
    - alpha = -1.56453968 (skewness parameter)
    - wavelength_center = 617.6 nm (initial guess)

    Dependencies:

    - SpectralFunctions: For wavelength/energy conversions, skew-Gaussian models,
      filter/dye spectra, and pixel quantum efficiencies
    - PSFFunctions: For wavelength-dependent PSF width calculations
    """

    # Lookup table for wavelength_center_for_peak, precomputed at the default
    # sigma_energy/alpha (0.1630104 eV / -1.56453968) over peak_wavelength =
    # 550-680 nm in 1 nm steps, via the same numerical optimisation
    # wavelength_center_for_peak falls back to for any other (sigma_energy, alpha)
    # or out-of-range peak_wavelength. Regenerate (loop wavelength_center_for_peak
    # over np.arange(550, 681, 1.0) with a fine wavelength_array, e.g.
    # np.linspace(400, 900, 100000)) if sigma_energy/alpha are ever refit -- this
    # table is only valid for the exact (sigma_energy, alpha) pair below. Round-trip
    # accuracy against generate_nile_red_spectrum's actual argmax: max abs error
    # 0.023 nm across the grid (limited by the 100000-point wavelength_array used to
    # generate it, not by the 1 nm table spacing / linear interpolation).
    _PEAK_LUT_SIGMA_ENERGY = 0.1630104
    _PEAK_LUT_ALPHA = -1.56453968
    _PEAK_LUT_PEAK_WAVELENGTHS = np.arange(550.0, 681.0, 1.0)
    _PEAK_LUT_WAVELENGTH_CENTERS = np.array([
        534.8743, 535.8275, 536.7781, 537.7312, 538.6843, 539.6454, 540.5985,
        541.5517, 542.5049, 543.4581, 544.4114, 545.3647, 546.3198, 547.2730,
        548.2231, 549.1787, 550.1326, 551.0865, 552.0402, 552.9939, 553.9472,
        554.8993, 555.8519, 556.8062, 557.7468, 558.6981, 559.6643, 560.6181,
        561.5701, 562.5243, 563.4784, 564.4315, 565.3842, 566.3354, 567.2905,
        568.2405, 569.1962, 570.1486, 571.0985, 572.0512, 573.0045, 573.9575,
        574.9107, 575.8563, 576.8154, 577.7661, 578.7157, 579.6715, 580.6240,
        581.5750, 582.5281, 583.4816, 584.4339, 585.3853, 586.3368, 587.2878,
        588.2438, 589.1937, 590.1459, 591.0950, 592.0470, 593.0010, 593.9534,
        594.9027, 595.8550, 596.8086, 597.7599, 598.7107, 599.6636, 600.6156,
        601.5684, 602.5193, 603.4702, 604.4218, 605.3742, 606.3260, 607.2746,
        608.2282, 609.1796, 610.1264, 611.0834, 612.0346, 612.9812, 613.9371,
        614.8892, 615.8389, 616.7912, 617.7428, 618.6952, 619.6452, 620.5961,
        621.5468, 622.4976, 623.4481, 624.3753, 625.3275, 626.3016, 627.2522,
        628.2025, 629.1550, 630.1025, 631.0546, 632.0032, 632.9550, 633.9054,
        634.8597, 635.8056, 636.7566, 637.7115, 638.6591, 639.6099, 640.5611,
        641.5119, 642.4610, 643.4118, 644.3654, 645.3159, 646.2633, 647.2149,
        648.1649, 649.1161, 650.0659, 651.0167, 651.9668, 652.9181, 653.8717,
        654.8218, 655.7714, 656.7184, 657.6687, 658.6190,
    ])

    def __init__(
        self,
        camera: str = "ximea",
        pixel_size: float | None = None,
        sigma_energy: float = 0.1630104,
        alpha: float = -1.56453968,
        wavelength_center_init: float = 617.6,
        config: AnalysisConfig = None,
        mosaic_unit=None,
    ):
        """Initialize Nile Red model with spectral parameters.

        Args:
            camera: Camera model name (``"ximea"`` or ``"zwo"``). Sets pixel_size
                used when converting localisation coordinates to nm.
            pixel_size: Physical pixel size in µm. If None, taken from camera defaults.
            sigma_energy: Gaussian width in energy space (eV), default from fit
            alpha: Skewness parameter, default from fit
            wavelength_center_init: Initial guess for central wavelength (nm)
            config: AnalysisConfig controlling progress and logging callbacks.
            mosaic_unit: Mosaic pattern. If None, taken from camera defaults.
        """
        import pyS3M.CameraDefaults as CameraDefaults
        _cam = CameraDefaults.get_camera_config(camera)
        self.pixel_size = pixel_size if pixel_size is not None else _cam.pixel_size
        self.camera = camera
        self.mosaic_unit = mosaic_unit if mosaic_unit is not None else _cam.mosaic_unit
        self.config = config if config is not None else AnalysisConfig()

        self.default_sigma_energy = sigma_energy
        self.default_alpha = alpha
        self.default_wavelength_center = wavelength_center_init

        # Initialize SpectralFunctions, PSFFunctions, and IOFunctions
        self.spectral_funcs = SpectralFunctions.Spectral_Funcs(camera=camera)
        self.psf_funcs = PSFFunctions.PSF_Functions()
        self.io = IOFunctions.IO_Functions(camera=camera, mosaic_unit=self.mosaic_unit)

    def setup_optical_system(
        self, filter_names: list
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Setup optical system by loading filter spectra and pixel efficiencies.

        Args:
            filter_names: List of filter/dichroic names to load

        Returns:
            wavelength_array: Wavelength grid (nm)
            pixel_QYs: Pixel quantum yields [B, G, R] shape (3, n_wavelengths)
            filter_spectra: Filter transmission curves (n_filters, n_wavelengths)
        """
        # Get pixel quantum efficiencies
        R, G, B, wavelength = self.spectral_funcs.getpixelefficiency()
        pixel_QYs = np.vstack([B, G, R])

        # Get filter transmission spectra
        filter_spectra = self.spectral_funcs.get_dye_or_filter_data(
            names=filter_names, wavelength=wavelength, dye_or_filter=False
        )

        return wavelength, pixel_QYs, filter_spectra

    def generate_nile_red_spectrum(
        self,
        wavelength_center: float,
        wavelength_array: np.ndarray,
        sigma_energy: Optional[float] = None,
        alpha: Optional[float] = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """Generate Nile Red emission spectrum using skew-Gaussian model.

        Args:
            wavelength_center: Central emission wavelength (nm) - FIT PARAMETER
            wavelength_array: Wavelength grid for spectrum (nm)
            sigma_energy: Gaussian width in energy space (eV), default from __init__
            alpha: Skewness parameter, default from __init__
            normalize: If True, normalize spectrum to unit sum

        Returns:
            spectrum: Emission spectrum I(λ) at wavelength_array points
        """
        # Use defaults if not provided
        if sigma_energy is None:
            sigma_energy = self.default_sigma_energy
        if alpha is None:
            alpha = self.default_alpha

        # Convert center wavelength to energy
        energy_center = self.spectral_funcs.wavelength_to_energy(
            np.array([wavelength_center])
        )[0]

        # Convert wavelength array to energy
        energy_array = self.spectral_funcs.wavelength_to_energy(wavelength_array)

        # Create skew-Gaussian in energy space
        # Amplitude will be normalized later, so set to 1 for now
        params = np.array([1.0, energy_center, sigma_energy, alpha])
        spectrum_energy = self.spectral_funcs.skew_gaussian_model(params, energy_array)

        # Transform to wavelength space with Jacobian and dipole moment weighting
        # I(λ) = I(E) / (E^(-3) * λ^2)
        weighting_factor = energy_array ** (-3) * wavelength_array**2
        spectrum_wavelength = spectrum_energy / weighting_factor

        # Normalize to unit sum if requested
        if normalize:
            total = np.trapz(spectrum_wavelength, wavelength_array)
            if total > 0:
                spectrum_wavelength = spectrum_wavelength / total

        return spectrum_wavelength

    def spectral_centre_of_mass(
        self,
        wavelength_center: float,
        wavelength_array: np.ndarray,
        sigma_energy: Optional[float] = None,
        alpha: Optional[float] = None,
    ) -> float:
        """First-moment mean wavelength <λ> = ∫I(λ)λdλ / ∫I(λ)dλ of the skew-Gaussian
        emission spectrum located at `wavelength_center`.

        Because the emission spectrum is skewed, its centre of mass is never equal to
        `wavelength_center` (the energy-space location/fit parameter) -- the two are
        different, both legitimate, quantities. This is the one to report/compare as
        "the wavelength": it's what a real emission spectrum's mean actually is, and
        what `fit_nile_red_wavelength` returns. When checking simulated fit bias against
        a known true `wavelength_center`, compare against *this* (the true spectrum's own
        centre of mass), not against `wavelength_center` directly, or the comparison
        conflates a genuine fit bias with the expected, non-bias gap between a skewed
        distribution's location parameter and its mean.

        Args:
            wavelength_center: Energy-space location parameter (nm)
            wavelength_array: Wavelength grid (nm)
            sigma_energy: Gaussian width in energy space (eV), default from __init__
            alpha: Skewness parameter, default from __init__

        Returns:
            centre_of_mass: Mean wavelength of the spectrum (nm)
        """
        spectrum = self.generate_nile_red_spectrum(
            wavelength_center, wavelength_array, sigma_energy=sigma_energy, alpha=alpha, normalize=True
        )
        denom = np.trapz(spectrum, wavelength_array)
        if denom > 0:
            return float(np.trapz(spectrum * wavelength_array, wavelength_array) / denom)
        return float(wavelength_center)

    def wavelength_center_for_peak(
        self,
        peak_wavelength: float,
        wavelength_array: np.ndarray,
        sigma_energy: Optional[float] = None,
        alpha: Optional[float] = None,
        bounds: tuple[float, float] = (450.0, 800.0),
    ) -> float:
        """Find the `wavelength_center` that makes `generate_nile_red_spectrum`'s
        actual rendered peak land at `peak_wavelength`.

        `wavelength_center` is a *location* parameter of the underlying energy-space
        skew-Gaussian, not the peak of the rendered spectrum: the energy<->wavelength
        Jacobian, the dipole-moment (E^3) weighting, and the skew term (alpha) each
        shift the spectrum's actual mode away from `wavelength_center`, by an amount
        that itself varies with `wavelength_center` (it's not a fixed offset) -- e.g.
        for the default sigma_energy/alpha, wavelength_center=617.6 nm renders a
        spectrum that actually peaks at ~636.9 nm. There is no closed-form inverse for
        this, so this solves for it numerically (bounded 1-D optimisation): for a
        candidate wavelength_center, render the spectrum and find where it actually
        peaks (argmax), and adjust wavelength_center until that peak matches
        `peak_wavelength`.

        Uses the precomputed `_PEAK_LUT_*` table (linear interpolation) when
        `sigma_energy`/`alpha` match the values it was generated at and
        `peak_wavelength` falls within its 550-680 nm range -- effectively instant,
        vs. re-running the optimisation below from scratch. Falls back to the
        numerical optimisation for any other (sigma_energy, alpha) or out-of-range
        peak_wavelength.

        Args:
            peak_wavelength: Desired peak (mode) wavelength of the rendered spectrum (nm)
            wavelength_array: Wavelength grid used both to render candidate spectra and
                to locate their peak -- the returned peak is only as precise as this
                grid's spacing, so use a fine enough grid for the precision you need.
                Unused when the LUT path is taken.
            sigma_energy: Gaussian width in energy space (eV), default from __init__
            alpha: Skewness parameter, default from __init__
            bounds: Search range for wavelength_center (nm). Unused when the LUT
                path is taken.

        Returns:
            wavelength_center: Value to pass to generate_nile_red_spectrum so the
                rendered spectrum peaks at (approximately) peak_wavelength
        """
        sigma_energy_resolved = self.default_sigma_energy if sigma_energy is None else sigma_energy
        alpha_resolved = self.default_alpha if alpha is None else alpha

        lut_peaks = self._PEAK_LUT_PEAK_WAVELENGTHS
        if (
            np.isclose(sigma_energy_resolved, self._PEAK_LUT_SIGMA_ENERGY)
            and np.isclose(alpha_resolved, self._PEAK_LUT_ALPHA)
            and lut_peaks[0] <= peak_wavelength <= lut_peaks[-1]
        ):
            return float(np.interp(peak_wavelength, lut_peaks, self._PEAK_LUT_WAVELENGTH_CENTERS))

        def _peak_of(wavelength_center: float) -> float:
            spectrum = self.generate_nile_red_spectrum(
                wavelength_center, wavelength_array, sigma_energy=sigma_energy,
                alpha=alpha, normalize=False,
            )
            return float(wavelength_array[np.argmax(spectrum)])

        def _objective(wavelength_center: float) -> float:
            return (_peak_of(wavelength_center) - peak_wavelength) ** 2

        result = minimize_scalar(_objective, bounds=bounds, method="bounded")
        return float(result.x)

    def wavelength_center_for_centre_of_mass(
        self,
        target_centre_of_mass: float,
        wavelength_array: np.ndarray,
        sigma_energy: Optional[float] = None,
        alpha: Optional[float] = None,
        bounds: tuple[float, float] = (450.0, 800.0),
    ) -> float:
        """Find the `wavelength_center` whose spectral centre of mass equals
        `target_centre_of_mass` -- the inverse of :meth:`spectral_centre_of_mass`.

        `fit_nile_red_wavelength` reports the spectral *centre of mass*, not the raw
        `wavelength_center` location parameter (see `spectral_centre_of_mass`'s
        docstring) -- so simulating ground truth data meant to be *recovered at* a
        known wavelength (e.g. for a synthetic test fixture) needs this inverse, not
        `wavelength_center_for_peak` (which targets the rendered spectrum's peak/mode,
        a different quantity again). Solved numerically (bounded 1-D optimisation),
        the same pattern as `wavelength_center_for_peak`.

        Args:
            target_centre_of_mass: Desired centre-of-mass wavelength of the rendered
                spectrum (nm) -- i.e. the value `fit_nile_red_wavelength` should recover.
            wavelength_array: Wavelength grid used to render candidate spectra and
                compute their centre of mass.
            sigma_energy: Gaussian width in energy space (eV), default from __init__
            alpha: Skewness parameter, default from __init__
            bounds: Search range for wavelength_center (nm).

        Returns:
            wavelength_center: Value to pass to generate_nile_red_spectrum so the
                rendered spectrum's centre of mass is (approximately)
                target_centre_of_mass.
        """
        def _objective(wavelength_center: float) -> float:
            com = self.spectral_centre_of_mass(
                wavelength_center, wavelength_array, sigma_energy=sigma_energy, alpha=alpha,
            )
            return (com - target_centre_of_mass) ** 2

        result = minimize_scalar(_objective, bounds=bounds, method="bounded")
        return float(result.x)

    def apply_optical_filters(
        self, spectrum: np.ndarray, filter_spectra: np.ndarray
    ) -> np.ndarray:
        """Apply optical filter transmission curves to emission spectrum.

        Args:
            spectrum: Emission spectrum I(λ)
            filter_spectra: Filter transmission curves (n_filters, n_wavelengths)

        Returns:
            spectrum_filtered: Spectrum after passing through optical path
        """
        # Multiply spectrum by all filter transmissions
        # filter_spectra should be (n_filters, n_wavelengths)
        total_transmission = np.prod(filter_spectra, axis=0)
        spectrum_filtered = spectrum * total_transmission

        return spectrum_filtered

    def calculate_rgb_from_spectrum(
        self,
        spectrum_filtered: np.ndarray,
        wavelength: np.ndarray,
        pixel_QYs: np.ndarray,
    ) -> np.ndarray:
        """Calculate expected R, G, B intensities on Bayer sensor.

        Args:
            spectrum_filtered: Filtered emission spectrum
            wavelength: Wavelength array (nm)
            pixel_QYs: Pixel quantum yields [B, G, R] shape (3, n_wavelengths)

        Returns:
            rgb_predicted: [R, G, B] intensities (normalized to unit sum)
        """
        # Integrate spectrum weighted by pixel quantum efficiencies
        B_predicted = np.trapz(spectrum_filtered * pixel_QYs[0], wavelength)
        G_predicted = np.trapz(spectrum_filtered * pixel_QYs[1], wavelength)
        R_predicted = np.trapz(spectrum_filtered * pixel_QYs[2], wavelength)

        # Return as array [R, G, B]
        rgb_predicted = np.array([R_predicted, G_predicted, B_predicted])

        # Normalize to unit sum
        total = np.sum(rgb_predicted)
        if total > 0:
            rgb_predicted = rgb_predicted / total

        return rgb_predicted

    def calculate_psf_width_from_spectrum(
        self, spectrum_filtered: np.ndarray, wavelength: np.ndarray, NA: float = 1.49
    ) -> float:
        """Calculate expected PSF width from polychromatic spectrum using 1st moment.

        Uses the first spectral moment to calculate PSF width:
        σ_PSF = σ(<λ>) where <λ> = ∫ I(λ) λ dλ / ∫ I(λ) dλ

        This is ~1000x faster than the weighted average method and gives
        <0.15% difference in results (validated by comparison tests).

        Args:
            spectrum_filtered: Filtered emission spectrum
            wavelength: Wavelength array (nm)
            NA: Numerical aperture (default: 1.49)

        Returns:
            sigma_psf_predicted: Expected PSF width (nm)
        """
        # Calculate first spectral moment (mean wavelength)
        denominator = np.trapz(spectrum_filtered, wavelength)

        if denominator > 0:
            lambda_avg = (
                np.trapz(spectrum_filtered * wavelength, wavelength) / denominator
            )
            # Calculate PSF width at the mean wavelength
            sigma_psf_predicted = self.psf_funcs.sigma_PSF(lambda_avg, NA)
        else:
            # Fallback: use center of wavelength range
            lambda_avg = np.mean(wavelength)
            sigma_psf_predicted = self.psf_funcs.sigma_PSF(lambda_avg, NA)

        return sigma_psf_predicted

    def nile_red_forward_model(
        self,
        wavelength_center: float,
        filter_spectra: np.ndarray,
        wavelength_array: np.ndarray,
        pixel_QYs: np.ndarray,
        NA: float = 1.49,
    ) -> dict[str, float]:
        """Complete forward model: wavelength_center → (R, G, B, σ_PSF).

        Args:
            wavelength_center: Central emission wavelength (nm) - FIT PARAMETER
            filter_spectra: Optical filter transmission curves (n_filters, n_wavelengths)
            wavelength_array: Wavelength grid (nm)
            pixel_QYs: Bayer pixel quantum yields [B, G, R] shape (3, n_wavelengths)
            NA: Numerical aperture (default: 1.49)

        Returns:
            predictions: dict with keys 'R', 'G', 'B', 'sigma_x', 'sigma_y'
        """
        # 1. Generate emission spectrum
        spectrum = self.generate_nile_red_spectrum(
            wavelength_center, wavelength_array, normalize=True
        )

        # 2. Apply optical filters
        spectrum_filtered = self.apply_optical_filters(spectrum, filter_spectra)

        # 3. Predict RGB values
        rgb = self.calculate_rgb_from_spectrum(
            spectrum_filtered, wavelength_array, pixel_QYs
        )

        # 4. Predict PSF width (assume circular PSF: σ_x = σ_y)
        sigma_psf = self.calculate_psf_width_from_spectrum(
            spectrum_filtered, wavelength_array, NA
        )

        predictions = {
            "R": rgb[0],
            "G": rgb[1],
            "B": rgb[2],
            "sigma_x": sigma_psf,
            "sigma_y": sigma_psf,
        }

        return predictions

    def residuals_nile_red(
        self,
        wavelength_center: np.ndarray,
        observed_data: dict[str, float],
        errors: dict[str, float],
        filter_spectra: np.ndarray,
        wavelength_array: np.ndarray,
        pixel_QYs: np.ndarray,
        NA: float = 1.49,
    ) -> NDArray[np.float64]:
        """Residual vector for least-squares fitting of Nile Red wavelength.

        This function returns the vector of weighted residuals rather than chi-squared,
        which is more appropriate for least-squares optimization algorithms.

        Args:
            wavelength_center: Central wavelength parameter (1D array with single element)
            observed_data: dict with 'R', 'G', 'B', 'sigma_x', 'sigma_y'
            errors: dict with same keys as observed_data
            filter_spectra: Optical filter transmission curves
            wavelength_array: Wavelength grid (nm)
            pixel_QYs: Pixel quantum yields
            NA: Numerical aperture (default: 1.49)

        Returns:
            residuals: Array of weighted residuals (observation - prediction) / error
        """
        # Extract scalar wavelength (least_squares passes 1D array)
        wl = (
            wavelength_center[0]
            if isinstance(wavelength_center, np.ndarray)
            else wavelength_center
        )

        # Get predictions from forward model
        predictions = self.nile_red_forward_model(
            wl, filter_spectra, wavelength_array, pixel_QYs, NA
        )

        # Build residual vector
        residuals = []
        for key in ["R", "G", "B", "sigma_x", "sigma_y"]:
            if key in observed_data and key in errors:
                if errors[key] > 0:
                    residual = (observed_data[key] - predictions[key]) / errors[key]
                    residuals.append(residual)

        return np.array(residuals)

    def _error_inflation_factor(self, snr: float) -> float:
        """Calculate error inflation factor based on SNR.

        At low SNR, fit errors systematically underestimate true uncertainty
        due to noise floor, non-negativity constraints, and normalization bias.
        This function provides empirical inflation factors to correct for this.

        Args:
            snr: Signal-to-noise ratio

        Returns:
            inflation_factor: Multiplicative factor for error (>= 1.0)
        """
        if snr < 2.0:
            return 3.0
        elif snr < 5.0:
            return 2.0
        elif snr < 10.0:
            return 1.5
        else:
            return 1.0

    def _calculate_channel_snr(
        self, observed_rgb: np.ndarray, total_photons: float, background_photons: float
    ) -> np.ndarray:
        """Calculate signal-to-noise ratio for each RGB channel.

        SNR = S / sqrt(S + B) where:
        - S = signal photons in channel
        - B = background photons in channel

        Args:
            observed_rgb: [R, G, B] measured intensities (normalized or absolute)
            total_photons: Total signal photon count
            background_photons: Total background photons (distributed evenly across RGB)

        Returns:
            snr_array: [SNR_R, SNR_G, SNR_B]
        """
        # Normalize RGB if needed
        rgb_fractions = observed_rgb / np.sum(observed_rgb)

        # Calculate signal photons per channel
        signal_photons = rgb_fractions * total_photons

        # Background photons per channel (assume uniform distribution)
        background_per_channel = background_photons / 3.0

        # Calculate SNR for each channel
        snr_array = signal_photons / np.sqrt(signal_photons + background_per_channel)

        return snr_array

    @staticmethod
    def _normalize_rgb_with_errors(
        rgb: np.ndarray, rgb_err: np.ndarray
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Normalize RGB values to unit sum and propagate errors via quadrature.

        Args:
            rgb: [R, G, B] intensities (unnormalized)
            rgb_err: [R_err, G_err, B_err] errors on intensities

        Returns:
            rgb_norm: [R, G, B] normalized to unit sum
            rgb_norm_err: propagated errors on normalized values
        """
        rgb_total = np.sum(rgb)
        rgb_norm = rgb / rgb_total

        total_err = np.linalg.norm(rgb_err)
        rgb_norm_err = np.array([
            (
                rgb_norm[i] * np.sqrt(
                    (rgb_err[i] / rgb[i]) ** 2 + (total_err / rgb_total) ** 2
                )
                if rgb[i] > 0
                else 1e-3
            )
            for i in range(3)
        ])

        return rgb_norm, rgb_norm_err

    @staticmethod
    def _weighted_average_with_error(
        values: np.ndarray, errors: np.ndarray
    ) -> tuple[float, float]:
        """Compute inverse-error-weighted average and propagated error.

        Args:
            values: Array of measurements
            errors: Array of measurement errors (must be > 0)

        Returns:
            weighted_avg: Weighted average of values
            propagated_err: Propagated error = 1 / sqrt(sum(1/err^2))
        """
        weights = 1.0 / errors
        weighted_avg = np.average(values, weights=weights)
        propagated_err = 1.0 / np.sqrt(np.sum(1.0 / errors ** 2))
        return weighted_avg, propagated_err

    def _parallel_fit_wavelengths(
        self,
        fit_args: list,
        n_workers: int,
        verbose: bool = True,
        label: str = "Fitting",
        progress_interval: int = 100,
    ) -> list[tuple[float, float]]:
        """Run wavelength fits in parallel with progress tracking.

        Args:
            fit_args: List of argument tuples for _fit_nile_red_wavelength_standalone
            n_workers: Number of parallel workers
            verbose: Print progress messages
            label: Label for progress output
            progress_interval: Print progress every N completions

        Returns:
            results: List of (wavelength, wavelength_error) tuples, one per input.
                     Failed fits return (NaN, NaN).
        """
        import time
        from concurrent import futures

        n_total = len(fit_args)
        results = [(np.nan, np.nan)] * n_total

        if verbose:
            msg = f"{label}: {n_total} tasks with {n_workers} workers..."
            logger.info(msg)
            if self.config.logging_callback:
                self.config.logging_callback(msg)
            start_time = time.time()

        with futures.ProcessPoolExecutor(n_workers) as executor:
            future_list = [
                executor.submit(_fit_nile_red_wavelength_standalone, *args)
                for args in fit_args
            ]

            completed = 0
            for idx, future in enumerate(future_list):
                try:
                    wl, wl_err = future.result(timeout=30)
                    results[idx] = (wl, wl_err)
                    completed += 1

                    if verbose and (
                        completed % progress_interval == 0
                        or completed == n_total
                    ):
                        elapsed = time.time() - start_time
                        rate = completed / elapsed if elapsed > 0 else 0
                        msg = (f"  {label} progress: {completed}/{n_total} "
                               f"({100*completed/n_total:.1f}%) - {rate:.1f} fits/s")
                        logger.debug(msg)
                        if self.config.progress_callback:
                            self.config.progress_callback(completed / n_total, msg)
                except Exception as e:
                    if verbose:
                        msg = f"\nWarning: {label} fit failed for index {idx}: {e}"
                        logger.info(msg)
                        if self.config.logging_callback:
                            self.config.logging_callback(msg)

        if verbose:
            elapsed = time.time() - start_time
            msg = f"\n  {label} complete: {completed}/{n_total} in {elapsed:.1f} s"
            logger.info(msg)
            if self.config.logging_callback:
                self.config.logging_callback(msg)

        return results

    def fit_nile_red_wavelength(
        self,
        observed_rgb: np.ndarray,
        observed_sigma_x: float,
        observed_sigma_y: float,
        rgb_errors: np.ndarray,
        sigma_x_error: float,
        sigma_y_error: float,
        filter_spectra: np.ndarray,
        wavelength_array: np.ndarray,
        pixel_QYs: np.ndarray,
        NA: float = 1.49,
        wavelength_bounds: tuple[float, float] = (500.0, 750.0),
        total_photons: Optional[float] = None,
        background_photons: float = 40.0,
        apply_snr_inflation: bool = True,
        wavelength_initial_guess: Optional[float] = None,
    ) -> tuple[float, dict[str, float]]:
        """Fit central wavelength of Nile Red emission from experimental data.

        Uses all three RGB channels for wavelength fitting. Implements SNR-based error
        inflation to correct for underestimated errors at low photon counts:
        - SNR < 2: inflate by 3.0x
        - SNR 2-5: inflate by 2.0x
        - SNR 5-10: inflate by 1.5x
        - SNR > 10: use as-is (1.0x)

        Args:
            observed_rgb: [R, G, B] measured intensities
            observed_sigma_x, observed_sigma_y: Measured PSF widths (nm)
            rgb_errors: Errors on [R, G, B]
            sigma_x_error, sigma_y_error: Errors on PSF widths (nm)
            filter_spectra: Optical filter transmission curves
            wavelength_array: Wavelength grid (nm)
            pixel_QYs: Pixel quantum yields
            NA: Numerical aperture (default: 1.49)
            wavelength_bounds: Search range for wavelength (nm)
            total_photons: Total photon count for SNR calculation (optional)
            background_photons: Background photon count (default: 40.0, distributed across RGB)
            apply_snr_inflation: Apply SNR-based error inflation (default: True)
            wavelength_initial_guess: Custom initial guess for wavelength (nm).
                If None, uses self.default_wavelength_center (617.6 nm).

        Returns:
            wavelength_center: Fitted central wavelength (nm)
            predictions: Predicted values at best-fit wavelength
        """
        # Normalize RGB to unit sum
        observed_rgb_norm = observed_rgb / np.sum(observed_rgb)
        rgb_errors_norm = rgb_errors / np.sum(observed_rgb)

        # Apply SNR-based error inflation if requested and total_photons is provided
        if apply_snr_inflation and total_photons is not None:
            # Calculate SNR for each channel
            snr_rgb = self._calculate_channel_snr(
                observed_rgb, total_photons, background_photons
            )

            # Apply inflation factors
            inflation_factors = np.array(
                [
                    self._error_inflation_factor(snr_rgb[0]),  # R
                    self._error_inflation_factor(snr_rgb[1]),  # G
                    self._error_inflation_factor(snr_rgb[2]),  # B
                ]
            )

            # Inflate errors
            rgb_errors_norm = rgb_errors_norm * inflation_factors

        # Use all three RGB channels for wavelength fitting
        observed_data = {
            "R": observed_rgb_norm[0],
            "G": observed_rgb_norm[1],
            "B": observed_rgb_norm[2],
            "sigma_x": observed_sigma_x,
            "sigma_y": observed_sigma_y,
        }

        errors = {
            "R": rgb_errors_norm[0],
            "G": rgb_errors_norm[1],
            "B": rgb_errors_norm[2],
            "sigma_x": sigma_x_error,
            "sigma_y": sigma_y_error,
        }

        # Initial guess: use custom guess, default central wavelength, or midpoint of bounds
        if wavelength_initial_guess is not None:
            x0 = np.array([wavelength_initial_guess])
        else:
            x0 = np.array([self.default_wavelength_center])
        if x0[0] < wavelength_bounds[0] or x0[0] > wavelength_bounds[1]:
            x0 = np.array([(wavelength_bounds[0] + wavelength_bounds[1]) / 2])

        # Fit using Trust Region Reflective algorithm (handles bounds well)
        result = least_squares(
            fun=self.residuals_nile_red,
            x0=x0,
            bounds=(wavelength_bounds[0], wavelength_bounds[1]),
            method="trf",  # Trust Region Reflective
            args=(
                observed_data,
                errors,
                filter_spectra,
                wavelength_array,
                pixel_QYs,
                NA,
            ),
        )

        wavelength_center = result.x[0]

        # Report the spectral centre of mass, not the raw energy-space location
        # parameter -- see spectral_centre_of_mass's docstring. IMPORTANT for anything
        # that checks bias against a known simulated wavelength: a noiseless roundtrip
        # (generate a spectrum at a known wavelength_center, fit it back with zero
        # noise) shows the raw TRF fit recovers wavelength_center with 0.00 nm bias,
        # while the *centre of mass* legitimately differs from wavelength_center by
        # +28 to +41 nm across 560-680 nm (a skewed distribution's mean is never equal
        # to its location parameter). That gap is not fit error -- so bias must be
        # computed as (fitted centre of mass) - (true spectrum's own centre of mass),
        # both via spectral_centre_of_mass, not against the raw wavelength_center/
        # nile_red_wavelength ground-truth value directly (see simulate_wavelength_precision).
        wavelength_mean = self.spectral_centre_of_mass(wavelength_center, wavelength_array)

        # Estimate wavelength error from Jacobian
        # s2 = residual variance, cov = s2 * inv(J^T J)
        J = result.jac          # (n_data, 1)
        n_data = len(result.fun)
        n_params = 1
        dof = max(n_data - n_params, 1)
        s2 = np.sum(result.fun**2) / dof
        JtJ = float(J.T @ J)
        if JtJ > 0:
            wavelength_error = np.sqrt(s2 / JtJ)
        else:
            wavelength_error = np.nan

        # Get predictions at best fit
        predictions = self.nile_red_forward_model(
            wavelength_center, filter_spectra, wavelength_array, pixel_QYs, NA
        )
        predictions["wavelength_error"] = wavelength_error

        return wavelength_mean, predictions

    def simulate_wavelength_precision(
        self,
        save_folder: str,
        wavelength_range: tuple[float, float] = (560.0, 620.0),
        wavelength_step: float = 5.0,
        photon_counts: np.ndarray = np.array([1000, 2000, 5000, 10000]),
        n_bootstrap: int = 1000,
        filter_names: Optional[list] = None,
        NA: float = 1.49,
        pixel_size: float | None = None,  # nm; None → self.pixel_size * 1000
        camera_parameters: Optional[dict] = None,
        image_size: int = 16,
        smoothing_function=None,
        background_photons: float = 40.0,
        starting_flag: str = "",
        save_raw_results: bool = True,
        cpu_fraction: float = 0.9,
        verbose: bool = True,
        use_tqdm: bool = False,
    ) -> None:
        """Simulate wavelength precision using two-stage workflow.

        Stage 1: Use Multicolour_Simulation_Functions to simulate images and fit them
        Stage 2: Post-process fit results to extract wavelengths using inverse model

        This method generates Nile Red spectra at different wavelengths, simulates
        imaging and fitting using existing infrastructure, then extracts wavelengths
        from the RGB+PSF fit results.

        `wavelength_range`/`wavelength_step` sweep the desired PEAK (mode) wavelength
        of each simulated spectrum, not the skew-Gaussian's energy-space location
        parameter -- the two differ by tens of nm (see
        `wavelength_center_for_peak`'s docstring). Internally, each sweep value is
        converted via `wavelength_center_for_peak` to the location parameter that
        actually renders a spectrum peaking there, *before* that spectrum is handed
        to the simulator (which applies the optical filters itself when generating
        images/RGB/PSF). Stage 2's bias reference is computed from that same
        converted location parameter (via `spectral_centre_of_mass`, the raw/
        un-filtered spectrum's own mean) -- using the raw peak value directly instead
        would introduce a ~20 nm spurious offset into the reported bias, dwarfing any
        genuine fit bias.

        Args:
            save_folder: Directory to save results (created if doesn't exist)
            wavelength_range: (min, max) PEAK wavelength range in nm (default: 560-620)
            wavelength_step: Wavelength step size in nm (default: 5)
            photon_counts: Array of photon counts to simulate (default: [1k, 2k, 5k, 10k])
            n_bootstrap: Number of Monte Carlo realizations per condition (default: 1000)
            filter_names: List of filter/dichroic names (default: Nile Red filters)
            NA: Numerical aperture (default: 1.49)
            pixel_size: Camera pixel size in nm (default: 69)
            camera_parameters: Camera calibration dict (default: ideal camera)
            image_size: Size of simulated images in pixels (default: 20)
            smoothing_function: Smoothing function for PSF (default: gaussian_filter)
            background_photons: Background photon count (default: 20)
            starting_flag: Prefix for saved files (default: "")
            save_raw_results: If True, save raw fit results (default: True)
            cpu_fraction: Fraction of CPUs to use for parallel processing
            verbose: Print progress messages (default: True)
            use_tqdm: Use tqdm for Jupyter-compatible progress bars (default: False)

        Saves per wavelength:
            - Standard simulation outputs from test_fit_method (RMSE_mean, RMSE_std, etc.)
            - wavelength_precision_summary.csv: Extracted wavelength statistics
        """
        import polars as pl
        if pixel_size is None:
            pixel_size = self.pixel_size * 1000  # µm → nm

        import time
        import pyS3M.Multicolour_Simulation_Functions as Multicolour_Simulation_Functions
        import pyS3M.MaskFunctions as MaskFunctions

        # Create save folder if it doesn't exist
        Path(save_folder).mkdir(parents=True, exist_ok=True)

        # Default filter configuration for Nile Red
        if filter_names is None:
            filter_names = [
                "semrock-ff01-650-200",
                "semrock-di03-r514-t1-25x36",
                "semrock-ff01-515-lp",
            ]

        # Setup optical system
        wavelength_array, pixel_QYs, _ = self.setup_optical_system(filter_names)

        # Setup default camera parameters if not provided
        if camera_parameters is None:
            M_F = MaskFunctions.Mask_Functions(camera=self.camera, mosaic_unit=self.mosaic_unit)
            masks = M_F.get_masks(size_x=image_size, size_y=image_size)

            # Use realistic camera parameters (median from calibrations)
            camera_parameters = {
                "gain": np.ones((image_size, image_size)) * 0.48,
                "offset": np.ones((image_size, image_size)) * 100.0,
                "variance": np.ones((image_size, image_size)) * 0.938,
                "readnoise": np.ones((image_size, image_size)) * 2.0,
                "rqe": np.ones((image_size, image_size)),
                "pixel_QYs": pixel_QYs,
                "pixel_order": ["B", "G", "R"],
                "pixel_order_indices": [0, 1, 2],
                "masks": masks,
            }

        # Initialize simulation
        MSF = Multicolour_Simulation_Functions.MultiC_Sim_Funcs()

        # Setup smoothing function
        if smoothing_function is None:
            import pyS3M.sCMOSFunctions as sCMOSFunctions
            import types

            sCMOS = sCMOSFunctions.sCMOS_Functions()
            smoothing_function = types.SimpleNamespace()
            smoothing_function.args = {"sigma": 1.5}
            smoothing_function.extent = 1.5
            smoothing_function.smoothing_function = sCMOS.gaussian_filter_stack
            smoothing_function.data_arg = "image"

        # Generate wavelength grid
        wavelengths_true = np.arange(
            wavelength_range[0], wavelength_range[1] + wavelength_step, wavelength_step
        )
        n_wavelengths = len(wavelengths_true)

        start_time = time.time()

        # Setup progress display
        if use_tqdm:
            try:
                from tqdm.auto import tqdm  # type: ignore
            except ImportError:
                if verbose:
                    logger.warning("Warning: tqdm not installed, falling back to print-based progress")
                use_tqdm = False

        if verbose:
            logger.info(f"\n{'='*60}")
            logger.info(f"Nile Red Wavelength Precision Simulation")
            logger.info(f"{'='*60}")
            logger.info(f"Wavelength range: {wavelength_range[0]}-{wavelength_range[1]} nm (step={wavelength_step} nm)")
            logger.info(f"Number of wavelengths: {n_wavelengths}")
            logger.info(f"Photon counts: {photon_counts}")
            logger.info(f"Bootstrap samples: {n_bootstrap}")
            logger.info(f"Save folder: {save_folder}")
            logger.info(f"{'='*60}\n")

        # STAGE 1: Simulate and fit images for each wavelength
        wavelength_iterator = enumerate(wavelengths_true)
        if use_tqdm and verbose:
            wavelength_iterator = tqdm(
                wavelength_iterator,
                total=n_wavelengths,
                desc="Stage 1: Simulating wavelengths",
            )

        for i, wl_true in wavelength_iterator:
            if self.config.progress_callback:
                self.config.progress_callback(i / n_wavelengths,
                    f"Stage 1: wavelength {i+1}/{n_wavelengths} ({wl_true:.1f} nm)")
            if verbose and not use_tqdm:
                elapsed = (time.time() - start_time) / 60.0
                msg = f"\n[{i+1}/{n_wavelengths}] Processing wavelength {wl_true:.1f} nm (elapsed: {elapsed:.1f} min)"
                logger.info(msg)
                if self.config.logging_callback:
                    self.config.logging_callback(msg)

            # wl_true is the desired PEAK wavelength, not the skew-Gaussian's
            # location parameter -- convert before generating the spectrum, so the
            # spectrum handed to the simulator actually peaks at wl_true (see
            # wavelength_center_for_peak's docstring; the gap is tens of nm).
            wavelength_center = self.wavelength_center_for_peak(wl_true, wavelength_array)
            spectrum = self.generate_nile_red_spectrum(
                wavelength_center, wavelength_array, normalize=True
            )

            # Create dye name for this wavelength (still keyed on the peak wl_true --
            # the human-meaningful sweep value -- for file naming/lookup below)
            dye_name = f"simulated_NileRed_{int(wl_true)}nm"

            # Run standard simulation using test_fit_method
            flag = f"{starting_flag}wl{int(wl_true)}_"

            # Create SimulationConfig
            import pyS3M.Multicolour_Simulation_Functions as MSF_module

            config = MSF_module.SimulationConfig(
                n_bootstrap=n_bootstrap,
                background_photons=background_photons,
                NA=NA,
                pixel_size=pixel_size,
                cpu_fraction=cpu_fraction,
                save_raw_results=save_raw_results,
                subtractx0y0=False,
                saverawimages=False,
            )

            MSF.test_fit_method(
                dye=dye_name,
                filters=filter_names,
                wavelength=wavelength_array,
                camera_parameters=camera_parameters,
                save_folder=save_folder,
                n_photon_space=photon_counts,
                smoothing_function=smoothing_function,
                starting_flag=flag,
                config=config,
                single_dye_spectrum=spectrum,  # Pass spectrum directly
                nile_red_wavelength=wl_true,  # Pass wavelength for inverse fitting
            )

        if verbose:
            total_elapsed = (time.time() - start_time) / 60.0
            logger.info(f"\n{'='*60}")
            logger.info(f"Stage 1 complete (simulation + wavelength fitting): {total_elapsed:.1f} min")
            logger.info(f"{'='*60}\n")
            logger.info("Starting Stage 2: Calculate statistics from fitted wavelengths...")

        # STAGE 2: Calculate statistics from wavelength columns in raw results
        wavelength_precision_results = []

        stats_iterator = enumerate(wavelengths_true)
        if use_tqdm and verbose:
            stats_iterator = tqdm(
                stats_iterator,
                total=n_wavelengths,
                desc="Stage 2: Calculating statistics",
            )

        for i, wl_true in stats_iterator:
            if verbose and not use_tqdm:
                logger.debug(f"  [{i+1}/{n_wavelengths}] Processing statistics for {wl_true:.1f} nm")

            flag = f"{starting_flag}wl{int(wl_true)}_"

            # Find raw results h5 files for this wavelength
            raw_files = [
                p.name
                for p in Path(save_folder).iterdir()
                if p.name.startswith(flag) and p.name.endswith("rawresults.h5")
            ]

            for raw_file in raw_files:
                # Load h5 file using pandas then convert to polars
                import pandas as pd

                file_path = Path(save_folder) / raw_file
                df_pandas = self.io.read_h5_database(file_path)
                df = pl.from_pandas(df_pandas)

                # Check if wavelength columns exist
                if "wl_fit" not in df.columns:
                    if verbose:
                        logger.info(f"\nWarning: No 'wl_fit' column found in {raw_file}")
                        logger.info("This may be from an older simulation. Re-run simulation to add wavelength fits.")
                    continue

                # Process each photon level separately
                if "photon_level" not in df.columns:
                    if verbose:
                        logger.info(f"\nWarning: No 'photon_level' column found in {raw_file}")
                    continue

                # wl_fit is the fitted spectral centre of mass of the RAW/un-filtered
                # spectrum (see fit_nile_red_wavelength/spectral_centre_of_mass),
                # evaluated at whatever location parameter the LSQ fit recovers.
                # wl_true, however, is the desired PEAK wavelength (see Stage 1 above)
                # -- not the location parameter, and not the raw spectrum's mean
                # either. To compare like-for-like, convert wl_true -> the location
                # parameter that actually renders a spectrum peaking there (the same
                # conversion, and the same resulting value, used in Stage 1 to
                # generate the simulated spectrum), then take *that* location
                # parameter's raw-spectrum centre of mass as the truth reference.
                # Skipping the peak->location-parameter conversion here (comparing
                # against spectral_centre_of_mass(wl_true, ...) directly) would
                # introduce a ~20 nm spurious offset into every reported bias --
                # confirmed empirically, not a small effect -- dwarfing any genuine
                # fit bias this sweep is meant to measure.
                wavelength_center_true = self.wavelength_center_for_peak(wl_true, wavelength_array)
                wl_true_com = self.spectral_centre_of_mass(wavelength_center_true, wavelength_array)

                for level in df["photon_level"].unique():
                    df_level = df.filter(pl.col("photon_level") == level)

                    # Get mean photon count for this level
                    n_photons = df_level["photons"].mean()

                    # Extract fitted wavelengths (excluding NaN values)
                    wavelengths_fitted = df_level["wl_fit"].to_numpy()
                    wavelengths_fitted = wavelengths_fitted[~np.isnan(wavelengths_fitted)]

                    # Calculate statistics
                    if len(wavelengths_fitted) > 0:
                        precision = np.std(wavelengths_fitted)
                        bias = np.mean(wavelengths_fitted) - wl_true_com
                        recovery_rate = len(wavelengths_fitted) / n_bootstrap

                        wavelength_precision_results.append(
                            {
                                "wavelength_true": wl_true,
                                "n_photons": n_photons,
                                "wavelength_precision": precision,
                                "wavelength_bias": bias,
                                "wavelength_mean": np.mean(wavelengths_fitted),
                                "recovery_rate": recovery_rate,
                                "n_successful": len(wavelengths_fitted),
                            }
                        )

        # Save wavelength precision summary
        if len(wavelength_precision_results) > 0:
            summary_df = pl.DataFrame(wavelength_precision_results)
            summary_file = Path(save_folder) / f"{starting_flag}wavelength_precision_summary.csv"
            summary_df.write_csv(summary_file)

            if verbose:
                logger.info(f"\n\n{'='*60}")
                logger.info(f"Simulation complete!")
                logger.info(f"Total time: {(time.time() - start_time) / 60.0:.1f} min")
                logger.info(f"Results saved to: {save_folder}")
                logger.info(f"Wavelength precision summary: {summary_file}")
                logger.info(f"{'='*60}\n")

    def fit_wavelengths_from_h5(
        self,
        h5_path: str,
        filter_names: list[str],
        camera_parameters: dict,
        wavelength_bounds: tuple[float, float] = (500.0, 750.0),
        NA: float = 1.49,
        pixel_size: float | None = None,  # nm; None → self.pixel_size * 1000
        output_path: Optional[str] = None,
        cpu_fraction: float = 0.9,
        verbose: bool = True,
        aggregate_id_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fit Nile Red wavelengths from localisations stored in HDF5 file.

        Convenience function that loads an HDF5 file containing localisation data
        (RGB intensities, PSF widths, and errors), fits the Nile Red wavelength
        for each localisation using parallel processing, and returns/saves the
        updated DataFrame with wavelength columns added.

        When aggregate_id_column is provided, uses a two-step fitting approach:

        1. Compute weighted-average A_R, A_G, A_B, s_x, s_y per aggregate and
           fit wavelength for each aggregate (higher SNR → more stable fits)
        2. Use each aggregate's fitted wavelength as the initial guess when
           fitting individual localisations within that aggregate

        Args:
            h5_path: Path to HDF5 file containing localisation data
            filter_names: List of filter/dichroic names used in optical path
            camera_parameters: Camera parameters dict containing:
                - 'pixel_QYs': Pixel quantum yields vs wavelength (if 'wavelength' not provided)
                - 'wavelength': Wavelength array (nm) - optional if pixel_QYs shape implies it
            wavelength_bounds: Search range for wavelength fitting (nm), default: (500, 750)
            NA: Numerical aperture, default: 1.49
            pixel_size: Camera pixel size in nm, default: 69.0
            output_path: Optional path to save updated HDF5 file (if None, doesn't save)
            cpu_fraction: Fraction of CPUs to use for parallel fitting, default: 0.9
            verbose: Print progress messages, default: True
            aggregate_id_column: Column name containing aggregate/punctum IDs (e.g. 'cluster_id').
                When provided, enables two-step fitting with aggregate-level priors.
                When None (default), all fits use the default initial guess.

        Returns:
            pd.DataFrame: Updated DataFrame with 'wl_fit' and 'wl_fit_err' columns added.
                When aggregate_id_column is provided, also adds 'wl_fit_aggregate' column
                containing the per-aggregate fitted wavelength.

        Required columns in HDF5 file:
            - A_R, A_G, A_B: RGB amplitudes (normalized)
            - s_x, s_y: PSF widths (pixels)
            - A_R_err, A_G_err, A_B_err: RGB amplitude errors
            - s_x_err, s_y_err: PSF width errors (pixels)
            - photons: Total photon count (for SNR calculation)
            - background_photons: Background photon count (for SNR calculation)

        Example:
            >>> import NileRedFunctions
            >>> nrf = NileRedFunctions.NileRed_Functions()
            >>>
            >>> # Define optical configuration
            >>> filters = [
            ...     "semrock-ff01-650-200",
            ...     "semrock-di03-r514-t1-25x36",
            ...     "semrock-ff01-515-lp",
            ... ]
            >>>
            >>> # Setup camera parameters (or load from calibration)
            >>> import SpectralFunctions
            >>> sf = SpectralFunctions.Spectral_Funcs()
            >>> R, G, B, wavelength = sf.getpixelefficiency()
            >>> pixel_QYs = np.vstack([B, G, R])
            >>> camera_params = {
            ...     'pixel_QYs': pixel_QYs,
            ...     'wavelength': wavelength,
            ... }
            >>>
            >>> # Fit wavelengths with aggregate priors
            >>> df_with_wavelengths = nrf.fit_wavelengths_from_h5(
            ...     h5_path='results_aggregatelocs.h5',
            ...     filter_names=filters,
            ...     camera_parameters=camera_params,
            ...     output_path='results_aggregatelocs.h5',
            ...     aggregate_id_column='cluster_id',
            ... )
        """
        if pixel_size is None:
            pixel_size = self.pixel_size * 1000  # µm → nm

        import pandas as pd
        import multiprocessing

        if verbose:
            logger.info(f"\n{'='*60}")
            logger.info(f"Fitting Nile Red Wavelengths from HDF5")
            logger.info(f"{'='*60}")
            logger.info(f"Input file: {h5_path}")
            logger.info(f"Wavelength bounds: {wavelength_bounds[0]}-{wavelength_bounds[1]} nm")
            logger.info(f"{'='*60}\n")

        # Load HDF5 file
        if not Path(h5_path).exists():
            raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

        if verbose:
            logger.info("Loading HDF5 file...")

        df = self.io.read_h5_database(h5_path)
        n_locs = len(df)

        if verbose:
            logger.info(f"Loaded {n_locs} localisations")

        # Check required columns
        required_cols = [
            "A_R",
            "A_G",
            "A_B",
            "s_x",
            "s_y",
            "A_R_err",
            "A_G_err",
            "A_B_err",
            "s_x_err",
            "s_y_err",
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"HDF5 file missing required columns: {missing_cols}\n"
                f"Available columns: {list(df.columns)}"
            )

        # Setup optical system
        # Get wavelength array from camera_parameters or use default
        if "wavelength" in camera_parameters:
            wavelength_array = camera_parameters["wavelength"]
        else:
            # Extract from pixel_QYs shape - assumes standard wavelength grid
            if verbose:
                logger.warning("Warning: 'wavelength' not in camera_parameters, using default from getpixelefficiency()")
            _, _, _, wavelength_array = self.spectral_funcs.getpixelefficiency()

        pixel_QYs = camera_parameters["pixel_QYs"]

        # Get filter spectra
        filter_spectra = self.spectral_funcs.get_dye_or_filter_data(
            names=filter_names, wavelength=wavelength_array, dye_or_filter=False
        )

        if verbose:
            logger.info(f"Optical system configured with {len(filter_names)} filters")
            logger.info(f"Wavelength array: {len(wavelength_array)} points")

        # Extract data from DataFrame
        R = df["A_R"].to_numpy()
        G = df["A_G"].to_numpy()
        B = df["A_B"].to_numpy()
        sigma_x = df["s_x"].to_numpy() * pixel_size  # Convert to nm
        sigma_y = df["s_y"].to_numpy() * pixel_size

        R_err = df["A_R_err"].to_numpy()
        G_err = df["A_G_err"].to_numpy()
        B_err = df["A_B_err"].to_numpy()
        sigma_x_err = df["s_x_err"].to_numpy() * pixel_size
        sigma_y_err = df["s_y_err"].to_numpy() * pixel_size

        # Extract photons and background for SNR calculation (if available)
        if "photons" in df.columns and "background_photons" in df.columns:
            fitted_photons = df["photons"].to_numpy()
            fitted_background_photons = df["background_photons"].to_numpy()
            use_snr = True
            if verbose:
                logger.info("Using photon counts for SNR-based error inflation")
        else:
            if verbose:
                logger.warning("Warning: 'photons' or 'background_photons' columns not found, skipping SNR error inflation")
            fitted_photons = None
            fitted_background_photons = None
            use_snr = False

        # --- Phase 1: Fit aggregate priors (if aggregate_id_column provided) ---
        aggregate_wl_map = {}  # {aggregate_id: fitted_wavelength}
        n_cpus = multiprocessing.cpu_count()
        n_workers = max(1, int(n_cpus * cpu_fraction))

        if aggregate_id_column is not None:
            if aggregate_id_column not in df.columns:
                raise ValueError(
                    f"aggregate_id_column '{aggregate_id_column}' not found in DataFrame. "
                    f"Available columns: {list(df.columns)}"
                )

            if verbose:
                logger.info(f"\n--- Phase 1: Fitting aggregate-level priors ---")
                logger.info(f"Grouping by '{aggregate_id_column}'...")

            aggregate_ids = df[aggregate_id_column].unique()
            aggregate_ids = aggregate_ids[~np.isnan(aggregate_ids)]
            n_aggregates = len(aggregate_ids)

            if verbose:
                logger.info(f"Found {n_aggregates} aggregates")

            # Compute weighted averages per aggregate and build fit args
            agg_fit_args = []
            agg_ids_ordered = []

            for agg_id in aggregate_ids:
                subset = df[df[aggregate_id_column] == agg_id]

                # Weighted averages and propagated errors per channel
                agg_R, agg_R_err = self._weighted_average_with_error(
                    subset["A_R"].to_numpy(), subset["A_R_err"].to_numpy())
                agg_G, agg_G_err = self._weighted_average_with_error(
                    subset["A_G"].to_numpy(), subset["A_G_err"].to_numpy())
                agg_B, agg_B_err = self._weighted_average_with_error(
                    subset["A_B"].to_numpy(), subset["A_B_err"].to_numpy())
                agg_sx, agg_sx_err = self._weighted_average_with_error(
                    subset["s_x"].to_numpy(), subset["s_x_err"].to_numpy())
                agg_sy, agg_sy_err = self._weighted_average_with_error(
                    subset["s_y"].to_numpy(), subset["s_y_err"].to_numpy())

                # Convert PSF widths from pixels to nm
                agg_sx *= pixel_size
                agg_sy *= pixel_size
                agg_sx_err *= pixel_size
                agg_sy_err *= pixel_size

                # Normalize RGB with error propagation
                agg_rgb = np.array([agg_R, agg_G, agg_B])
                agg_rgb_err = np.array([agg_R_err, agg_G_err, agg_B_err])
                if np.sum(agg_rgb) <= 0:
                    continue
                rgb_norm, rgb_norm_err = self._normalize_rgb_with_errors(agg_rgb, agg_rgb_err)

                # Sum photons across aggregate for SNR
                agg_photons = subset["photons"].sum() if use_snr else None
                agg_bg = subset["background_photons"].sum() if use_snr else None

                agg_fit_args.append((
                    rgb_norm, agg_sx, agg_sy, rgb_norm_err, agg_sx_err, agg_sy_err,
                    filter_spectra, wavelength_array, pixel_QYs, NA,
                    agg_photons, agg_bg, wavelength_bounds, None,
                ))
                agg_ids_ordered.append(agg_id)

            # Fit aggregates in parallel
            if len(agg_fit_args) > 0:
                agg_results = self._parallel_fit_wavelengths(
                    agg_fit_args, n_workers, verbose,
                    label="Aggregate fitting", progress_interval=50,
                )

                for idx, (wl, _) in enumerate(agg_results):
                    if not np.isnan(wl):
                        aggregate_wl_map[agg_ids_ordered[idx]] = wl

                if verbose and len(aggregate_wl_map) > 0:
                    agg_wls = np.array(list(aggregate_wl_map.values()))
                    logger.info(f"  Aggregate wavelength range: {np.min(agg_wls):.1f} - {np.max(agg_wls):.1f} nm")
                    logger.info(f"  Aggregate median wavelength: {np.median(agg_wls):.1f} nm")

            if verbose:
                logger.info(f"\n--- Phase 2: Fitting individual localisations with aggregate priors ---")

        # --- Phase 2: Prepare arguments for per-localisation fitting ---
        if verbose:
            logger.info(f"\nPreparing {n_locs} fitting tasks...")

        fit_args = []
        valid_indices = []

        # Look up aggregate IDs for initial guesses
        if aggregate_id_column is not None:
            loc_agg_ids = df[aggregate_id_column].to_numpy()
        else:
            loc_agg_ids = None

        for j in range(n_locs):
            # Skip if RGB total is zero or negative
            rgb_j = np.array([R[j], G[j], B[j]])
            if np.sum(rgb_j) <= 0:
                continue

            rgb_err_j = np.array([R_err[j], G_err[j], B_err[j]])
            rgb_norm, rgb_norm_err = self._normalize_rgb_with_errors(rgb_j, rgb_err_j)

            # Look up aggregate prior initial guess
            wl_guess = None
            if loc_agg_ids is not None and not np.isnan(loc_agg_ids[j]):
                wl_guess = aggregate_wl_map.get(loc_agg_ids[j], None)

            fit_args.append((
                rgb_norm, sigma_x[j], sigma_y[j],
                rgb_norm_err, sigma_x_err[j], sigma_y_err[j],
                filter_spectra, wavelength_array, pixel_QYs, NA,
                fitted_photons[j] if use_snr else None,
                fitted_background_photons[j] if use_snr else None,
                wavelength_bounds, wl_guess,
            ))
            valid_indices.append(j)

        if verbose:
            logger.info(f"Valid fitting tasks: {len(fit_args)}/{n_locs}")
            if aggregate_id_column is not None:
                n_with_prior = sum(
                    1 for j in valid_indices
                    if not np.isnan(loc_agg_ids[j]) and loc_agg_ids[j] in aggregate_wl_map
                )
                logger.info(f"Localisations with aggregate prior: {n_with_prior}/{len(fit_args)}")

        # Parallel wavelength fitting
        wl_fits = np.full(n_locs, np.nan)
        wl_fit_errs = np.full(n_locs, np.nan)

        if len(fit_args) > 0:
            results = self._parallel_fit_wavelengths(
                fit_args, n_workers, verbose,
                label="Localisation fitting", progress_interval=100,
            )

            for idx, (wl, wl_err) in enumerate(results):
                wl_fits[valid_indices[idx]] = wl
                wl_fit_errs[valid_indices[idx]] = wl_err

        # Add wavelength columns to DataFrame
        df["wl_fit"] = wl_fits
        df["wl_fit_err"] = wl_fit_errs

        # Add per-aggregate fitted wavelength column (if using aggregate priors)
        if aggregate_id_column is not None and len(aggregate_wl_map) > 0:
            wl_agg = np.full(n_locs, np.nan)
            for j in range(n_locs):
                agg_id = loc_agg_ids[j]
                if not np.isnan(agg_id) and agg_id in aggregate_wl_map:
                    wl_agg[j] = aggregate_wl_map[agg_id]
            df["wl_fit_aggregate"] = wl_agg

        # Calculate success rate
        n_successful = np.sum(~np.isnan(wl_fits))
        success_rate = 100 * n_successful / n_locs

        if verbose:
            logger.info(f"\nResults:")
            logger.info(f"  Successful fits: {n_successful}/{n_locs} ({success_rate:.1f}%)")
            logger.info(f"  Wavelength range: {np.nanmin(wl_fits):.1f} - {np.nanmax(wl_fits):.1f} nm")
            logger.info(f"  Median wavelength: {np.nanmedian(wl_fits):.1f} nm")
            logger.info(f"  Std wavelength: {np.nanstd(wl_fits):.1f} nm")

        # Save to output file if requested
        if output_path is not None:
            if verbose:
                logger.info(f"\nSaving results to: {output_path}")

            self.io.write_h5_database(df, output_path, normalise_photons=False)

            if verbose:
                logger.info("Save complete!")

        if verbose:
            logger.info(f"\n{'='*60}\n")

        return df

    def fit_wavelengths_pixelated(
        self,
        h5_path: str,
        filter_names: list[str],
        camera_parameters: dict,
        pixel_size_nm: float = 50.0,
        wavelength_bounds: tuple[float, float] = (500.0, 750.0),
        NA: float = 1.49,
        camera_pixel_size: float | None = None,  # nm; None → self.pixel_size * 1000
        min_localisations: int = 3,
        output_path: Optional[str] = None,
        cpu_fraction: float = 0.9,
        verbose: bool = True,
        aggregate_id_column: Optional[str] = None,
        return_grid: bool = True,
    ) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
        """Fit Nile Red wavelengths on a spatial pixel grid.

        Discretises localisations onto a regular grid of user-defined pixel size,
        computes inverse-error-weighted averages of RGB intensities and PSF widths
        per pixel, then fits a single wavelength per pixel. This produces a spatial
        wavelength map with higher per-pixel precision (more photons) and far fewer
        fits than per-localisation fitting.

        Args:
            h5_path: Path to HDF5 file containing localisation data.
            filter_names: List of filter/dichroic names used in optical path.
            camera_parameters: Camera parameters dict containing:
                - 'pixel_QYs': Pixel quantum yields vs wavelength
                - 'wavelength': Wavelength array (nm) - optional
            pixel_size_nm: Grid pixel size in nm. Controls spatial resolution
                vs averaging trade-off. Default: 50.0 nm.
            wavelength_bounds: Search range for wavelength fitting (nm).
            NA: Numerical aperture. Default: 1.49.
            camera_pixel_size: Camera pixel size in nm. Default: 69.0.
            min_localisations: Minimum localisations per pixel to attempt a fit.
                Pixels with fewer localisations get NaN. Default: 3.
            output_path: Optional path to save updated HDF5 file.
            cpu_fraction: Fraction of CPUs to use for parallel fitting.
            verbose: Print progress messages.
            aggregate_id_column: Column name for aggregate/punctum IDs.
                When provided, pixels are grouped within each aggregate so
                overlapping structures are not mixed.
            return_grid: If True, return (DataFrame, grid_info dict).
                If False, return only the DataFrame.

        Returns:
            If return_grid is True:
                Tuple of (DataFrame, grid_info) where grid_info contains:
                - 'wl_grid': (ny, nx) wavelength array (nm), NaN where no fit
                - 'wl_err_grid': (ny, nx) wavelength error array (nm)
                - 'n_locs_grid': (ny, nx) localisation count per pixel
                - 'total_photons_grid': (ny, nx) summed photons per pixel
                - 'mean_photons_grid': (ny, nx) mean photons per loc per pixel
                - 'pixel_size_nm': grid pixel size
                - 'origin_nm': (x_min, y_min) of grid origin in nm
                - 'grid_shape': (ny, nx)
                - 'n_pixels_fitted': number of pixels with successful fits
                - 'n_pixels_skipped': pixels below min_localisations threshold
            If return_grid is False:
                DataFrame with added columns:
                - 'wl_pixel': fitted wavelength (nm) per localisation
                - 'wl_pixel_err': wavelength error (nm) per localisation
                - 'pixel_ix', 'pixel_iy': grid indices

        Notes:
            When ``aggregate_id_column`` is provided, an aggregate-level
            wavelength fit is performed first using all localisations in each
            aggregate. Pixels that fall below ``min_localisations`` (or whose
            fit fails) are assigned the aggregate-level wavelength and error
            instead of NaN, preventing gaps from arbitrary grid boundaries
            splitting small aggregates.

        Required columns in HDF5 file:
            - xc, yc: Localisation positions (camera pixels)
            - A_R, A_G, A_B: RGB amplitudes
            - s_x, s_y: PSF widths (camera pixels)
            - A_R_err, A_G_err, A_B_err: RGB amplitude errors
            - s_x_err, s_y_err: PSF width errors (camera pixels)
            - photons, background_photons: (optional) for SNR-based error inflation
        """
        if camera_pixel_size is None:
            camera_pixel_size = self.pixel_size * 1000  # µm → nm

        import multiprocessing

        if verbose:
            logger.info(f"\n{'='*60}")
            logger.info(f"Pixelated Nile Red Wavelength Fitting")
            logger.info(f"{'='*60}")
            logger.info(f"Input file: {h5_path}")
            logger.info(f"Grid pixel size: {pixel_size_nm} nm")
            logger.info(f"Min localisations per pixel: {min_localisations}")
            logger.info(f"Wavelength bounds: {wavelength_bounds[0]}-{wavelength_bounds[1]} nm")
            logger.info(f"{'='*60}\n")

        # --- Load HDF5 ---
        if not Path(h5_path).exists():
            raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

        df = self.io.read_h5_database(h5_path)
        df = df.loc[:, ~df.columns.duplicated()]
        n_locs = len(df)

        if verbose:
            logger.info(f"Loaded {n_locs} localisations")

        # --- Check required columns ---
        required_cols = [
            "xc", "yc", "A_R", "A_G", "A_B", "s_x", "s_y",
            "A_R_err", "A_G_err", "A_B_err", "s_x_err", "s_y_err",
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"HDF5 file missing required columns: {missing_cols}\n"
                f"Available columns: {list(df.columns)}"
            )

        # --- Setup optical system ---
        if "wavelength" in camera_parameters:
            wavelength_array = camera_parameters["wavelength"]
        else:
            if verbose:
                logger.warning("Warning: 'wavelength' not in camera_parameters, " "using default from getpixelefficiency()")
            _, _, _, wavelength_array = self.spectral_funcs.getpixelefficiency()

        pixel_QYs = camera_parameters["pixel_QYs"]

        filter_spectra = self.spectral_funcs.get_dye_or_filter_data(
            names=filter_names, wavelength=wavelength_array, dye_or_filter=False
        )

        if verbose:
            logger.info(f"Optical system configured with {len(filter_names)} filters")

        # --- Check SNR columns ---
        if "photons" in df.columns and "background_photons" in df.columns:
            use_snr = True
            if verbose:
                logger.info("Using photon counts for SNR-based error inflation")
        else:
            use_snr = False
            if verbose:
                logger.warning("Warning: 'photons' or 'background_photons' columns not found, " "skipping SNR error inflation")

        # --- Aggregate-level fitting (fallback for sub-threshold pixels) ---
        aggregate_wl_map = {}      # {agg_id: fitted_wavelength}
        aggregate_wl_err_map = {}  # {agg_id: fitted_wavelength_error}

        if aggregate_id_column is not None:
            if aggregate_id_column not in df.columns:
                raise ValueError(
                    f"aggregate_id_column '{aggregate_id_column}' not found. "
                    f"Available columns: {list(df.columns)}"
                )

            if verbose:
                logger.info(f"\n--- Fitting aggregate-level wavelengths ---")
                logger.info(f"Grouping by '{aggregate_id_column}'...")

            agg_ids_unique = df[aggregate_id_column].unique()
            agg_ids_unique = agg_ids_unique[~np.isnan(agg_ids_unique)]
            n_aggregates = len(agg_ids_unique)

            if verbose:
                logger.info(f"Found {n_aggregates} aggregates")

            agg_fit_args = []
            agg_ids_ordered = []

            for agg_id in agg_ids_unique:
                subset = df[df[aggregate_id_column] == agg_id]

                try:
                    agg_R, agg_R_err = self._weighted_average_with_error(
                        subset["A_R"].to_numpy(), subset["A_R_err"].to_numpy())
                    agg_G, agg_G_err = self._weighted_average_with_error(
                        subset["A_G"].to_numpy(), subset["A_G_err"].to_numpy())
                    agg_B, agg_B_err = self._weighted_average_with_error(
                        subset["A_B"].to_numpy(), subset["A_B_err"].to_numpy())
                    agg_sx, agg_sx_err = self._weighted_average_with_error(
                        subset["s_x"].to_numpy(), subset["s_x_err"].to_numpy())
                    agg_sy, agg_sy_err = self._weighted_average_with_error(
                        subset["s_y"].to_numpy(), subset["s_y_err"].to_numpy())
                except (ZeroDivisionError, ValueError):
                    continue

                # Convert PSF widths from camera pixels to nm
                agg_sx *= camera_pixel_size
                agg_sy *= camera_pixel_size
                agg_sx_err *= camera_pixel_size
                agg_sy_err *= camera_pixel_size

                # Normalize RGB with error propagation
                agg_rgb = np.array([agg_R, agg_G, agg_B])
                agg_rgb_err = np.array([agg_R_err, agg_G_err, agg_B_err])
                if np.sum(agg_rgb) <= 0:
                    continue
                rgb_norm, rgb_norm_err = self._normalize_rgb_with_errors(
                    agg_rgb, agg_rgb_err)

                # Sum photons across aggregate for SNR
                agg_photons = float(subset["photons"].sum()) if use_snr else None
                agg_bg = (float(subset["background_photons"].sum())
                          if use_snr else None)

                agg_fit_args.append((
                    rgb_norm, agg_sx, agg_sy, rgb_norm_err, agg_sx_err,
                    agg_sy_err, filter_spectra, wavelength_array, pixel_QYs,
                    NA, agg_photons, agg_bg, wavelength_bounds, None,
                ))
                agg_ids_ordered.append(agg_id)

            # Fit aggregates in parallel
            if len(agg_fit_args) > 0:
                n_cpus_agg = multiprocessing.cpu_count()
                n_workers_agg = max(1, int(n_cpus_agg * cpu_fraction))

                agg_results = self._parallel_fit_wavelengths(
                    agg_fit_args, n_workers_agg, verbose,
                    label="Aggregate fitting", progress_interval=50,
                )

                for idx, (wl, wl_err) in enumerate(agg_results):
                    if not np.isnan(wl):
                        aggregate_wl_map[agg_ids_ordered[idx]] = wl
                        aggregate_wl_err_map[agg_ids_ordered[idx]] = wl_err

                if verbose and len(aggregate_wl_map) > 0:
                    agg_wls = np.array(list(aggregate_wl_map.values()))
                    logger.info(f"  Aggregate wavelength range: " f"{np.min(agg_wls):.1f} - {np.max(agg_wls):.1f} nm")
                    logger.info(f"  Median: {np.median(agg_wls):.1f} nm")
                    logger.info(f"  Fitted: {len(aggregate_wl_map)}/{n_aggregates} " f"aggregates")

        # --- Step 1: Discretise onto grid ---
        x_nm = df["xc"].to_numpy() * camera_pixel_size
        y_nm = df["yc"].to_numpy() * camera_pixel_size

        x_min = np.floor(np.min(x_nm) / pixel_size_nm) * pixel_size_nm
        y_min = np.floor(np.min(y_nm) / pixel_size_nm) * pixel_size_nm

        pixel_ix = np.floor((x_nm - x_min) / pixel_size_nm).astype(int)
        pixel_iy = np.floor((y_nm - y_min) / pixel_size_nm).astype(int)

        nx = pixel_ix.max() + 1
        ny = pixel_iy.max() + 1

        if verbose:
            logger.info(f"Grid dimensions: {nx} x {ny} pixels " f"({nx * pixel_size_nm:.0f} x {ny * pixel_size_nm:.0f} nm)")

        # --- Step 2: Build pixel groups ---
        # Pixel key includes aggregate ID when provided to avoid mixing structures
        if aggregate_id_column is not None:
            agg_ids = df[aggregate_id_column].to_numpy()
            # Build composite key: (agg_id, ix, iy)
            # Use a dict mapping composite key -> list of localisation indices
            pixel_groups = {}
            for j in range(n_locs):
                if np.isnan(agg_ids[j]):
                    continue
                key = (agg_ids[j], pixel_ix[j], pixel_iy[j])
                if key not in pixel_groups:
                    pixel_groups[key] = []
                pixel_groups[key].append(j)
        else:
            # Simple spatial grouping: key = (ix, iy)
            pixel_groups = {}
            for j in range(n_locs):
                key = (pixel_ix[j], pixel_iy[j])
                if key not in pixel_groups:
                    pixel_groups[key] = []
                pixel_groups[key].append(j)

        n_total_pixels = len(pixel_groups)

        # --- Step 3: Compute weighted averages per pixel and build fit args ---
        A_R = df["A_R"].to_numpy()
        A_G = df["A_G"].to_numpy()
        A_B = df["A_B"].to_numpy()
        s_x = df["s_x"].to_numpy()
        s_y = df["s_y"].to_numpy()
        A_R_err = df["A_R_err"].to_numpy()
        A_G_err = df["A_G_err"].to_numpy()
        A_B_err = df["A_B_err"].to_numpy()
        s_x_err = df["s_x_err"].to_numpy()
        s_y_err = df["s_y_err"].to_numpy()
        photons = df["photons"].to_numpy() if use_snr else None
        bg_photons = df["background_photons"].to_numpy() if use_snr else None

        fit_args = []
        pixel_keys_ordered = []
        pixel_metadata = []  # (n_locs, total_photons, mean_photons) per pixel
        n_skipped = 0

        for key, indices in pixel_groups.items():
            n_in_pixel = len(indices)
            if n_in_pixel < min_localisations:
                n_skipped += 1
                continue

            idx = np.array(indices)

            # Weighted averages per channel
            try:
                avg_R, avg_R_err = self._weighted_average_with_error(
                    A_R[idx], A_R_err[idx])
                avg_G, avg_G_err = self._weighted_average_with_error(
                    A_G[idx], A_G_err[idx])
                avg_B, avg_B_err = self._weighted_average_with_error(
                    A_B[idx], A_B_err[idx])
                avg_sx, avg_sx_err = self._weighted_average_with_error(
                    s_x[idx], s_x_err[idx])
                avg_sy, avg_sy_err = self._weighted_average_with_error(
                    s_y[idx], s_y_err[idx])
            except (ZeroDivisionError, ValueError):
                n_skipped += 1
                continue

            # Convert PSF widths from camera pixels to nm
            avg_sx *= camera_pixel_size
            avg_sy *= camera_pixel_size
            avg_sx_err *= camera_pixel_size
            avg_sy_err *= camera_pixel_size

            # Normalize RGB with error propagation
            rgb = np.array([avg_R, avg_G, avg_B])
            rgb_err = np.array([avg_R_err, avg_G_err, avg_B_err])
            if np.sum(rgb) <= 0:
                n_skipped += 1
                continue
            rgb_norm, rgb_norm_err = self._normalize_rgb_with_errors(rgb, rgb_err)

            # Photon sums for SNR
            pix_photons = float(np.sum(photons[idx])) if photons is not None else None
            pix_bg = float(np.sum(bg_photons[idx])) if bg_photons is not None else None

            fit_args.append((
                rgb_norm, avg_sx, avg_sy, rgb_norm_err, avg_sx_err, avg_sy_err,
                filter_spectra, wavelength_array, pixel_QYs, NA,
                pix_photons, pix_bg, wavelength_bounds, None,
            ))
            pixel_keys_ordered.append(key)
            pixel_metadata.append((
                n_in_pixel,
                pix_photons if pix_photons is not None else np.nan,
                (pix_photons / n_in_pixel) if pix_photons is not None else np.nan,
            ))

        n_to_fit = len(fit_args)

        if verbose:
            logger.info(f"\nPixel groups: {n_total_pixels} total")
            logger.info(f"  Fitting: {n_to_fit} pixels (>= {min_localisations} localisations)")
            logger.info(f"  Skipped: {n_skipped} pixels (< {min_localisations} localisations)")
            if n_to_fit > 0:
                locs_per_pixel = [m[0] for m in pixel_metadata]
                logger.info(f"  Locs/pixel: median {np.median(locs_per_pixel):.0f}, " f"range [{np.min(locs_per_pixel)}-{np.max(locs_per_pixel)}]")

        # --- Step 4: Parallel wavelength fitting ---
        n_cpus = multiprocessing.cpu_count()
        n_workers = max(1, int(n_cpus * cpu_fraction))

        pixel_wl = {}      # key -> fitted wavelength
        pixel_wl_err = {}  # key -> fitted wavelength error

        if n_to_fit > 0:
            results = self._parallel_fit_wavelengths(
                fit_args, n_workers, verbose,
                label="Pixel fitting", progress_interval=50,
            )

            n_success = 0
            for i, (wl, wl_err) in enumerate(results):
                if not np.isnan(wl):
                    pixel_wl[pixel_keys_ordered[i]] = wl
                    pixel_wl_err[pixel_keys_ordered[i]] = wl_err
                    n_success += 1

            if verbose:
                logger.info(f"\nFit results: {n_success}/{n_to_fit} pixels converged")
                if n_success > 0:
                    wls = np.array(list(pixel_wl.values()))
                    logger.info(f"  Wavelength range: {np.min(wls):.1f} - {np.max(wls):.1f} nm")
                    logger.info(f"  Median: {np.median(wls):.1f} nm, Std: {np.std(wls):.1f} nm")

        # --- Step 5: Build output grids ---
        wl_grid = np.full((ny, nx), np.nan)
        wl_err_grid = np.full((ny, nx), np.nan)
        n_locs_grid = np.zeros((ny, nx), dtype=int)
        total_photons_grid = np.full((ny, nx), np.nan)
        mean_photons_grid = np.full((ny, nx), np.nan)

        for i, key in enumerate(pixel_keys_ordered):
            # Extract grid coordinates from key
            if aggregate_id_column is not None:
                _, ix, iy = key
            else:
                ix, iy = key

            n_loc, tot_ph, mean_ph = pixel_metadata[i]
            n_locs_grid[iy, ix] += n_loc
            # Accumulate photons (handles NaN from first write)
            existing = total_photons_grid[iy, ix]
            if np.isnan(existing):
                total_photons_grid[iy, ix] = tot_ph
            elif not np.isnan(tot_ph):
                total_photons_grid[iy, ix] = existing + tot_ph
            mean_photons_grid[iy, ix] = mean_ph

            if key in pixel_wl:
                wl_grid[iy, ix] = pixel_wl[key]
                wl_err_grid[iy, ix] = pixel_wl_err.get(key, np.nan)

        # Fill grid gaps with aggregate-level wavelengths
        if aggregate_id_column is not None and len(aggregate_wl_map) > 0:
            for key, indices in pixel_groups.items():
                if key not in pixel_wl:
                    agg_id, ix, iy = key
                    if agg_id in aggregate_wl_map:
                        wl_grid[iy, ix] = aggregate_wl_map[agg_id]
                        wl_err_grid[iy, ix] = aggregate_wl_err_map.get(
                            agg_id, np.nan)
                        n_locs_grid[iy, ix] += len(indices)

        # --- Step 6: Assign pixel wavelength back to localisations ---
        wl_pixel = np.full(n_locs, np.nan)
        wl_pixel_err = np.full(n_locs, np.nan)
        n_from_pixel = 0
        n_from_aggregate = 0

        for key, indices in pixel_groups.items():
            if key in pixel_wl:
                # Pixel-level fit succeeded
                for j in indices:
                    wl_pixel[j] = pixel_wl[key]
                    wl_pixel_err[j] = pixel_wl_err.get(key, np.nan)
                n_from_pixel += len(indices)
            elif aggregate_id_column is not None:
                # Fall back to aggregate-level fit
                agg_id = key[0]  # key is (agg_id, ix, iy)
                if agg_id in aggregate_wl_map:
                    for j in indices:
                        wl_pixel[j] = aggregate_wl_map[agg_id]
                        wl_pixel_err[j] = aggregate_wl_err_map.get(
                            agg_id, np.nan)
                    n_from_aggregate += len(indices)

        df["wl_pixel"] = wl_pixel
        df["wl_pixel_err"] = wl_pixel_err
        df["pixel_ix"] = pixel_ix
        df["pixel_iy"] = pixel_iy

        n_assigned = np.sum(~np.isnan(wl_pixel))
        if verbose:
            logger.info(f"\nLocalisations with pixel wavelength: " f"{n_assigned}/{n_locs} ({100*n_assigned/n_locs:.1f}%)")
            if aggregate_id_column is not None:
                logger.info(f"  From pixel fits: {n_from_pixel}")
                logger.info(f"  From aggregate fallback: {n_from_aggregate}")

        # --- Save ---
        if output_path is not None:
            if verbose:
                logger.info(f"\nSaving results to: {output_path}")
            self.io.write_h5_database(df, output_path, normalise_photons=False)
            if verbose:
                logger.info("Save complete!")

        if verbose:
            logger.info(f"\n{'='*60}\n")

        if return_grid:
            grid_info = {
                "wl_grid": wl_grid,
                "wl_err_grid": wl_err_grid,
                "n_locs_grid": n_locs_grid,
                "total_photons_grid": total_photons_grid,
                "mean_photons_grid": mean_photons_grid,
                "pixel_size_nm": pixel_size_nm,
                "origin_nm": (x_min, y_min),
                "grid_shape": (ny, nx),
                "n_pixels_fitted": len(pixel_wl),
                "n_pixels_skipped": n_skipped,
            }
            return df, grid_info
        else:
            return df


# Module-level standalone function for parallel Nile Red wavelength fitting (must be pickleable)
def _fit_nile_red_wavelength_standalone(
    rgb: np.ndarray,
    sigma_x: float,
    sigma_y: float,
    rgb_err: np.ndarray,
    sigma_x_err: float,
    sigma_y_err: float,
    filter_spectra: np.ndarray,
    wavelength_array: np.ndarray,
    pixel_QYs: np.ndarray,
    NA: float,
    total_photons: Optional[float] = None,
    background_photons: Optional[float] = None,
    wavelength_bounds: tuple[float, float] = (500.0, 750.0),
    wavelength_initial_guess: Optional[float] = None,
) -> tuple[float, float]:
    """Standalone function for fitting Nile Red wavelength from a single localization.

    Must be a module-level function (not a method) to be pickleable for multiprocessing.

    Args:
        rgb: Normalized [R, G, B] intensities
        sigma_x, sigma_y: PSF widths in nm
        rgb_err, sigma_x_err, sigma_y_err: Errors on measurements
        filter_spectra, wavelength_array, pixel_QYs: Optical system parameters
        NA: Numerical aperture
        total_photons: Fitted total photon count (for SNR-based error inflation)
        background_photons: Fitted background photon count (for SNR-based error inflation)
        wavelength_bounds: Search range for wavelength (nm)
        wavelength_initial_guess: Custom initial guess for wavelength (nm).
            If None, uses default (617.6 nm).

    Returns:
        Tuple of (fitted_wavelength, wavelength_error)
        Returns (NaN, NaN) if fit fails
    """
    try:
        nrf = NileRed_Functions()

        wl, predictions = nrf.fit_nile_red_wavelength(
            observed_rgb=rgb,
            observed_sigma_x=sigma_x,
            observed_sigma_y=sigma_y,
            rgb_errors=rgb_err,
            sigma_x_error=sigma_x_err,
            sigma_y_error=sigma_y_err,
            filter_spectra=filter_spectra,
            wavelength_array=wavelength_array,
            pixel_QYs=pixel_QYs,
            NA=NA,
            wavelength_bounds=wavelength_bounds,
            total_photons=total_photons,
            background_photons=background_photons,
            apply_snr_inflation=True if total_photons is not None else False,
            wavelength_initial_guess=wavelength_initial_guess,
        )
        wl_err = predictions.get("wavelength_error", np.nan)
        return (wl, wl_err)
    except Exception:
        return (np.nan, np.nan)
