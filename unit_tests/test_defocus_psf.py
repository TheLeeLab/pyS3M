"""Unit tests for VectorialPSF and DipolePSF (src/simulation/defocus_psf.py).

Tests cover:
  - PSF normalisation (unit sum per z-plane)
  - In-focus symmetry (4-fold symmetry for zero lateral offset)
  - PSF centration (peak at centre for in-focus, zero shift)
  - Defocus broadening (PSF widens monotonically with |z|)
  - Defocus z-symmetry (|z| gives same PSF when SA=0)
  - Lateral shift (sub-pixel x0 moves peak)
  - Spectral integration shape and normalisation
  - Padding / output shape consistency across wavelengths
"""
import sys
from pathlib import Path
import numpy as np
import pytest


from pyS3M.simulation.defocus_psf import VectorialPSF


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def psf():
    """VectorialPSF configured for a 100× oil-immersion objective."""
    return VectorialPSF(
        NA=1.49,
        n_medium=1.33,
        n_immersion=1.515,
        pix_obj_um=0.069,   # Ximea: 69 nm object-space pixel
        psf_size=21,
        N_pupil=128,        # reduced for test speed; 256 for production
    )


WAVELENGTHS_UM = np.array([0.55, 0.62, 0.69])
Z_UM = np.array([0.0, 0.1, 0.3])


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

class TestNormalisation:
    def test_monochromatic_sums_to_one(self, psf):
        stack = psf.compute_psf_stack(Z_UM, WAVELENGTHS_UM)
        # Each (z, lambda) patch should sum to 1
        sums = stack.sum(axis=(-2, -1))
        np.testing.assert_allclose(sums, 1.0, rtol=1e-6)

    def test_spectral_integrated_sums_to_one(self, psf):
        n_lam = len(WAVELENGTHS_UM)
        n_ch = 3
        # Uniform spectral weights
        weights = np.ones((n_lam, n_ch), dtype=np.float64) / n_lam
        stack = psf.compute_psf_stack(Z_UM, WAVELENGTHS_UM, spectral_weights=weights)
        sums = stack.sum(axis=(-2, -1))
        np.testing.assert_allclose(sums, 1.0, rtol=1e-6)

    def test_field_normalisation_keeps_energy_outside_patch(self, psf):
        patch = psf.compute_psf_stack(Z_UM, WAVELENGTHS_UM)
        field = psf.compute_psf_stack(Z_UM, WAVELENGTHS_UM, normalise="field")
        sums = field.sum(axis=(-2, -1))
        # Part of the energy (Airy tails, defocus rings) lies outside the 21-px patch
        assert np.all(sums < 1.0) and np.all(sums > 0.5)
        # Same shape, only the scale differs
        np.testing.assert_allclose(field / sums[..., None, None], patch, rtol=1e-6, atol=1e-12)

    def test_field_fraction_grows_with_patch_size(self):
        kwargs = dict(NA=1.49, n_medium=1.33, n_immersion=1.515, pix_obj_um=0.069, N_pupil=128)
        small = VectorialPSF(psf_size=11, **kwargs).compute_psf_stack([0.0], [0.6], normalise="field")
        large = VectorialPSF(psf_size=41, **kwargs).compute_psf_stack([0.0], [0.6], normalise="field")
        assert small.sum() < large.sum() <= 1.0

    def test_bad_normalise_raises(self, psf):
        with pytest.raises(ValueError, match="normalise"):
            psf.compute_psf_stack(Z_UM, WAVELENGTHS_UM, normalise="peak")
        with pytest.raises(ValueError, match="monochromatic"):
            psf.compute_psf_stack(Z_UM, WAVELENGTHS_UM, spectral_weights=np.ones((3, 3)), normalise="field")


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------

class TestOutputShape:
    def test_monochromatic_shape(self, psf):
        stack = psf.compute_psf_stack(Z_UM, WAVELENGTHS_UM)
        assert stack.shape == (len(Z_UM), len(WAVELENGTHS_UM), 21, 21)

    def test_spectral_shape(self, psf):
        weights = np.ones((len(WAVELENGTHS_UM), 3)) / len(WAVELENGTHS_UM)
        stack = psf.compute_psf_stack(Z_UM, WAVELENGTHS_UM, spectral_weights=weights)
        assert stack.shape == (len(Z_UM), 3, 21, 21)

    def test_single_z_single_lambda(self, psf):
        stack = psf.compute_psf_stack(np.array([0.0]), np.array([0.60]))
        assert stack.shape == (1, 1, 21, 21)


# ---------------------------------------------------------------------------
# In-focus symmetry
# ---------------------------------------------------------------------------

class TestInFocusSymmetry:
    """In-focus PSF (z=0, no SA, no shift) should be 4-fold symmetric."""

    def _get_infocus(self, psf):
        stack = psf.compute_psf_stack(
            np.array([0.0]), np.array([0.60])
        )
        return stack[0, 0]  # (21, 21)

    def test_left_right_symmetry(self, psf):
        I = self._get_infocus(psf)
        np.testing.assert_allclose(I, np.fliplr(I), atol=1e-8)

    def test_up_down_symmetry(self, psf):
        I = self._get_infocus(psf)
        np.testing.assert_allclose(I, np.flipud(I), atol=1e-8)

    def test_peak_at_center(self, psf):
        I = self._get_infocus(psf)
        peak = np.unravel_index(np.argmax(I), I.shape)
        centre = (psf.psf_size // 2, psf.psf_size // 2)
        assert peak == centre, f"Peak {peak} not at centre {centre}"


# ---------------------------------------------------------------------------
# Defocus broadening
# ---------------------------------------------------------------------------

class TestDefocusBroadening:
    """PSF should broaden (lower peak, larger sigma) with increasing |z|."""

    def test_peak_decreases_with_defocus(self, psf):
        z_vals = np.array([0.0, 0.05, 0.1, 0.2, 0.4])
        stack = psf.compute_psf_stack(z_vals, np.array([0.60]))
        peaks = stack[:, 0].max(axis=(-2, -1))
        # Each successive z should have a lower or equal peak
        assert np.all(np.diff(peaks) <= 1e-8), \
            f"Peak did not decrease monotonically: {peaks}"


# ---------------------------------------------------------------------------
# Defocus z-symmetry (no spherical aberration)
# ---------------------------------------------------------------------------

class TestDefocusSymmetry:
    """Without depth-induced SA, PSF at +z0 should equal PSF at -z0."""

    def test_positive_negative_z_equal(self, psf):
        z_pos = np.array([0.2])
        z_neg = np.array([-0.2])
        lam = np.array([0.60])

        I_pos = psf.compute_psf_stack(z_pos, lam, distance_from_coverslip_um=0.0)[0, 0]
        I_neg = psf.compute_psf_stack(z_neg, lam, distance_from_coverslip_um=0.0)[0, 0]
        # Should be identical (defocus is symmetric without SA)
        np.testing.assert_allclose(I_pos, I_neg, atol=1e-8)


# ---------------------------------------------------------------------------
# SA breaks z-symmetry
# ---------------------------------------------------------------------------

class TestSphericalAberration:
    def test_sa_breaks_z_symmetry(self, psf):
        z_pos = np.array([0.2])
        z_neg = np.array([-0.2])
        lam = np.array([0.60])
        d = 0.5  # 500 nm into sample

        I_pos = psf.compute_psf_stack(z_pos, lam, distance_from_coverslip_um=d)[0, 0]
        I_neg = psf.compute_psf_stack(z_neg, lam, distance_from_coverslip_um=d)[0, 0]
        assert not np.allclose(I_pos, I_neg, atol=1e-6), \
            "SA should break z-symmetry but PSFs are identical"

    def test_sa_zero_matches_no_sa(self, psf):
        lam = np.array([0.60])
        z = np.array([0.2])
        I_sa0 = psf.compute_psf_stack(z, lam, distance_from_coverslip_um=0.0)[0, 0]
        I_sa_explicit = psf.compute_psf_stack(
            z, lam, distance_from_coverslip_um=0.0
        )[0, 0]
        np.testing.assert_allclose(I_sa0, I_sa_explicit, atol=1e-12)


# ---------------------------------------------------------------------------
# Lateral shift
# ---------------------------------------------------------------------------

class TestLateralShift:
    def test_shift_moves_peak(self, psf):
        lam = np.array([0.60])
        z = np.array([0.0])
        shift_um = psf.pix_obj_um  # shift by exactly one pixel

        I_centre = psf.compute_psf_stack(z, lam, x0_um=0.0)[0, 0]
        I_shift = psf.compute_psf_stack(z, lam, x0_um=shift_um)[0, 0]

        peak_c = np.unravel_index(np.argmax(I_centre), I_centre.shape)
        peak_s = np.unravel_index(np.argmax(I_shift), I_shift.shape)
        assert peak_s != peak_c, "Lateral shift did not move peak"


# ---------------------------------------------------------------------------
# Aperture clipping
# ---------------------------------------------------------------------------

class TestApertureClipping:
    """For NA > n_medium, rho_prop < 1; check aperture is correctly clipped."""

    def test_propagating_aperture_fraction(self, psf):
        # rho_prop = n_medium / NA = 1.33 / 1.49 ≈ 0.893
        rho_prop = psf.n_medium / psf.NA
        expected_fraction = np.pi * rho_prop ** 2 / 4  # circle-in-square
        actual_fraction = psf._aperture.sum() / psf._aperture.size
        # Should be within 5% of the circular fraction
        assert abs(actual_fraction - expected_fraction) < 0.05, \
            f"Aperture fraction {actual_fraction:.3f} != expected {expected_fraction:.3f}"


# ---------------------------------------------------------------------------
# Phase mask passthrough
# ---------------------------------------------------------------------------

class TestPhaseMask:
    def test_flat_phase_mask_identity(self, psf):
        """Flat (all-zero) phase mask should give same result as no mask."""
        lam = np.array([0.60])
        z = np.array([0.0])
        flat_mask = np.zeros((psf.N_pupil, psf.N_pupil))

        I_no_mask = psf.compute_psf_stack(z, lam)[0, 0]
        I_flat_mask = psf.compute_psf_stack(z, lam, phase_mask=flat_mask)[0, 0]
        np.testing.assert_allclose(I_no_mask, I_flat_mask, atol=1e-10)


# ---------------------------------------------------------------------------
# build_spectral_weights (smoke test, no database required to be mocked)
# ---------------------------------------------------------------------------

class TestBuildSpectralWeights:
    def test_output_shape_and_non_negative(self):
        """Mock spectral_functions to check build_spectral_weights contract."""

        n_lam = 50
        n_ch = 3
        wl_nm = np.linspace(500, 750, n_lam)
        pixel_QYs = np.random.uniform(0, 0.5, size=(n_ch, n_lam))

        class MockSpectral:
            def get_spectral_data(self, names, wl, dtype):
                # Gaussian emission spectrum
                return np.exp(-((wl - 620) ** 2) / (2 * 30 ** 2))[np.newaxis, :]

            def getobjectiveefficiency(self, wl):
                return np.ones_like(wl)

        weights = VectorialPSF.build_spectral_weights(
            spectral_functions=MockSpectral(),
            dye="mock_dye",
            filters=None,
            wavelengths_nm=wl_nm,
            pixel_QYs=pixel_QYs,
            include_objective=True,
        )

        assert weights.shape == (n_lam, n_ch)
        assert np.all(weights >= 0), "Weights should be non-negative"
        assert np.any(weights > 0), "At least some weights should be positive"

    def test_real_spectral_functions_with_filters_and_no_objective(self):
        """Real pyS3M.SpectralFunctions.Spectral_Funcs instance (not a mock):
        exercises the SpectralDataType-lookup-via-instance-module branch
        (only reachable when the class's own module already has
        SpectralDataType, unlike a plain mock class), plus filters != None
        and include_objective=False."""
        import pyS3M.SpectralFunctions as SpectralFunctions

        sf = SpectralFunctions.Spectral_Funcs()
        wl_nm = np.linspace(600, 700, 20)
        n_ch = 3
        pixel_QYs = np.random.uniform(0, 0.5, size=(n_ch, len(wl_nm)))

        weights = VectorialPSF.build_spectral_weights(
            spectral_functions=sf,
            dye=sf.dye_names[0],
            filters=[sf.filter_names[0]],
            wavelengths_nm=wl_nm,
            pixel_QYs=pixel_QYs,
            include_objective=False,
        )
        assert weights.shape == (len(wl_nm), n_ch)
        assert np.all(weights >= 0)


# ---------------------------------------------------------------------------
# _compute_padding
# ---------------------------------------------------------------------------

class TestComputePadding:
    def test_odd_total_bumped_to_even(self, psf):
        # wavelength=0.400um with this psf's NA/pix_obj_um/N_pupil gives an
        # odd N_total_exact rounding, exercising the "+= 1" parity fix-up.
        pad = psf._compute_padding(0.400)
        assert pad == 61


# ---------------------------------------------------------------------------
# DipolePSF -- vectorial isotropic-dipole PSF (Fazel et al. 2024)
# ---------------------------------------------------------------------------

from scipy.integrate import quad
from scipy.special import j1

from pyS3M.simulation.defocus_psf import DipolePSF

N_WATER, N_OIL = 1.33, 1.515


def _fwhm(x, profile):
    above = x[profile >= 0.5 * profile.max()]
    return above.max() - above.min()


def _fresnel_t(s1, n1=N_WATER, n2=N_OIL):
    """Transmission (field) coefficients water -> glass at sample-side sin(theta) s1 (subcritical)."""
    c1 = np.sqrt(1 - s1 ** 2)
    c2 = np.sqrt(1 - (n1 * s1 / n2) ** 2)
    return c1, c2, 2 * n1 * c1 / (n1 * c1 + n2 * c2), 2 * n1 * c1 / (n2 * c1 + n1 * c2)


class TestDipolePSFFazelFig18:
    """Fazel Fig. 18: NA 1.2 water objective, emitter in water, 500 nm, focal-plane x profiles."""

    x = np.linspace(-0.8, 0.8, 1601)

    @pytest.fixture(scope="class")
    def profiles(self):
        dpsf = DipolePSF(NA=1.2, n_medium=N_WATER, n_immersion=N_WATER, N_pupil=256)
        prof = {o: dpsf.image(0.5, self.x, [0.0], orientation=o)[0] for o in ("iso", "x")}
        v = 2 * np.pi * 1.2 * np.abs(self.x) / 0.5
        prof["airy"] = np.where(v > 0, (2 * j1(v) / np.where(v > 0, v, 1)) ** 2, 1.0)  # Eq. 66
        return prof

    def test_width_ordering(self, profiles):
        # random dipole widest, x dipole (along its own axis) next, scalar Airy narrowest
        w = {k: _fwhm(self.x, p) for k, p in profiles.items()}
        assert w["iso"] > w["x"] > w["airy"]
        assert 1.05 < w["iso"] / w["airy"] < 1.12  # 232 vs 214 nm

    def test_random_dipole_fills_first_minimum(self, profiles):
        # the z dipole's doughnut fills the Airy zero at 0.61 lambda / NA = 0.254 um
        i = np.argmin(np.abs(self.x - 0.254))
        assert profiles["airy"][i] / profiles["airy"].max() < 0.002
        assert 0.02 < profiles["iso"][i] / profiles["iso"].max() < 0.05


class TestDipolePSFEnergy:
    """Collected power against independent calculations that do not use the pupil grid."""

    def test_homogeneous_matches_free_dipole(self):
        # n_d = n: x dipole radiates 3/(8 pi) (1 - sin^2 cos^2 phi) per solid angle
        NA, n = 1.2, N_WATER
        th_max = np.arcsin(NA / n)
        expected = quad(lambda t: 3 / (8 * np.pi) * np.pi * (1 + np.cos(t) ** 2) * np.sin(t), 0, th_max)[0]
        got = DipolePSF(NA=NA, n_medium=n, n_immersion=n, N_pupil=512).collected_power("x")
        np.testing.assert_allclose(got, expected, rtol=3e-3)

    @pytest.mark.parametrize("orientation", ["x", "z"])
    def test_subcritical_energy_conservation(self, orientation):
        # Power into the glass = free dipole's power x Fresnel power transmittance T, per direction.
        # All subcritical light (theta_glass < 61 deg) is inside NA 1.49.
        def integrand(t1):
            s1 = np.sin(t1)
            c1, c2, t_s, t_p = _fresnel_t(s1)
            T_s = N_OIL * c2 / (N_WATER * c1) * t_s ** 2
            T_p = N_OIL * c2 / (N_WATER * c1) * t_p ** 2
            if orientation == "z":
                return 3 / (8 * np.pi) * T_p * s1 ** 2 * 2 * np.pi * s1
            return 3 / (8 * np.pi) * np.pi * (T_p * c1 ** 2 + T_s) * s1

        expected = quad(integrand, 0, np.pi / 2, limit=200)[0]
        dpsf = DipolePSF(NA=1.49, n_medium=N_WATER, n_immersion=N_OIL, N_pupil=512, supercritical=False)
        np.testing.assert_allclose(dpsf.collected_power(orientation), expected, rtol=3e-3)

    @pytest.mark.parametrize("orientation", ["x", "z"])
    def test_supercritical_power_matches_ldos(self, orientation):
        # Total power of a dipole at the interface from its reflection coefficients (Novotny & Hecht
        # Eqs. 10.26-27), minus the power back into the water and the subcritical power into the glass,
        # leaves the supercritical power into the glass.
        e1, e2, m = N_WATER ** 2, N_OIL ** 2, N_OIL / N_WATER
        kz = lambda s, n: np.sqrt(complex((n / N_WATER) ** 2 - s ** 2))
        rp = lambda s: (e2 * kz(s, N_WATER) - e1 * kz(s, N_OIL)) / (e2 * kz(s, N_WATER) + e1 * kz(s, N_OIL))
        rs = lambda s: (kz(s, N_WATER) - kz(s, N_OIL)) / (kz(s, N_WATER) + kz(s, N_OIL))
        if orientation == "z":
            core = lambda s: 1.5 * s ** 3 * rp(s)
        else:
            core = lambda s: 0.75 * s * (rs(s) - kz(s, N_WATER) ** 2 * rp(s))
        # 1/kz singularity at s = 1 removed: s = sin(a) below, s = sqrt(1 + t^2) above
        total = 1 + quad(lambda a: core(np.sin(a)).real, 0, np.pi / 2, limit=200)[0] \
            + quad(lambda t: (core(np.sqrt(1 + t ** 2)) / (1j * np.sqrt(1 + t ** 2))).real,
                   0, np.sqrt(m ** 2 - 1), limit=200)[0]

        def up(th):  # direct + reflected wave back into the water
            s, c = np.sin(th), np.cos(th)
            if orientation == "z":
                return 3 / (8 * np.pi) * s ** 2 * abs(1 + rp(s)) ** 2 * 2 * np.pi * s
            return 3 / (8 * np.pi) * np.pi * (c ** 2 * abs(1 - rp(s)) ** 2 + abs(1 + rs(s)) ** 2) * s

        sub = DipolePSF(NA=1.49, n_medium=N_WATER, n_immersion=N_OIL, N_pupil=512,
                        supercritical=False).collected_power(orientation)
        expected = total - quad(up, 0, np.pi / 2, limit=200)[0] - sub
        # NA just below n_oil, so the whole glass hemisphere is collected
        dpsf = DipolePSF(NA=0.9999 * N_OIL, n_medium=N_WATER, n_immersion=N_OIL, N_pupil=1024)
        np.testing.assert_allclose(dpsf.collected_power(orientation, supercritical_only=True), expected,
                                   rtol=5e-3)

    def test_supercritical_light_decays_with_height(self):
        kw = dict(NA=1.49, n_medium=N_WATER, n_immersion=N_OIL, N_pupil=256)
        p = [DipolePSF(z_dipole_um=z, **kw).collected_power(supercritical_only=True) for z in (0.0, 0.05, 0.2)]
        # decays on a ~50-100 nm scale (slower just past the critical angle)
        assert p[0] > p[1] > p[2] and p[2] < 0.3 * p[0]


class TestDipolePSFPatches:
    @pytest.fixture(scope="class")
    def dpsf(self):
        return DipolePSF(NA=1.49, n_medium=N_WATER, n_immersion=N_OIL, N_pupil=256)

    def test_patch_is_a_probability_and_grows_with_size(self, dpsf):
        small = dpsf.pixel_patches([0.6], 0.069, 15, oversample=3).sum()
        large = dpsf.pixel_patches([0.6], 0.069, 61, oversample=3).sum()
        assert 0.7 < small < large < 1.0
        assert large > 0.9

    def test_centred_and_symmetric(self, dpsf):
        patch = dpsf.pixel_patches([0.6], 0.069, 21, oversample=5)[0]
        assert np.unravel_index(np.argmax(patch), patch.shape) == (10, 10)
        np.testing.assert_allclose(patch, patch.T, rtol=1e-6, atol=1e-12)
        np.testing.assert_allclose(patch, patch[::-1, ::-1], rtol=1e-6, atol=1e-12)

    def test_offset_shifts_the_image_exactly(self, dpsf):
        # a whole-pixel offset equals the unshifted patch moved by one column (the image is evaluated
        # on the shifted grid, so sub-pixel offsets are exact too)
        base = dpsf.pixel_patches([0.6], 0.069, 21, oversample=5)[0]
        shifted = dpsf.pixel_patches([0.6], 0.069, 21, oversample=5, x0_um=0.069)[0]
        np.testing.assert_allclose(shifted[:, 1:], base[:, :-1], rtol=1e-9, atol=1e-15)
        half = dpsf.pixel_patches([0.6], 0.069, 21, oversample=5, x0_um=0.5 * 0.069)[0]
        np.testing.assert_allclose(half[:, 11], half[:, 10], rtol=1e-9)  # symmetric about the pixel edge

    def test_bad_arguments_raise(self, dpsf):
        with pytest.raises(ValueError, match="odd"):
            dpsf.pixel_patches([0.6], 0.069, 21, oversample=4)
        with pytest.raises(ValueError, match="orientation"):
            dpsf.image(0.6, [0.0], [0.0], orientation="w")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
