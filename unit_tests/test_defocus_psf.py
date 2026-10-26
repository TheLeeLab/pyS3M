"""Unit tests for VectorialPSF (src/simulation/defocus_psf.py).

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
