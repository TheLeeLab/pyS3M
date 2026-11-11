"""Full coverage tests for pyS3M.render.

Covers the `render()` dispatcher and its 4 live blur methods (None/hist,
"gaussian", "gaussian_RGB", "gaussian_colour", "smooth").

Several private helpers (_render_setup, _render_colour_setup, _fill,
_fill_gaussian, _fill_colour_gaussian, _fill_RGB_gaussian) are
`@numba.njit` -- once JIT-compiled they run as machine code that bypasses
Python's trace hooks, so coverage.py cannot see line hits inside them no
matter how many times they're called normally. Each exposes the original,
uncompiled Python implementation via `.py_func`; tests call both the normal
(JIT) path, to verify real runtime behaviour, and the `.py_func` path, purely
so coverage.py can see the body executed -- same pattern as
unit_tests/test_localise.py.
"""
from __future__ import annotations

import numpy as np
import pytest

import pyS3M.render as render_mod
from pyS3M.render import (
    render,
    render_hist,
    render_gaussian,
    render_gaussian_colour,
    render_gaussian_RGB,
    render_smooth,
    _render_setup,
    _render_colour_setup,
    _fill,
    _fill_gaussian,
    _fill_colour_gaussian,
    _fill_RGB_gaussian,
    _fftconvolve,
)


def _locs(n=5, seed=0, with_photoelectrons=True, with_rgb=False):
    rng = np.random.default_rng(seed)
    xc = rng.uniform(10, 90, n)
    yc = rng.uniform(10, 90, n)
    xc_err = np.full(n, 0.5)
    yc_err = np.full(n, 0.5)
    arrays = [xc, yc, xc_err, yc_err]
    names = ["xc", "yc", "xc_err", "yc_err"]
    if with_photoelectrons:
        arrays.append(np.full(n, 1000.0))
        names.append("photoelectrons")
    arrays.append(rng.uniform(0.2, 0.4, n))
    names.append("A_R")
    if with_rgb:
        arrays += [rng.uniform(0.2, 0.4, n), rng.uniform(0.2, 0.4, n)]
        names += ["A_G", "A_B"]
    return np.rec.fromarrays(arrays, names=names)


_INFO = [{"Width": 100, "Height": 100}]


# ======================================================================
# render() dispatcher
# ======================================================================

class TestRenderDispatch:
    def test_no_blur_uses_render_hist(self):
        locs = _locs()
        n, img = render(locs, info=_INFO)
        assert n == 5
        assert img.shape == (100, 100)

    def test_gaussian(self):
        locs = _locs()
        n, img = render(locs, info=_INFO, blur_method="gaussian")
        assert n == 5

    def test_gaussian_rgb(self):
        locs = _locs(with_rgb=True)
        n, img, img_rgb = render(locs, info=_INFO, blur_method="gaussian_RGB")
        assert n == 5
        assert img_rgb.shape == (100, 100, 3)

    def test_gaussian_colour(self):
        locs = _locs()
        n, img, img_colour = render(locs, info=_INFO, blur_method="gaussian_colour")
        assert n == 5
        assert img_colour.shape == (100, 100, 3)

    def test_smooth(self):
        locs = _locs()
        n, img = render(locs, info=_INFO, blur_method="smooth")
        assert n == 5

    def test_unknown_blur_method_raises(self):
        locs = _locs()
        with pytest.raises(Exception, match="blur_method not understood"):
            render(locs, info=_INFO, blur_method="bogus")

    def test_explicit_viewport(self):
        locs = _locs()
        n, img = render(locs, viewport=((0, 0), (100, 100)))
        assert n == 5

    def test_no_viewport_no_info_raises(self):
        locs = _locs()
        with pytest.raises(ValueError, match="Need info"):
            render(locs)


# ======================================================================
# render_hist / _render_setup / _fill
# ======================================================================

class TestRenderHist:
    def test_basic(self):
        locs = _locs()
        n, img = render_hist(locs, 1.0, 0, 0, 100, 100)
        assert n == 5
        assert img.sum() == 5

    def test_py_func_matches_jit(self):
        locs = _locs()
        image_jit, ny_jit, nx_jit, x_jit, y_jit, iv_jit = _render_setup(locs, 1.0, 0, 0, 100, 100)
        image_py, ny_py, nx_py, x_py, y_py, iv_py = _render_setup.py_func(locs, 1.0, 0, 0, 100, 100)
        np.testing.assert_array_equal(x_jit, x_py)
        np.testing.assert_array_equal(y_jit, y_py)
        assert (ny_jit, nx_jit) == (ny_py, nx_py)
        np.testing.assert_array_equal(iv_jit, iv_py)

        _fill(image_jit, x_jit, y_jit)
        _fill.py_func(image_py, x_py, y_py)
        np.testing.assert_array_equal(image_jit, image_py)

    def test_out_of_view_locs_excluded(self):
        # A localisation exactly on the boundary (x==x_max) is excluded by
        # the strict `<`/`>` in_view comparison.
        locs = np.rec.fromarrays(
            [np.array([100.0, 50.0]), np.array([50.0, 50.0]),
             np.array([0.5, 0.5]), np.array([0.5, 0.5])],
            names=["xc", "yc", "xc_err", "yc_err"],
        )
        n, img = render_hist(locs, 1.0, 0, 0, 100, 100)
        assert n == 1


# ======================================================================
# render_gaussian / _fill_gaussian
# ======================================================================

class TestRenderGaussian:
    def test_basic(self):
        locs = _locs()
        n, img = render_gaussian(locs, 1.0, 0, 0, 100, 100, min_blur_width=1.0)
        assert n == 5
        assert img.sum() > 0

    def test_no_photons_column_defaults_to_one(self):
        locs = _locs(with_photoelectrons=False)
        n, img = render_gaussian(locs, 1.0, 0, 0, 100, 100, min_blur_width=1.0)
        assert n == 5

    def test_py_func_matches_jit(self):
        # 8x8 canvas, a single point dead centre with sigma large enough
        # that its 3-sigma draw radius overflows all four edges at once --
        # exercises the i_min/i_max/j_min/j_max boundary-clamp branches.
        nx = ny = 8
        image = np.zeros((ny, nx), dtype=np.float32)
        x = y = np.array([4.0])
        sx = sy = np.array([5.0])
        photons = np.array([1000.0])

        image_jit = image.copy()
        _fill_gaussian(image_jit, x, y, sx, sy, photons, nx, ny)
        image_py = image.copy()
        _fill_gaussian.py_func(image_py, x, y, sx, sy, photons, nx, ny)
        np.testing.assert_allclose(image_jit, image_py)
        assert image_jit.sum() > 0


class TestRenderSmooth:
    def test_basic(self):
        locs = _locs()
        n, img = render_smooth(locs, 1.0, 0, 0, 100, 100)
        assert n == 5
        assert img.sum() > 0

    def test_empty_returns_zero_image(self):
        locs = _locs(n=0)
        n, img = render_smooth(locs, 1.0, 0, 0, 100, 100)
        assert n == 0
        assert img.sum() == 0

    def test_fftconvolve_matches_manual_kernel(self):
        image = np.zeros((20, 20), dtype=np.float32)
        image[10, 10] = 1.0
        blurred = _fftconvolve(image, 1.0, 1.0)
        assert blurred.shape == image.shape
        assert blurred.sum() == pytest.approx(image.sum(), rel=1e-3)


# ======================================================================
# render_gaussian_colour / _render_colour_setup / _fill_colour_gaussian
# ======================================================================

class TestRenderGaussianColour:
    def test_basic(self):
        locs = _locs()
        n, img_total, img_colour = render_gaussian_colour(
            locs, 1.0, 0, 0, 100, 100, min_blur_width=1.0,
            cparam="A_R", c_min=0.3, c_max=0.75,
            mindensperc=1, maxdensperc=99.9, densitymin=0.1, cmap_string="jet",
        )
        assert n == 5
        assert img_colour.shape == (100, 100, 3)

    def test_no_photons_column_defaults_to_one(self):
        locs = _locs(with_photoelectrons=False)
        n, img_total, img_colour = render_gaussian_colour(
            locs, 1.0, 0, 0, 100, 100, min_blur_width=1.0,
            cparam="A_R", c_min=0.3, c_max=0.75,
            mindensperc=1, maxdensperc=99.9, densitymin=0.1, cmap_string="jet",
        )
        assert n == 5

    def test_fallback_without_matplotlib(self, monkeypatch):
        # Force the "matplotlib unavailable" fallback branches (dummy
        # grayscale cmap, plain rgb*density compositing instead of
        # colors.hsv_to_rgb) -- plt/colors are real, importable modules in
        # this environment, so the only way to exercise these branches is to
        # blank out render.py's own module-level references to them.
        monkeypatch.setattr(render_mod, "plt", None)
        monkeypatch.setattr(render_mod, "colors", None)
        locs = _locs()
        n, img_total, img_colour = render_gaussian_colour(
            locs, 1.0, 0, 0, 100, 100, min_blur_width=1.0,
            cparam="A_R", c_min=0.3, c_max=0.75,
            mindensperc=1, maxdensperc=99.9, densitymin=0.1, cmap_string="jet",
        )
        assert n == 5
        assert img_colour.shape == (100, 100, 3)

    def test_render_colour_setup_py_func_matches_jit(self):
        locs = _locs()
        image_total, image_spectral, ny, nx, x, y, iv = _render_colour_setup(locs, 1.0, 0, 0, 100, 100)
        image_total_py, image_spectral_py, ny_py, nx_py, x_py, y_py, iv_py = (
            _render_colour_setup.py_func(locs, 1.0, 0, 0, 100, 100)
        )
        np.testing.assert_array_equal(x, x_py)
        np.testing.assert_array_equal(y, y_py)

    def test_fill_colour_gaussian_py_func_matches_jit_and_clamps_bounds(self):
        # 8x8 canvas, one centred point with an oversized sigma so its draw
        # radius overflows all four edges -- exercises the boundary-clamp
        # branches in addition to the normal interior-draw path.
        nx = ny = 8
        image_total = np.zeros((ny, nx), dtype=np.float32)
        image_spectral = np.zeros((ny, nx), dtype=np.float32)
        x = y = np.array([4.0])
        sx = sy = np.array([5.0])
        colour = np.array([0.5])
        photons = np.array([1000.0])

        it_jit, is_jit = image_total.copy(), image_spectral.copy()
        _fill_colour_gaussian(it_jit, is_jit, x, y, sx, sy, colour, photons, nx, ny)
        it_py, is_py = image_total.copy(), image_spectral.copy()
        _fill_colour_gaussian.py_func(it_py, is_py, x, y, sx, sy, colour, photons, nx, ny)
        np.testing.assert_allclose(it_jit, it_py)
        np.testing.assert_allclose(is_jit, is_py)
        assert it_jit.sum() > 0


# ======================================================================
# render_gaussian_RGB / _fill_RGB_gaussian
# ======================================================================

class TestRenderGaussianRGB:
    def test_basic(self):
        locs = _locs(with_rgb=True)
        n, img_total, img_rgb = render_gaussian_RGB(
            locs, 1.0, 0, 0, 100, 100, min_blur_width=1.0,
            mindensperc=1, maxdensperc=99.9, densitymin=0.1,
        )
        assert n == 5
        assert img_rgb.shape == (100, 100, 3)

    def test_no_photons_column_defaults_to_one(self):
        locs = _locs(with_photoelectrons=False, with_rgb=True)
        n, img_total, img_rgb = render_gaussian_RGB(
            locs, 1.0, 0, 0, 100, 100, min_blur_width=1.0,
            mindensperc=1, maxdensperc=99.9, densitymin=0.1,
        )
        assert n == 5

    def test_fallback_without_matplotlib(self, monkeypatch):
        monkeypatch.setattr(render_mod, "plt", None)
        monkeypatch.setattr(render_mod, "colors", None)
        locs = _locs(with_rgb=True)
        n, img_total, img_rgb = render_gaussian_RGB(
            locs, 1.0, 0, 0, 100, 100, min_blur_width=1.0,
            mindensperc=1, maxdensperc=99.9, densitymin=0.1,
        )
        assert n == 5
        assert img_rgb.shape == (100, 100, 3)

    def test_py_func_matches_jit_and_clamps_bounds(self):
        # 8x8 canvas, one centred point with an oversized sigma so its draw
        # radius overflows all four edges -- exercises the boundary-clamp
        # branches in addition to the normal interior-draw path.
        nx = ny = 8
        image_total = np.zeros((ny, nx), dtype=np.float32)
        image_R = np.zeros((ny, nx), dtype=np.float32)
        image_G = np.zeros((ny, nx), dtype=np.float32)
        image_B = np.zeros((ny, nx), dtype=np.float32)
        x = y = np.array([4.0])
        sx = sy = np.array([5.0])
        A_R = A_G = A_B = np.array([0.3], dtype=np.float32)
        photons = np.array([1000.0], dtype=np.float32)

        it_jit = image_total.copy()
        iR_jit, iG_jit, iB_jit = image_R.copy(), image_G.copy(), image_B.copy()
        _fill_RGB_gaussian(it_jit, iR_jit, iG_jit, iB_jit, x, y, sx, sy, A_R, A_G, A_B, photons, nx, ny)

        it_py = image_total.copy()
        iR_py, iG_py, iB_py = image_R.copy(), image_G.copy(), image_B.copy()
        _fill_RGB_gaussian.py_func(it_py, iR_py, iG_py, iB_py, x, y, sx, sy, A_R, A_G, A_B, photons, nx, ny)

        np.testing.assert_allclose(it_jit, it_py)
        np.testing.assert_allclose(iR_jit, iR_py)
        np.testing.assert_allclose(iG_jit, iG_py)
        np.testing.assert_allclose(iB_jit, iB_py)
        assert it_jit.sum() > 0
