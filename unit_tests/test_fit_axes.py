"""Axis convention of the fitted localisations.

Convention (see Constants.ResultColumns): images are numpy arrays indexed ``[row, col]``;
``xc`` is the column and ``yc`` the row, in full-frame pixels; ``s_x`` is the PSF width along
x (columns), ``s_y`` along y (rows); ``*_err`` follow their parameter; ELLIPTICAL's ``theta``
is the angle of the ``s_x`` axis from +x towards +y (y down the image).

The spots here are written straight into ``img[row, col]`` with numpy, independently of the
gaussoptfuncs models, and placed so that every x quantity differs from its y counterpart: a
swap anywhere (ROI origin, parameter order, error order) fails a test.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from pyS3M.ImageAnalysisFunctions import (
    CircularFittingProcessor,
    EllipticalFittingProcessor,
    FittingStrategy,
    Image_Analysis_Functions,
    NoColourFittingProcessor,
    StandardDataFittingProcessor,
)

SIZE = 14
BG = 5.0


def _bayer(shape):
    rows, cols = shape
    masks = np.zeros((rows, cols, 3), dtype=np.bool_)
    masks[0::2, 0::2, 0] = True
    masks[0::2, 1::2, 1] = masks[1::2, 0::2, 1] = True
    masks[1::2, 1::2, 2] = True
    return masks


def _spot(shape, x, y, s_x, s_y, n=2e5, phi=0.0):
    """Pixel-centre sampled Gaussian in img[row, col]; its s_x axis at phi from +x towards +y."""
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float64)
    dx, dy = cols - x, rows - y
    u = dx * np.cos(phi) + dy * np.sin(phi)
    v = -dx * np.sin(phi) + dy * np.cos(phi)
    g = np.exp(-0.5 * ((u / s_x) ** 2 + (v / s_y) ** 2))
    return (n * g / (2 * np.pi * s_x * s_y)).astype(np.float32)


def _fit(processor, img, origin, masks=True):
    from scipy.ndimage import gaussian_filter
    smoothed = gaussian_filter(img, 1.0).astype(np.float32)
    weights = (1.0 / (img + 2.0)).astype(np.float32)
    kw = {"masks": _bayer(img.shape)} if masks else {}
    return processor.fit_single_punctum(img, smoothed, weights, list(origin), **kw)


ORIGIN = (40.0, 10.0)            # ROI origin: column 40, row 10
X, Y, SX, SY = 6.3, 7.6, 1.0, 1.7  # within the ROI: x = column, y = row


class TestOutputAxes:
    @pytest.mark.parametrize("processor", [StandardDataFittingProcessor(readnoise=1.0)])
    def test_standard_position_width_and_errors(self, processor):
        img = _spot((SIZE, SIZE), X, Y, SX, SY) + BG
        pfit, perr = _fit(processor, img, ORIGIN)
        xc, yc, s_x, s_y = pfit[:4]
        assert xc == pytest.approx(ORIGIN[0] + X, abs=0.02)
        assert yc == pytest.approx(ORIGIN[1] + Y, abs=0.02)
        assert s_x == pytest.approx(SX, abs=0.02)
        assert s_y == pytest.approx(SY, abs=0.02)
        # the wider axis is the less precise one: errors follow their own parameter
        assert perr[1] > 1.3 * perr[0]   # yc_err > xc_err
        assert perr[3] > 1.15 * perr[2]  # s_y_err > s_x_err

    def test_circular_position(self):
        img = _spot((SIZE, SIZE), X, Y, 1.3, 1.3) + BG
        pfit, _ = _fit(CircularFittingProcessor(readnoise=1.0), img, ORIGIN)
        assert pfit[0] == pytest.approx(ORIGIN[0] + X, abs=0.02)
        assert pfit[1] == pytest.approx(ORIGIN[1] + Y, abs=0.02)

    def test_nocolour_position_and_width(self):
        img = _spot((SIZE, SIZE), X, Y, SX, SY) + BG
        pfit, _ = _fit(NoColourFittingProcessor(), img, ORIGIN, masks=False)
        assert pfit[0] == pytest.approx(ORIGIN[0] + X, abs=0.02)
        assert pfit[1] == pytest.approx(ORIGIN[1] + Y, abs=0.02)
        assert pfit[2] == pytest.approx(SX, abs=0.05)
        assert pfit[3] == pytest.approx(SY, abs=0.05)

    def test_elliptical_axis_aligned(self):
        img = _spot((SIZE, SIZE), X, Y, SX, SY) + BG
        pfit, _ = _fit(EllipticalFittingProcessor(), img, ORIGIN)
        assert pfit[0] == pytest.approx(ORIGIN[0] + X, abs=0.02)
        assert pfit[1] == pytest.approx(ORIGIN[1] + Y, abs=0.02)
        # theta near 0 (mod pi) keeps s_x on the column axis
        k = np.round(pfit[4] / (np.pi / 2))
        s_x, s_y = (pfit[2], pfit[3]) if k % 2 == 0 else (pfit[3], pfit[2])
        assert s_x == pytest.approx(SX, abs=0.03)
        assert s_y == pytest.approx(SY, abs=0.03)

    @pytest.mark.parametrize("phi", [0.5, -0.5])
    def test_elliptical_theta_direction(self, phi):
        img = _spot((SIZE, SIZE), 6.8, 6.4, 1.0, 2.0, phi=phi) + BG
        pfit, _ = _fit(EllipticalFittingProcessor(), img, ORIGIN)
        s_x, s_y, theta = pfit[2], pfit[3], pfit[4]
        major = theta if s_x > s_y else theta + np.pi / 2   # direction of the major axis
        true_major = phi + np.pi / 2                         # the spot's s_y (=2.0) axis
        d = (major - true_major + np.pi / 2) % np.pi - np.pi / 2
        assert abs(d) < 0.05


class TestFullPipelineAxes:
    """Detection-to-fit through SR_Functions' ROI extraction, on a non-square frame."""

    def test_spots_land_on_their_columns_and_rows(self):
        from pyS3M import SR_Functions, sCMOSFunctions

        height, width = 40, 64
        truth = [(12.3, 9.8), (45.7, 27.2), (30.1, 14.6)]   # (x = column, y = row)
        img = np.full((height, width), BG, dtype=np.float32)
        for x, y in truth:
            img += _spot((height, width), x, y, 1.2, 1.2, n=5e4)
        raw = img[np.newaxis]
        detected = np.array([[int(round(y)), int(round(x)), 0] for x, y in truth])
        # put each detection one pixel off along a different axis, so the ROI origins differ
        detected[0, 0] += 1
        detected[1, 1] -= 1

        SR = SR_Functions.SuperRes_Functions()
        sC = sCMOSFunctions.sCMOS_Functions()
        sf = types.SimpleNamespace(args={"sigma": 1.5}, extent=1.5,
                                   smoothing_function=sC.gaussian_filter_stack, data_arg="image")
        puncta, smoothed, masks, weights, coords, planes, _, readnoise = (
            SR._process_detected_puncta_batch(
                raw, detected, width, height, 12, sf, 1.0, _bayer((height, width)),
                gain_map=1.0, offset_map=0.0, rqe=1.0, frame_offset=0, is_multi_frame=True,
            )
        )
        pfit, _ = Image_Analysis_Functions(fit_gate=None).fit_puncta_method(
            puncta, smoothed, weights, coords, planes, FittingStrategy.STANDARD_DATA,
            masks=masks, readnoise_maps=readnoise,
        )
        for (x, y), row in zip(truth, pfit):
            assert row[0] == pytest.approx(x, abs=0.02)
            assert row[1] == pytest.approx(y, abs=0.02)


class TestSimulationRoundTrip:
    """gen_camera_image_stack places x0y0[..., 0, :] on the columns; the fit must return it as xc."""

    def test_simulated_spot_is_fitted_where_it_was_placed(self):
        from pyS3M import MaskFunctions, SpectralFunctions, sCMOSFunctions
        from pyS3M.simulation.multicolour import MultiC_Sim_Funcs_Refactored

        sim = MultiC_Sim_Funcs_Refactored(camera="ximea")
        R, G, B, wl = SpectralFunctions.Spectral_Funcs().getpixelefficiency()
        height, width = 14, 20
        cam = {"gain": np.ones((height, width)), "offset": np.full((height, width), 100.0),
               "variance": np.full((height, width), 1e-12), "readnoise": 1e-6,
               "rqe": np.ones((height, width)), "pixel_QYs": np.vstack([B, G, R]),
               "masks": MaskFunctions.Mask_Functions().get_masks(size_x=height, size_y=width),
               "pixel_order": ["B", "G", "R"], "pixel_order_indices": {"B": 0, "G": 1, "R": 2},
               "mosaic_unit": None}
        px_nm = sim.pixel_size * 1000
        x_true, y_true = 12.3, 6.6
        x0y0 = {"dye": np.zeros((1, 2, 1))}
        x0y0["dye"][0, 0, 0] = x_true * px_nm
        x0y0["dye"][0, 1, 0] = y_true * px_nm
        sf = types.SimpleNamespace(args={"sigma": 0.85}, data_arg="image", extent=0.85,
                                   smoothing_function=sCMOSFunctions.sCMOS_Functions().gaussian_filter_stack)
        _, _, pe = sim.gen_camera_image_stack(
            cam, wl, 600.0, np.array([0.3, 0.3, 0.4]), {"dye": np.array([1e6])}, x0y0,
            smoothing_function=sf, return_normal_image=True, return_photoelectrons=True,
            emission_spectra={"dye": np.exp(-0.5 * ((wl - 600.0) / 20.0) ** 2)},
        )
        img = np.asarray(pe, dtype=np.float32).reshape(height, width)
        roi = img[0:14, 6:20]   # square ROI at column 6, row 0
        pfit, _ = _fit(NoColourFittingProcessor(), roi + BG, (6.0, 0.0), masks=False)
        assert pfit[0] == pytest.approx(x_true, abs=0.05)
        assert pfit[1] == pytest.approx(y_true, abs=0.05)
