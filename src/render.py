"""
    jsb92 updated render functions
    
    original code
    picasso.render
    ~~~~~~~~~~~~~~~~~~~~

    Render single molecule localizations to a super-resolution image

    :original author: Joerg Schnitzbauer, 2015
    :copyright: Copyright (c) MIT License
"""

from __future__ import annotations

from pathlib import Path
import sys

import numba
import numpy as np
import scipy.signal as signal
from numpy.typing import NDArray

sys.path.append(str(Path(__file__).parent))
from pyS3M.ImportManager import get_module

plt = get_module("matplotlib.pyplot")
colors = get_module("matplotlib.colors")

_DRAW_MAX_SIGMA = 3


def render(
    locs: np.recarray,
    info: list[dict] | None = None,
    oversampling: float = 1,
    viewport: tuple[tuple[float, float], tuple[float, float]] | None = None,
    blur_method: str | None = None,
    min_blur_width: float = 0,
    cparam: str = "A_R",
    c_min: float = 0.3,
    c_max: float = 0.75,
    mindensperc: float = 1,
    maxdensperc: float = 99.9,
    densitymin: float = 0.1,
    cmap_string: str = "jet",
) -> tuple[int, NDArray[np.float32]] | tuple[int, NDArray[np.float32], NDArray[np.float32]]:
    """
    Renders locs.

    Parameters
    ----------
    locs : np.recarray
        Localizations to be rendered
    info : dict (default=None)
        Contains metadata for locs. Needed only if no viewport
        specified
    oversampling : float (default=1)
        Number of super-resolution pixels per camera pixel
    viewport : list or tuple (default=None)
        Field of view to be rendered. If None, all locs are rendered
    blur_method : str (default=None)
        Defines localizations' blur. The string has to be one of
        'gaussian', 'gaussian_colour', 'smooth'. If None,
        no blurring is applied.
    min_blur_width : float (default=0)
        Minimum size of blur (pixels)

    Raises
    ------
    Exception
        If blur_method not one of 'gaussian', 'gaussian_colour', 'smooth' or None

    Returns
    -------
    int
        Number of localizations rendered
    np.array
        Rendered image
    """

    if viewport is None:
        try:
            # all locs
            viewport = [(0, 0), (info[0]["Height"], info[0]["Width"])]
        except TypeError:
            raise ValueError("Need info if no viewport is provided.")
    (y_min, x_min), (y_max, x_max) = viewport
    if blur_method is None:
        # no blur
        return render_hist(
            locs,
            oversampling,
            y_min,
            x_min,
            y_max,
            x_max,
        )
    elif blur_method == "gaussian":
        # individual localization precision
        return render_gaussian(
            locs,
            oversampling,
            y_min,
            x_min,
            y_max,
            x_max,
            min_blur_width,
        )
    elif blur_method == "gaussian_RGB":
        return render_gaussian_RGB(
            locs,
            oversampling,
            y_min,
            x_min,
            y_max,
            x_max,
            min_blur_width,
            mindensperc,
            maxdensperc,
            densitymin,
        )
    elif blur_method == "gaussian_colour":
        return render_gaussian_colour(
            locs,
            oversampling,
            y_min,
            x_min,
            y_max,
            x_max,
            min_blur_width,
            cparam,
            c_min,
            c_max,
            mindensperc,
            maxdensperc,
            densitymin,
            cmap_string,
        )
    elif blur_method == "smooth":
        # one pixel blur
        return render_smooth(
            locs,
            oversampling,
            y_min,
            x_min,
            y_max,
            x_max,
        )
    else:
        raise Exception("blur_method not understood.")


@numba.njit
def _render_colour_setup(
    locs,
    oversampling,
    y_min,
    x_min,
    y_max,
    x_max,
):
    """
    Finds coordinates to be rendered and sets up an empty image array.

    Parameters
    ----------
    locs : np.recarray
        Localizations
    oversampling : float
        Number of super-resolution pixels per camera pixel
    y_min : float
        Minimum y coordinate to be rendered (pixels)
    x_min : float
        Minimum x coordinate to be rendered (pixels)
    y_max : float
        Maximum y coordinate to be rendered (pixels)
    x_max : float
        Maximum x coordinate to be rendered (pixels)

    Returns
    -------
    np.array
        Empty image array
    int
        Number of pixels in y
    int
        Number of pixels in x
    np.array
        x coordinates to be rendered
    np.array
        y coordinates to be rendered
    np.array
        Indeces of locs to be rendered
    """

    n_pixel_y = int(np.ceil(oversampling * (y_max - y_min)))
    n_pixel_x = int(np.ceil(oversampling * (x_max - x_min)))
    x = locs.xc
    y = locs.yc
    in_view = (x > x_min) & (y > y_min) & (x < x_max) & (y < y_max)
    x = x[in_view]
    y = y[in_view]
    x = oversampling * (x - x_min)
    y = oversampling * (y - y_min)
    image_total = np.zeros((n_pixel_y, n_pixel_x), dtype=np.float32)
    image_spectral = np.zeros((n_pixel_y, n_pixel_x), dtype=np.float32)
    return image_total, image_spectral, n_pixel_y, n_pixel_x, x, y, in_view


@numba.njit
def _render_setup(
    locs,
    oversampling,
    y_min,
    x_min,
    y_max,
    x_max,
):
    """
    Finds coordinates to be rendered and sets up an empty image array.

    Parameters
    ----------
    locs : np.recarray
        Localizations
    oversampling : float
        Number of super-resolution pixels per camera pixel
    y_min : float
        Minimum y coordinate to be rendered (pixels)
    x_min : float
        Minimum x coordinate to be rendered (pixels)
    y_max : float
        Maximum y coordinate to be rendered (pixels)
    x_max : float
        Maximum x coordinate to be rendered (pixels)

    Returns
    -------
    np.array
        Empty image array
    int
        Number of pixels in y
    int
        Number of pixels in x
    np.array
        x coordinates to be rendered
    np.array
        y coordinates to be rendered
    np.array
        Indeces of locs to be rendered
    """

    n_pixel_y = int(np.ceil(oversampling * (y_max - y_min)))
    n_pixel_x = int(np.ceil(oversampling * (x_max - x_min)))
    x = locs.xc
    y = locs.yc
    in_view = (x > x_min) & (y > y_min) & (x < x_max) & (y < y_max)
    x = x[in_view]
    y = y[in_view]
    x = oversampling * (x - x_min)
    y = oversampling * (y - y_min)
    image = np.zeros((n_pixel_y, n_pixel_x), dtype=np.float32)
    return image, n_pixel_y, n_pixel_x, x, y, in_view


@numba.njit
def _fill(image, x, y):
    """
    Fills image with x and y coordinates.
    Image is not blurred.

    Parameters
    ----------
    image : np.array
        Empty image array
    x : np.array
        x coordinates to be rendered
    y : np.array
        y coordinates to be rendered
    """

    x = x.astype(np.int32)
    y = y.astype(np.int32)
    for i, j in zip(x, y):
        image[j, i] += 1


@numba.njit
def _fill_colour_gaussian(
    image_total, image_spectral, x, y, sx, sy, colour, photons, n_pixel_x, n_pixel_y
):
    """
    Fills image with blurred x and y coordinates.
    Localization precisions (sx and sy) are treated as standard
    deviations of the guassians to be rendered.

    Parameters
    ----------
    image : np.array
        Empty image array
    x : np.array
        x coordinates to be rendered
    y : np.array
        y coordinates to be rendered
    sx : np.array
        Localization precision in x for each loc
    sy : np.array
        Localization precision in y for each loc
    colour: np.array
        Colour information for each loc
    photons : np.array
        Photon count for each loc (area under Gaussian)
    n_pixel_x : int
        Number of pixels in x
    n_pixel_y : int
        Number of pixels in y
    """

    # render each localization separately
    for x_, y_, sx_, sy_, colour_, photons_ in zip(x, y, sx, sy, colour, photons):

        # get min and max indeces to draw the given localization
        max_y = _DRAW_MAX_SIGMA * sy_
        i_min = np.int32(y_ - max_y)
        if i_min < 0:
            i_min = 0
        i_max = np.int32(y_ + max_y + 1)
        if i_max > n_pixel_y:
            i_max = n_pixel_y
        max_x = _DRAW_MAX_SIGMA * sx_
        j_min = np.int32(x_ - max_x)
        if j_min < 0:
            j_min = 0
        j_max = np.int32(x_ + max_x) + 1
        if j_max > n_pixel_x:
            j_max = n_pixel_x

        # draw a localization as a 2D gaussian weighted by its photoelectrons
        # Area under Gaussian = photons_ (the weight; not normalized to 1.0)
        for i in range(i_min, i_max):
            for j in range(j_min, j_max):
                val = photons_ * np.exp(
                    -(
                        (j - x_ + 0.5) ** 2 / (2 * sx_**2)
                        + (i - y_ + 0.5) ** 2 / (2 * sy_**2)
                    )
                ) / (2 * np.pi * sx_ * sy_)
                image_total[i, j] += val
                image_spectral[i, j] += val * colour_


@numba.njit
def _fill_gaussian(image, x, y, sx, sy, photons, n_pixel_x, n_pixel_y):
    """
    Fills image with blurred x and y coordinates.
    Localization precisions (sx and sy) are treated as standard
    deviations of the guassians to be rendered.

    Parameters
    ----------
    image : np.array
        Empty image array
    x : np.array
        x coordinates to be rendered
    y : np.array
        y coordinates to be rendered
    sx : np.array
        Localization precision in x for each loc
    sy : np.array
        Localization precision in y for each loc
    photons : np.array
        Photon count for each loc (area under Gaussian)
    n_pixel_x : int
        Number of pixels in x
    n_pixel_y : int
        Number of pixels in y
    """

    # render each localization separately
    for x_, y_, sx_, sy_, photons_ in zip(x, y, sx, sy, photons):

        # get min and max indeces to draw the given localization
        max_y = _DRAW_MAX_SIGMA * sy_
        i_min = np.int32(y_ - max_y)
        if i_min < 0:
            i_min = 0
        i_max = np.int32(y_ + max_y + 1)
        if i_max > n_pixel_y:
            i_max = n_pixel_y
        max_x = _DRAW_MAX_SIGMA * sx_
        j_min = np.int32(x_ - max_x)
        if j_min < 0:
            j_min = 0
        j_max = np.int32(x_ + max_x) + 1
        if j_max > n_pixel_x:
            j_max = n_pixel_x

        # draw a localization as a 2D gaussian weighted by its photoelectrons
        # Area under Gaussian = photons_ (the weight; not normalized to 1.0)
        for i in range(i_min, i_max):
            for j in range(j_min, j_max):
                image[i, j] += photons_ * np.exp(
                    -(
                        (j - x_ + 0.5) ** 2 / (2 * sx_**2)
                        + (i - y_ + 0.5) ** 2 / (2 * sy_**2)
                    )
                ) / (2 * np.pi * sx_ * sy_)


def render_hist(
    locs: np.recarray,
    oversampling: float,
    y_min: float,
    x_min: float,
    y_max: float,
    x_max: float,
) -> tuple[int, NDArray[np.float32]]:
    """
    Renders locs with no blur.

    Parameters
    ----------
    locs : np.recarray
        Localizations to be rendered
    oversampling : float (default=1)
        Number of super-resolution pixels per camera pixel
    y_min : float
        Minimum y coordinate to be rendered (pixels)
    x_min : float
        Minimum x coordinate to be rendered (pixels)
    y_max : float
        Maximum y coordinate to be rendered (pixels)
    x_max : float
        Maximum x coordinate to be rendered (pixels)

    Returns
    -------
    int
        Number of localizations rendered
    np.array
        Rendered image
    """

    image, n_pixel_y, n_pixel_x, x, y, in_view = _render_setup(
        locs,
        oversampling,
        y_min,
        x_min,
        y_max,
        x_max,
    )
    _fill(image, x, y)
    return len(x), image


def render_gaussian_colour(
    locs: np.recarray,
    oversampling: float,
    y_min: float,
    x_min: float,
    y_max: float,
    x_max: float,
    min_blur_width: float,
    cparam: str,
    c_min: float,
    c_max: float,
    mindensperc: float,
    maxdensperc: float,
    densitymin: float,
    cmap_string: str,
) -> tuple[int, NDArray[np.float32], NDArray[np.float32]]:
    """
    Renders locs with with individual localization precision which
    differs in x and y. Renders a spectral image based on a colour parameter.

    Parameters
    ----------
    locs : np.recarray
        Localizations to be rendered
    oversampling : float (default=1)
        Number of super-resolution pixels per camera pixel
    y_min : float
        Minimum y coordinate to be rendered (pixels)
    x_min : float
        Minimum x coordinate to be rendered (pixels)
    y_max : float
        Maximum y coordinate to be rendered (pixels)
    x_max : float
        Maximum x coordinate to be rendered (pixels)
    min_blur_width : float
        Minimum localization precision (pixels)
    ang : tuple (default=None)
        Rotation angles of locs around x, y and z axes. If None,
        locs are not rotated.

    Returns
    -------
    int
        Number of localizations rendered
    np.array
        Rendered image
    """

    image_total, image_spectral, n_pixel_y, n_pixel_x, x, y, in_view = (
        _render_colour_setup(
            locs,
            oversampling,
            y_min,
            x_min,
            y_max,
            x_max,
        )
    )

    blur_width = oversampling * np.maximum(locs.xc_err, min_blur_width)
    blur_height = oversampling * np.maximum(locs.yc_err, min_blur_width)
    sy = blur_height[in_view]
    sx = blur_width[in_view]
    color = locs[cparam][in_view]

    # Weight each localisation by its detected photoelectrons (1.0 if not available)
    if hasattr(locs, 'photoelectrons'):
        photons = locs.photoelectrons[in_view]
    else:
        photons = np.ones(len(x), dtype=np.float32)

    _fill_colour_gaussian(
        image_total, image_spectral, x, y, sx, sy, color, photons, n_pixel_x, n_pixel_y
    )

    non_zero = image_total > 0
    image_spectral[non_zero] = image_spectral[non_zero] / image_total[non_zero]
    image_spectral[~non_zero] = 0

    min_density = np.percentile(image_total, mindensperc)
    max_density = np.percentile(image_total, maxdensperc)
    if plt:
        cmap = plt.get_cmap(cmap_string)
    else:
        # Simple fallback colormap when matplotlib not available
        def cmap(x):
            # Create a simple grayscale colormap
            return np.stack([x, x, x, np.ones_like(x)], axis=-1)

    normalised_density = np.clip(
        (image_total - min_density) / (max_density - min_density), 0, 1
    )
    # Suppress isolated noise below the density floor before colour mapping
    normalised_density[normalised_density < densitymin] = 0.0

    normalised_wl = np.clip((image_spectral - c_min) / (c_max - c_min), 0, 1)
    rgb = cmap(normalised_wl)[..., :3]
    if colors:
        hsv = colors.rgb_to_hsv(rgb)
        hsv[..., 2] = normalised_density  # brightness = density: fades to black
        image_colour_gaussian = colors.hsv_to_rgb(hsv)
    else:
        # Fallback when matplotlib.colors not available
        image_colour_gaussian = rgb * normalised_density[..., None]

    return len(x), image_total, image_colour_gaussian


@numba.njit
def _fill_RGB_gaussian(
    image_total, image_R, image_G, image_B,
    x, y, sx, sy, A_R, A_G, A_B, photons,
    n_pixel_x, n_pixel_y,
):
    for x_, y_, sx_, sy_, r_, g_, b_, ph_ in zip(x, y, sx, sy, A_R, A_G, A_B, photons):
        max_y = _DRAW_MAX_SIGMA * sy_
        i_min = np.int32(y_ - max_y)
        if i_min < 0:
            i_min = 0
        i_max = np.int32(y_ + max_y + 1)
        if i_max > n_pixel_y:
            i_max = n_pixel_y
        max_x = _DRAW_MAX_SIGMA * sx_
        j_min = np.int32(x_ - max_x)
        if j_min < 0:
            j_min = 0
        j_max = np.int32(x_ + max_x) + 1
        if j_max > n_pixel_x:
            j_max = n_pixel_x
        for i in range(i_min, i_max):
            for j in range(j_min, j_max):
                val = ph_ * np.exp(
                    -(
                        (j - x_ + 0.5) ** 2 / (2 * sx_**2)
                        + (i - y_ + 0.5) ** 2 / (2 * sy_**2)
                    )
                ) / (2 * np.pi * sx_ * sy_)
                image_total[i, j] += val
                image_R[i, j] += val * r_
                image_G[i, j] += val * g_
                image_B[i, j] += val * b_


def render_gaussian_RGB(
    locs: np.recarray,
    oversampling: float,
    y_min: float,
    x_min: float,
    y_max: float,
    x_max: float,
    min_blur_width: float = 1.0,
    mindensperc: float = 1,
    maxdensperc: float = 99.9,
    densitymin: float = 0.1,
) -> tuple[int, NDArray[np.float32], NDArray[np.float32]]:
    """Render locs into a true-colour RGB image using A_R, A_G, A_B fractions.

    Each pixel's hue is the photoelectron-weighted mean (A_R, A_G, A_B) of the
    localisations that contribute to it; brightness is modulated by the
    local density, fading to black where density is low.  This gives
    spectrally correct colours — A_G-dominant data appears green, not blue.

    Returns
    -------
    int
        Number of localisations rendered.
    np.ndarray (H, W)
        Total-photoelectron density image.
    np.ndarray (H, W, 3)
        RGB image, float32 in [0, 1].
    """
    image_total, _, n_pixel_y, n_pixel_x, x, y, in_view = _render_colour_setup(
        locs, oversampling, y_min, x_min, y_max, x_max
    )
    image_R = np.zeros((n_pixel_y, n_pixel_x), dtype=np.float32)
    image_G = np.zeros((n_pixel_y, n_pixel_x), dtype=np.float32)
    image_B = np.zeros((n_pixel_y, n_pixel_x), dtype=np.float32)

    blur_width  = oversampling * np.maximum(locs.xc_err, min_blur_width)
    blur_height = oversampling * np.maximum(locs.yc_err, min_blur_width)
    sx = blur_width[in_view]
    sy = blur_height[in_view]
    ch_R = locs.A_R[in_view].astype(np.float32)
    ch_G = locs.A_G[in_view].astype(np.float32)
    ch_B = locs.A_B[in_view].astype(np.float32)
    photons = locs.photoelectrons[in_view].astype(np.float32) if hasattr(locs, "photoelectrons") \
              else np.ones(len(x), dtype=np.float32)

    _fill_RGB_gaussian(
        image_total, image_R, image_G, image_B,
        x, y, sx, sy, ch_R, ch_G, ch_B, photons,
        n_pixel_x, n_pixel_y,
    )

    non_zero = image_total > 0
    for ch in (image_R, image_G, image_B):
        ch[non_zero] /= image_total[non_zero]
        ch[~non_zero] = 0.0

    min_density = np.percentile(image_total, mindensperc)
    max_density = np.percentile(image_total, maxdensperc)
    normalised_density = np.clip(
        (image_total - min_density) / (max_density - min_density + 1e-9), 0, 1
    )
    normalised_density[normalised_density < densitymin] = 0.0

    rgb = np.stack([image_R, image_G, image_B], axis=-1)
    if colors:
        hsv = colors.rgb_to_hsv(rgb)
        hsv[..., 2] = normalised_density
        image_rgb = colors.hsv_to_rgb(hsv).astype(np.float32)
    else:
        image_rgb = (rgb * normalised_density[..., np.newaxis]).astype(np.float32)

    return len(x), image_total, image_rgb


def render_gaussian(
    locs: np.recarray,
    oversampling: float,
    y_min: float,
    x_min: float,
    y_max: float,
    x_max: float,
    min_blur_width: float,
) -> tuple[int, NDArray[np.float32]]:
    """
    Renders locs with with individual localization precision which
    differs in x and y.

    Parameters
    ----------
    locs : np.recarray
        Localizations to be rendered
    oversampling : float (default=1)
        Number of super-resolution pixels per camera pixel
    y_min : float
        Minimum y coordinate to be rendered (pixels)
    x_min : float
        Minimum x coordinate to be rendered (pixels)
    y_max : float
        Maximum y coordinate to be rendered (pixels)
    x_max : float
        Maximum x coordinate to be rendered (pixels)
    min_blur_width : float
        Minimum localization precision (pixels)
    ang : tuple (default=None)
        Rotation angles of locs around x, y and z axes. If None,
        locs are not rotated.

    Returns
    -------
    int
        Number of localizations rendered
    np.array
        Rendered image
    """

    image, n_pixel_y, n_pixel_x, x, y, in_view = _render_setup(
        locs,
        oversampling,
        y_min,
        x_min,
        y_max,
        x_max,
    )

    blur_width = oversampling * np.maximum(locs.xc_err, min_blur_width)
    blur_height = oversampling * np.maximum(locs.yc_err, min_blur_width)
    sy = blur_height[in_view]
    sx = blur_width[in_view]

    # Weight each localisation by its detected photoelectrons (1.0 if not available)
    if hasattr(locs, 'photoelectrons'):
        photons = locs.photoelectrons[in_view]
    else:
        photons = np.ones(len(x), dtype=np.float32)

    _fill_gaussian(image, x, y, sx, sy, photons, n_pixel_x, n_pixel_y)

    return len(x), image


def render_smooth(
    locs: np.recarray,
    oversampling: float,
    y_min: float,
    x_min: float,
    y_max: float,
    x_max: float,
) -> tuple[int, NDArray[np.float32]]:
    """
    Renders locs with with blur of one display pixel (set by
    oversampling)

    Parameters
    ----------
    locs : np.recarray
        Localizations to be rendered
    oversampling : float (default=1)
        Number of super-resolution pixels per camera pixel
    y_min : float
        Minimum y coordinate to be rendered (pixels)
    x_min : float
        Minimum x coordinate to be rendered (pixels)
    y_max : float
        Maximum y coordinate to be rendered (pixels)
    x_max : float
        Maximum x coordinate to be rendered (pixels)
    ang : tuple (default=None)
        Rotation angles of locs around x, y and z axes. If None,
        locs are not rotated.

    Returns
    -------
    int
        Number of localizations rendered
    np.array
        Rendered image
    """

    image, n_pixel_y, n_pixel_x, x, y, in_view = _render_setup(
        locs,
        oversampling,
        y_min,
        x_min,
        y_max,
        x_max,
    )

    n = len(x)
    if n == 0:
        return 0, image
    else:
        _fill(image, x, y)
        return n, _fftconvolve(image, 1, 1)


def _fftconvolve(image, blur_width, blur_height):
    """
    Blurs (convolves) 2D image using fast fourier transform.

    Parameters
    ----------
    image : np.array
        Image with renderd but not blurred locs
    blur_width : float
        Blur width
    blur_height
        Blur height

    Returns
    -------
    np.array
        Blurred image
    """

    kernel_width = 10 * int(np.round(blur_width)) + 1
    kernel_height = 10 * int(np.round(blur_height)) + 1
    kernel_y = signal.windows.gaussian(kernel_height, blur_height)
    kernel_x = signal.windows.gaussian(kernel_width, blur_width)
    kernel = np.outer(kernel_y, kernel_x)
    kernel /= kernel.sum()
    return signal.fftconvolve(image, kernel, mode="same")
