#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 14:55:31 2025

@author: jbeckwith
"""
import numpy as np
from numba import jit


@jit(nopython=True, nogil=True, cache=True)
def gaussian_unscaled_model(
    array_tofill: np.ndarray,
    x: np.ndarray,
    size: int,
    x0: float,
    y0: float,
    sigma_x: float,
    sigma_y: float,
) -> np.ndarray:
    """gauss2d: Returns 2D gaussian

    Args:
        array_tofill (np.ndarray): array to be filled
        x (np.1darray): x array
        size (int): how big the array is
        x0 (float): x-coordinate of the center of the gaussian
        y0 (float): y-coordinate of the center of the gaussian
        sigma (float): standard deviation of the gaussian

    Returns:
        signal (float): signal at particular (x, y) location"""
    norm_x = 0.3989422804014327 / sigma_x  # 1/√(2π)
    norm_y = 0.3989422804014327 / sigma_y  # 1/√(2π)
    xg = norm_x * np.exp(-0.5 * ((x - x0) / sigma_x) ** 2)
    yg = norm_y * np.exp(-0.5 * ((x - y0) / sigma_y) ** 2)
    for i in range(size):
        for j in range(size):
            array_tofill[i, j] = xg[i] * yg[j]
    return array_tofill


@jit(nopython=True, nogil=True, cache=True)
def WLS_justcolour_model_nobounds(
    params,
    x,
    gauss_2d,
    locparams,
):
    """
    Calculate a coloured gaussian based on input params.


    Args:
        params (numpy.ndarray): Input parameters.
        data (numpy.ndarray): Data to fit.
        masks (np.3darray): 3d array of colour masks

    Returns:
        gauss_2d (numpy.ndarray): 2d bayer filtered gaussian.
    """

    gauss_2d[:, :] = (
        np.multiply(
            params[0] ** 2,
            gaussian_unscaled_model(
                gauss_2d[:, :],
                x,
                len(x),
                locparams[0],
                locparams[1],
                locparams[2],
                locparams[3],
            ),
        )
        + params[1] ** 2
    )
    return gauss_2d


@jit(nopython=True, nogil=True, cache=True)
def WLS_nocolour_model_nobounds(
    params,
    data,
    x,
    gauss_2d,
):
    """
    Calculate a coloured gaussian based on input params.


    Args:
        params (numpy.ndarray): Input parameters.
        data (numpy.ndarray): Data to fit.
        masks (np.3darray): 3d array of colour masks

    Returns:
        gauss_2d (numpy.ndarray): 2d bayer filtered gaussian.
    """

    gauss_2d[:, :] = (
        np.multiply(
            params[5] ** 2,
            gaussian_unscaled_model(
                gauss_2d[:, :],
                x,
                len(x),
                params[0],
                params[1],
                params[2],
                params[3],
            ),
        )
        + params[4] ** 2
    )
    return gauss_2d


@jit(nopython=True, nogil=True, cache=True)
def WLS_rawcolour_model_nobounds(
    params,
    data,
    masks,
    background_bayer_matrix,
    bayer_matrix,
    x,
    gauss_2d,
    locparams,
):
    """
    Calculate a coloured gaussian based on input params.


    Args:
        params (numpy.ndarray): Input parameters.
        data (numpy.ndarray): Data to fit.
        masks (np.3darray): 3d array of colour masks

    Returns:
        gauss_2d (numpy.ndarray): 2d bayer filtered gaussian.
    """

    for i in np.arange(masks.shape[-1]):
        pixels = masks[:, :, i].ravel()
        bayer_matrix[pixels] = params[-3 + i] ** 2
        background_bayer_matrix[pixels] = params[-6 + i] ** 2
    bayer_matrix = bayer_matrix.reshape(len(x), len(x))
    background_bayer_matrix = background_bayer_matrix.reshape(len(x), len(x))
    gauss_2d[:, :] = (
        np.multiply(
            bayer_matrix,
            gaussian_unscaled_model(
                gauss_2d[:, :],
                x,
                len(x),
                locparams[0],
                locparams[1],
                locparams[2],
                locparams[3],
            ),
        )
        + background_bayer_matrix
    )
    return gauss_2d


@jit(nopython=True, nogil=True, cache=True)
def WLS_model_nobounds(
    params,
    masks,
    x,
    gauss_2d,
):
    """
    Calculate a coloured gaussian based on input params.


    Args:
        params (numpy.ndarray): Input parameters.
        masks (np.3darray): 3d array of colour masks
        x (numpy.ndarray): coordinate array
        gauss_2d (numpy.ndarray): output array to fill

    Returns:
        gauss_2d (numpy.ndarray): 2d bayer filtered gaussian.
    """

    n_ch = masks.shape[-1]
    len_x = len(x)

    # Build per-channel lookup tables from params layout: [x,y,sy,sx, bg0,...,bg_{n_ch-1}, A0,...,A_{n_ch-1}]
    bg_lookup = np.zeros(n_ch)
    amp_lookup = np.zeros(n_ch)
    for i in range(n_ch):
        bg_lookup[i] = params[4 + i] * params[4 + i]
        amp_lookup[i] = params[4 + n_ch + i] * params[4 + n_ch + i]

    # First compute the Gaussian using the exact same method as gaussian_unscaled_model
    gauss_2d = gaussian_unscaled_model(
        gauss_2d, x, len_x, params[0], params[1], params[2], params[3]
    )

    # Apply colour pattern per channel
    for channel in range(n_ch):
        for i in range(len_x):
            for j in range(len_x):
                if masks[j, i, channel]:
                    gauss_2d[j, i] = (
                        amp_lookup[channel] * gauss_2d[j, i] + bg_lookup[channel]
                    )

    return gauss_2d


@jit(nopython=True, nogil=True, cache=True)
def WLS_chi_nobounds(params, data, masks, weights, size, ravelsize):
    """
    Calculate the chi vector for the weighted least squares model.

    Args:
        params (numpy.ndarray): Input parameters.
        data (numpy.ndarray): Data to fit.
        masks (np.3darray): 3d array of colour masks
        weights (np.ndarray): weights for the chi

    Returns:
        chi (numpy.ndarray): Vector of chi.
    """
    x = np.arange(size, dtype=np.float32)
    gauss_2d = np.zeros((size, size), dtype=np.float32)
    chi = np.zeros((size, size), dtype=np.float32)
    gauss_2d[:, :] = WLS_model_nobounds(params, masks, x, gauss_2d)
    chi[:, :] = np.multiply(weights, np.square(np.subtract(data, gauss_2d)))
    return np.sqrt(chi.ravel())


@jit(nopython=True, nogil=True, cache=True)
def WLS_chi_circular_nobounds(params, data, masks, weights, size, ravelsize):
    """
    Calculate the chi vector for the weighted least squares model, constrained to a
    single (circular) PSF width instead of independent sigma_x/sigma_y.

    ``params`` carries one shared width: [x, y, s, bg_0,...,bg_{n_ch-1}, A_0,...,A_{n_ch-1}]
    -- one element shorter than WLS_chi_nobounds's [x, y, sy, sx, bg..., A...]. That single
    width is duplicated into both the sigma_y and sigma_x slots before delegating to the
    existing (unchanged) WLS_model_nobounds, so the underlying coloured-Gaussian model and
    its covariance/error propagation are identical to STANDARD_DATA -- only the number of
    free width parameters the optimiser sees differs.

    Args:
        params (numpy.ndarray): Input parameters, one shared width (see above).
        data (numpy.ndarray): Data to fit.
        masks (np.3darray): 3d array of colour masks
        weights (np.ndarray): weights for the chi

    Returns:
        chi (numpy.ndarray): Vector of chi.
    """
    full_params = np.empty(len(params) + 1, dtype=params.dtype)
    full_params[0] = params[0]  # x
    full_params[1] = params[1]  # y
    full_params[2] = params[2]  # s -> sigma_y slot
    full_params[3] = params[2]  # s -> sigma_x slot
    full_params[4:] = params[3:]  # bg..., A...

    x = np.arange(size, dtype=np.float32)
    gauss_2d = np.zeros((size, size), dtype=np.float32)
    chi = np.zeros((size, size), dtype=np.float32)
    gauss_2d[:, :] = WLS_model_nobounds(full_params, masks, x, gauss_2d)
    chi[:, :] = np.multiply(weights, np.square(np.subtract(data, gauss_2d)))
    return np.sqrt(chi.ravel())


@jit(nopython=True, nogil=True, cache=True)
def WLS_chi_nocolour_nobounds(params, data, weights, size, ravelsize):
    """
    Calculate the chi vector for the weighted least squares model.

    Args:
        params (numpy.ndarray): Input parameters.
        data (numpy.ndarray): Data to fit.
        weights (np.ndarray): weights for the chi

    Returns:
        chi (numpy.ndarray): Vector of chi.
    """
    x = np.arange(size, dtype=np.float32)
    gauss_2d = np.zeros((size, size), dtype=np.float32)
    chi = np.zeros((size, size), dtype=np.float32)
    gauss_2d[:, :] = WLS_nocolour_model_nobounds(params, data, x, gauss_2d)
    chi[:, :] = np.multiply(weights, np.square(np.subtract(data, gauss_2d)))
    return np.sqrt(chi.ravel())


@jit(nopython=True, nogil=True, cache=True)
def WLS_chi_justcolour_nobounds(params, data, weights, size, locparams):
    """
    Calculate the chi vector for the weighted least squares model.

    Args:
        params (numpy.ndarray): Input parameters.
        data (numpy.ndarray): Data to fit.
        weights (np.ndarray): weights for the chi

    Returns:
        chi (numpy.ndarray): Vector of chi.
    """
    x = np.arange(size, dtype=np.float32)
    gauss_2d = np.zeros((size, size), dtype=np.float32)
    chi = np.zeros((size, size), dtype=np.float32)
    gauss_2d[:, :] = WLS_justcolour_model_nobounds(params, x, gauss_2d, locparams)
    chi[:, :] = np.multiply(weights, np.square(np.subtract(data, gauss_2d)))
    return np.sqrt(chi.ravel())


@jit(nopython=True, nogil=True, cache=True)
def WLS_rawcolour_chi_nobounds(
    params, data, masks, weights, size, ravelsize, locparams
):
    """
    Calculate the chi vector for the weighted least squares model.

    Args:
        params (numpy.ndarray): Input parameters.
        data (numpy.ndarray): Data to fit.
        masks (np.3darray): 3d array of colour masks
        weights (np.ndarray): weights for the chi

    Returns:
        chi (numpy.ndarray): Vector of chi.
    """
    x = np.arange(size, dtype=np.float32)
    background_bayer_matrix = np.zeros(ravelsize, dtype=np.float32)
    bayer_matrix = np.zeros(ravelsize, dtype=np.float32)
    gauss_2d = np.zeros((size, size), dtype=np.float32)
    chi = np.zeros((size, size), dtype=np.float32)
    gauss_2d[:, :] = WLS_rawcolour_model_nobounds(
        params,
        data,
        masks,
        background_bayer_matrix,
        bayer_matrix,
        x,
        gauss_2d,
        locparams,
    )
    chi[:, :] = np.multiply(weights, np.square(np.subtract(data, gauss_2d)))
    return np.sqrt(chi.ravel())


@jit(nopython=True, nogil=True, cache=True)
def _sum_and_centre_of_mass(smoothed_data, size):
    x_ig = 0.0
    y_ig = 0.0
    A = 0.0
    for i in range(size):
        for j in range(size):
            x_ig += smoothed_data[i, j] * i
            y_ig += smoothed_data[i, j] * j
            A += smoothed_data[i, j]
    x_ig /= A
    y_ig /= A
    return np.abs(A), np.abs(x_ig), np.abs(y_ig)


@jit(nopython=True, nogil=True, cache=True)
def _initial_sigma(smoothed_data, x_ig, y_ig, A, size):
    sum_deviation_y = 0.0
    sum_deviation_x = 0.0
    for i in range(size):
        for j in range(size):
            sum_deviation_y += smoothed_data[i, j] * (i - y_ig) ** 2
            sum_deviation_x += smoothed_data[i, j] * (j - x_ig) ** 2
    sy = np.sqrt(sum_deviation_y / A)
    sx = np.sqrt(sum_deviation_x / A)
    # Apply empirical correction factor of 0.5 to compensate for smoothing bias
    # Smoothing inflates sigma by ~50%, so divide by 2 to get closer to true value
    return np.abs(sy) * 0.5, np.abs(sx) * 0.5


@jit(nopython=True, cache=True)
def initial_guess(smoothed_data, raw_data, masks):
    """
    initial_guess of gaussian input parameters.

    Args:
        smoothed_data (np.2darray): smoothed photoelectron data matrix.
        raw_data (np.2darray): raw photoelectron data matrix
        masks (np.3darray): mask matrix

    Returns:
        x_ig (float): centroid guess in x.
        y_ig (float): centroid guess in y.
        sigma (float): sigma guess
        bB (float): background guess Blue (sqrt of actual background)
        bG (float): background guess Green (sqrt of actual background)
        bR (float): background guess Red (sqrt of actual background)
        A_ig (float): amplitude guess for all colours (sqrt of actual amplitude)

    Note:
        Background and amplitude parameters are returned as sqrt(value) because
        WLS_model_nobounds squares them. This prevents catastrophic initial guess
        errors at high photon counts (>30k) that cause LM fitting to fail.
    """
    n_ch = masks.shape[-1]
    BG_matrix = np.zeros(n_ch)
    flattened_rawdata = raw_data.ravel()
    for i in range(n_ch):
        pixels = masks[:, :, i].ravel()
        BG_matrix[i] = np.min(np.abs(flattened_rawdata[pixels]))

    ig_data = np.abs(smoothed_data)
    bs_data = ig_data - np.abs(np.min(ig_data))
    size = bs_data.shape[0]
    A, x_ig, y_ig = _sum_and_centre_of_mass(bs_data, size)
    sigma_y, sigma_x = _initial_sigma(bs_data, x_ig, y_ig, A, size)
    A_ig = A / n_ch

    # Return array: [x, y, sigma_y, sigma_x, sqrt(bg_0), ..., sqrt(bg_{n_ch-1}), sqrt(A), ..., sqrt(A)]
    # Model squares bg and A values; using sqrt prevents catastrophic errors at high photon counts.
    result = np.zeros(4 + 2 * n_ch)
    result[0] = x_ig
    result[1] = y_ig
    result[2] = sigma_y
    result[3] = sigma_x
    for i in range(n_ch):
        result[4 + i] = np.sqrt(np.abs(BG_matrix[i]))
        result[4 + n_ch + i] = np.sqrt(np.abs(A_ig))
    return result


@jit(nopython=True, nogil=True, cache=True)
def gaussian_unscaled_model_elliptical(
    array_tofill: np.ndarray,
    x: np.ndarray,
    size: int,
    x0: float,
    y0: float,
    sigma_x: float,
    sigma_y: float,
    theta: float,
) -> np.ndarray:
    """2D rotated elliptical Gaussian.

    Same coordinate convention as gaussian_unscaled_model: x0 is the row-centre,
    y0 is the column-centre, sigma_x is the half-width along the rotated row axis,
    sigma_y is the half-width along the rotated column axis, and theta (radians)
    rotates the ellipse axes relative to the row/column frame.

    Normalisation matches the separable case: integrates to 1 over the plane.
    """
    norm = 0.15915494309189535 / (sigma_x * sigma_y)  # 1 / (2π σ_x σ_y)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    for i in range(size):
        di = x[i] - x0
        for j in range(size):
            dj = x[j] - y0
            ir = di * cos_t + dj * sin_t
            jr = -di * sin_t + dj * cos_t
            array_tofill[i, j] = norm * np.exp(
                -0.5 * ((ir / sigma_x) ** 2 + (jr / sigma_y) ** 2)
            )
    return array_tofill


@jit(nopython=True, nogil=True, cache=True)
def WLS_model_elliptical_nobounds(
    params,
    masks,
    x,
    gauss_2d,
):
    """11-parameter elliptical Gaussian model with Bayer colour masks.

    params layout:
        [0]  x0         row centre
        [1]  y0         column centre
        [2]  sigma_x    half-width along rotated row axis
        [3]  sigma_y    half-width along rotated column axis
        [4]  theta      rotation angle (radians)
        [5]  √bg_B      sqrt(background Blue)
        [6]  √bg_G      sqrt(background Green)
        [7]  √bg_R      sqrt(background Red)
        [8]  √A_B       sqrt(amplitude Blue)
        [9]  √A_G       sqrt(amplitude Green)
        [10] √A_R       sqrt(amplitude Red)
    """
    len_x = len(x)

    amp_lookup = np.array(
        [params[8] * params[8], params[9] * params[9], params[10] * params[10]]
    )
    bg_lookup = np.array(
        [params[5] * params[5], params[6] * params[6], params[7] * params[7]]
    )

    gauss_2d = gaussian_unscaled_model_elliptical(
        gauss_2d, x, len_x,
        params[0], params[1], params[2], params[3], params[4],
    )

    for channel in range(3):
        for i in range(len_x):
            for j in range(len_x):
                if masks[j, i, channel]:
                    gauss_2d[j, i] = (
                        amp_lookup[channel] * gauss_2d[j, i] + bg_lookup[channel]
                    )
    return gauss_2d


@jit(nopython=True, nogil=True, cache=True)
def WLS_chi_elliptical_nobounds(params, data, masks, weights, size, ravelsize):
    """Chi residual vector for the 11-parameter elliptical Gaussian model.

    Args:
        params: 11-element parameter vector (see WLS_model_elliptical_nobounds).
        data: (size × size) observed image.
        masks: (size × size × 3) Bayer colour masks.
        weights: (size × size) per-pixel weights.
        size: Image dimension.
        ravelsize: size × size (pre-computed).

    Returns:
        Flattened sqrt(weights × (data − model)²) — used directly by leastsq.
    """
    x = np.arange(size, dtype=np.float32)
    gauss_2d = np.zeros((size, size), dtype=np.float32)
    chi = np.zeros((size, size), dtype=np.float32)
    gauss_2d[:, :] = WLS_model_elliptical_nobounds(params, masks, x, gauss_2d)
    chi[:, :] = np.multiply(weights, np.square(np.subtract(data, gauss_2d)))
    return np.sqrt(chi.ravel())


@jit(nopython=True, cache=True)
def _initial_theta(smoothed_data, x0, y0, size):
    """Estimate rotation angle from image second moments (principal axis).

    Uses the inertia tensor of the smoothed intensity distribution.
    Returns theta in radians, in the range (-π/2, π/2).
    """
    I_rr = 0.0  # row variance
    I_cc = 0.0  # column variance
    I_rc = 0.0  # cross moment
    A = 0.0
    for i in range(size):
        di = i - x0
        for j in range(size):
            v = smoothed_data[i, j]
            if v > 0.0:
                dj = j - y0
                I_rr += v * di * di
                I_cc += v * dj * dj
                I_rc += v * di * dj
                A += v
    if A > 0.0:
        I_rr /= A
        I_cc /= A
        I_rc /= A
    # Principal axis angle: 0.5 * atan2(2*I_rc, I_rr - I_cc)
    theta = 0.5 * np.arctan2(2.0 * I_rc, I_rr - I_cc)
    return theta


@jit(nopython=True, cache=True)
def initial_guess_elliptical(smoothed_data, raw_data, masks):
    """Initial parameter guess for elliptical Gaussian fitting.

    Extends initial_guess with a moment-based estimate of the rotation angle.

    Returns:
        11-element array:
        [x0, y0, sigma_x, sigma_y, theta,
         √bg_B, √bg_G, √bg_R, √A_B, √A_G, √A_R]
    """
    BG_matrix = np.zeros(masks.shape[-1])
    flattened_rawdata = raw_data.ravel()
    for i in np.arange(masks.shape[-1]):
        pixels = masks[:, :, i].ravel()
        BG_matrix[i] = np.min(np.abs(flattened_rawdata[pixels]))
    bB, bG, bR = BG_matrix
    ig_data = np.abs(smoothed_data)
    bs_data = ig_data - np.abs(np.min(ig_data))
    size = bs_data.shape[0]
    A, x_ig, y_ig = _sum_and_centre_of_mass(bs_data, size)
    sigma_y, sigma_x = _initial_sigma(bs_data, x_ig, y_ig, A, size)
    theta = _initial_theta(bs_data, x_ig, y_ig, size)
    A_ig = A / 3.0
    return (
        x_ig, y_ig, sigma_y, sigma_x, theta,
        np.sqrt(np.abs(bB)), np.sqrt(np.abs(bG)), np.sqrt(np.abs(bR)),
        np.sqrt(np.abs(A_ig)), np.sqrt(np.abs(A_ig)), np.sqrt(np.abs(A_ig)),
    )


@jit(nopython=True, cache=True)
def compute_A_median(smoothed_data):
    """Compute median-subtracted amplitude: sum(smoothed - median(smoothed)).
    Returns ~0 for symmetric noise, >0 for real spots."""
    flat = smoothed_data.ravel().copy()
    flat.sort()
    n = len(flat)
    if n % 2 == 0:
        median_val = (flat[n // 2 - 1] + flat[n // 2]) / 2.0
    else:
        median_val = flat[n // 2]
    A = 0.0
    for i in range(n):
        A += flat[i] - median_val
    return A
