#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone worker for one (spectral pattern, dye) simulation task from
``Figure1_DifferentPixelSpectra.ipynb``'s pattern sweep.

Same rationale and pattern as ``_mask_pattern_sweep_task.py`` (read its
docstring for the full explanation): the sweep cell shells out to this
script once per (pattern, dye) combination via ``subprocess.run`` instead of
calling in-process, so each task runs in a genuinely fresh Python process
and its RSS is fully reclaimed by the OS on exit rather than accumulating
across the sweep's iterations inside one long-lived kernel.

Run directly for one task::

    python _spectral_pattern_sweep_task.py --pattern CYYM --dye "ATTO 647N" \\
        --save-folder /path/to/output

Spectral pattern definitions here are a deliberate duplicate of
``Figure1_DifferentPixelSpectra.ipynb`` cells 5/10 (not an import from the
notebook), so this file stays a plain, independently-runnable script -- the
notebook's own visualisation cells are unaffected by this refactor.
"""
from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[3] / "src"))

import pyS3M.IOFunctions as IOFunctions
import pyS3M.MaskFunctions as MaskFunctions
import pyS3M.SpectralFunctions as SpectralFunctions
import pyS3M.sCMOSFunctions as sCMOSFunctions
from pyS3M.simulation.multicolour import (
    FittingStrategy,
    SimulationConfig,
    MultiC_Sim_Funcs,
)

DEFAULT_CALIB_DIR = str(
    Path(__file__).resolve().parents[3] / "Camera_Calibrations" / "Ximea_Camera"
)


def _gauss(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def build_registry(wavelength: np.ndarray, peak_G: float) -> dict[str, dict]:
    """name -> dict(mosaic, pixel_QYs, pixel_order, pixel_order_indices, flag).

    Six spectral filter designs, all on a 2x2 spatial mosaic (see notebook
    cell 5 for the physical rationale behind each).
    """
    R_base, G_base, B_base, _ = SpectralFunctions.Spectral_Funcs().getpixelefficiency()

    C_spec = peak_G * _gauss(wavelength, 500, 50) + 0.15 * _gauss(wavelength, 650, 40)
    Y_spec = peak_G * _gauss(wavelength, 575, 80)
    M_spec = peak_G * _gauss(wavelength, 450, 50) + peak_G * _gauss(wavelength, 650, 50)

    E_spec = peak_G * _gauss(wavelength, 520, 15)

    W_spec = np.full_like(wavelength, peak_G, dtype=float)

    lam_min, lam_max = 400.0, 780.0
    phi = (wavelength - lam_min) / (lam_max - lam_min)
    S1_spec = np.clip(0.5 * (1.0 + np.sin(2 * np.pi * phi)) * peak_G, 0.0, None)
    C1_spec = np.clip(0.5 * (1.0 + np.cos(2 * np.pi * phi)) * peak_G, 0.0, None)
    S2_spec = np.clip(0.5 * (1.0 + np.sin(4 * np.pi * phi)) * peak_G, 0.0, None)
    C2_spec = np.clip(0.5 * (1.0 + np.cos(4 * np.pi * phi)) * peak_G, 0.0, None)
    vis_mask = (wavelength >= lam_min) & (wavelength <= lam_max)
    for sp in [S1_spec, C1_spec, S2_spec, C2_spec]:
        sp[~vis_mask] = 0.0

    N_spec = peak_G * _gauss(wavelength, 750, 40)

    return {
        "Bayer": {
            "mosaic": np.array([["R", "G"], ["G", "B"]]),
            "pixel_QYs": np.vstack([B_base, G_base, R_base]),
            "pixel_order": ["B", "G", "R"],
            "pixel_order_indices": {"B": 0, "G": 1, "R": 2},
            "flag": "spec_bayer_",
        },
        "CYYM": {
            "mosaic": np.array([["C", "Y"], ["Y", "M"]]),
            "pixel_QYs": np.vstack([C_spec, Y_spec, M_spec]),
            "pixel_order": ["C", "Y", "M"],
            "pixel_order_indices": {"C": 0, "Y": 1, "M": 2},
            "flag": "spec_cyym_",
        },
        "RGBE": {
            "mosaic": np.array([["R", "G"], ["B", "E"]]),
            "pixel_QYs": np.vstack([B_base, G_base, R_base, E_spec]),
            "pixel_order": ["B", "G", "R", "E"],
            "pixel_order_indices": {"B": 0, "G": 1, "R": 2, "E": 3},
            "flag": "spec_rgbe_",
        },
        "RGBW": {
            "mosaic": np.array([["R", "G"], ["B", "W"]]),
            "pixel_QYs": np.vstack([B_base, G_base, R_base, W_spec]),
            "pixel_order": ["B", "G", "R", "W"],
            "pixel_order_indices": {"B": 0, "G": 1, "R": 2, "W": 3},
            "flag": "spec_rgbw_",
        },
        "SinCos": {
            "mosaic": np.array([["P", "Q"], ["U", "V"]]),
            "pixel_QYs": np.vstack([S1_spec, C1_spec, S2_spec, C2_spec]),
            "pixel_order": ["P", "Q", "U", "V"],
            "pixel_order_indices": {"P": 0, "Q": 1, "U": 2, "V": 3},
            "flag": "spec_sincos_",
        },
        "NIR": {
            "mosaic": np.array([["R", "G"], ["B", "N"]]),
            "pixel_QYs": np.vstack([B_base, G_base, R_base, N_spec]),
            "pixel_order": ["B", "G", "R", "N"],
            "pixel_order_indices": {"B": 0, "G": 1, "R": 2, "N": 3},
            "flag": "spec_nir_",
        },
    }


def load_calibration(calib_dir: str):
    io = IOFunctions.IO_Functions()
    gain = io.read_tiff(os.path.join(calib_dir, "gain.tif"))
    offset = io.read_tiff(os.path.join(calib_dir, "offset.tif"))
    variance = io.read_tiff(os.path.join(calib_dir, "variance.tif"))
    readnoise = io.read_tiff(os.path.join(calib_dir, "readnoise.tif"))
    rqe = io.read_tiff(os.path.join(calib_dir, "rqe.tif"))
    return gain, offset, variance, readnoise, rqe


def build_camera_parameters(pdict: dict, calib, image_size: int) -> dict:
    gain, offset, variance, readnoise, rqe = calib
    masks = MaskFunctions.Mask_Functions().get_masks(
        size_x=image_size, size_y=image_size, mosaic_unit=pdict["mosaic"]
    )
    return {
        "gain": np.full((image_size, image_size), np.median(gain)),
        "offset": np.full((image_size, image_size), np.median(offset)),
        "variance": np.full((image_size, image_size), np.median(variance)),
        "readnoise": float(np.median(readnoise)),
        "rqe": np.full((image_size, image_size), np.median(rqe)),
        "masks": masks,
        "pixel_QYs": pdict["pixel_QYs"],
        "pixel_order": pdict["pixel_order"],
        "pixel_order_indices": pdict["pixel_order_indices"],
        "mosaic_unit": pdict["mosaic"],
    }


def make_smoothing_function(sigma: float = 1.5):
    sf = types.SimpleNamespace()
    sf.args = {"sigma": sigma}
    sf.extent = sigma
    sf.smoothing_function = sCMOSFunctions.sCMOS_Functions().gaussian_filter_stack
    sf.data_arg = "image"
    return sf


def run_one(
    pattern_name: str,
    dye: str,
    calib_dir: str,
    save_folder: str,
    image_size: int = 14,
    pixel_size: int = 69,
    NA: float = 1.49,
    n_bootstrap: int = 100_000,
    n_photon_levels: int = 200,
    photon_min: float = 500.0,
    photon_max: float = 50_000.0,
    background_photons: float = 5.0,
) -> None:
    S_F = SpectralFunctions.Spectral_Funcs()
    _, G_base, _, wavelength = S_F.getpixelefficiency()
    peak_G = float(np.max(G_base))

    registry = build_registry(wavelength, peak_G)
    if pattern_name not in registry:
        raise ValueError(f"Unknown pattern {pattern_name!r}; choices: {list(registry)}")
    pdict = registry[pattern_name]
    n_ch = len(pdict["pixel_order"])

    calib = load_calibration(calib_dir)
    cam_sim = build_camera_parameters(pdict, calib, image_size)

    n_photon_space = np.unique(
        np.around(np.logspace(np.log10(photon_min), np.log10(photon_max), n_photon_levels) / 5) * 5
    )

    # background_colour length must match the number of pixel types
    config = SimulationConfig(
        n_bootstrap=n_bootstrap,
        background_photons=background_photons,
        background_colour=[1.0] * n_ch,
        NA=NA,
        pixel_size=pixel_size,
        save_raw_results=True,
        subtractx0y0=False,
        saverawimages=False,
        verbose=False,
        use_stochastic_photons=True,
        n_unit_cells=7,
    )

    MSF = MultiC_Sim_Funcs()
    MSF.test_simulation_method(
        dye=dye,
        filters=[],
        wavelength=wavelength,
        camera_parameters=cam_sim,
        save_folder=save_folder,
        n_photon_space=n_photon_space,
        smoothing_function=make_smoothing_function(),
        strategy=FittingStrategy.STANDARD,
        starting_flag=pdict["flag"],
        config=config,
        overwrite=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # choices computed lazily inside run_one (needs wavelength/peak_G); keep
    # the CLI permissive here and let run_one raise on an unknown name.
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--dye", required=True)
    parser.add_argument("--calib-dir", default=DEFAULT_CALIB_DIR)
    parser.add_argument("--save-folder", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=100_000)
    parser.add_argument("--n-photon-levels", type=int, default=200)
    args = parser.parse_args()

    run_one(
        args.pattern,
        args.dye,
        args.calib_dir,
        args.save_folder,
        n_bootstrap=args.n_bootstrap,
        n_photon_levels=args.n_photon_levels,
    )


if __name__ == "__main__":
    main()
