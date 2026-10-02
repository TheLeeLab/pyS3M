#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone worker for one (noise model, dye) simulation task from
``Bayesian_Comparison.ipynb``.

Bayer (RGGB) only. Two camera-noise models are compared:

- ``median``: every pixel of every bootstrap frame gets the sensor-wide median
  gain/offset/variance/rqe (what every other SI sweep does).
- ``chip``: every bootstrap frame gets its own randomly-located crop of the real
  full-chip Ximea calibration maps (``full_chip_calibration``), so hot/defective
  pixels appear at their true frequency. The same crop is used to generate the
  frame and to convert/weight it for fitting.

As with ``_mask_pattern_sweep_task.py``, the notebook runs each task in its own
subprocess so RSS is fully reclaimed between tasks.

Run directly for one task::

    python _bayer_noise_sweep_task.py --noise-model chip --dye "ATTO 647N" \\
        --save-folder /path/to/output
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

CHIP_SAMPLING_SEED = 2026
MOSAIC_BAYER = np.array([["R", "G"], ["G", "B"]])
NOISE_MODELS = {"median": "bayer_median_noise_", "chip": "bayer_chip_noise_"}

DEFAULT_CALIB_DIR = str(
    Path(__file__).resolve().parents[3] / "Camera_Calibrations" / "Ximea_Camera"
)


def photon_space(n_levels: int, photon_min: float = 200.0, photon_max: float = 50_000.0) -> np.ndarray:
    """Log-spaced photon levels, rounded to the nearest 5 (shared with the notebook)."""
    return np.unique(
        np.around(np.logspace(np.log10(photon_min), np.log10(photon_max), n_levels) / 5) * 5
    )


def load_calibration(calib_dir: str) -> dict[str, np.ndarray]:
    io = IOFunctions.IO_Functions()
    return {
        k: io.read_tiff(os.path.join(calib_dir, f"{k}.tif"))
        for k in ("gain", "offset", "variance", "readnoise", "rqe")
    }


def build_camera_parameters(
    noise_model: str, calib: dict[str, np.ndarray], image_size: int, pixel_QYs: np.ndarray
) -> dict:
    flat = lambda k: np.full((image_size, image_size), np.median(calib[k]))  # noqa: E731
    cam = {
        "gain": flat("gain"),
        "offset": flat("offset"),
        "variance": flat("variance"),
        "readnoise": float(np.median(calib["readnoise"])),
        "rqe": flat("rqe"),
        "masks": MaskFunctions.Mask_Functions().get_masks(
            size_x=image_size, size_y=image_size, mosaic_unit=MOSAIC_BAYER
        ),
        "pixel_QYs": pixel_QYs,
        "pixel_order": ["B", "G", "R"],
        "pixel_order_indices": {"B": 0, "G": 1, "R": 2},
        "mosaic_unit": MOSAIC_BAYER,
    }
    if noise_model == "chip":
        cam["full_chip_calibration"] = {
            k: calib[k] for k in ("gain", "offset", "variance", "rqe")
        }
    return cam


def make_smoothing_function(sigma: float = 1.5):
    sf = types.SimpleNamespace()
    sf.args = {"sigma": sigma}
    sf.extent = sigma
    sf.smoothing_function = sCMOSFunctions.sCMOS_Functions().gaussian_filter_stack
    sf.data_arg = "image"
    return sf


def run_one(
    noise_model: str,
    dye: str,
    calib_dir: str,
    save_folder: str,
    pixel_size: int = 69,
    NA: float = 1.49,
    n_bootstrap: int = 100_000,
    n_photon_levels: int = 200,
    background_photons: float = 5.0,
    n_unit_cells: int = 7,
) -> None:
    if noise_model not in NOISE_MODELS:
        raise ValueError(f"Unknown noise model {noise_model!r}; choices: {list(NOISE_MODELS)}")

    calib = load_calibration(calib_dir)
    R_sim, G_sim, B_sim, wavelength_sim = SpectralFunctions.Spectral_Funcs().getpixelefficiency()
    pixel_QYs_sim = np.vstack([B_sim, G_sim, R_sim])

    config = SimulationConfig(
        n_bootstrap=n_bootstrap,
        background_photons=background_photons,
        NA=NA,
        pixel_size=pixel_size,
        save_raw_results=True,
        subtractx0y0=False,
        saverawimages=False,
        verbose=False,
        use_stochastic_photons=True,
        n_unit_cells=n_unit_cells,
        chip_sampling_seed=CHIP_SAMPLING_SEED,
    )

    image_size = n_unit_cells * MOSAIC_BAYER.shape[0]
    cam_sim = build_camera_parameters(noise_model, calib, image_size, pixel_QYs_sim)

    MultiC_Sim_Funcs().test_simulation_method(
        dye=dye,
        filters=[],
        wavelength=wavelength_sim,
        camera_parameters=cam_sim,
        save_folder=save_folder,
        n_photon_space=photon_space(n_photon_levels),
        smoothing_function=make_smoothing_function(),
        strategy=FittingStrategy.STANDARD,
        starting_flag=NOISE_MODELS[noise_model],
        config=config,
        overwrite=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise-model", required=True, choices=list(NOISE_MODELS))
    parser.add_argument("--dye", required=True)
    parser.add_argument("--calib-dir", default=DEFAULT_CALIB_DIR)
    parser.add_argument("--save-folder", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=100_000)
    parser.add_argument("--n-photon-levels", type=int, default=200)
    args = parser.parse_args()

    run_one(
        args.noise_model,
        args.dye,
        args.calib_dir,
        args.save_folder,
        n_bootstrap=args.n_bootstrap,
        n_photon_levels=args.n_photon_levels,
    )


if __name__ == "__main__":
    main()
