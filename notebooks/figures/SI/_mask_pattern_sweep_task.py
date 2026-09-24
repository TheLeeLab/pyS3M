#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone worker for one (pattern, dye) simulation task from
``FigureSI_DifferentMaskPatterns.ipynb``'s pattern sweep.

The sweep cell that used to call this in-process, once per (pattern, dye)
combination (6 patterns x 3 dyes = 18 iterations, each n_bootstrap=100_000 x
200 photon levels), now shells out to this script via ``subprocess.run`` --
one completely fresh Python process per task, joined before the next one
starts. Whatever CPython/glibc's allocator does or doesn't hand back to the
OS between iterations no longer matters: RSS is fully reclaimed the moment
each subprocess exits, so peak memory across the whole sweep is bounded by
one task's footprint rather than whatever got retained across all 18.

Run directly for one task::

    python _mask_pattern_sweep_task.py --pattern Bayer --dye "ATTO 647N" \\
        --save-folder /path/to/output

Mosaic/pattern definitions here are a deliberate duplicate of
``FigureSI_DifferentMaskPatterns.ipynb`` cells 6/9 (not an import from the
notebook) so this file stays a plain, independently-runnable script -- the
notebook's own visualisation cells (6-9) are unaffected by this refactor.
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

CHIP_SEED = 42
RANDOM_BAYER_SEED = 123

DEFAULT_CALIB_DIR = str(
    Path(__file__).resolve().parents[3] / "Camera_Calibrations" / "Ximea_Camera"
)


def build_chip_pattern(chip_shape: tuple[int, int], seed: int = CHIP_SEED) -> np.ndarray:
    """Chip-wide, unconstrained-shuffle random Bayer pattern (see notebook cell 6)."""
    chip_h, chip_w = chip_shape
    total = chip_h * chip_w
    n_G = total // 2
    n_R = (total - n_G) // 2
    n_B = total - n_G - n_R
    rng = np.random.default_rng(seed)
    flat = np.empty(total, dtype="U1")
    flat[:n_R] = "R"
    flat[n_R:n_R + n_G] = "G"
    flat[n_R + n_G:] = "B"
    rng.shuffle(flat)
    return flat.reshape(chip_h, chip_w)


def sample_chip_patches_batch(
    pattern: np.ndarray, patch_size: int, n_samples: int, rng: np.random.Generator
) -> np.ndarray:
    """n_samples independent random (patch_size x patch_size) crops of pattern,
    each at its own independently-random offset -- one per bootstrap sample."""
    h, w = pattern.shape
    r0 = rng.integers(0, h - patch_size, size=n_samples)
    c0 = rng.integers(0, w - patch_size, size=n_samples)
    rows = r0[:, None, None] + np.arange(patch_size)[None, :, None]
    cols = c0[:, None, None] + np.arange(patch_size)[None, None, :]
    return pattern[rows, cols]  # (n_samples, patch_size, patch_size)


def build_registry() -> dict[str, tuple[np.ndarray | None, str]]:
    """name -> (mosaic_unit or None, starting_flag). Random_Bayer's mosaic
    slot is None -- its masks are per-frame, built in build_camera_parameters."""
    return {
        "Bayer": (np.array([["R", "G"], ["G", "B"]]), "bayer_rggb_"),
        "GGBR": (np.array([["G", "G"], ["R", "B"]]), "ggbr_"),
        "RGBR": (np.array([["R", "G"], ["B", "R"]]), "rgbr_"),
        "RRGB": (np.array([["R", "R"], ["G", "B"]]), "rrgb_"),
        "X-Trans": (
            np.array([
                ["G", "B", "G", "G", "R", "G"],
                ["R", "G", "R", "B", "G", "B"],
                ["G", "B", "G", "G", "R", "G"],
                ["G", "R", "G", "G", "B", "G"],
                ["B", "G", "B", "R", "G", "R"],
                ["G", "R", "G", "G", "B", "G"],
            ]),
            "xtrans_",
        ),
        "Random_Bayer": (None, "random_bayer_s42_"),
    }


def load_calibration(calib_dir: str):
    io = IOFunctions.IO_Functions()
    gain = io.read_tiff(os.path.join(calib_dir, "gain.tif"))
    offset = io.read_tiff(os.path.join(calib_dir, "offset.tif"))
    variance = io.read_tiff(os.path.join(calib_dir, "variance.tif"))
    readnoise = io.read_tiff(os.path.join(calib_dir, "readnoise.tif"))
    rqe = io.read_tiff(os.path.join(calib_dir, "rqe.tif"))
    return gain, offset, variance, readnoise, rqe


def build_camera_parameters(
    pattern_name: str,
    mosaic: np.ndarray | None,
    calib,
    image_size: int,
    n_bootstrap: int,
    pixel_QYs: np.ndarray,
) -> dict:
    gain, offset, variance, readnoise, rqe = calib
    if pattern_name == "Random_Bayer":
        chip_pattern = build_chip_pattern(gain.shape)
        rng = np.random.default_rng(RANDOM_BAYER_SEED)
        patches = sample_chip_patches_batch(chip_pattern, image_size, n_bootstrap, rng)
        masks = {"B": patches == "B", "G": patches == "G", "R": patches == "R"}
        mosaic_unit = None
    else:
        masks = MaskFunctions.Mask_Functions().get_masks(
            size_x=image_size, size_y=image_size, mosaic_unit=mosaic
        )
        mosaic_unit = mosaic

    return {
        "gain": np.full((image_size, image_size), np.median(gain)),
        "offset": np.full((image_size, image_size), np.median(offset)),
        "variance": np.full((image_size, image_size), np.median(variance)),
        "readnoise": float(np.median(readnoise)),
        "rqe": np.full((image_size, image_size), np.median(rqe)),
        "masks": masks,
        "pixel_QYs": pixel_QYs,
        "pixel_order": ["B", "G", "R"],
        "pixel_order_indices": {"B": 0, "G": 1, "R": 2},
        "mosaic_unit": mosaic_unit,
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
    registry = build_registry()
    if pattern_name not in registry:
        raise ValueError(f"Unknown pattern {pattern_name!r}; choices: {list(registry)}")
    mosaic, flag = registry[pattern_name]

    calib = load_calibration(calib_dir)
    S_F = SpectralFunctions.Spectral_Funcs()
    R_sim, G_sim, B_sim, wavelength_sim = S_F.getpixelefficiency()
    pixel_QYs_sim = np.vstack([B_sim, G_sim, R_sim])

    n_photon_space = np.unique(
        np.around(np.logspace(np.log10(photon_min), np.log10(photon_max), n_photon_levels) / 5) * 5
    )

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
        n_unit_cells=7,
    )

    cam_sim = build_camera_parameters(
        pattern_name, mosaic, calib, image_size, config.n_bootstrap, pixel_QYs_sim
    )

    MSF = MultiC_Sim_Funcs()
    MSF.test_simulation_method(
        dye=dye,
        filters=[],
        wavelength=wavelength_sim,
        camera_parameters=cam_sim,
        save_folder=save_folder,
        n_photon_space=n_photon_space,
        smoothing_function=make_smoothing_function(),
        strategy=FittingStrategy.STANDARD,
        starting_flag=flag,
        config=config,
        overwrite=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", required=True, choices=list(build_registry()))
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
