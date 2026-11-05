#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test suite for multi-FOV single molecule extraction functionality.

Tests the batch processing methods:
- _extract_fov_name()
- extract_single_molecules_batch()
- build_photon_accumulation_database()
- analyze_multi_fov_dataset()

Created: 2025-10-22
"""

import numpy as np
import pandas as pd
import os

import pytest

import pyS3M.SM_extractionfunctions as SM_extractionfunctions
import pyS3M.IOFunctions as IOFunctions

# Kept deliberately small -- this only needs to exercise the batch-processing
# code paths (FOV bookkeeping, clustering, accumulation), not statistically
# realistic dataset sizes. min_cluster_size=3 below requires >=3 frames per
# molecule; frames_per_molecule=8 with jitter (-3..+3) keeps a safety margin
# above that floor.
N_MOLECULES_PER_FOV = [6, 8, 10]
FRAMES_PER_MOLECULE = 8


def create_synthetic_localization_data(n_molecules, frames_per_molecule, rng):
    """
    Create synthetic localization data for testing.

    Args:
        n_molecules (int): Number of molecules to simulate
        frames_per_molecule (int): Frames per molecule (with some variation)
        rng (np.random.Generator): Random generator for reproducibility

    Returns:
        pd.DataFrame: Synthetic localization data with realistic columns
    """
    records = []

    for mol_idx in range(n_molecules):
        # Random position for this molecule (same across frames with small jitter)
        x_center = rng.uniform(10, 500)
        y_center = rng.uniform(10, 500)

        # Random number of frames (around frames_per_molecule)
        n_frames = max(3, frames_per_molecule + rng.integers(-3, 4))

        for frame_idx in range(n_frames):
            # Position jitter (localization precision ~20 nm = 0.3 pixels at 69 nm/pixel)
            xc = x_center + rng.normal(0, 0.3)
            yc = y_center + rng.normal(0, 0.3)

            # Random photon counts (realistic range)
            photons = rng.uniform(500, 5000)

            # Random RGB fractions (normalized)
            rgb = rng.dirichlet([2, 3, 1])  # Bias toward G
            A_R = rgb[0] * photons
            A_G = rgb[1] * photons
            A_B = rgb[2] * photons

            # Realistic errors (scales as 1/sqrt(photons), well under
            # FilteringConstants.MAX_COLOUR_ERROR=0.15 fractional threshold)
            A_R_err = 0.05
            A_G_err = 0.05
            A_B_err = 0.05

            # Small, non-zero background per channel (filter_quality_localisations
            # requires bg_R/G/B_err columns, fractional error under the same 0.15
            # threshold as the amplitude errors above)
            bg_R = rng.uniform(5, 20)
            bg_G = rng.uniform(5, 20)
            bg_B = rng.uniform(5, 20)
            bg_R_err = 0.05
            bg_G_err = 0.05
            bg_B_err = 0.05

            # PSF widths (within FilteringConstants' MIN/MAX_SIGMA_NM bounds at
            # the default 69 nm/px pixel size: 75-250 nm -> ~1.09-3.62 px)
            s_x = rng.uniform(1.2, 1.5)
            s_y = rng.uniform(1.2, 1.5)
            # PSF width errors, well under MAX_SIGMA_ERROR_NM=40nm -> ~0.58px
            s_x_err = 0.05
            s_y_err = 0.05

            # Chi-squared (good fits)
            chi_sqr = rng.gamma(5, 0.2)  # Mean ~1.0

            record = {
                "frame": frame_idx * 5,  # Non-consecutive frames
                "xc": xc,
                "yc": yc,
                "xc_err": 0.3,
                "yc_err": 0.3,
                "A_R": A_R,
                "A_G": A_G,
                "A_B": A_B,
                "A_R_err": A_R_err,
                "A_G_err": A_G_err,
                "A_B_err": A_B_err,
                "bg_R": bg_R,
                "bg_G": bg_G,
                "bg_B": bg_B,
                "bg_R_err": bg_R_err,
                "bg_G_err": bg_G_err,
                "bg_B_err": bg_B_err,
                "photons": photons,
                "s_x": s_x,
                "s_y": s_y,
                "s_x_err": s_x_err,
                "s_y_err": s_y_err,
                "chi_sqr": chi_sqr,
            }
            records.append(record)

    return pd.DataFrame(records)


@pytest.fixture(scope="module")
def fov_files(tmp_path_factory):
    """Write a small set of synthetic multi-FOV localisation .h5 files."""
    IO = IOFunctions.IO_Functions()
    rng = np.random.default_rng(42)

    fov_dir = tmp_path_factory.mktemp("sm_extraction_fovs")
    files = []
    for i, n_molecules in enumerate(N_MOLECULES_PER_FOV):
        fov_name = f"Pos{i}"
        loc_data = create_synthetic_localization_data(
            n_molecules=n_molecules,
            frames_per_molecule=FRAMES_PER_MOLECULE,
            rng=rng,
        )
        filepath = fov_dir / f"{fov_name}_undrifted_locs.h5"
        IO._write_h5_database(loc_data, str(filepath), normalise_photons=False, append=False)
        files.append(str(filepath))

    return files


@pytest.fixture(scope="module")
def batch_result(fov_files):
    """Run extract_single_molecules_batch once and share across tests."""
    SM_E = SM_extractionfunctions.extract_SMs()
    sm_db, sf_db = SM_E.extract_single_molecules_batch(
        fov_files,
        clustering_method="HDBSCAN",
        min_cluster_size=3,
        verbose=False,
    )
    return sm_db, sf_db


def test_extract_fov_name():
    """Test FOV name extraction from filenames.

    _extract_fov_name returns the full filename with known suffixes/extension
    stripped (not a short "PosN" pattern) -- this ensures uniqueness across
    datasets where a short suffix like 'Pos0' may repeat (see its docstring
    in clustering/batch.py).
    """
    SM_E = SM_extractionfunctions.extract_SMs()

    test_cases = [
        ("/path/to/Pos0_undrifted_locs.h5", "Pos0_undrifted_locs"),
        ("/path/to/Pos15_data.h5", "Pos15_data"),
        ("/path/to/Pos123_test.h5", "Pos123_test"),
        ("/path/to/nopattern.h5", "nopattern"),
        ("/path/to/position_5.h5", "position_5"),
        ("Pos7_file.h5", "Pos7_file"),
    ]

    for filepath, expected in test_cases:
        result = SM_E._extract_fov_name(filepath)
        assert result == expected, f"{filepath} -> {result} (expected {expected})"


def test_batch_processing(fov_files, batch_result):
    """Test multi-FOV batch processing."""
    sm_db, sf_db = batch_result
    n_fovs = len(fov_files)

    # Check FOV columns exist
    required_cols = ["fov_index", "fov_name", "molecular_index"]
    for col in required_cols:
        assert col in sm_db.columns and col in sf_db.columns, \
            f"Column '{col}' missing from sm_db/sf_db"

    # Check FOV names (_extract_fov_name returns the full filename stem, e.g.
    # "Pos0_undrifted_locs", not a short "Pos0" pattern -- see test_extract_fov_name)
    unique_fov_names = sorted(sm_db["fov_name"].unique())
    expected_fov_names = [f"Pos{i}_undrifted_locs" for i in range(n_fovs)]
    assert unique_fov_names == expected_fov_names, \
        f"FOV names incorrect: {unique_fov_names} (expected {expected_fov_names})"

    # Check molecular_index uniqueness
    unique_mol_ids = sm_db["molecular_index"].unique()
    assert len(unique_mol_ids) == len(sm_db), "Duplicate molecular_index values"

    # Check frame database is larger than the per-molecule summary database
    assert len(sf_db) > len(sm_db), "Frame database should be larger than molecule database"


def test_photon_accumulation(batch_result):
    """Test photon accumulation database building."""
    _, sf_db = batch_result

    SM_E = SM_extractionfunctions.extract_SMs()
    pa_db = SM_E.build_photon_accumulation_database(sf_db, verbose=False)

    # Check required columns
    required_cols = [
        "molecular_index", "frames_accumulated", "photons_accumulated",
        "A_R", "A_G", "A_B", "A_R_err", "A_G_err", "A_B_err",
        "xc_mean", "yc_mean", "xc_std", "yc_std"
    ]
    for col in required_cols:
        assert col in pa_db.columns, f"Column '{col}' missing from pa_db"

    # A_R/A_G/A_B here are running inverse-variance-weighted mean *amplitudes*
    # (same units as per-frame `photons`), not normalised fractions -- see
    # build_photon_accumulation_database's docstring/weighted-mean computation.
    # There's no sum-to-1 invariant to check; verify they're finite and positive.
    for col in ("A_R", "A_G", "A_B"):
        assert np.all(np.isfinite(pa_db[col])), f"Non-finite values in '{col}'"
        assert np.all(pa_db[col] > 0), f"Non-positive values in '{col}'"

    # Check photons_accumulated is monotonically increasing per molecule
    unique_mols = pa_db["molecular_index"].unique()
    for mol_id in unique_mols:
        mol_data = pa_db[pa_db["molecular_index"] == mol_id]
        photons = mol_data["photons_accumulated"].values
        assert np.all(np.diff(photons) > 0), \
            f"Photons not monotonically increasing for molecule {mol_id}"

    # Check frames_accumulated is sequential (1, 2, 3, ...) per molecule
    for mol_id in unique_mols:
        mol_data = pa_db[pa_db["molecular_index"] == mol_id]
        frames = mol_data["frames_accumulated"].values
        expected_frames = np.arange(1, len(mol_data) + 1)
        assert np.array_equal(frames, expected_frames), \
            f"frames_accumulated not sequential for molecule {mol_id}"


def test_full_workflow(fov_files, tmp_path):
    """Test complete analyze_multi_fov_dataset workflow with saving."""
    SM_E = SM_extractionfunctions.extract_SMs()

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    sm_db, sf_db, pa_db = SM_E.analyse_multi_fov_dataset(
        fov_files,
        clustering_method="HDBSCAN",
        build_accumulation=True,
        min_cluster_size=3,
        output_folder=str(output_dir),
        output_prefix="test_analysis",
        verbose=False,
    )

    expected_files = [
        "test_analysis_single_molecules.h5",
        "test_analysis_single_frames.h5",
        "test_analysis_photon_accumulation.h5",
    ]

    for filename in expected_files:
        filepath = output_dir / filename
        assert filepath.exists(), f"{filename} missing from output"

    # Test loading saved files round-trips to the same size
    sm_loaded = pd.read_hdf(output_dir / "test_analysis_single_molecules.h5")
    sf_loaded = pd.read_hdf(output_dir / "test_analysis_single_frames.h5")
    pa_loaded = pd.read_hdf(output_dir / "test_analysis_photon_accumulation.h5")

    assert len(sm_loaded) == len(sm_db), "Single molecule database size mismatch after reload"
    assert len(sf_loaded) == len(sf_db), "Single frame database size mismatch after reload"
    assert len(pa_loaded) == len(pa_db), "Photon accumulation database size mismatch after reload"


def test_max_background_error_separates_background_cut():
    """max_colour_error cuts only A_*_err; bg_*_err is cut only by max_background_error."""
    import numpy as np
    import pandas as pd
    from pyS3M.SM_extractionfunctions import extract_SMs
    from pyS3M.Constants import FilteringCriteria

    n = 4
    df = pd.DataFrame({
        "xc": np.arange(n, dtype=float), "yc": np.zeros(n), "frame": np.arange(n),
        "chi_sqr": np.ones(n), "xc_err": np.full(n, 0.1), "yc_err": np.full(n, 0.1),
        "s_x": np.full(n, 1.5), "s_y": np.full(n, 1.5),
        "s_x_err": np.full(n, 0.1), "s_y_err": np.full(n, 0.1),
        "A_R": np.full(n, 0.5), "A_G": np.full(n, 0.3), "A_B": np.full(n, 0.2),
        "A_R_err": np.full(n, 0.05), "A_G_err": np.full(n, 0.05), "A_B_err": np.full(n, 0.05),
        "bg_R": np.full(n, 0.5), "bg_G": np.full(n, 0.5), "bg_B": np.zeros(n),
        "bg_R_err": [0.1, 5.0, 20.0, 0.1], "bg_G_err": np.full(n, 0.1),
        "bg_B_err": np.full(n, 0.1), "photons": np.full(n, 1000.0),
    })
    sm = extract_SMs()
    kw = dict(chi_val=3, max_colour_error=1.0, min_photons=0)
    assert len(sm.filter_quality_localisations(df, **kw)) == 4
    assert len(sm.filter_quality_localisations(df, max_background_error=10, **kw)) == 3
    assert len(sm.filter_quality_localisations(df, max_background_error=1, **kw)) == 2
    assert len(sm.filter_quality_localisations(df, max_background_error=np.inf, **kw)) == 4
    crit = FilteringCriteria(chi_val=3, max_colour_error=1.0, min_photons=0,
                             max_background_error=1.0)
    assert len(sm.filter_quality_localisations(df, criteria=crit)) == 2
    crit = FilteringCriteria(chi_val=3, max_colour_error=1.0, min_photons=0)
    assert len(sm.filter_quality_localisations(df, criteria=crit)) == 4
    # A colour error above max_colour_error is still cut.
    df.loc[0, "A_R_err"] = 2.0
    assert len(sm.filter_quality_localisations(df, **kw)) == 3
