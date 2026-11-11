#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test suite for GMM-based dye misidentification analysis.

Tests:
- Ground truth assignment via GMM
- Photon-dependent misidentification analysis
- Integration with photon accumulation database

Created: 2025-10-22
"""

import numpy as np
import pandas as pd

import pytest

import pyS3M.SM_extractionfunctions as SM_extractionfunctions

# Kept deliberately small -- this only needs two well-separated dye
# populations for the GMM to distinguish, not statistically realistic
# dataset scale.
N_MOLECULES_PER_POPULATION = 15
MAX_FRAMES_PER_MOLECULE = 25


def create_synthetic_2dye_accumulation_data():
    """
    Create synthetic photon accumulation database with 2 dye populations.

    Population 0 (Red): High A_R (~0.7), Low A_G (~0.2)
    Population 1 (Green): Low A_R (~0.2), High A_G (~0.7)

    Returns:
        pd.DataFrame: Photon accumulation database
    """
    np.random.seed(42)

    records = []
    mol_idx = 0

    for true_A_R, true_A_G, true_A_B in [(0.70, 0.20, 0.10), (0.20, 0.70, 0.10)]:
        for _ in range(N_MOLECULES_PER_POPULATION):
            n_frames = np.random.randint(8, MAX_FRAMES_PER_MOLECULE)

            for frame_num in range(n_frames):
                photons_this_frame = np.random.uniform(100, 1000)
                photons_accumulated = photons_this_frame * (frame_num + 1)

                # Add noise that decreases with more photons
                noise_scale = 0.1 / np.sqrt(photons_accumulated / 1000)
                A_R = true_A_R + np.random.normal(0, noise_scale)
                A_G = true_A_G + np.random.normal(0, noise_scale)
                A_B = true_A_B + np.random.normal(0, noise_scale)

                # Normalize
                total = A_R + A_G + A_B
                A_R /= total
                A_G /= total
                A_B /= total

                # Errors decrease with photons
                A_R_err = 0.05 / np.sqrt(photons_accumulated / 1000)
                A_G_err = 0.05 / np.sqrt(photons_accumulated / 1000)
                A_B_err = 0.05 / np.sqrt(photons_accumulated / 1000)

                records.append({
                    "molecular_index": mol_idx,
                    "frames_accumulated": frame_num + 1,
                    "photoelectrons_accumulated": photons_accumulated,
                    "A_R": A_R,
                    "A_G": A_G,
                    "A_B": A_B,
                    "A_R_err": A_R_err,
                    "A_G_err": A_G_err,
                    "A_B_err": A_B_err,
                    "xc_mean": np.random.uniform(100, 400),
                    "yc_mean": np.random.uniform(100, 400),
                    "xc_std": 0.3,
                    "yc_std": 0.3,
                    "s_x_mean": 1.3,
                    "s_y_mean": 1.3,
                    "fov_index": 0,
                    "fov_name": "Pos0",
                })

            mol_idx += 1

    return pd.DataFrame(records)


def create_synthetic_2dye_singlemolecule_data():
    """
    Create synthetic single molecule database with 2 dye populations.

    Population 0 (Red): High A_R (~0.7), Low A_G (~0.2)
    Population 1 (Green): Low A_R (~0.2), High A_G (~0.7)

    Returns:
        pd.DataFrame: Single molecule database
    """
    np.random.seed(42)

    records = []
    mol_idx = 0

    for true_A_R, true_A_G, true_A_B in [(0.70, 0.20, 0.10), (0.20, 0.70, 0.10)]:
        for _ in range(N_MOLECULES_PER_POPULATION):
            photons = np.random.uniform(1000, 100000)

            noise_scale = 0.1 / np.sqrt(photons / 1000)
            A_R = true_A_R + np.random.normal(0, noise_scale)
            A_G = true_A_G + np.random.normal(0, noise_scale)
            A_B = true_A_B + np.random.normal(0, noise_scale)

            total = A_R + A_G + A_B
            A_R /= total
            A_G /= total
            A_B /= total

            records.append({
                "molecular_index": mol_idx,
                "photoelectrons": photons,
                "A_R": A_R,
                "A_G": A_G,
                "A_B": A_B,
                "xc": np.random.uniform(100, 400),
                "yc": np.random.uniform(100, 400),
                "s_x": 1.3,
                "s_y": 1.3,
                "fov_index": 0,
                "fov_name": "Pos0",
            })
            mol_idx += 1

    return pd.DataFrame(records)


@pytest.fixture(scope="module")
def pa_db():
    return create_synthetic_2dye_accumulation_data()


@pytest.fixture(scope="module")
def sm_db():
    return create_synthetic_2dye_singlemolecule_data()


@pytest.fixture(scope="module")
def mode_a_result(pa_db):
    """Reference means/assignments from the photon accumulation DB (Mode A)."""
    SM_E = SM_extractionfunctions.extract_SMs()
    fixed_means, ref_db, gmm = SM_E.extract_reference_means(
        pa_db, reference_photoelectron_threshold=5000, verbose=False
    )
    return fixed_means, ref_db, gmm


def test_extract_reference_means_mode_a(pa_db, mode_a_result):
    """Test extracting reference means from photon accumulation DB (Mode A)."""
    fixed_means, ref_db, gmm = mode_a_result

    assert gmm.converged_, "GMM did not converge"
    assert fixed_means.shape == (2, 2), f"Fixed means shape is {fixed_means.shape} (expected (2, 2))"

    n_molecules = pa_db["molecular_index"].nunique()
    assert len(ref_db) >= n_molecules * 0.9, \
        f"Reference database has {len(ref_db)} molecules (expected ~{n_molecules})"

    required_cols = ["molecular_index", "true_label", "max_photoelectrons", "A_R_ref", "A_G_ref",
                      "posterior_prob_0", "posterior_prob_1"]
    for col in required_cols:
        assert col in ref_db.columns, f"Column '{col}' missing from ref_db"

    assert sorted(ref_db["true_label"].unique()) == [0, 1], \
        f"Labels are {sorted(ref_db['true_label'].unique())} (expected [0, 1])"

    posterior_sums = ref_db["posterior_prob_0"] + ref_db["posterior_prob_1"]
    assert np.allclose(posterior_sums, 1.0), "Posterior probabilities don't sum to 1.0"

    # Components should have well-separated A_R means (true populations differ by 0.5)
    mean_diff = np.abs(fixed_means[0, 0] - fixed_means[1, 0])
    assert mean_diff > 0.1, f"Components not well separated (A_R difference: {mean_diff:.3f})"

    assert np.all(fixed_means >= 0) and np.all(fixed_means <= 1), \
        "Fixed means outside valid range [0, 1]"


def test_extract_reference_means_mode_b(sm_db):
    """Test extracting reference means from single molecule DB (Mode B)."""
    SM_E = SM_extractionfunctions.extract_SMs()

    # Mode B.1: no photon threshold (all molecules)
    fixed_means_all, ref_db_all, gmm_all = SM_E.extract_reference_means(
        sm_db, reference_photoelectron_threshold=None, verbose=False
    )

    assert gmm_all.converged_, "GMM did not converge (Mode B.1, all molecules)"
    assert fixed_means_all.shape == (2, 2), \
        f"Fixed means shape is {fixed_means_all.shape} (expected (2, 2))"
    assert len(ref_db_all) == len(sm_db), \
        f"Expected {len(sm_db)} molecules, got {len(ref_db_all)}"

    required_cols_all = ["molecular_index", "true_label", "photoelectrons", "A_R_ref", "A_G_ref",
                         "posterior_prob_0", "posterior_prob_1"]
    for col in required_cols_all:
        assert col in ref_db_all.columns, f"Column '{col}' missing from ref_db_all"

    assert np.all(fixed_means_all >= 0) and np.all(fixed_means_all <= 1), \
        "Fixed means outside valid range [0, 1] (Mode B.1)"

    # Mode B.2: with photon threshold (filter low-photon molecules)
    fixed_means_filtered, ref_db_filtered, gmm_filtered = SM_E.extract_reference_means(
        sm_db, reference_photoelectron_threshold=50000, verbose=False
    )

    assert gmm_filtered.converged_, "GMM did not converge (Mode B.2, filtered)"

    n_expected_filtered = len(sm_db[sm_db["photoelectrons"] >= 50000])
    assert len(ref_db_filtered) == n_expected_filtered, \
        f"Expected {n_expected_filtered} filtered, got {len(ref_db_filtered)}"

    # High-photon-only means should be similar to all-molecule means (less noisy subset
    # of the same populations, not a different distribution)
    mean_diff = np.max(np.abs(fixed_means_all - fixed_means_filtered))
    assert mean_diff < 0.1, \
        f"Means differ too much between modes (max diff: {mean_diff:.4f})"


def test_analytical_misidentification_analysis(pa_db, mode_a_result):
    """Test analytical photon-dependent misidentification analysis."""
    fixed_means, ref_db, _ = mode_a_result

    SM_E = SM_extractionfunctions.extract_SMs()

    # Fewer, narrower bins than production use (this only needs to exercise the
    # per-bin M-estimator fit + analytical overlap calculation, not a fine-grained
    # production sweep), and fewer Monte Carlo samples (numerical precision of the
    # analytical overlap integral, not a correctness-affecting parameter). Upper
    # edge kept below this fixture's max accumulated-photon value so every bin has
    # enough molecules for a well-conditioned per-component covariance fit (a
    # too-sparse top bin degenerates to a singular covariance -> log(0) warning
    # from the mixture pdf, harmless to the result but noisy).
    photon_bins = np.logspace(3, 4.2, 6)
    summary_db = SM_E.analyze_photon_dependent_misidentification_analytical(
        pa_db,
        fixed_means,
        ref_db,
        photon_bins,
        use_earliest_entry=True,
        n_mc_samples=2000,
        verbose=False,
    )

    assert len(summary_db) > 0, "Summary database empty"

    required_summary_cols = ["photoelectron_bin_min", "photoelectron_bin_max", "n_molecules",
                            "converged", "overall_accuracy", "overall_error_rate",
                            "component_0_accuracy", "component_1_accuracy",
                            "cov_0_AR_AR", "cov_0_AR_AG", "cov_0_AG_AG",
                            "cov_1_AR_AR", "cov_1_AR_AG", "cov_1_AG_AG",
                            "weight_0", "weight_1"]
    missing_cols = [col for col in required_summary_cols if col not in summary_db.columns]
    assert not missing_cols, f"Missing columns: {missing_cols}"

    assert (summary_db["overall_accuracy"] >= 0).all() and (summary_db["overall_accuracy"] <= 1).all(), \
        "Some accuracy values outside [0, 1]"

    assert np.allclose(
        summary_db["overall_accuracy"] + summary_db["overall_error_rate"], 1.0
    ), "Accuracy + error rate != 1.0"

    weight_sums = summary_db["weight_0"] + summary_db["weight_1"]
    assert np.allclose(weight_sums, 1.0), "Component weights don't sum to 1.0"

    # Covariance diagonal elements (variances) must be positive
    assert (summary_db["cov_0_AR_AR"] > 0).all() and (summary_db["cov_0_AG_AG"] > 0).all(), \
        "Some component-0 covariance diagonal elements are non-positive"
    assert (summary_db["cov_1_AR_AR"] > 0).all() and (summary_db["cov_1_AG_AG"] > 0).all(), \
        "Some component-1 covariance diagonal elements are non-positive"

    assert summary_db["converged"].all(), \
        f"Not all bins converged ({summary_db['converged'].sum()}/{len(summary_db)})"
