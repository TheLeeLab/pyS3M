"""Coverage tests for pyS3M.clustering.batch's remaining branches not already
exercised by test_sm_extraction_batch.py: the `config=` override path, all
three `clustering_method` branches (DBSCAN/linked/unknown-raises), the
zero-molecules-in-a-FOV and all-FOVs-empty early-returns, the `fov_name is
None` fallback, verbose logging branches, build_photon_accumulation_database's
missing-columns error and its no-error-columns/no-PSF-width-columns fallback
branches, and analyse_multi_fov_dataset's verbose + build_accumulation=False
paths.

Reuses the small synthetic-localisation-data generator and h5 fixture pattern
from test_sm_extraction_batch.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pyS3M.SM_extractionfunctions as SM_extractionfunctions
import pyS3M.IOFunctions as IOFunctions
from pyS3M.clustering import ClusteringConfig


def create_synthetic_localization_data(n_molecules, frames_per_molecule, rng):
    """Small synthetic localisation-data generator (mirrors
    test_sm_extraction_batch.py's helper of the same name -- duplicated
    locally rather than imported, since pytest doesn't put unit_tests/ on
    sys.path as an importable package)."""
    records = []
    for mol_idx in range(n_molecules):
        x_center = rng.uniform(10, 500)
        y_center = rng.uniform(10, 500)
        n_frames = max(3, frames_per_molecule + rng.integers(-3, 4))
        for frame_idx in range(n_frames):
            xc = x_center + rng.normal(0, 0.3)
            yc = y_center + rng.normal(0, 0.3)
            photons = rng.uniform(500, 5000)
            rgb = rng.dirichlet([2, 3, 1])
            A_R, A_G, A_B = rgb[0] * photons, rgb[1] * photons, rgb[2] * photons
            records.append({
                "frame": frame_idx * 5, "xc": xc, "yc": yc,
                "xc_err": 0.3, "yc_err": 0.3,
                "A_R": A_R, "A_G": A_G, "A_B": A_B,
                "A_R_err": 0.05, "A_G_err": 0.05, "A_B_err": 0.05,
                "bg_R": rng.uniform(5, 20), "bg_G": rng.uniform(5, 20), "bg_B": rng.uniform(5, 20),
                "bg_R_err": 0.05, "bg_G_err": 0.05, "bg_B_err": 0.05,
                "photoelectrons": photons,
                "s_x": rng.uniform(1.2, 1.5), "s_y": rng.uniform(1.2, 1.5),
                "s_x_err": 0.05, "s_y_err": 0.05,
                "chi_sqr": rng.gamma(5, 0.2),
            })
    return pd.DataFrame(records)


@pytest.fixture
def fov_files_small(tmp_path):
    IO = IOFunctions.IO_Functions()
    rng = np.random.default_rng(123)
    files = []
    for i, n_molecules in enumerate([6, 8]):
        loc_data = create_synthetic_localization_data(n_molecules, 8, rng)
        filepath = tmp_path / f"Pos{i}_locs.h5"
        IO._write_h5_database(loc_data, str(filepath), normalise_photons=False, append=False)
        files.append(str(filepath))
    return files


class TestExtractSingleMoleculesBatchConfig:
    def test_config_overrides_kwargs(self, fov_files_small):
        SM_E = SM_extractionfunctions.extract_SMs()
        cfg = ClusteringConfig(
            clustering_method="HDBSCAN", min_cluster_size=3, verbose=False
        )
        sm_db, sf_db = SM_E.extract_single_molecules_batch(
            fov_files_small, config=cfg, min_cluster_size=999,  # overridden by config
        )
        assert len(sm_db) > 0

    def test_non_list_localisation_files_raises(self):
        SM_E = SM_extractionfunctions.extract_SMs()
        with pytest.raises(ValueError, match="localisation_files must be"):
            SM_E.extract_single_molecules_batch("not_a_list.h5")


class TestExtractSingleMoleculesBatchMethods:
    def test_dbscan_method(self, fov_files_small):
        SM_E = SM_extractionfunctions.extract_SMs()
        sm_db, sf_db = SM_E.extract_single_molecules_batch(
            fov_files_small, clustering_method="DBSCAN", min_cluster_size=3, verbose=False,
        )
        assert isinstance(sm_db, pd.DataFrame)

    def test_linked_method(self, fov_files_small):
        SM_E = SM_extractionfunctions.extract_SMs()
        sm_db, sf_db = SM_E.extract_single_molecules_batch(
            fov_files_small, clustering_method="linked", max_distance=2.0, max_frames=10, verbose=False,
        )
        assert isinstance(sm_db, pd.DataFrame)

    def test_unknown_method_raises(self, fov_files_small):
        SM_E = SM_extractionfunctions.extract_SMs()
        with pytest.raises(ValueError, match="Unknown clustering_method"):
            SM_E.extract_single_molecules_batch(fov_files_small, clustering_method="bogus")

    def test_verbose_true_happy_path(self, fov_files_small):
        SM_E = SM_extractionfunctions.extract_SMs()
        sm_db, sf_db = SM_E.extract_single_molecules_batch(
            fov_files_small, clustering_method="HDBSCAN", min_cluster_size=3, verbose=True,
        )
        assert len(sm_db) > 0

    def test_start_frame_filters_locs(self, tmp_path):
        IO = IOFunctions.IO_Functions()
        rng = np.random.default_rng(5)
        loc_data = create_synthetic_localization_data(6, 8, rng)
        filepath = tmp_path / "Pos0_locs.h5"
        IO._write_h5_database(loc_data, str(filepath), normalise_photons=False, append=False)

        SM_E = SM_extractionfunctions.extract_SMs()
        sm_db, sf_db = SM_E.extract_single_molecules_batch(
            [str(filepath)], clustering_method="HDBSCAN", min_cluster_size=3,
            start_frame=1000, verbose=False,
        )
        # Every localisation has frame < 1000 (max frame ~ 7*5=35), so nothing survives.
        assert len(sm_db) == 0


class TestExtractSingleMoleculesBatchEmptyBranches:
    def test_fov_name_none_falls_back_to_fov_index(self, fov_files_small, monkeypatch):
        SM_E = SM_extractionfunctions.extract_SMs()
        monkeypatch.setattr(SM_E, "_extract_fov_name", lambda filepath: None)
        sm_db, sf_db = SM_E.extract_single_molecules_batch(
            fov_files_small, clustering_method="HDBSCAN", min_cluster_size=3, verbose=True,
        )
        assert set(sm_db["fov_name"].unique()) == {"fov_0", "fov_1"}

    def test_fov_with_zero_molecules_is_skipped(self, tmp_path):
        IO = IOFunctions.IO_Functions()
        rng = np.random.default_rng(7)
        # Too few localisations per molecule to ever form a min_cluster_size=10
        # cluster (frames_per_molecule=2 -> n_frames clamped to >=3, always
        # under 10); the "real" FOV uses frames_per_molecule=15 (n_frames
        # 12-18 after jitter) to comfortably clear it.
        sparse_loc_data = create_synthetic_localization_data(2, 2, rng)
        real_loc_data = create_synthetic_localization_data(6, 15, rng)
        sparse_path = tmp_path / "Pos0_sparse.h5"
        real_path = tmp_path / "Pos1_real.h5"
        IO._write_h5_database(sparse_loc_data, str(sparse_path), normalise_photons=False, append=False)
        IO._write_h5_database(real_loc_data, str(real_path), normalise_photons=False, append=False)

        SM_E = SM_extractionfunctions.extract_SMs()
        sm_db, sf_db = SM_E.extract_single_molecules_batch(
            [str(sparse_path), str(real_path)],
            clustering_method="HDBSCAN", min_cluster_size=10, verbose=True,
        )
        # Only the FOV with enough locs per molecule should contribute.
        assert set(sm_db["fov_index"].unique()) == {1}

    def test_all_fovs_empty_returns_empty_dataframes(self, tmp_path):
        IO = IOFunctions.IO_Functions()
        rng = np.random.default_rng(9)
        sparse_loc_data = create_synthetic_localization_data(1, 8, rng)
        filepath = tmp_path / "Pos0_sparse.h5"
        IO._write_h5_database(sparse_loc_data, str(filepath), normalise_photons=False, append=False)

        SM_E = SM_extractionfunctions.extract_SMs()
        sm_db, sf_db = SM_E.extract_single_molecules_batch(
            [str(filepath)], clustering_method="HDBSCAN", min_cluster_size=50, verbose=True,
        )
        assert len(sm_db) == 0
        assert len(sf_db) == 0


class TestBuildPhotonAccumulationDatabaseBranches:
    def _sf_db(self, rng, with_errors=True, with_psf=True):
        loc_data = create_synthetic_localization_data(2, 8, rng)
        loc_data["molecular_index"] = 0
        loc_data.loc[loc_data.index[len(loc_data) // 2:], "molecular_index"] = 1
        if not with_errors:
            loc_data = loc_data.drop(columns=["A_R_err", "A_G_err", "A_B_err"])
        if not with_psf:
            loc_data = loc_data.drop(columns=["s_x", "s_y"])
        return loc_data

    def test_missing_columns_raises(self):
        SM_E = SM_extractionfunctions.extract_SMs()
        bad_df = pd.DataFrame({"molecular_index": [0], "frame": [0]})
        with pytest.raises(ValueError, match="Missing required columns"):
            SM_E.build_photon_accumulation_database(bad_df)

    def test_verbose_true(self):
        SM_E = SM_extractionfunctions.extract_SMs()
        rng = np.random.default_rng(11)
        sf_db = self._sf_db(rng)
        pa_db = SM_E.build_photon_accumulation_database(sf_db, verbose=True)
        assert len(pa_db) > 0

    def test_no_amplitude_error_columns_uses_simple_mean(self):
        SM_E = SM_extractionfunctions.extract_SMs()
        rng = np.random.default_rng(13)
        sf_db = self._sf_db(rng, with_errors=False)
        pa_db = SM_E.build_photon_accumulation_database(sf_db, verbose=False)
        # Fallback branch sets A_*_err to exactly zero (no weighting available).
        assert np.all(pa_db["A_R_err"] == 0.0)

    def test_no_psf_width_columns_uses_zero(self):
        SM_E = SM_extractionfunctions.extract_SMs()
        rng = np.random.default_rng(17)
        sf_db = self._sf_db(rng, with_psf=False)
        pa_db = SM_E.build_photon_accumulation_database(sf_db, verbose=False)
        assert np.all(pa_db["s_x_mean"] == 0.0)
        assert np.all(pa_db["s_y_mean"] == 0.0)


class TestAnalyseMultiFovDataset:
    def test_verbose_true_with_output_folder(self, fov_files_small, tmp_path):
        SM_E = SM_extractionfunctions.extract_SMs()
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        sm_db, sf_db, pa_db = SM_E.analyse_multi_fov_dataset(
            fov_files_small, clustering_method="HDBSCAN", min_cluster_size=3,
            build_accumulation=True, output_folder=str(output_dir), verbose=True,
        )
        assert (output_dir / "analysis_single_molecules.h5").exists()

    def test_build_accumulation_false_returns_two_tuple(self, fov_files_small):
        SM_E = SM_extractionfunctions.extract_SMs()
        result = SM_E.analyse_multi_fov_dataset(
            fov_files_small, clustering_method="HDBSCAN", min_cluster_size=3,
            build_accumulation=False, verbose=False,
        )
        assert len(result) == 2
