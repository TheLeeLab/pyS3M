"""Coverage tests for pyS3M.SM_extractionfunctions's remaining branches not
already exercised by test_sm_extraction_batch.py / test_clustering_batch.py /
test_clustering_linked.py / test_spectral_lap_linking.py: `_load_localisation_files`'s
file-path-list-loading and start_frame-filter branches, `filter_quality_localisations`'s
`criteria=` override path and its "photons column missing" fallback, and
`average_parameters`'s "index" column skip (both loops) and its
photons-column-absent manual-computation fallback.

Small synthetic localisation DataFrames throughout (mirrors the column set used
in test_sm_extraction_batch.py's generator).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pyS3M.SM_extractionfunctions as SM_extractionfunctions
import pyS3M.IOFunctions as IOFunctions
from pyS3M.Constants import FilteringCriteria


def _make_loc_df(n=20, seed=0, with_photons=True):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "frame": np.arange(n),
        "xc": rng.uniform(0, 100, n),
        "yc": rng.uniform(0, 100, n),
        "xc_err": 0.05,
        "yc_err": 0.05,
        "A_R": rng.uniform(200, 400, n),
        "A_G": rng.uniform(200, 400, n),
        "A_B": rng.uniform(200, 400, n),
        "A_R_err": 0.05,
        "A_G_err": 0.05,
        "A_B_err": 0.05,
        "bg_R": rng.uniform(5, 20, n),
        "bg_G": rng.uniform(5, 20, n),
        "bg_B": rng.uniform(5, 20, n),
        "bg_R_err": 0.05,
        "bg_G_err": 0.05,
        "bg_B_err": 0.05,
        "s_x": rng.uniform(1.2, 1.5, n),
        "s_y": rng.uniform(1.2, 1.5, n),
        "s_x_err": 0.05,
        "s_y_err": 0.05,
        "chi_sqr": rng.gamma(5, 0.2, n),
    })
    if with_photons:
        df["photons"] = df["A_R"] + df["A_G"] + df["A_B"]
    return df


class TestLoadLocalisationFiles:
    def test_loads_and_concatenates_file_list(self, tmp_path):
        IO = IOFunctions.IO_Functions()
        SM_E = SM_extractionfunctions.extract_SMs()
        df1 = _make_loc_df(5, seed=1)
        df2 = _make_loc_df(5, seed=2)
        p1 = tmp_path / "a.h5"
        p2 = tmp_path / "b.h5"
        IO._write_h5_database(df1, str(p1), normalise_photons=False, append=False)
        IO._write_h5_database(df2, str(p2), normalise_photons=False, append=False)

        loaded = SM_E._load_localisation_files([str(p1), str(p2)])
        assert len(loaded) == 10

    def test_empty_file_list_returns_empty_dataframe(self):
        SM_E = SM_extractionfunctions.extract_SMs()
        loaded = SM_E._load_localisation_files([])
        assert isinstance(loaded, pd.DataFrame)
        assert len(loaded) == 0

    def test_start_frame_filters_out_early_frames(self):
        SM_E = SM_extractionfunctions.extract_SMs()
        df = _make_loc_df(20)
        loaded = SM_E._load_localisation_files(df, start_frame=10)
        assert loaded["frame"].min() >= 10
        assert len(loaded) == 10


class TestFilterQualityLocalisationsCriteriaAndFallback:
    def test_criteria_object_overrides_kwargs(self):
        SM_E = SM_extractionfunctions.extract_SMs()
        df = _make_loc_df(30)
        criteria = FilteringCriteria(
            chi_val=10.0, max_localisation_error=1.0, max_colour_error=1.0,
            min_sigma=0.01, max_sigma=10.0, max_sigma_error=10.0,
            min_photons=1.0, max_photons=1e9,
        )
        filtered = SM_E.filter_quality_localisations(
            df, criteria=criteria,
            # deliberately-restrictive kwargs that would filter everything if used
            chi_val=0.0001, min_photons=1e12,
        )
        assert len(filtered) > 0

    def test_missing_photons_column_computed_via_add_photon_columns(self):
        SM_E = SM_extractionfunctions.extract_SMs()
        df = _make_loc_df(30, with_photons=False)
        assert "photons" not in df.columns
        filtered = SM_E.filter_quality_localisations(
            df, chi_val=10.0, max_localisation_error=1.0, max_colour_error=1.0,
            min_sigma=0.01, max_sigma=10.0, max_sigma_error=10.0,
            min_photons=1.0, max_photons=1e9,
        )
        assert "photons" in filtered.columns
        assert len(filtered) > 0


    def test_photoelectron_bounds_cut_on_detected_photoelectrons(self):
        SM_E = SM_extractionfunctions.extract_SMs()
        df = IOFunctions.IO_Functions()._add_photon_columns(_make_loc_df(40, with_photons=False))
        df = df.drop(columns=["photoelectrons", "background_photoelectrons"])
        pe = df["photons"] * (df["A_B"] / 4 + df["A_G"] / 2 + df["A_R"] / 4)
        lo, hi = pe.quantile([0.25, 0.75])
        filtered = SM_E.filter_quality_localisations(
            df, criteria=FilteringCriteria(
                chi_val=10.0, max_localisation_error=1.0, max_colour_error=1.0,
                min_sigma=0.01, max_sigma=10.0, max_sigma_error=10.0,
                min_photons=1.0, min_photoelectrons=lo, max_photoelectrons=hi),
        )
        np.testing.assert_allclose(np.sort(filtered["photoelectrons"]), np.sort(pe[(pe > lo) & (pe < hi)]))

    def test_no_photoelectron_bounds_leaves_photon_cut_unchanged(self):
        SM_E = SM_extractionfunctions.extract_SMs()
        df = _make_loc_df(30)
        kwargs = dict(chi_val=10.0, max_localisation_error=1.0, max_colour_error=1.0,
                      min_sigma=0.01, max_sigma=10.0, max_sigma_error=10.0, min_photons=900.0)
        filtered = SM_E.filter_quality_localisations(df, **kwargs)
        assert len(filtered) == int((df["photons"] > 900.0).sum())
        assert "photoelectrons" not in filtered.columns


class TestAverageParametersBranches:
    def test_index_column_is_skipped_in_both_loops(self):
        SM_E = SM_extractionfunctions.extract_SMs()
        df = _make_loc_df(20)
        df["index"] = np.arange(len(df))
        labels = np.array([0] * 10 + [1] * 10)
        result = SM_E.average_parameters(df, labels)
        assert "index" not in result.columns
        assert len(result) == 2

    def test_missing_photons_column_uses_weighted_amplitude_sum(self):
        SM_E = SM_extractionfunctions.extract_SMs()
        df = _make_loc_df(20, with_photons=False)
        assert "photons" not in df.columns
        labels = np.array([0] * 10 + [1] * 10)
        result = SM_E.average_parameters(df, labels)
        assert np.all(np.isfinite(result["photons"]))
        assert np.all(result["photons"] > 0)
