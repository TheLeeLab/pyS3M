"""Coverage tests for pyS3M.clustering.linked_clusterer's remaining branches
not already exercised by test_spectral_lap_linking.py: the `config=` override
path on both extract_single_molecules_linked and extract_single_molecules_spectral_lap,
flag_static_localisations directly, spectral_lap_link's D_prior gap-scaled-cutoff
branch, the mid-sequence new-track-creation branch (a molecule appearing after
frame 0 that can't be matched to any active track), the verbose progress-log
branch (needs >=501 unique frames to hit the `fi % 500 == 0` checkpoint), and
extract_single_molecules_spectral_lap's remove_static=True + verbose branches.

Reuses the crossing-trajectory generator pattern from test_spectral_lap_linking.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pyS3M.SM_extractionfunctions as SM_extractionfunctions
from pyS3M.clustering import ClusteringConfig


def _make_two_molecules(n_frames=20, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(n_frames):
        rows.append({
            "frame": f, "xc": 5.0 + f * 0.1 + rng.normal(0, 0.02),
            "yc": 10.0 + rng.normal(0, 0.02),
            "A_R": 0.7, "A_G": 0.2, "A_B": 0.1,
        })
        rows.append({
            "frame": f, "xc": 30.0 + rng.normal(0, 0.02),
            "yc": 30.0 + rng.normal(0, 0.02),
            "A_R": 0.1, "A_G": 0.7, "A_B": 0.2,
        })
    df = pd.DataFrame(rows)
    df["A_R_err"] = 0.03
    df["A_G_err"] = 0.03
    df["A_B_err"] = 0.03
    df["bg_R_err"] = 0.01
    df["bg_G_err"] = 0.01
    df["bg_B_err"] = 0.01
    df["s_x"] = 1.5
    df["s_y"] = 1.5
    df["s_x_err"] = 0.1
    df["s_y_err"] = 0.1
    df["xc_err"] = 0.05
    df["yc_err"] = 0.05
    df["chi_sqr"] = 1.0
    df["photoelectrons"] = 1000.0
    return df


class TestExtractSingleMoleculesLinkedConfig:
    def test_config_overrides_kwargs(self):
        SM_E = SM_extractionfunctions.extract_SMs()
        df = _make_two_molecules(n_frames=10)
        cfg = ClusteringConfig(max_distance=2.0, max_frames=5, start_frame=0)
        sm_db, sf_db = SM_E.extract_single_molecules_linked(
            df, config=cfg, max_distance=0.001,  # overridden by config
        )
        assert isinstance(sm_db, pd.DataFrame)


class TestFlagStaticLocalisations:
    def test_flags_dense_cluster_as_static(self):
        SM_E = SM_extractionfunctions.extract_SMs()
        rng = np.random.default_rng(3)
        # 15 tightly-clustered points (static) + 15 widely-scattered points (mobile).
        static_xy = rng.normal(0, 0.01, size=(15, 2)) + [10.0, 10.0]
        mobile_xy = rng.uniform(0, 100, size=(15, 2))
        xy = np.vstack([static_xy, mobile_xy])
        loc_data = pd.DataFrame({"xc": xy[:, 0], "yc": xy[:, 1]})

        flags = SM_E.flag_static_localisations(loc_data, eps=0.5, min_samples=10)
        assert flags.dtype == bool
        assert flags[:15].all()
        assert not flags[15:].any()


class TestSpectralLapLinkDPrior:
    def test_d_prior_gap_scaled_cutoff(self):
        SM_E = SM_extractionfunctions.extract_SMs()
        df = _make_two_molecules(n_frames=15)
        link_groups = SM_E.spectral_lap_link(
            df.sort_values("frame"), max_distance=2.0, max_dark_time=2,
            D_prior=0.1, dt=1.0, sigma_loc=0.05, alpha=3.0,
        )
        assert len(link_groups) == len(df)
        assert link_groups.dtype == np.int32


class TestSpectralLapLinkNewTrackMidSequence:
    def test_molecule_appearing_after_frame_zero_starts_new_track(self):
        SM_E = SM_extractionfunctions.extract_SMs()
        rng = np.random.default_rng(21)
        rows = []
        # Molecule A present frames 0-9.
        for f in range(10):
            rows.append({"frame": f, "xc": 5.0, "yc": 5.0, "A_R": 0.7, "A_G": 0.2, "A_B": 0.1})
        # Molecule B appears only from frame 5 onward, far from A (so it can't
        # be matched to A's active track -> exercises the "new track created
        # for an unmatched current localisation" branch, distinct from the
        # frame-0 n_active==0 initial-track-creation branch).
        for f in range(5, 10):
            rows.append({"frame": f, "xc": 80.0, "yc": 80.0, "A_R": 0.1, "A_G": 0.7, "A_B": 0.2})
        df = pd.DataFrame(rows)
        for col, val in [("A_R_err", 0.03), ("A_G_err", 0.03), ("A_B_err", 0.03)]:
            df[col] = val

        link_groups = SM_E.spectral_lap_link(
            df.sort_values("frame"), max_distance=2.0, max_dark_time=1,
        )
        n_tracks = len(np.unique(link_groups[link_groups >= 0]))
        assert n_tracks == 2


class TestSpectralLapLinkVerboseProgress:
    def test_verbose_progress_log_at_500_frames(self):
        SM_E = SM_extractionfunctions.extract_SMs()
        rng = np.random.default_rng(31)
        n_frames = 505
        rows = [
            {
                "frame": f, "xc": 5.0 + rng.normal(0, 0.02), "yc": 5.0 + rng.normal(0, 0.02),
                "A_R": 0.7, "A_G": 0.2, "A_B": 0.1,
            }
            for f in range(n_frames)
        ]
        df = pd.DataFrame(rows)
        for col, val in [("A_R_err", 0.03), ("A_G_err", 0.03), ("A_B_err", 0.03)]:
            df[col] = val

        link_groups = SM_E.spectral_lap_link(
            df.sort_values("frame"), max_distance=2.0, max_dark_time=1, verbose=True,
        )
        assert len(link_groups) == n_frames


class TestExtractSingleMoleculesSpectralLapExtra:
    def test_config_overrides_kwargs(self):
        SM_E = SM_extractionfunctions.extract_SMs()
        df = _make_two_molecules(n_frames=10)
        cfg = ClusteringConfig(
            max_distance=2.0, max_dark_time=1, w_spatial=1.0, w_spectral=0.5,
            min_frames=1, remove_static=False, verbose=False,
        )
        sm_db, sf_db = SM_E.extract_single_molecules_spectral_lap(
            df.copy(), config=cfg, max_distance=0.001,  # overridden by config
            chi_val=10.0, max_localisation_error=1.0, min_photoelectrons=100, max_photoelectrons=1e6,
        )
        assert isinstance(sm_db, pd.DataFrame)

    def test_remove_static_true_with_verbose(self):
        SM_E = SM_extractionfunctions.extract_SMs()
        rng = np.random.default_rng(41)
        rows = []
        # Static molecule: tight cluster, 15 frames.
        for f in range(15):
            rows.append({
                "frame": f, "xc": 10.0 + rng.normal(0, 0.01), "yc": 10.0 + rng.normal(0, 0.01),
                "A_R": 0.1, "A_G": 0.1, "A_B": 0.8,
            })
        # Mobile molecule: moves steadily, 15 frames.
        for f in range(15):
            rows.append({
                "frame": f, "xc": 30.0 + f * 0.5, "yc": 30.0 + rng.normal(0, 0.02),
                "A_R": 0.7, "A_G": 0.2, "A_B": 0.1,
            })
        df = pd.DataFrame(rows)
        for col, val in [
            ("A_R_err", 0.03), ("A_G_err", 0.03), ("A_B_err", 0.03),
            ("bg_R_err", 0.01), ("bg_G_err", 0.01), ("bg_B_err", 0.01),
            ("s_x", 1.5), ("s_y", 1.5), ("s_x_err", 0.1), ("s_y_err", 0.1),
            ("xc_err", 0.05), ("yc_err", 0.05), ("chi_sqr", 1.0), ("photoelectrons", 1000.0),
        ]:
            df[col] = val

        sm_db, sf_db = SM_E.extract_single_molecules_spectral_lap(
            df.copy(), max_distance=2.0, max_dark_time=1, min_frames=1,
            chi_val=10.0, max_localisation_error=1.0, min_photoelectrons=100, max_photoelectrons=1e6,
            remove_static=True, static_eps=0.5, static_min_samples=10, verbose=True,
        )
        # Only the mobile molecule should survive static removal.
        assert len(sm_db) == 1
