"""Tests for spectral-assisted LAP linking in SM_extractionfunctions.

Creates synthetic localisation data with known molecule identities and
crossing trajectories to verify that spectral LAP linking correctly
maintains track identity using colour information.
"""

import sys
import os
import numpy as np
import pandas as pd
import unittest


import pyS3M.SM_extractionfunctions as SM_extractionfunctions


def _make_crossing_trajectories(
    n_frames=20,
    seed=42,
):
    """Create 3 molecules with distinct colours whose paths cross.

    Molecule A (red):   moves right across FOV
    Molecule B (green): moves left across FOV  (crosses A around frame 10)
    Molecule C (blue):  stationary

    Returns:
        DataFrame with one row per localisation, columns:
            frame, xc, yc, A_R, A_G, A_B, A_R_err, A_G_err, A_B_err,
            s_x, s_y, s_x_err, s_y_err, xc_err, yc_err, chi_sqr,
            photons, true_id
    """
    rng = np.random.default_rng(seed)
    rows = []

    for f in range(n_frames):
        # Molecule A: red, moving right
        rows.append({
            "frame": f,
            "xc": 5.0 + f * 0.5 + rng.normal(0, 0.05),
            "yc": 10.0 + rng.normal(0, 0.05),
            "A_R": 0.7 + rng.normal(0, 0.02),
            "A_G": 0.2 + rng.normal(0, 0.02),
            "A_B": 0.1 + rng.normal(0, 0.02),
            "true_id": 0,
        })

        # Molecule B: green, moving left (crosses A near frame 10)
        rows.append({
            "frame": f,
            "xc": 15.0 - f * 0.5 + rng.normal(0, 0.05),
            "yc": 10.0 + rng.normal(0, 0.05),
            "A_R": 0.15 + rng.normal(0, 0.02),
            "A_G": 0.7 + rng.normal(0, 0.02),
            "A_B": 0.15 + rng.normal(0, 0.02),
            "true_id": 1,
        })

        # Molecule C: blue, stationary
        rows.append({
            "frame": f,
            "xc": 10.0 + rng.normal(0, 0.05),
            "yc": 5.0 + rng.normal(0, 0.05),
            "A_R": 0.1 + rng.normal(0, 0.02),
            "A_G": 0.15 + rng.normal(0, 0.02),
            "A_B": 0.75 + rng.normal(0, 0.02),
            "true_id": 2,
        })

    df = pd.DataFrame(rows)

    # Add required error/quality columns
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


def _track_purity(link_groups, true_ids):
    """Compute purity: fraction of locs in each track with the majority true_id."""
    linked_mask = link_groups >= 0
    lg = link_groups[linked_mask]
    ti = true_ids[linked_mask]

    if len(lg) == 0:
        return 0.0

    correct = 0
    total = 0
    for track_id in np.unique(lg):
        mask = lg == track_id
        ids_in_track = ti[mask]
        # Most common true_id in this track
        majority = np.bincount(ids_in_track.astype(int)).argmax()
        correct += np.sum(ids_in_track == majority)
        total += len(ids_in_track)

    return correct / total


class TestSpectralLAPLinking(unittest.TestCase):
    """Test spectral LAP linking on synthetic crossing trajectories."""

    @classmethod
    def setUpClass(cls):
        cls.df = _make_crossing_trajectories(n_frames=20, seed=42)
        cls.sm = SM_extractionfunctions.extract_SMs()

    def test_basic_linking_output(self):
        """spectral_lap_link returns correct shape and dtype."""
        link_groups = self.sm.spectral_lap_link(
            self.df.sort_values("frame"),
            max_distance=2.0,
            max_dark_time=1,
        )
        self.assertEqual(len(link_groups), len(self.df))
        self.assertEqual(link_groups.dtype, np.int32)
        # All locs should be linked (3 molecules, continuous frames)
        self.assertTrue(np.all(link_groups >= 0))

    def test_three_tracks_found(self):
        """Should find exactly 3 tracks for 3 molecules."""
        link_groups = self.sm.spectral_lap_link(
            self.df.sort_values("frame"),
            max_distance=2.0,
            max_dark_time=1,
        )
        n_tracks = len(np.unique(link_groups[link_groups >= 0]))
        self.assertEqual(n_tracks, 3, f"Expected 3 tracks, got {n_tracks}")

    def test_spectral_lap_maintains_identity(self):
        """Spectral LAP should maintain track identity through crossing."""
        link_groups = self.sm.spectral_lap_link(
            self.df.sort_values("frame"),
            max_distance=2.0,
            max_dark_time=1,
            w_spatial=1.0,
            w_spectral=0.5,
        )
        true_ids = self.df.sort_values("frame")["true_id"].to_numpy()
        purity = _track_purity(link_groups, true_ids)

        self.assertGreaterEqual(
            purity, 0.95,
            f"Track purity {purity:.2%} is below 95% — "
            f"spectral LAP should maintain identity through crossing"
        )

    def test_greedy_nn_swaps_at_crossing(self):
        """Greedy NN (no spectral info) should have lower purity at crossing."""
        import pyS3M.postprocess as postprocess

        df_sorted = self.df.sort_values("frame")
        loc_array = df_sorted.to_records(index=False)
        group = np.zeros(len(loc_array), dtype=np.int32)

        link_groups_nn = postprocess.get_link_groups(
            loc_array, 2.0, 1, group
        )
        true_ids = df_sorted["true_id"].to_numpy()
        purity_nn = _track_purity(link_groups_nn, true_ids)

        # Also run spectral LAP for comparison
        link_groups_lap = self.sm.spectral_lap_link(
            df_sorted,
            max_distance=2.0,
            max_dark_time=1,
            w_spatial=1.0,
            w_spectral=0.5,
        )
        purity_lap = _track_purity(link_groups_lap, true_ids)

        print(f"\n  Track purity comparison:")
        print(f"    Greedy NN:     {purity_nn:.2%}")
        print(f"    Spectral LAP:  {purity_lap:.2%}")

        # Spectral LAP should be >= greedy NN
        self.assertGreaterEqual(
            purity_lap, purity_nn,
            "Spectral LAP should have equal or better purity than greedy NN"
        )

    def test_extract_single_molecules_spectral_lap(self):
        """Public API method should return compatible output format."""
        sm_db, sf_db = self.sm.extract_single_molecules_spectral_lap(
            self.df.copy(),
            max_distance=2.0,
            max_dark_time=1,
            w_spatial=1.0,
            w_spectral=0.5,
            chi_val=10.0,  # permissive filter
            max_localisation_error=1.0,
            min_photoelectrons=100,
            max_photoelectrons=1e6,
            # Molecule C is deliberately stationary (see _make_crossing_trajectories),
            # and extract_single_molecules_spectral_lap's remove_static=True default
            # correctly flags/drops it (DBSCAN eps=1.0, min_samples=10 vs. C's 20
            # localisations jittering by ~0.05px) -- a real, working feature, not a
            # bug. This test is about the linking/extraction pipeline's molecule
            # count, not static removal (no other test in this file exercises it),
            # so disable it here to keep all 3 synthetic molecules.
            remove_static=False,
        )

        # Should return DataFrames
        self.assertIsInstance(sm_db, pd.DataFrame)
        self.assertIsInstance(sf_db, pd.DataFrame)

        # Single molecule database should have molecular_index
        self.assertIn("molecular_index", sm_db.columns)
        self.assertIn("molecular_index", sf_db.columns)

        # Should have 3 molecules
        self.assertEqual(len(sm_db), 3, f"Expected 3 molecules, got {len(sm_db)}")

    def test_dark_time_gap_closing(self):
        """Tracks should persist across dark frames within max_dark_time."""
        # Remove frame 10 for molecule A to create a gap
        df_gap = self.df.copy()
        drop_mask = (df_gap["frame"] == 10) & (df_gap["true_id"] == 0)
        df_gap = df_gap[~drop_mask].copy()

        link_groups = self.sm.spectral_lap_link(
            df_gap.sort_values("frame"),
            max_distance=2.0,
            max_dark_time=2,  # allow 2-frame gap
        )

        # Molecule A should still be one track despite the gap
        true_ids = df_gap.sort_values("frame")["true_id"].to_numpy()
        purity = _track_purity(link_groups, true_ids)
        n_tracks = len(np.unique(link_groups[link_groups >= 0]))

        self.assertEqual(n_tracks, 3,
                         f"Expected 3 tracks with gap closing, got {n_tracks}")
        self.assertGreaterEqual(purity, 0.95)


if __name__ == "__main__":
    unittest.main(verbosity=2)
