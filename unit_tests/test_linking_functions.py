#!/usr/bin/env python3
"""
Full coverage tests for pyS3M.LinkingFunctions -- post-hoc linking of repeated
blink detections across frames (`link_localisations`) and joint
spatial+spectral clustering of linked events (`joint_spectral_spatial_cluster`).

Part of the coverage push (claude/TODO.md PRIORITY 1). Unlike localise.py,
nothing here is dead: `link_localisations` is used in 6 developer-branch
research notebooks (confirmed via `git grep ... developer`, not present on
`main`), `joint_spectral_spatial_cluster` is used for real in
`channel_unmixing.py`, and the five `_link_group_*` numba helpers are
re-exported and used by `postprocess.py`. `test_spectral_lap_linking.py`
covers a different, unrelated linking algorithm (spectral LAP, in
clustering/) and doesn't touch this file at all -- confirmed 0% baseline.

The five `_link_group_*` helpers and the two `_get_*` greedy-chain helpers are
all `@numba.jit(nopython=True)` -- once JIT-compiled they bypass Python's
trace hooks entirely, so coverage.py can't see hits inside them no matter how
often they're called. Tests call each via both its normal (JIT) path, to
verify real behaviour, and its `.py_func` attribute (numba's uncompiled
escape hatch), purely so coverage.py can see the body execute.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pyS3M.LinkingFunctions as lf


# ======================================================================
# Low-level numba helpers
# ======================================================================

class TestLinkGroupCount:
    def test_counts_per_group(self):
        link_group = np.array([0, 0, 1, 1, 1, 2], dtype=np.int32)
        result = lf._link_group_count(link_group, 6, 3)
        np.testing.assert_array_equal(result, [2, 3, 1])

    def test_py_func_matches_jit(self):
        link_group = np.array([0, 1, 1], dtype=np.int32)
        np.testing.assert_array_equal(
            lf._link_group_count(link_group, 3, 2),
            lf._link_group_count.py_func(link_group, 3, 2),
        )


class TestLinkGroupSum:
    def test_sums_per_group(self):
        link_group = np.array([0, 0, 1], dtype=np.int32)
        col = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = lf._link_group_sum(col, link_group, 3, 2)
        np.testing.assert_allclose(result, [3.0, 3.0])

    def test_py_func_matches_jit(self):
        link_group = np.array([0, 0, 1], dtype=np.int32)
        col = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        np.testing.assert_allclose(
            lf._link_group_sum(col, link_group, 3, 2),
            lf._link_group_sum.py_func(col, link_group, 3, 2),
        )


class TestLinkGroupMean:
    def test_means_per_group(self):
        link_group = np.array([0, 0, 1], dtype=np.int32)
        col = np.array([1.0, 3.0, 5.0], dtype=np.float32)
        npg = lf._link_group_count(link_group, 3, 2)
        result = lf._link_group_mean(col, link_group, 3, 2, npg)
        np.testing.assert_allclose(result, [2.0, 5.0])

    def test_py_func_matches_jit(self):
        link_group = np.array([0, 0, 1], dtype=np.int32)
        col = np.array([1.0, 3.0, 5.0], dtype=np.float32)
        npg = lf._link_group_count(link_group, 3, 2)
        np.testing.assert_allclose(
            lf._link_group_mean(col, link_group, 3, 2, npg),
            lf._link_group_mean.py_func(col, link_group, 3, 2, npg),
        )


class TestLinkGroupWeightedMean:
    def test_weighted_mean(self):
        link_group = np.array([0, 0, 1], dtype=np.int32)
        col = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        weights = np.array([1.0, 2.0, 1.0], dtype=np.float32)
        npg = lf._link_group_count(link_group, 3, 2)
        result, sum_w = lf._link_group_weighted_mean(col, weights, link_group, 3, 2, npg)
        np.testing.assert_allclose(result, [5.0 / 3.0, 3.0])
        np.testing.assert_allclose(sum_w, [3.0, 1.0])

    def test_zero_weight_group_returns_zero(self):
        """sum_weights[i] == 0 -> the `else 0.0` branch."""
        link_group = np.array([0, 0, 1], dtype=np.int32)
        col = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        weights = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        npg = lf._link_group_count(link_group, 3, 2)
        result, sum_w = lf._link_group_weighted_mean(col, weights, link_group, 3, 2, npg)
        assert result[0] == 0.0
        assert sum_w[0] == 0.0

    def test_py_func_matches_jit(self):
        link_group = np.array([0, 0, 1], dtype=np.int32)
        col = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        weights = np.array([1.0, 2.0, 1.0], dtype=np.float32)
        npg = lf._link_group_count(link_group, 3, 2)
        r_jit = lf._link_group_weighted_mean(col, weights, link_group, 3, 2, npg)
        r_py = lf._link_group_weighted_mean.py_func(col, weights, link_group, 3, 2, npg)
        np.testing.assert_allclose(r_jit[0], r_py[0])
        np.testing.assert_allclose(r_jit[1], r_py[1])


class TestLinkGroupMinMax:
    def test_min_max_per_group(self):
        link_group = np.array([0, 0, 1], dtype=np.int32)
        col = np.array([1.0, 5.0, 3.0], dtype=np.float32)
        mn, mx = lf._link_group_min_max(col, link_group, 3, 2)
        np.testing.assert_allclose(mn, [1.0, 3.0])
        np.testing.assert_allclose(mx, [5.0, 3.0])

    def test_py_func_matches_jit(self):
        link_group = np.array([0, 0, 1], dtype=np.int32)
        col = np.array([1.0, 5.0, 3.0], dtype=np.float32)
        mn_jit, mx_jit = lf._link_group_min_max(col, link_group, 3, 2)
        mn_py, mx_py = lf._link_group_min_max.py_func(col, link_group, 3, 2)
        np.testing.assert_allclose(mn_jit, mn_py)
        np.testing.assert_allclose(mx_jit, mx_py)


# ======================================================================
# Greedy-chain linking core
# ======================================================================

class TestGetLinkGroups:
    def _basic_track(self):
        # Emitter A: frames 1,2,3 near (10,10). Emitter B: frame 8, isolated.
        frame = np.array([1, 2, 3, 8], dtype=np.int32)
        x = np.array([10.0, 10.01, 9.99, 30.0], dtype=np.float32)
        y = np.array([10.0, 9.99, 10.01, 30.0], dtype=np.float32)
        group = np.zeros(4, dtype=np.int32)
        return frame, x, y, group

    def test_links_nearby_consecutive_frames(self):
        frame, x, y, group = self._basic_track()
        link_group = lf._get_link_groups(frame, x, y, group, np.float32(0.5), 1)
        assert link_group[0] == link_group[1] == link_group[2]
        assert link_group[3] != link_group[0]

    def test_dark_time_gap_bridged_when_within_max_dark_time(self):
        frame = np.array([1, 4], dtype=np.int32)  # 2 dark frames in between
        x = np.array([10.0, 10.0], dtype=np.float32)
        y = np.array([10.0, 10.0], dtype=np.float32)
        group = np.zeros(2, dtype=np.int32)
        linked = lf._get_link_groups(frame, x, y, group, np.float32(0.5), max_dark_time=3)
        assert linked[0] == linked[1]
        not_linked = lf._get_link_groups(frame, x, y, group, np.float32(0.5), max_dark_time=1)
        assert not_linked[0] != not_linked[1]

    def test_different_group_ids_never_link(self):
        """group[] gates linking -- e.g. per-spectral-channel grouping."""
        frame = np.array([1, 2], dtype=np.int32)
        x = np.array([10.0, 10.0], dtype=np.float32)
        y = np.array([10.0, 10.0], dtype=np.float32)
        group = np.array([0, 1], dtype=np.int32)
        link_group = lf._get_link_groups(frame, x, y, group, np.float32(0.5), 1)
        assert link_group[0] != link_group[1]

    def test_py_func_matches_jit(self):
        frame, x, y, group = self._basic_track()
        jit = lf._get_link_groups(frame, x, y, group, np.float32(0.5), 1)
        py = lf._get_link_groups.py_func(frame, x, y, group, np.float32(0.5), 1)
        np.testing.assert_array_equal(jit, py)

    def test_get_next_loc_index_py_func_matches_jit(self):
        frame, x, y, group = self._basic_track()
        link_group = -np.ones(4, dtype=np.int32)
        link_group[0] = 0
        args = (0, link_group, 4, frame, x, y, np.float32(0.5), 1, group)
        assert lf._get_next_loc_index(*args) == lf._get_next_loc_index.py_func(*args)

    def test_get_next_loc_index_no_match_returns_minus_one(self):
        """Isolated localisation, nothing within range -> falls through to
        the final `return -1`."""
        frame, x, y, group = self._basic_track()
        link_group = -np.ones(4, dtype=np.int32)
        link_group[3] = 0
        args = (3, link_group, 4, frame, x, y, np.float32(0.5), 1, group)
        assert lf._get_next_loc_index(*args) == -1
        assert lf._get_next_loc_index.py_func(*args) == -1

    def test_get_next_loc_index_skips_duplicate_frame(self):
        """Two localisations sharing the current frame (e.g. different
        spectral channels detected in the same frame) -> the
        `frame[min_index] < min_frame` skip-forward loop actually runs."""
        frame = np.array([1, 1, 2], dtype=np.int32)
        x = np.array([10.0, 50.0, 10.0], dtype=np.float32)
        y = np.array([10.0, 50.0, 10.0], dtype=np.float32)
        group = np.zeros(3, dtype=np.int32)
        link_group = -np.ones(3, dtype=np.int32)
        link_group[0] = 0
        args = (0, link_group, 3, frame, x, y, np.float32(0.5), 1, group)
        assert lf._get_next_loc_index(*args) == 2
        assert lf._get_next_loc_index.py_func(*args) == 2


# ======================================================================
# link_localisations
# ======================================================================

def _base_df(n_extra_cols=True):
    df = pd.DataFrame({
        "frame": [1, 2, 3, 5, 6, 8],
        "xc": [10.0, 10.01, 9.99, 20.0, 20.02, 30.0],
        "yc": [10.0, 9.99, 10.01, 20.0, 19.98, 30.0],
        "xc_err": [0.02] * 6,
        "yc_err": [0.02] * 6,
    })
    if n_extra_cols:
        df["photoelectrons"] = [100.0, 110.0, 90.0, 200.0, 210.0, 50.0]
        df["bg_B"] = [5] * 6
        df["bg_G"] = [5] * 6
        df["bg_R"] = [5] * 6
        df["A_B"] = [0.3] * 6
        df["A_G"] = [0.3] * 6
        df["A_R"] = [0.4] * 6
        df["A_B_err"] = [0.01] * 6
        df["A_G_err"] = [0.01] * 6
        df["A_R_err"] = [0.01] * 6
        df["s_x"] = [1.2] * 6
        df["s_y"] = [1.2] * 6
        df["chi_sqr"] = [1.0] * 6
        df["extra_numeric_col"] = [1, 2, 3, 4, 5, 6]
    return df


class TestLinkLocalisations:
    def test_empty_dataframe(self):
        empty = pd.DataFrame({"frame": [], "xc": [], "yc": [], "xc_err": [], "yc_err": []})
        out = lf.link_localisations(empty, 10)
        assert len(out) == 0

    def test_full_columns_link_and_isolated(self):
        df = _base_df()
        linked = lf.link_localisations(df, n_frames=10, r_max=0.5, max_dark_time=1)
        assert len(linked) == 3
        assert set(linked["n"]) == {3, 2, 1}
        assert "photoelectron_rate" in linked.columns
        np.testing.assert_allclose(linked.loc[linked["n"] == 3, "photoelectron_rate"], 100.0)
        # extra_numeric_col passed through and averaged, not silently dropped
        assert "extra_numeric_col" in linked.columns

    def test_photoelectrons_summed_over_the_on_time(self):
        # A linked event's detected signal is its total over the on-time
        df = _base_df()
        linked = lf.link_localisations(df, n_frames=10, r_max=0.5, max_dark_time=1)
        np.testing.assert_allclose(linked.loc[linked["n"] == 3, "photoelectrons"], 300.0)
        np.testing.assert_allclose(linked.loc[linked["n"] == 2, "photoelectrons"], 410.0)

    def test_minimal_columns_no_photoelectrons_no_rate(self):
        df = _base_df(n_extra_cols=False)
        linked = lf.link_localisations(df, n_frames=10)
        assert "photoelectron_rate" not in linked.columns
        assert "photoelectrons" not in linked.columns

    def test_zero_error_guarded_by_nanmedian_fallback(self):
        df = pd.DataFrame({
            "frame": [1, 2, 3],
            "xc": [10.0, 10.0, 10.0],
            "yc": [10.0, 10.0, 10.0],
            "xc_err": [0.0, 0.02, 0.02],
            "yc_err": [0.02, 0.0, 0.02],
        })
        linked = lf.link_localisations(df, n_frames=10)
        assert np.all(np.isfinite(linked["xc_err"]))
        assert np.all(np.isfinite(linked["yc_err"]))

    def test_a_column_without_err_falls_back_to_simple_mean(self):
        df = pd.DataFrame({
            "frame": [1, 2, 3],
            "xc": [10.0] * 3, "yc": [10.0] * 3,
            "xc_err": [0.02] * 3, "yc_err": [0.02] * 3,
            "A_B": [0.3, 0.31, 0.29],
        })
        linked = lf.link_localisations(df, n_frames=10)
        assert linked["A_B"].iloc[0] == pytest.approx(0.3, abs=1e-5)
        assert "A_B_err" not in linked.columns

    def test_a_column_with_err_uses_weighted_mean(self):
        df = pd.DataFrame({
            "frame": [1, 2, 3],
            "xc": [10.0] * 3, "yc": [10.0] * 3,
            "xc_err": [0.02] * 3, "yc_err": [0.02] * 3,
            "A_R": [0.4, 0.42, 0.38], "A_R_err": [0.01, 0.02, 0.01],
        })
        linked = lf.link_localisations(df, n_frames=10)
        assert "A_R_err" in linked.columns
        assert np.isfinite(linked["A_R_err"].iloc[0])

    def test_chi_sqr_averaged(self):
        df = pd.DataFrame({
            "frame": [1, 2],
            "xc": [10.0, 10.0], "yc": [10.0, 10.0],
            "xc_err": [0.02, 0.02], "yc_err": [0.02, 0.02],
            "chi_sqr": [1.0, 2.0],
        })
        linked = lf.link_localisations(df, n_frames=10)
        assert linked["chi_sqr"].iloc[0] == pytest.approx(1.5)

    def test_non_numeric_extra_column_ignored(self):
        df = pd.DataFrame({
            "frame": [1, 2], "xc": [10.0, 10.0], "yc": [10.0, 10.0],
            "xc_err": [0.02, 0.02], "yc_err": [0.02, 0.02],
            "label": ["a", "b"],
        })
        linked = lf.link_localisations(df, n_frames=10)
        assert "label" not in linked.columns

    def test_remove_ambiguous_lengths_true_vs_false(self):
        df = _base_df(n_extra_cols=False)
        # n_frames=9 -> last group (frame 8) sits on the last frame (8 == 9-1)
        with_removal = lf.link_localisations(df, n_frames=9, remove_ambiguous_lengths=True)
        without_removal = lf.link_localisations(df, n_frames=9, remove_ambiguous_lengths=False)
        assert len(with_removal) < len(without_removal)


# ======================================================================
# joint_spectral_spatial_cluster
# ======================================================================

def _make_group(cx, cy, n, seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "n": [1] * n,
        "xc": cx + rng.normal(0, 0.01, n), "yc": cy + rng.normal(0, 0.01, n),
        "A_R": [0.4] * n, "A_G": [0.3] * n,
        "xc_err": [0.05] * n, "yc_err": [0.05] * n,
        "A_R_err": [0.02] * n, "A_G_err": [0.02] * n,
    })


class TestJointSpectralSpatialCluster:
    def test_missing_n_column_raises(self):
        df = _make_group(1, 1, 2, 0).drop(columns=["n"])
        with pytest.raises(ValueError, match="column 'n' missing"):
            lf.joint_spectral_spatial_cluster(df)

    def test_missing_required_feature_column_raises(self):
        df = _make_group(1, 1, 2, 0).drop(columns=["A_G"])
        with pytest.raises(ValueError, match="Required column 'A_G' not found"):
            lf.joint_spectral_spatial_cluster(df)

    def test_no_pairs_within_threshold_all_isolated(self):
        n = 5
        df = pd.DataFrame({
            "n": [1] * n,
            "xc": np.arange(n) * 1000.0, "yc": np.arange(n) * 1000.0,
            "A_R": [0.4] * n, "A_G": [0.3] * n,
            "xc_err": [0.05] * n, "yc_err": [0.05] * n,
            "A_R_err": [0.02] * n, "A_G_err": [0.02] * n,
        })
        out = lf.joint_spectral_spatial_cluster(df)
        assert (out["joint_cluster_id"] == -1).all()

    def test_two_clusters_found(self):
        df = pd.concat([_make_group(10, 10, 4, 1), _make_group(50, 50, 4, 2)], ignore_index=True)
        out = lf.joint_spectral_spatial_cluster(df, d_threshold=3.0, min_cluster_size=3)
        ids = out["joint_cluster_id"].to_numpy()
        assert set(ids[:4]) == {ids[0]}
        assert set(ids[4:]) == {ids[4]}
        assert ids[0] != ids[4]
        assert ids[0] != -1 and ids[4] != -1

    def test_cluster_below_min_size_marked_isolated(self):
        df = _make_group(90, 90, 2, 3)
        out = lf.joint_spectral_spatial_cluster(df, d_threshold=3.0, min_cluster_size=3)
        assert (out["joint_cluster_id"] == -1).all()

    def test_custom_column_names(self):
        df = _make_group(10, 10, 4, 1).rename(columns={
            "xc": "x", "yc": "y", "A_R": "red", "A_G": "green",
            "xc_err": "x_err", "yc_err": "y_err", "A_R_err": "red_err", "A_G_err": "green_err",
        })
        out = lf.joint_spectral_spatial_cluster(
            df,
            spatial_cols=["x", "y"], spectral_cols=["red", "green"],
            spatial_err_cols=["x_err", "y_err"], spectral_err_cols=["red_err", "green_err"],
            d_threshold=3.0, min_cluster_size=3,
        )
        assert (out["joint_cluster_id"] != -1).all()
