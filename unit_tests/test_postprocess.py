#!/usr/bin/env python3
"""
Full coverage tests for pyS3M.postprocess -- picking/grouping, temporal linking,
picked-based undrift, image-based aggregate segmentation, and fiducial removal.

`segment_locs_by_rendered_image` is also exercised by
`test_verbose_segmentation.py`/`test_image_based_segmentation.py`; this file
targets the gaps those leave, mainly the real valid-aggregate success path
Step 5 extraction.
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import pyS3M.postprocess as pp
from pyS3M.Constants import AnalysisConfig


def _locs(n=200, seed=0, width=100.0, height=100.0, n_frames=50):
    rng = np.random.default_rng(seed)
    xc = rng.uniform(0, width, n).astype(np.float32)
    yc = rng.uniform(0, height, n).astype(np.float32)
    frame = rng.integers(0, n_frames, n).astype(np.int32)
    return np.rec.fromarrays([xc, yc, frame], names=["xc", "yc", "frame"])


# ======================================================================
# get_index_blocks / index_blocks_shape / get_block_locs_at
# ======================================================================

class TestIndexBlocks:
    def test_index_blocks_shape(self):
        assert pp.index_blocks_shape(100.0, 50.0, 5.0) == (10, 20)

    def test_get_index_blocks_and_lookup(self):
        locs = _locs(n=300)
        index_blocks = pp.get_index_blocks(locs, 100.0, 100.0, 5.0)
        assert len(index_blocks) == 8
        block_locs = pp.get_block_locs_at(50.0, 50.0, index_blocks)
        assert isinstance(block_locs, np.recarray)

    def test_fill_index_block_py_func_matches_jit(self):
        block_starts = np.zeros((2, 2), dtype=np.uint32)
        block_ends = np.zeros((2, 2), dtype=np.uint32)
        x_index = np.array([0, 0, 1], dtype=np.uint32)
        y_index = np.array([0, 0, 1], dtype=np.uint32)
        jit = pp._fill_index_block(block_starts.copy(), block_ends.copy(), 3, x_index, y_index, 0, 0, 0)
        py = pp._fill_index_block.py_func(block_starts.copy(), block_ends.copy(), 3, x_index, y_index, 0, 0, 0)
        assert jit == py

    def test_fill_index_blocks_py_func_matches_jit(self):
        x_index = np.array([0, 0, 1], dtype=np.uint32)
        y_index = np.array([0, 0, 1], dtype=np.uint32)
        bs1, be1 = np.zeros((2, 2), dtype=np.uint32), np.zeros((2, 2), dtype=np.uint32)
        bs2, be2 = np.zeros((2, 2), dtype=np.uint32), np.zeros((2, 2), dtype=np.uint32)
        pp._fill_index_blocks(bs1, be1, x_index, y_index)
        pp._fill_index_blocks.py_func(bs2, be2, x_index, y_index)
        np.testing.assert_array_equal(bs1, bs2)
        np.testing.assert_array_equal(be1, be2)


# ======================================================================
# picked_locs -- Circle / Rectangle / Polygon dispatch
# ======================================================================

class TestPickedLocs:
    def test_circle_picks(self):
        locs = _locs(n=200)
        picks = [(float(locs.xc[0]), float(locs.yc[0])), (float(locs.xc[5]), float(locs.yc[5]))]
        out = pp.picked_locs(locs, 100, 100, picks, "Circle", pick_size=5.0, add_group=True)
        assert len(out) == 2
        assert "group" in out[0].dtype.names

    def test_circle_picks_no_group(self):
        locs = _locs(n=200)
        picks = [(float(locs.xc[0]), float(locs.yc[0]))]
        out = pp.picked_locs(locs, 100, 100, picks, "Circle", pick_size=5.0, add_group=False)
        assert "group" not in out[0].dtype.names

    def test_rectangle_picks(self):
        locs = _locs(n=200)
        picks = [((10.0, 10.0), (20.0, 20.0))]
        out = pp.picked_locs(locs, 100, 100, picks, "Rectangle", pick_size=5.0)
        assert len(out) == 1
        assert "x_pick_rot" in out[0].dtype.names

    def test_polygon_picks(self):
        locs = _locs(n=200)
        poly = [(0.0, 0.0), (50.0, 0.0), (50.0, 50.0), (0.0, 50.0), (0.0, 0.0)]
        out = pp.picked_locs(locs, 100, 100, [poly], "Polygon")
        assert len(out) == 1

    def test_polygon_degenerate_pick_skipped(self):
        """Not a closed polygon (<3 points) -> get_pick_polygon_corners
        returns (None, None), hitting the `if X is None: continue` branch."""
        locs = _locs(n=200)
        calls = []
        out = pp.picked_locs(
            locs, 100, 100, [[(0.0, 0.0), (1.0, 1.0)]], "Polygon",
            callback=lambda i: calls.append(i),
        )
        assert out == []
        assert calls == [1]

    def test_invalid_pick_shape_raises(self):
        locs = _locs(n=10)
        with pytest.raises(ValueError, match="Invalid pick shape"):
            pp.picked_locs(locs, 100, 100, [(1.0, 1.0)], "Triangle")

    def test_empty_picks_returns_none(self):
        locs = _locs(n=10)
        assert pp.picked_locs(locs, 100, 100, [], "Circle") is None

    def test_console_callback_circle_and_rectangle(self):
        locs = _locs(n=200)
        picks = [(float(locs.xc[0]), float(locs.yc[0])), (float(locs.xc[1]), float(locs.yc[1]))]
        out = pp.picked_locs(locs, 100, 100, picks, "Circle", pick_size=5.0, callback="console")
        assert len(out) == 2
        out_r = pp.picked_locs(
            locs, 100, 100, [((10.0, 10.0), (20.0, 20.0))], "Rectangle", pick_size=5.0, callback="console"
        )
        assert len(out_r) == 1

    def test_console_callback_polygon_valid_and_degenerate(self):
        """Covers both the degenerate-pick (`X is None`) and normal-pick
        branches of the Polygon path with callback='console'."""
        locs = _locs(n=200)
        poly = [(0.0, 0.0), (50.0, 0.0), (50.0, 50.0), (0.0, 50.0), (0.0, 0.0)]
        degenerate = [(0.0, 0.0), (1.0, 1.0)]
        out = pp.picked_locs(locs, 100, 100, [degenerate, poly], "Polygon", callback="console")
        assert len(out) == 1  # only the valid polygon produced a result

    def test_callable_callback_polygon_valid_pick(self):
        locs = _locs(n=200)
        poly = [(0.0, 0.0), (50.0, 0.0), (50.0, 50.0), (0.0, 50.0), (0.0, 0.0)]
        calls = []
        out = pp.picked_locs(locs, 100, 100, [poly], "Polygon", callback=lambda i: calls.append(i))
        assert len(out) == 1
        assert calls == [1]

    def test_callable_callback_circle_and_rectangle(self):
        locs = _locs(n=200)
        calls = []
        pp.picked_locs(
            locs, 100, 100, [(float(locs.xc[0]), float(locs.yc[0]))], "Circle",
            pick_size=5.0, callback=lambda i: calls.append(("c", i)),
        )
        pp.picked_locs(
            locs, 100, 100, [((10.0, 10.0), (20.0, 20.0))], "Rectangle",
            pick_size=5.0, callback=lambda i: calls.append(("r", i)),
        )
        assert ("c", 1) in calls and ("r", 1) in calls


# ======================================================================
# get_link_groups / _get_next_loc_index_in_link_group
# ======================================================================

class TestGetLinkGroups:
    def _track(self):
        return np.rec.fromarrays(
            [
                np.array([1, 2, 3, 8], dtype=np.int32),
                np.array([10.0, 10.01, 9.99, 30.0], dtype=np.float32),
                np.array([10.0, 9.99, 10.01, 30.0], dtype=np.float32),
            ],
            names=["frame", "xc", "yc"],
        )

    def test_links_nearby_frames(self):
        locs = self._track()
        group = np.zeros(4, dtype=np.int32)
        lg = pp.get_link_groups(locs, np.float32(0.5), 1, group)
        assert lg[0] == lg[1] == lg[2]
        assert lg[3] != lg[0]

    def test_py_func_matches_jit(self):
        locs = self._track()
        group = np.zeros(4, dtype=np.int32)
        jit = pp.get_link_groups(locs, np.float32(0.5), 1, group)
        py = pp.get_link_groups.py_func(locs, np.float32(0.5), 1, group)
        np.testing.assert_array_equal(jit, py)

    def test_no_match_returns_minus_one(self):
        """current_index=2 (not the last index) so the `for min_index in
        range(current_index+1, N)` loop always has >=1 iteration in both
        the JIT and .py_func paths -- see test below for what happens at
        the very last index, which the two paths handle differently."""
        locs = self._track()
        link_group = -np.ones(4, dtype=np.int32)
        link_group[2] = 0
        group = np.zeros(4, dtype=np.int32)
        args = (2, link_group, 4, locs.frame, locs.xc, locs.yc, np.float32(0.5), 1, group)
        assert pp._get_next_loc_index_in_link_group(*args) == -1
        assert pp._get_next_loc_index_in_link_group.py_func(*args) == -1

    def test_last_index_jit_vs_py_func_diverge(self):
        """A real, latent JIT/interpreted-Python divergence found while
        writing this test: at the very last index (current_index+1 == N),
        `for min_index in range(current_index+1, N): ...` has zero
        iterations, leaving `min_index` unbound. Numba's nopython mode
        tolerates this (the loop variable ends up usable, and the function
        returns -1 correctly); plain interpreted Python does not --
        `.py_func` raises UnboundLocalError. Only reachable in practice if
        this function were ever called without its @numba.jit decorator
        (e.g. via .py_func, or if numba were ever disabled/removed) --
        the always-JIT-compiled real code path never hits it. Documented
        here rather than "fixed" since fixing would mean changing the
        numba-compiled function's real behaviour to chase an interpreted-
        mode-only edge case."""
        locs = self._track()
        link_group = -np.ones(4, dtype=np.int32)
        link_group[3] = 0
        group = np.zeros(4, dtype=np.int32)
        args = (3, link_group, 4, locs.frame, locs.xc, locs.yc, np.float32(0.5), 1, group)
        assert pp._get_next_loc_index_in_link_group(*args) == -1
        with pytest.raises(UnboundLocalError):
            pp._get_next_loc_index_in_link_group.py_func(*args)

    def test_max_index_falls_off_end_via_for_else(self):
        """current_index is near the end of the array so the max_index
        for-loop runs to completion without a `break` -> the `for...else:
        max_index = N` branch (distinct from the `break`-out path)."""
        locs = self._track()  # last frame = 8, only 4 locs total
        link_group = -np.ones(4, dtype=np.int32)
        link_group[2] = 0  # current_index = 2 (frame=3); nothing left within
        group = np.zeros(4, dtype=np.int32)
        args = (2, link_group, 4, locs.frame, locs.xc, locs.yc, np.float32(50.0), 10, group)
        # roi is huge and dark time huge -> max_index loop runs off the end
        result = pp._get_next_loc_index_in_link_group(*args)
        assert result == 3  # links to the last (index 3) within the huge window
        assert pp._get_next_loc_index_in_link_group.py_func(*args) == 3


# ======================================================================
# undrift_from_picked / _undrift_from_picked_coordinate
# ======================================================================

class TestUndriftFromPicked:
    def _pick(self, cx, cy, frames, seed):
        rng = np.random.default_rng(seed)
        frames = np.array(frames)
        xc = cx + rng.normal(0, 0.02, len(frames))
        yc = cy + rng.normal(0, 0.02, len(frames))
        return np.rec.fromarrays([frames, xc, yc], names=["frame", "xc", "yc"])

    def test_multiple_picks_full_coverage(self):
        n_frames = 10
        picks = [self._pick(5, 5, range(10), 1), self._pick(10, 10, range(10), 2)]
        drift = pp.undrift_from_picked(picks, n_frames)
        assert drift.dtype.names == ("xc", "yc")
        assert len(drift.xc) == n_frames
        assert np.all(np.isfinite(drift.xc))

    def test_picks_with_gaps_interpolated(self):
        """One pick missing some frames -> NaN gaps get linearly interpolated."""
        n_frames = 10
        picks = [self._pick(5, 5, range(10), 1), self._pick(10, 10, [0, 1, 2, 7, 8, 9], 2)]
        drift = pp.undrift_from_picked(picks, n_frames)
        assert np.all(np.isfinite(drift.xc))
        assert np.all(np.isfinite(drift.yc))


# ======================================================================
# Parallel / serial pick-processing branches
# ======================================================================

class TestParallelAndSerialPicking:
    def _many_picks_rect(self, locs, n=10):
        return [
            ((float(locs.xc[i] - 2), float(locs.yc[i] - 2)), (float(locs.xc[i] + 2), float(locs.yc[i] + 2)))
            for i in range(n)
        ]

    def _many_picks_circ(self, locs, n=10):
        return [(float(locs.xc[i]), float(locs.yc[i])) for i in range(n)]

    def test_parallel_rectangle_success_console(self):
        locs = _locs(n=500)
        picks = self._many_picks_rect(locs)
        out = pp.picked_locs(locs, 100, 100, picks, "Rectangle", pick_size=2.0, parallel=True, callback="console")
        assert len(out) == 10

    def test_parallel_rectangle_success_callable(self):
        locs = _locs(n=500)
        picks = self._many_picks_rect(locs)
        calls = []
        out = pp._parallel_picked_locs_rectangle(locs, 100, 100, picks, 2.0, callback=lambda i: calls.append(i))
        assert len(out) == 10
        assert len(calls) == 10

    def test_parallel_circle_success_console(self):
        locs = _locs(n=500)
        picks = self._many_picks_circ(locs)
        out = pp.picked_locs(locs, 100, 100, picks, "Circle", pick_size=2.0, parallel=True, callback="console")
        assert len(out) == 10

    def test_parallel_circle_success_callable(self):
        locs = _locs(n=500)
        picks = self._many_picks_circ(locs)
        calls = []
        out = pp._parallel_picked_locs_circle(locs, 100, 100, picks, 2.0, callback=lambda i: calls.append(i))
        assert len(out) == 10
        assert len(calls) == 10

    def test_parallel_below_threshold_falls_back_to_serial_path(self):
        """< 8 picks with parallel=True never enters the ThreadPoolExecutor
        branch in _parallel_picked_locs_* -- it calls the serial function
        directly instead."""
        locs = _locs(n=200)
        picks = self._many_picks_rect(locs, n=3)
        out = pp._parallel_picked_locs_rectangle(locs, 100, 100, picks, 2.0)
        assert len(out) == 3
        picks_c = self._many_picks_circ(locs, n=3)
        out_c = pp._parallel_picked_locs_circle(locs, 100, 100, picks_c, 2.0)
        assert len(out_c) == 3

    def test_parallel_rectangle_empty_picks(self):
        locs = _locs(n=10)
        assert pp._parallel_picked_locs_rectangle(locs, 100, 100, [], 2.0) == []

    def test_parallel_circle_empty_picks(self):
        locs = _locs(n=10)
        assert pp._parallel_picked_locs_circle(locs, 100, 100, [], 2.0) == []

    def test_process_single_rectangle_pick_direct(self):
        locs = _locs(n=200)
        idx, result = pp._process_single_rectangle_pick(locs, 5, ((10.0, 10.0), (20.0, 20.0)), 5.0, True)
        assert idx == 5
        assert "group" in result.dtype.names

    def test_process_single_rectangle_pick_error_returns_empty(self):
        """Malformed pick (wrong tuple shape) -> internal except -> empty result."""
        locs = _locs(n=10)
        idx, result = pp._process_single_rectangle_pick(locs, 2, "not-a-pick", 5.0, True)
        assert idx == 2
        assert len(result) == 0

    def test_process_single_circle_pick_direct(self):
        locs = _locs(n=200)
        idx, result = pp._process_single_circle_pick(locs, 100, 100, 3, (float(locs.xc[0]), float(locs.yc[0])), 5.0, True)
        assert idx == 3

    def test_process_single_circle_pick_error_returns_empty(self):
        locs = _locs(n=10)
        idx, result = pp._process_single_circle_pick(locs, 100, 100, 1, "not-a-pick", 5.0, True)
        assert idx == 1
        assert len(result) == 0

    def test_parallel_rectangle_individual_future_failure_caught(self, monkeypatch):
        """A pick that raises *outside* _process_single_rectangle_pick's own
        try/except (simulated by making the function itself raise) is caught
        by the outer `except Exception` around future.result(). Uses a
        callable callback (not console) so both the success-branch and
        failure-branch callable-callback lines get exercised too."""
        locs = _locs(n=500)
        picks = self._many_picks_rect(locs)
        calls = []

        def _boom(*args, **kwargs):
            raise RuntimeError("synthetic future failure")

        monkeypatch.setattr(pp, "_process_single_rectangle_pick", _boom)
        out = pp._parallel_picked_locs_rectangle(locs, 100, 100, picks, 2.0, callback=lambda i: calls.append(i))
        assert len(out) == 10
        assert all(len(o) == 0 for o in out)
        assert len(calls) == 10

    def test_parallel_rectangle_individual_future_failure_console(self, monkeypatch):
        """Same failure injection as above, but with callback='console' so
        the `if callback == "console" and progress: progress.update(1)`
        line inside the failure branch (distinct from the callable-callback
        line) gets exercised too."""
        locs = _locs(n=500)
        picks = self._many_picks_rect(locs)

        def _boom(*args, **kwargs):
            raise RuntimeError("synthetic future failure")

        monkeypatch.setattr(pp, "_process_single_rectangle_pick", _boom)
        out = pp._parallel_picked_locs_rectangle(locs, 100, 100, picks, 2.0, callback="console")
        assert len(out) == 10
        assert all(len(o) == 0 for o in out)

    def test_parallel_circle_individual_future_failure_caught(self, monkeypatch):
        locs = _locs(n=500)
        picks = self._many_picks_circ(locs)
        calls = []

        def _boom(*args, **kwargs):
            raise RuntimeError("synthetic future failure")

        monkeypatch.setattr(pp, "_process_single_circle_pick", _boom)
        out = pp._parallel_picked_locs_circle(locs, 100, 100, picks, 2.0, callback=lambda i: calls.append(i))
        assert len(out) == 10
        assert all(len(o) == 0 for o in out)
        assert len(calls) == 10

    def test_parallel_circle_individual_future_failure_console(self, monkeypatch):
        locs = _locs(n=500)
        picks = self._many_picks_circ(locs)

        def _boom(*args, **kwargs):
            raise RuntimeError("synthetic future failure")

        monkeypatch.setattr(pp, "_process_single_circle_pick", _boom)
        out = pp._parallel_picked_locs_circle(locs, 100, 100, picks, 2.0, callback="console")
        assert len(out) == 10
        assert all(len(o) == 0 for o in out)

    def test_parallel_rectangle_overall_failure_falls_back_to_serial(self, monkeypatch):
        """_parallel_picked_locs_rectangle does `from concurrent.futures
        import ThreadPoolExecutor` *locally*, so patching pp.ThreadPoolExecutor
        has no effect -- patch multiprocessing.cpu_count instead, which is
        called (via a local `import multiprocessing as mp`, but that binds
        to the same real module object) before the ThreadPoolExecutor block
        even starts, inside the same outer try."""
        import multiprocessing

        locs = _locs(n=500)
        picks = self._many_picks_rect(locs)

        def _boom():
            raise RuntimeError("synthetic cpu_count failure")

        monkeypatch.setattr(multiprocessing, "cpu_count", _boom)
        out = pp._parallel_picked_locs_rectangle(locs, 100, 100, picks, 2.0)
        assert len(out) == 10  # serial fallback still succeeds

    def test_parallel_circle_overall_failure_falls_back_to_serial(self, monkeypatch):
        import multiprocessing

        locs = _locs(n=500)
        picks = self._many_picks_circ(locs)

        def _boom():
            raise RuntimeError("synthetic cpu_count failure")

        monkeypatch.setattr(multiprocessing, "cpu_count", _boom)
        out = pp._parallel_picked_locs_circle(locs, 100, 100, picks, 2.0)
        assert len(out) == 10

    def test_parallel_rectangle_fills_none_safety_check(self, monkeypatch):
        """If as_completed() yields nothing (contrived here), every position
        stays None after the main loop -- exercises the
        `if picked_locs[i] is None: picked_locs[i] = empty` safety net."""
        import concurrent.futures

        locs = _locs(n=500)
        picks = self._many_picks_rect(locs)
        monkeypatch.setattr(concurrent.futures, "as_completed", lambda fs: iter(()))
        out = pp._parallel_picked_locs_rectangle(locs, 100, 100, picks, 2.0)
        assert len(out) == 10
        assert all(len(o) == 0 for o in out)

    def test_parallel_circle_fills_none_safety_check(self, monkeypatch):
        import concurrent.futures

        locs = _locs(n=500)
        picks = self._many_picks_circ(locs)
        monkeypatch.setattr(concurrent.futures, "as_completed", lambda fs: iter(()))
        out = pp._parallel_picked_locs_circle(locs, 100, 100, picks, 2.0)
        assert len(out) == 10
        assert all(len(o) == 0 for o in out)

    def test_serial_rectangle_console_callback_and_per_pick_error(self, monkeypatch):
        locs = _locs(n=200)
        picks = [((10.0, 10.0), (20.0, 20.0)), "not-a-pick"]
        out = pp._serial_picked_locs_rectangle(locs, 100, 100, picks, 5.0, callback="console")
        assert len(out) == 2
        assert len(out[1]) == 0  # malformed pick -> caught, empty result

    def test_serial_circle_console_callback_and_per_pick_error(self):
        locs = _locs(n=200)
        picks = [(float(locs.xc[0]), float(locs.yc[0])), "not-a-pick"]
        out = pp._serial_picked_locs_circle(locs, 100, 100, picks, 5.0, callback="console")
        assert len(out) == 2
        assert len(out[1]) == 0

    def test_serial_rectangle_callable_callback_success_and_failure(self):
        """Callable callback on both the success path and the per-pick
        exception path (a malformed second pick)."""
        locs = _locs(n=50)
        calls = []
        out = pp._serial_picked_locs_rectangle(
            locs, 100, 100, [((10.0, 10.0), (20.0, 20.0)), "not-a-pick"], 5.0,
            callback=lambda i: calls.append(i),
        )
        assert calls == [1, 2]
        assert len(out[1]) == 0

    def test_serial_circle_callable_callback_success_and_failure(self):
        locs = _locs(n=50)
        calls = []
        out = pp._serial_picked_locs_circle(
            locs, 100, 100, [(float(locs.xc[0]), float(locs.yc[0])), "not-a-pick"], 5.0,
            callback=lambda i: calls.append(i),
        )
        assert calls == [1, 2]
        assert len(out[1]) == 0

    def test_serial_rectangle_lib_import_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pyS3M.lib", None)
        locs = _locs(n=10)
        with pytest.raises(ImportError, match="lib module required for rectangle picking"):
            pp._serial_picked_locs_rectangle(locs, 100, 100, [((0.0, 0.0), (1.0, 1.0))], 1.0)

    def test_serial_circle_lib_import_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pyS3M.lib", None)
        locs = _locs(n=10)
        with pytest.raises(ImportError, match="lib module required for circle picking"):
            pp._serial_picked_locs_circle(locs, 100, 100, [(0.0, 0.0)], 1.0)


# ======================================================================
# segment_locs_by_rendered_image
# ======================================================================

def _cluster_df(n=300, cx=20.0, cy=20.0, seed=0, with_errs=True):
    rng = np.random.default_rng(seed)
    data = {
        "xc": rng.normal(cx, 1.0, n),
        "yc": rng.normal(cy, 1.0, n),
        "frame": rng.integers(0, 100, n),
    }
    if with_errs:
        data["photoelectrons"] = rng.uniform(500, 2000, n)
        data["xc_err"] = np.full(n, 0.02)
        data["yc_err"] = np.full(n, 0.02)
    return pd.DataFrame(data)


class TestSegmentLocsByRenderedImage:
    def test_missing_xy_columns_raises(self):
        df = _cluster_df().drop(columns=["xc"])
        with pytest.raises(ValueError, match="'xc' and 'yc' columns"):
            pp.segment_locs_by_rendered_image(df, width=40, height=40)

    def test_invalid_threshold_method_raises(self):
        df = _cluster_df()
        with pytest.raises(ValueError, match="Unknown threshold_method"):
            pp.segment_locs_by_rendered_image(df, width=40, height=40, threshold_method="bogus")

    def test_success_otsu_with_weighted_and_summed_stats(self):
        df = _cluster_df()
        agg_locs, stats = pp.segment_locs_by_rendered_image(
            df, width=40, height=40, oversampling=4, pixel_size_nm=100.0,
            min_area_nm2=10.0, min_localisations=10, threshold_method="otsu",
        )
        assert len(stats) == 1
        assert len(agg_locs) > 0
        assert "photoelectrons" in stats.columns  # summed
        assert "xc_err" in stats.columns   # weighted-mean error propagation
        assert "frame" in stats.columns

    def test_success_li_method(self):
        df = _cluster_df(with_errs=False)
        agg_locs, stats = pp.segment_locs_by_rendered_image(
            df, width=40, height=40, oversampling=4, min_area_nm2=10.0,
            min_localisations=10, threshold_method="li",
        )
        assert len(stats) == 1

    def test_success_percentile_method(self):
        df = _cluster_df(with_errs=False)
        agg_locs, stats = pp.segment_locs_by_rendered_image(
            df, width=40, height=40, oversampling=4, min_area_nm2=10.0,
            min_localisations=10, threshold_method="percentile",
        )
        assert len(stats) == 1

    def test_recarray_input(self):
        df = _cluster_df(with_errs=False)
        rec = df.to_records(index=False)
        agg_locs, stats = pp.segment_locs_by_rendered_image(
            rec, width=40, height=40, oversampling=4, min_area_nm2=10.0, min_localisations=10,
        )
        assert len(stats) == 1

    def test_no_valid_regions_returns_empty(self):
        df = _cluster_df(with_errs=False)
        agg_locs, stats = pp.segment_locs_by_rendered_image(
            df, width=40, height=40, oversampling=4, min_localisations=10**6,
        )
        assert len(agg_locs) == 0
        assert len(stats) == 0
        assert "aggregate_id" in agg_locs.columns

    def test_verbose_plots_with_a_rejected_region(self):
        """Two clusters, both crossing the Otsu binary threshold as
        separate detected regions, but the second is too sparse to pass
        min_localisations -> exercises the "rejected region" display branch
        inside the verbose plotting block, not just the "valid region" one
        (confirmed both regions are actually detected via a direct
        render+threshold+label reproduction before relying on it here)."""
        rng = np.random.default_rng(0)
        big = pd.DataFrame({
            "xc": rng.normal(10.0, 1.0, 300), "yc": rng.normal(10.0, 1.0, 300),
            "frame": rng.integers(0, 100, 300),
        })
        small = pd.DataFrame({
            "xc": rng.normal(35.0, 0.3, 20), "yc": rng.normal(35.0, 0.3, 20),
            "frame": rng.integers(0, 100, 20),
        })
        df = pd.concat([big, small], ignore_index=True)
        agg_locs, stats = pp.segment_locs_by_rendered_image(
            df, width=40, height=40, oversampling=4, min_area_nm2=10.0,
            min_localisations=50, verbose=True,
        )
        assert len(stats) == 1  # only the big cluster passes min_localisations

    def test_skimage_import_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "skimage", None)
        monkeypatch.setitem(sys.modules, "skimage.filters", None)
        monkeypatch.setitem(sys.modules, "skimage.measure", None)
        df = _cluster_df(with_errs=False)
        with pytest.raises(ImportError, match="Required module not found"):
            pp.segment_locs_by_rendered_image(df, width=40, height=40)

    def test_verbose_plot_exception_is_caught(self, monkeypatch):
        df = _cluster_df(with_errs=False)
        import pyS3M.PlottingBase as pb
        monkeypatch.setattr(pb, "AnalysisPlotter", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        agg_locs, stats = pp.segment_locs_by_rendered_image(
            df, width=40, height=40, oversampling=4, min_area_nm2=10.0,
            min_localisations=10, verbose=True,
        )
        assert len(stats) == 1  # verbose failure doesn't break the main result

    def test_console_callback_and_config_callbacks(self):
        df = _cluster_df(with_errs=False)
        calls = []
        cfg = AnalysisConfig(
            progress_callback=lambda f, m: calls.append(("p", f)),
            logging_callback=lambda m: calls.append(("l", m)),
        )
        agg_locs, stats = pp.segment_locs_by_rendered_image(
            df, width=40, height=40, oversampling=4, min_area_nm2=10.0,
            min_localisations=10, callback="console", config=cfg,
        )
        assert len(calls) > 0
        assert any(c[0] == "p" for c in calls)
        assert any(c[0] == "l" for c in calls)

    def test_callable_callback(self):
        df = _cluster_df(with_errs=False)
        calls = []
        pp.segment_locs_by_rendered_image(
            df, width=40, height=40, oversampling=4, min_area_nm2=10.0,
            min_localisations=10, callback=lambda i: calls.append(i),
        )
        assert len(calls) > 0

    def test_no_regions_at_all_diagnostic_branch(self):
        """A single, unblurred point: the 95th percentile of a one-valued
        nonzero-pixel array equals that same value, so `rendered_image >
        threshold` (strict) is False everywhere -> len(regions) == 0,
        distinct from the "regions found but none valid" branch above."""
        df = pd.DataFrame({"xc": [20.0], "yc": [20.0], "frame": [0]})
        agg_locs, stats = pp.segment_locs_by_rendered_image(
            df, width=40, height=40, oversampling=1, threshold_method="percentile",
            blur_method=None, min_area_nm2=1.0, min_localisations=1,
        )
        assert len(stats) == 0


# ======================================================================
# remove_fiducials
# ======================================================================

def _agg_and_stats():
    agg_locs = pd.DataFrame({
        "aggregate_id": [0, 0, 1, 1, 2, 2],
        "xc": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        "yc": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
    })
    stats = pd.DataFrame({
        "aggregate_id": [0, 1, 2],
        "n_localisations": [1000, 50, 20],
        "A_R": [0.6, 0.2, 0.9],
        "A_G": [0.1, 0.1, 0.1],
    })
    return agg_locs, stats


class TestRemoveFiducials:
    def test_density_only(self):
        agg_locs, stats = _agg_and_stats()
        filt_locs, filt_stats, mask = pp.remove_fiducials(agg_locs, stats, n_frames=1000, density_threshold=0.5)
        np.testing.assert_array_equal(mask, [True, False, False])
        assert len(filt_stats) == 2

    def test_a_r_float_threshold_defaults_above(self):
        agg_locs, stats = _agg_and_stats()
        _, _, mask = pp.remove_fiducials(agg_locs, stats, n_frames=1000, A_R_threshold=0.5, density_threshold=None)
        np.testing.assert_array_equal(mask, [True, False, True])

    def test_a_r_tuple_below(self):
        agg_locs, stats = _agg_and_stats()
        _, _, mask = pp.remove_fiducials(agg_locs, stats, n_frames=1000, A_R_threshold=(0.5, "below"), density_threshold=None)
        np.testing.assert_array_equal(mask, [False, True, False])

    def test_a_g_threshold(self):
        agg_locs, stats = _agg_and_stats()
        _, _, mask = pp.remove_fiducials(agg_locs, stats, n_frames=1000, A_G_threshold=0.05, density_threshold=None)
        np.testing.assert_array_equal(mask, [True, True, True])

    def test_require_all_true(self):
        agg_locs, stats = _agg_and_stats()
        _, _, mask = pp.remove_fiducials(agg_locs, stats, n_frames=1000, A_R_threshold=0.5, density_threshold=0.5, require_all=True)
        np.testing.assert_array_equal(mask, [True, False, False])

    def test_bad_tuple_length_raises(self):
        agg_locs, stats = _agg_and_stats()
        with pytest.raises(ValueError, match="got length 3"):
            pp.remove_fiducials(agg_locs, stats, n_frames=1000, A_R_threshold=(0.5, "below", "extra"), density_threshold=None)

    def test_bad_direction_raises(self):
        agg_locs, stats = _agg_and_stats()
        with pytest.raises(ValueError, match="'above' or 'below'"):
            pp.remove_fiducials(agg_locs, stats, n_frames=1000, A_R_threshold=(0.5, "sideways"), density_threshold=None)

    def test_missing_spectral_column_raises(self):
        agg_locs, stats = _agg_and_stats()
        with pytest.raises(ValueError, match="'A_R' column not found"):
            pp.remove_fiducials(agg_locs, stats.drop(columns=["A_R"]), n_frames=1000, A_R_threshold=0.5, density_threshold=None)

    def test_no_active_criteria_raises(self):
        agg_locs, stats = _agg_and_stats()
        with pytest.raises(ValueError, match="At least one criterion"):
            pp.remove_fiducials(agg_locs, stats, n_frames=1000, density_threshold=None)

    def test_missing_id_column_raises(self):
        agg_locs, stats = _agg_and_stats()
        with pytest.raises(ValueError, match="'aggregate_id' or 'cluster_id'"):
            pp.remove_fiducials(agg_locs, stats.rename(columns={"aggregate_id": "foo"}), n_frames=1000, density_threshold=0.5)

    def test_missing_nloc_column_raises(self):
        agg_locs, stats = _agg_and_stats()
        with pytest.raises(ValueError, match="'n_localisations' column"):
            pp.remove_fiducials(agg_locs, stats.rename(columns={"n_localisations": "foo"}), n_frames=1000, density_threshold=0.5)

    def test_alternate_column_names(self):
        agg_locs, stats = _agg_and_stats()
        stats_alt = stats.rename(columns={"aggregate_id": "cluster_id", "n_localisations": "n_locs"})
        agg_locs_alt = agg_locs.rename(columns={"aggregate_id": "cluster_id"})
        _, _, mask = pp.remove_fiducials(agg_locs_alt, stats_alt, n_frames=1000, density_threshold=0.5)
        np.testing.assert_array_equal(mask, [True, False, False])

    def test_id_col_mismatch_falls_back_to_aggregate_id_in_locs(self):
        """id_col resolves to 'cluster_id' via stats, but aggregate_locs
        only has 'aggregate_id' -> the `elif 'aggregate_id' in ...` branch."""
        agg_locs = pd.DataFrame({"aggregate_id": [0, 0, 1], "xc": [1.0, 1.0, 2.0], "yc": [1.0, 1.0, 2.0]})
        stats = pd.DataFrame({"cluster_id": [0, 1], "n_localisations": [1000, 20]})
        _, filt_locs, mask = pp.remove_fiducials(agg_locs, stats, n_frames=1000, density_threshold=0.5)
        np.testing.assert_array_equal(mask, [True, False])

    def test_id_col_mismatch_falls_back_to_cluster_id_in_locs(self):
        """The reverse: id_col resolves to 'aggregate_id' via stats, but
        aggregate_locs only has 'cluster_id' -> the final
        `elif 'cluster_id' in ...` branch."""
        agg_locs = pd.DataFrame({"cluster_id": [0, 0, 1], "xc": [1.0, 1.0, 2.0], "yc": [1.0, 1.0, 2.0]})
        stats = pd.DataFrame({"aggregate_id": [0, 1], "n_localisations": [1000, 20]})
        _, _, mask = pp.remove_fiducials(agg_locs, stats, n_frames=1000, density_threshold=0.5)
        np.testing.assert_array_equal(mask, [True, False])

    def test_aggregate_locs_missing_id_column_raises(self):
        agg_locs = pd.DataFrame({"foo": [0, 0, 1], "xc": [1.0, 1.0, 2.0], "yc": [1.0, 1.0, 2.0]})
        stats = pd.DataFrame({"aggregate_id": [0, 1], "n_localisations": [1000, 20]})
        with pytest.raises(ValueError, match="aggregate_locs must contain"):
            pp.remove_fiducials(agg_locs, stats, n_frames=1000, density_threshold=0.5)

    def test_verbose_and_config_logging_callback(self):
        agg_locs, stats = _agg_and_stats()
        calls = []
        cfg = AnalysisConfig(logging_callback=lambda m: calls.append(m))
        pp.remove_fiducials(agg_locs, stats, n_frames=1000, density_threshold=0.5, verbose=True, config=cfg)
        assert len(calls) >= 1
