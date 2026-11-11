"""Full coverage tests for pyS3M.clustering.hdbscan_clusterer -- HDBSCAN-based
single-molecule extraction mixin. The real happy path is already exercised via
unit_tests/test_analysis_pipeline.py's HDBSCAN-mode parametrisation; this file
closes the two gaps that path doesn't reach: the too-few-localisations
early-return, and the fast_hdbscan-not-installed import fallback.
"""
import sys

import numpy as np
import pandas as pd
import pytest

import pyS3M.SM_extractionfunctions as SM_extractionfunctions
import pyS3M.clustering.hdbscan_clusterer as hdbscan_clusterer


def _tiny_loc_df(n=2):
    rng = np.random.default_rng(0)
    photons = rng.uniform(500, 5000, n)
    rgb = rng.dirichlet([2, 3, 1], n)
    return pd.DataFrame({
        "frame": np.arange(n),
        "xc": rng.uniform(0, 40, n),
        "yc": rng.uniform(0, 40, n),
        "xc_err": np.full(n, 0.02),
        "yc_err": np.full(n, 0.02),
        "A_R": rgb[:, 0] * photons,
        "A_G": rgb[:, 1] * photons,
        "A_B": rgb[:, 2] * photons,
        "A_R_err": np.full(n, 0.05),
        "A_G_err": np.full(n, 0.05),
        "A_B_err": np.full(n, 0.05),
        "bg_R": rng.uniform(5, 20, n),
        "bg_G": rng.uniform(5, 20, n),
        "bg_B": rng.uniform(5, 20, n),
        "bg_R_err": np.full(n, 0.05),
        "bg_G_err": np.full(n, 0.05),
        "bg_B_err": np.full(n, 0.05),
        "photoelectrons": photons,
        "s_x": rng.uniform(1.2, 1.5, n),
        "s_y": rng.uniform(1.2, 1.5, n),
        "s_x_err": np.full(n, 0.05),
        "s_y_err": np.full(n, 0.05),
        "chi_sqr": rng.gamma(5, 0.2, n),
    })


class TestExtractSingleMoleculesHdbscanTooFewLocs:
    def test_returns_empty_dataframes(self):
        sm = SM_extractionfunctions.extract_SMs()
        df = _tiny_loc_df(n=2)
        sm_db, sf_db = sm.extract_single_molecules_HDBSCAN(
            df, min_cluster_size=10, min_photoelectrons=0,
        )
        assert sm_db.empty
        assert sf_db.empty


class TestGetHdbscanCls:
    """_get_hdbscan_cls is a lazy, memoised loader (see its docstring in
    hdbscan_clusterer.py for why: fast_hdbscan's own __init__.py
    unconditionally JIT-compiles its entire numba codebase on import, ~14s
    with no on-disk cache -- deferring the import out of this module's own
    top level keeps that cost off every AnalysisPipeline import)."""

    def test_prefers_fast_hdbscan_when_available(self, monkeypatch):
        monkeypatch.setattr(hdbscan_clusterer, "_HDBSCAN_cls", None)
        cls = hdbscan_clusterer._get_hdbscan_cls()
        assert hdbscan_clusterer.HDBSCAN_BACKEND == "fast_hdbscan"
        from fast_hdbscan import HDBSCAN as FastHDBSCAN
        assert cls is FastHDBSCAN

    def test_import_error_falls_back_to_sklearn(self, monkeypatch):
        # fast_hdbscan is installed in this environment, so the fallback
        # branch only triggers if its import genuinely fails -- block it in
        # sys.modules and force re-resolution by clearing the memoised class.
        monkeypatch.setattr(hdbscan_clusterer, "_HDBSCAN_cls", None)
        monkeypatch.setitem(sys.modules, "fast_hdbscan", None)
        cls = hdbscan_clusterer._get_hdbscan_cls()
        assert hdbscan_clusterer.HDBSCAN_BACKEND == "sklearn"
        from sklearn.cluster import HDBSCAN as SklearnHDBSCAN
        assert cls is SklearnHDBSCAN

    def test_result_is_memoised(self, monkeypatch):
        monkeypatch.setattr(hdbscan_clusterer, "_HDBSCAN_cls", None)
        first = hdbscan_clusterer._get_hdbscan_cls()
        second = hdbscan_clusterer._get_hdbscan_cls()
        assert first is second
