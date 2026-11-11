"""Full coverage tests for pyS3M.clustering.dbscan_clusterer -- DBSCAN-based
single-molecule extraction mixin. The real happy path is already exercised via
unit_tests/test_analysis_pipeline.py's DBSCAN-mode parametrisation; this file
closes the gaps that path doesn't reach: the too-few-localisations
early-return, and the zero-precision guard against a degenerate error column.
"""
import numpy as np
import pandas as pd
import pytest

import pyS3M.SM_extractionfunctions as SM_extractionfunctions


def _tiny_loc_df(n=2, zero_errors=False):
    rng = np.random.default_rng(0)
    photons = rng.uniform(500, 5000, n)
    rgb = rng.dirichlet([2, 3, 1], n)
    err = 0.0 if zero_errors else 0.02
    return pd.DataFrame({
        "frame": np.arange(n),
        "xc": rng.uniform(0, 40, n),
        "yc": rng.uniform(0, 40, n),
        "xc_err": np.full(n, err),
        "yc_err": np.full(n, err),
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


class TestExtractSingleMoleculesDbscanTooFewLocs:
    def test_returns_empty_dataframes(self):
        sm = SM_extractionfunctions.extract_SMs()
        df = _tiny_loc_df(n=2)
        sm_db, sf_db = sm.extract_single_molecules_DBSCAN(
            df, min_cluster_size=10, min_photoelectrons=0,
        )
        assert sm_db.empty
        assert sf_db.empty


class TestExtractSingleMoleculesDbscanZeroPrecision:
    def test_zero_error_columns_raise(self):
        sm = SM_extractionfunctions.extract_SMs()
        df = _tiny_loc_df(n=15, zero_errors=True)
        with pytest.raises(ValueError, match="loc_precision"):
            sm.extract_single_molecules_DBSCAN(
                df, min_cluster_size=3, min_photoelectrons=0,
            )
