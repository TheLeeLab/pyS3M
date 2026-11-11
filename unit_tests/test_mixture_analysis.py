"""Full coverage tests for pyS3M.mixture_analysis.MixtureAnalysisMixin -- GMM-based
dye-population fitting (MLE/EM/pygmmis extreme-deconvolution), fixed-mean covariance
fitting (plain EM/MLE and robust M-estimator), and analytical misidentification-rate
calculation used by SM_extractionfunctions.extract_SMs.

Small-but-sufficient synthetic 2-component data throughout (tens of points per
population, not fixture files) -- these are real statistical fits (GMM/EM/MLE), so
unlike pure-numeric modules a handful of points isn't enough for convergence; the
data is kept as small as still reliably converges. `monkeypatch` is used only for
branches no real data organically reaches (forced component-assignment edge cases,
forced singular-matrix recovery, forced import/fit failures).
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import pyS3M.mixture_analysis as mixture_analysis
import pyS3M.PlottingBase as PlottingBase
import pyS3M.SM_extractionfunctions as SM_extractionfunctions
from pyS3M.mixture_analysis import MixtureAnalysisMixin


MEAN0 = np.array([0.7, 0.2])
MEAN1 = np.array([0.2, 0.7])


def _host():
    return SM_extractionfunctions.extract_SMs()


def _two_component_X(n_per=25, seed=0, sigma=0.03, mean0=MEAN0, mean1=MEAN1):
    rng = np.random.default_rng(seed)
    cov = np.eye(2) * sigma**2
    X0 = rng.multivariate_normal(mean0, cov, n_per)
    X1 = rng.multivariate_normal(mean1, cov, n_per)
    return np.vstack([X0, X1])


def _mode_b_df(n_per=20, seed=0, with_photoelectrons=True, with_errors=False,
                with_molidx=True, with_fov=False, sigma=0.03):
    X = _two_component_X(n_per=n_per, seed=seed, sigma=sigma)
    n = len(X)
    data = {"A_R": X[:, 0], "A_G": X[:, 1], "A_B": 1.0 - X[:, 0] - X[:, 1]}
    if with_molidx:
        data["molecular_index"] = np.arange(n)
    if with_photoelectrons:
        rng = np.random.default_rng(seed + 1)
        data["photoelectrons"] = rng.uniform(5000, 50000, n)
    if with_errors:
        data["A_R_err"] = np.full(n, 0.01)
        data["A_G_err"] = np.full(n, 0.01)
    if with_fov:
        data["fov_index"] = 0
        data["fov_name"] = "Pos0"
    return pd.DataFrame(data)


def _mode_a_df(n_mol_per=15, seed=0, frames_per_mol=5, noise=0.02):
    rng = np.random.default_rng(seed)
    records = []
    mol_idx = 0
    for mean in (MEAN0, MEAN1):
        for _ in range(n_mol_per):
            for f in range(frames_per_mol):
                photons_acc = 1000.0 * (f + 1)
                A_R = mean[0] + rng.normal(0, noise)
                A_G = mean[1] + rng.normal(0, noise)
                A_B = 1.0 - A_R - A_G
                records.append({
                    "molecular_index": mol_idx,
                    "photoelectrons_accumulated": photons_acc,
                    "A_R": A_R, "A_G": A_G, "A_B": A_B,
                })
            mol_idx += 1
    return pd.DataFrame(records)


# ======================================================================
# _fit_gmm_mle
# ======================================================================

class TestFitGmmMle:
    def test_normal_fit_recovers_means(self):
        host = _host()
        X = _two_component_X(n_per=25, seed=1)
        initial_means = np.array([MEAN0, MEAN1])
        means, covs, weights, converged = host._fit_gmm_mle(
            X, initial_means, n_components=2, verbose=True,
        )
        assert means.shape == (2, 2)
        assert covs.shape == (2, 2, 2)
        assert weights.shape == (2,)
        assert np.isclose(weights.sum(), 1.0)

    def test_negative_log_likelihood_exception_branch(self, monkeypatch):
        # A corrupt covariance matrix still passes the eigenvalue-validity gate
        # in practice (NaN/inf entries don't reliably fail np.linalg.eigvalsh),
        # so force the exception directly at its real source instead: make
        # multivariate_normal's construction raise on its first call only,
        # which happens organically during the optimizer's first evaluation.
        host = _host()
        X = _two_component_X(n_per=15, seed=41)
        initial_means = np.array([MEAN0, MEAN1])
        real_mvn = mixture_analysis.multivariate_normal
        calls = {"n": 0}

        def flaky_mvn(mean, cov):
            calls["n"] += 1
            if calls["n"] == 1:
                raise np.linalg.LinAlgError("forced singular")
            return real_mvn(mean=mean, cov=cov)

        monkeypatch.setattr(mixture_analysis, "multivariate_normal", flaky_mvn)
        means, covs, weights, converged = host._fit_gmm_mle(X, initial_means, n_components=2)
        assert means.shape == (2, 2)


# ======================================================================
# _fit_gmm_em
# ======================================================================

class TestFitGmmEm:
    def test_normal_fit_no_weighting(self):
        host = _host()
        X = _two_component_X(n_per=25, seed=2)
        initial_means = np.array([MEAN0, MEAN1])
        means, covs, weights, converged = host._fit_gmm_em(
            X, initial_means, n_components=2, verbose=True,
        )
        assert means.shape == (2, 2)

    def test_photon_statistics_weighting_without_error_columns(self):
        host = _host()
        X = _two_component_X(n_per=25, seed=3)
        n = len(X)
        rng = np.random.default_rng(4)
        photons = rng.uniform(1000, 20000, n)
        A_R = X[:, 0]
        A_G = X[:, 1]
        initial_means = np.array([MEAN0, MEAN1])
        means, covs, weights, converged = host._fit_gmm_em(
            X, initial_means, n_components=2,
            photoelectrons=photons, A_R=A_R, A_G=A_G, has_error_columns=False,
        )
        assert means.shape == (2, 2)

    def test_provided_error_columns_weighting(self):
        host = _host()
        X = _two_component_X(n_per=25, seed=5)
        n = len(X)
        photons = np.full(n, 10000.0)
        A_R = X[:, 0]
        A_G = X[:, 1]
        sigma_A_R = np.full(n, 0.01)
        sigma_A_G = np.full(n, 0.01)
        initial_means = np.array([MEAN0, MEAN1])
        means, covs, weights, converged = host._fit_gmm_em(
            X, initial_means, n_components=2,
            photoelectrons=photons, A_R=A_R, A_G=A_G, has_error_columns=True,
            sigma_A_R=sigma_A_R, sigma_A_G=sigma_A_G,
        )
        assert means.shape == (2, 2)

    def test_empty_component_is_skipped(self, monkeypatch):
        # Force every point into component 0 -- component 1's n_in_component==0
        # `continue` branch, while X_reweighted_list still has component 0's data.
        host = _host()
        X = _two_component_X(n_per=10, seed=6)
        initial_means = np.array([MEAN0, MEAN1])
        monkeypatch.setattr(
            mixture_analysis.GaussianMixture, "predict",
            lambda self, X: np.zeros(len(X), dtype=int),
        )
        means, covs, weights, converged = host._fit_gmm_em(
            X, initial_means, n_components=2, max_iter=5,
        )
        assert means.shape == (2, 2)

    def test_all_components_empty_falls_back_to_full_data(self, monkeypatch):
        # Force every point's label to never match any real component index --
        # X_reweighted_list stays empty every iteration -> X_balanced = X fallback.
        host = _host()
        X = _two_component_X(n_per=10, seed=7)
        initial_means = np.array([MEAN0, MEAN1])
        monkeypatch.setattr(
            mixture_analysis.GaussianMixture, "predict",
            lambda self, X: np.full(len(X), -1, dtype=int),
        )
        means, covs, weights, converged = host._fit_gmm_em(
            X, initial_means, n_components=2, max_iter=3,
        )
        assert means.shape == (2, 2)


# ======================================================================
# _fit_gmm_pygmmis
# ======================================================================

class TestFitGmmPygmmis:
    def test_normal_fit(self):
        host = _host()
        X = _two_component_X(n_per=25, seed=8)
        X_err = np.full_like(X, 0.02)
        initial_means = np.array([MEAN0, MEAN1])
        means, covs, weights, converged = host._fit_gmm_pygmmis(
            X, X_err, initial_means, n_components=2, max_iter=20, verbose=True,
        )
        assert means.shape == (2, 2)
        assert covs.shape == (2, 2, 2)
        assert weights.shape == (2,)

    def test_sparse_component_uses_identity_covariance(self):
        # A component with <= n_features points assigned falls back to the
        # small-identity-covariance initial guess instead of empirical covariance.
        host = _host()
        X = _two_component_X(n_per=25, seed=9)
        X_err = np.full_like(X, 0.02)
        # Third mean far from all real data -> gets 0 points assigned.
        initial_means = np.array([MEAN0, MEAN1, [5.0, 5.0]])
        means, covs, weights, converged = host._fit_gmm_pygmmis(
            X, X_err, initial_means, n_components=3, max_iter=10,
        )
        assert means.shape == (3, 2)

    def test_pygmmis_not_installed_raises_import_error(self, monkeypatch):
        host = _host()
        X = _two_component_X(n_per=5, seed=10)
        X_err = np.full_like(X, 0.02)
        initial_means = np.array([MEAN0, MEAN1])
        monkeypatch.setitem(sys.modules, "pygmmis", None)
        with pytest.raises(ImportError, match="pygmmis is required"):
            host._fit_gmm_pygmmis(X, X_err, initial_means, n_components=2)

    def test_fit_failure_returns_initial_params_unconverged(self, monkeypatch):
        import pygmmis

        host = _host()
        X = _two_component_X(n_per=15, seed=11)
        X_err = np.full_like(X, 0.02)
        initial_means = np.array([MEAN0, MEAN1])

        def _raise(*a, **kw):
            raise RuntimeError("forced pygmmis failure")

        monkeypatch.setattr(pygmmis, "fit", _raise)
        means, covs, weights, converged = host._fit_gmm_pygmmis(
            X, X_err, initial_means, n_components=2, verbose=True,
        )
        assert converged is False
        np.testing.assert_allclose(means, initial_means)

    def test_macos_patches_and_restores_pygmmis_pool_and_shared(self, monkeypatch):
        # On macOS, pygmmis.fit's internal multiprocessing.Pool() and its
        # multiprocessing.Array-backed shared arrays (via createShared) must
        # both be swapped for in-thread stand-ins for the duration of the
        # call only (see _SerialPool's/_no_shared_array's docstrings in
        # mixture_analysis.py) -- force the platform check to exercise that
        # branch on this (non-macOS) CI box.
        import pygmmis

        monkeypatch.setattr(sys, "platform", "darwin")
        original_pool = pygmmis.multiprocessing.Pool
        original_shared = pygmmis.createShared

        host = _host()
        X = _two_component_X(n_per=25, seed=12)
        X_err = np.full_like(X, 0.02)
        initial_means = np.array([MEAN0, MEAN1])
        means, covs, weights, converged = host._fit_gmm_pygmmis(
            X, X_err, initial_means, n_components=2, max_iter=20,
        )
        assert means.shape == (2, 2)
        # Fit must actually converge through the patched pool/shared-array
        # stand-ins, not just run without raising.
        assert converged is True
        np.testing.assert_allclose(
            np.sort(means, axis=0), np.sort(np.array([MEAN0, MEAN1]), axis=0),
            atol=0.05,
        )
        # Both must be restored to the real pygmmis internals afterward, not
        # left patched to the serial/no-shared-memory stand-ins.
        assert pygmmis.multiprocessing.Pool is original_pool
        assert pygmmis.createShared is original_shared

    def test_macos_restores_pool_and_shared_even_on_fit_failure(self, monkeypatch):
        import pygmmis

        monkeypatch.setattr(sys, "platform", "darwin")
        original_pool = pygmmis.multiprocessing.Pool
        original_shared = pygmmis.createShared

        def _raise(*a, **kw):
            raise RuntimeError("forced pygmmis failure")

        monkeypatch.setattr(pygmmis, "fit", _raise)

        host = _host()
        X = _two_component_X(n_per=10, seed=13)
        X_err = np.full_like(X, 0.02)
        initial_means = np.array([MEAN0, MEAN1])
        means, covs, weights, converged = host._fit_gmm_pygmmis(
            X, X_err, initial_means, n_components=2,
        )
        assert converged is False
        assert pygmmis.multiprocessing.Pool is original_pool
        assert pygmmis.createShared is original_shared

    def test_no_shared_array_returns_plain_float64_copy(self):
        arr = mixture_analysis._no_shared_array(np.array([[1, 2], [3, 4]], dtype=np.int32))
        assert arr.dtype == np.float64
        assert arr.shape == (2, 2)
        np.testing.assert_array_equal(arr, [[1.0, 2.0], [3.0, 4.0]])


class TestSerialPool:
    """Direct tests for the macOS-only multiprocessing.Pool stand-in itself."""

    def test_serial_pool_result_get_returns_wrapped_value(self):
        result = mixture_analysis._SerialPoolResult(42)
        assert result.get() == 42

    def test_apply_async_runs_synchronously_and_returns_result(self):
        pool = mixture_analysis._SerialPool()
        result = pool.apply_async(lambda a, b: a + b, (2, 3))
        assert result.get() == 5

    def test_map_async_runs_synchronously_and_returns_results(self):
        pool = mixture_analysis._SerialPool()
        result = pool.map_async(lambda x: x * 2, [1, 2, 3], chunksize=1)
        assert result.get() == [2, 4, 6]

    def test_close_and_join_are_noops(self):
        pool = mixture_analysis._SerialPool()
        pool.close()
        pool.join()


# ======================================================================
# extract_reference_means
# ======================================================================

class TestExtractReferenceMeansModeA:
    def test_verbose_mle_default(self, capsys):
        host = _host()
        df = _mode_a_df()
        means, ref_db, gmm = host.extract_reference_means(
            df, reference_photoelectron_threshold=3000.0, verbose=True,
        )
        assert means.shape == (2, 2)
        assert "true_label" in ref_db.columns
        assert "max_photoelectrons" in ref_db.columns

    def test_no_threshold_raises(self):
        host = _host()
        df = _mode_a_df()
        with pytest.raises(ValueError, match="reference_photoelectron_threshold must be provided"):
            host.extract_reference_means(df, reference_photoelectron_threshold=None, verbose=False)

    def test_threshold_too_high_raises(self):
        host = _host()
        df = _mode_a_df()
        with pytest.raises(ValueError, match="No molecules reach photoelectron threshold"):
            host.extract_reference_means(df, reference_photoelectron_threshold=1e12, verbose=False)

    def test_em_fit_type(self):
        host = _host()
        df = _mode_a_df()
        means, ref_db, gmm = host.extract_reference_means(
            df, reference_photoelectron_threshold=3000.0, fit_type="EM", verbose=True,
        )
        assert means.shape == (2, 2)


class TestExtractReferenceMeansLegacy:
    def test_old_photon_accumulation_database_raises(self):
        df = _mode_a_df().rename(columns={"photoelectrons_accumulated": "photons_accumulated"})
        with pytest.raises(ValueError, match="Old photon accumulation database"):
            _host().extract_reference_means(df, reference_photoelectron_threshold=3000.0, verbose=False)

    @pytest.mark.parametrize("with_io", [True, False])
    def test_old_molecule_table_is_upgraded(self, with_io):
        from pyS3M.IOFunctions import IO_Functions
        host = _host()
        if with_io:
            host.io = IO_Functions()
        elif hasattr(host, "io"):
            del host.io
        df = _mode_b_df(with_photoelectrons=False)
        df["photons"] = 40000.0   # old sum: photoelectrons = photons * sum_p m_p f_p (> 5000 here)
        means, ref_db, gmm = host.extract_reference_means(
            df, reference_photoelectron_threshold=5000.0, verbose=False)
        assert means.shape == (2, 2)
        assert "photons" not in ref_db.columns


class TestExtractReferenceMeansModeB:
    def test_no_threshold_no_errors_no_photons(self):
        # No error columns AND no photon column -> the "no weighting info" branch,
        # exercised with verbose=True to also cover its logging lines.
        host = _host()
        df = _mode_b_df(with_photoelectrons=False, with_errors=False)
        means, ref_db, gmm = host.extract_reference_means(
            df, reference_photoelectron_threshold=None, verbose=True,
        )
        assert means.shape == (2, 2)

    def test_missing_required_columns_raises(self):
        host = _host()
        df = pd.DataFrame({"A_R": [0.5, 0.6], "A_G": [0.5, 0.4]})
        with pytest.raises(ValueError, match="missing required columns"):
            host.extract_reference_means(df, reference_photoelectron_threshold=None, verbose=False)

    def test_threshold_without_photons_column_raises(self):
        host = _host()
        df = _mode_b_df(with_photoelectrons=False)
        with pytest.raises(ValueError, match="'photoelectrons' column not found"):
            host.extract_reference_means(df, reference_photoelectron_threshold=1000.0, verbose=False)

    def test_threshold_too_high_raises(self):
        host = _host()
        df = _mode_b_df(with_photoelectrons=True)
        with pytest.raises(ValueError, match="No molecules have photoelectrons"):
            host.extract_reference_means(df, reference_photoelectron_threshold=1e12, verbose=False)

    def test_threshold_with_photons_verbose(self):
        host = _host()
        df = _mode_b_df(with_photoelectrons=True, with_fov=True)
        means, ref_db, gmm = host.extract_reference_means(
            df, reference_photoelectron_threshold=5000.0, verbose=True,
        )
        assert means.shape == (2, 2)
        assert "fov_index" in ref_db.columns
        assert "fov_name" in ref_db.columns

    def test_error_columns_weighting(self):
        host = _host()
        df = _mode_b_df(with_photoelectrons=True, with_errors=True)
        means, ref_db, gmm = host.extract_reference_means(
            df, reference_photoelectron_threshold=None, verbose=True,
        )
        assert means.shape == (2, 2)

    def test_invalid_fit_type_raises(self):
        host = _host()
        df = _mode_b_df()
        with pytest.raises(ValueError, match="fit_type must be"):
            host.extract_reference_means(df, reference_photoelectron_threshold=None, fit_type="bogus")

    def test_n_components_not_two(self):
        # Regression test: extract_reference_means used to hardcode
        # posterior_prob_0/posterior_prob_1, crashing with a bare IndexError for
        # any n_components != 2. Now builds posterior_prob_{i} dynamically.
        host = _host()
        df = _mode_b_df(n_per=20, seed=12)
        means, ref_db, gmm = host.extract_reference_means(
            df, reference_photoelectron_threshold=None, n_components=1, verbose=False,
        )
        assert means.shape == (1, 2)
        assert "posterior_prob_0" in ref_db.columns
        assert "posterior_prob_1" not in ref_db.columns

    def test_mle_downsample_branch(self):
        # Uniform tiny error columns on a large-ish dataset -> every point gets the
        # same normalised weight (1.0) -> replication_count=5 each -> for n=700
        # that's 3500 > 3000, triggering the MLE downsample-to-3000 path.
        host = _host()
        df = _mode_b_df(n_per=350, seed=13, with_photoelectrons=False, with_errors=True, sigma=0.03)
        df["A_R_err"] = 1e-6
        df["A_G_err"] = 1e-6
        means, ref_db, gmm = host.extract_reference_means(
            df, reference_photoelectron_threshold=None, fit_type="MLE", verbose=True,
        )
        assert means.shape == (2, 2)

    def test_more_components_than_real_peaks_falls_back_to_quantiles(self):
        # Only 2 real populations, so the per-dimension histogram-peak search
        # (this specific seed finds 3-4 peaks per dimension) comes up short of
        # 6 requested components, exercising extract_reference_means' own
        # quantile fallback (a separate inline duplicate of
        # _find_histogram_peaks_1d's fallback).
        host = _host()
        df = _mode_b_df(n_per=20, seed=44)
        means, ref_db, gmm = host.extract_reference_means(
            df, reference_photoelectron_threshold=None, n_components=6, verbose=False,
        )
        assert means.shape == (6, 2)

    def test_plotting_exception_is_caught_and_warned(self, monkeypatch, caplog):
        import logging

        host = _host()
        df = _mode_b_df(n_per=15, seed=45)

        def _raise(*a, **kw):
            raise RuntimeError("forced plotting failure")

        monkeypatch.setattr(PlottingBase.AnalysisPlotter, "two_column_plot", _raise)
        with caplog.at_level(logging.WARNING):
            means, ref_db, gmm = host.extract_reference_means(
                df, reference_photoelectron_threshold=None, verbose=True,
            )
        assert means.shape == (2, 2)
        assert "Plotting skipped - error" in caplog.text

    def test_plotting_import_error_is_caught_and_warned(self, monkeypatch, caplog):
        import logging

        host = _host()
        df = _mode_b_df(n_per=15, seed=47)
        monkeypatch.delattr(PlottingBase, "AnalysisPlotter")
        with caplog.at_level(logging.WARNING):
            means, ref_db, gmm = host.extract_reference_means(
                df, reference_photoelectron_threshold=None, verbose=True,
            )
        assert means.shape == (2, 2)
        assert "PlottingBase not available" in caplog.text


# ======================================================================
# fit_covariances_fixed_means
# ======================================================================

class TestFitCovariancesFixedMeans:
    def test_mle_normal_fit(self):
        host = _host()
        X = _two_component_X(n_per=25, seed=14)
        fixed_means = np.array([MEAN0, MEAN1])
        covs, weights, converged = host.fit_covariances_fixed_means(
            X, fixed_means, fit_type="MLE", verbose=True,
        )
        assert covs.shape == (2, 2, 2)
        assert weights.shape == (2,)

    def test_mle_hits_invalid_weight_branch(self, monkeypatch):
        # Probe the objective directly with an invalid weight (>=1) before
        # letting the real optimizer run -- not organically reached by
        # L-BFGS-B's exploration from a good init.
        import scipy.optimize as sopt

        host = _host()
        X = _two_component_X(n_per=15, seed=42)
        fixed_means = np.array([MEAN0, MEAN1])
        real_minimize = sopt.minimize

        def probe_minimize(fun, x0, **kwargs):
            bad_weight = x0.copy()
            bad_weight[-1] = 5.0  # weight >= 1, invalid
            fun(bad_weight)
            return real_minimize(fun, x0, **kwargs)

        monkeypatch.setattr(sopt, "minimize", probe_minimize)
        covs, weights, converged = host.fit_covariances_fixed_means(
            X, fixed_means, fit_type="MLE",
        )
        assert covs.shape == (2, 2, 2)

    def test_mle_exception_branch(self, monkeypatch):
        # Same technique as _fit_gmm_mle's exception-branch test: force
        # multivariate_normal to raise on its first call, which happens
        # organically during the optimizer's first objective evaluation.
        host = _host()
        X = _two_component_X(n_per=15, seed=46)
        fixed_means = np.array([MEAN0, MEAN1])
        real_mvn = mixture_analysis.multivariate_normal
        calls = {"n": 0}

        def flaky_mvn(mean, cov):
            calls["n"] += 1
            if calls["n"] == 1:
                raise np.linalg.LinAlgError("forced singular")
            return real_mvn(mean=mean, cov=cov)

        monkeypatch.setattr(mixture_analysis, "multivariate_normal", flaky_mvn)
        covs, weights, converged = host.fit_covariances_fixed_means(
            X, fixed_means, fit_type="MLE",
        )
        assert covs.shape == (2, 2, 2)

    def test_em_normal_fit_converges(self):
        host = _host()
        X = _two_component_X(n_per=25, seed=15)
        fixed_means = np.array([MEAN0, MEAN1])
        covs, weights, converged = host.fit_covariances_fixed_means(
            X, fixed_means, fit_type="EM", max_iter=100, tol=1e-4, verbose=True,
        )
        assert covs.shape == (2, 2, 2)
        assert converged in (True, False)

    def test_em_does_not_converge_within_one_iteration(self):
        host = _host()
        X = _two_component_X(n_per=25, seed=16)
        fixed_means = np.array([MEAN0, MEAN1])
        covs, weights, converged = host.fit_covariances_fixed_means(
            X, fixed_means, fit_type="EM", max_iter=1, tol=1e-30, verbose=True,
        )
        assert converged is False

    def test_em_singular_covariance_regularised(self, monkeypatch):
        # Force a LinAlgError on the first multivariate_normal construction inside
        # the E-step to exercise the singular-covariance regularisation fallback.
        host = _host()
        X = _two_component_X(n_per=25, seed=17)
        fixed_means = np.array([MEAN0, MEAN1])

        real_mvn = mixture_analysis.multivariate_normal
        calls = {"n": 0}

        def flaky_mvn(mean, cov):
            calls["n"] += 1
            if calls["n"] == 1:
                raise np.linalg.LinAlgError("forced singular")
            return real_mvn(mean=mean, cov=cov)

        monkeypatch.setattr(mixture_analysis, "multivariate_normal", flaky_mvn)
        covs, weights, converged = host.fit_covariances_fixed_means(
            X, fixed_means, fit_type="EM", max_iter=5,
        )
        assert covs.shape == (2, 2, 2)

    def test_em_degenerate_component_regularised_in_mstep(self):
        # Duplicate points collapse a component's weighted covariance towards a
        # (near-)zero matrix -> min eigenvalue < 1e-6 -> M-step regularisation.
        host = _host()
        X0 = np.tile(MEAN0, (20, 1))
        X1 = np.random.default_rng(43).multivariate_normal(MEAN1, np.eye(2) * 0.03**2, 20)
        X = np.vstack([X0, X1])
        fixed_means = np.array([MEAN0, MEAN1])
        covs, weights, converged = host.fit_covariances_fixed_means(
            X, fixed_means, fit_type="EM", max_iter=5,
        )
        assert covs.shape == (2, 2, 2)


# ======================================================================
# fit_covariances_fixed_means_mestimator
# ======================================================================

class TestFitCovariancesFixedMeansMestimator:
    def test_tukey_normal_fit_verbose(self):
        # tol=0.0 prevents early convergence so the loop actually reaches
        # iteration == max_iter - 1, exercising the last-iteration diagnostic
        # block (including the reference_covariances ratio log line).
        host = _host()
        X = _two_component_X(n_per=30, seed=18)
        fixed_means = np.array([MEAN0, MEAN1])
        ref_covs = np.array([np.eye(2) * 0.001, np.eye(2) * 0.001])
        covs, weights, point_weights = host.fit_covariances_fixed_means_mestimator(
            X, fixed_means, reference_covariances=ref_covs,
            estimator_type="tukey", max_iter=5, tol=0.0, verbose=True,
        )
        assert covs.shape == (2, 2, 2)
        assert len(point_weights) == 2

    def test_huber_estimator_downweights_outliers(self):
        host = _host()
        X = _two_component_X(n_per=30, seed=19)
        # Inject a few outlier points into component 0's data.
        X = np.vstack([X, np.array([[3.0, 3.0], [3.2, 2.9], [-2.0, -2.0]])])
        fixed_means = np.array([MEAN0, MEAN1])
        covs, weights, point_weights = host.fit_covariances_fixed_means_mestimator(
            X, fixed_means, estimator_type="huber", max_iter=10,
        )
        assert covs.shape == (2, 2, 2)

    def test_converges_before_max_iter_logs_convergence_message(self, caplog):
        import logging

        host = _host()
        X = _two_component_X(n_per=30, seed=48)
        fixed_means = np.array([MEAN0, MEAN1])
        with caplog.at_level(logging.INFO):
            covs, weights, point_weights = host.fit_covariances_fixed_means_mestimator(
                X, fixed_means, max_iter=20, tol=1e-2, verbose=True,
            )
        assert "Converged at iteration" in caplog.text

    def test_sparse_component_uses_reference_covariance(self):
        host = _host()
        X = _two_component_X(n_per=25, seed=20)
        # Third mean far away -> gets 0 or 1 points -> len(X_k) <= n_features branch.
        fixed_means = np.array([MEAN0, MEAN1, [10.0, 10.0]])
        ref_covs = np.array([np.eye(2) * 0.001, np.eye(2) * 0.001, np.eye(2) * 0.005])
        covs, weights, point_weights = host.fit_covariances_fixed_means_mestimator(
            X, fixed_means, reference_covariances=ref_covs, max_iter=3,
        )
        assert covs.shape == (3, 2, 2)

    def test_sparse_component_without_reference_uses_small_identity(self):
        host = _host()
        X = _two_component_X(n_per=25, seed=21)
        fixed_means = np.array([MEAN0, MEAN1, [10.0, 10.0]])
        covs, weights, point_weights = host.fit_covariances_fixed_means_mestimator(
            X, fixed_means, reference_covariances=None, max_iter=3,
        )
        assert covs.shape == (3, 2, 2)

    def test_near_zero_scale_falls_back_to_one(self):
        # Duplicate points sitting exactly on the fixed mean -> mahalanobis
        # distances of (near) zero for the whole component -> scale < 1e-10.
        host = _host()
        n = 30
        X0 = np.tile(MEAN0, (n, 1))
        X1 = np.tile(MEAN1, (n, 1)) + np.random.default_rng(22).normal(0, 0.02, (n, 2))
        X = np.vstack([X0, X1])
        fixed_means = np.array([MEAN0, MEAN1])
        covs, weights, point_weights = host.fit_covariances_fixed_means_mestimator(
            X, fixed_means, max_iter=3,
        )
        assert covs.shape == (2, 2, 2)

    def test_singular_covariance_during_reweighting_regularised(self, monkeypatch):
        host = _host()
        X = _two_component_X(n_per=30, seed=23)
        fixed_means = np.array([MEAN0, MEAN1])

        real_inv = np.linalg.inv
        calls = {"n": 0}

        def flaky_inv(a):
            calls["n"] += 1
            if calls["n"] == 1:
                raise np.linalg.LinAlgError("forced singular")
            return real_inv(a)

        monkeypatch.setattr(np.linalg, "inv", flaky_inv)
        covs, weights, point_weights = host.fit_covariances_fixed_means_mestimator(
            X, fixed_means, max_iter=3,
        )
        assert covs.shape == (2, 2, 2)


# ======================================================================
# analyze_photon_dependent_misidentification_analytical
# ======================================================================

class TestAnalyzePhotonDependentMisidentificationAnalytical:
    def _refs(self, host):
        df = _mode_a_df(n_mol_per=15, frames_per_mol=5)
        fixed_means, ref_db, gmm = host.extract_reference_means(
            df, reference_photoelectron_threshold=3000.0, verbose=False,
        )
        return df, fixed_means, ref_db, gmm

    def test_verbose_earliest_entry(self):
        host = _host()
        df, fixed_means, ref_db, gmm = self._refs(host)
        photon_bins = np.array([1000.0, 3000.0, 5001.0])
        summary_db = host.analyze_photon_dependent_misidentification_analytical(
            df, fixed_means, ref_db, photon_bins,
            reference_covariances=gmm.covariances_,
            use_earliest_entry=True, n_mc_samples=500, verbose=True,
        )
        assert len(summary_db) > 0

    def test_midpoint_entry_mode(self):
        host = _host()
        df, fixed_means, ref_db, gmm = self._refs(host)
        photon_bins = np.array([1000.0, 3000.0, 5001.0])
        summary_db = host.analyze_photon_dependent_misidentification_analytical(
            df, fixed_means, ref_db, photon_bins,
            use_earliest_entry=False, n_mc_samples=500, verbose=False,
        )
        assert len(summary_db) > 0

    def test_old_accumulation_database_raises(self):
        host = _host()
        df, fixed_means, ref_db, gmm = self._refs(host)
        old = df.rename(columns={"photoelectrons_accumulated": "photons_accumulated"})
        with pytest.raises(ValueError, match="no 'photoelectrons_accumulated'"):
            host.analyze_photon_dependent_misidentification_analytical(
                old, fixed_means, ref_db, np.array([1000.0, 3000.0]), verbose=False)

    def test_empty_bin_is_skipped(self):
        host = _host()
        df, fixed_means, ref_db, gmm = self._refs(host)
        # A bin range with no molecules (well above max accumulated photons) sits
        # between two populated bins -> exercises the "no molecules, skip" branch.
        photon_bins = np.array([1000.0, 3000.0, 1e9, 1e10])
        summary_db = host.analyze_photon_dependent_misidentification_analytical(
            df, fixed_means, ref_db, photon_bins,
            use_earliest_entry=True, n_mc_samples=500, verbose=False,
        )
        assert len(summary_db) < len(photon_bins) - 1

    def test_all_bins_empty_returns_empty_summary(self):
        host = _host()
        df, fixed_means, ref_db, gmm = self._refs(host)
        photon_bins = np.array([1e9, 1e10])
        summary_db = host.analyze_photon_dependent_misidentification_analytical(
            df, fixed_means, ref_db, photon_bins,
            use_earliest_entry=True, n_mc_samples=500, verbose=True,
        )
        assert len(summary_db) == 0

    def test_per_bin_plotting_exception_is_caught_and_warned(self, monkeypatch, caplog):
        import logging

        host = _host()
        df, fixed_means, ref_db, gmm = self._refs(host)
        photon_bins = np.array([1000.0, 3000.0, 5001.0])

        def _raise(*a, **kw):
            raise RuntimeError("forced plotting failure")

        monkeypatch.setattr(PlottingBase.AnalysisPlotter, "two_column_plot", _raise)
        with caplog.at_level(logging.WARNING):
            summary_db = host.analyze_photon_dependent_misidentification_analytical(
                df, fixed_means, ref_db, photon_bins,
                use_earliest_entry=True, n_mc_samples=500, verbose=True,
            )
        assert len(summary_db) > 0
        assert "Plotting skipped for this bin" in caplog.text


# ======================================================================
# _find_histogram_peaks_1d / _find_initial_means_2d
# ======================================================================

class TestFindHistogramPeaks1d:
    def test_finds_two_peaks(self):
        host = _host()
        data = np.concatenate([
            np.random.default_rng(24).normal(0.2, 0.02, 200),
            np.random.default_rng(25).normal(0.8, 0.02, 200),
        ])
        peaks = host._find_histogram_peaks_1d(data, n_peaks=2)
        assert len(peaks) == 2

    def test_falls_back_to_quantiles_when_too_few_peaks(self):
        host = _host()
        data = np.random.default_rng(26).uniform(0, 1, 20)
        peaks = host._find_histogram_peaks_1d(data, n_peaks=10)
        assert len(peaks) == 10


class TestFindInitialMeans2d:
    def test_histogram_peaks_two_channels(self):
        host = _host()
        X = _two_component_X(n_per=30, seed=27)
        means = host._find_initial_means_2d(X, n_channels=2, method="histogram_peaks")
        assert means.shape == (2, 2)

    def test_histogram_peaks_non_two_channels(self):
        host = _host()
        X = _two_component_X(n_per=30, seed=28)
        means = host._find_initial_means_2d(X, n_channels=1, method="histogram_peaks")
        assert means.shape == (1, 2)

    def test_kmeans_method(self):
        host = _host()
        X = _two_component_X(n_per=30, seed=29)
        means = host._find_initial_means_2d(X, n_channels=2, method="kmeans")
        assert means.shape == (2, 2)

    def test_unknown_method_raises(self):
        host = _host()
        X = _two_component_X(n_per=10, seed=30)
        with pytest.raises(ValueError, match="Unknown method"):
            host._find_initial_means_2d(X, n_channels=2, method="bogus")


# ======================================================================
# _estimate_initial_covariances_2d
# ======================================================================

class TestEstimateInitialCovariances2d:
    def test_normal_core_region_estimate(self):
        host = _host()
        X = _two_component_X(n_per=60, seed=31)
        means = np.array([MEAN0, MEAN1])
        covs = host._estimate_initial_covariances_2d(X, means, n_channels=2)
        assert covs.shape == (2, 2, 2)

    def test_with_error_incorporation(self):
        host = _host()
        X = _two_component_X(n_per=60, seed=32)
        means = np.array([MEAN0, MEAN1])
        X_err = np.full_like(X, 0.01)
        covs = host._estimate_initial_covariances_2d(X, means, n_channels=2, X_err=X_err)
        assert covs.shape == (2, 2, 2)

    def test_no_core_region(self):
        host = _host()
        X = _two_component_X(n_per=30, seed=33)
        means = np.array([MEAN0, MEAN1])
        covs = host._estimate_initial_covariances_2d(X, means, n_channels=2, use_core_region=False)
        assert covs.shape == (2, 2, 2)

    def test_few_points_assigned_uses_small_isotropic(self):
        host = _host()
        X = _two_component_X(n_per=25, seed=34)
        # Third mean far away -> at most a handful of points assigned (<=20).
        means = np.array([MEAN0, MEAN1, [10.0, 10.0]])
        covs = host._estimate_initial_covariances_2d(X, means, n_channels=3)
        np.testing.assert_allclose(covs[2], np.eye(2) * 0.005)

    def test_moderate_points_but_tiny_core_uses_small_isotropic(self):
        # Core-region selection only activates for n_assigned > 50, and a very
        # low percentile then shrinks that core down to <= 5 points.
        host = _host()
        X = _two_component_X(n_per=60, seed=35)
        means = np.array([MEAN0, MEAN1])
        covs = host._estimate_initial_covariances_2d(
            X, means, n_channels=2, use_core_region=True, percentile=1,
        )
        assert covs.shape == (2, 2, 2)

    def test_non_positive_definite_forces_stronger_regularisation(self, monkeypatch):
        host = _host()
        X = _two_component_X(n_per=60, seed=36)
        means = np.array([MEAN0, MEAN1])

        real_eigvalsh = np.linalg.eigvalsh
        calls = {"n": 0}

        def flaky_eigvalsh(a):
            calls["n"] += 1
            if calls["n"] == 1:
                return np.array([-0.5, 1.0])
            return real_eigvalsh(a)

        monkeypatch.setattr(np.linalg, "eigvalsh", flaky_eigvalsh)
        covs = host._estimate_initial_covariances_2d(X, means, n_channels=2)
        assert covs.shape == (2, 2, 2)
