# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from typing import Any, Optional
from numpy.typing import NDArray
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
import logging
from pyS3M.PlottingBase import _safe_tight_layout
logger = logging.getLogger(__name__)


class _SerialPoolResult:
    """`multiprocessing.pool.AsyncResult`-alike wrapping an already-computed value."""

    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


class _SerialPool:
    """In-thread, no-subprocess stand-in for `multiprocessing.Pool`.

    `pygmmis.fit`/`pygmmis.GMM.logL` always spawn a bare `multiprocessing.Pool()`
    internally with no way to disable it. Creating a real `Pool` from a
    background `QThread` (the GUI's `AnalysisWorker`) is a known-fragile
    combination on macOS (Cocoa runtime + `spawn` start method) -- seen as a
    GUI crash with leaked-semaphore `resource_tracker` warnings during Channel
    Unmixing, not reproducible on Linux. Swapping `pygmmis`'s `multiprocessing.
    Pool` for this serial stand-in (macOS only, see `_fit_gmm_pygmmis`) avoids
    spawning any subprocess at all, at the cost of pygmmis's internal
    parallelism on that platform.

    pygmmis hands this pool to `parmap.starmap`/`parmap.map` as `pm_pool`,
    which call `pool.map_async(func, iterable, chunksize)`, not
    `apply_async` -- both are implemented here. `chunksize` is accepted for
    signature compatibility and ignored.
    """

    def apply_async(self, func, args=()):
        return _SerialPoolResult(func(*args))

    def map_async(self, func, iterable, chunksize=None):
        return _SerialPoolResult([func(item) for item in iterable])

    def close(self):
        pass

    def join(self):
        pass


def _no_shared_array(a, dtype=None):
    """Serial-mode stand-in for `pygmmis.createShared`.

    `pygmmis.fit` also builds several `multiprocessing.Array`-backed shared
    arrays (`data_`, `covar_`, `log_S`, ...) via its own `createShared`
    helper, independently of the `multiprocessing.Pool` object patched by
    `_SerialPool` above -- constructing that POSIX shared memory (`mmap` +
    an internal `Lock`) is itself a second, lower-level macOS crash surface
    (seen as a `bus error` even after `_SerialPool` alone stopped the
    `Pool`-related crash and cut the leaked-semaphore count from 7 to 1 --
    that remaining 1 was this shared array's own lock). Under `_SerialPool`
    there are no subprocess workers to share memory with in the first
    place, so a plain in-process `float64` array is a behaviourally
    identical, zero-risk substitute -- `createShared` itself just returns a
    flatten-then-reshape copy of the input, no different from `np.array`.
    """
    return np.array(a, dtype=np.float64)


class MixtureAnalysisMixin:
    """Mixin providing GMM mixture-analysis methods for extract_SMs."""
    def _fit_gmm_mle(
        self,
        X: NDArray[np.float64],
        initial_means: NDArray[np.float64],
        n_components: int,
        covariance_type: str = "full",
        max_iter: int = 500,
        verbose: bool = False,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], bool]:
        """
        Fit Gaussian Mixture Model using Maximum Likelihood Estimation (MLE).

        Uses scipy.optimize to directly minimize negative log-likelihood.
        This is more robust than EM when dealing with weighted data.

        Args:
            X (np.ndarray): Data points, shape (n_samples, n_features)
            initial_means (np.ndarray): Initial mean positions, shape (n_components, n_features)
            n_components (int): Number of mixture components
            covariance_type (str): Type of covariance ('full', 'tied', 'diag', 'spherical')
            max_iter (int): Maximum optimization iterations
            verbose (bool): Print optimization progress

        Returns:
            tuple: (means, covariances, weights, converged)
                - means (np.ndarray): Fitted means, shape (n_components, n_features)
                - covariances (np.ndarray): Fitted covariances, shape (n_components, n_features, n_features)
                - weights (np.ndarray): Component weights, shape (n_components,)
                - converged (bool): Whether optimization succeeded
        """
        from scipy.optimize import minimize

        def pack_params(means, covariances, weights):
            """Pack GMM parameters into flat array."""
            params = []
            # Means
            params.extend(means.flatten())
            # Covariances (upper triangular elements)
            for k in range(n_components):
                cov = covariances[k]
                params.extend([cov[0, 0], cov[0, 1], cov[1, 1]])
            # Weights (n-1 to maintain sum=1 constraint)
            params.extend(weights[:-1])
            return np.array(params)

        def unpack_params(params):
            """Unpack flat array into GMM parameters."""
            idx = 0
            # Means
            means = params[:n_components * 2].reshape(n_components, 2)
            idx += n_components * 2
            # Covariances
            covariances = []
            for k in range(n_components):
                cov = np.array([
                    [params[idx], params[idx + 1]],
                    [params[idx + 1], params[idx + 2]]
                ])
                covariances.append(cov)
                idx += 3
            # Weights
            weights = np.zeros(n_components)
            weights[:-1] = params[idx:]
            weights[-1] = 1.0 - weights[:-1].sum()
            return means, np.array(covariances), weights

        def negative_log_likelihood(params):
            """Negative log-likelihood for minimization."""
            try:
                means, covariances, weights = unpack_params(params)

                # Check validity
                for k in range(n_components):
                    eigvals = np.linalg.eigvalsh(covariances[k])
                    if np.any(eigvals <= 0):
                        return 1e10
                if np.any(weights <= 0) or np.any(weights >= 1):
                    return 1e10

                # Calculate log-likelihood
                log_probs = np.zeros((len(X), n_components))
                for k in range(n_components):
                    mvn = multivariate_normal(mean=means[k], cov=covariances[k])
                    log_probs[:, k] = mvn.logpdf(X) + np.log(weights[k])

                log_probs_max = log_probs.max(axis=1, keepdims=True)
                log_likelihood = np.sum(log_probs_max + np.log(np.exp(log_probs - log_probs_max).sum(axis=1)))

                return -log_likelihood

            except (np.linalg.LinAlgError, ValueError):
                return 1e10

        # Initialize with sklearn GMM
        gmm_init = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=50,
            n_init=1,
            means_init=initial_means,
        )
        gmm_init.fit(X)

        # Pack initial parameters
        params_init = pack_params(gmm_init.means_, gmm_init.covariances_, gmm_init.weights_)

        if verbose:
            logger.info(f"  Running MLE optimization (L-BFGS-B)...")

        # Optimize with MLE
        result = minimize(
            negative_log_likelihood,
            params_init,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-6}
        )

        # Unpack optimized parameters
        means_opt, covariances_opt, weights_opt = unpack_params(result.x)

        return means_opt, covariances_opt, weights_opt, result.success

    def _fit_gmm_em(
        self,
        X: NDArray[np.float64],
        initial_means: NDArray[np.float64],
        n_components: int,
        covariance_type: str = "full",
        max_iter: int = 100,
        n_reweighting_iterations: int = 2,
        photoelectrons: NDArray[np.float64] | None = None,
        A_R: NDArray[np.float64] | None = None,
        A_G: NDArray[np.float64] | None = None,
        has_error_columns: bool = False,
        sigma_A_R: NDArray[np.float64] | None = None,
        sigma_A_G: NDArray[np.float64] | None = None,
        verbose: bool = False,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], bool]:
        """
        Fit Gaussian Mixture Model using Expectation-Maximization (EM) with iterative re-weighting.

        This approach normalizes weights within each component to handle population imbalance better.

        Args:
            X (np.ndarray): Data points, shape (n_samples, n_features)
            initial_means (np.ndarray): Initial mean positions, shape (n_components, n_features)
            n_components (int): Number of mixture components
            covariance_type (str): Type of covariance ('full', 'tied', 'diag', 'spherical')
            max_iter (int): Maximum EM iterations
            n_reweighting_iterations (int): Number of re-weighting iterations (default: 2)
            photoelectrons (np.ndarray): Photoelectron counts for weighting (optional)
            A_R (np.ndarray): A_R values for uncertainty calculation (optional)
            A_G (np.ndarray): A_G values for uncertainty calculation (optional)
            has_error_columns (bool): Whether error columns are available
            sigma_A_R (np.ndarray): A_R errors (optional)
            sigma_A_G (np.ndarray): A_G errors (optional)
            verbose (bool): Print progress

        Returns:
            tuple: (means, covariances, weights, converged)
                - means (np.ndarray): Fitted means, shape (n_components, n_features)
                - covariances (np.ndarray): Fitted covariances, shape (n_components, n_features, n_features)
                - weights (np.ndarray): Component weights, shape (n_components,)
                - converged (bool): Whether EM converged
        """
        # Initial fit using histogram-based initialization
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=max_iter,
            n_init=1,
            means_init=initial_means,
        )
        gmm.fit(X)

        # Iterative re-weighting
        for iteration in range(n_reweighting_iterations):
            if verbose:
                logger.info(f"  Re-weighting iteration {iteration + 1}/{n_reweighting_iterations}...")

            # Get component assignments on original data
            labels = gmm.predict(X)

            # Build within-component weighted dataset
            X_reweighted_list = []
            for k in range(n_components):
                component_mask = labels == k
                n_in_component = component_mask.sum()

                if n_in_component == 0:
                    continue

                component_data = X[component_mask]

                # Apply error-based weighting within this component only
                if photoelectrons is not None and A_R is not None and A_G is not None:
                    component_photoelectrons = photoelectrons[component_mask]
                    component_A_R = A_R[component_mask]
                    component_A_G = A_G[component_mask]

                    # Calculate uncertainty for this component
                    if has_error_columns and sigma_A_R is not None and sigma_A_G is not None:
                        component_sigma_A_R = sigma_A_R[component_mask]
                        component_sigma_A_G = sigma_A_G[component_mask]
                        sigma_combined_k = np.sqrt(component_sigma_A_R**2 + component_sigma_A_G**2)
                    else:
                        sigma_A_R_k = np.sqrt(component_A_R * (1 - component_A_R) / (component_photoelectrons + 1))
                        sigma_A_G_k = np.sqrt(component_A_G * (1 - component_A_G) / (component_photoelectrons + 1))
                        sigma_combined_k = np.sqrt(sigma_A_R_k**2 + sigma_A_G_k**2)

                    # Weight = 1/sigma
                    weights_k = 1.0 / (sigma_combined_k + 1e-10)
                    weights_k = weights_k / weights_k.mean()

                    # Convert to integer replication counts
                    rep_counts_k = np.round(weights_k * 10).astype(int)
                    rep_counts_k = np.maximum(rep_counts_k, 1)

                    X_reweighted_list.append(np.repeat(component_data, rep_counts_k, axis=0))
                else:
                    # No photoelectron weighting - just use the data as is
                    X_reweighted_list.append(component_data)

            # Combine all components
            if len(X_reweighted_list) > 0:
                X_balanced = np.vstack(X_reweighted_list)
            else:
                X_balanced = X

            # Refit GMM with balanced weights
            gmm.fit(X_balanced)

        return gmm.means_, gmm.covariances_, gmm.weights_, gmm.converged_

    def _fit_gmm_pygmmis(
        self,
        X,
        X_err,
        initial_means,
        n_components,
        max_iter=100,
        verbose=False,
    ):
        """
        Fit Gaussian Mixture Model using pygmmis Extreme Deconvolution.

        This method properly handles per-point measurement uncertainties by deconvolving
        measurement noise from the intrinsic distribution. This is the theoretically correct
        approach for SMLM data where each localization has its own fitting uncertainty.

        Unlike point replication methods, Extreme Deconvolution:
        - Treats measurement covariances as part of the model
        - Deconvolves noise to recover the true error-free distribution
        - Scales to millions of points without replication overhead
        - Provides proper uncertainty quantification

        Args:
            X (np.ndarray): Data points, shape (n_samples, n_features)
            X_err (np.ndarray): Per-point uncertainties, shape (n_samples, n_features)
                These should be the standard errors (sigma), not variances.
            initial_means (np.ndarray): Initial mean positions, shape (n_components, n_features)
            n_components (int): Number of mixture components
            max_iter (int): Maximum iterations for extreme deconvolution (default: 100)
            verbose (bool): Print progress

        Returns:
            tuple: (means, covariances, weights, converged)
                - means (np.ndarray): Fitted means, shape (n_components, n_features)
                - covariances (np.ndarray): Fitted covariances, shape (n_components, n_features, n_features)
                - weights (np.ndarray): Component weights, shape (n_components,)
                - converged (bool): Whether fitting converged

        References:
            Bovy, Hogg & Roweis (2011) "Extreme deconvolution: Inferring complete
            distribution functions from noisy, heterogeneous and incomplete observations"
            https://github.com/pmelchior/pygmmis
        """
        try:
            import pygmmis
        except ImportError:
            raise ImportError(
                "pygmmis is required for extreme deconvolution fitting. "
                "Install with: pip install pygmmis"
            )

        n_samples, n_features = X.shape

        if verbose:
            logger.info(f"  Extreme Deconvolution fitting with pygmmis...")
            logger.info(f"    Data: {n_samples} points, {n_features} features")
            logger.info(f"    Components: {n_components}")
            logger.info(f"    Mean errors: {X_err.mean(axis=0)}")

        # Prepare per-point covariance matrices (diagonal, since A_R and A_G errors are independent)
        covar = np.zeros((n_samples, n_features, n_features))
        for i in range(n_samples):
            # Convert standard errors to variances (sigma^2)
            covar[i] = np.diag(X_err[i]**2)

        # Initialize GMM with k-means or provided means
        gmm = pygmmis.GMM(K=n_components, D=n_features)

        # Set initial means
        gmm.mean = initial_means.copy()

        # Initialize covariances using simple empirical estimate
        # Assign each point to nearest mean and estimate covariance
        from scipy.spatial.distance import cdist
        distances = cdist(X, initial_means)
        labels = np.argmin(distances, axis=1)

        initial_covariances = np.zeros((n_components, n_features, n_features))
        initial_weights = np.zeros(n_components)

        for k in range(n_components):
            mask = labels == k
            n_k = mask.sum()

            if n_k > n_features:  # Need enough points to estimate covariance
                X_k = X[mask]
                # Empirical covariance
                diff = X_k - initial_means[k]
                cov_k = (diff.T @ diff) / n_k
                # Add regularization to ensure positive definite
                cov_k += np.eye(n_features) * 1e-4
                initial_covariances[k] = cov_k
                initial_weights[k] = n_k / n_samples
            else:
                # Not enough points, use identity
                initial_covariances[k] = np.eye(n_features) * 0.01
                initial_weights[k] = 1.0 / n_components

        # Ensure weights sum to 1
        initial_weights /= initial_weights.sum()

        gmm.covar = initial_covariances
        gmm.amp = initial_weights

        if verbose:
            logger.info(f"    Initial weights: {gmm.amp}")
            logger.info(f"    Running extreme deconvolution (max_iter={max_iter})...")

        # Run extreme deconvolution
        # pygmmis returns log-likelihood and component assignments
        #
        # On macOS, pygmmis.fit's internal multiprocessing.Pool() and its
        # multiprocessing.Array-backed shared arrays (via createShared) must
        # not be created from a background QThread (see _SerialPool's and
        # _no_shared_array's docstrings) -- swap both for in-thread stand-ins
        # for the duration of this call only, then restore them.
        patch_pool = sys.platform == "darwin"
        if patch_pool:
            original_pool = pygmmis.multiprocessing.Pool
            original_shared = pygmmis.createShared
            pygmmis.multiprocessing.Pool = lambda *a, **k: _SerialPool()
            pygmmis.createShared = _no_shared_array
        try:
            logL, U = pygmmis.fit(
                gmm,
                data=X,
                covar=covar,
                init_method='none',   # Use our provided initialization
                w=1e-6,              # Covariance regularization (small value)
                cutoff=5.0,          # Mahalanobis distance cutoff for outliers
                maxiter=max_iter,
                tol=1e-3,            # Convergence tolerance
            )

            converged = True  # pygmmis doesn't explicitly report convergence

            if verbose:
                logger.info(f"    Final log-likelihood: {logL:.2f}")
                logger.info(f"    Final weights: {gmm.amp}")

        except Exception as e:
            if verbose:
                logger.info(f"    Warning: Extreme deconvolution failed: {e}")
                logger.info(f"    Returning initial parameters")
            converged = False
        finally:
            if patch_pool:
                pygmmis.multiprocessing.Pool = original_pool
                pygmmis.createShared = original_shared

        # Extract results
        means = gmm.mean.copy()
        covariances = gmm.covar.copy()
        weights = gmm.amp.copy()

        return means, covariances, weights, converged

    def extract_reference_means(
        self,
        data_db: pd.DataFrame,
        reference_photoelectron_threshold: float | None = None,
        n_components: int = 2,
        covariance_type: str = "full",
        fit_type: str = "MLE",
        random_state: int = 42,
        verbose: bool = True,
    ) -> tuple[NDArray[np.float64], pd.DataFrame, Any]:
        """
        Extract reference means from molecule data using GMM (analytical approach).

        Fits a Gaussian Mixture Model to establish reference mean positions for each dye
        population. These fixed means are then used with photoelectron-dependent covariance fitting
        to analytically calculate misidentification rates.

        This is the first step in the analytical approach:
        1. Extract means from reference data (this function)
        2. Fit covariances at each photoelectron level with fixed means
        3. Analytically calculate overlap/error rates from distributions

        Two modes of operation:

        - **Photon accumulation database + threshold**: Use highest-photoelectron data only.
          Pass photon_accumulation_db with reference_photoelectron_threshold to extract
          molecules reaching threshold for stable mean estimates.
        - **Single molecule database**: Use all molecules (no threshold). Pass
          single_molecule_database with reference_photoelectron_threshold=None to use
          averaged RGB values from all molecules.

        Args:
            data_db (pd.DataFrame): Either:
                - Photon accumulation database (with 'photoelectrons_accumulated' column)
                - Single molecule database (with 'A_R', 'A_G', 'A_B' columns)
            reference_photoelectron_threshold (float, optional): Minimum photoelectrons for reference.
                - If provided: Filters accumulation DB to molecules >= threshold
                - If None: Uses all molecules (assumes single molecule database)
                (default: None - use all molecules)
            n_components (int): Number of Gaussian components (default: 2)
            covariance_type (str): GMM covariance type - "full", "tied", "diag", "spherical"
                (default: "full" allows correlation between A_R and A_G)
            fit_type (str): Fitting algorithm - "MLE" or "EM" (default: "MLE")
                - "MLE": Direct maximum likelihood via scipy.optimize (more robust)
                - "EM": Iterative EM with within-component re-weighting (faster, handles imbalance)
            random_state (int): Random seed for reproducibility
            verbose (bool): Print progress and statistics

        Returns:
            tuple: (reference_means, reference_db, gmm_model)
                - reference_means (np.ndarray): Shape (n_components, 2) - fixed means for [A_R, A_G]
                - reference_db (pd.DataFrame): Reference molecules with assignments — molecular_index (unique molecule ID), true_label (0 or 1, e.g. 0=Red, 1=Green), A_R_ref/A_G_ref/A_B_ref (reference RGB values), posterior_prob_0/posterior_prob_1 (posterior probabilities), photoelectrons or max_photoelectrons (from single molecule DB or accumulation DB respectively), fov_index/fov_name (FOV tracking, if available)
                - gmm_model (GaussianMixture): Fitted GMM (for reference only)

        Examples:
            >>> # Mode A: High-photon data from accumulation database
            >>> means, ref_db, gmm = SM_E.extract_reference_means(
            ...     pa_db,
            ...     reference_photoelectron_threshold=200000,
            ...     verbose=True
            ... )

            >>> # Mode B: All molecules from single molecule database
            >>> means, ref_db, gmm = SM_E.extract_reference_means(
            ...     sm_db,
            ...     reference_photoelectron_threshold=None,
            ...     verbose=True
            ... )

            >>> print("Fixed mean positions:")
            >>> print(f"  Component 0: A_R={means[0,0]:.3f}, A_G={means[0,1]:.3f}")
            >>> print(f"  Component 1: A_R={means[1,0]:.3f}, A_G={means[1,1]:.3f}")
        """
        if verbose:
            logger.info("=" * 60)
            logger.info("Extracting Reference Means (Analytical Approach)")
            logger.info("=" * 60)

        if "photons_accumulated" in data_db.columns and "photoelectrons_accumulated" not in data_db.columns:
            raise ValueError(
                "Old photon accumulation database ('photons_accumulated', the overcounted A_B + A_G + A_R): "
                "rebuild it with build_photon_accumulation_database (photoelectrons_accumulated)."
            )
        if any(c in data_db.columns for c in ("photons", "background_photons")):   # database written before 2026-11-11
            io = getattr(self, "io", None)
            if io is None:
                from pyS3M.IOFunctions import IO_Functions
                io = IO_Functions()
            data_db = io.add_photoelectron_columns(data_db)

        # Detect database type and extract reference data accordingly
        is_photon_accumulation_db = "photoelectrons_accumulated" in data_db.columns

        if is_photon_accumulation_db:
            # Mode A: Photon accumulation database with threshold
            if reference_photoelectron_threshold is None:
                raise ValueError(
                    "reference_photoelectron_threshold must be provided when using photon accumulation database. "
                    "To use all molecules, pass a single molecule database instead."
                )

            if verbose:
                logger.info("Mode: Photon Accumulation Database")
                logger.info(f"Using highest-photoelectron data (threshold: {reference_photoelectron_threshold:,.0f})")

            # Get maximum photoelectrons accumulated for each molecule
            max_photoelectrons_per_mol = (
                data_db.groupby("molecular_index")["photoelectrons_accumulated"]
                .max()
                .reset_index()
            )
            max_photoelectrons_per_mol.columns = ["molecular_index", "max_photoelectrons"]

            if verbose:
                logger.info(f"Total molecules in database: {len(max_photoelectrons_per_mol)}")

            # Filter molecules that reach reference threshold
            qualified_molecules = max_photoelectrons_per_mol[
                max_photoelectrons_per_mol["max_photoelectrons"] >= reference_photoelectron_threshold
            ]

            if verbose:
                n_qualified = len(qualified_molecules)
                n_total = len(max_photoelectrons_per_mol)
                pct_qualified = 100 * n_qualified / n_total
                logger.info(f"Molecules reaching threshold: {n_qualified}/{n_total} ({pct_qualified:.1f}%)")

            if len(qualified_molecules) == 0:
                raise ValueError(
                    f"No molecules reach photoelectron threshold {reference_photoelectron_threshold}. "
                    f"Maximum photoelectrons in dataset: {max_photoelectrons_per_mol['max_photoelectrons'].max():.0f}"
                )

            # Get data at maximum photoelectrons for each qualified molecule
            reference_data = []
            for mol_idx in qualified_molecules["molecular_index"]:
                mol_data = data_db[data_db["molecular_index"] == mol_idx]
                # Get row with maximum photoelectrons
                max_row = mol_data.loc[mol_data["photoelectrons_accumulated"].idxmax()]
                reference_data.append(max_row)

            reference_df = pd.DataFrame(reference_data).reset_index(drop=True)
            photoelectron_column = "photoelectrons_accumulated"  # For later reference

        else:
            # Mode B: Single molecule database
            if verbose:
                logger.info("Mode: Single Molecule Database")
                if reference_photoelectron_threshold is not None:
                    logger.info(f"Filtering molecules with photoelectrons >= {reference_photoelectron_threshold:,.0f}")
                else:
                    logger.info("Using all molecules (no photoelectron threshold)")

            # Check required columns
            required_cols = ["A_R", "A_G", "A_B"]
            missing_cols = [col for col in required_cols if col not in data_db.columns]
            if missing_cols:
                raise ValueError(
                    f"Single molecule database missing required columns: {missing_cols}. "
                    f"Available columns: {list(data_db.columns)}"
                )

            # Apply photoelectron threshold if provided
            if reference_photoelectron_threshold is not None:
                if "photoelectrons" not in data_db.columns:
                    raise ValueError(
                        "reference_photoelectron_threshold provided but 'photoelectrons' column not found in database. "
                        "Either provide a database with 'photoelectrons' column or set reference_photoelectron_threshold=None."
                    )
                reference_df = data_db[data_db["photoelectrons"] >= reference_photoelectron_threshold].copy()

                if verbose:
                    n_qualified = len(reference_df)
                    n_total = len(data_db)
                    pct_qualified = 100 * n_qualified / n_total if n_total > 0 else 0
                    logger.info(f"Total molecules in database: {n_total}")
                    logger.info(f"Molecules passing threshold: {n_qualified}/{n_total} ({pct_qualified:.1f}%)")

                if len(reference_df) == 0:
                    raise ValueError(
                        f"No molecules have photoelectrons >= {reference_photoelectron_threshold}. "
                        f"Maximum photoelectrons in dataset: {data_db['photoelectrons'].max():.0f}"
                    )
            else:
                reference_df = data_db.copy()
                if verbose:
                    logger.info(f"Total molecules in database: {len(reference_df)}")

            photoelectron_column = "photoelectrons" if "photoelectrons" in reference_df.columns else None

        if verbose:
            logger.info(f"\nFitting {n_components}-component Gaussian Mixture Model...")
            logger.info(f"  Covariance type: {covariance_type}")
            logger.info(f"  Features: A_R, A_G")

        # Prepare data for GMM: (A_R, A_G) coordinates
        X = reference_df[["A_R", "A_G"]].values

        # Apply error-based weighting if available
        # Weight by inverse of uncertainty in A_R and A_G measurements
        # We do this by replicating high-precision samples more times

        # Check if error columns exist in the database
        has_error_columns = "A_R_err" in reference_df.columns and "A_G_err" in reference_df.columns

        if has_error_columns:
            # Use pre-calculated errors from database
            sigma_A_R = reference_df["A_R_err"].values
            sigma_A_G = reference_df["A_G_err"].values

            # Combined uncertainty (quadrature sum for 2D Gaussian)
            sigma_combined = np.sqrt(sigma_A_R**2 + sigma_A_G**2)

            # Weight = 1/sigma (inverse of uncertainty)
            weights = 1.0 / (sigma_combined + 1e-10)  # Small constant to avoid infinity
            weights = weights / weights.mean()  # Normalize

            # Store for later use in EM
            photoelectrons = reference_df[photoelectron_column].values if photoelectron_column is not None else None
            A_R = reference_df["A_R"].values
            A_G = reference_df["A_G"].values
            A_B = reference_df["A_B"].values if "A_B" in reference_df.columns else None

        elif photoelectron_column is not None:
            # Fall back to calculating uncertainty from photoelectron statistics
            photoelectrons = reference_df[photoelectron_column].values

            # Get amplitude ratios
            A_R = reference_df["A_R"].values
            A_G = reference_df["A_G"].values
            A_B = reference_df["A_B"].values

            # Uncertainty in A_R = R/(R+G+B) using error propagation
            # σ(A_R) ≈ sqrt(A_R*(1-A_R)/total_photoelectrons) for Poisson statistics

            sigma_A_R = np.sqrt(A_R * (1 - A_R) / (photoelectrons + 1))  # +1 to avoid division by zero
            sigma_A_G = np.sqrt(A_G * (1 - A_G) / (photoelectrons + 1))

            # Combined uncertainty (quadrature sum for 2D Gaussian)
            sigma_combined = np.sqrt(sigma_A_R**2 + sigma_A_G**2)

            # Weight = 1/sigma (inverse of uncertainty)
            weights = 1.0 / (sigma_combined + 1e-10)  # Small constant to avoid infinity
            weights = weights / weights.mean()  # Normalize
        else:
            # No weighting information available
            weights = None
            photoelectrons = None
            A_R = None
            A_G = None
            A_B = None
            sigma_combined = None

        if weights is not None:

            # Convert weights to integer replication counts
            # Use factor of 5 for MLE (faster) and factor of 10 for EM
            # MLE is more expensive per iteration so we use fewer replications
            replication_factor = 5 if fit_type.upper() == "MLE" else 10
            replication_counts = np.round(weights * replication_factor).astype(int)
            replication_counts = np.maximum(replication_counts, 1)  # At least 1 copy

            # Create weighted dataset by replicating samples
            X_weighted = np.repeat(X, replication_counts, axis=0)

            # For MLE, cap total samples at 3000 to keep optimization tractable
            if fit_type.upper() == "MLE" and len(X_weighted) > 3000:
                # Downsample while preserving weight distribution
                downsample_ratio = 3000 / len(X_weighted)
                new_rep_counts = np.maximum(np.round(replication_counts * downsample_ratio).astype(int), 1)
                X_weighted = np.repeat(X, new_rep_counts, axis=0)
                if verbose:
                    logger.info(f"  MLE optimization: downsampled to {len(X_weighted)} samples for tractability")

            if verbose:
                if has_error_columns:
                    logger.info(f"  Using error-weighted GMM fitting (1/σ weighting from A_R_err, A_G_err columns)")
                else:
                    logger.info(f"  Using error-weighted GMM fitting (1/σ weighting calculated from photoelectron statistics)")
                if photoelectrons is not None:
                    logger.info(f"    Photoelectron range: {photoelectrons.min():.0f} - {photoelectrons.max():.0f}")
                logger.info(f"    Uncertainty range: σ={sigma_combined.min():.4f} - {sigma_combined.max():.4f}")
                logger.info(f"    Weight range: {weights.min():.2f} - {weights.max():.2f}")
                logger.info(f"    Replication range: {replication_counts.min()} - {replication_counts.max()} copies")
                logger.info(f"    Original samples: {len(X)}, Weighted samples: {len(X_weighted)}")

            X_fit = X_weighted
        else:
            if verbose:
                logger.info(f"  Using uniform weights (no photoelectron column available)")
            X_fit = X

        # Validate fit_type parameter
        if fit_type.upper() not in ["MLE", "EM"]:
            raise ValueError(f"fit_type must be 'MLE' or 'EM', got '{fit_type}'")

        # Initialize means using histogram-based peak finding
        # This gives better initial guesses than k-means++
        from scipy.signal import find_peaks

        def fit_histogram_gaussians(data, n_peaks=2):
            """Fit Gaussians to histogram peaks to get initial guesses."""
            # Create histogram
            hist, bin_edges = np.histogram(data, bins=50, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # Find peaks in histogram
            peaks, properties = find_peaks(hist, height=0, distance=5)

            if len(peaks) >= n_peaks:
                # Sort by height and take top n_peaks
                peak_heights = hist[peaks]
                top_peak_indices = np.argsort(peak_heights)[-n_peaks:]
                peak_positions = bin_centers[peaks[top_peak_indices]]
                return np.sort(peak_positions)
            else:
                # Fallback: use quantiles
                return np.quantile(data, np.linspace(0.2, 0.8, n_peaks))

        # Get initial means from histograms
        A_R_peaks = fit_histogram_gaussians(X[:, 0], n_components)
        A_G_peaks = fit_histogram_gaussians(X[:, 1], n_components)

        # Combine to form initial 2D means
        # Match peaks: if A_R high, A_G should be low (and vice versa)
        initial_means = np.zeros((n_components, 2))
        if n_components == 2:
            # Sort A_R peaks ascending, A_G peaks descending
            A_R_sorted = np.sort(A_R_peaks)
            A_G_sorted = np.sort(A_G_peaks)[::-1]
            initial_means[:, 0] = A_R_sorted
            initial_means[:, 1] = A_G_sorted
        else:
            # For n_components != 2, use a simpler approach
            initial_means[:, 0] = A_R_peaks
            initial_means[:, 1] = A_G_peaks

        if verbose:
            logger.info(f"  Histogram-based initialization:")
            for i in range(n_components):
                logger.info(f"    Component {i}: A_R={initial_means[i, 0]:.4f}, A_G={initial_means[i, 1]:.4f}")

        # Fit GMM using selected method
        if fit_type.upper() == "MLE":
            # Use helper function for MLE fitting
            means_opt, covariances_opt, weights_opt, converged = self._fit_gmm_mle(
                X_fit,
                initial_means,
                n_components,
                covariance_type=covariance_type,
                max_iter=500,
                verbose=verbose,
            )

            # Create GMM object with optimized parameters for compatibility
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type=covariance_type,
                random_state=random_state,
            )
            gmm.means_ = means_opt
            gmm.covariances_ = covariances_opt
            gmm.weights_ = weights_opt
            gmm.converged_ = converged

            # Compute precisions_cholesky_ for sklearn compatibility
            # This is needed for predict() and predict_proba() methods
            gmm.precisions_cholesky_ = np.empty((n_components, 2, 2))
            for k in range(n_components):
                cov_chol = np.linalg.cholesky(covariances_opt[k])
                gmm.precisions_cholesky_[k] = np.linalg.solve(cov_chol, np.eye(2)).T

            # Predict labels and posterior probabilities on original unweighted data
            labels = gmm.predict(X)
            posteriors = gmm.predict_proba(X)

            if verbose:
                logger.info(f"  Converged: {gmm.converged_}")
                logger.info(f"  BIC: {gmm.bic(X):.2f}")
                logger.info(f"  AIC: {gmm.aic(X):.2f}")
                logger.info("\nGMM Component Parameters:")
                for i in range(n_components):
                    logger.info(f"  Component {i}:")
                    logger.info(f"    Mean A_R: {gmm.means_[i, 0]:.4f}")
                    logger.info(f"    Mean A_G: {gmm.means_[i, 1]:.4f}")
                    logger.info(f"    Weight: {gmm.weights_[i]:.4f}")
                    n_assigned = np.sum(labels == i)
                    logger.info(f"    Molecules assigned: {n_assigned} ({100*n_assigned/len(labels):.1f}%)")

        elif fit_type.upper() == "EM":
            # Use helper function for EM fitting
            means_opt, covariances_opt, weights_opt, converged = self._fit_gmm_em(
                X,
                initial_means,
                n_components,
                covariance_type=covariance_type,
                max_iter=100,
                n_reweighting_iterations=2,
                photoelectrons=photoelectrons,
                A_R=A_R,
                A_G=A_G,
                has_error_columns=has_error_columns,
                sigma_A_R=sigma_A_R if has_error_columns else None,
                sigma_A_G=sigma_A_G if has_error_columns else None,
                verbose=verbose,
            )

            # Create GMM object with optimized parameters for compatibility
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type=covariance_type,
                random_state=random_state,
            )
            gmm.means_ = means_opt
            gmm.covariances_ = covariances_opt
            gmm.weights_ = weights_opt
            gmm.converged_ = converged

            # Compute precisions_cholesky_ for sklearn compatibility
            gmm.precisions_cholesky_ = np.empty((n_components, 2, 2))
            for k in range(n_components):
                cov_chol = np.linalg.cholesky(covariances_opt[k])
                gmm.precisions_cholesky_[k] = np.linalg.solve(cov_chol, np.eye(2)).T

            # Predict labels and posterior probabilities on original data
            labels = gmm.predict(X)
            posteriors = gmm.predict_proba(X)

            if verbose:
                logger.info(f"  Converged: {gmm.converged_}")
                logger.info(f"  BIC: {gmm.bic(X):.2f}")
                logger.info(f"  AIC: {gmm.aic(X):.2f}")
                logger.info("\nGMM Component Parameters:")
                for i in range(n_components):
                    logger.info(f"  Component {i}:")
                    logger.info(f"    Mean A_R: {gmm.means_[i, 0]:.4f}")
                    logger.info(f"    Mean A_G: {gmm.means_[i, 1]:.4f}")
                    logger.info(f"    Weight: {gmm.weights_[i]:.4f}")
                    n_assigned = np.sum(labels == i)
                    logger.info(f"    Molecules assigned: {n_assigned} ({100*n_assigned/len(labels):.1f}%)")


        # Build reference database - handle both modes
        ref_db_dict = {
            "true_label": labels,
            "A_R_ref": reference_df["A_R"].values,
            "A_G_ref": reference_df["A_G"].values,
            "A_B_ref": reference_df["A_B"].values,
        }
        for i in range(n_components):
            ref_db_dict[f"posterior_prob_{i}"] = posteriors[:, i]

        # Add molecular_index if available
        if "molecular_index" in reference_df.columns:
            ref_db_dict["molecular_index"] = reference_df["molecular_index"].values

        # Add photoelectron information based on database type
        if photoelectron_column is not None:
            if is_photon_accumulation_db:
                ref_db_dict["max_photoelectrons"] = reference_df[photoelectron_column].values
            else:
                ref_db_dict["photoelectrons"] = reference_df[photoelectron_column].values

        reference_db = pd.DataFrame(ref_db_dict)

        # Add FOV tracking if available
        if "fov_index" in reference_df.columns:
            reference_db["fov_index"] = reference_df["fov_index"].values
        if "fov_name" in reference_df.columns:
            reference_db["fov_name"] = reference_df["fov_name"].values

        if verbose:
            logger.info("\n" + "=" * 60)
            logger.info("Reference Means Extraction Complete!")
            logger.info("=" * 60)
            logger.info(f"\nExtracted {n_components} fixed mean positions:")
            for i in range(n_components):
                logger.info(f"  Component {i}: A_R={gmm.means_[i, 0]:.4f}, A_G={gmm.means_[i, 1]:.4f}")

            # Plot histograms with fitted means
            try:
                from pyS3M.PlottingBase import AnalysisPlotter

                plotter = AnalysisPlotter()
                fig, (ax1, ax2) = plotter.two_column_plot(nrows=1, ncols=2, height=3)

                # Define colors for each component
                colors = ['red', 'green', 'blue', 'orange', 'purple'][:n_components]

                # Plot A_R histogram
                ax1.hist(X[:, 0], bins=50, alpha=0.3, color='gray', label='All data')
                for i in range(n_components):
                    component_mask = labels == i
                    ax1.hist(X[component_mask, 0], bins=50, alpha=0.5,
                            color=colors[i], label=f'Component {i}')
                    ax1.axvline(gmm.means_[i, 0], color=colors[i], linestyle='--',
                              linewidth=2, label=f'Mean {i}: {gmm.means_[i, 0]:.3f}')
                ax1.set_xlabel('A_R')
                ax1.set_ylabel('Count')
                ax1.set_title('A_R Distribution with Fitted Means')
                ax1.legend(fontsize=8)
                ax1.grid(True, alpha=0.3)

                # Plot A_G histogram
                ax2.hist(X[:, 1], bins=50, alpha=0.3, color='gray', label='All data')
                for i in range(n_components):
                    component_mask = labels == i
                    ax2.hist(X[component_mask, 1], bins=50, alpha=0.5,
                            color=colors[i], label=f'Component {i}')
                    ax2.axvline(gmm.means_[i, 1], color=colors[i], linestyle='--',
                              linewidth=2, label=f'Mean {i}: {gmm.means_[i, 1]:.3f}')
                ax2.set_xlabel('A_G')
                ax2.set_ylabel('Count')
                ax2.set_title('A_G Distribution with Fitted Means')
                ax2.legend(fontsize=8)
                ax2.grid(True, alpha=0.3)

                _safe_tight_layout(fig)
                plotter.save_or_show(fig, save_path=None)  # Show only

                logger.info("\n  (Close the plot window to continue)")

            except ImportError:
                logger.warning("\n  (Plotting skipped - PlottingBase not available)")
            except Exception as e:
                logger.warning(f"\n  (Plotting skipped - error: {e})")

        return gmm.means_, reference_db, gmm

    def fit_covariances_fixed_means(
        self,
        X: NDArray[np.float64],
        fixed_means: NDArray[np.float64],
        fit_type: str = "EM",
        max_iter: int = 100,
        tol: float = 1e-6,
        verbose: bool = False,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], bool]:
        """
        Fit covariance matrices with fixed means using either MLE or EM algorithm.

        This is the core of the analytical approach: given fixed mean positions
        from high-photoelectron data, fit covariances at a specific photoelectron level to
        characterize measurement uncertainty.

        Args:
            X (np.ndarray): Data points, shape (n_samples, n_features)
                Typically [:, 0] = A_R, [:, 1] = A_G
            fixed_means (np.ndarray): Fixed mean positions, shape (n_components, n_features)
            fit_type (str): Fitting method - "MLE" or "EM" (default: "EM")
                - "MLE": Direct maximum likelihood optimization (more robust)
                - "EM": Expectation-Maximization algorithm (faster)
            max_iter (int): Maximum iterations (default: 100)
            tol (float): Convergence tolerance for log-likelihood (default: 1e-6)
            verbose (bool): Print iteration progress

        Returns:
            tuple: (covariances, weights, converged)
                - covariances (np.ndarray): Shape (n_components, n_features, n_features)
                - weights (np.ndarray): Component weights, shape (n_components,)
                - converged (bool): Whether fitting converged

        Example:
            >>> # Get reference means from high-photoelectron data
            >>> means, ref_db, _ = SM_E.extract_reference_means(pa_db, threshold=200000)
            >>>
            >>> # Fit covariances at lower photoelectron level using MLE
            >>> low_pe_data = pa_db[pa_db['photoelectrons_accumulated'].between(5000, 6000)]
            >>> X = low_pe_data[['A_R', 'A_G']].values
            >>> cov, weights, converged = SM_E.fit_covariances_fixed_means(X, means, fit_type="MLE")
        """
        n_samples, n_features = X.shape
        n_components = len(fixed_means)

        if fit_type.upper() == "MLE":
            # Use MLE optimization for fixed means
            from scipy.optimize import minimize

            def pack_params_fixed_means(covariances, weights):
                """Pack covariances and weights (means are fixed)."""
                params = []
                # Covariances (upper triangular elements)
                for k in range(n_components):
                    cov = covariances[k]
                    params.extend([cov[0, 0], cov[0, 1], cov[1, 1]])
                # Weights (n-1 to maintain sum=1 constraint)
                params.extend(weights[:-1])
                return np.array(params)

            def unpack_params_fixed_means(params):
                """Unpack covariances and weights."""
                idx = 0
                # Covariances
                covariances = []
                for k in range(n_components):
                    cov = np.array([
                        [params[idx], params[idx + 1]],
                        [params[idx + 1], params[idx + 2]]
                    ])
                    covariances.append(cov)
                    idx += 3
                # Weights
                weights = np.zeros(n_components)
                weights[:-1] = params[idx:]
                weights[-1] = 1.0 - weights[:-1].sum()
                return np.array(covariances), weights

            def negative_log_likelihood_fixed(params):
                """Negative log-likelihood with fixed means."""
                try:
                    covariances, weights = unpack_params_fixed_means(params)

                    # Check validity
                    for k in range(n_components):
                        eigvals = np.linalg.eigvalsh(covariances[k])
                        if np.any(eigvals <= 0):
                            return 1e10
                    if np.any(weights <= 0) or np.any(weights >= 1):
                        return 1e10

                    # Calculate log-likelihood with fixed means
                    log_probs = np.zeros((len(X), n_components))
                    for k in range(n_components):
                        mvn = multivariate_normal(mean=fixed_means[k], cov=covariances[k])
                        log_probs[:, k] = mvn.logpdf(X) + np.log(weights[k])

                    log_probs_max = log_probs.max(axis=1, keepdims=True)
                    log_likelihood = np.sum(log_probs_max + np.log(np.exp(log_probs - log_probs_max).sum(axis=1)))

                    return -log_likelihood

                except (np.linalg.LinAlgError, ValueError):
                    return 1e10

            # Initialize covariances and weights
            covariances_init = np.array([np.eye(n_features) for _ in range(n_components)])
            weights_init = np.ones(n_components) / n_components

            params_init = pack_params_fixed_means(covariances_init, weights_init)

            if verbose:
                logger.info(f"  Running MLE optimization with fixed means...")

            # Optimize
            result = minimize(
                negative_log_likelihood_fixed,
                params_init,
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': tol}
            )

            covariances, weights = unpack_params_fixed_means(result.x)
            converged = result.success

            return covariances, weights, converged

        elif fit_type.upper() == "EM":
            # EM algorithm for fitting covariances with fixed means
            # Initialize covariances and weights
            covariances = [np.eye(n_features) for _ in range(n_components)]
            weights = np.ones(n_components) / n_components

            log_likelihood_old = -np.inf
            converged = False

            for iteration in range(max_iter):
                # E-step: Calculate responsibilities
                log_probs = np.zeros((n_samples, n_components))

                for k in range(n_components):
                    try:
                        mvn = multivariate_normal(mean=fixed_means[k], cov=covariances[k])
                        log_probs[:, k] = mvn.logpdf(X) + np.log(weights[k])
                    except np.linalg.LinAlgError:
                        # Singular covariance - add regularization
                        cov_reg = covariances[k] + 1e-6 * np.eye(n_features)
                        mvn = multivariate_normal(mean=fixed_means[k], cov=cov_reg)
                        log_probs[:, k] = mvn.logpdf(X) + np.log(weights[k])

                # Normalize to get responsibilities (log-sum-exp trick for stability)
                log_probs_max = log_probs.max(axis=1, keepdims=True)
                probs = np.exp(log_probs - log_probs_max)
                responsibilities = probs / probs.sum(axis=1, keepdims=True)

                # Calculate log-likelihood
                log_likelihood = np.sum(np.log(probs.sum(axis=1)) + log_probs_max.flatten())

                # Check convergence
                if abs(log_likelihood - log_likelihood_old) < tol:
                    converged = True
                    if verbose:
                        logger.info(f"  Converged at iteration {iteration+1}")
                    break
                log_likelihood_old = log_likelihood

                # M-step: Update weights and covariances (NOT means!)
                weights = responsibilities.mean(axis=0)

                for k in range(n_components):
                    # Weighted covariance around fixed mean
                    centered = X - fixed_means[k]
                    weighted = responsibilities[:, k:k+1] * centered

                    cov_k = (weighted.T @ centered) / responsibilities[:, k].sum()

                    # Regularize to ensure positive definite
                    min_eig = np.linalg.eigvalsh(cov_k)[0]
                    if min_eig < 1e-6:
                        cov_k += (1e-6 - min_eig) * np.eye(n_features)

                    covariances[k] = cov_k

            if not converged and verbose:
                logger.info(f"  Warning: Did not converge after {max_iter} iterations")

            return np.array(covariances), weights, converged

    def fit_covariances_fixed_means_mestimator(
        self,
        X: NDArray[np.float64],
        fixed_means: NDArray[np.float64],
        reference_covariances: NDArray[np.float64] | None = None,
        estimator_type: str = "tukey",
        max_iter: int = 20,
        tol: float = 1e-4,
        verbose: bool = False,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], list[NDArray[np.float64]]]:
        """
        Robustly fit covariances using M-estimators with iterative re-weighting.

        This method handles outliers in multimodal mixture data by:
        1. Hard-assigning points to nearest component
        2. Using M-estimator weight functions to down-weight outliers
        3. Iteratively re-fitting covariances with weighted data

        M-estimators provide soft rejection of outliers (gradual down-weighting)
        rather than hard thresholding, making them robust while retaining efficiency.

        Args:
            X (np.ndarray): Data matrix, shape (n_samples, n_features)
            fixed_means (np.ndarray): Fixed component means, shape (n_components, n_features)
            reference_covariances (np.ndarray, optional): Reference covariances from high-photoelectron
                data for comparison/diagnostics, shape (n_components, n_features, n_features)
            estimator_type (str): Type of M-estimator to use:
                - "huber": Huber loss (moderate robustness, c=1.345)
                - "tukey": Tukey bisquare (aggressive robustness, c=4.685) [default]
            max_iter (int): Maximum number of iterations for re-weighting (default 20)
            tol (float): Convergence tolerance on covariance change (default 1e-4)
            verbose (bool): Print diagnostic information

        Returns:
            covariances (np.ndarray): Robust covariances, shape (n_components, n_features, n_features)
            weights (np.ndarray): Component weights based on assigned counts, shape (n_components,)
            point_weights (list): Per-point weights for each component (for diagnostics)

        Example:
            >>> # Get reference means from high-photoelectron fit
            >>> ref_means, ref_db, gmm = SM_E.extract_reference_means(high_pe_data)
            >>> ref_covs = gmm.covariances_
            >>>
            >>> # Robustly fit at lower photoelectron level using M-estimators
            >>> low_pe_data = pa_db[pa_db['photoelectrons'].between(5000, 6000)]
            >>> X = low_pe_data[['A_R', 'A_G']].values
            >>> cov, weights, pt_wts = SM_E.fit_covariances_fixed_means_mestimator(
            ...     X, ref_means, reference_covariances=ref_covs, estimator_type="tukey"
            ... )
        """
        from scipy.spatial.distance import cdist
        from scipy.stats import norm

        n_samples, n_features = X.shape
        n_components = len(fixed_means)

        if verbose:
            logger.info(f"  M-estimator robust fitting (type={estimator_type}, max_iter={max_iter})")

        # Hard assignment to nearest component (Euclidean distance)
        distances = cdist(X, fixed_means, metric='euclidean')
        assignments = np.argmin(distances, axis=1)

        if verbose:
            for k in range(n_components):
                n_k = (assignments == k).sum()
                logger.info(f"  Component {k}: {n_k} points assigned")

        # M-estimator weight functions
        def huber_weight(r, c=1.345):
            """Huber weight function (moderate robustness)"""
            return np.where(np.abs(r) <= c, 1.0, c / np.abs(r))

        def tukey_weight(r, c=4.685):
            """Tukey bisquare weight function (aggressive robustness)"""
            return np.where(np.abs(r) <= c, (1 - (r/c)**2)**2, 0.0)

        weight_func = tukey_weight if estimator_type == "tukey" else huber_weight

        # Initialize covariances with standard MLE per component
        covariances = []
        for k in range(n_components):
            X_k = X[assignments == k]
            if len(X_k) > n_features:
                diff = X_k - fixed_means[k]
                cov_k = (diff.T @ diff) / len(X_k)
            else:
                # Not enough points
                if reference_covariances is not None:
                    cov_k = reference_covariances[k]
                else:
                    cov_k = np.eye(n_features) * 0.001
            covariances.append(cov_k)

        covariances = np.array(covariances)

        # Iterative re-weighting
        all_point_weights = []

        for iteration in range(max_iter):
            old_covs = covariances.copy()
            iteration_weights = []

            for k in range(n_components):
                X_k = X[assignments == k]
                n_k = len(X_k)

                if n_k <= n_features:
                    # Too few points, keep current covariance
                    iteration_weights.append(np.ones(n_k))
                    continue

                # Calculate Mahalanobis distances
                try:
                    inv_cov = np.linalg.inv(covariances[k])
                except np.linalg.LinAlgError:
                    # Singular matrix, regularize
                    covariances[k] += np.eye(n_features) * 1e-6
                    inv_cov = np.linalg.inv(covariances[k])

                diff = X_k - fixed_means[k]
                mahal = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))

                # Normalize by robust scale estimate (median absolute deviation)
                scale = np.median(mahal) / np.sqrt(norm.ppf(0.75))  # Chi distribution with df=n_features
                if scale < 1e-10:  # Avoid division by zero
                    scale = 1.0
                standardized = mahal / scale

                # Calculate M-estimator weights
                weights_k = weight_func(standardized)

                # Re-fit covariance with weights
                weighted_diff = diff * np.sqrt(weights_k[:, np.newaxis])
                weight_sum = weights_k.sum()

                if weight_sum > 0:
                    covariances[k] = (weighted_diff.T @ weighted_diff) / weight_sum

                iteration_weights.append(weights_k)

                if verbose and iteration == max_iter - 1:  # Print on last iteration
                    n_downweighted = (weights_k < 0.5).sum()
                    pct = 100 * n_downweighted / len(weights_k)
                    det_k = np.linalg.det(covariances[k])
                    logger.info(f"  Component {k}: {n_downweighted}/{len(weights_k)} heavily downweighted (<0.5 weight, {pct:.1f}%)")
                    logger.info(f"    Covariance determinant: {det_k:.6f}")

                    if reference_covariances is not None:
                        det_ref = np.linalg.det(reference_covariances[k])
                        ratio = det_k / det_ref
                        logger.info(f"    Ratio to reference: {ratio:.2f}x")

            all_point_weights = iteration_weights

            # Check convergence (Frobenius norm of covariance change)
            max_change = np.max([np.linalg.norm(covariances[k] - old_covs[k], 'fro')
                                 for k in range(n_components)])

            if verbose and (iteration == 0 or iteration == max_iter - 1 or max_change < tol):
                logger.info(f"  Iteration {iteration + 1}: max covariance change = {max_change:.6f}")

            if max_change < tol:
                if verbose:
                    logger.info(f"  Converged at iteration {iteration + 1}")
                break

        # Compute final component weights (based on number of assigned points)
        weights = np.array([np.sum(assignments == k) for k in range(n_components)], dtype=float)
        weights /= weights.sum()

        return covariances, weights, all_point_weights

    def calculate_analytical_misidentification(
        self,
        fixed_means: NDArray[np.float64],
        covariances: NDArray[np.float64],
        weights: NDArray[np.float64],
        n_samples: int = 10000,
        random_state: int = 42,
    ) -> dict[str, Any]:
        """
        Analytically calculate misidentification rates from Gaussian overlap.

        For 2-component case: Uses Monte Carlo integration to calculate the probability
        that a sample from component i is classified as component j.

        Method:
        1. Generate synthetic samples from each component
        2. Classify using Bayes decision rule (posterior probabilities)
        3. Calculate confusion matrix and error rates

        This is more accurate than empirical measurement because it characterizes
        the noise model itself, not specific noisy measurements.

        Args:
            fixed_means (np.ndarray): Mean positions, shape (n_components, n_features)
            covariances (np.ndarray): Covariance matrices, shape (n_components, n_features, n_features)
            weights (np.ndarray): Component weights, shape (n_components,)
            n_samples (int): Number of Monte Carlo samples per component (default: 10,000)
            random_state (int): Random seed for reproducibility

        Returns:
            dict: Misidentification statistics with keys:
                - 'confusion_matrix': shape (n_components, n_components); entry [i, j] = P(classified as j | true component i)
                - 'accuracy_per_component': shape (n_components,); probability of correct classification for each component
                - 'overall_accuracy': float; weighted average accuracy
                - 'error_rate_per_component': shape (n_components,); probability of misclassification for each component
                - 'overall_error_rate': float; weighted average error rate

        Example:
            >>> # After fitting covariances at specific photoelectron level
            >>> cov, weights, _ = SM_E.fit_covariances_fixed_means(X, fixed_means)
            >>> stats = SM_E.calculate_analytical_misidentification(
            ...     fixed_means, cov, weights, n_samples=10000
            ... )
            >>> print(f"Overall accuracy: {stats['overall_accuracy']:.3f}")
            >>> print(f"Confusion matrix:\\n{stats['confusion_matrix']}")
        """
        np.random.seed(random_state)

        n_components = len(fixed_means)
        n_features = fixed_means.shape[1]

        # Generate synthetic samples from each component
        all_samples = []
        all_true_labels = []

        for k in range(n_components):
            # Generate samples from component k
            samples = np.random.multivariate_normal(
                mean=fixed_means[k],
                cov=covariances[k],
                size=n_samples
            )
            all_samples.append(samples)
            all_true_labels.append(np.full(n_samples, k))

        # Combine all samples
        X = np.vstack(all_samples)
        y_true = np.concatenate(all_true_labels)

        # Classify using Bayes decision rule
        log_probs = np.zeros((len(X), n_components))

        for k in range(n_components):
            mvn = multivariate_normal(mean=fixed_means[k], cov=covariances[k])
            log_probs[:, k] = mvn.logpdf(X) + np.log(weights[k])

        # Predict labels (argmax of posterior)
        y_pred = np.argmax(log_probs, axis=1)

        # Calculate confusion matrix
        # Entry [i, j] = number of samples from component i classified as j
        conf_matrix_counts = np.zeros((n_components, n_components), dtype=int)
        for i in range(n_components):
            mask = y_true == i
            for j in range(n_components):
                conf_matrix_counts[i, j] = np.sum(y_pred[mask] == j)

        # Normalize to get probabilities (rows sum to 1)
        conf_matrix = conf_matrix_counts.astype(float) / n_samples

        # Calculate accuracies
        accuracy_per_component = np.diag(conf_matrix)
        overall_accuracy = np.average(accuracy_per_component, weights=weights)

        # Calculate error rates
        error_rate_per_component = 1.0 - accuracy_per_component
        overall_error_rate = 1.0 - overall_accuracy

        return {
            'confusion_matrix': conf_matrix,
            'accuracy_per_component': accuracy_per_component,
            'overall_accuracy': overall_accuracy,
            'error_rate_per_component': error_rate_per_component,
            'overall_error_rate': overall_error_rate,
        }

    def analyze_photon_dependent_misidentification_analytical(
        self,
        photon_accumulation_db: pd.DataFrame,
        fixed_means: NDArray[np.float64],
        reference_db: pd.DataFrame,
        photoelectron_bins: NDArray[np.float64] | list[float],
        reference_covariances: NDArray[np.float64] | None = None,
        use_earliest_entry: bool = True,
        n_mc_samples: int = 10000,
        estimator_type: str = "tukey",
        max_iter: int = 20,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Analytically analyze misidentification rates across photoelectron bins using robust M-estimator fitting.

        Fits covariances at each photoelectron level with fixed means using M-estimators (Huber or Tukey),
        then analytically calculates misidentification rates from distribution overlap using Monte Carlo
        integration.

        This approach:
        1. Separates signal (means - dye properties) from noise (covariances - measurement uncertainty)
        2. Robustly characterizes the noise model at each photoelectron level (soft down-weighting of outliers)
        3. Provides stable and interpretable error rate predictions

        Workflow:

        1. For each photon bin, extract data at that photoelectron level, robustly fit
           covariances with fixed means using M-estimators, and analytically
           calculate misidentification from overlap
        2. Return summary of error rates vs photoelectron count

        Args:
            photon_accumulation_db (pd.DataFrame): Photon accumulation database
            fixed_means (np.ndarray): Fixed mean positions from extract_reference_means(),
                shape (n_components, 2) for [A_R, A_G]
            reference_db (pd.DataFrame): Reference molecules from extract_reference_means()
            photoelectron_bins (array-like): Photoelectron bin edges (e.g., [1000, 2000, 5000, 10000])
            reference_covariances (np.ndarray, optional): Reference covariances from high-photoelectron fit
                for comparison/diagnostics. Can be obtained from ``gmm.covariances_`` after extract_reference_means()
            use_earliest_entry (bool): If True, use earliest crossing into each bin.
                If False, use midpoint of bin. (default: True)
            n_mc_samples (int): Monte Carlo samples for analytical error calculation (default: 10,000)
            estimator_type (str): M-estimator type: "huber" (moderate) or "tukey" (aggressive, default)
            max_iter (int): Maximum iterations for M-estimator re-weighting (default: 20)
            verbose (bool): Print progress and statistics

        Returns:
            pd.DataFrame: Summary database with columns:
                - photoelectron_bin_min, photoelectron_bin_max: Bin edges
                - n_molecules: Number of molecules in bin
                - converged: Whether fitting converged
                - overall_accuracy: Predicted classification accuracy (analytical)
                - overall_error_rate: Predicted error rate (analytical)
                - component_0_accuracy, component_1_accuracy: Per-component accuracies
                - component_0_error_rate, component_1_error_rate: Per-component error rates
                - confusion_matrix_00, confusion_matrix_01, etc.: Full confusion matrix
                - cov_0_AR_AR, cov_0_AR_AG, cov_0_AG_AG: Fitted covariance components for component 0
                - cov_1_AR_AR, cov_1_AR_AG, cov_1_AG_AG: Fitted covariance components for component 1
                - weight_0, weight_1: Fitted component weights

        Example:
            >>> # Extract fixed means from high-photoelectron data
            >>> means, ref_db, gmm = SM_E.extract_reference_means(pa_db, threshold=200000)
            >>> ref_covs = gmm.covariances_  # Get reference covariances
            >>>
            >>> # Analyze error rates across photoelectron levels using M-estimators
            >>> photoelectron_bins = np.logspace(3, 5, 11)  # 1k to 100k photoelectrons
            >>> summary = SM_E.analyze_photon_dependent_misidentification_analytical(
            ...     pa_db, means, ref_db, photoelectron_bins,
            ...     reference_covariances=ref_covs,
            ...     estimator_type="tukey",  # Aggressive outlier down-weighting
            ...     verbose=True
            ... )
            >>>
            >>> # Plot results
            >>> plt.plot(summary['photoelectron_bin_min'], summary['overall_accuracy'])
            >>> plt.xlabel('Photons')
            >>> plt.ylabel('Predicted Accuracy')
        """
        if verbose:
            logger.info("=" * 60)
            logger.info("Analytical Misidentification Analysis")
            logger.info("=" * 60)
            logger.info(f"Photoelectron bins: {len(photoelectron_bins)-1} bins")
            logger.info(f"  Range: {photoelectron_bins[0]:,.0f} - {photoelectron_bins[-1]:,.0f} photoelectrons")
            logger.info(f"Reference molecules: {len(reference_db)}")
            logger.info(f"Fixed means: {fixed_means.shape[0]} components")
            logger.info("")

        if "photoelectrons_accumulated" not in photon_accumulation_db.columns:
            raise ValueError(
                "photon_accumulation_db has no 'photoelectrons_accumulated' (an old database with "
                "'photons_accumulated'?): rebuild it with build_photon_accumulation_database."
            )

        # Filter photon accumulation data to only include reference molecules
        reference_mol_ids = set(reference_db["molecular_index"].values)
        pa_filtered = photon_accumulation_db[
            photon_accumulation_db["molecular_index"].isin(reference_mol_ids)
        ]

        if verbose:
            logger.info(f"Photon accumulation rows (reference molecules only): {len(pa_filtered)}")

        # Storage for summary results
        all_summaries = []
        n_components = len(fixed_means)

        # Process each photon bin
        for i in range(len(photoelectron_bins) - 1):
            bin_min = photoelectron_bins[i]
            bin_max = photoelectron_bins[i + 1]

            if verbose:
                logger.info(f"\nBin {i+1}/{len(photoelectron_bins)-1}: [{bin_min:,.0f}, {bin_max:,.0f}) photoelectrons...")

            # Get molecules in this bin
            bin_data = pa_filtered[
                (pa_filtered["photoelectrons_accumulated"] >= bin_min)
                & (pa_filtered["photoelectrons_accumulated"] < bin_max)
            ]

            if len(bin_data) == 0:
                if verbose:
                    logger.info(f"  No molecules in this bin, skipping...")
                continue

            # Get one row per molecule (earliest entry or midpoint)
            if use_earliest_entry:
                bin_molecules = (
                    bin_data.sort_values("photoelectrons_accumulated")
                    .groupby("molecular_index")
                    .first()
                    .reset_index()
                )
            else:
                bin_midpoint = (bin_min + bin_max) / 2
                bin_data["dist_to_midpoint"] = np.abs(
                    bin_data["photoelectrons_accumulated"] - bin_midpoint
                )
                bin_molecules = (
                    bin_data.sort_values("dist_to_midpoint")
                    .groupby("molecular_index")
                    .first()
                    .reset_index()
                )
                bin_molecules = bin_molecules.drop(columns=["dist_to_midpoint"])

            n_molecules = len(bin_molecules)
            if verbose:
                logger.info(f"  Molecules in bin: {n_molecules}")

            # Extract A_R, A_G data
            X = bin_molecules[["A_R", "A_G"]].values

            # Step 1: Robustly fit covariances with fixed means using M-estimators
            if verbose:
                logger.info(f"  Fitting covariances (M-estimator: {estimator_type})...")
            covariances, weights, point_weights = self.fit_covariances_fixed_means_mestimator(
                X, fixed_means,
                reference_covariances=reference_covariances,
                estimator_type=estimator_type,
                max_iter=max_iter,
                verbose=False
            )
            converged = True  # M-estimators converge if they finish iterating

            if verbose:
                status = "converged" if converged else "did not converge"
                logger.info(f"  Covariance fitting: {status}")

            # Step 2: Analytically calculate misidentification
            if verbose:
                logger.info(f"  Calculating analytical error rates...")
            stats = self.calculate_analytical_misidentification(
                fixed_means, covariances, weights, n_samples=n_mc_samples, random_state=42
            )

            # Plot distributions for this bin if verbose
            if verbose:
                try:
                    from pyS3M.PlottingBase import AnalysisPlotter

                    # Predict labels for molecules in this bin
                    log_probs = np.zeros((len(X), n_components))
                    for k in range(n_components):
                        mvn = multivariate_normal(mean=fixed_means[k], cov=covariances[k])
                        log_probs[:, k] = mvn.logpdf(X) + np.log(weights[k])
                    labels = np.argmax(log_probs, axis=1)

                    plotter = AnalysisPlotter()
                    fig, (ax1, ax2) = plotter.two_column_plot(nrows=1, ncols=2, height=3)

                    colors = ['red', 'green', 'blue', 'orange', 'purple'][:n_components]

                    # Plot A_R histogram
                    ax1.hist(X[:, 0], bins=30, alpha=0.3, color='gray', label='All data')
                    for k in range(n_components):
                        component_mask = labels == k
                        if component_mask.sum() > 0:
                            ax1.hist(X[component_mask, 0], bins=30, alpha=0.5,
                                    color=colors[k], label=f'Component {k}')
                        ax1.axvline(fixed_means[k, 0], color=colors[k], linestyle='--',
                                  linewidth=2, label=f'Mean {k}: {fixed_means[k, 0]:.3f}')
                    ax1.set_xlabel('A_R')
                    ax1.set_ylabel('Count')
                    ax1.set_title(f'Bin {i+1}: [{bin_min:.0f}, {bin_max:.0f}) photoelectrons\nA_R Distribution (Acc: {stats["overall_accuracy"]:.3f})')
                    ax1.legend(fontsize=8)
                    ax1.grid(True, alpha=0.3)

                    # Plot A_G histogram
                    ax2.hist(X[:, 1], bins=30, alpha=0.3, color='gray', label='All data')
                    for k in range(n_components):
                        component_mask = labels == k
                        if component_mask.sum() > 0:
                            ax2.hist(X[component_mask, 1], bins=30, alpha=0.5,
                                    color=colors[k], label=f'Component {k}')
                        ax2.axvline(fixed_means[k, 1], color=colors[k], linestyle='--',
                                  linewidth=2, label=f'Mean {k}: {fixed_means[k, 1]:.3f}')
                    ax2.set_xlabel('A_G')
                    ax2.set_ylabel('Count')
                    ax2.set_title(f'A_G Distribution\nσ²(0)={covariances[0,0,0]+covariances[0,1,1]:.4f}, σ²(1)={covariances[1,0,0]+covariances[1,1,1]:.4f}')
                    ax2.legend(fontsize=8)
                    ax2.grid(True, alpha=0.3)

                    _safe_tight_layout(fig)
                    plotter.save_or_show(fig, save_path=None)

                    logger.info(f"  (Close plot to continue to next bin)")

                except Exception as e:
                    logger.warning(f"  (Plotting skipped for this bin - error: {e})")

            # Build summary row
            summary_row = {
                "photoelectron_bin_min": bin_min,
                "photoelectron_bin_max": bin_max,
                "n_molecules": n_molecules,
                "converged": converged,
                "overall_accuracy": stats['overall_accuracy'],
                "overall_error_rate": stats['overall_error_rate'],
            }

            # Add per-component accuracies and error rates
            for k in range(n_components):
                summary_row[f'component_{k}_accuracy'] = stats['accuracy_per_component'][k]
                summary_row[f'component_{k}_error_rate'] = stats['error_rate_per_component'][k]

            # Add confusion matrix elements
            for k1 in range(n_components):
                for k2 in range(n_components):
                    summary_row[f'confusion_matrix_{k1}{k2}'] = stats['confusion_matrix'][k1, k2]

            # Add covariance elements for each component
            for k in range(n_components):
                cov = covariances[k]
                summary_row[f'cov_{k}_AR_AR'] = cov[0, 0]
                summary_row[f'cov_{k}_AR_AG'] = cov[0, 1]
                summary_row[f'cov_{k}_AG_AG'] = cov[1, 1]
                summary_row[f'weight_{k}'] = weights[k]

            if verbose:
                logger.info(f"  Overall accuracy (analytical): {stats['overall_accuracy']:.3f}")
                logger.info(f"  Component accuracies: {stats['accuracy_per_component']}")

            all_summaries.append(summary_row)

        # Combine results
        if len(all_summaries) > 0:
            summary_db = pd.DataFrame(all_summaries)

            if verbose:
                logger.info("\n" + "=" * 60)
                logger.info("Analytical Analysis Complete!")
                logger.info("=" * 60)
                logger.info(f"Summary database: {len(summary_db)} bins")
                logger.info(f"Overall accuracy range: {summary_db['overall_accuracy'].min():.3f} - {summary_db['overall_accuracy'].max():.3f}")
        else:
            summary_db = pd.DataFrame()
            if verbose:
                logger.info("\n" + "=" * 60)
                logger.info("No results generated (no molecules in bins)")
                logger.info("=" * 60)

        return summary_db

    def _find_histogram_peaks_1d(self, data, n_peaks, bins=50):
        """
        Find peaks in 1D histogram for initial channel guess.

        Uses the same approach as extract_reference_means() for consistency.

        Args:
            data (np.ndarray): 1D array of values
            n_peaks (int): Number of peaks to find
            bins (int): Number of histogram bins (default: 50, matches extract_reference_means)

        Returns:
            np.ndarray: Peak positions, shape (n_peaks,)
        """
        from scipy.signal import find_peaks

        # Create histogram with density normalization
        hist, bin_edges = np.histogram(data, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Find peaks in histogram (distance=5 prevents very close peaks)
        peaks, properties = find_peaks(hist, height=0, distance=5)

        if len(peaks) >= n_peaks:
            # Sort by height and take top n_peaks
            peak_heights = hist[peaks]
            top_peak_indices = np.argsort(peak_heights)[-n_peaks:]
            peak_positions = bin_centers[peaks[top_peak_indices]]
            return np.sort(peak_positions)
        else:
            # Fallback: use quantiles
            return np.quantile(data, np.linspace(0.2, 0.8, n_peaks))

    def _find_initial_means_2d(self, X, n_channels, method="histogram_peaks"):
        """
        Find initial channel means in 2D feature space.

        Uses the same approach as extract_reference_means() for consistency.
        For 2-channel case, assumes anticorrelated colors (high A_R → low A_G).

        Args:
            X (np.ndarray): Data matrix, shape (n_samples, 2)
            n_channels (int): Number of channels
            method (str): Method for finding means

        Returns:
            np.ndarray: Initial means, shape (n_channels, 2)
        """
        if method == "histogram_peaks":
            # Find peaks in each dimension separately
            peaks_dim0 = self._find_histogram_peaks_1d(X[:, 0], n_channels)
            peaks_dim1 = self._find_histogram_peaks_1d(X[:, 1], n_channels)

            # Combine to form initial 2D means
            # Match peaks: if dim0 high, dim1 should be low (and vice versa)
            # This assumes anticorrelated channels (typical for multicolor SMLM)
            initial_means = np.zeros((n_channels, 2))

            if n_channels == 2:
                # Sort dim0 ascending, dim1 descending to pair anticorrelated peaks
                dim0_sorted = np.sort(peaks_dim0)
                dim1_sorted = np.sort(peaks_dim1)[::-1]
                initial_means[:, 0] = dim0_sorted
                initial_means[:, 1] = dim1_sorted
            else:
                # For n_channels != 2, use simpler approach (no assumed correlation)
                initial_means[:, 0] = peaks_dim0
                initial_means[:, 1] = peaks_dim1

            return initial_means

        elif method == "kmeans":
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=n_channels, n_init=10, random_state=42)
            kmeans.fit(X)
            return kmeans.cluster_centers_

        else:
            raise ValueError(f"Unknown method: {method}")

    def _estimate_initial_covariances_2d(self, X, initial_means, n_channels,
                                          X_err=None, use_core_region=True,
                                          percentile=50, scale=0.7):
        """
        Estimate initial covariances conservatively from core regions around means.

        Strategy:
        1. Hard assign points to nearest mean
        2. Take only the CORE points (e.g., 50th percentile by distance) for each component
        3. Calculate robust covariance from core region
        4. Optionally incorporate fitting errors
        5. Scale down by factor (default 0.7) to prevent EM from over-expanding

        This prevents the initial guess from being too broad by focusing on the
        well-separated core of each distribution.

        Args:
            X (np.ndarray): Data matrix, shape (n_samples, n_features)
            initial_means (np.ndarray): Initial means, shape (n_channels, n_features)
            n_channels (int): Number of channels
            X_err (np.ndarray, optional): Error matrix, shape (n_samples, n_features)
            use_core_region (bool): If True, use only core percentile of each component
            percentile (float): Percentile threshold for core region (default: 50)
            scale (float): Scaling factor for covariances (default: 0.7)

        Returns:
            np.ndarray: Initial covariances, shape (n_channels, n_features, n_features)
        """
        from scipy.spatial.distance import cdist

        n_samples, n_features = X.shape

        # Hard assignment: assign each point to nearest mean
        distances = cdist(X, initial_means, metric='euclidean')
        assignments = np.argmin(distances, axis=1)

        # Calculate sample covariance for each component
        initial_covariances = np.zeros((n_channels, n_features, n_features))

        for k in range(n_channels):
            mask = assignments == k
            n_assigned = mask.sum()

            if n_assigned > 20:  # Need reasonable number of points
                X_k = X[mask]

                # Use only core region for more conservative estimate
                if use_core_region and n_assigned > 50:
                    # Calculate distances from mean
                    dists_k = np.linalg.norm(X_k - initial_means[k], axis=1)
                    # Take only the closest percentile of points
                    threshold = np.percentile(dists_k, percentile)
                    core_mask = dists_k <= threshold
                    X_k_core = X_k[core_mask]
                else:
                    X_k_core = X_k

                if len(X_k_core) > 5:
                    # Calculate robust sample covariance from core
                    centered = X_k_core - initial_means[k]
                    cov_k = (centered.T @ centered) / len(X_k_core)

                    # Ensure symmetry (numerical precision can break this)
                    cov_k = (cov_k + cov_k.T) / 2.0

                    # If we have error information, incorporate it
                    if X_err is not None:
                        # Average measurement error for this component
                        X_err_k = X_err[mask]
                        if use_core_region and n_assigned > 50:
                            X_err_k = X_err_k[core_mask]

                        # Mean squared error (diagonal covariance contribution)
                        mean_err_sq = np.mean(X_err_k**2, axis=0)
                        err_cov = np.diag(mean_err_sq)

                        # Add measurement error to intrinsic spread
                        # But cap it so errors don't dominate
                        err_cov_capped = np.minimum(err_cov, cov_k * 0.5)
                        cov_k = cov_k + err_cov_capped

                    # Scale down covariance to be conservative (prevents overly broad initial guess)
                    cov_k = cov_k * scale

                    # Add regularization to ensure positive definite
                    # Use scale-dependent regularization based on diagonal variance
                    diag_var = np.diag(cov_k)
                    reg_scale = np.maximum(1e-6, np.mean(diag_var) * 1e-3)
                    cov_k += np.eye(n_features) * reg_scale

                    # Final validation: ensure positive definite
                    eigvals = np.linalg.eigvalsh(cov_k)
                    if np.min(eigvals) <= 0:
                        # Add stronger regularization
                        cov_k += np.eye(n_features) * (np.abs(np.min(eigvals)) + 1e-4)

                    initial_covariances[k] = cov_k
                else:
                    # Not enough core points, use small isotropic
                    initial_covariances[k] = np.eye(n_features) * 0.005
            else:
                # Very few points assigned, use small isotropic covariance
                initial_covariances[k] = np.eye(n_features) * 0.005

        return initial_covariances

