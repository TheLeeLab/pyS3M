# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
from typing import Any, Optional
from numpy.typing import NDArray
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
import logging
from pyS3M.PlottingBase import _safe_tight_layout
logger = logging.getLogger(__name__)


class ChannelUnmixingMixin:
    """Mixin providing channel-unmixing methods for extract_SMs."""
    def unmix_channels(
        self,
        loc_data: pd.DataFrame,
        n_channels: int,
        channels_to_use: list[str] = ["A_R", "A_G"],
        confidence_threshold: float = 0.95,
        false_positive_rate: float | None = None,
        initial_guess_method: str = "histogram_peaks",
        gmm_fit_method: str = "EM",
        covariance_type: str = "full",
        max_iter: int = 500,
        outlier_rejection: str = "mahalanobis",
        mestimator_type: str = "tukey",
        initial_guess_percentile: float = 50,
        initial_guess_scale: float = 0.7,
        verbose: bool = True,
        plot_results: bool = False,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        Separate SMLM localizations into N channels based on RGB amplitude ratios.

        This function uses Gaussian Mixture Model (GMM) fitting to separate multi-color
        SMLM data into distinct channels based on spectral signatures (A_R, A_G, A_B).
        Assignments are confidence-based with optional outlier rejection.

        Args:
            loc_data (pd.DataFrame): Localization data with columns:
                - A_R, A_G, A_B: Normalized RGB amplitudes
                - A_R_err, A_G_err, A_B_err: Fitting uncertainties (optional)
                - xc, yc: Localization coordinates
                - frame: Frame number

            n_channels (int): Number of distinct color channels (2-5 typical)

            channels_to_use (list): Which amplitude channels to use for separation
                - ['A_R', 'A_G']: 2D separation (typical for 2-3 color)
                - ['A_R', 'A_G', 'A_B']: 3D separation (for >3 colors)
                - ['A_R']: 1D separation (single ratio)
                Note: Corresponding error columns (e.g., A_R_err, A_G_err) are ALWAYS
                used for weighting if available. Higher errors = lower trust.

            confidence_threshold (float): Minimum posterior probability for assignment (0-1)
                Higher = more conservative (fewer assignments, higher purity)

            false_positive_rate (float, optional): Maximum acceptable FPR for assignment
                If specified, calculates confidence threshold from analytical overlap

            initial_guess_method (str): Method for initial channel centers
                - 'histogram_peaks': Find peaks in 1D histograms (default)
                - 'kmeans': K-means clustering

            gmm_fit_method (str): GMM fitting algorithm. ``'EM'`` auto-selects
                the best method (pygmmis Extreme Deconvolution when error
                columns are present, otherwise sklearn EM). ``'EM_weighted'``
                uses photoelectron-based weighting (legacy). ``'fixed'`` skips EM
                and uses the initial guess directly (most conservative).

            covariance_type (str): GMM covariance structure
                - 'full': Full covariance (allows correlation, default)
                - 'tied': Same covariance for all components
                - 'diag': Diagonal (no correlation)
                - 'spherical': Single variance per component

            max_iter (int): Maximum GMM fitting iterations

            outlier_rejection (str): Outlier handling method
                - 'none': No outlier rejection (default)
                - 'mahalanobis': Hard threshold on Mahalanobis distance

            mestimator_type (str): If outlier_rejection='mestimator'
                - 'huber': Moderate robustness
                - 'tukey': Aggressive robustness

            initial_guess_percentile (float): Percentile for core region selection (0-100)
                Lower = tighter initial guess. Default: 50 (median)
                Try 25-30 for very conservative separation

            initial_guess_scale (float): Scaling factor for initial covariances
                Lower = tighter ellipses. Default: 0.7
                Try 0.4-0.5 to prevent EM from over-expanding

            verbose (bool): Print progress and diagnostics

            plot_results (bool): Create diagnostic plots

        Returns:
            assigned_locs (pd.DataFrame): Input data with added columns:
                - 'channel': Assigned channel (0 to n_channels-1, or -1 for unassigned)
                - 'channel_confidence': Posterior probability for assigned channel
                - 'channel_probability_0', ...: Posterior for each channel
                - 'mahalanobis_distance': Distance to assigned channel mean
                - 'is_outlier': Boolean flag for outliers

            metadata (dict): Diagnostic information:
                - 'means': Fitted channel means
                - 'covariances': Fitted covariances
                - 'weights': Fitted channel weights
                - 'converged': Whether GMM converged
                - 'n_assigned': Number per channel
                - 'n_unassigned': Number rejected
                - 'confusion_matrix': Expected confusion matrix (if FPR specified)

        Example:
            >>> # 2-color separation (ATTO655 + Cy3B)
            >>> assigned, metadata = SM_E.unmix_channels(
            ...     loc_data,
            ...     n_channels=2,
            ...     channels_to_use=['A_R', 'A_G'],
            ...     confidence_threshold=0.95,
            ...     verbose=True,
            ...     plot_results=True
            ... )
            >>> print(f"Channel 0: {metadata['n_assigned'][0]} locs")
            >>> print(f"Channel 1: {metadata['n_assigned'][1]} locs")
        """
        if verbose:
            logger.info("=" * 70)
            logger.info("Channel Unmixing")
            logger.info("=" * 70)
            logger.info(f"Input: {len(loc_data)} localizations")
            logger.info(f"Channels: {n_channels}")
            logger.info(f"Features: {channels_to_use}")
            logger.info("")

        # ===== Phase 1: Input Validation and Preprocessing =====
        # Check required columns
        for col in channels_to_use:
            if col not in loc_data.columns:
                raise ValueError(f"Column '{col}' not found in loc_data")

        # Check for error columns and auto-generate if needed
        # Make a copy of loc_data to avoid modifying the original
        loc_data_work = loc_data.copy()

        error_cols = [f"{col}_err" for col in channels_to_use]
        missing_errors = [col for col in error_cols if col not in loc_data_work.columns]

        if missing_errors:
            if verbose:
                logger.info(f"Missing error columns: {missing_errors}")
                logger.info("Attempting to auto-generate errors...")

            for error_col in missing_errors:
                # Extract base column name (remove '_err' suffix)
                base_col = error_col.replace('_err', '')

                if base_col == 'photoelectrons':
                    # Poisson statistics: σ = sqrt(N)
                    loc_data_work[error_col] = np.sqrt(np.maximum(loc_data_work[base_col].values, 1))
                    if verbose:
                        mean_err = loc_data_work[error_col].mean()
                        logger.info(f"  {error_col}: Generated from Poisson statistics (mean={mean_err:.2f})")
                elif base_col in loc_data_work.columns:
                    # For other columns, use a small fraction of the value as error estimate
                    # This is a conservative guess - user should provide real errors if possible
                    loc_data_work[error_col] = loc_data_work[base_col].values * 0.05  # 5% relative error
                    if verbose:
                        mean_err = loc_data_work[error_col].mean()
                        logger.info(f"  {error_col}: Estimated as 5% of {base_col} (mean={mean_err:.4f})")
                        logger.info(f"    WARNING: Using estimated errors. Provide measured errors for better results.")
                else:
                    raise ValueError(f"Cannot auto-generate error for '{error_col}': column '{base_col}' not found")

        # Extract feature matrix
        X = loc_data_work[channels_to_use].values
        n_features = X.shape[1]

        if verbose:
            logger.info(f"Feature matrix: {X.shape}")
            logger.info(f"Feature ranges:")
            for i, col in enumerate(channels_to_use):
                logger.info(f"  {col}: [{X[:, i].min():.3f}, {X[:, i].max():.3f}], mean={X[:, i].mean():.3f}")
            logger.info("")

        # ===== Phase 2: Initial Guess for Channel Means =====
        if verbose:
            logger.info(f"Finding initial channel means (method: {initial_guess_method})...")

        if n_features == 1:
            # 1D case
            initial_means = self._find_histogram_peaks_1d(
                X[:, 0], n_channels
            ).reshape(-1, 1)
        elif n_features == 2:
            # 2D case
            initial_means = self._find_initial_means_2d(
                X, n_channels, method=initial_guess_method
            )
        else:
            # 3D or higher - use k-means
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=n_channels, n_init=10, random_state=42)
            kmeans.fit(X)
            initial_means = kmeans.cluster_centers_

        if verbose:
            logger.info(f"Initial means:")
            for k in range(n_channels):
                mean_str = ", ".join(
                    [f"{channels_to_use[i]}={initial_means[k, i]:.3f}" for i in range(n_features)]
                )
                logger.info(f"  Channel {k}: {mean_str}")
            logger.info("")

        # ===== Phase 2.5: Estimate Initial Covariances =====
        # Two-stage initialization: means from histograms, covariances from data
        if initial_guess_method == "histogram_peaks":
            if verbose:
                logger.info("Estimating initial covariances from core regions (conservative)...")

            # Extract error matrix if available (now works for any n_features)
            error_cols = [f"{col}_err" for col in channels_to_use]
            if all(col in loc_data_work.columns for col in error_cols):
                X_err = loc_data_work[error_cols].values
            else:
                X_err = None  # pragma: no cover -- Phase 1 above always
                # back-fills every f"{col}_err" (or raises), so error_cols
                # is always fully present here; unreachable defensively.

            initial_covariances = self._estimate_initial_covariances_2d(
                X, initial_means, n_channels, X_err=X_err,
                use_core_region=True, percentile=initial_guess_percentile,
                scale=initial_guess_scale
            )

            if verbose:
                for k in range(n_channels):
                    det_k = np.linalg.det(initial_covariances[k])
                    # Calculate standard deviations along principal axes
                    eigvals = np.linalg.eigvalsh(initial_covariances[k])
                    sigma_str = ", ".join([f"σ{i+1}={np.sqrt(eigvals[i]):.3f}" for i in range(n_features)])
                    logger.info(f"  Channel {k}: det(cov)={det_k:.6f}, {sigma_str}")
                logger.info("")

            # Create diagnostic plot showing initial guess (only for 2D)
            if plot_results and n_features == 2:
                self._plot_initial_guess_2d(
                    X, channels_to_use, initial_means, initial_covariances, n_channels
                )
            elif plot_results and n_features > 2:
                if verbose:
                    logger.info(f"  Skipping initial guess plot (only available for 2D, current: {n_features}D)")
        else:
            initial_covariances = None

        # ===== Phase 3: GMM Fitting =====
        # Extract errors - ALWAYS use them if available (now includes auto-generated ones)
        error_cols = [f"{col}_err" for col in channels_to_use]
        has_errors = all(col in loc_data_work.columns for col in error_cols)

        if has_errors:
            X_err = loc_data_work[error_cols].values
        else:
            X_err = None  # pragma: no cover -- see Phase 2.5's identical note

        # Intelligent method selection: Use pygmmis if errors available, sklearn otherwise
        if gmm_fit_method == "EM":
            if has_errors:
                # Use pygmmis Extreme Deconvolution (theoretically optimal for per-point errors)
                actual_method = "extreme_deconvolution"
                if verbose:
                    logger.info(f"Fitting GMM (method: EM → Extreme Deconvolution, covariance: {covariance_type})...")
                    logger.info(f"  Auto-selected pygmmis (error columns detected)")
                    logger.info(f"  Mean errors: {X_err.mean(axis=0)}")
            else:  # pragma: no cover -- has_errors is always True (see above);
                # 'sklearn_EM' itself is still reachable by naming it directly
                # via gmm_fit_method, just never through this auto-select arm.
                # Use sklearn EM (no errors available)
                actual_method = "sklearn_EM"
                if verbose:
                    logger.info(f"Fitting GMM (method: EM → sklearn, covariance: {covariance_type})...")
                    logger.info("  No error columns found, using sklearn EM without error weighting")
        else:
            actual_method = gmm_fit_method
            if verbose:
                logger.info(f"Fitting GMM (method: {gmm_fit_method}, covariance: {covariance_type})...")
                if has_errors:
                    logger.info(f"  Error columns available (mean errors: {X_err.mean(axis=0)})")
                else:
                    logger.info("  No error columns found")  # pragma: no cover -- has_errors always True

        if actual_method == "sklearn_EM":
            # Use sklearn GMM without error weighting (pure EM)
            from sklearn.mixture import GaussianMixture

            # Prepare precisions_init if we have initial covariances
            if initial_covariances is not None and covariance_type == "full":
                # sklearn uses precisions (inverse covariances) for initialization
                precisions_init = np.zeros_like(initial_covariances)
                for k in range(n_channels):
                    try:
                        precisions_init[k] = np.linalg.inv(initial_covariances[k])
                    except np.linalg.LinAlgError:
                        # Singular, use regularized version
                        cov_reg = initial_covariances[k] + np.eye(n_features) * 1e-3
                        precisions_init[k] = np.linalg.inv(cov_reg)
            else:
                precisions_init = None

            gmm = GaussianMixture(
                n_components=n_channels,
                covariance_type=covariance_type,
                max_iter=max_iter,
                n_init=1,
                means_init=initial_means,
                precisions_init=precisions_init,
                random_state=42,
            )
            gmm.fit(X)  # No replication - pure sklearn EM
            means = gmm.means_
            covariances = gmm.covariances_
            weights = gmm.weights_
            converged = gmm.converged_

        elif gmm_fit_method == "EM_weighted":
            # Use existing weighted EM implementation
            photoelectrons = (loc_data_work["photoelectrons"].values
                              if "photoelectrons" in loc_data_work.columns else None)
            A_R = loc_data_work["A_R"].values if "A_R" in loc_data_work.columns else None
            A_G = loc_data_work["A_G"].values if "A_G" in loc_data_work.columns else None

            # ALWAYS use error columns if available (match channels_to_use)
            # NOTE: EM_weighted only supports 2D (legacy method, use 'EM' for N-D support)
            if has_errors and len(channels_to_use) == 2:
                # Extract the specific error columns for the channels being used
                if channels_to_use[0] == 'A_R':
                    sigma_dim0 = loc_data_work["A_R_err"].values
                elif channels_to_use[0] == 'A_G':
                    sigma_dim0 = loc_data_work["A_G_err"].values
                elif channels_to_use[0] == 'A_B':
                    sigma_dim0 = loc_data_work["A_B_err"].values
                else:
                    sigma_dim0 = loc_data_work[f"{channels_to_use[0]}_err"].values

                if channels_to_use[1] == 'A_R':
                    sigma_dim1 = loc_data_work["A_R_err"].values
                elif channels_to_use[1] == 'A_G':
                    sigma_dim1 = loc_data_work["A_G_err"].values
                elif channels_to_use[1] == 'A_B':
                    sigma_dim1 = loc_data_work["A_B_err"].values
                else:
                    sigma_dim1 = loc_data_work[f"{channels_to_use[1]}_err"].values

                # For compatibility with _fit_gmm_em (expects A_R and A_G)
                sigma_A_R = sigma_dim0
                sigma_A_G = sigma_dim1
            else:
                if len(channels_to_use) != 2:
                    raise ValueError("EM_weighted method only supports 2D (len(channels_to_use)==2). Use gmm_fit_method='EM' for N-D support.")
                # has_errors is always True (Phase 1 back-fills every _err
                # column or raises), so this else is only ever entered via
                # len(channels_to_use) != 2 -- which the guard above always
                # raises on first. Unreachable defensively.
                sigma_A_R = None  # pragma: no cover
                sigma_A_G = None  # pragma: no cover

            means, covariances, weights, converged = self._fit_gmm_em(
                X=X,
                initial_means=initial_means,
                n_components=n_channels,
                covariance_type=covariance_type,
                max_iter=max_iter,
                photoelectrons=photoelectrons,
                A_R=A_R,
                A_G=A_G,
                has_error_columns=has_errors,
                sigma_A_R=sigma_A_R,
                sigma_A_G=sigma_A_G,
                verbose=False,
            )
            gmm = None  # Not using sklearn GMM object

        elif actual_method == "extreme_deconvolution":
            # Use pygmmis Extreme Deconvolution for proper error handling
            if not has_errors:  # pragma: no cover -- has_errors always True (see Phase 3 above)
                raise ValueError(
                    "Extreme Deconvolution requires error columns (A_R_err, A_G_err, etc.). "
                    "This should not happen when auto-selected by gmm_fit_method='EM'."
                )

            means, covariances, weights, converged = self._fit_gmm_pygmmis(
                X=X,
                X_err=X_err,
                initial_means=initial_means,
                n_components=n_channels,
                max_iter=max_iter,
                verbose=verbose,
            )
            gmm = None  # Not using sklearn GMM object

        elif gmm_fit_method == "fixed":
            # Use initial guess without EM refinement (most conservative)
            # This prevents EM from expanding the Gaussians
            if initial_covariances is None:
                raise ValueError(
                    "gmm_fit_method='fixed' requires initial_guess_method='histogram_peaks' "
                    "to compute initial covariances. Either change initial_guess_method or "
                    "use a different gmm_fit_method."
                )

            means = initial_means
            covariances = initial_covariances

            # Validate all covariances are positive definite
            for k in range(n_channels):
                eigvals = np.linalg.eigvalsh(covariances[k])
                if np.min(eigvals) <= 0:
                    if verbose:
                        logger.info(f"  Warning: Channel {k} covariance not positive definite (min eigenvalue={np.min(eigvals):.2e})")
                        logger.info(f"           Adding regularization...")
                    # Add regularization
                    diag_mean = np.mean(np.diag(covariances[k]))
                    reg_amount = np.abs(np.min(eigvals)) + np.maximum(1e-6, diag_mean * 1e-3)
                    covariances[k] += np.eye(n_features) * reg_amount

            # Calculate weights by hard assignment
            from scipy.spatial.distance import cdist
            distances = cdist(X, initial_means, metric='euclidean')
            assignments = np.argmin(distances, axis=1)
            weights = np.array([np.sum(assignments == k) / len(X) for k in range(n_channels)])

            converged = True  # No iteration needed
            gmm = None

            if verbose:
                logger.info("Using fixed initial guess (no EM refinement)")

        else:
            raise ValueError(f"Unknown gmm_fit_method: {gmm_fit_method}")

        if verbose:
            status = "converged" if converged else "did not converge"
            logger.info(f"GMM fitting: {status}")
            logger.info(f"Fitted means:")
            for k in range(n_channels):
                mean_str = ", ".join(
                    [f"{channels_to_use[i]}={means[k, i]:.3f}" for i in range(n_features)]
                )
                weight_pct = weights[k] * 100
                logger.info(f"  Channel {k}: {mean_str} (weight: {weight_pct:.1f}%)")
            logger.info("")

        # ===== Phase 4: Channel Assignment with Confidence =====
        if verbose:
            logger.info("Calculating posterior probabilities and assignments...")

        # Calculate posterior probabilities
        n_locs = len(X)
        log_probs = np.zeros((n_locs, n_channels))

        for k in range(n_channels):
            mvn = multivariate_normal(mean=means[k], cov=covariances[k])
            log_probs[:, k] = mvn.logpdf(X) + np.log(weights[k])

        # Normalize to get posteriors (log-sum-exp trick)
        log_probs_max = log_probs.max(axis=1, keepdims=True)
        probs = np.exp(log_probs - log_probs_max)
        posterior_probs = probs / probs.sum(axis=1, keepdims=True)

        # Most likely channel
        channel_assignments = np.argmax(posterior_probs, axis=1)

        # Confidence = posterior of assigned channel
        confidence = posterior_probs[np.arange(n_locs), channel_assignments]

        # Calculate analytical confusion matrix if FPR specified
        if false_positive_rate is not None:
            if verbose:
                logger.info(f"Calculating analytical FPR to determine confidence threshold (target: {false_positive_rate:.3f})...")

            stats = self.calculate_analytical_misidentification(
                means, covariances, weights, n_samples=10000, random_state=42
            )

            if verbose:
                logger.info(f"Analytical accuracy: {stats['overall_accuracy']:.3f}")
                logger.info(f"Confusion matrix:")
                logger.info(stats["confusion_matrix"])
                logger.info("")

            # Use simple threshold based on FPR
            # Higher FPR tolerance → lower threshold → more assignments
            confidence_threshold = 1.0 - (false_positive_rate / n_channels)

            if verbose:
                logger.info(f"Setting confidence threshold to {confidence_threshold:.3f} (from FPR={false_positive_rate:.3f})")

        # Apply confidence threshold
        is_assigned = confidence >= confidence_threshold
        channel_assignments_filtered = channel_assignments.copy()
        channel_assignments_filtered[~is_assigned] = -1

        # ===== Phase 5: Outlier Detection =====
        is_outlier = np.zeros(n_locs, dtype=bool)

        if outlier_rejection == "mahalanobis":
            if verbose:
                logger.info("Applying Mahalanobis distance outlier rejection...")

            from scipy.stats import chi2

            # Calculate Mahalanobis distance to assigned channel
            mahalanobis_distances = np.zeros(n_locs)

            for k in range(n_channels):
                channel_k_mask = channel_assignments_filtered == k
                if not np.any(channel_k_mask):
                    continue

                X_k = X[channel_k_mask]
                try:
                    inv_cov_k = np.linalg.inv(covariances[k])
                except np.linalg.LinAlgError:
                    # Singular covariance, regularize
                    cov_reg = covariances[k] + 1e-6 * np.eye(n_features)
                    inv_cov_k = np.linalg.inv(cov_reg)

                diff_k = X_k - means[k]
                mahal_k = np.sqrt(np.sum(diff_k @ inv_cov_k * diff_k, axis=1))
                mahalanobis_distances[channel_k_mask] = mahal_k

            # Flag outliers (99.9% quantile of chi-squared)
            outlier_threshold = np.sqrt(chi2.ppf(0.999, df=n_features))
            is_outlier = mahalanobis_distances > outlier_threshold
            channel_assignments_filtered[is_outlier] = -1

            if verbose:
                n_outliers = is_outlier.sum()
                logger.info(f"  Outliers detected: {n_outliers} ({100*n_outliers/n_locs:.2f}%, threshold={outlier_threshold:.2f})")

        else:
            # Calculate Mahalanobis distance anyway for diagnostics
            mahalanobis_distances = np.zeros(n_locs)
            for k in range(n_channels):
                channel_k_mask = channel_assignments_filtered == k
                if not np.any(channel_k_mask):
                    continue

                X_k = X[channel_k_mask]
                try:
                    inv_cov_k = np.linalg.inv(covariances[k])
                except np.linalg.LinAlgError:
                    cov_reg = covariances[k] + 1e-6 * np.eye(n_features)
                    inv_cov_k = np.linalg.inv(cov_reg)

                diff_k = X_k - means[k]
                mahal_k = np.sqrt(np.sum(diff_k @ inv_cov_k * diff_k, axis=1))
                mahalanobis_distances[channel_k_mask] = mahal_k

        # ===== Phase 6: Create Output DataFrame =====
        assigned_locs = loc_data.copy()
        assigned_locs["channel"] = channel_assignments_filtered
        assigned_locs["channel_confidence"] = confidence
        assigned_locs["mahalanobis_distance"] = mahalanobis_distances
        assigned_locs["is_outlier"] = is_outlier

        # Add per-channel posterior probabilities
        for k in range(n_channels):
            assigned_locs[f"channel_probability_{k}"] = posterior_probs[:, k]

        # Create metadata
        n_assigned_per_channel = {
            k: np.sum(channel_assignments_filtered == k) for k in range(n_channels)
        }
        n_unassigned = np.sum(channel_assignments_filtered == -1)

        metadata = {
            "means": means,
            "covariances": covariances,
            "weights": weights,
            "converged": converged,
            "n_assigned": n_assigned_per_channel,
            "n_unassigned": n_unassigned,
            "initial_means": initial_means,
            "channels_used": channels_to_use,
            "confidence_threshold": confidence_threshold,
        }

        # Add confusion matrix if calculated
        if false_positive_rate is not None:
            metadata["confusion_matrix"] = stats["confusion_matrix"]
            metadata["assignment_purity"] = np.diag(stats["confusion_matrix"])
            metadata["false_positive_rates"] = 1.0 - np.diag(stats["confusion_matrix"])

        if verbose:
            logger.info("\n" + "=" * 70)
            logger.info("Unmixing Complete")
            logger.info("=" * 70)
            logger.info(f"Assignments:")
            for k in range(n_channels):
                n_k = n_assigned_per_channel[k]
                pct_k = 100 * n_k / n_locs
                logger.info(f"  Channel {k}: {n_k:,} ({pct_k:.1f}%)")
            pct_unassigned = 100 * n_unassigned / n_locs
            logger.info(f"  Unassigned: {n_unassigned:,} ({pct_unassigned:.1f}%)")
            logger.info("")

        # ===== Phase 7: Diagnostic Plotting =====
        if plot_results:
            self._plot_unmixing_results(
                X, channels_to_use, channel_assignments_filtered, confidence,
                means, covariances, weights, n_channels, confidence_threshold,
                metadata
            )

        return assigned_locs, metadata

    def _plot_initial_guess_2d(self, X, channels_to_use, initial_means,
                                initial_covariances, n_channels, display: bool = True):
        """
        Plot 2D histogram with initial guess overlaid (means and 2σ ellipses).

        Args:
            X (np.ndarray): Data matrix, shape (n_samples, 2)
            channels_to_use (list): Channel names
            initial_means (np.ndarray): Initial means, shape (n_channels, 2)
            initial_covariances (np.ndarray): Initial covariances, shape (n_channels, 2, 2)
            n_channels (int): Number of channels
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
        from pyS3M.PlottingBase import PublicationPlotter

        # Use colors that don't conflict with channel names (avoid red/green for R/G channels)
        colors_ch = ['blue', 'orange', 'purple', 'cyan', 'magenta', 'brown'][:n_channels]

        plotter = PublicationPlotter()
        fig, ax = plotter.two_column_plot(nrows=1, ncols=1)  # Use default height (3 inches)

        # 2D histogram
        hist_2d, xedges, yedges = np.histogram2d(X[:, 0], X[:, 1], bins=100)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        im = ax.imshow(
            hist_2d.T, origin='lower', extent=extent,
            cmap='gray', aspect='auto', interpolation='nearest'
        )
        plt.colorbar(im, ax=ax, label='Count')

        # Overlay initial means and 2σ ellipses
        for k in range(n_channels):
            # Plot mean
            ax.scatter(
                initial_means[k, 0], initial_means[k, 1],
                s=200, marker='x', color=colors_ch[k],
                linewidths=4, label=f'Channel {k}', zorder=10
            )

            # Plot 2σ confidence ellipse
            eigvals, eigvecs = np.linalg.eigh(initial_covariances[k])
            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
            width, height = 2 * 2 * np.sqrt(eigvals)  # 2σ

            ellipse = Ellipse(
                initial_means[k], width, height, angle=angle,
                edgecolor=colors_ch[k], facecolor='none', linewidth=3, zorder=9
            )
            ax.add_patch(ellipse)

        ax.set_xlabel(channels_to_use[0])
        ax.set_ylabel(channels_to_use[1])
        ax.set_title('Initial Guess: 2D Histogram + Means + 2σ Ellipses', weight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Don't use tight_layout() - it conflicts with colorbar layout engine
        fig.subplots_adjust(right=0.85)  # Make room for colorbar
        if display:
            plt.show()

    def _plot_unmixing_results(
        self, X, channels_to_use, assignments, confidence,
        means, covariances, weights, n_channels, confidence_threshold,
        metadata, display: bool = True
    ):
        """Create diagnostic plots for channel unmixing results."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
        from scipy.stats import norm
        from pyS3M.PlottingBase import PublicationPlotter

        n_features = X.shape[1]

        # Plot 1: 1D Histograms with GMM overlay
        plotter = PublicationPlotter()
        # Use 2.5 inches per panel (reasonable for stacked histograms)
        fig, axes = plotter.one_column_plot(npanels=n_features, height=2.5 * n_features)
        if n_features == 1:
            axes = [axes]

        # Use colors that don't conflict with channel names (avoid red/green for R/G channels)
        colors = ['blue', 'orange', 'purple', 'cyan', 'magenta', 'brown'][:n_channels]

        for i, channel_name in enumerate(channels_to_use):
            ax = axes[i]

            # Histogram of all data
            ax.hist(X[:, i], bins=500, alpha=0.3, color='gray', label='All data', density=True)

            # Histograms per assigned channel
            for k in range(n_channels):
                mask = assignments == k
                if mask.sum() > 0:
                    ax.hist(
                        X[mask, i], bins=200, alpha=0.5,
                        color=colors[k], label=f'Channel {k}', density=True
                    )

            # GMM components (marginal distributions)
            x_range = np.linspace(X[:, i].min(), X[:, i].max(), 1000)
            gmm_pdf = np.zeros_like(x_range)

            for k in range(n_channels):
                # Project to 1D (marginal)
                if n_features == 1:
                    mean_1d = means[k, 0]
                    var_1d = covariances[k, 0, 0]
                else:
                    mean_1d = means[k, i]
                    var_1d = covariances[k, i, i]

                pdf_k = weights[k] * norm.pdf(x_range, mean_1d, np.sqrt(var_1d))
                ax.plot(x_range, pdf_k, color=colors[k], linewidth=2, linestyle='--',
                       label=f'GMM Ch{k}')
                gmm_pdf += pdf_k

            ax.plot(x_range, gmm_pdf, 'k-', linewidth=2, label='GMM total')

            ax.set_xlabel(channel_name)
            ax.set_ylabel('Density')
            ax.set_title(f'{channel_name} Distribution with GMM Fit')
            ax.legend()
            ax.grid(True, alpha=0.3)

        _safe_tight_layout(fig)  # Use safe wrapper to avoid layout warnings
        if display:
            plt.show()

        # Plot 2: 2D Scatter (if 2D data)
        if n_features == 2:
            fig, axes = plotter.two_column_plot(nrows=1, ncols=2)  # Use default height (3 inches)

            # Left: GMM ellipses
            ax = axes[0]
            ax.scatter(X[:, 0], X[:, 1], s=1, alpha=0.2, c='gray', rasterized=True)

            for k in range(n_channels):
                # 2σ confidence ellipse
                eigvals, eigvecs = np.linalg.eigh(covariances[k])
                angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
                width, height = 2 * 2 * np.sqrt(eigvals)

                ellipse = Ellipse(
                    means[k], width, height, angle=angle,
                    edgecolor=colors[k], facecolor='none', linewidth=3
                )
                ax.add_patch(ellipse)
                ax.scatter(
                    means[k, 0], means[k, 1], s=200, marker='x',
                    color=colors[k], linewidths=4, label=f'Channel {k}'
                )

            ax.set_xlabel(channels_to_use[0])
            ax.set_ylabel(channels_to_use[1])
            ax.set_title('GMM Fit (2σ ellipses)')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Right: Assignments
            ax = axes[1]
            for k in range(n_channels):
                mask = assignments == k
                if mask.sum() > 0:
                    ax.scatter(
                        X[mask, 0], X[mask, 1], s=1, alpha=0.5,
                        color=colors[k], label=f'Ch {k} (n={mask.sum():,})',
                        rasterized=True
                    )

            # Unassigned
            unassigned_mask = assignments == -1
            if unassigned_mask.sum() > 0:
                ax.scatter(
                    X[unassigned_mask, 0], X[unassigned_mask, 1],
                    s=1, alpha=0.3, color='black',
                    label=f'Unassigned (n={unassigned_mask.sum():,})',
                    rasterized=True
                )

            ax.set_xlabel(channels_to_use[0])
            ax.set_ylabel(channels_to_use[1])
            ax.set_title(f'Assignments (threshold={confidence_threshold:.2f})')
            ax.legend()
            ax.grid(True, alpha=0.3)

            _safe_tight_layout(fig)  # Use safe wrapper to avoid layout warnings
            if display:
                plt.show()

        # Plot 3: Confidence histogram
        fig, ax = plotter.one_column_plot(npanels=1)  # Use default height (3 inches)

        for k in range(n_channels):
            mask = (assignments == k)
            if mask.sum() > 0:
                ax.hist(
                    confidence[mask], bins=100, alpha=0.6,
                    color=colors[k], label=f'Channel {k}'
                )

        ax.axvline(
            confidence_threshold, color='red', linestyle='--',
            linewidth=3, label=f'Threshold ({confidence_threshold:.2f})'
        )
        ax.set_xlabel('Assignment Confidence')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Assignment Confidences')
        ax.legend()
        ax.grid(True, alpha=0.3)
        _safe_tight_layout(fig)  # Use safe wrapper to avoid layout warnings
        if display:
            plt.show()

        # Plot 4: Confusion matrix (if available)
        if 'confusion_matrix' in metadata:
            fig, ax = plotter.one_column_plot(npanels=1, height=3.5)
            conf_mat = metadata['confusion_matrix']

            im = ax.imshow(conf_mat, cmap='Blues', vmin=0, vmax=1)

            # Annotate cells
            for i in range(n_channels):
                for j in range(n_channels):
                    text_color = 'white' if conf_mat[i, j] > 0.5 else 'black'
                    ax.text(
                        j, i, f"{conf_mat[i, j]:.3f}",
                        ha="center", va="center",
                        color=text_color, weight='bold'
                    )

            ax.set_xticks(np.arange(n_channels))
            ax.set_yticks(np.arange(n_channels))
            ax.set_xticklabels([f'Ch {k}' for k in range(n_channels)])
            ax.set_yticklabels([f'Ch {k}' for k in range(n_channels)])
            ax.set_xlabel('Predicted Channel')
            ax.set_ylabel('True Channel')
            ax.set_title('Expected Confusion Matrix (Analytical)')
            plt.colorbar(im, ax=ax, label='Probability')
            # Don't use tight_layout() - it conflicts with colorbar layout engine
            fig.subplots_adjust(right=0.85)  # Make room for colorbar
            if display:
                plt.show()

    # ========================================================================
    # Joint Spatial-Spectral Clustering Unmixing
    # ========================================================================
    # Added: 2026-06-24
    # Groups temporally-linked blink events into per-molecule clusters using
    # a joint Mahalanobis distance in (x, y, A_R, A_G) / error space, then
    # fits the GMM to per-cluster spectral means rather than individual locs.
    # ========================================================================

    def unmix_channels_joint_cluster(
        self,
        loc_data: pd.DataFrame,
        n_channels: int = 2,
        channels_to_use: list = None,
        spatial_cols: list = None,
        spatial_err_cols: list = None,
        d_threshold: float = 2.0,
        min_cluster_size: int = 3,
        confidence_threshold_isolated: float = 0.90,
        verbose: bool = True,
        plot_results: bool = True,
    ) -> tuple:
        """Unmix spectral channels via joint spatial-spectral clustering.

        Requires temporally-linked input (column ``'n'`` must be present).
        Each blink event is one row; contiguous blink frames should already have
        been collapsed by ``link_localisations``.

        Algorithm
        ---------
        1. Cluster blink events with ``joint_spectral_spatial_cluster`` using a
           4-D Mahalanobis distance in (x, y, A_R, A_G) / error space.  Each
           cluster corresponds to one molecule of one species.
        2. Compute photon-weighted per-cluster spectral means (weight by ``n``).
        3. Fit a GMM with EM to the cluster means — populations are tight because
           each point is a multi-blink molecular average, not a shot-noise-limited
           single frame.
        4. Assign each cluster a channel by GMM posterior.
        5. Propagate channel labels to every blink event in the cluster.
        6. Classify isolated blink events (no cluster) with the cluster-fitted GMM
           at ``confidence_threshold_isolated``.

        Parameters
        ----------
        loc_data : pd.DataFrame
            Temporally-linked localisation table with column ``'n'``.
        n_channels : int
            Number of spectral channels. Default 2.
        channels_to_use : list
            Spectral feature columns. Default ['A_R', 'A_G'].
        spatial_cols : list
            Spatial coordinate columns. Default ['xc', 'yc'].
        spatial_err_cols : list
            Spatial uncertainty columns. Default ['xc_err', 'yc_err'].
        d_threshold : float
            Mahalanobis gate for clustering (combined-σ units). Default 2.0.
        min_cluster_size : int
            Minimum blink events to form a valid cluster. Default 3.
        confidence_threshold_isolated : float
            GMM posterior threshold for isolated blink events. Default 0.90.
        verbose : bool
            Print progress to logger.
        plot_results : bool
            Show diagnostic scatter plots after assignment.

        Returns
        -------
        result : pd.DataFrame
            Copy of ``loc_data`` with added columns:
            ``'joint_cluster_id'``, ``'channel'`` (−1 = unassigned),
            ``'channel_confidence'``, ``'assignment_code'``
            (0 = unassigned, 1 = from cluster, 2 = isolated GMM).
        metadata : dict
            Keys: ``n_clusters``, ``n_clustered``, ``n_isolated``,
            ``means``, ``covariances``, ``weights``, ``gmm``,
            ``n_assigned`` (dict per channel), ``n_unassigned``,
            ``n_assigned_from_clusters``, ``n_assigned_from_isolated``.
        """
        from pyS3M.LinkingFunctions import joint_spectral_spatial_cluster

        if channels_to_use is None:
            channels_to_use = ['A_R', 'A_G']
        if spatial_cols is None:
            spatial_cols = ['xc', 'yc']
        if spatial_err_cols is None:
            spatial_err_cols = ['xc_err', 'yc_err']
        spectral_err_cols = [c + '_err' for c in channels_to_use]

        if verbose:
            logger.info("=" * 70)
            logger.info("Joint Spatial-Spectral Channel Unmixing")
            logger.info("=" * 70)
            logger.info(f"Input: {len(loc_data):,} blink events")
            logger.info(f"Channels: {n_channels},  Features: {channels_to_use}")
            logger.info(f"d_threshold: {d_threshold},  min_cluster_size: {min_cluster_size}")
            logger.info("")

        # ── Step 1: joint spatial-spectral clustering ──────────────────────
        if verbose:
            logger.info("Step 1: Joint spatial-spectral clustering...")

        clustered = joint_spectral_spatial_cluster(
            loc_data,
            spatial_cols=spatial_cols,
            spectral_cols=channels_to_use,
            spatial_err_cols=spatial_err_cols,
            spectral_err_cols=spectral_err_cols,
            d_threshold=d_threshold,
            min_cluster_size=min_cluster_size,
        )

        n_clusters = int(clustered['joint_cluster_id'].max()) + 1
        n_isolated = int((clustered['joint_cluster_id'] == -1).sum())
        n_in_clusters = len(clustered) - n_isolated

        if verbose:
            logger.info(f"  {n_clusters} clusters,  "
                        f"{n_in_clusters:,} blink events in clusters,  "
                        f"{n_isolated:,} isolated")
            logger.info("")

        if n_clusters < n_channels:
            raise ValueError(
                f"Only {n_clusters} cluster(s) found but {n_channels} channels "
                "requested.  Try reducing d_threshold or min_cluster_size."
            )

        # ── Step 2: inverse-variance weighted per-cluster spectral means ──────
        # Weight each blink event by 1/σ² where σ = A_X_err.  This is the
        # statistically optimal weight for combining independent estimates:
        # blink events with tighter spectral fits contribute more to the mean.
        # (link_localisations already did this within a blink; we do it again
        # across blinks of the same molecule.)
        if verbose:
            logger.info("Step 2: Computing per-cluster spectral means "
                        "(inverse-variance weighted by A_X_err)...")

        in_cluster = clustered[clustered['joint_cluster_id'] >= 0]
        cluster_means = (
            in_cluster
            .groupby('joint_cluster_id')
            .apply(
                lambda g: pd.Series({
                    col: np.average(
                        g[col].to_numpy(),
                        weights=1.0 / np.maximum(
                            g[col + '_err'].to_numpy() ** 2, 1e-20
                        ),
                    )
                    for col in channels_to_use
                })
            )
            .reset_index()
        )
        cluster_X = cluster_means[channels_to_use].to_numpy(dtype=np.float64)

        if verbose:
            for i, col in enumerate(channels_to_use):
                logger.info(f"  Cluster {col}: "
                            f"{cluster_X[:, i].mean():.3f} ± {cluster_X[:, i].std():.3f}")
            logger.info("")

        # ── Step 3: GMM on cluster means ──────────────────────────────────
        if verbose:
            logger.info("Step 3: Fitting GMM (EM) to cluster means...")

        gmm = GaussianMixture(
            n_components=n_channels,
            covariance_type='full',
            max_iter=300,
            n_init=10,
            random_state=42,
        )
        gmm.fit(cluster_X)

        means = gmm.means_
        covariances = gmm.covariances_
        weights = gmm.weights_

        if verbose:
            for k in range(n_channels):
                feat_str = ", ".join(
                    f"{channels_to_use[i]}={means[k, i]:.3f}"
                    for i in range(len(channels_to_use))
                )
                logger.info(f"  Channel {k}: {feat_str}  (weight {weights[k]*100:.1f}%)")
            logger.info("")

        # ── Step 4: assign each cluster a channel ─────────────────────────
        if verbose:
            logger.info("Step 4: Assigning clusters by GMM posterior...")

        cluster_posteriors = gmm.predict_proba(cluster_X)
        cluster_channel = np.argmax(cluster_posteriors, axis=1)
        cluster_confidence = cluster_posteriors[np.arange(len(cluster_X)), cluster_channel]

        cluster_ids_vals = cluster_means['joint_cluster_id'].to_numpy()
        id_to_channel = dict(zip(cluster_ids_vals, cluster_channel))
        id_to_confidence = dict(zip(cluster_ids_vals, cluster_confidence))

        # ── Step 5: propagate labels to individual blink events ────────────
        # assignment_code: 0 = unassigned, 1 = from cluster, 2 = isolated GMM
        result = clustered.copy()
        result['channel'] = np.int32(-1)
        result['channel_confidence'] = np.float32(np.nan)
        result['assignment_code'] = np.uint8(0)

        in_mask = result['joint_cluster_id'] >= 0
        result.loc[in_mask, 'channel'] = (
            result.loc[in_mask, 'joint_cluster_id'].map(id_to_channel).astype(np.int32)
        )
        result.loc[in_mask, 'channel_confidence'] = (
            result.loc[in_mask, 'joint_cluster_id'].map(id_to_confidence).astype(np.float32)
        )
        result.loc[in_mask, 'assignment_code'] = np.uint8(1)

        if verbose:
            for k in range(n_channels):
                n_k = int((result.loc[in_mask, 'channel'] == k).sum())
                logger.info(f"  Channel {k}: {n_k:,} blink events from clusters")

        # ── Step 6: classify isolated blink events ─────────────────────────
        iso_mask = result['joint_cluster_id'] == -1
        n_iso_assigned = 0

        if iso_mask.sum() > 0:
            if verbose:
                logger.info("")
                logger.info(f"Step 5: Classifying {n_isolated:,} isolated blink events "
                            f"(threshold = {confidence_threshold_isolated})...")

            X_iso = result.loc[iso_mask, channels_to_use].to_numpy(dtype=np.float64)
            iso_post = gmm.predict_proba(X_iso)
            iso_ch = np.argmax(iso_post, axis=1)
            iso_conf = iso_post[np.arange(len(X_iso)), iso_ch]
            passes = iso_conf >= confidence_threshold_isolated

            iso_idx = result.index[iso_mask]
            result.loc[iso_idx[passes], 'channel'] = iso_ch[passes].astype(np.int32)
            result.loc[iso_idx[passes], 'channel_confidence'] = iso_conf[passes].astype(np.float32)
            result.loc[iso_idx[passes], 'assignment_code'] = np.uint8(2)
            n_iso_assigned = int(passes.sum())

            if verbose:
                logger.info(f"  Assigned: {n_iso_assigned:,},  "
                            f"Rejected: {int((~passes).sum()):,}")

        # ── Summary ────────────────────────────────────────────────────────
        n_total_assigned = int((result['channel'] >= 0).sum())
        n_total_rejected = int((result['channel'] == -1).sum())

        if verbose:
            logger.info("")
            logger.info("=" * 70)
            pct = 100.0 * n_total_assigned / max(len(result), 1)
            logger.info(f"Total assigned : {n_total_assigned:,} / {len(result):,}  ({pct:.1f}%)")
            logger.info(f"Total rejected : {n_total_rejected:,}")
            logger.info("=" * 70)

        metadata = {
            'n_clusters': n_clusters,
            'n_clustered': n_in_clusters,
            'n_isolated': n_isolated,
            'means': means,
            'covariances': covariances,
            'weights': weights,
            'gmm': gmm,
            'n_assigned': {k: int((result['channel'] == k).sum()) for k in range(n_channels)},
            'n_unassigned': n_total_rejected,
            'n_assigned_from_clusters': int(
                (result.loc[in_mask, 'channel'] >= 0).sum()
            ),
            'n_assigned_from_isolated': n_iso_assigned,
        }

        if plot_results:
            self._plot_joint_cluster_results(
                result, channels_to_use, gmm, n_channels, cluster_X, loc_data
            )

        return result, metadata

    def _plot_joint_cluster_results(
        self,
        result: pd.DataFrame,
        channels_to_use: list,
        gmm,
        n_channels: int,
        cluster_X: np.ndarray,
        loc_data_original: pd.DataFrame = None,
    ) -> None:
        """Diagnostic plots for joint-cluster unmixing.

        Figure 1 — spectral scatter:
          Left:  per-cluster means (regular scatter, small N).
          Right: individual blink events coloured by channel, rendered with
                 datashader so N=10⁶+ points draw in milliseconds.

        Figure 2 — spectral histograms:
          One panel per spectral feature (A_R, A_G, …).
          Grey: original distribution before unmixing.
          Coloured: per-channel assigned distribution.
          The per-channel distributions should be substantially narrower than
          the original, confirming that unmixing has separated the species.
        """
        try:
            # Consistent colour palette for channels
            colours_ch  = ['tab:blue', 'tab:red',   'tab:green', 'tab:orange']
            colours_hex = ['#1f77b4',  '#d62728',   '#2ca02c',   '#ff7f0e'  ]

            cluster_ch = gmm.predict(cluster_X)

            # ── Figure 1: scatter ─────────────────────────────────────────
            fig1, axs = plt.subplots(1, 2, figsize=(10, 4))

            # Left panel — cluster means (hundreds–thousands of points)
            for k in range(n_channels):
                mask = cluster_ch == k
                axs[0].scatter(
                    cluster_X[mask, 0], cluster_X[mask, 1],
                    s=6, alpha=0.6, c=colours_ch[k % len(colours_ch)],
                    label=f'Channel {k}', rasterized=True,
                )
            axs[0].set_xlabel(channels_to_use[0])
            axs[0].set_ylabel(channels_to_use[1])
            axs[0].set_title('Per-cluster spectral means')
            axs[0].legend(markerscale=3, fontsize=8)

            # Right panel — individual blink events, per-layer datashader
            # Each channel is rendered separately then composited with tf.stack,
            # avoiding the integer category-key issues in plot_multi_dataset_scatter.
            import matplotlib.colors as mcolors
            layers_info = []
            for k in range(n_channels):
                sub = result[result['channel'] == k]
                if len(sub) == 0:
                    continue
                layers_info.append((
                    sub[channels_to_use[0]].to_numpy(dtype=np.float64),
                    sub[channels_to_use[1]].to_numpy(dtype=np.float64),
                    colours_hex[k % len(colours_hex)],
                    f'Channel {k} (n={len(sub):,})',
                ))
            unassigned = result[result['channel'] == -1]
            if len(unassigned):
                layers_info.append((
                    unassigned[channels_to_use[0]].to_numpy(dtype=np.float64),
                    unassigned[channels_to_use[1]].to_numpy(dtype=np.float64),
                    '#aaaaaa',
                    f'Unassigned (n={len(unassigned):,})',
                ))

            if layers_info:
                try:
                    import datashader as ds
                    import matplotlib.colors as mcolors

                    all_x = np.concatenate([li[0] for li in layers_info])
                    all_y = np.concatenate([li[1] for li in layers_info])
                    x_range = (float(all_x.min()), float(all_x.max()))
                    y_range = (float(all_y.min()), float(all_y.max()))
                    W, H = 500, 500
                    cvs = ds.Canvas(plot_width=W, plot_height=H,
                                    x_range=x_range, y_range=y_range)

                    # Additive RGBA composite built directly from count arrays.
                    # Avoids datashader colormap alpha issues entirely.
                    composite = np.zeros((H, W, 4), dtype=np.float32)
                    for x_arr, y_arr, colour, _ in layers_info:
                        df_i = pd.DataFrame({'x': x_arr, 'y': y_arr})
                        density = cvs.points(df_i, 'x', 'y').values.astype(np.float32)
                        density = np.where(np.isnan(density), 0.0, density)
                        log_d = np.log1p(np.maximum(density, 0.0))
                        max_ld = float(log_d.max())
                        alpha_ch = (log_d / max_ld) if max_ld > 0 else log_d
                        rgb = np.array(mcolors.to_rgb(colour), dtype=np.float32)
                        composite[:, :, :3] += (
                            rgb[np.newaxis, np.newaxis, :] * alpha_ch[:, :, np.newaxis]
                        )
                        composite[:, :, 3] = np.clip(
                            composite[:, :, 3] + alpha_ch, 0.0, 1.0
                        )

                    img_arr = (np.clip(composite, 0.0, 1.0) * 255).astype(np.uint8)
                    axs[1].imshow(img_arr,
                                  extent=[x_range[0], x_range[1],
                                          y_range[0], y_range[1]],
                                  origin='lower', aspect='auto',
                                  interpolation='bilinear')
                    total_pts = sum(len(li[0]) for li in layers_info)
                    axs[1].text(0.02, 0.98, f'Datashader: {total_pts:,} pts',
                                transform=axs[1].transAxes, fontsize=7, va='top',
                                alpha=0.7, bbox=dict(boxstyle='round',
                                facecolor='white', alpha=0.5))

                except ImportError:
                    import matplotlib.colors as mcolors
                    for x_arr, y_arr, colour, _ in layers_info:
                        axs[1].scatter(x_arr, y_arr, s=1, alpha=0.2,
                                       c=colour, rasterized=True)

                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor=li[2], label=li[3]) for li in layers_info
                ]
                axs[1].legend(handles=legend_elements, fontsize=7)

            axs[1].set_xlabel(channels_to_use[0])
            axs[1].set_ylabel(channels_to_use[1])
            axs[1].set_title('Individual blink events')

            _safe_tight_layout(fig1)
            plt.show()

            # ── Figure 2: spectral histograms ─────────────────────────────
            n_feat = len(channels_to_use)
            fig2, axs2 = plt.subplots(1, n_feat, figsize=(4.5 * n_feat, 3.5))
            if n_feat == 1:  # pragma: no cover -- Figure 1 above unconditionally
                # indexes cluster_X[:, 1] and channels_to_use[1], so a real
                # single-feature call raises there first (caught by this
                # function's own broad except below) and never reaches here.
                axs2 = [axs2]

            n_bins = 120
            orig_df = loc_data_original if loc_data_original is not None else result

            for i, col in enumerate(channels_to_use):
                ax = axs2[i]

                # Common bin range across original + result
                col_min = min(orig_df[col].min(), result[col].min())
                col_max = max(orig_df[col].max(), result[col].max())
                bins = np.linspace(col_min, col_max, n_bins + 1)

                # Original distribution — grey
                orig_vals = orig_df[col].to_numpy()
                orig_h, _ = np.histogram(orig_vals, bins=bins, density=True)
                ax.stairs(orig_h, bins, fill=True, alpha=0.20,
                          color='gray', label=f'Original (n={len(orig_vals):,})')
                ax.stairs(orig_h, bins, fill=False, alpha=0.60,
                          color='gray', linewidth=0.8)

                # Per-channel distributions
                for k in range(n_channels):
                    sub_vals = result.loc[result['channel'] == k, col].to_numpy()
                    if len(sub_vals) == 0:
                        continue
                    h, _ = np.histogram(sub_vals, bins=bins, density=True)
                    c = colours_ch[k % len(colours_ch)]
                    ax.stairs(h, bins, fill=True, alpha=0.30,
                              color=c, label=f'Channel {k} (n={len(sub_vals):,})')
                    ax.stairs(h, bins, fill=False, alpha=0.90,
                              color=c, linewidth=1.0)

                ax.set_xlabel(col, fontsize=9)
                ax.set_ylabel('Density' if i == 0 else '', fontsize=9)
                ax.set_title(f'{col} distribution', fontsize=9)
                ax.legend(fontsize=7)

            _safe_tight_layout(fig2)
            plt.show()

        except Exception as e:
            logger.warning(f"Could not create joint-cluster diagnostic plot: {e}")

    def find_exemplar_dye_pair(
        self,
        sf_db: pd.DataFrame,
        mean_0: NDArray[np.float64],
        mean_1: NDArray[np.float64],
        spectral_tol: float = 0.05,
        min_spatial_dist_nm: float = 500.0,
        max_spatial_dist_nm: float | None = None,
        min_photoelectrons: float = 1000.0,
        pixel_size: float | None = None,
        n_top: int = 10,
    ) -> pd.DataFrame | None:
        """Find a co-localised pair of single-frame localisations representing two dye classes.

        Searches the **single-frame** database so that both localisations must
        appear in the same frame of the same FOV — i.e. they are simultaneously
        visible.  Candidates in each class are those whose (A_R, A_G) lies within
        `spectral_tol` of the class mean.  Pairs are ranked by spatial separation
        so the result can be used directly as an exemplar figure.

        Args:
            sf_db: Single-frame DataFrame as returned by extract_single_molecules_*.
                   Must contain: xc, yc, A_R, A_G, photoelectrons, frame, fov_index, molecular_index.
            mean_0: (A_R, A_G) mean for class 0 (e.g. from GMM fixed_means[0]).
            mean_1: (A_R, A_G) mean for class 1 (e.g. from GMM fixed_means[1]).
            spectral_tol: Maximum Euclidean distance in (A_R, A_G) space for a
                          localisation to be accepted as a candidate for a class.
                          Start with 0.05; relax if no pairs found.
            min_spatial_dist_nm: Minimum separation (nm) — excludes pairs that are
                                 too close to resolve in the raw image (default 500).
            max_spatial_dist_nm: Maximum separation (nm).  None means no upper limit.
            min_photoelectrons: Minimum detected photoelectrons for a localisation to be
                         considered (default 1000).
            pixel_size: Camera pixel size in nm (default 69.0 nm for Ximea).
            n_top: Number of best pairs to return (ranked by spatial_dist_nm).

        Returns:
            pd.DataFrame with one row per candidate pair, sorted by
            spatial_dist_nm (closest first), containing:
              fov_index, frame, mol_0_idx, mol_1_idx,
              xc_0, yc_0, A_R_0, A_G_0, spec_dist_0,
              xc_1, yc_1, A_R_1, A_G_1, spec_dist_1,
              spatial_dist_nm, spectral_score
            Returns None if no pairs satisfy the constraints.
        """
        if pixel_size is None:
            pixel_size = self.pixel_size * 1000  # µm → nm

        import pandas as pd
        from scipy.spatial.distance import cdist

        mean_0 = np.asarray(mean_0, dtype=float)
        mean_1 = np.asarray(mean_1, dtype=float)

        sf_db = sf_db[sf_db['photoelectrons'] >= min_photoelectrons]

        ar = sf_db['A_R'].to_numpy()
        ag = sf_db['A_G'].to_numpy()

        spec_dist_0 = np.sqrt((ar - mean_0[0])**2 + (ag - mean_0[1])**2)
        spec_dist_1 = np.sqrt((ar - mean_1[0])**2 + (ag - mean_1[1])**2)

        cands_0 = sf_db[spec_dist_0 <= spectral_tol].copy()
        cands_0['spec_dist'] = spec_dist_0[spec_dist_0 <= spectral_tol]

        cands_1 = sf_db[spec_dist_1 <= spectral_tol].copy()
        cands_1['spec_dist'] = spec_dist_1[spec_dist_1 <= spectral_tol]

        if len(cands_0) == 0 or len(cands_1) == 0:
            logger.info(f"No candidates found within spectral_tol={spectral_tol:.3f}. " f"Class 0: {len(cands_0)} candidates, Class 1: {len(cands_1)} candidates. " "Try increasing spectral_tol.")
            return None

        fov_col = 'fov_index' if 'fov_index' in sf_db.columns else 'fov_name'

        # Group by (fov, frame) so both localisations must be in the same frame.
        common_groups = set(
            map(tuple, cands_0[[fov_col, 'frame']].drop_duplicates().values)
        ) & set(
            map(tuple, cands_1[[fov_col, 'frame']].drop_duplicates().values)
        )

        if len(common_groups) == 0:
            logger.info("No (FOV, frame) group contains candidates from both classes.")
            return None

        all_pairs = []
        for fov, frame in common_groups:
            f0 = cands_0[(cands_0[fov_col] == fov) & (cands_0['frame'] == frame)]
            f1 = cands_1[(cands_1[fov_col] == fov) & (cands_1['frame'] == frame)]

            xy_0 = f0[['xc', 'yc']].to_numpy() * pixel_size  # nm
            xy_1 = f1[['xc', 'yc']].to_numpy() * pixel_size

            dists_nm = cdist(xy_0, xy_1)  # (n0, n1)

            i_flat, j_flat = np.meshgrid(
                np.arange(len(f0)), np.arange(len(f1)), indexing='ij'
            )
            i_flat = i_flat.ravel()
            j_flat = j_flat.ravel()
            d_flat = dists_nm.ravel()

            keep = d_flat >= min_spatial_dist_nm
            if max_spatial_dist_nm is not None:
                keep &= d_flat <= max_spatial_dist_nm
            i_flat, j_flat, d_flat = i_flat[keep], j_flat[keep], d_flat[keep]

            if len(d_flat) == 0:
                continue

            mol0 = f0.iloc[i_flat]
            mol1 = f1.iloc[j_flat]

            sd0 = mol0['spec_dist'].to_numpy()
            sd1 = mol1['spec_dist'].to_numpy()

            pairs = pd.DataFrame({
                fov_col:           fov,
                'frame':           frame,
                'mol_0_idx':       mol0['molecular_index'].to_numpy(),
                'xc_0':            mol0['xc'].to_numpy(),
                'yc_0':            mol0['yc'].to_numpy(),
                'A_R_0':           mol0['A_R'].to_numpy(),
                'A_G_0':           mol0['A_G'].to_numpy(),
                'spec_dist_0':     sd0,
                'mol_1_idx':       mol1['molecular_index'].to_numpy(),
                'xc_1':            mol1['xc'].to_numpy(),
                'yc_1':            mol1['yc'].to_numpy(),
                'A_R_1':           mol1['A_R'].to_numpy(),
                'A_G_1':           mol1['A_G'].to_numpy(),
                'spec_dist_1':     sd1,
                'spatial_dist_nm': d_flat,
                'spectral_score':  sd0 + sd1,
            })
            all_pairs.append(pairs)

        if not all_pairs:
            logger.info("No pairs found within spatial distance constraints.")
            return None

        result = pd.concat(all_pairs, ignore_index=True)
        # A molecule within spectral_tol of both means would appear in both cands_0 and
        # cands_1, producing a self-pair.  Remove any row where mol_0 == mol_1.
        result = result[result['mol_0_idx'] != result['mol_1_idx']]
        if len(result) == 0:
            logger.info("All pairs were self-pairs (same molecule in both classes). " "Try reducing spectral_tol.")
            return None
        result = result.sort_values('spatial_dist_nm').head(n_top).reset_index(drop=True)

        logger.info(f"Found {len(result)} candidate pairs from {len(common_groups)} (FOV, frame) group(s).")
        logger.info(f"Best pair: FOV={result.iloc[0][fov_col]}, frame={result.iloc[0]['frame']}, " f"dist={result.iloc[0]['spatial_dist_nm']:.0f} nm, " f"mol_0 A_R={result.iloc[0]['A_R_0']:.3f} A_G={result.iloc[0]['A_G_0']:.3f}, " f"mol_1 A_R={result.iloc[0]['A_R_1']:.3f} A_G={result.iloc[0]['A_G_1']:.3f}")
        return result

    def get_exemplar_crop(
        self,
        pair_row: pd.Series,
        data_folder: str | Path,
        crop_size_px: int = 30,
    ) -> tuple[NDArray[np.float32], pd.DataFrame]:
        """Load the raw TIFF for the FOV in pair_row and return a single-frame crop.

        The TIFF is located by sorting all TIFFs in data_folder and picking the
        one at position fov_index.  The specific frame stored in pair_row['frame']
        is extracted directly — no projection — so the crop shows exactly what the
        camera captured when both molecules were localised.

        Args:
            pair_row: A single row (pd.Series) from find_exemplar_dye_pair,
                      typically result.iloc[0] for the best pair.
            data_folder: Directory containing the raw TIFF files.
            crop_size_px: Half-width of the square crop in camera pixels.
                          The returned image is (2*crop_size_px) × (2*crop_size_px).

        Returns:
            crop: 2D np.ndarray — the raw single-frame crop.
            pair_info: pd.DataFrame (single row) — the input pair_row with extra
                       columns: xc_0_crop, yc_0_crop, xc_1_crop, yc_1_crop
                       giving molecule positions relative to the crop origin.
        """
        fov_index = int(pair_row['fov_index'])
        frame_index = int(pair_row['frame'])

        # Collect and sort all TIFFs in the folder, then pick the Nth one.
        tif_files = sorted(
            p for p in Path(data_folder).glob("*.tif*")
            if p.suffix != '.h5'
        )

        if len(tif_files) == 0:
            raise FileNotFoundError(f"No TIFF files found in '{data_folder}'.")
        if fov_index >= len(tif_files):
            raise IndexError(
                f"fov_index={fov_index} but only {len(tif_files)} TIFFs found in "
                f"'{data_folder}'."
            )
        tif_path = tif_files[fov_index]
        logger.info(f"Loading FOV {fov_index}, frame {frame_index}: {Path(tif_path).name}")

        # Load a single frame directly via IO.read_tiff
        projected = self.io.read_tiff(tif_path, frame=frame_index)

        # Crop centre: midpoint of the two molecule positions (camera pixels)
        cx = int(round(0.5 * (pair_row['xc_0'] + pair_row['xc_1'])))
        cy = int(round(0.5 * (pair_row['yc_0'] + pair_row['yc_1'])))

        H, W = projected.shape
        x0 = max(cx - crop_size_px, 0)
        x1 = min(cx + crop_size_px, W)
        y0 = max(cy - crop_size_px, 0)
        y1 = min(cy + crop_size_px, H)

        crop = projected[y0:y1, x0:x1]

        # Molecule positions relative to crop origin
        pair_info = pair_row.to_frame().T.copy().reset_index(drop=True)
        pair_info['xc_0'] = pair_row['xc_0'] - x0
        pair_info['yc_0'] = pair_row['yc_0'] - y0
        pair_info['xc_1'] = pair_row['xc_1'] - x0
        pair_info['yc_1'] = pair_row['yc_1'] - y0
        pair_info['crop_x0_px'] = x0
        pair_info['crop_y0_px'] = y0

        return crop, pair_info
