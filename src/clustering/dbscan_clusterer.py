# -*- coding: utf-8 -*-
"""
clustering/dbscan_clusterer.py

DBSCAN-based single-molecule extraction mixin.
Extracted from SM_extractionfunctions.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pyS3M.Constants import FilteringConstants, FilteringCriteria
from ._config import ClusteringConfig
from sklearn.cluster import DBSCAN
import logging
logger = logging.getLogger(__name__)


class DBSCANMixin:
    """Mixin supplying DBSCAN-based single-molecule extraction."""

    def extract_single_molecules_DBSCAN(
        self,
        loc_data: pd.DataFrame,
        min_cluster_size: int = 10,
        criteria: FilteringCriteria = None,
        chi_val: float | None = None,
        max_localisation_error: float = FilteringConstants.MAX_LOCALISATION_ERROR_PX,
        max_colour_error: float = FilteringConstants.MAX_COLOUR_ERROR,
        min_sigma: float | None = None,
        max_sigma: float | None = None,
        max_sigma_error: float | None = None,
        min_photons: float = FilteringConstants.MIN_PHOTONS,
        max_photons: float | None = None,
        min_photoelectrons: float | None = None,
        max_photoelectrons: float | None = None,
        epsilon_multiplier: float = 1.0,
        start_frame: int = 0,
        config: ClusteringConfig = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract single molecules from localisation data by DBSCAN clustering.

        Args:
            loc_data (pd.DataFrame): Localisation data to process.
            min_cluster_size (int): Minimum cluster size (min_samples for DBSCAN).
            criteria (FilteringCriteria, optional): Quality filter bundle.
            chi_val: Quality filter parameters (chi_val, max_localisation_error,
                max_colour_error, min_sigma, max_sigma, max_sigma_error,
                min_photons, max_photons, min_photoelectrons, max_photoelectrons — see
                filter_quality_localisations).
            epsilon_multiplier (float): Multiplier applied to the mean localisation
                precision to derive the DBSCAN epsilon radius.
            start_frame (int): Discard localisations before this frame.

        Returns:
            tuple: (single_molecule_database, single_frame_database) as DataFrames.
                single_frame_database includes molecular_index column and
                excludes unassigned localisations.
        """
        if config is not None:
            min_cluster_size   = config.min_cluster_size
            epsilon_multiplier = config.epsilon_multiplier
            start_frame        = config.start_frame

        loc_data = self._prepare_locs(
            loc_data, start_frame, criteria, chi_val,
            max_localisation_error, max_colour_error,
            min_sigma, max_sigma, max_sigma_error,
            min_photons, max_photons,
            min_photoelectrons, max_photoelectrons,
        )

        if not self._check_min_locs(loc_data, min_cluster_size):
            return pd.DataFrame(), pd.DataFrame()

        X = np.vstack([loc_data["xc"], loc_data["yc"]]).T
        loc_precision = 0.5 * (
            np.mean(loc_data["xc_err"]) + np.mean(loc_data["yc_err"])
        )
        if loc_precision <= 0:
            raise ValueError(
                f"loc_precision = {loc_precision:.6f} (computed from mean xc_err / yc_err). "
                "Error columns are zero or NaN — this usually means chi_sqr was very small "
                "(bright spots cause pcov * chisqr → 0) or errors were not saved. "
                "Check that ImageAnalysisFunctions.calculate_errors returns non-zero values "
                "for your data, or pass epsilon_multiplier with an explicit eps value."
            )

        hdb = DBSCAN(
            min_samples=min_cluster_size,
            eps=loc_precision * epsilon_multiplier,
        )
        hdb.fit(X)

        return self._finish_clustering(loc_data, hdb.labels_)
