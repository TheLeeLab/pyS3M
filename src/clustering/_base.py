# -*- coding: utf-8 -*-
"""
clustering/_base.py

Shared boilerplate helpers for clustering mixins.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pyS3M.Constants import FilteringConstants, FilteringCriteria
import logging
logger = logging.getLogger(__name__)


class ClusteringBaseMixin:
    """Shared helpers composed into DBSCAN, HDBSCAN, and linked clusterers."""

    def _prepare_locs(
        self,
        loc_data: pd.DataFrame,
        start_frame: int,
        criteria: FilteringCriteria | None,
        chi_val: float | None,
        max_localisation_error: float,
        max_colour_error: float,
        min_sigma: float | None,
        max_sigma: float | None,
        max_sigma_error: float | None,
        min_photons: float,
        max_photons: float | None,
        min_photoelectrons: float | None = None,
        max_photoelectrons: float | None = None,
    ) -> pd.DataFrame:
        """Load from file if needed, then quality-filter localisations."""
        loc_data = self._load_localisation_files(loc_data, start_frame=start_frame)
        loc_data = self.filter_quality_localisations(
            loc_data=loc_data, criteria=criteria,
            chi_val=chi_val, max_localisation_error=max_localisation_error,
            min_photons=min_photons, max_photons=max_photons,
            min_photoelectrons=min_photoelectrons, max_photoelectrons=max_photoelectrons,
            max_colour_error=max_colour_error,
            min_sigma=min_sigma, max_sigma=max_sigma,
            max_sigma_error=max_sigma_error,
        )
        return loc_data

    def _check_min_locs(self, loc_data: pd.DataFrame, min_count: int) -> bool:
        """Return False (and warn) if too few localisations remain after filtering."""
        if len(loc_data) == 0:
            logger.warning(
                "Warning: No localizations remaining after filtering. "
                "Returning empty databases."
            )
            return False
        if len(loc_data) < min_count:
            logger.warning(
                f"Warning: Only {len(loc_data)} localizations remaining after filtering, "
                f"but min_cluster_size={min_count}. Need at least {min_count} points. "
                f"Returning empty databases."
            )
            return False
        return True

    def _finish_clustering(
        self,
        loc_data: pd.DataFrame,
        labels: np.ndarray,
        molecular_index_offset: int = 0,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Assign labels → average per cluster → return (molecule_db, frame_db)."""
        assigned_mask = labels >= 0
        loc_data_assigned = loc_data[assigned_mask].copy()
        labels_assigned = labels[assigned_mask]
        loc_data_assigned["molecular_index"] = labels_assigned + molecular_index_offset
        df = self.average_parameters(loc_data_assigned, labels_assigned)
        df["molecular_index"] = df.index + molecular_index_offset
        return df, loc_data_assigned
