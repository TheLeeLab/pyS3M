# -*- coding: utf-8 -*-
"""
This class contains functions pertaining to analysis of images,
relating to the bayerSMLM concept.
jsb92, 2024/01/02
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
import pyS3M.IOFunctions as IOFunctions
from pyS3M.Constants import FilteringConstants, FilteringCriteria
from pyS3M.clustering import ClusteringBaseMixin, HDBSCANMixin, DBSCANMixin, LinkedMixin, BatchMixin
from pyS3M.mixture_analysis import MixtureAnalysisMixin
from pyS3M.channel_unmixing import ChannelUnmixingMixin



class extract_SMs(HDBSCANMixin, DBSCANMixin, LinkedMixin, BatchMixin,
                  MixtureAnalysisMixin, ChannelUnmixingMixin,
                  ClusteringBaseMixin):
    def __init__(self, camera: str = "ximea", pixel_size: float = None, io_functions=None, mosaic_unit=None) -> None:
        """Single molecule extraction functions for clustering localizations into single molecules.

        Args:
            camera: Camera model name (``"ximea"`` or ``"zwo"``). Sets pixel_size
                and therefore sigma bounds if not overridden explicitly.
            pixel_size: Physical pixel size in µm. If None, taken from camera defaults.
            io_functions: IO functions instance (default: creates new instance)
            mosaic_unit: Mosaic pattern; sets the pixel fractions used for ``photoelectrons``.
                If None, taken from camera defaults.
        """
        import pyS3M.CameraDefaults as CameraDefaults
        config = CameraDefaults.get_camera_config(camera)
        self.pixel_size = pixel_size if pixel_size is not None else config.pixel_size

        # Dependency injection with sensible defaults
        self.io = (
            io_functions if io_functions is not None else IOFunctions.IO_Functions(camera=camera, mosaic_unit=mosaic_unit)
        )

    def _load_localisation_files(self, loc_data, start_frame=0):
        """Load localisations from HDF5 file paths if needed.

        Args:
            loc_data: Either a pd.DataFrame (returned as-is) or a list/array of
                      HDF5 file paths (concatenated and returned as pd.DataFrame).
            start_frame (int): Discard all localisations with frame < start_frame.
                               Default 0 (keep all).

        Returns:
            pd.DataFrame of localisation data.
        """
        if isinstance(loc_data, pd.DataFrame):
            df = loc_data
        else:
            # Treat as iterable of file paths
            dfs = []
            for f in loc_data:
                dfs.append(self.io.read_h5_database(str(f)))
            if not dfs:
                return pd.DataFrame()
            df = pd.concat(dfs, ignore_index=True)

        if start_frame > 0:
            df = df[df["frame"] >= start_frame].reset_index(drop=True)
        return df

    def filter_quality_localisations(
        self,
        loc_data,
        chi_val=None,
        max_localisation_error=FilteringConstants.MAX_LOCALISATION_ERROR_PX,
        max_colour_error=FilteringConstants.MAX_COLOUR_ERROR,
        min_sigma=None,       # px; None → FilteringConstants.MIN_SIGMA_NM / self.pixel_size_nm
        max_sigma=None,       # px; None → FilteringConstants.MAX_SIGMA_NM / self.pixel_size_nm
        max_sigma_error=None, # px; None → FilteringConstants.MAX_SIGMA_ERROR_NM / self.pixel_size_nm
        min_photons=FilteringConstants.MIN_PHOTONS,
        max_photons=None,
        min_photoelectrons=None,
        max_photoelectrons=None,
        criteria: FilteringCriteria = None,
    ):
        """Apply quality filters to localisation data.

        Individual keyword arguments are used when ``criteria`` is ``None``
        (backwards-compatible path).  Pass a :class:`~Constants.FilteringCriteria`
        instance to replace all individual kwargs with a single object.

        Args:
            loc_data (pd.DataFrame): Localization data to filter.
            chi_val: Chi-squared threshold (localisations with ``chi_sqr`` at or
                above this are dropped); ``None`` → no chi-squared filtering.
            max_localisation_error: Maximum localisation precision (pixels).
            max_colour_error: Maximum amplitude error fraction.
            min_sigma, max_sigma, max_sigma_error: PSF sigma bounds (pixels);
                ``None`` → derived from ``FilteringConstants`` / pixel size.
            min_photons, max_photons: Bounds on ``photons`` (A_B + A_G + A_R, which overcounts
                the detected photoelectrons 2-4x; see ``IO_Functions.add_photoelectron_columns``).
            min_photoelectrons, max_photoelectrons: Bounds on the detected ``photoelectrons``
                (sum_p f_p * A_p); ``None`` → no cut. Applied in addition to the photon bounds.
            criteria: If provided, overrides all individual kwargs above.

        Returns:
            pd.DataFrame: Filtered localisation data.
        """
        if criteria is not None:
            chi_val               = criteria.chi_val
            max_localisation_error = criteria.max_localisation_error
            max_colour_error      = criteria.max_colour_error
            min_sigma             = criteria.min_sigma
            max_sigma             = criteria.max_sigma
            max_sigma_error       = criteria.max_sigma_error
            min_photons           = criteria.min_photons
            max_photons           = criteria.max_photons
            min_photoelectrons    = criteria.min_photoelectrons
            max_photoelectrons    = criteria.max_photoelectrons
        # Resolve sigma bounds from camera pixel size if not explicitly provided
        pixel_size_nm = self.pixel_size * 1000  # µm → nm
        if min_sigma is None:
            min_sigma = FilteringConstants.MIN_SIGMA_NM / pixel_size_nm
        if max_sigma is None:
            max_sigma = FilteringConstants.MAX_SIGMA_NM / pixel_size_nm
        if max_sigma_error is None:
            max_sigma_error = FilteringConstants.MAX_SIGMA_ERROR_NM / pixel_size_nm

        # Apply quality filters. chi_val=None means "no chi-squared filter" -- unlike
        # every other threshold here, there's no data-independent default worth
        # picking (a per-fit chi-squared depends on PSF model, background, and photon
        # count), so we skip the cut entirely rather than fabricate one from the
        # data's own median (which would silently discard the worse-fitting half of
        # *any* input, however good, and make results depend on batch composition).
        filtered_data = loc_data.copy()
        if chi_val is not None:
            filtered_data = filtered_data[filtered_data["chi_sqr"] < chi_val]
        filtered_data = filtered_data[filtered_data["xc_err"] < max_localisation_error]
        filtered_data = filtered_data[filtered_data["yc_err"] < max_localisation_error]
        for key in ['A_R_err', 'A_G_err', 'A_B_err', 'bg_R_err', 'bg_G_err', 'bg_B_err']:
            filtered_data = filtered_data[filtered_data[key] < max_colour_error]
        for key in ['s_x_err', 's_y_err']:
            filtered_data = filtered_data[filtered_data[key] < max_sigma_error]
        for key in ['s_x', 's_y']:
            filtered_data = filtered_data[filtered_data[key] < max_sigma]
            filtered_data = filtered_data[filtered_data[key] > min_sigma]

        # Add photons column using centralized method and apply photon count filters
        if "photons" not in filtered_data.columns:
            filtered_data = self.io._add_photon_columns(filtered_data, normalise=True)
        if max_photons is not None:
            filtered_data = filtered_data[filtered_data["photons"] < max_photons]
        filtered_data = filtered_data[filtered_data["photons"] > min_photons]
        if min_photoelectrons is not None or max_photoelectrons is not None:
            if "photoelectrons" not in filtered_data.columns:
                filtered_data = self.io.add_photoelectron_columns(filtered_data)
            if min_photoelectrons is not None:
                filtered_data = filtered_data[filtered_data["photoelectrons"] > min_photoelectrons]
            if max_photoelectrons is not None:
                filtered_data = filtered_data[filtered_data["photoelectrons"] < max_photoelectrons]

        return filtered_data.reset_index(drop=True)

    def average_parameters(self, data, dbscan_labels):
        """
        Average parameters for clustered single molecule localizations.

        Args:
            data (pd.DataFrame): Localization data with fitting parameters
            dbscan_labels (np.array): Cluster labels from DBSCAN/HDBSCAN

        Returns:
            pd.DataFrame: Averaged parameters per single molecule
        """
        labels = np.sort(np.unique(dbscan_labels))
        labels = labels[labels > -1]
        dict_obj = {}
        dict_obj["photons"] = np.zeros(len(labels))
        dict_obj["frames"] = np.zeros(len(labels))
        for column in np.array(data.columns):
            if column == "index":
                continue
            else:
                dict_obj[column] = np.zeros(len(labels))

        for label in labels:
            for column in np.array(data.columns):
                if column == "index":
                    continue
                elif column in ["A_B", "A_G", "A_R"]:
                    dict_obj[column][label] = np.average(
                        data[column][dbscan_labels == label],
                        weights=data[column + "_err"][dbscan_labels == label] ** -2,
                    )
                else:
                    dict_obj[column][label] = np.mean(
                        data[column][dbscan_labels == label]
                    )
            dict_obj["frames"][label] = len(data[column][dbscan_labels == label])
            # Use existing photons column if available, otherwise calculate manually
            if "photons" in data.columns:
                dict_obj["photons"][label] = np.sum(
                    data["photons"][dbscan_labels == label]
                )
            else:
                dict_obj["photons"][label] = (
                    np.average(
                        data["A_B"][dbscan_labels == label],
                        weights=data["A_B_err"][dbscan_labels == label] ** -2,
                    )
                    + np.average(
                        data["A_G"][dbscan_labels == label],
                        weights=data["A_G_err"][dbscan_labels == label] ** -2,
                    )
                    + np.average(
                        data["A_R"][dbscan_labels == label],
                        weights=data["A_R_err"][dbscan_labels == label] ** -2,
                    )
                )
        df = pd.DataFrame.from_dict(dict_obj)
        # Normalise photon fractions using centralised IOFunctions method
        return df

    # ------------------------------------------------------------------
    # Clustering methods: HDBSCANMixin, DBSCANMixin, LinkedMixin, BatchMixin
    #   → src/clustering/
    # GMM mixture-analysis methods: MixtureAnalysisMixin
    #   → src/mixture_analysis.py
    # Channel-unmixing methods: ChannelUnmixingMixin
    #   → src/channel_unmixing.py
    # ------------------------------------------------------------------

