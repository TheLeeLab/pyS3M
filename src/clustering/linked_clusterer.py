# -*- coding: utf-8 -*-
"""
clustering/linked_clusterer.py

Temporal-linking and spectral-LAP single-molecule extraction mixins.
Extracted from SM_extractionfunctions.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pyS3M.Constants import FilteringConstants, FilteringCriteria
from ._config import ClusteringConfig
import pyS3M.postprocess as postprocess
import logging
logger = logging.getLogger(__name__)



class LinkedMixin:
    """Mixin supplying temporal-linking and spectral-LAP extraction methods."""

    # ------------------------------------------------------------------
    # Temporal linking (postprocess.get_link_groups)
    # ------------------------------------------------------------------

    def extract_single_molecules_linked(
        self,
        loc_data: pd.DataFrame,
        max_distance: float = 1.0,
        max_frames: int = 10,
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
        start_frame: int = 0,
        config: ClusteringConfig = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract single molecules using temporal linking (postprocess.get_link_groups).

        Args:
            loc_data (pd.DataFrame): Localisation data to process.
            max_distance (float): Maximum linking distance in pixels.
            max_frames (int): Maximum frame gap for linking.
            criteria (FilteringCriteria, optional): Quality filter bundle.
            chi_val (float, optional): Quality filter parameter (see filter_quality_localisations).
            max_localisation_error (float): Quality filter parameter (see filter_quality_localisations).
            max_colour_error (float): Quality filter parameter (see filter_quality_localisations).
            min_sigma (float, optional): Quality filter parameter (see filter_quality_localisations).
            max_sigma (float, optional): Quality filter parameter (see filter_quality_localisations).
            max_sigma_error (float, optional): Quality filter parameter (see filter_quality_localisations).
            min_photons (float): Quality filter parameter (see filter_quality_localisations).
            max_photons (float, optional): Quality filter parameter (see filter_quality_localisations).
            min_photoelectrons, max_photoelectrons (float, optional): Bounds on the detected
                photoelectrons (see filter_quality_localisations); None → no cut.
            start_frame (int): Discard localisations before this frame.

        Returns:
            tuple: (single_molecule_database, single_frame_database) as DataFrames.
                   single_frame_database includes molecular_index column and
                   excludes unlinked localisations.
        """
        if config is not None:
            max_distance = config.max_distance
            max_frames   = config.max_frames
            start_frame  = config.start_frame

        loc_data = self._prepare_locs(
            loc_data, start_frame, criteria, chi_val,
            max_localisation_error, max_colour_error,
            min_sigma, max_sigma, max_sigma_error,
            min_photons, max_photons,
            min_photoelectrons, max_photoelectrons,
        )

        loc_data_sorted = loc_data.sort_values("frame")
        loc_array = loc_data_sorted.to_records(index=False)
        group = np.zeros(len(loc_array), dtype=np.int32)

        link_groups = postprocess.get_link_groups(
            loc_array, max_distance, max_frames, group
        )

        return self._finish_clustering(loc_data_sorted, link_groups)

    # ------------------------------------------------------------------
    # Spectral-assisted LAP linking (Jaqaman et al. 2008)
    # ------------------------------------------------------------------

    def flag_static_localisations(self, loc_data: pd.DataFrame, eps: float, min_samples: int) -> NDArray[np.bool_]:
        """Identify static (non-diffusing) emitters via DBSCAN.

        Localisations belonging to a dense spatial cluster (≥ ``min_samples``
        observations within ``eps`` pixels) are flagged as static.  A
        diffusing molecule scatters across many pixels and appears as DBSCAN
        noise (label −1); a stuck molecule accumulates a tight cloud.

        The recommended eps is 3× per-axis localisation precision (px).

        Args:
            loc_data (pd.DataFrame): Localisation table with columns xc, yc.
            eps (float): DBSCAN neighbourhood radius in pixels.
            min_samples (int): Minimum cluster size to be called static.

        Returns:
            np.ndarray: Boolean array, True where the localisation is static.
        """
        from sklearn.cluster import DBSCAN

        xy = loc_data[["xc", "yc"]].to_numpy()
        labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(xy)
        return labels >= 0

    def spectral_lap_link(
        self,
        loc_data: pd.DataFrame,
        max_distance: float = 1.0,
        max_dark_time: int = 1,
        w_spatial: float = 1.0,
        w_spectral: float = 0.5,
        spectral_tol: float = FilteringConstants.MAX_COLOUR_ERROR,
        spectral_columns: tuple[str, ...] = ("A_R", "A_G", "A_B"),
        D_prior: float | None = None,
        dt: float = 1.0,
        sigma_loc: float = 0.0,
        alpha: float = 3.0,
        verbose: bool = False,
    ) -> NDArray[np.int32]:
        """Link localisations across frames using LAP with spectral cost.

        Builds a combined spatial + spectral cost matrix per frame and solves
        the Linear Assignment Problem (Hungarian algorithm) for globally
        optimal assignments.  Active tracks persist across dark frames up to
        ``max_dark_time``.

        Cost function:
            cost = w_spatial * (d_spatial / max_dist)²
                 + w_spectral * (d_spectral / spectral_tol)²

        When ``D_prior`` is supplied the spatial search radius scales as:
            max_dist(gap) = alpha * sqrt(4 * D_prior * gap * dt + 2 * sigma_loc²)

        References:
            Jaqaman et al. (2008) Nat. Methods 5, 695–702.
            Crocker & Grier (1996) J. Colloid Interface Sci. 179, 298–310.
            Chenouard et al. (2014) Nat. Methods 11, 281–289.
            Sergé et al. (2008) Nat. Methods 5, 687–694.

        Args:
            loc_data (pd.DataFrame): Localisations with columns frame, xc, yc
                and the columns listed in spectral_columns.
            max_distance (float): Hard spatial cutoff in pixels (used when
                D_prior is None or as fallback).
            max_dark_time (int): Maximum frame gap between consecutive
                localisations in a track.
            w_spatial (float): Weight for the spatial cost term.
            w_spectral (float): Weight for the spectral cost term.
            spectral_tol (float): Normalisation scale for spectral distance.
            spectral_columns (tuple): Column names for the spectral channels.
            D_prior (float or None): Diffusion coefficient in px²/frame (with
                dt in same units) for gap-scaled linking.  None → fixed cutoff.
            dt (float): Frame interval in seconds (used when D_prior is set).
            sigma_loc (float): Per-axis localisation precision in pixels
                (used when D_prior is set).
            alpha (float): σ-multiplier for gap-scaled cutoff (default 3).
            verbose (bool): Print progress every 500 frames.

        Returns:
            np.ndarray: Integer array of length len(loc_data); each element is
            a track ID (≥ 0) or −1 for unlinked localisations.
        """
        from scipy.optimize import linear_sum_assignment

        frames_arr = loc_data["frame"].to_numpy()
        xc = loc_data["xc"].to_numpy()
        yc = loc_data["yc"].to_numpy()

        spec_cols = [loc_data[c].to_numpy() for c in spectral_columns]
        spectra = np.column_stack(spec_cols)
        spec_sum = spectra.sum(axis=1, keepdims=True)
        spec_sum[spec_sum == 0] = 1.0
        spectra_norm = spectra / spec_sum

        n_locs = len(loc_data)
        link_group = np.full(n_locs, -1, dtype=np.int32)
        unique_frames = np.unique(frames_arr)

        cutoff = w_spatial + w_spectral
        active_tracks = []
        next_track_id = 0

        for fi, current_frame in enumerate(unique_frames):
            if verbose and fi % 500 == 0 and fi > 0:
                logger.info(f"  Frame {fi}/{len(unique_frames)}, " f"{next_track_id} tracks so far")

            frame_mask = frames_arr == current_frame
            cur_indices = np.where(frame_mask)[0]
            n_cur = len(cur_indices)

            if n_cur == 0:  # pragma: no cover -- unreachable: current_frame is
                continue    # drawn from np.unique(frames_arr), so the mask always matches

            cur_x = xc[cur_indices]
            cur_y = yc[cur_indices]
            cur_spec = spectra_norm[cur_indices]

            active_tracks = [
                t for t in active_tracks
                if t["last_frame"] >= current_frame - max_dark_time - 1
            ]
            n_active = len(active_tracks)

            if n_active == 0:
                for j, idx in enumerate(cur_indices):
                    link_group[idx] = next_track_id
                    active_tracks.append({
                        "id": next_track_id,
                        "last_frame": current_frame,
                        "last_x": cur_x[j],
                        "last_y": cur_y[j],
                        "last_spec": cur_spec[j],
                    })
                    next_track_id += 1
                continue

            trk_x = np.array([t["last_x"] for t in active_tracks])
            trk_y = np.array([t["last_y"] for t in active_tracks])
            trk_spec = np.array([t["last_spec"] for t in active_tracks])

            dx = trk_x[:, None] - cur_x[None, :]
            dy = trk_y[:, None] - cur_y[None, :]
            d_spatial = np.sqrt(dx**2 + dy**2)

            d_spectral = np.sqrt(
                ((trk_spec[:, None, :] - cur_spec[None, :, :]) ** 2).sum(axis=2)
            )

            if D_prior is not None:
                gap_n = np.array(
                    [current_frame - t["last_frame"] for t in active_tracks],
                    dtype=float,
                )
                max_dist_t = alpha * np.sqrt(
                    4.0 * D_prior * gap_n * dt + 2.0 * sigma_loc ** 2
                )
                max_dist_t = max_dist_t[:, None]
            else:
                max_dist_t = max_distance

            cost = (w_spatial * (d_spatial / max_dist_t) ** 2
                    + w_spectral * (d_spectral / spectral_tol) ** 2)
            cost[d_spatial > max_dist_t] = 1e9

            dim = n_active + n_cur
            aug = np.full((dim, dim), 1e9)
            aug[:n_active, :n_cur] = cost
            for i in range(n_active):
                aug[i, n_cur + i] = cutoff
            for j in range(n_cur):
                aug[n_active + j, j] = cutoff
            aug[n_active:, n_cur:] = 0.0

            row_ind, col_ind = linear_sum_assignment(aug)

            assigned_cur = set()
            for r, c in zip(row_ind, col_ind):
                if r < n_active and c < n_cur and aug[r, c] < cutoff:
                    trk = active_tracks[r]
                    idx = cur_indices[c]
                    link_group[idx] = trk["id"]
                    trk["last_frame"] = current_frame
                    trk["last_x"] = cur_x[c]
                    trk["last_y"] = cur_y[c]
                    trk["last_spec"] = cur_spec[c]
                    assigned_cur.add(c)

            for j in range(n_cur):
                if j not in assigned_cur:
                    idx = cur_indices[j]
                    link_group[idx] = next_track_id
                    active_tracks.append({
                        "id": next_track_id,
                        "last_frame": current_frame,
                        "last_x": cur_x[j],
                        "last_y": cur_y[j],
                        "last_spec": cur_spec[j],
                    })
                    next_track_id += 1

        if verbose:
            n_tracks = len(np.unique(link_group[link_group >= 0]))
            n_linked = np.sum(link_group >= 0)
            logger.info(f"  Linking complete: {n_tracks} tracks from " f"{n_linked}/{n_locs} localisations")

        return link_group

    def extract_single_molecules_spectral_lap(
        self,
        loc_data: pd.DataFrame,
        max_distance: float = 1.0,
        max_dark_time: int = 1,
        w_spatial: float = 1.0,
        w_spectral: float = 0.5,
        spectral_tol: float = FilteringConstants.MAX_COLOUR_ERROR,
        spectral_columns: tuple[str, ...] = ("A_R", "A_G", "A_B"),
        min_frames: int = 3,
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
        D_prior: float | None = None,
        dt: float = 1.0,
        sigma_loc: float = 0.0,
        alpha: float = 3.0,
        remove_static: bool = True,
        static_eps: float | None = None,
        static_min_samples: int = 10,
        verbose: bool = False,
        config: ClusteringConfig = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Extract single molecules using spectral-assisted LAP linking.

        Same interface and output format as extract_single_molecules_linked()
        but uses a Linear Assignment Problem (Hungarian algorithm) with
        combined spatial + spectral cost, following Jaqaman et al. (2008)
        Nat. Methods 5, 695–702.

        Args:
            loc_data (pd.DataFrame): Localisation data with columns frame,
                xc, yc, A_R, A_G, A_B, etc.
            max_distance (float): Maximum linking distance in pixels.
            max_dark_time (int): Maximum frame gap for linking.
            w_spatial (float): Weight for spatial cost term.
            w_spectral (float): Weight for spectral cost term.
            spectral_tol (float): Spectral distance tolerance.
            spectral_columns (tuple): Column names for spectral channels.
            min_frames (int): Minimum localisations per track (shorter
                tracks are discarded).
            criteria: Quality filter bundle (see filter_quality_localisations).
                chi_val, max_localisation_error, max_colour_error, min_sigma,
                max_sigma, max_sigma_error, min_photons, max_photons are applied
                when criteria is None.
            D_prior (float, optional): Gap-scaled linking parameter (see spectral_lap_link).
            dt (float): Frame interval for gap scaling.
            sigma_loc (float): Localisation precision for gap scaling.
            alpha (float): Gap penalty exponent.
            remove_static (bool): Remove static localisations before linking.
            static_eps (float, optional): DBSCAN eps for static detection;
                default 3*sigma_loc if sigma_loc>0 else 1.0.
            static_min_samples (int): DBSCAN min_samples for static detection.
            verbose (bool): Print progress messages.

        Returns:
            tuple: (single_molecule_database, single_frame_database).
        """
        if config is not None:
            max_distance        = config.max_distance
            max_dark_time       = config.max_dark_time
            w_spatial           = config.w_spatial
            w_spectral          = config.w_spectral
            spectral_tol        = config.spectral_tol
            spectral_columns    = config.spectral_columns
            min_frames          = config.min_frames
            D_prior             = config.D_prior
            dt                  = config.dt
            sigma_loc           = config.sigma_loc
            alpha               = config.alpha
            remove_static       = config.remove_static
            static_eps          = config.static_eps
            static_min_samples  = config.static_min_samples
            verbose             = config.verbose

        loc_data = self.filter_quality_localisations(
            loc_data=loc_data,
            criteria=criteria,
            chi_val=chi_val,
            max_localisation_error=max_localisation_error,
            min_photons=min_photons,
            max_photons=max_photons,
            min_photoelectrons=min_photoelectrons,
            max_photoelectrons=max_photoelectrons,
            max_colour_error=max_colour_error,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            max_sigma_error=max_sigma_error,
        )

        loc_data_sorted = loc_data.sort_values("frame").reset_index(drop=True).copy()

        if remove_static:
            eps = static_eps if static_eps is not None else (
                3.0 * sigma_loc if sigma_loc > 0 else 1.0
            )
            static_mask = self.flag_static_localisations(
                loc_data_sorted, eps=eps, min_samples=static_min_samples,
            )
            n_static = int(static_mask.sum())
            if verbose:
                logger.info(f"  Static removal: {n_static}/{len(loc_data_sorted)} " f"localisations flagged ({100*n_static/max(len(loc_data_sorted),1):.1f}%)")
            loc_data_sorted = loc_data_sorted[~static_mask].reset_index(drop=True)

        if verbose:
            logger.info(f"Spectral LAP linking: {len(loc_data_sorted)} localisations, " f"max_distance={max_distance}, max_dark_time={max_dark_time}")

        link_groups = self.spectral_lap_link(
            loc_data_sorted,
            max_distance=max_distance,
            max_dark_time=max_dark_time,
            w_spatial=w_spatial,
            w_spectral=w_spectral,
            spectral_tol=spectral_tol,
            spectral_columns=spectral_columns,
            D_prior=D_prior,
            dt=dt,
            sigma_loc=sigma_loc,
            alpha=alpha,
            verbose=verbose,
        )

        linked_mask = link_groups >= 0
        loc_data_linked = loc_data_sorted[linked_mask].copy()
        link_groups_linked = link_groups[linked_mask]

        unique_ids = np.unique(link_groups_linked)
        id_map = {old: new for new, old in enumerate(unique_ids)}
        link_groups_linked = np.array(
            [id_map[g] for g in link_groups_linked], dtype=np.int32
        )

        loc_data_linked["molecular_index"] = link_groups_linked

        single_molecule_database = self.average_parameters(
            loc_data_linked, link_groups_linked
        )
        single_molecule_database["molecular_index"] = np.arange(
            len(single_molecule_database)
        )

        if min_frames > 1:
            keep_mask = single_molecule_database["frames"] >= min_frames
            removed_ids = set(
                single_molecule_database.loc[~keep_mask, "molecular_index"]
            )
            kept_old_ids = single_molecule_database.loc[keep_mask, "molecular_index"].values
            old_to_new = {int(old): new for new, old in enumerate(kept_old_ids)}

            single_molecule_database = single_molecule_database[keep_mask].reset_index(drop=True)
            single_molecule_database["molecular_index"] = np.arange(
                len(single_molecule_database)
            )
            loc_data_linked = loc_data_linked[
                ~loc_data_linked["molecular_index"].isin(removed_ids)
            ].copy()
            loc_data_linked["molecular_index"] = loc_data_linked["molecular_index"].map(old_to_new)

        if verbose:
            logger.info(f"Result: {len(single_molecule_database)} molecules from " f"{len(loc_data_linked)} linked localisations " f"(min_frames={min_frames})")

        return single_molecule_database, loc_data_linked
