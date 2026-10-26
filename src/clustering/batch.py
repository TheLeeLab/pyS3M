# -*- coding: utf-8 -*-
"""
clustering/batch.py

Batch extraction and multi-FOV orchestration mixin.
Extracted from SM_extractionfunctions.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from pyS3M.Constants import FilteringConstants, FilteringCriteria
from ._config import ClusteringConfig
import logging
logger = logging.getLogger(__name__)



class BatchMixin:
    """Mixin supplying batch / multi-FOV single-molecule extraction."""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_fov_name(self, filepath: Path | str) -> str:
        """Extract FOV identifier from filename, stripping all known suffixes.

        Returns the full filename (without directory or extension) to ensure
        uniqueness across datasets where a short suffix like 'Pos0' may repeat.

        Args:
            filepath (str): Full path to localisation file.

        Returns:
            str: Filename without extension (e.g. "Pos15_undrifted_locs").
        """
        _known = {'.h5', '.hdf5', '.tif', '.tiff', '.ome', '.txt', '.csv', '.json'}
        p = Path(filepath)
        while p.suffix.lower() in _known:
            p = p.with_suffix('')
        return p.name

    # ------------------------------------------------------------------
    # Single-batch extraction
    # ------------------------------------------------------------------

    def extract_single_molecules_batch(
        self,
        localisation_files: list[str | Path],
        clustering_method: str = "HDBSCAN",
        min_cluster_size: int = 10,
        criteria: FilteringCriteria = None,
        chi_val: float | None = None,
        max_localisation_error: float = FilteringConstants.MAX_LOCALISATION_ERROR_PX,
        max_colour_error: float = FilteringConstants.MAX_COLOUR_ERROR,
        min_sigma: float | None = None,
        max_sigma: float | None = None,
        max_sigma_error: float | None = None,
        min_photons: float = FilteringConstants.MIN_PHOTONS,
        max_photons: float = 1e6,
        min_photoelectrons: float | None = None,
        max_photoelectrons: float | None = None,
        max_distance: float = 0.5,
        max_frames: int = 10,
        epsilon_multiplier: float = 1.0,
        start_frame: int = 0,
        verbose: bool = True,
        config: ClusteringConfig = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract single molecules from multiple localisation files (FOVs).

        Processes multiple FOV files and combines them into unified databases
        with FOV tracking columns.  Each molecule gets a globally unique
        molecular_index.

        Args:
            localisation_files (list): List of HDF5 localisation file paths.
            clustering_method (str): "HDBSCAN", "DBSCAN", or "linked".
            min_cluster_size (int): Minimum cluster size for HDBSCAN/DBSCAN.
            criteria (FilteringCriteria, optional): Quality filter bundle.
            chi_val (float, optional): Quality filter parameter.
            max_localisation_error (float): Quality filter parameter.
            max_colour_error (float): Quality filter parameter.
            min_sigma (float, optional): Quality filter parameter.
            max_sigma (float, optional): Quality filter parameter.
            max_sigma_error (float, optional): Quality filter parameter.
            min_photons (float): Quality filter parameter.
            max_photons (float): Quality filter parameter.
            min_photoelectrons, max_photoelectrons (float, optional): Bounds on the detected
                photoelectrons (see filter_quality_localisations); None → no cut.
            max_distance (float): Maximum distance for linked method (pixels).
            max_frames (int): Maximum frame gap for linked method.
            epsilon_multiplier (float): DBSCAN epsilon multiplier.
            start_frame (int): Discard localisations before this frame.
            verbose (bool): Print progress information.

        Returns:
            tuple: (single_molecule_database, single_frame_database).
                Both DataFrames include fov_index and fov_name columns.
                molecular_index is globally unique across all FOVs.
        """
        if config is not None:
            clustering_method  = config.clustering_method
            min_cluster_size   = config.min_cluster_size
            epsilon_multiplier = config.epsilon_multiplier
            max_distance       = config.max_distance
            max_frames         = config.max_frames
            start_frame        = config.start_frame
            verbose            = config.verbose

        if not isinstance(localisation_files, (list, np.ndarray)):
            raise ValueError("localisation_files must be a list or array of file paths")

        molecular_index_offset = 0
        all_molecule_dbs = []
        all_frame_dbs = []

        for fov_idx, loc_file in enumerate(localisation_files):
            if verbose:
                logger.debug(f"Processing FOV {fov_idx + 1}/{len(localisation_files)} ({Path(loc_file).name})...")

            loc_data = self.io.read_h5_database(loc_file)
            if start_frame > 0:
                loc_data = loc_data[loc_data["frame"] >= start_frame].reset_index(drop=True)

            if verbose:
                logger.debug(f" {len(loc_data)} localizations...")

            fov_name = self._extract_fov_name(loc_file)
            if fov_name is None:
                fov_name = f"fov_{fov_idx}"
                if verbose:
                    logger.debug(f" (no Pos pattern, using {fov_name})...")

            if clustering_method.upper() == "HDBSCAN":
                sm_db, sf_db = self.extract_single_molecules_HDBSCAN(
                    loc_data,
                    min_cluster_size=min_cluster_size,
                    criteria=criteria,
                    chi_val=chi_val,
                    max_localisation_error=max_localisation_error,
                    max_colour_error=max_colour_error,
                    min_sigma=min_sigma,
                    max_sigma=max_sigma,
                    max_sigma_error=max_sigma_error,
                    min_photons=min_photons,
                    max_photons=max_photons,
                    min_photoelectrons=min_photoelectrons,
                    max_photoelectrons=max_photoelectrons,
                )
            elif clustering_method.upper() == "DBSCAN":
                sm_db, sf_db = self.extract_single_molecules_DBSCAN(
                    loc_data,
                    min_cluster_size=min_cluster_size,
                    criteria=criteria,
                    chi_val=chi_val,
                    max_localisation_error=max_localisation_error,
                    max_colour_error=max_colour_error,
                    min_sigma=min_sigma,
                    max_sigma=max_sigma,
                    max_sigma_error=max_sigma_error,
                    min_photons=min_photons,
                    max_photons=max_photons,
                    min_photoelectrons=min_photoelectrons,
                    max_photoelectrons=max_photoelectrons,
                    epsilon_multiplier=epsilon_multiplier,
                )
            elif clustering_method.lower() == "linked":
                sm_db, sf_db = self.extract_single_molecules_linked(
                    loc_data,
                    max_distance=max_distance,
                    max_frames=max_frames,
                    criteria=criteria,
                    chi_val=chi_val,
                    max_localisation_error=max_localisation_error,
                    max_colour_error=max_colour_error,
                    min_sigma=min_sigma,
                    max_sigma=max_sigma,
                    max_sigma_error=max_sigma_error,
                    min_photons=min_photons,
                    max_photons=max_photons,
                    min_photoelectrons=min_photoelectrons,
                    max_photoelectrons=max_photoelectrons,
                )
            else:
                raise ValueError(
                    f"Unknown clustering_method: {clustering_method}. "
                    f"Choose from 'HDBSCAN', 'DBSCAN', or 'linked'"
                )

            if verbose:
                logger.info(f" Found {len(sm_db)} molecules")

            if len(sm_db) == 0:
                continue

            sm_db["fov_index"] = fov_idx
            sm_db["fov_name"] = fov_name
            sf_db["fov_index"] = fov_idx
            sf_db["fov_name"] = fov_name

            sm_db["molecular_index"] = sm_db["molecular_index"] + molecular_index_offset
            sf_db["molecular_index"] = sf_db["molecular_index"] + molecular_index_offset

            all_molecule_dbs.append(sm_db)
            all_frame_dbs.append(sf_db)

            molecular_index_offset += len(sm_db)

        if verbose:
            logger.info("\nCombining databases...")

        if len(all_molecule_dbs) == 0:
            if verbose:
                logger.warning("Warning: No molecules found in any FOV. Returning empty databases.")
            return pd.DataFrame(), pd.DataFrame()

        single_molecule_database = pd.concat(all_molecule_dbs, ignore_index=True)
        single_frame_database = pd.concat(all_frame_dbs, ignore_index=True)

        if verbose:
            logger.info(f"Complete! Summary:")
            logger.info(f"  - Total FOVs: {len(localisation_files)}")
            logger.info(f"  - Total molecules: {len(single_molecule_database)}")
            logger.info(f"  - Total localizations: {len(single_frame_database)}")

        return single_molecule_database, single_frame_database

    # ------------------------------------------------------------------
    # Photon accumulation
    # ------------------------------------------------------------------

    def build_photon_accumulation_database(self, single_frame_database: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Build cumulative photon accumulation database for precision analysis.

        For each molecule, creates a cumulative sum trace showing how A_R, A_G,
        A_B evolve as more frames are accumulated, enabling analysis of
        parameter precision vs. photon count.

        Args:
            single_frame_database (pd.DataFrame): Output from
                extract_single_molecules_*.  Must contain columns
                molecular_index, frame, A_R, A_G, A_B, photons, A_R_err,
                A_G_err, A_B_err, xc, yc.
            verbose (bool): Print progress information.

        Returns:
            pd.DataFrame: Photon accumulation database with one row per
            (molecule, accumulated-frame-count).  Key columns:
            molecular_index, frames_accumulated, photons_accumulated,
            A_R, A_G, A_B, A_R_err, A_G_err, A_B_err, xc_mean, yc_mean,
            xc_std, yc_std, s_x_mean, s_y_mean (plus fov_index/fov_name
            when present in input).
        """
        required_cols = ["molecular_index", "frame", "A_R", "A_G", "A_B", "photons"]
        missing_cols = [col for col in required_cols if col not in single_frame_database.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        unique_molecules = np.sort(single_frame_database["molecular_index"].unique())

        if verbose:
            logger.info(f"Building photon accumulation database for {len(unique_molecules)} molecules...")

        accumulation_records = []

        for mol_idx in unique_molecules:
            mol_data = single_frame_database[
                single_frame_database["molecular_index"] == mol_idx
            ].copy()
            mol_data = mol_data.sort_values("frame")

            fov_index = mol_data["fov_index"].iloc[0] if "fov_index" in mol_data.columns else None
            fov_name = mol_data["fov_name"].iloc[0] if "fov_name" in mol_data.columns else None

            n_frames = len(mol_data)

            if verbose and (mol_idx % 100 == 0 or mol_idx == unique_molecules[0]):
                logger.debug(f"  Molecule {mol_idx + 1}/{len(unique_molecules)} ({n_frames} frames)")

            cumsum_photons = np.cumsum(mol_data["photons"].values)
            frames_range = np.arange(1, n_frames + 1)

            if "A_R_err" in mol_data.columns:
                A_R_vals = mol_data["A_R"].values
                A_G_vals = mol_data["A_G"].values
                A_B_vals = mol_data["A_B"].values
                A_R_errs = mol_data["A_R_err"].values
                A_G_errs = mol_data["A_G_err"].values
                A_B_errs = mol_data["A_B_err"].values

                eps = 1e-12
                A_R_weights = 1.0 / (A_R_errs**2 + eps)
                A_G_weights = 1.0 / (A_G_errs**2 + eps)
                A_B_weights = 1.0 / (A_B_errs**2 + eps)

                cumsum_A_R_weighted = np.cumsum(A_R_weights * A_R_vals)
                cumsum_A_G_weighted = np.cumsum(A_G_weights * A_G_vals)
                cumsum_A_B_weighted = np.cumsum(A_B_weights * A_B_vals)
                cumsum_A_R_weights = np.cumsum(A_R_weights)
                cumsum_A_G_weights = np.cumsum(A_G_weights)
                cumsum_A_B_weights = np.cumsum(A_B_weights)

                A_R_mean = cumsum_A_R_weighted / cumsum_A_R_weights
                A_G_mean = cumsum_A_G_weighted / cumsum_A_G_weights
                A_B_mean = cumsum_A_B_weighted / cumsum_A_B_weights
                A_R_mean_err = 1.0 / np.sqrt(cumsum_A_R_weights)
                A_G_mean_err = 1.0 / np.sqrt(cumsum_A_G_weights)
                A_B_mean_err = 1.0 / np.sqrt(cumsum_A_B_weights)
            else:
                cumsum_A_R = np.cumsum(mol_data["A_R"].values)
                cumsum_A_G = np.cumsum(mol_data["A_G"].values)
                cumsum_A_B = np.cumsum(mol_data["A_B"].values)
                A_R_mean = cumsum_A_R / frames_range
                A_G_mean = cumsum_A_G / frames_range
                A_B_mean = cumsum_A_B / frames_range
                A_R_mean_err = np.zeros(n_frames)
                A_G_mean_err = np.zeros(n_frames)
                A_B_mean_err = np.zeros(n_frames)

            xc_cumsum = np.cumsum(mol_data["xc"].values)
            yc_cumsum = np.cumsum(mol_data["yc"].values)
            xc_mean = xc_cumsum / frames_range
            yc_mean = yc_cumsum / frames_range

            xc_std = np.zeros(n_frames)
            yc_std = np.zeros(n_frames)
            for i in range(n_frames):
                if i > 0:
                    xc_std[i] = np.std(mol_data["xc"].values[: i + 1])
                    yc_std[i] = np.std(mol_data["yc"].values[: i + 1])

            if "s_x" in mol_data.columns:
                s_x_mean = np.cumsum(mol_data["s_x"].values) / frames_range
                s_y_mean = np.cumsum(mol_data["s_y"].values) / frames_range
            else:
                s_x_mean = np.zeros(n_frames)
                s_y_mean = np.zeros(n_frames)

            for i in range(n_frames):
                record = {
                    "molecular_index": mol_idx,
                    "frames_accumulated": i + 1,
                    "photons_accumulated": cumsum_photons[i],
                    "A_R": A_R_mean[i],
                    "A_G": A_G_mean[i],
                    "A_B": A_B_mean[i],
                    "A_R_err": A_R_mean_err[i],
                    "A_G_err": A_G_mean_err[i],
                    "A_B_err": A_B_mean_err[i],
                    "xc_mean": xc_mean[i],
                    "yc_mean": yc_mean[i],
                    "xc_std": xc_std[i],
                    "yc_std": yc_std[i],
                    "s_x_mean": s_x_mean[i],
                    "s_y_mean": s_y_mean[i],
                }
                if fov_index is not None:
                    record["fov_index"] = fov_index
                if fov_name is not None:
                    record["fov_name"] = fov_name
                accumulation_records.append(record)

        if verbose:
            logger.info("\n  Converting to DataFrame...")

        photon_accumulation_db = pd.DataFrame(accumulation_records)

        if verbose:
            total_rows = len(photon_accumulation_db)
            avg_frames_per_mol = total_rows / len(unique_molecules)
            logger.info(f"Complete! Photon accumulation database:")
            logger.info(f"  - Total rows: {total_rows}")
            logger.info(f"  - Average frames per molecule: {avg_frames_per_mol:.1f}")

        return photon_accumulation_db

    # ------------------------------------------------------------------
    # High-level multi-FOV workflow
    # ------------------------------------------------------------------

    def analyse_multi_fov_dataset(
        self,
        localisation_files: list[str | Path],
        clustering_method: str = "HDBSCAN",
        build_accumulation: bool = True,
        min_cluster_size: int = 10,
        chi_val: float | None = None,
        max_localisation_error: float = 1.0,
        max_colour_error: float = FilteringConstants.MAX_COLOUR_ERROR,
        min_sigma: float | None = None,
        max_sigma: float | None = None,
        max_sigma_error: float | None = None,
        min_photons: float = FilteringConstants.MIN_PHOTONS,
        max_photons: float = 1e6,
        min_photoelectrons: float | None = None,
        max_photoelectrons: float | None = None,
        max_distance: float = 0.5,
        max_frames: int = 10,
        epsilon_multiplier: float = 0.5,
        output_folder: Path | str | None = None,
        output_prefix: str = "analysis",
        verbose: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame]:
        """
        Complete workflow: extract single molecules from multiple FOVs and
        optionally build photon accumulation database.

        Args:
            localisation_files (list): List of HDF5 localisation file paths.
            clustering_method (str): "HDBSCAN", "DBSCAN", or "linked".
            build_accumulation (bool): Build photon accumulation database?
            min_cluster_size (int): Minimum cluster size for HDBSCAN/DBSCAN.
            chi_val (float, optional): Quality filter parameter.
            max_localisation_error (float): Quality filter parameter.
            max_colour_error (float): Quality filter parameter.
            min_sigma (float, optional): Quality filter parameter.
            max_sigma (float, optional): Quality filter parameter.
            max_sigma_error (float, optional): Quality filter parameter.
            min_photons (float): Quality filter parameter.
            max_photons (float): Quality filter parameter.
            min_photoelectrons, max_photoelectrons (float, optional): Bounds on the detected
                photoelectrons (see filter_quality_localisations); None → no cut.
            max_distance (float): Maximum distance for linked method (pixels).
            max_frames (int): Maximum frame gap for linked method.
            epsilon_multiplier (float): DBSCAN epsilon multiplier.
            output_folder (str, optional): Save databases here if provided.
            output_prefix (str): Prefix for output filenames.
            verbose (bool): Print progress information.

        Returns:
            tuple: (single_molecule_db, single_frame_db, photon_accumulation_db)
                   if build_accumulation=True, else (single_molecule_db, single_frame_db).
        """
        if verbose:
            logger.info("=" * 60)
            logger.info("Multi-FOV Single Molecule Analysis")
            logger.info("=" * 60)

        single_molecule_db, single_frame_db = self.extract_single_molecules_batch(
            localisation_files=localisation_files,
            clustering_method=clustering_method,
            min_cluster_size=min_cluster_size,
            chi_val=chi_val,
            max_localisation_error=max_localisation_error,
            max_colour_error=max_colour_error,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            max_sigma_error=max_sigma_error,
            min_photons=min_photons,
            max_photons=max_photons,
            min_photoelectrons=min_photoelectrons,
            max_photoelectrons=max_photoelectrons,
            max_distance=max_distance,
            max_frames=max_frames,
            epsilon_multiplier=epsilon_multiplier,
            verbose=verbose,
        )

        photon_accumulation_db = None
        if build_accumulation:
            photon_accumulation_db = self.build_photon_accumulation_database(
                single_frame_db, verbose=verbose
            )

        if output_folder is not None:
            if verbose:
                logger.info(f"\nSaving databases to {output_folder}...")
            Path(output_folder).mkdir(parents=True, exist_ok=True)

            sm_path = Path(output_folder) / f"{output_prefix}_single_molecules.h5"
            self.io.write_h5_database(single_molecule_db, sm_path, normalise_photons=False)
            if verbose:
                logger.info(f"  Saved: {Path(sm_path).name}")

            sf_path = Path(output_folder) / f"{output_prefix}_single_frames.h5"
            self.io.write_h5_database(
                single_frame_db, sf_path, normalise_photons=False, append=False, verbose=verbose
            )
            if verbose:
                logger.info(f"  Saved: {Path(sf_path).name}")

            if photon_accumulation_db is not None:
                pa_path = Path(output_folder) / f"{output_prefix}_photon_accumulation.h5"
                self.io.write_h5_database(photon_accumulation_db, pa_path, normalise_photons=False)
                if verbose:
                    logger.info(f"  Saved: {Path(pa_path).name}")

        if verbose:
            logger.info("\n" + "=" * 60)
            logger.info("Analysis Complete!")
            logger.info("=" * 60)

        if build_accumulation:
            return single_molecule_db, single_frame_db, photon_accumulation_db
        else:
            return single_molecule_db, single_frame_db
