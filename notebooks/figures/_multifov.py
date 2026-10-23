"""Shared pipeline for multi-position single-molecule acquisitions (Figures 2 and 3).

The single-molecule acquisitions on BioStudies (single dyes, QDots) are grids of fields of view:
one TIFF plus one ``*_metadata.txt`` per position. Each acquisition goes through

1. download (``FAST_MODE``: one position),
2. fit each position on its own (``fit_SM_data``, one ``.h5`` per TIFF),
3. optionally, AIM drift correction per position,
4. clustering of localisations into molecules across all positions
   (``analyse_multi_fov_dataset``; ``fov_index`` follows the sorted file order).

Fits, drift-corrected files and clusterings are cached beside the data and reused while the
files and settings are unchanged.
"""
import json
from pathlib import Path

import pandas as pd

import _biostudies_download as bd

PIXEL_SIZE_NM = 69.0


def metadata_for(tif: Path) -> Path:
    """``x_MMStack_2-Pos000_000.ome.tif`` -> ``x_MMStack_2-Pos000_000_metadata.txt``."""
    return tif.with_name(tif.name.split(".")[0] + "_metadata.txt")


def tif_for(h5: Path) -> Path:
    """Per-position fit (or ``*_undrifted_locs.h5``) -> the TIFF it came from."""
    return h5.with_name(h5.name.replace("_undrifted_locs.h5", ".h5").replace(".h5", ".ome.tif"))


def prepare_acquisition(remote_folder, dest_dir, fast_mode, fitter, calib, smoothing_function, io,
                        fit_params, drift_corrector=None, drift_params=None, position=None,
                        refit=False):
    """Download and fit one acquisition (optionally drift-correct it); return per-position files.

    Args:
        remote_folder: Deposition folder of the acquisition.
        dest_dir: Local folder for this acquisition only (the fit reads every TIFF in it).
        fast_mode: Download one position only.
        fitter: ``SR_Functions.SuperRes_Functions`` instance.
        calib: Dict of full-sensor calibration maps (gain_map, offset_map, rqe, read_noise,
            variance).
        smoothing_function: Spot-detection smoothing namespace.
        io: ``IOFunctions.IO_Functions`` instance.
        fit_params: ``fit_SM_data`` keyword arguments (pfa, sigma, ROI_size, ...).
        drift_corrector: ``Drift_Correction_Functions`` instance, or None to skip drift correction.
        drift_params: AIM settings (segmentation, intersect_d, roi_r), camera px.
        position: In ``fast_mode``, the position to take (e.g. ``"Pos001_001"``); default the first.
        refit: Fit again even if a fit with the same files and settings exists.

    Returns:
        list[Path]: One localisation file per position (``*_undrifted_locs.h5`` if drift
        corrected, else the fit's ``.h5``), in sorted position order.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    stem = None
    if fast_mode and position is not None:
        names = bd.list_remote_folder(remote_folder)
        stem = next(n for n in names if f"-{position}." in n).split(".")[0]
    downloaded = bd.download_acquisition(remote_folder, dest_dir, stem=stem, first_file_only=fast_mode)
    tifs = sorted(downloaded["images"], key=lambda p: bd._natural_key(p.name))

    record_file = dest_dir / "fit_record.json"
    record = {"images": [p.name for p in tifs], **fit_params}
    fits = [p.with_name(p.name.split(".")[0] + ".h5") for p in tifs]
    previous = json.loads(record_file.read_text()) if record_file.exists() else None
    if refit or previous != record or not all(f.exists() for f in fits):
        for f in fits:   # fit_SM_data appends to existing per-file outputs
            f.unlink(missing_ok=True)
        fitter.fit_SM_data(dest_dir, smoothing_function, image_type=".tif", **calib, **fit_params)
        record_file.write_text(json.dumps(record, indent=1))
        for f in dest_dir.glob("*_undrifted_locs.h5"):
            f.unlink()
    else:
        print(f"Reusing the fits in {dest_dir} (same files and settings)")

    if drift_corrector is None:
        return fits

    undrifted = []
    for tif, fit in zip(tifs, fits):
        out = fit.with_name(fit.stem + "_undrifted_locs.h5")
        undrifted.append(out)
        if out.exists():
            continue
        meta = metadata_for(tif)
        _, _, width, height = io.metadata_reader_imageJ(meta)
        info = [{"Width": float(width), "Height": float(height),
                 "Frames": float(io.metadata_nframes_reader_imageJ(meta)),
                 "Pixelsize": PIXEL_SIZE_NM}]
        locs = pd.read_hdf(fit)
        corrected, _ = drift_corrector.undrift(locs.to_records(index=False), info, method="aim",
                                               **drift_params)
        io.write_h5_database(pd.DataFrame(corrected), out, normalise_photons=False, verbose=False)
    return undrifted


def cluster_sample(sm_extractor, files, out_dir, prefix, params, recluster=False):
    """Cluster a sample's localisations into molecules (cached on disk).

    Args:
        sm_extractor: ``SM_extractionfunctions.extract_SMs`` instance.
        files: Per-position localisation files (``fov_index`` = position in this list).
        out_dir: Where the databases are written.
        prefix: Database file-name prefix.
        params: ``analyse_multi_fov_dataset`` settings (method, cluster size, quality cuts).
        recluster: Cluster again even if a clustering with the same files and settings exists.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: single-molecule and single-frame databases.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sm_file = out_dir / f"{prefix}_single_molecules.h5"
    sf_file = out_dir / f"{prefix}_single_frames.h5"
    record_file = out_dir / f"{prefix}_cluster_record.json"
    record = {"files": [str(Path(f).resolve()) for f in files],
              "mtimes": [Path(f).stat().st_mtime for f in files], **params}
    previous = json.loads(record_file.read_text()) if record_file.exists() else None
    if not recluster and previous == record and sm_file.exists() and sf_file.exists():
        return pd.read_hdf(sm_file), pd.read_hdf(sf_file)
    sm_db, sf_db = sm_extractor.analyse_multi_fov_dataset(
        [str(f) for f in files], build_accumulation=False, output_folder=str(out_dir),
        output_prefix=prefix, verbose=False, **params)
    record_file.write_text(json.dumps(record, indent=1))
    return sm_db, sf_db
