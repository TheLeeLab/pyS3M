"""Shared helper for downloading files from the paper's BioStudies deposition
(accession S-BIAD3210) over plain HTTPS -- no auth needed, confirmed live
against the FTP mirror (see claude/paper_figures_reproduction.md §3.1).

Reused by every notebooks/figures/ notebook that needs real acquired data
("bucket B" in that doc) rather than reinventing a fetch helper per notebook.
Deliberately plain `requests`, no new heavyweight dependency (e.g. `pooch`) --
these are plain scoped file GETs, nothing a fuller download-manager buys much
over.
"""
from pathlib import Path

import requests

ACCESSION = "S-BIAD3210"
BASE_URL = f"https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/210/{ACCESSION}/Files"


def human_size(n_bytes: float) -> str:
    if n_bytes < 0:
        return "unknown size"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} PB"


def download_file(remote_rel_path: str, dest_dir: Path, chunk_size: int = 1 << 20,
                   force: bool = False) -> Path:
    """Download one file from the deposition to *dest_dir*, keeping only its
    basename (the deposition's own folder structure isn't reproduced locally).

    Skips the download if a file of the exact same size already exists at the
    destination (resume/re-run friendly, same size-based-skip convention as
    ``notebooks/EBI_Upload/*.ipynb``'s upload helpers). Downloads to a
    ``.part`` temp file first so an interrupted download can never look like
    a complete one.
    """
    url = f"{BASE_URL}/{remote_rel_path}"
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / Path(remote_rel_path).name

    with requests.head(url, timeout=30, allow_redirects=True) as r:
        r.raise_for_status()
        remote_size = int(r.headers.get("Content-Length", -1))

    if not force and dest.exists() and remote_size >= 0 and dest.stat().st_size == remote_size:
        print(f"Already downloaded: {dest.name} ({human_size(remote_size)})")
        return dest

    print(f"Downloading {remote_rel_path.rsplit('/', 1)[-1]} ({human_size(remote_size)}) ...")
    tmp = dest.with_name(dest.name + ".part")
    try:
        from tqdm.notebook import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        iterator = r.iter_content(chunk_size=chunk_size)
        with tmp.open("wb") as f:
            if has_tqdm and remote_size > 0:
                with tqdm(total=remote_size, unit="B", unit_scale=True) as pbar:
                    for chunk in iterator:
                        f.write(chunk)
                        pbar.update(len(chunk))
            else:
                for chunk in iterator:
                    f.write(chunk)

    tmp.rename(dest)
    print(f"  -> {dest}")
    return dest


def download_files(remote_rel_paths: list, dest_dir: Path, **kwargs) -> list:
    """Download each path in *remote_rel_paths* (see ``download_file``)."""
    return [download_file(p, dest_dir, **kwargs) for p in remote_rel_paths]
