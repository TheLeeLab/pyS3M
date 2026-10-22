"""Shared helper for downloading files from the paper's BioStudies deposition
(accession S-BIAD3210) over plain HTTPS -- no auth needed, confirmed live
against the FTP mirror (see claude/paper_figures_reproduction.md §3.1).

Reused by every notebooks/figures/ notebook that needs real acquired data
("bucket B" in that doc) rather than reinventing a fetch helper per notebook.
Deliberately plain `requests`, no new heavyweight dependency (e.g. `pooch`) --
these are plain scoped file GETs, nothing a fuller download-manager buys much
over.
"""
import time
from pathlib import Path

import requests

ACCESSION = "S-BIAD3210"
BASE_URL = f"https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/210/{ACCESSION}/Files"

# (connect, read) seconds. ftp.ebi.ac.uk's TLS handshake routinely takes 20-30 s, sometimes
# over 60 s (measured 2026-10-22). urllib3 counts the handshake against the *connect* timeout
# (though it reports it as a read timeout), so that one is generous too, and failed requests
# are retried with backoff.
TIMEOUT = (180, 300)
MAX_ATTEMPTS = 6
# Ask for the raw bytes: with transfer compression, Content-Length (compressed) would never
# match the decoded size written to disk, breaking the size checks and Range resume.
HEADERS = {"Accept-Encoding": "identity"}


def human_size(n_bytes: float) -> str:
    if n_bytes < 0:
        return "unknown size"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} PB"


def _retry(action, what: str, max_attempts: int = None, progress=None):
    """Call *action()*, retrying on connection errors/timeouts with exponential backoff.

    If *progress()* (e.g. bytes written so far) grew since the last failure, the attempt
    count resets, so a long download that keeps moving is never abandoned.
    """
    max_attempts = max_attempts or MAX_ATTEMPTS
    attempt, last = 0, progress() if progress else None
    while True:
        attempt += 1
        try:
            return action()
        except (requests.ConnectionError, requests.Timeout) as e:
            if progress is not None and progress() > last:
                attempt, last = 1, progress()
            if attempt >= max_attempts:
                raise
            wait = min(10 * 2 ** (attempt - 1), 300)
            print(f"  {what}: {type(e).__name__} (attempt {attempt}/{max_attempts}), retrying in {wait} s ...")
            time.sleep(wait)


def download_file(remote_rel_path: str, dest_dir: Path, chunk_size: int = 1 << 20,
                   force: bool = False) -> Path:
    """Download one file from the deposition to *dest_dir*, keeping only its
    basename (the deposition's own folder structure isn't reproduced locally).

    Skips the download if a file of the exact same size already exists at the
    destination (resume/re-run friendly, same size-based-skip convention as
    ``notebooks/EBI_Upload/*.ipynb``'s upload helpers). Downloads to a
    ``.part`` temp file first so an interrupted download can never look like
    a complete one. An existing ``.part`` file (from a dropped connection or a
    previous run) is resumed with an HTTP Range request rather than restarted.
    """
    url = f"{BASE_URL}/{remote_rel_path}"
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / Path(remote_rel_path).name

    def head():
        with requests.head(url, timeout=TIMEOUT, allow_redirects=True, headers=HEADERS) as r:
            r.raise_for_status()
            return int(r.headers.get("Content-Length", -1))

    remote_size = _retry(head, f"HEAD {dest.name}")

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

    if force and tmp.exists():
        tmp.unlink()

    def fetch():
        """Fetch the rest of the file into *tmp*, resuming from its current size."""
        start = tmp.stat().st_size if tmp.exists() else 0
        if remote_size >= 0 and start > remote_size:  # stale .part from a different file
            tmp.unlink()
            start = 0
        if remote_size >= 0 and start == remote_size:
            return
        headers = {**HEADERS, "Range": f"bytes={start}-"} if start else HEADERS
        with requests.get(url, stream=True, timeout=TIMEOUT, headers=headers) as r:
            if start and r.status_code == 416:  # server rejected the resume range: start over
                tmp.unlink()
                raise requests.ConnectionError("resume range rejected (HTTP 416), restarting")
            r.raise_for_status()
            if start and r.status_code != 206:  # server ignored the Range: start over
                start = 0
            if start:
                print(f"  resuming from {human_size(start)}")
            with tmp.open("ab" if start else "wb") as f:
                pbar = (tqdm(total=remote_size, initial=start, unit="B", unit_scale=True)
                        if has_tqdm and remote_size > 0 else None)
                try:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                        if pbar is not None:
                            pbar.update(len(chunk))
                finally:
                    if pbar is not None:
                        pbar.close()
        if remote_size >= 0 and tmp.stat().st_size != remote_size:
            raise requests.ConnectionError(
                f"incomplete: got {human_size(tmp.stat().st_size)} of {human_size(remote_size)}")

    _retry(fetch, f"GET {dest.name}", progress=lambda: tmp.stat().st_size if tmp.exists() else 0)

    tmp.rename(dest)
    print(f"  -> {dest}")
    return dest


def download_files(remote_rel_paths: list, dest_dir: Path, **kwargs) -> list:
    """Download each path in *remote_rel_paths* (see ``download_file``)."""
    return [download_file(p, dest_dir, **kwargs) for p in remote_rel_paths]
