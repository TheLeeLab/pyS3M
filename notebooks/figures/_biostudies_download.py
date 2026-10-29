"""Shared helper for downloading files from the paper's BioStudies deposition
(accession S-BIAD3210) over plain HTTPS -- no auth needed, confirmed live
against the FTP mirror (see claude/paper_figures_reproduction.md §3.1).

Reused by every notebooks/figures/ notebook that needs real acquired data
("bucket B" in that doc) rather than reinventing a fetch helper per notebook.
Deliberately plain `requests`, no new heavyweight dependency (e.g. `pooch`) --
these are plain scoped file GETs, nothing a fuller download-manager buys much
over.
"""
import re
import time
from pathlib import Path
from urllib.parse import unquote

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
# A connection dropped mid-body surfaces as ChunkedEncodingError, which is not a
# ConnectionError subclass.
RETRYABLE = (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError)

# Optional local copy of the deposition (see use_local_mirror): (local root, prefix to strip).
_LOCAL_MIRROR = None


def use_local_mirror(local_root, strip_prefix: str = "2026_Multicolour_Paper/") -> None:
    """Read the deposition from a local copy instead of downloading it.

    With a mirror set, ``list_remote_folder`` lists the local folder and ``download_file``
    symlinks the local file into *dest_dir* instead of fetching it, so the fits land in
    *dest_dir* and the copy itself is never written to. A deposition path
    ``<strip_prefix><rest>`` maps to ``<local_root>/<rest>``.

    Args:
        local_root: Folder holding the deposition's tree (below *strip_prefix*), or None to
            go back to downloading.
        strip_prefix: Leading part of the deposition paths that the local tree lacks.
    """
    global _LOCAL_MIRROR
    _LOCAL_MIRROR = None if local_root is None else (Path(local_root), strip_prefix)


def _mirrored(remote_rel_path: str) -> Path:
    """Local path of a deposition path under the mirror set by ``use_local_mirror``."""
    root, prefix = _LOCAL_MIRROR
    rel = remote_rel_path.strip("/")
    if prefix and rel.startswith(prefix):
        rel = rel[len(prefix):]
    return root / rel


def _link_from_mirror(remote_rel_path: str, dest_dir: Path) -> Path:
    """Symlink one file of the local mirror into *dest_dir* (the mirror's ``download_file``)."""
    src = _mirrored(remote_rel_path)
    if not src.is_file():
        raise FileNotFoundError(f"{src} (local copy of {remote_rel_path}) not found")
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dest.is_symlink() and dest.resolve() == src.resolve():
        return dest
    if dest.exists() or dest.is_symlink():
        raise FileExistsError(f"{dest} exists and is not a link to {src}; move it out first")
    dest.symlink_to(src)
    print(f"Linked {src}")
    return dest


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
        except RETRYABLE as e:
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

    The ``.part`` file lives in a sibling folder (see ``_partial_dir``), not in
    *dest_dir*: ``Helper_Functions.file_search`` matches substrings recursively,
    so ``x.ome.tif.part`` would be fitted as a TIFF and ``x_metadata.txt.part``
    read as the metadata.

    With a local mirror set (``use_local_mirror``), links the local file instead.
    """
    if _LOCAL_MIRROR is not None:
        return _link_from_mirror(remote_rel_path, dest_dir)
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
    tmp = _partial_dir(dest_dir) / (dest.name + ".part")
    tmp.parent.mkdir(exist_ok=True)
    legacy_tmp = dest.with_name(dest.name + ".part")  # where earlier versions kept it
    if legacy_tmp.exists() and not tmp.exists():
        legacy_tmp.rename(tmp)
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


def _partial_dir(dest_dir: Path) -> Path:
    """Folder for in-progress ``.part`` files: a hidden sibling of *dest_dir*, outside it."""
    dest_dir = Path(dest_dir).resolve()
    return dest_dir.parent / f".{dest_dir.name}.partial"


def download_files(remote_rel_paths: list, dest_dir: Path, **kwargs) -> list:
    """Download each path in *remote_rel_paths* (see ``download_file``)."""
    return [download_file(p, dest_dir, **kwargs) for p in remote_rel_paths]


def _natural_key(name: str):
    """Sort key putting ``x_2.ome.tif`` before ``x_10.ome.tif`` (cf. ``file_search``)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"([0-9]+)", name)]


def list_remote_folder(remote_rel_folder: str) -> list:
    """Return the names of the files in one deposition folder (sub-folders excluded).

    Parses the mirror's HTML directory index (or lists the local copy set by
    ``use_local_mirror``); the names come back naturally sorted.
    """
    if _LOCAL_MIRROR is not None:
        folder = _mirrored(remote_rel_folder)
        if not folder.is_dir():
            raise FileNotFoundError(f"{folder} (local copy of {remote_rel_folder}) not found")
        return sorted((p.name for p in folder.iterdir() if p.is_file()), key=_natural_key)
    url = f"{BASE_URL}/{remote_rel_folder.strip('/')}/"

    def get():
        with requests.get(url, timeout=TIMEOUT) as r:
            r.raise_for_status()
            return r.text

    html = _retry(get, f"LIST {remote_rel_folder}")
    names = {unquote(h) for h in re.findall(r'href="([^"]+)"', html)}
    # Skip sort links (?C=N;O=D), the parent link (absolute path) and sub-folders (trailing /).
    names = [n for n in names if not n.startswith(("?", "/")) and "/" not in n]
    return sorted(names, key=_natural_key)


def download_acquisition(remote_folder: str, dest_dir: Path, stem: str = None,
                         first_file_only: bool = False,
                         image_suffixes: tuple = (".tif", ".tiff")) -> dict:
    """Download one MicroManager acquisition: its TIFFs *and* its ``*_metadata.txt``.

    The metadata sidecar carries the camera ROI the acquisition was cropped to, which
    ``fit_imaging_data`` needs to crop the calibration maps
    (``Helper_Functions.load_metadata_roi`` looks for it beside the TIFFs), so it always
    comes too, also with *first_file_only*. Downloads land flat in *dest_dir*, and
    re-running skips what is already complete (see ``download_file``).

    Args:
        remote_folder: Folder path relative to the deposition's ``Files/`` root.
        dest_dir: Local folder; keep one acquisition per folder, since the fitting
            routines read every TIFF and the first metadata file they find in it.
        stem: Only take files whose name starts with this (e.g.
            ``"..._MMStack_Pos0"`` when one folder holds several positions).
            Default: everything in the folder.
        first_file_only: Download only the first TIFF of the series (``FAST_MODE``).
        image_suffixes: Name endings counted as image files.

    Returns:
        ``{"images": [Path, ...], "metadata": [Path, ...]}`` of the local files.

    Raises:
        FileNotFoundError: if the folder has no matching TIFFs or no metadata file.
        RuntimeError: if *dest_dir* already holds image files outside this selection.
    """
    names = list_remote_folder(remote_folder)
    if stem:
        names = [n for n in names if n.startswith(stem)]
    images = [n for n in names if n.lower().endswith(image_suffixes)]
    metadata = [n for n in names if n.endswith("metadata.txt")]
    if not images:
        raise FileNotFoundError(f"No {image_suffixes} files in {remote_folder} (stem={stem!r})")
    if not metadata:
        raise FileNotFoundError(
            f"No *metadata.txt in {remote_folder} (stem={stem!r}); without it the fit "
            "can't find the acquisition's camera ROI")
    if first_file_only:
        images = images[:1]

    # The fit reads every TIFF under the folder, so files from another acquisition, or the
    # rest of this series left by a full run, would be fitted too.
    dest_dir = Path(dest_dir)
    extra = sorted(str(p.relative_to(dest_dir)) for p in dest_dir.rglob("*")
                   if p.is_file() and p.name.lower().endswith(image_suffixes)
                   and p.name not in images) if dest_dir.exists() else []
    if extra:
        raise RuntimeError(
            f"{dest_dir} already holds {len(extra)} image file(s) outside this selection "
            f"(e.g. {extra[0]}), which the fit would also read. Use a separate folder per "
            "acquisition and per FAST_MODE setting, or move them out.")

    total = len(images) + len(metadata)
    print(f"{remote_folder}: {len(images)} image file(s) + {len(metadata)} metadata file(s)")
    for i, name in enumerate(metadata + images, 1):
        print(f"[{i}/{total}] ", end="")
        download_file(f"{remote_folder.strip('/')}/{name}", dest_dir)

    return {"images": [dest_dir / n for n in images],
            "metadata": [dest_dir / n for n in metadata]}
