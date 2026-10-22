"""Read the EBI BioImage Archive FTP credentials from a local text file.

The credentials are never stored in the notebooks (they are tracked in git). Instead each
machine that uploads keeps a copy of the credentials file, by default at
``~/.config/pyS3M/ebi_ftp_credentials.txt`` (outside the repository), or wherever the
``EBI_CREDENTIALS_FILE`` environment variable points. Format, one ``KEY=VALUE`` per line
(``#`` comments allowed; the file can also be ``source``d by a shell for ``lftp``)::

    WEBIN_USER=bs-upload
    WEBIN_PASS=...
    EBI_FTP_DIR=/dc/...
"""
import os
from pathlib import Path

DEFAULT_PATH = Path.home() / ".config" / "pyS3M" / "ebi_ftp_credentials.txt"
REQUIRED_KEYS = ("WEBIN_USER", "WEBIN_PASS", "EBI_FTP_DIR")


def credentials_path() -> Path:
    return Path(os.environ.get("EBI_CREDENTIALS_FILE", DEFAULT_PATH)).expanduser()


def load_ebi_credentials(path=None) -> dict:
    """Return {"WEBIN_USER", "WEBIN_PASS", "EBI_FTP_DIR"} from the credentials file."""
    path = Path(path).expanduser() if path else credentials_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"EBI credentials file not found: {path}\n"
            "Copy ebi_ftp_credentials.txt to this machine (see notebooks/EBI_Upload/_ebi_credentials.py), "
            "or point EBI_CREDENTIALS_FILE at it."
        )
    creds = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        creds[key.strip()] = value.strip().strip('"').strip("'")
    missing = [k for k in REQUIRED_KEYS if not creds.get(k)]
    if missing:
        raise ValueError(f"{path} is missing: {', '.join(missing)}")
    if path.stat().st_mode & 0o077:
        print(f"Note: {path} is readable by other users -- consider `chmod 600 {path}`.")
    return {k: creds[k] for k in REQUIRED_KEYS}
