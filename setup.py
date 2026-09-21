"""pyproject.toml declares all real project metadata; this file exists solely to run a
build-time check, before setuptools packages anything, that Spectra/spectral_data.duckdb is
the real Git LFS content and not an unfetched pointer stub -- see README.md's "Installation"
section for the Git LFS setup this depends on.
"""
import sys
from pathlib import Path

from setuptools import setup

_DUCKDB_PATH = Path(__file__).parent / "Spectra" / "spectral_data.duckdb"
_LFS_POINTER_SIGNATURE = b"version https://git-lfs.github.com/spec/v1"

if _DUCKDB_PATH.exists() and _DUCKDB_PATH.read_bytes()[:64].startswith(_LFS_POINTER_SIGNATURE):
    sys.exit(
        f"\n{_DUCKDB_PATH} is a Git LFS pointer, not the real database file -- Git LFS "
        "content wasn't fetched for this checkout.\n\n"
        "Install Git LFS and run `git lfs pull` from the repository root, then retry.\n"
        "See the 'Installation' section of README.md for platform-specific instructions.\n"
    )

setup()
