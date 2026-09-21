## pyS3M

[![Tests](https://github.com/JBeckwith/pyS3M/actions/workflows/tests.yml/badge.svg)](https://github.com/JBeckwith/pyS3M/actions/workflows/tests.yml)
[![Coverage](https://JBeckwith.github.io/pyS3M/badges/coverage.svg)](https://github.com/JBeckwith/pyS3M/actions/workflows/tests.yml)
[![Documentation](https://readthedocs.org/projects/pys3m/badge/?version=latest)](https://pys3m.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/1197322070.svg)](https://zenodo.org/badge/latestdoi/1197322070)

`pyS3M` (written in support of https://www.biorxiv.org/content/10.64898/2026.04.08.715690v1)
is a Python package of classes for analysing spatial-spectral single-molecule localisation
microscopy data — fitting, quality filtering, clustering, drift correction, FRC, and
simulation — usable from scripts, notebooks, or its desktop GUI. Example notebooks are
provided under `notebooks/analyses/` (fitting through resolution estimation) and
`notebooks/simulations/` (generating your own synthetic acquisitions), each running
end-to-end against data already bundled with the repo.

Documentation: https://pys3m.readthedocs.io/en/latest/index.html

## Installation

Requires Python >=3.11, <3.13 (tested on 3.12.3).

Install into a virtual environment, not your system Python — pyS3M pulls in a large,
version-pinned dependency tree (numpy, numba, scikit-learn, PyQt6, ...) that can otherwise
clash with other projects. See the [venv docs](https://docs.python.org/3/library/venv.html)
if you're not already using one:

```bash
python -m venv .venv
source .venv/bin/activate   # .venv\Scripts\activate on Windows
```

This repo uses [Git LFS](https://git-lfs.com) to store a large (~246 MB) spectral database
file (`Spectra/spectral_data.duckdb`). **Install Git LFS before cloning** — without it, the
checkout gets a small Git LFS pointer file instead of the real database, and `pip install .`
will refuse to proceed, aborting with a clear error pointing back here.

<details>
<summary>Install Git LFS (macOS / Windows / Linux)</summary>

- **macOS**: `brew install git-lfs`, or download the installer from
  [git-lfs.com](https://git-lfs.com).
- **Windows**: Git LFS ships with recent [Git for Windows](https://gitforwindows.org)
  installers by default — check with `git lfs version`. If it's missing, install via
  `winget install GitHub.GitLFS`, `choco install git-lfs`, or the installer from
  [git-lfs.com](https://git-lfs.com).
- **Linux**: `sudo apt install git-lfs` (Debian/Ubuntu), `sudo dnf install git-lfs` (Fedora),
  `sudo pacman -S git-lfs` (Arch), or download from [git-lfs.com](https://git-lfs.com).

Then, once per machine:

```bash
git lfs install
```

**Already cloned without Git LFS set up?** Don't re-clone — install Git LFS as above, then
from the repository root run `git lfs pull` to fetch the real content for any LFS pointer
stubs already checked out.

</details>

Clone the repository, then from its root:

```bash
pip install .
```

This installs `pyS3M` as a real package (`import pyS3M.SR_Functions`, etc. works from
anywhere — no `sys.path` hacks needed) along with its core analysis dependencies. Optional
extras layer on top as needed:

```bash
pip install .[notebooks]  # jupyterlab, seaborn, xarray, plotly, ...
pip install .[docs]       # Sphinx + the Read the Docs theme, for building docs locally
pip install .[dev]        # pytest, coverage, black, build
```

Extras can be combined, e.g. `pip install .[notebooks,dev]`. For an editable install while
developing `pyS3M` itself, add `-e`: `pip install -e .[dev]`.

## Running the GUI

```bash
pys3m-gui
```

(installed as a console script by `pip install .`), or equivalently `python run_gui.py`
from the repository root without installing.

## Quickstart

See the Getting Started guide for a minimal worked example and installation/GUI details:
https://pys3m.readthedocs.io/en/latest/getting-started.html

See `notebooks/analyses/` for fuller worked examples (single- and multi-FOV fitting, drift
correction, clustering, channel unmixing, Nile Red, FRC) and `notebooks/simulations/` for how
to generate your own synthetic acquisitions.

## License

Copyright © 2026, Cambridge Enterprise Limited, all rights reserved. This software is
provided **for academic use only** — see `LICENSE` for the full text. For commercial use,
contact `ls.ipportfolio@enterprise.cam.ac.uk` quoting LEE-11475-25.

## Contributing

Patches and contributions are very welcome! Please see `CONTRIBUTING.md` and
`CODE_OF_CONDUCT.md` for more details.
