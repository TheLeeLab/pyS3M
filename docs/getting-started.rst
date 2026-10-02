Introduction
============

``pyS3M`` is a Python package written in support of spatial-spectral single-molecule
spectroscopy analysis by Joseph S. Beckwith.  The preprint describing the method can
be found on `bioRxiv <https://www.biorxiv.org/content/10.64898/2026.04.08.715690v1>`_.

The package provides a set of Python classes that can be used from scripts,
interactive Jupyter notebooks, or a bundled PyQt6 GUI to:

- Fit single-molecule data from spatially patterned detectors
- Correct lateral drift using fiducial beads or AIM (Adaptive Intersection Maximization)
- Cluster and co-localise multi-channel localisations
- Simulate multicolour SMLM experiments for method validation
- Detect and characterise single-molecule steps and FRET transitions

Requires Python >=3.11, <3.13 (tested on 3.12.3) — the ceiling comes from a real
dependency constraint (``colour-demosaicing`` caps at <3.13), not an arbitrary choice.

Installation
============

Install into a virtual environment, not your system Python — ``pyS3M`` pulls in a large,
version-pinned dependency tree (numpy, numba, scikit-learn, PyQt6, ...) that can otherwise
clash with other projects. See the `venv docs <https://docs.python.org/3/library/venv.html>`_
if you're not already using one:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate   # .venv\Scripts\activate on Windows

Clone the repository, then from its root:

.. code-block:: bash

   pip install .

This installs ``pyS3M`` as a real package (``import pyS3M.SR_Functions``, etc. works
from anywhere — no ``sys.path`` hacks needed) along with its core analysis
dependencies and the Jupyter stack needed to run the example and paper-figure notebooks
(``jupyter lab`` or ``jupyter notebook``). Optional extras layer on top as needed:

.. code-block:: bash

   pip install ".[docs]"     # Sphinx + the Read the Docs theme, for building docs locally
   pip install ".[dev]"      # pytest, coverage, black, build

Extras can be combined, e.g. ``pip install ".[docs,dev]"`` (the quotes stop shells like
zsh treating the brackets as a glob). For an editable install while developing ``pyS3M``
itself, add ``-e``: ``pip install -e ".[dev]"``.

Running the GUI
================

.. code-block:: bash

   pys3m-gui

(installed as a console script by ``pip install .``), or equivalently
``python run_gui.py`` from the repository root without installing.

Getting Started
===============

Example notebooks covering the main workflows are provided in ``notebooks/analyses/``
(single- and multi-FOV fitting, drift correction, clustering, channel unmixing, Nile Red,
FRC) and ``notebooks/simulations/`` (generating your own synthetic acquisitions) — this is the
exact worked example from ``01_getting_started_smlm_fitting.ipynb``, runnable end-to-end
against data already bundled with the repo.
A minimal analysis looks like, using :class:`~pyS3M.AnalysisPipeline.AnalysisPipeline`
configured via :class:`~pyS3M.AnalysisPipeline.FittingConfig` and
:class:`~pyS3M.Constants.AnalysisConfig`, then
:meth:`~pyS3M.AnalysisPipeline.AnalysisPipeline.load_calibration`,
:meth:`~pyS3M.AnalysisPipeline.AnalysisPipeline.fit`,
:meth:`~pyS3M.AnalysisPipeline.AnalysisPipeline.load_localisations`, and
:meth:`~pyS3M.AnalysisPipeline.AnalysisPipeline.filter_and_cluster` (see the
:doc:`Core Analysis API reference <api_core>` for full parameter documentation):

.. code-block:: python

   from pathlib import Path
   from pyS3M.AnalysisPipeline import AnalysisPipeline, FittingConfig
   from pyS3M.Constants import AnalysisConfig

   data_dir = Path("path/to/tiffs/")

   cfg = AnalysisConfig(display=False, save_figures=True,
                        output_dir=Path("results/"), dpi=150)
   pipe = AnalysisPipeline(camera="ximea", config=cfg)
   pipe.load_calibration(Path("Camera_Calibrations/Ximea_Camera"))

   fc = FittingConfig(peak_wavelength=0.638, pfa=1e-3)
   pipe.fit(data_dir, mode="smlm", fitting_config=fc)

   locs = pipe.load_localisations(data_dir)
   sm_db, sf_db = pipe.filter_and_cluster(locs)

See ``notebooks/analyses/`` for fuller worked examples (single- and multi-FOV fitting,
drift correction, clustering, channel unmixing, Nile Red, FRC) and ``notebooks/simulations/``
for how to generate your own synthetic acquisitions.

Contributing
============

Patches and contributions are welcome.  Please see ``CONTRIBUTING.md`` and
``CODE_OF_CONDUCT.md`` in the repository root for details.
