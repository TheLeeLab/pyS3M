# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Optional, Tuple
from pyS3M.Constants import FilteringConstants


@dataclass
class ClusteringConfig:
    """Algorithm parameters for single-molecule extraction methods.

    Groups the non-filtering arguments that are passed identically on every
    call to ``extract_single_molecules_*``.  Pass a single
    ``ClusteringConfig`` instance via the ``config=`` keyword instead of the
    individual keyword arguments to reduce call-site verbosity.

    Filtering thresholds (chi_val, min_photoelectrons, etc.) remain in a separate
    :class:`~Constants.FilteringCriteria` object so the two concerns stay
    independent.

    Fields are annotated with which method(s) they apply to:

    - **all** — every ``extract_single_molecules_*`` method
    - **batch** — ``extract_single_molecules_batch`` only
    - **HDBSCAN** — ``extract_single_molecules_HDBSCAN``
    - **DBSCAN** — ``extract_single_molecules_DBSCAN``
    - **linked** — ``extract_single_molecules_linked``
    - **spectral_lap** — ``extract_single_molecules_spectral_lap``

    Example::

        from pyS3M.clustering import ClusteringConfig
        from pyS3M.Constants import FilteringCriteria

        cfg = ClusteringConfig(
            clustering_method="HDBSCAN",
            min_cluster_size=5,
            start_frame=100,
        )
        filt = FilteringCriteria(min_photoelectrons=200)

        sm_db, sf_db = SM_E.extract_single_molecules_batch(
            files, config=cfg, criteria=filt
        )
    """

    # ── Method selection ──────────────────────────────────────────────────
    clustering_method: str = "HDBSCAN"
    """(batch) Which algorithm to apply to each FOV."""

    # ── Common: HDBSCAN, DBSCAN, batch ───────────────────────────────────
    min_cluster_size: int = 10
    """(HDBSCAN, DBSCAN, batch) Minimum number of localisations per cluster."""

    # ── DBSCAN / batch-with-DBSCAN ────────────────────────────────────────
    epsilon_multiplier: float = 1.0
    """(DBSCAN, batch) Multiplier on median localisation precision to set ε."""

    # ── Linked / spectral LAP / batch-with-linked ─────────────────────────
    max_distance: float = 1.0
    """(linked, spectral_lap, batch) Maximum linking distance in pixels."""

    max_frames: int = 10
    """(linked, batch) Maximum frame gap for temporal linking."""

    # ── All methods ───────────────────────────────────────────────────────
    start_frame: int = 0
    """(all) Discard localisations with frame < start_frame."""

    verbose: bool = False
    """(all) Emit detailed progress via the module logger."""

    # ── Spectral LAP only ─────────────────────────────────────────────────
    max_dark_time: int = 1
    """(spectral_lap) Maximum frame gap allowed when linking tracks."""

    w_spatial: float = 1.0
    """(spectral_lap) Weight of the spatial cost term."""

    w_spectral: float = 0.5
    """(spectral_lap) Weight of the spectral cost term."""

    spectral_tol: float = FilteringConstants.MAX_COLOUR_ERROR
    """(spectral_lap) Maximum spectral distance for a valid link."""

    spectral_columns: Tuple[str, ...] = ("A_R", "A_G", "A_B")
    """(spectral_lap) DataFrame columns used as the spectral feature vector."""

    min_frames: int = 3
    """(spectral_lap) Minimum track length (frames) to retain a molecule."""

    D_prior: Optional[float] = None
    """(spectral_lap) Diffusion coefficient prior (µm²/s); None → data-driven."""

    dt: float = 1.0
    """(spectral_lap) Frame interval in seconds."""

    sigma_loc: float = 0.0
    """(spectral_lap) Localisation precision (pixels) used in cost calculation."""

    alpha: float = 3.0
    """(spectral_lap) Gap-penalty scale factor."""

    remove_static: bool = True
    """(spectral_lap) Remove immobile (static) localisations before linking."""

    static_eps: Optional[float] = None
    """(spectral_lap) DBSCAN ε for static-locs detection; None → auto."""

    static_min_samples: int = 10
    """(spectral_lap) DBSCAN min_samples for static-locs detection."""
