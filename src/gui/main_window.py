import enum
import logging
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from PyQt6.QtWidgets import (
    QMainWindow, QDockWidget, QWidget, QVBoxLayout,
    QScrollArea, QLabel, QMessageBox, QApplication, QTabWidget,
)
from PyQt6.QtCore import Qt, pyqtSignal, QSettings

from pyS3M.gui.panels.calibration_panel import CalibrationCalcPanel
from pyS3M.gui.panels.setup_panel import SetupPanel
from pyS3M.gui.panels.fitting_panel import FittingPanel
from pyS3M.gui.panels.postproc_panel import PostProcPanel
from pyS3M.gui.panels.drift_panel import DriftPanel
from pyS3M.gui.panels.frc_panel import FRCPanel
from pyS3M.gui.panels.channel_unmixing_panel import ChannelUnmixingPanel
from pyS3M.gui.panels.nile_red_panel import NileRedPanel
from pyS3M.gui.panels.results_panel import ResultsPanel
from pyS3M.gui.panels.simulation_panel import SimulationPanel, _PHOTON_LEVELS
from pyS3M.gui.widgets.log_widget import LogWidget, QtLogHandler
from pyS3M.gui.widgets.progress_widget import ProgressWidget
from pyS3M.gui.worker import AnalysisWorker

logger = logging.getLogger(__name__)


class AppState(enum.Enum):
    IDLE = "idle"
    CALIBRATED = "calibrated"
    FITTED = "fitted"
    UNDRIFTED = "undrifted"
    CLUSTERED = "clustered"


class MainWindow(QMainWindow):
    state_changed = pyqtSignal(str)

    # Controls-dock tab index -> ResultsPanel context key (see _on_ctrl_tab_changed).
    _CTRL_TAB_CONTEXTS = ("analysis", "simulation", "frc", "channel_unmixing", "nile_red", "cmos_calibration")

    def __init__(self):
        super().__init__()
        self.setWindowTitle("pyS3M")
        self.resize(1400, 900)

        self._state = AppState.IDLE
        self.pipeline = None
        self._worker: AnalysisWorker | None = None      # main pipeline worker
        self._aux_worker: AnalysisWorker | None = None  # viz-only worker (non-blocking)
        self._fitted_data_dir: str | None = None
        self._fov_data: list = []                       # [(locs_df, tif_path)] per FOV
        self._undrifted_locs = None                      # DataFrame, set once undrift succeeds
        self._sm_db = None
        self._sf_db = None
        self._nile_red_db = None                          # per-loc DataFrame + wl_pixel columns
        self._nile_red_grid = None                        # grid_info dict from fit_wavelengths_pixelated
        self._nile_red_input_df = None                     # explicitly-loaded override for sf_db

        self._build_ui()
        self._connect_signals()
        self._install_log_handler()
        self._restore_settings()
        self._update_state(AppState.IDLE)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        self.results_panel = ResultsPanel(self)
        self.setCentralWidget(self.results_panel)

        self.calibration_panel = CalibrationCalcPanel(self)
        self.setup_panel = SetupPanel(self)
        self.fitting_panel = FittingPanel(self)
        self.postproc_panel = PostProcPanel(self)
        self.drift_panel = DriftPanel(self)
        self.simulation_panel = SimulationPanel(self)
        self.frc_panel = FRCPanel(self)
        self.channel_unmixing_panel = ChannelUnmixingPanel(self)
        self.nile_red_panel = NileRedPanel(self)

        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(4, 4, 4, 4)
        vbox.setSpacing(8)
        vbox.addWidget(self.setup_panel)
        vbox.addWidget(self.fitting_panel)
        vbox.addWidget(self.postproc_panel)
        vbox.addWidget(self.drift_panel)
        vbox.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(container)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.ctrl_tabs = QTabWidget()
        self.ctrl_tabs.addTab(scroll, "Analysis")

        sim_scroll = QScrollArea()
        sim_scroll.setWidget(self.simulation_panel)
        sim_scroll.setWidgetResizable(True)
        sim_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.ctrl_tabs.addTab(sim_scroll, "Simulation")

        self.ctrl_tabs.addTab(self.frc_panel, "FRC")
        self.ctrl_tabs.addTab(self.channel_unmixing_panel, "Unmixing")
        self.ctrl_tabs.addTab(self.nile_red_panel, "Nile Red")
        self.ctrl_tabs.addTab(self.calibration_panel, "CMOS Calibration")

        ctrl_dock = QDockWidget("Controls", self)
        ctrl_dock.setWidget(self.ctrl_tabs)
        ctrl_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        ctrl_dock.setMinimumWidth(330)
        ctrl_dock.setMaximumWidth(440)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, ctrl_dock)

        bottom = QWidget()
        blay = QVBoxLayout(bottom)
        blay.setContentsMargins(4, 4, 4, 4)
        blay.setSpacing(4)
        self.progress_widget = ProgressWidget(bottom)
        self.log_widget = LogWidget(bottom)
        blay.addWidget(self.progress_widget)
        blay.addWidget(self.log_widget)

        log_dock = QDockWidget("Log", self)
        log_dock.setWidget(bottom)
        log_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        log_dock.setMaximumHeight(220)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, log_dock)

        self._status_label = QLabel("Idle")
        self.statusBar().addPermanentWidget(self._status_label)

    def _connect_signals(self):
        self.state_changed.connect(self.setup_panel.on_state_changed)
        self.state_changed.connect(self.fitting_panel.on_state_changed)
        self.state_changed.connect(self.postproc_panel.on_state_changed)
        self.state_changed.connect(self.drift_panel.on_state_changed)
        self.state_changed.connect(self.frc_panel.on_state_changed)
        self.state_changed.connect(self.channel_unmixing_panel.on_state_changed)
        self.state_changed.connect(self.simulation_panel.on_state_changed)
        self.state_changed.connect(self.nile_red_panel.on_state_changed)

        self.calibration_panel.calibration_compute_requested.connect(self._on_run_calibration)
        self.setup_panel.calibration_requested.connect(self._on_load_calibration)
        self.fitting_panel.preview_requested.connect(self._on_preview_fitting)
        self.fitting_panel.fit_requested.connect(self._on_run_fitting)
        self.fitting_panel.stats_refresh_requested.connect(self._on_stats_refresh)
        self.postproc_panel.load_locs_requested.connect(self._on_load_locs)
        self.postproc_panel.cluster_requested.connect(self._on_run_clustering)
        self.postproc_panel.save_requested.connect(self._on_save_clustering)
        self.postproc_panel.clear_requested.connect(self._on_clear_clustering)
        self.drift_panel.undrift_requested.connect(self._on_undrift)
        self.drift_panel.clear_requested.connect(self._on_clear_drift)
        self.frc_panel.frc_requested.connect(self._on_run_frc)
        self.channel_unmixing_panel.unmixing_requested.connect(self._on_channel_unmixing)
        self.channel_unmixing_panel.frc_per_channel_requested.connect(self._on_run_frc_per_channel)
        self.nile_red_panel.fit_requested.connect(self._on_run_nile_red)
        self.nile_red_panel.clear_requested.connect(self._on_clear_nile_red)
        self.nile_red_panel.load_locs_requested.connect(self._on_load_nile_red_locs)
        self.results_panel.fov_requested.connect(self._on_fov_requested)
        self.simulation_panel.simulation_requested.connect(self._on_run_simulation)
        self.simulation_panel.pattern_simulation_requested.connect(self._on_run_pattern_simulation)
        self.simulation_panel.clear_pattern_requested.connect(self._on_clear_pattern_simulation)
        self.fitting_panel.clear_requested.connect(self._on_clear_fitting)
        self.ctrl_tabs.currentChanged.connect(self._on_ctrl_tab_changed)
        self._on_ctrl_tab_changed(self.ctrl_tabs.currentIndex())

    def _on_ctrl_tab_changed(self, index: int):
        if 0 <= index < len(self._CTRL_TAB_CONTEXTS):
            self.results_panel.set_context(self._CTRL_TAB_CONTEXTS[index])

    def _install_log_handler(self):
        self._log_handler = QtLogHandler()
        self._log_handler.setFormatter(
            logging.Formatter("%(asctime)s  %(name)s  %(message)s", datefmt="%H:%M:%S")
        )
        self._log_handler.message.connect(self.log_widget.append)
        logging.getLogger().addHandler(self._log_handler)
        logging.getLogger().setLevel(logging.INFO)

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def _update_state(self, new_state: AppState):
        self._state = new_state
        self._status_label.setText(new_state.value.replace("_", " ").title())
        self.state_changed.emit(new_state.value)

    # ------------------------------------------------------------------
    # Settings persistence
    # ------------------------------------------------------------------

    def _restore_settings(self):
        s = QSettings()
        if cal := s.value("cal_dir", ""):
            self.setup_panel.set_cal_dir(cal)
        # data_dir is deliberately not restored — FittingPanel's FolderPicker
        # already opens into test_tiffs/ by default (see fitting_panel.py's
        # _TEST_TIFFS_DIR); restoring a remembered last-used path here would
        # permanently shadow that default once anything else was ever picked.
        s.remove("data_dir")

    def _save_settings(self):
        s = QSettings()
        s.setValue("cal_dir", self.setup_panel.cal_dir)

    # ------------------------------------------------------------------
    # Worker lifecycle helpers
    # ------------------------------------------------------------------

    def _worker_running(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def _invalidate_nile_red(self):
        """Nile Red results are derived from sf_db — call whenever fitting,
        drift correction, or clustering are (re-)run or cleared, so a stale
        wavelength map never lingers against data it no longer matches."""
        self._nile_red_db = None
        self._nile_red_grid = None
        self.nile_red_panel.set_clear_enabled(False)
        self.results_panel.clear_nile_red_figure()

    def _reset_busy(self):
        self.calibration_panel.set_busy(False)
        self.setup_panel.set_busy(False)
        self.fitting_panel.set_busy(False)
        self.postproc_panel.set_busy(False)
        self.drift_panel.set_busy(False)
        self.frc_panel.set_busy(False)
        self.channel_unmixing_panel.set_busy(False)
        self.simulation_panel.set_busy(False)
        self.nile_red_panel.set_busy(False)
        # set_busy(True) unconditionally disables each panel's "Clear Results"
        # button; restore it to whatever's actually true (matters after a
        # failed re-run when a still-valid prior result exists to clear).
        self.fitting_panel.set_clear_enabled(self._fitted_data_dir is not None)
        self.drift_panel.set_clear_enabled(self._undrifted_locs is not None)
        self.postproc_panel.set_clear_enabled(self._sm_db is not None)
        self.nile_red_panel.set_clear_enabled(self._nile_red_db is not None)
        self.progress_widget.reset()

    def _start_worker(self, fn) -> AnalysisWorker:
        """Create, store, and wire a main-pipeline worker.

        The worker clears ``self._worker`` via ``finished`` (after the thread
        has fully exited) rather than inside result/error slots, which would
        drop the Python reference while the C++ QThread is still cleaning up.
        """
        worker = AnalysisWorker(fn)
        self._worker = worker
        worker.progress.connect(self.progress_widget.update)
        worker.log.connect(self.log_widget.append)
        worker.error.connect(self._on_worker_error)
        # Clear the reference only after the OS thread is fully done.
        worker.finished.connect(lambda w=worker: self._on_main_worker_finished(w))
        return worker

    def _on_main_worker_finished(self, w: AnalysisWorker):
        if self._worker is w:
            self._worker = None

    def _on_worker_error(self, msg: str):
        self._reset_busy()
        self.log_widget.append(f"ERROR:\n{msg}")
        self._status_label.setText("Error")
        QMessageBox.critical(self, "Pipeline error", msg[:600])

    # ------------------------------------------------------------------
    # Figure helpers (use Figure directly — no pyplot, thread-safe)
    # ------------------------------------------------------------------

    def _compute_mip(self, tif_path: str, max_frames: int = 200) -> np.ndarray:
        """Max-intensity projection over the first max_frames frames of a TIFF."""
        import tifffile
        mip = None
        chunk_size = 50
        with tifffile.TiffFile(tif_path) as tif:
            n_frames = min(len(tif.pages), max_frames)
            for start in range(0, n_frames, chunk_size):
                end = min(start + chunk_size, n_frames)
                chunk = np.stack(
                    [tif.pages[i].asarray() for i in range(start, end)], axis=0
                ).astype(np.float32)
                chunk_mip = chunk.max(axis=0)
                mip = chunk_mip if mip is None else np.maximum(mip, chunk_mip)
        return mip

    def _make_locs_render_figure(
        self,
        locs,
        data_dir: str | None = None,
        tif_path: str | None = None,
        title: str = "Localisations",
    ) -> Figure:
        """Side-by-side MIP + gaussian_colour render.

        *tif_path* — specific TIFF file to use for the MIP (preferred).
        *data_dir* — folder; first TIFF found is used when *tif_path* is absent.
        Falls back to the plain scatter figure if the render cannot be produced.
        """
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.insert(0, str(_Path(__file__).parent.parent))

        # --- MIP ---
        mip = None
        if tif_path is not None:
            try:
                mip = self._compute_mip(tif_path)
            except Exception:
                pass
        elif data_dir is not None:
            tif_files = sorted(_Path(data_dir).glob("*.tif")) + sorted(_Path(data_dir).glob("*.tiff"))
            if tif_files:
                try:
                    mip = self._compute_mip(str(tif_files[0]))
                except Exception:
                    pass

        # --- render: true RGB if all three spectral columns present, else scalar colourmap ---
        render_img = None
        spectral_cols = ("A_R", "A_G", "A_B")
        has_rgb = all(c in locs.columns for c in spectral_cols)
        if all(c in locs.columns for c in ("xc", "yc", "xc_err", "yc_err")) and len(locs) > 0:
            try:
                import pyS3M.render as _render
                base_cols = ["xc", "yc", "xc_err", "yc_err"]
                extra = [c for c in (("A_R", "A_G", "A_B") if has_rgb else ("A_R",))
                         if c in locs.columns]
                if "photons" in locs.columns:
                    extra.append("photons")
                subset = locs[base_cols + extra].dropna()
                if len(subset) > 0:
                    locs_rec = subset.to_records(index=False)
                    y_min = max(0.0, float(locs_rec["yc"].min()) - 1)
                    x_min = max(0.0, float(locs_rec["xc"].min()) - 1)
                    y_max = float(locs_rec["yc"].max()) + 1
                    x_max = float(locs_rec["xc"].max()) + 1
                    if has_rgb and all(c in locs_rec.dtype.names for c in spectral_cols):
                        _, _, render_img = _render.render_gaussian_RGB(
                            locs_rec, 1.0,
                            y_min, x_min, y_max, x_max,
                            min_blur_width=1.0,
                            mindensperc=1, maxdensperc=99.9,
                            densitymin=0.1,
                        )
                    else:
                        _, _, render_img = _render.render_gaussian_colour(
                            locs_rec, 1.0,
                            y_min, x_min, y_max, x_max,
                            min_blur_width=1.0,
                            cparam="A_R",
                            c_min=0.3, c_max=0.75,
                            mindensperc=1, maxdensperc=99.9,
                            densitymin=0.1,
                            cmap_string="jet",
                        )
            except Exception:
                pass

        if render_img is None:
            return self._make_scatter_figure(locs, title)

        ncols = 2 if mip is not None else 1
        fig = Figure(figsize=(6 * ncols, 5), dpi=100, layout="constrained")

        if mip is not None:
            ax_mip = fig.add_subplot(1, 2, 1)
            ax_mip.imshow(
                mip, cmap="gray", aspect="equal",
                vmin=np.percentile(mip, 1), vmax=np.percentile(mip, 99.8),
            )
            ax_mip.set_title("Max Intensity Projection")
            ax_mip.set_xticks([])
            ax_mip.set_yticks([])
            ax_render = fig.add_subplot(1, 2, 2)
        else:
            ax_render = fig.add_subplot(1, 1, 1)

        ax_render.imshow(render_img, aspect="equal", origin="upper")
        ax_render.set_title(f"{title}  ({len(locs):,})")
        ax_render.set_xticks([])
        ax_render.set_yticks([])

        return fig

    def _make_scatter_figure(self, locs, title: str = "Localisations") -> Figure:
        # Defensive: this is the last-resort fallback in _make_locs_render_figure,
        # reached whenever the RGB/scalar render can't be built -- including when
        # *locs* isn't actually localisation data at all (e.g. a differently-shaped
        # companion .h5 glob-matched alongside real results; see claude/LOG.md
        # "GUI bug reports from real Windows/Linux usage"). A missing xc/yc here
        # must not crash the whole fitting/loading worker -- show a placeholder
        # instead of raising, so the real results already recorded upstream of
        # this figure aren't lost along with it.
        if not all(c in locs.columns for c in ("xc", "yc")):
            fig = Figure(figsize=(6, 6), dpi=100, layout="constrained")
            ax = fig.add_subplot(111)
            ax.text(
                0.5, 0.5, "No xc/yc columns to render\n(not localisation data?)",
                ha="center", va="center", transform=ax.transAxes, color="gray",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title)
            return fig

        pixel_size_nm = self.pipeline.pixel_size * 1000
        x = locs["xc"].values * pixel_size_nm
        y = locs["yc"].values * pixel_size_nm

        fig = Figure(figsize=(6, 6), dpi=100, layout="constrained")
        ax = fig.add_subplot(111)

        spectral_cols = ("A_R", "A_G", "A_B")
        if all(c in locs.columns for c in spectral_cols):
            total = locs["A_R"].values + locs["A_G"].values + locs["A_B"].values
            total = np.where(total == 0, 1.0, total)
            c = locs["A_R"].values / total
            sc = ax.scatter(x, y, c=c, s=1, cmap="RdYlBu_r",
                            vmin=0, vmax=1, rasterized=True, alpha=0.5, linewidths=0)
            fig.colorbar(sc, ax=ax, label="A_R fraction", shrink=0.8)
        else:
            ax.scatter(x, y, s=1, rasterized=True, alpha=0.5)

        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
        ax.set_title(f"{title}  ({len(locs):,})")
        ax.set_aspect("equal")
        ax.invert_yaxis()
        return fig

    def _make_stats_figure(
        self,
        locs,
        photon_range=(100, 1_000_000),
        sm_locs=None,
    ) -> Figure:
        # After H5 write, A_R/G/B are normalised fractions; the true total is in "photons".
        # Use "photons" if present; fall back to raw sum only for un-normalised data.
        # When sm_locs is provided (post-clustering), render a 2×3 grid comparing
        # per-localisation (locs / sf_db) with per-molecule (sm_locs / sm_db) stats.
        col_colors = [("A_R", "tomato"), ("A_G", "mediumseagreen"), ("A_B", "cornflowerblue")]
        min_ph, max_ph = photon_range

        def _extract(df):
            """Return (filtered_df, total_photons_array) applying photon_range mask."""
            has_photons = "photons" in df.columns
            spectral_cols = ("A_R", "A_G", "A_B")
            has_spectral = has_photons or all(c in df.columns for c in spectral_cols)
            if not has_spectral:
                return None, None
            total_all = (
                df["photons"].values if has_photons
                else df["A_R"].values + df["A_G"].values + df["A_B"].values
            )
            mask = np.isfinite(total_all) & (total_all >= min_ph) & (total_all <= max_ph)
            return df[mask], total_all[mask]

        two_rows = sm_locs is not None and not sm_locs.empty

        if two_rows:
            fig = Figure(figsize=(13, 6), dpi=100, layout="constrained")
            rows = [
                ("Localisations", locs),
                ("Molecules",     sm_locs),
            ]
            for row_idx, (row_label, df) in enumerate(rows):
                filtered, total = _extract(df)
                axes = [fig.add_subplot(2, 4, row_idx * 4 + col + 1) for col in range(4)]
                if filtered is None:
                    axes[0].text(0.5, 0.5, "No data", ha="center", va="center",
                                 transform=axes[0].transAxes, color="gray")
                    continue
                for ax, (col, color) in zip(axes[:3], col_colors):
                    ch = filtered[col].values if col in filtered.columns else np.array([])
                    ch = ch[np.isfinite(ch)]
                    ax.hist(ch, bins=60, color=color, alpha=0.8, range=(0, 1))
                    ax.set_xlabel(f"{col} fraction")
                    ax.set_ylabel(f"{row_label}\nCount" if col == "A_R" else "Count")
                axes[3].hist(total, bins=60, color="slategray", alpha=0.8)
                axes[3].set_xlabel("Total photons")
                axes[3].set_ylabel("Count")
                axes[3].set_title(f"n = {len(filtered):,}")
        else:
            filtered, total = _extract(locs)
            fig = Figure(figsize=(10, 3), dpi=100, layout="constrained")
            if filtered is None:
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, "No photon data in file", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
            else:
                axes = fig.subplots(1, 4)
                for ax, (col, color) in zip(axes[:3], col_colors):
                    ch = filtered[col].values if col in filtered.columns else np.array([])
                    ch = ch[np.isfinite(ch)]
                    ax.hist(ch, bins=60, color=color, alpha=0.8, range=(0, 1))
                    ax.set_xlabel(f"{col} fraction")
                    ax.set_ylabel("Count")
                    ax.set_title(col)
                axes[3].hist(total, bins=60, color="slategray", alpha=0.8)
                axes[3].set_xlabel("Total photons")
                axes[3].set_ylabel("Count")
                axes[3].set_title(f"Total  (n={len(filtered):,})")
        return fig

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def _on_load_calibration(self, camera: str, pixel_size: float, cal_dir: str):
        if self._worker_running():
            return

        def _do():
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).parent.parent))
            from pyS3M.AnalysisPipeline import AnalysisPipeline
            from pyS3M.Constants import AnalysisConfig
            cfg = AnalysisConfig(
                display=False,
                progress_callback=lambda f, m: worker.progress.emit(f, m),
                logging_callback=lambda m: worker.log.emit(m),
            )
            pipe = AnalysisPipeline(camera=camera, pixel_size=pixel_size, config=cfg)
            pipe.load_calibration(Path(cal_dir))
            return pipe

        worker = self._start_worker(_do)
        worker.result.connect(self._on_calibration_done)
        self.setup_panel.set_busy(True)
        worker.start()

    def _on_calibration_done(self, pipe):
        self.pipeline = pipe
        self._save_settings()
        self.setup_panel.set_busy(False)
        shape = pipe.gain_map.shape
        self.setup_panel.show_calibration_status(f"✓ {shape[0]}×{shape[1]}")
        self.progress_widget.update(1.0, "Calibration loaded")
        self._update_state(AppState.CALIBRATED)

    def _on_run_calibration(self, camera: str, raw_dir: str, mode: str = "rgb"):
        if self._worker_running():
            return

        def _do():
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).parent.parent))
            from pyS3M.AnalysisPipeline import AnalysisPipeline
            from pyS3M.Constants import AnalysisConfig
            cfg = AnalysisConfig(
                display=False,
                progress_callback=lambda f, m: worker.progress.emit(f, m),
                logging_callback=lambda m: worker.log.emit(m),
            )
            pipe = AnalysisPipeline(camera=camera, config=cfg)
            pipe.calibrate(raw_dir, mode=mode)
            fig = self._make_calibration_figure(pipe)
            return pipe, fig

        worker = self._start_worker(_do)
        worker.result.connect(self._on_calibration_computed)
        self.calibration_panel.set_busy(True)
        worker.start()

    def _on_calibration_computed(self, result):
        pipe, fig = result
        self.pipeline = pipe
        self._save_settings()
        self.calibration_panel.set_busy(False)
        shape = pipe.gain_map.shape
        self.calibration_panel.show_calibration_status(f"✓ {shape[0]}×{shape[1]}")
        self.progress_widget.update(1.0, "Calibration computed")
        self._update_state(AppState.CALIBRATED)
        if fig is not None:
            self.results_panel.set_calibration_figure(fig)

    def _make_calibration_figure(self, pipe) -> Figure:
        """5-panel imshow grid of the computed calibration maps, direct-Figure
        style matching _make_drift_figure/_make_frc_figure rather than reusing
        any notebook-oriented plotting method."""
        fig = Figure(figsize=(12, 4), dpi=100, layout="constrained")
        axes = fig.subplots(1, 5)
        panels = [
            ("Gain (ADU/e⁻)", pipe.gain_map),
            ("Offset (ADU)", pipe.offset_map),
            ("Variance (ADU²)", pipe.variance),
            ("Read noise (e⁻)", pipe.read_noise),
            ("Relative QE", pipe.rqe),
        ]
        for ax, (title, data) in zip(axes, panels):
            im = ax.imshow(data, cmap="viridis")
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        return fig

    # ------------------------------------------------------------------
    # Preview fitting — runs on the main thread because example_spots_singleframe
    # calls plt.subplots() via PlottingBase, which creates a Qt canvas.
    # Qt widgets must be created on the main thread.
    # ------------------------------------------------------------------

    def _on_preview_fitting(self, data_dir: str, fitting_config):
        if self._worker_running() or self.pipeline is None:
            return

        self.fitting_panel.set_preview_busy(True)
        self.progress_widget.update(0.0, "Running preview…")
        self._status_label.setText("Preview running…")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()   # flush UI updates before blocking

        try:
            sf = self.pipeline.make_smoothing_function(fitting_config.sigma)
            result = self.pipeline.sr.example_spots_singleframe(
                image_folder=Path(data_dir),
                smoothing_function=sf,
                gain_map=self.pipeline.gain_map,
                offset_map=self.pipeline.offset_map,
                rqe=self.pipeline.rqe,
                read_noise=self.pipeline.read_noise,
                variance=self.pipeline.variance,
                pfa=fitting_config.pfa,
                ROI_size=fitting_config.ROI_size,
                peak_wavelength=fitting_config.peak_wavelength,
                NA=fitting_config.NA,
                pixel_size=self.pipeline.pixel_size,
                sigma=fitting_config.sigma,
                fraction_true=fitting_config.fraction_true,
                use_variance_aware_demosaic=fitting_config.use_variance_aware_demosaic,
            )
            if result is not None:
                fig, _ = result
                self.results_panel.set_preview_figure(fig)
            self.progress_widget.update(1.0, "Preview ready")
        except Exception:
            import traceback as _tb
            self.log_widget.append(f"Preview error:\n{_tb.format_exc()}")
            self.progress_widget.update(0.0, "Preview failed")
        finally:
            QApplication.restoreOverrideCursor()
            self.fitting_panel.set_preview_busy(False)
            self._update_state(self._state)  # restore status bar label

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def _on_run_fitting(self, data_dir: str, mode: str, fitting_config, extra_kwargs: dict):
        if self._worker_running() or self.pipeline is None:
            return

        def _do():
            self.pipeline.config.progress_callback = lambda f, m: worker.progress.emit(f, m)
            self.pipeline.config.logging_callback = lambda m: worker.log.emit(m)
            # fov_data: list of (locs_df, tif_path_or_None), one entry per TIFF/H5
            fov_data = self.pipeline.fit(
                Path(data_dir), mode=mode, fitting_config=fitting_config, **extra_kwargs
            )
            return data_dir, fov_data

        worker = self._start_worker(_do)
        worker.result.connect(self._on_fitting_done)
        self.fitting_panel.set_fit_busy(True)
        worker.start()

    def _on_fitting_done(self, result):
        # Record real results (already written to disk by pipeline.fit()) and
        # advance app state before touching the preview figures below -- a
        # rendering failure (e.g. a malformed companion file glob-matched
        # alongside the real results, as ground_truth.h5 briefly was) must
        # never prevent _fov_data/_fitted_data_dir/state from being set, or
        # every downstream action (channel unmixing, clustering, ...) that
        # gates on them silently no-ops with no error and nothing in the log.
        # Rendering itself is deferred to _refresh_render_figure's own aux
        # worker, matching the pattern _on_clustering_done already uses.
        data_dir, fov_data = result
        self._fitted_data_dir = data_dir
        self._fov_data = fov_data
        self._undrifted_locs = None
        self._sm_db = None
        self._sf_db = None
        self.fitting_panel.set_fit_busy(False)
        self.fitting_panel.set_clear_enabled(True)
        self.drift_panel.set_clear_enabled(False)
        self.postproc_panel.set_clear_enabled(False)
        self.postproc_panel.clear_result()
        self.channel_unmixing_panel.set_available_channels([])
        self._invalidate_nile_red()
        self.progress_widget.update(1.0, "Fitting complete")
        self._update_state(AppState.FITTED)
        self.results_panel.set_fov_count(len(fov_data))
        if not fov_data:
            return
        n = len(fov_data)
        first_df, first_tif = fov_data[0]
        title = f"FOV 1 / {n}" if n > 1 else "Localisations"
        all_locs = pd.concat([df for df, _ in fov_data], ignore_index=True)
        self._refresh_render_figure(first_df, title=title, stats_locs=all_locs, tif_path=first_tif)

    def _on_clear_fitting(self):
        """Discard fitting results (and everything downstream that depended
        on them) so the user can re-fit with different parameters without
        re-selecting the data folder."""
        self._fitted_data_dir = None
        self._fov_data = []
        self._undrifted_locs = None
        self._sm_db = None
        self._sf_db = None
        self.fitting_panel.set_clear_enabled(False)
        self.drift_panel.set_clear_enabled(False)
        self.postproc_panel.set_clear_enabled(False)
        self.postproc_panel.clear_result()
        self.channel_unmixing_panel.set_available_channels([])
        self._invalidate_nile_red()
        self.results_panel.set_fov_count(1)
        self.results_panel.clear_localisations_figure()
        self.results_panel.clear_stats_figure()
        self.results_panel.clear_drift_figure()
        self.results_panel.clear_unmixing_figure()
        self.results_panel.clear_frc_figure()
        self.progress_widget.reset()
        self._update_state(AppState.CALIBRATED if self.pipeline is not None else AppState.IDLE)
        self.log_widget.append("Cleared fitting results — ready to fit again.")

    # ------------------------------------------------------------------
    # FOV navigation
    # ------------------------------------------------------------------

    def _on_fov_requested(self, idx: int):
        if not self._fov_data or idx >= len(self._fov_data):
            return
        if self._aux_worker is not None and self._aux_worker.isRunning():
            return
        locs_df, tif_path = self._fov_data[idx]
        n = len(self._fov_data)
        title = f"FOV {idx + 1} / {n}"

        def _do():
            return self._make_locs_render_figure(locs_df, tif_path=tif_path, title=title)

        aux = AnalysisWorker(_do)
        self._aux_worker = aux
        aux.result.connect(
            lambda fig: self.results_panel.set_localisations_figure(fig) if fig is not None else None
        )
        aux.error.connect(
            lambda msg: self.log_widget.append(f"Warning: FOV render failed:\n{msg}")
        )
        aux.finished.connect(lambda w=aux: self._on_aux_worker_finished(w))
        aux.start()

    # ------------------------------------------------------------------
    def _on_aux_worker_finished(self, w: AnalysisWorker):
        if self._aux_worker is w:
            self._aux_worker = None

    def _on_stats_refresh(self, photon_range: tuple):
        if self._fitted_data_dir is None or self.pipeline is None:
            return
        if self._aux_worker is not None and self._aux_worker.isRunning():
            return

        def _do():
            locs = self.pipeline.load_localisations(self._fitted_data_dir)
            if locs.empty:
                return None
            return self._make_stats_figure(locs, photon_range=photon_range)

        aux = AnalysisWorker(_do)
        self._aux_worker = aux
        aux.result.connect(lambda fig: self.results_panel.set_stats_figure(fig) if fig is not None else None)
        aux.error.connect(lambda msg: self.log_widget.append(f"Stats refresh failed:\n{msg}"))
        aux.finished.connect(lambda w=aux: self._on_aux_worker_finished(w))
        aux.start()

    # ------------------------------------------------------------------
    # Load existing H5 (skip fitting)
    # ------------------------------------------------------------------

    def _on_load_locs(self, h5_path: str):
        if self._worker_running():
            return

        h5 = Path(h5_path)
        folder = h5.parent
        filename = h5.name
        photon_range = self.fitting_panel.photon_range

        if self.pipeline is None:
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).parent.parent))
            from pyS3M.AnalysisPipeline import AnalysisPipeline
            from pyS3M.Constants import AnalysisConfig
            self.pipeline = AnalysisPipeline(
                config=AnalysisConfig(display=False)
            )
            logger.info("Created default pipeline for H5 load (no calibration)")

        def _do():
            locs = self.pipeline.load_localisations(folder, pattern=filename)
            if locs.empty:
                raise ValueError(f"No localisations found in {h5}")
            locs_fig = self._make_locs_render_figure(locs, data_dir=str(folder), title="Localisations (loaded)")
            stats = self._make_stats_figure(locs, photon_range=photon_range)
            return str(folder), locs, locs_fig, stats

        aux = AnalysisWorker(_do)
        self._aux_worker = aux
        aux.result.connect(self._on_h5_loaded)
        aux.error.connect(lambda msg: self.log_widget.append(f"Load H5 failed: {msg[:300]}"))
        aux.finished.connect(lambda w=aux: self._on_aux_worker_finished(w))
        aux.start()

    def _on_h5_loaded(self, result):
        folder, locs, scatter_fig, stats_fig = result
        self._fitted_data_dir = folder
        # Store the loaded table itself, not just its figures -- state advances
        # to FITTED below, which enables Channel Unmixing/undrift/FRC, but
        # _unmixing_source_locs() (and undrift/FRC's own fallback reloads) need
        # actual data to act on. Leaving _fov_data empty here previously meant
        # those buttons looked enabled but had nothing to run against, so
        # clicking them silently no-op'd -- no error, no log, nothing.
        self._fov_data = [(locs, None)]
        self._undrifted_locs = None
        self._sm_db = None
        self._sf_db = None
        self.fitting_panel.set_clear_enabled(True)
        self.drift_panel.set_clear_enabled(False)
        self.postproc_panel.set_clear_enabled(False)
        self.postproc_panel.clear_result()
        self.channel_unmixing_panel.set_available_channels([])
        self._invalidate_nile_red()
        self.results_panel.set_fov_count(1)
        self.results_panel.set_localisations_figure(scatter_fig)
        self.results_panel.set_stats_figure(stats_fig)
        self.progress_widget.update(1.0, "Localisations loaded")
        self._update_state(AppState.FITTED)

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def _on_run_clustering(self, criteria, clustering_config):
        if self._worker_running() or self.pipeline is None or self._fitted_data_dir is None:
            return

        def _do():
            # Prefer the drift-corrected localisations if undrift has been run
            # (they live in memory only, not reloaded from _fitted_data_dir).
            if self._undrifted_locs is not None:
                locs = self._undrifted_locs
            elif len(self._fov_data) > 1:
                # Multiple independent FOVs: xc/yc are local pixel coordinates
                # within each FOV's own field of view, so clustering across a
                # flat concatenation of all FOVs risks merging unrelated
                # molecules that happen to share similar local coordinates in
                # different FOVs. Cluster each FOV separately instead, then
                # concatenate with fov_index/fov_name columns and a running
                # molecular_index offset to keep indices globally unique.
                sm_parts, sf_parts = [], []
                molecular_index_offset = 0
                for fov_idx, (fov_locs, tif_path) in enumerate(self._fov_data):
                    fov_name = Path(tif_path).stem if tif_path else f"fov_{fov_idx}"
                    sm_fov, sf_fov = self.pipeline.filter_and_cluster(
                        fov_locs, criteria=criteria, clustering_config=clustering_config
                    )
                    if sm_fov.empty:
                        continue
                    sm_fov = sm_fov.assign(
                        fov_index=fov_idx, fov_name=fov_name,
                        molecular_index=sm_fov["molecular_index"] + molecular_index_offset,
                    )
                    sf_fov = sf_fov.assign(
                        fov_index=fov_idx, fov_name=fov_name,
                        molecular_index=sf_fov["molecular_index"] + molecular_index_offset,
                    )
                    sm_parts.append(sm_fov)
                    sf_parts.append(sf_fov)
                    molecular_index_offset = int(sm_fov["molecular_index"].max()) + 1
                if not sm_parts:
                    return pd.DataFrame(), pd.DataFrame()
                return (
                    pd.concat(sm_parts, ignore_index=True),
                    pd.concat(sf_parts, ignore_index=True),
                )
            else:
                locs = self.pipeline.load_localisations(self._fitted_data_dir)
            return self.pipeline.filter_and_cluster(
                locs, criteria=criteria, clustering_config=clustering_config
            )

        worker = self._start_worker(_do)
        worker.result.connect(self._on_clustering_done)
        self.postproc_panel.set_busy(True)
        worker.start()

    def _on_clustering_done(self, result):
        sm_db, sf_db = result
        self._sm_db = sm_db
        self._sf_db = sf_db
        self.postproc_panel.set_busy(False)
        self.postproc_panel.show_result(len(sm_db), len(sf_db))
        self.postproc_panel.set_clear_enabled(True)
        self._invalidate_nile_red()
        self.progress_widget.update(1.0, "Clustering complete")
        self._update_state(AppState.CLUSTERED)
        if not sm_db.empty:
            self._refresh_render_figure(sm_db, title="Single molecules", stats_locs=sf_db, stats_sm=sm_db)

    def _on_clear_clustering(self):
        """Discard clustering results (and channel-unmixing/per-channel-FRC
        results derived from them) so the user can re-cluster with different
        parameters — also re-enables the Drift Correction panel, which is
        disabled once the app state moves past "fitted"/"undrifted"."""
        self._sm_db = None
        self._sf_db = None
        self.postproc_panel.clear_result()
        self.postproc_panel.set_clear_enabled(False)
        self.channel_unmixing_panel.set_available_channels([])
        self._invalidate_nile_red()
        self.results_panel.clear_unmixing_figure()
        self.results_panel.clear_frc_figure()
        self.progress_widget.reset()
        new_state = AppState.UNDRIFTED if self._undrifted_locs is not None else AppState.FITTED
        self._update_state(new_state)
        self.log_widget.append("Cleared clustering results — ready to cluster again.")

    def _refresh_render_figure(
        self, locs, title: str = "Localisations", stats_locs=None, stats_sm=None,
        tif_path: str | None = None,
    ):
        """Launch an aux worker to build and display the render (and optionally stats) figure.

        *tif_path*, when given, is used for the MIP instead of globbing
        ``self._fitted_data_dir`` — needed right after fitting, where *locs* is
        one specific FOV's table and its matching TIFF may not be the first
        one found by a folder glob.
        """
        data_dir = self._fitted_data_dir
        photon_range = self.fitting_panel.photon_range

        def _do():
            locs_fig = self._make_locs_render_figure(
                locs, data_dir=data_dir, tif_path=tif_path, title=title
            )
            stats_fig = (
                self._make_stats_figure(stats_locs, photon_range=photon_range, sm_locs=stats_sm)
                if stats_locs is not None and not stats_locs.empty
                else None
            )
            return locs_fig, stats_fig

        def _on_result(result):
            locs_fig, stats_fig = result
            if locs_fig is not None:
                self.results_panel.set_localisations_figure(locs_fig)
            if stats_fig is not None:
                self.results_panel.set_stats_figure(stats_fig)

        aux = AnalysisWorker(_do)
        self._aux_worker = aux
        aux.result.connect(_on_result)
        aux.error.connect(
            lambda msg: self.log_widget.append(f"Warning: render failed:\n{msg}")
        )
        aux.finished.connect(lambda w=aux: self._on_aux_worker_finished(w))
        aux.start()

    def _on_save_clustering(self):
        if self._sm_db is None or self._sm_db.empty:
            QMessageBox.warning(self, "No data", "No clustering results to save.")
            return

        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Save clustering results", "", "HDF5 files (*.h5)"
        )
        if not path:
            return
        if not path.endswith(".h5"):
            path += ".h5"

        try:
            self._sm_db.to_hdf(path, key="sm_db", mode="w")
            self._sf_db.to_hdf(path, key="sf_db", mode="a")
            self.log_widget.append(
                f"Saved {len(self._sm_db)} molecules / {len(self._sf_db)} localisations → {path}"
            )
            self.progress_widget.update(1.0, "Results saved")
        except Exception:
            import traceback as _tb
            self.log_widget.append(f"Save failed:\n{_tb.format_exc()}")
            QMessageBox.critical(self, "Save failed", "Could not write HDF5 file — see log.")

    def _fov_dimensions(self, locs) -> tuple[int, int]:
        """(height, width) in camera pixels, for undrift/FRC binning.

        Prefers the loaded calibration's gain_map shape (the true sensor
        extent, guaranteed to cover every localisation). Falls back to the
        localisations' own coordinate extent when no calibration is loaded --
        gain_map is None after "Load Localisations" on an existing .h5
        without recalibrating first (see _on_load_locs), which previously
        crashed undrift/FRC/per-channel-FRC with an unhelpful AttributeError.
        """
        if self.pipeline.gain_map is not None:
            return self.pipeline.gain_map.shape[:2]
        height = int(np.ceil(locs["yc"].max())) + 1
        width = int(np.ceil(locs["xc"].max())) + 1
        return height, width

    # ------------------------------------------------------------------
    # Drift correction
    # ------------------------------------------------------------------

    def _on_undrift(self, segmentation: int, intersect_d_nm: float, roi_r_nm: float):
        if self._worker_running() or self.pipeline is None or self._fitted_data_dir is None:
            return

        def _do():
            # Always reload the raw (not previously undrifted) localisations,
            # so re-running undrift never compounds a prior correction.
            locs_df = self.pipeline.load_localisations(self._fitted_data_dir)
            if locs_df.empty:
                raise ValueError("No localisations to undrift.")

            pixel_size_nm = self.pipeline.pixel_size * 1000.0
            height, width = self._fov_dimensions(locs_df)
            n_frames = int(locs_df["frame"].max()) + 1

            info = [{
                "Width": int(width),
                "Height": int(height),
                "Frames": n_frames,
                "Pixelsize": pixel_size_nm,
            }]

            locs_rec = locs_df.to_records(index=False)
            corrected_locs, drift_result = self.pipeline.undrift(
                locs_rec, info, method="aim",
                segmentation=segmentation,
                intersect_d=intersect_d_nm / pixel_size_nm,
                roi_r=roi_r_nm / pixel_size_nm,
                pixel_size_nm=pixel_size_nm,
            )
            corrected_df = pd.DataFrame(corrected_locs)

            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).parent.parent))
            from pyS3M.IOFunctions import IO_Functions
            out_path = str(Path(self._fitted_data_dir) / "undrifted_locs.h5")
            IO_Functions().write_h5_database(corrected_df, out_path)

            drift_fig = self._make_drift_figure(drift_result, pixel_size_nm)
            return corrected_df, drift_fig

        worker = self._start_worker(_do)
        worker.result.connect(self._on_undrift_done)
        self.drift_panel.set_busy(True)
        worker.start()

    def _on_undrift_done(self, result):
        corrected_df, drift_fig = result
        self._undrifted_locs = corrected_df
        # Any prior clustering was run against the pre-undrift (or a
        # differently-parameterised undrift) locs — no longer valid.
        self._sm_db = None
        self._sf_db = None
        self.drift_panel.set_busy(False)
        self.drift_panel.set_clear_enabled(True)
        self.postproc_panel.clear_result()
        self.postproc_panel.set_clear_enabled(False)
        self.channel_unmixing_panel.set_available_channels([])
        self._invalidate_nile_red()
        self.results_panel.clear_unmixing_figure()
        self.results_panel.clear_frc_figure()
        self.progress_widget.update(1.0, "Undrift complete")
        self._update_state(AppState.UNDRIFTED)
        if drift_fig is not None:
            self.results_panel.set_drift_figure(drift_fig)
        self._refresh_render_figure(
            corrected_df, title="Undrifted localisations", stats_locs=corrected_df,
        )

    def _on_clear_drift(self):
        """Discard drift-correction results (and clustering/unmixing/FRC
        results derived from them) so the user can re-run undrift with
        different segmentation/intersect_d/roi_r parameters."""
        # Undrift also writes undrifted_locs.h5 next to the fitted data
        # (see the run-undrift worker below) so it survives a session
        # restart. Leaving it on disk after clearing means the *next*
        # load_localisations(self._fitted_data_dir) call -- e.g. the one
        # undrift itself does to fetch its input -- globs it back in
        # alongside the original fit output, silently duplicating locs.
        if self._fitted_data_dir is not None:
            stale_path = Path(self._fitted_data_dir) / "undrifted_locs.h5"
            try:
                stale_path.unlink(missing_ok=True)
            except OSError as e:
                logger.warning("Could not remove stale %s: %s", stale_path, e)
        self._undrifted_locs = None
        self._sm_db = None
        self._sf_db = None
        self.drift_panel.set_clear_enabled(False)
        self.postproc_panel.clear_result()
        self.postproc_panel.set_clear_enabled(False)
        self.channel_unmixing_panel.set_available_channels([])
        self._invalidate_nile_red()
        self.results_panel.clear_drift_figure()
        self.results_panel.clear_unmixing_figure()
        self.results_panel.clear_frc_figure()
        self.progress_widget.reset()
        self._update_state(AppState.FITTED)
        self.log_widget.append("Cleared drift-correction results — ready to undrift again.")

    def _make_drift_figure(self, drift_result, pixel_size_nm: float) -> Figure:
        fig = Figure(figsize=(7, 4), dpi=100, layout="constrained")
        ax = fig.add_subplot(111)
        frames = np.arange(len(drift_result.drift_x))
        ax.plot(frames, drift_result.drift_x * pixel_size_nm, label="x", color="tab:blue", lw=1.0)
        ax.plot(frames, drift_result.drift_y * pixel_size_nm, label="y", color="tab:orange", lw=1.0)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Drift (nm)")
        ax.set_title(f"Drift trace ({drift_result.method_used.value})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return fig

    # ------------------------------------------------------------------
    # FRC (Fourier Ring Correlation) resolution
    # ------------------------------------------------------------------

    def _on_run_frc(self, zoom: float, n_blocks: int, reps: int):
        if self._worker_running() or self.pipeline is None or self._fitted_data_dir is None:
            return

        def _do():
            # Prefer the drift-corrected localisations if undrift has been run
            # (in-memory only), same preference order as clustering.
            locs = (
                self._undrifted_locs if self._undrifted_locs is not None
                else self.pipeline.load_localisations(self._fitted_data_dir)
            )
            if locs.empty:
                raise ValueError("No localisations available for FRC.")
            pixel_size_nm = self.pipeline.pixel_size * 1000.0
            height, width = self._fov_dimensions(locs)
            return self._make_frc_figure(locs, width, height, pixel_size_nm, zoom, n_blocks, reps)

        worker = self._start_worker(_do)
        worker.result.connect(self._on_frc_done)
        self.frc_panel.set_busy(True)
        worker.start()

    def _on_frc_done(self, fig):
        self.frc_panel.set_busy(False)
        self.progress_widget.update(1.0, "FRC complete")
        if fig is not None:
            self.results_panel.set_frc_figure(fig)

    def _make_frc_figure(
        self,
        locs,
        width: int,
        height: int,
        pixel_size_nm: float,
        zoom: float,
        n_blocks: int,
        reps: int,
    ) -> Figure:
        """Spatial resolution (FIRE) over all current localisations. Splitting by
        dye/channel is deliberately not done here — that requires a categorical
        `channel` column, which only exists once Channel Unmixing has run; when
        that panel is built, FRC should be re-runnable per selected channel from
        there rather than by any spectral-fraction heuristic in this panel."""
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent.parent))
        import pyS3M.FRCFunctions as FRCFunctions

        fig = Figure(figsize=(6, 5), dpi=100, layout="constrained")
        ax = fig.add_subplot(111)

        res_nm, frc_curve, _, _ = FRCFunctions.fire(
            locs, nx=width, ny=height, zoom=zoom,
            n_blocks=n_blocks, reps=reps, pixel_size_nm=pixel_size_nm,
        )
        res_label = f"{res_nm:.0f} nm" if np.isfinite(res_nm) else "unresolved"

        if len(frc_curve) > 0:
            sz = max(int(width * zoom), int(height * zoom))
            q_per_nm = (np.arange(len(frc_curve)) / sz) / (pixel_size_nm / zoom)
            ax.plot(
                q_per_nm, frc_curve, color="black", lw=1.2,
                label=f"FIRE: {res_label}  (n={len(locs):,})",
            )

        ax.axhline(1.0 / 7.0, color="gray", ls="--", lw=0.8, label="1/7 threshold")
        ax.set_xlabel("Spatial frequency (nm⁻¹)")
        ax.set_ylabel("FRC")
        ax.set_ylim(-0.2, 1.05)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_title(f"Fourier Ring Correlation — resolution: {res_label}")
        ax.grid(True, alpha=0.3)
        return fig

    # ------------------------------------------------------------------
    # Channel unmixing
    # ------------------------------------------------------------------

    def _unmixing_source_locs(self) -> pd.DataFrame | None:
        """Best-available localisation table for channel unmixing.

        Prefers the post-clustering per-molecule table (sm_db) — its averaged
        A_R/A_G values give much tighter spectral clusters than any
        single-frame row — but falls back to undrifted or raw per-FOV fitted
        locs so a user can try unmixing as soon as there's any analysed data,
        not only once clustering has produced sm_db.
        """
        if self._sm_db is not None and not self._sm_db.empty:
            return self._sm_db
        if self._undrifted_locs is not None and not self._undrifted_locs.empty:
            return self._undrifted_locs
        if self._fov_data:
            return pd.concat([locs for locs, _ in self._fov_data], ignore_index=True)
        return None

    def _on_channel_unmixing(
        self, n_channels: int, channels_to_use: list, confidence_threshold: float,
        outlier_rejection: str,
    ):
        if self._worker_running() or self.pipeline is None:
            return
        loc_data = self._unmixing_source_locs()
        if loc_data is None or loc_data.empty:
            return  # button is state-gated, but guard anyway
        # sm_db identity check (not equality) at completion time to decide
        # whether to write results back into it, since the run's own input
        # will otherwise be indistinguishable from an sm_db-derived result.
        ran_on_sm_db = loc_data is self._sm_db

        def _do():
            assigned, metadata = self.pipeline.sm.unmix_channels(
                loc_data,
                n_channels=n_channels,
                channels_to_use=channels_to_use,
                confidence_threshold=confidence_threshold,
                outlier_rejection=outlier_rejection,
                verbose=True,
                plot_results=False,
            )
            fig = self._make_channel_unmixing_figure(assigned, channels_to_use, metadata)
            return assigned, fig, ran_on_sm_db

        worker = self._start_worker(_do)
        worker.result.connect(self._on_channel_unmixing_done)
        self.channel_unmixing_panel.set_busy(True)
        worker.start()

    def _on_channel_unmixing_done(self, result):
        assigned, fig, ran_on_sm_db = result
        self.channel_unmixing_panel.set_busy(False)
        self.progress_widget.update(1.0, "Channel unmixing complete")
        if ran_on_sm_db:
            self._sm_db = assigned  # adds 'channel'/'channel_confidence'/... columns in place
            channels = sorted(int(c) for c in assigned["channel"].unique() if c >= 0)
            self.channel_unmixing_panel.set_available_channels(channels)
        else:
            # Pre-clustering preview: show the assignment figure but don't
            # promote it into sm_db -- Per-Channel FRC and downstream steps
            # still need the real clustered table.
            self.channel_unmixing_panel.set_available_channels([])
        if fig is not None:
            self.results_panel.set_unmixing_figure(fig)

    def _make_channel_unmixing_figure(
        self, assigned: pd.DataFrame, channels_to_use: list, metadata: dict,
    ) -> Figure:
        """Two panels: spectral-feature space coloured by assigned channel, and
        the spatial (xc, yc) scatter in the same per-channel colours — the
        actual multi-colour overlay this analysis is for. Deliberately a plain
        coloured scatter, not a composite-colormap render (matching
        _make_frc_figure/_make_drift_figure's precedent of a direct, simple
        embeddable figure over reusing the backend's own notebook-oriented
        diagnostic plotting methods)."""
        fig = Figure(figsize=(9, 4.5), dpi=100, layout="constrained")
        ax_spec, ax_spatial = fig.subplots(1, 2)

        n_channels = int(assigned["channel"].max()) + 1 if (assigned["channel"] >= 0).any() else 0
        colors = ["tab:red", "tab:green", "tab:blue", "tab:purple", "tab:orange"]

        unassigned = assigned["channel"] < 0
        ax_spec.scatter(
            assigned.loc[unassigned, channels_to_use[0]],
            assigned.loc[unassigned, channels_to_use[1]],
            s=4, alpha=0.3, color="lightgray", label="unassigned", rasterized=True,
        )
        ax_spatial.scatter(
            assigned.loc[unassigned, "xc"], assigned.loc[unassigned, "yc"],
            s=2, alpha=0.2, color="lightgray", rasterized=True,
        )
        for k in range(n_channels):
            mask = assigned["channel"] == k
            n_k = int(mask.sum())
            c = colors[k % len(colors)]
            ax_spec.scatter(
                assigned.loc[mask, channels_to_use[0]], assigned.loc[mask, channels_to_use[1]],
                s=4, alpha=0.5, color=c, label=f"channel {k} (n={n_k:,})", rasterized=True,
            )
            ax_spatial.scatter(
                assigned.loc[mask, "xc"], assigned.loc[mask, "yc"],
                s=2, alpha=0.6, color=c, rasterized=True,
            )

        ax_spec.set_xlabel(channels_to_use[0])
        ax_spec.set_ylabel(channels_to_use[1] if len(channels_to_use) > 1 else "density")
        ax_spec.set_title("Spectral assignment")
        ax_spec.legend(fontsize=7, loc="best")
        ax_spec.grid(True, alpha=0.3)

        ax_spatial.set_xlabel("xc (px)")
        ax_spatial.set_ylabel("yc (px)")
        ax_spatial.set_aspect("equal")
        ax_spatial.set_title("Spatial overlay")
        ax_spatial.invert_yaxis()

        return fig

    # ------------------------------------------------------------------
    # FRC per channel (uses sm_db's `channel` column from Channel Unmixing)
    # ------------------------------------------------------------------

    def _on_run_frc_per_channel(self, channels: list, zoom: float, n_blocks: int, reps: int):
        if self._worker_running() or self.pipeline is None:
            return
        if self._sm_db is None or self._sm_db.empty or "channel" not in self._sm_db.columns:
            return
        if not channels:
            return

        def _do():
            pixel_size_nm = self.pipeline.pixel_size * 1000.0
            height, width = self._fov_dimensions(self._sm_db)
            return self._make_frc_per_channel_figure(
                self._sm_db, channels, width, height, pixel_size_nm, zoom, n_blocks, reps,
            )

        worker = self._start_worker(_do)
        worker.result.connect(self._on_frc_per_channel_done)
        self.channel_unmixing_panel.set_busy(True)
        worker.start()

    def _on_frc_per_channel_done(self, fig):
        self.channel_unmixing_panel.set_busy(False)
        self.progress_widget.update(1.0, "Per-channel FRC complete")
        if fig is not None:
            self.results_panel.set_frc_figure(fig)

    def _make_frc_per_channel_figure(
        self,
        sm_db: pd.DataFrame,
        channels: list,
        width: int,
        height: int,
        pixel_size_nm: float,
        zoom: float,
        n_blocks: int,
        reps: int,
    ) -> Figure:
        """One FRC curve per selected channel, overlaid on shared axes — the
        per-channel counterpart to _make_frc_figure's single combined curve.
        Runs on sm_db (per-molecule, post-clustering) rather than raw locs
        since `channel` only exists there. Colours match
        _make_channel_unmixing_figure's channel colouring."""
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent.parent))
        import pyS3M.FRCFunctions as FRCFunctions

        fig = Figure(figsize=(6, 5), dpi=100, layout="constrained")
        ax = fig.add_subplot(111)
        colors = ["tab:red", "tab:green", "tab:blue", "tab:purple", "tab:orange"]
        sz = max(int(width * zoom), int(height * zoom))

        for ch in channels:
            locs_ch = sm_db[sm_db["channel"] == ch]
            c = colors[int(ch) % len(colors)]
            if locs_ch.empty:
                continue
            res_nm, frc_curve, _, _ = FRCFunctions.fire(
                locs_ch, nx=width, ny=height, zoom=zoom,
                n_blocks=n_blocks, reps=reps, pixel_size_nm=pixel_size_nm,
            )
            res_label = f"{res_nm:.0f} nm" if np.isfinite(res_nm) else "unresolved"
            if len(frc_curve) > 0:
                q_per_nm = (np.arange(len(frc_curve)) / sz) / (pixel_size_nm / zoom)
                ax.plot(
                    q_per_nm, frc_curve, color=c, lw=1.2,
                    label=f"channel {ch}: {res_label}  (n={len(locs_ch):,})",
                )

        ax.axhline(1.0 / 7.0, color="gray", ls="--", lw=0.8, label="1/7 threshold")
        ax.set_xlabel("Spatial frequency (nm⁻¹)")
        ax.set_ylabel("FRC")
        ax.set_ylim(-0.2, 1.05)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_title("Fourier Ring Correlation — per channel")
        ax.grid(True, alpha=0.3)
        return fig

    # ------------------------------------------------------------------
    # Nile Red — pixelated wavelength fit (uses sf_db, post-clustering)
    # ------------------------------------------------------------------

    def _on_run_nile_red(
        self, filter_ids: list, wl_min: float, wl_max: float, na: float,
        pixel_size_nm: float, min_locs: int,
    ):
        if self._worker_running() or self.pipeline is None:
            return
        locs = self._nile_red_input_df if self._nile_red_input_df is not None else self._sf_db
        if locs is None or locs.empty:
            return  # button is state-gated to "clustered" (or a loaded file), but guard anyway
        aggregate_id_column = "molecular_index" if "molecular_index" in locs.columns else None

        def _do():
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).parent.parent))
            import tempfile
            import os
            from pyS3M.IOFunctions import IO_Functions
            import pyS3M.SpectralFunctions as SpectralFunctions

            fd, tmp_path = tempfile.mkstemp(suffix=".h5")
            os.close(fd)
            try:
                IO_Functions().write_h5_database(
                    locs, tmp_path, normalise_photons=False, verbose=False,
                )
                R_qe, G_qe, B_qe, wl = SpectralFunctions.Spectral_Funcs(
                    camera=self.pipeline.camera
                ).getpixelefficiency()
                camera_parameters = {
                    "pixel_QYs": np.vstack([B_qe, G_qe, R_qe]), "wavelength": wl,
                }
                df, grid_info = self.pipeline.nile_red.fit_wavelengths_pixelated(
                    tmp_path, filter_ids, camera_parameters,
                    pixel_size_nm=pixel_size_nm,
                    wavelength_bounds=(wl_min, wl_max),
                    NA=na,
                    min_localisations=min_locs,
                    aggregate_id_column=aggregate_id_column,
                    verbose=True,
                    return_grid=True,
                )
            finally:
                os.unlink(tmp_path)
            fig = self._make_nile_red_figure(grid_info, wl_min, wl_max)
            return df, grid_info, fig

        worker = self._start_worker(_do)
        worker.result.connect(self._on_nile_red_done)
        self.nile_red_panel.set_busy(True)
        worker.start()

    def _on_nile_red_done(self, result):
        db, grid_info, fig = result
        self._nile_red_db = db
        self._nile_red_grid = grid_info
        self.nile_red_panel.set_busy(False)
        self.nile_red_panel.set_clear_enabled(True)
        self.progress_widget.update(
            1.0,
            f"Nile Red fit complete ({grid_info['n_pixels_fitted']} pixels fitted, "
            f"{grid_info['n_pixels_skipped']} skipped)",
        )
        if fig is not None:
            self.results_panel.set_nile_red_figure(fig)

    def _on_clear_nile_red(self):
        self._nile_red_db = None
        self._nile_red_grid = None
        self._nile_red_input_df = None
        self.nile_red_panel.set_clear_enabled(False)
        self.results_panel.clear_nile_red_figure()
        self.log_widget.append("Cleared Nile Red results — ready to fit again.")

    def _on_load_nile_red_locs(self, h5_path: str):
        if self._aux_worker is not None and self._aux_worker.isRunning():
            return

        def _do():
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).parent.parent))
            from pyS3M.IOFunctions import IO_Functions
            df = IO_Functions().read_h5_database(h5_path)
            required = ["xc", "yc", "A_R", "A_G", "A_B", "s_x", "s_y",
                        "A_R_err", "A_G_err", "A_B_err", "s_x_err", "s_y_err"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError(
                    f"{h5_path} is missing required columns: {missing}\n"
                    f"Available columns: {list(df.columns)}"
                )
            return df

        aux = AnalysisWorker(_do)
        self._aux_worker = aux
        aux.result.connect(lambda df: self._on_nile_red_locs_loaded(df, h5_path))
        aux.error.connect(lambda msg: self.log_widget.append(f"Load H5 failed: {msg[:300]}"))
        aux.finished.connect(lambda w=aux: self._on_aux_worker_finished(w))
        aux.start()

    def _on_nile_red_locs_loaded(self, df: pd.DataFrame, h5_path: str):
        self._nile_red_input_df = df
        self.nile_red_panel.set_loaded(h5_path, len(df))
        self.log_widget.append(f"Loaded {len(df):,} localisations from {h5_path} for Nile Red fitting.")

    def _make_nile_red_figure(self, grid_info: dict, wl_min: float, wl_max: float) -> Figure:
        """Left: histogram of the fitted per-pixel wavelengths. Right: the
        pixelated wavelength map itself (grid_info['wl_grid']) rendered
        directly as an image — the real spatial output of the pixelated fit,
        and the right choice for performance too: imshow's cost is fixed by
        the grid size (bounded by the field of view), not by how many
        localisations fed into it, unlike a per-localisation scatter which
        gets slower and heavier the denser the data (exactly the dense,
        overlapping-coverage regime this panel targets).

        Colour range is the fitted values' own 1st-99th percentile, not the
        full fit search range (wl_min/wl_max, e.g. 500-750 nm) — real fitted
        wavelengths usually span a much narrower band, and stretching the
        colour scale across the whole search range would wash out any real
        spatial contrast. Unfit grid pixels (NaN) render as black — set via
        a masked array + cmap.set_bad, not the default (transparent, reads
        as white against this app's light background)."""
        wl_grid = grid_info["wl_grid"]
        pixel_size_nm = grid_info["pixel_size_nm"]
        x0, y0 = grid_info["origin_nm"]
        ny, nx = grid_info["grid_shape"]

        fig = Figure(figsize=(9, 4.5), dpi=100, layout="constrained")
        ax_hist, ax_map = fig.subplots(1, 2)

        valid = wl_grid[~np.isnan(wl_grid)]
        vmin, vmax = (
            tuple(np.percentile(valid, [1, 99])) if valid.size > 0 else (wl_min, wl_max)
        )
        if vmin == vmax:
            vmin, vmax = wl_min, wl_max

        ax_hist.hist(valid, bins=40, range=(vmin, vmax), color="tab:purple", alpha=0.8)
        ax_hist.set_xlabel("Wavelength (nm)")
        ax_hist.set_ylabel("Pixel count")
        ax_hist.set_title(
            f"Wavelength distribution\n"
            f"({grid_info['n_pixels_fitted']} fitted, {grid_info['n_pixels_skipped']} skipped)"
        )
        ax_hist.grid(True, alpha=0.3)

        cmap = matplotlib.colormaps["nipy_spectral"].copy()
        cmap.set_bad(color="black")
        extent = (x0, x0 + nx * pixel_size_nm, y0, y0 + ny * pixel_size_nm)
        im = ax_map.imshow(
            np.ma.masked_invalid(wl_grid), cmap=cmap, vmin=vmin, vmax=vmax,
            origin="lower", extent=extent, aspect="equal",
        )
        fig.colorbar(im, ax=ax_map, label="Wavelength (nm)", fraction=0.046, pad=0.04)
        ax_map.set_xlabel("x (nm)")
        ax_map.set_ylabel("y (nm)")
        ax_map.set_title("Pixelated wavelength map")

        return fig

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _on_run_simulation(
        self,
        dye: str,
        filters: list,
        bg_photons: float,
        na: float,
        pixel_size_nm: float,
        read_noise_e: float,
        peak_qe: float,
        n_rep: int,
    ):
        if self._aux_worker is not None and self._aux_worker.isRunning():
            return

        self.simulation_panel.set_busy(True)
        self.progress_widget.update(0.0, "Building simulation…")

        def _do():
            return self._make_simulation_figure(
                dye, filters, bg_photons, na, pixel_size_nm, read_noise_e, peak_qe, n_rep
            )

        aux = AnalysisWorker(_do)
        self._aux_worker = aux
        aux.result.connect(self._on_simulation_done)
        aux.error.connect(
            lambda msg: (
                self.log_widget.append(f"Simulation failed:\n{msg}"),
                self.simulation_panel.set_busy(False),
                self.progress_widget.update(0.0, "Simulation failed"),
            )
        )
        aux.finished.connect(lambda w=aux: self._on_aux_worker_finished(w))
        aux.start()

    def _on_simulation_done(self, fig):
        self.simulation_panel.set_busy(False)
        self.progress_widget.update(1.0, "Simulation ready")
        if fig is not None:
            self.results_panel.set_simulation_figure(fig)

    def _make_simulation_figure(
        self,
        dye: str,
        filters: list,
        bg_photons: float,
        na: float,
        pixel_size_nm: float,
        read_noise_e: float,
        peak_qe: float,
        n_rep: int,
    ) -> Figure:
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.insert(0, str(_Path(__file__).parent.parent))

        import numpy as np
        import pyS3M.SpectralFunctions as SpectralFunctions
        from pyS3M.PSFFunctions import PSF_Functions

        S_F = SpectralFunctions.Spectral_Funcs()
        R_qe, G_qe, B_qe, wl = S_F.getpixelefficiency()

        # Scale Bayer QE curves so the peak across all channels equals peak_qe.
        # This changes the relative colour fractions (the filter+QE overlap per channel)
        # while keeping the peak QE at the user-specified value.
        raw_peak = max(R_qe.max(), G_qe.max(), B_qe.max())
        scale = peak_qe / raw_peak if raw_peak > 0 else 1.0
        R_sc, G_sc, B_sc = R_qe * scale, G_qe * scale, B_qe * scale

        # pixel_QYs convention: (n_channels, n_wavelengths) in BGR order
        pixel_QYs = np.vstack([B_sc, G_sc, R_sc])

        # normalized=True → per-channel colour fractions (sum to 1).
        # The Bayer-geometry weighting means the QE curves can't be summed
        # directly to get a total efficiency; instead we use peak_qe as the
        # overall detection efficiency and fracs for the colour split.
        avg_wl, fracs = S_F.get_pixel_fractions_dye_and_filters(
            dyes=[dye],
            filters=filters if filters else None,
            wavelength=wl,
            pixel_QYs=pixel_QYs,
            normalized=True,
        )
        # fracs shape (3,): [B_frac, G_frac, R_frac], sum = 1
        b_frac, g_frac, r_frac = fracs
        # Total photoelectrons = n_photons × peak_qe; split by fracs
        b_eff, g_eff, r_eff = b_frac * peak_qe, g_frac * peak_qe, r_frac * peak_qe

        # PSF sigma from dye emission wavelength (avg_wl in nm) and NA
        sigma_nm = PSF_Functions.sigma_PSF(float(avg_wl) * 1e-9, na) * 1e9  # m → nm
        sigma_px = sigma_nm / pixel_size_nm

        # Normalised Gaussian PSF kernel
        patch_size = 13
        c = patch_size // 2
        yy, xx = np.mgrid[:patch_size, :patch_size]
        psf = np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2.0 * sigma_px ** 2))
        psf /= psf.sum()

        # RGGB Bayer masks over the patch
        r_mask = (yy % 2 == 0) & (xx % 2 == 0)
        g_mask = ((yy % 2 == 0) & (xx % 2 == 1)) | ((yy % 2 == 1) & (xx % 2 == 0))
        b_mask = (yy % 2 == 1) & (xx % 2 == 1)

        rng = np.random.default_rng(42)
        photon_levels = _PHOTON_LEVELS
        n_rows = len(photon_levels)

        fig = Figure(
            figsize=(1.8 * n_rep + 1.0, 1.8 * n_rows + 0.7),
            dpi=100,
            layout="constrained",
        )
        fig.patch.set_facecolor("#1a1a1a")
        axes = fig.subplots(n_rows, n_rep, squeeze=False)

        for row_idx, n_ph in enumerate(photon_levels):
            # Generate all replicates for this photon level first so we can
            # compute a shared percentile scale across the row.
            row_patches = []
            for col_idx in range(n_rep):
                bayer = np.zeros((patch_size, patch_size))
                for mask, eff in ((r_mask, r_eff), (g_mask, g_eff), (b_mask, b_eff)):
                    n_px = int(mask.sum())
                    sig = rng.poisson(n_ph * eff * psf[mask] + bg_photons).astype(float)
                    sig += rng.normal(0.0, read_noise_e, size=n_px)
                    bayer[mask] = sig - bg_photons
                row_patches.append(bayer)

            # Per-row percentile stretch so every photon level is clearly visible
            all_vals = np.concatenate([p.ravel() for p in row_patches])
            vmin = float(np.percentile(all_vals, 0.1))
            vmax = float(np.percentile(all_vals, 99.9))
            if vmax <= vmin:
                vmax = vmin + 1.0

            for col_idx, bayer in enumerate(row_patches):
                ax = axes[row_idx, col_idx]
                ax.set_facecolor("#1a1a1a")

                ax.imshow(
                    bayer,
                    cmap="gray",
                    origin="upper",
                    interpolation="nearest",
                    aspect="equal",
                    vmin=vmin,
                    vmax=vmax,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_edgecolor("#333333")
                    spine.set_linewidth(0.5)

                if col_idx == 0:
                    ax.set_ylabel(
                        f"{n_ph:,} ph",
                        color="white",
                        fontsize=7,
                        rotation=0,
                        labelpad=34,
                        va="center",
                    )

        filter_label = ", ".join(f.split("-", 1)[-1] for f in filters) if filters else "no filter"
        fig.suptitle(
            (
                f"{dye}  |  {filter_label}  |  "
                f"λ={float(avg_wl):.0f} nm  σ={sigma_nm:.0f} nm  "
                f"QE={peak_qe:.2f}  BG={bg_photons:.0f} ph/px  "
                f"RN={read_noise_e:.1f} e⁻"
            ),
            color="white",
            fontsize=8,
        )
        return fig

    # ------------------------------------------------------------------
    # Image-driven STORM/PAINT pattern simulation
    # ------------------------------------------------------------------

    def _on_run_pattern_simulation(
        self,
        image_path: str,
        colour_to_dye: dict,
        n_frames: int,
        density_per_um2: float,
        modality: str,
        on_rate: float,
        off_rate: float,
        bleach_after_cycles: int,
        photon_min: float,
        photon_max: float,
        background_photons: float,
        na: float,
        output_dir: str,
        run_name: str,
    ):
        if self._aux_worker is not None and self._aux_worker.isRunning():
            return
        if self.pipeline is None or self.pipeline.gain_map is None:
            return
        if not image_path or not colour_to_dye or not output_dir or not run_name:
            return

        self.simulation_panel.set_busy(True)
        self.progress_widget.update(0.0, "Simulating acquisition…")

        def _do():
            return self._simulate_pattern_acquisition(
                image_path, colour_to_dye, n_frames, density_per_um2, modality,
                on_rate, off_rate, bleach_after_cycles, photon_min, photon_max,
                background_photons, na, output_dir, run_name,
            )

        aux = AnalysisWorker(_do)
        self._aux_worker = aux
        aux.result.connect(self._on_pattern_simulation_done)
        aux.error.connect(
            lambda msg: (
                self.log_widget.append(f"Pattern simulation failed:\n{msg}"),
                self.simulation_panel.set_busy(False),
                self.progress_widget.update(0.0, "Pattern simulation failed"),
            )
        )
        aux.finished.connect(lambda w=aux: self._on_aux_worker_finished(w))
        aux.start()

    def _on_pattern_simulation_done(self, result):
        fig, out_dir, avg_emission_wavelength_nm = result
        self.simulation_panel.set_busy(False)
        self.simulation_panel.set_clear_enabled(True)
        self.progress_widget.update(1.0, "Pattern simulation complete")
        peak_wavelength_um = avg_emission_wavelength_nm / 1000.0
        self.log_widget.append(
            f"Synthetic acquisition written to {out_dir}\n"
            f"  Recommended Peak λ for fitting this: {peak_wavelength_um:.3f} µm "
            f"(the shared PSF wavelength this simulation actually used)"
        )
        if fig is not None:
            self.results_panel.set_simulation_figure(fig)

    def _on_clear_pattern_simulation(self):
        """Clear the simulation preview so the user can try again with
        different parameters. The written acquisition on disk is untouched —
        this only clears the in-GUI preview/state."""
        self.simulation_panel.set_clear_enabled(False)
        self.results_panel.clear_simulation_figure()
        self.log_widget.append("Cleared simulation preview.")

    def _simulate_pattern_acquisition(
        self,
        image_path: str,
        colour_to_dye: dict,
        n_frames: int,
        density_per_um2: float,
        modality: str,
        on_rate: float,
        off_rate: float,
        bleach_after_cycles: int,
        photon_min: float,
        photon_max: float,
        background_photons: float,
        na: float,
        output_dir: str,
        run_name: str,
    ):
        """Render a synthetic STORM/PAINT frame stack from a pattern image and
        write it to disk as a real, fittable acquisition (TIFF stack +
        metadata.txt + ground-truth H5), reusing the calibration already
        loaded on self.pipeline. The actual simulation is pure/GUI-free —
        see pattern_source.simulate_acquisition, which this wraps — so the
        same code path is reusable from non-GUI tooling
        (claude/generate_test_fixtures.py).
        """
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent.parent))
        from pyS3M.simulation import pattern_source
        import pyS3M.IOFunctions as IOFunctions

        camera_pixel_size_nm = self.pipeline.pixel_size * 1000.0
        width, height = pattern_source.image_fov_camera_pixels(image_path, camera_pixel_size_nm)

        bayer_stack, ground_truth, width, height, avg_emission_wavelength_nm = pattern_source.simulate_acquisition(
            image=image_path,
            colour_to_dye=colour_to_dye,
            camera=self.pipeline.camera,
            pixel_size_um=self.pipeline.pixel_size,
            gain_map=self.pipeline.gain_map[:height, :width],
            offset_map=self.pipeline.offset_map[:height, :width],
            variance_map=self.pipeline.variance[:height, :width],
            rqe_map=self.pipeline.rqe[:height, :width],
            n_frames=n_frames,
            density_per_um2=density_per_um2,
            modality=modality,
            on_rate=on_rate,
            off_rate=off_rate,
            bleach_after_cycles=bleach_after_cycles,
            photon_range=(photon_min, photon_max),
            background_photons=background_photons,
            na=na,
        )

        out_dir = Path(output_dir) / run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        io = IOFunctions.IO_Functions()
        io.write_tiff(bayer_stack, out_dir / f"{run_name}_MMStack_Default.ome.tif", bit="uint16", pixel_size=self.pipeline.pixel_size)

        metadata = {"FrameKey-0-0-0": {"ROI": f"0-0-{width}-{height}"}}
        import json
        with open(out_dir / f"{run_name}_MMStack_Default_metadata.txt", "w") as f:
            json.dump(metadata, f)

        # Written to a subfolder, under a non-default key, so a later fit run on
        # out_dir doesn't glob this file in as if it were localisation results:
        # load_localisations[_per_fov]'s glob is non-recursive (won't descend
        # into ground_truth/), and even if it did, they already skip files that
        # lack a "data" key, which now includes this one. Two independent
        # layers deliberately -- see claude/LOG.md "GUI bug reports from real
        # Windows/Linux usage".
        gt_dir = out_dir / "ground_truth"
        gt_dir.mkdir(parents=True, exist_ok=True)
        io.write_h5_database(ground_truth, gt_dir / "ground_truth.h5", verbose=False, key="ground_truth")

        fig = self._make_pattern_simulation_figure(
            bayer_stack, ground_truth, colour_to_dye, width, height,
        )
        return fig, str(out_dir), avg_emission_wavelength_nm

    def _make_pattern_simulation_figure(
        self, bayer_stack, ground_truth: pd.DataFrame, colour_to_dye: dict,
        width: int, height: int,
    ) -> Figure:
        """Representative frame (max-intensity projection, to actually show
        something given the ON duty cycle is only a few percent) alongside
        the ground-truth candidate positions coloured per dye."""
        fig = Figure(figsize=(9, 4.5), dpi=100, layout="constrained")
        ax_frame, ax_gt = fig.subplots(1, 2)

        projection = np.max(bayer_stack, axis=0)
        ax_frame.imshow(projection.T, cmap="gray", origin="upper")
        ax_frame.set_title("Max-intensity projection")
        ax_frame.set_xticks([])
        ax_frame.set_yticks([])

        for colour, dye in colour_to_dye.items():
            sub = ground_truth[ground_truth["dye"] == dye]
            if sub.empty:
                continue
            ax_gt.scatter(
                sub["xc_nm"] / 1000.0, sub["yc_nm"] / 1000.0,
                s=4, color=[c / 255.0 for c in colour], label=dye, rasterized=True,
            )
        ax_gt.set_xlabel("x (µm)")
        ax_gt.set_ylabel("y (µm)")
        ax_gt.set_aspect("equal")
        ax_gt.invert_yaxis()
        ax_gt.set_title(f"Ground truth ({len(ground_truth):,} candidates)")
        ax_gt.legend(fontsize=7, loc="best")

        return fig

    # ------------------------------------------------------------------
    # Qt lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        for w in (self._worker, self._aux_worker):
            if w is not None and w.isRunning():
                w.quit()
                w.wait(2000)
        logging.getLogger().removeHandler(self._log_handler)
        super().closeEvent(event)
