from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QLabel, QSizePolicy, QPushButton,
)
from PyQt6.QtCore import Qt, pyqtSignal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure


class _DraggableZoomRect:
    """Moves paired Rectangle patches across two full-field axes in sync and
    updates a matching pair of zoom axes live.  Must be instantiated after the
    FigureCanvasQTAgg is created so mpl_connect targets the live Qt canvas."""

    def __init__(self, rects, zoom_axes, image_shape):
        self._rects = rects
        self._zoom_axes = zoom_axes
        self._h, self._w = image_shape[:2]
        self._press = None

        canvas = rects[0].figure.canvas
        canvas.mpl_connect("button_press_event", self._on_press)
        canvas.mpl_connect("button_release_event", self._on_release)
        canvas.mpl_connect("motion_notify_event", self._on_motion)

    def _clamp(self, x0, y0, w, h):
        return max(0, min(x0, self._w - w)), max(0, min(y0, self._h - h))

    def _on_press(self, event):
        if event.inaxes is None:
            return
        for rect in self._rects:
            if rect.axes is not event.inaxes:
                continue
            contains, _ = rect.contains(event)
            if contains:
                x0, y0 = rect.get_xy()
                self._press = (x0, y0, event.xdata, event.ydata)
                return

    def _on_release(self, _event):
        self._press = None

    def _on_motion(self, event):
        if self._press is None or event.inaxes is None:
            return
        x0, y0, xpress, ypress = self._press
        r = self._rects[0]
        w, h = r.get_width(), r.get_height()
        new_x, new_y = self._clamp(x0 + event.xdata - xpress, y0 + event.ydata - ypress, w, h)
        for rect in self._rects:
            rect.set_xy((new_x, new_y))
        for ax in self._zoom_axes:
            ax.set_xlim(new_x, new_x + w)
            ax.set_ylim(new_y, new_y + h)
        r.figure.canvas.draw_idle()


class _FigureTab(QWidget):
    """Tab that shows a placeholder until a figure is provided."""

    def __init__(self, placeholder_msg: str, parent=None):
        super().__init__(parent)
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._placeholder = QLabel(placeholder_msg)
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet("color: gray; font-size: 13pt;")
        self._lay.addWidget(self._placeholder)
        self._canvas = None
        self._toolbar = None
        self._zoom_drag = None  # keep drag handler alive with the tab

    def set_figure(self, fig: Figure):
        if self._canvas is not None:
            self._lay.removeWidget(self._toolbar)
            self._lay.removeWidget(self._canvas)
            self._toolbar.deleteLater()
            self._canvas.deleteLater()
        self._zoom_drag = None
        self._placeholder.setVisible(False)

        # Reduce internal whitespace on figures that have no layout engine set.
        # constrained_layout figures manage their own spacing; for plain figures
        # use subplots_adjust with near-zero margins rather than tight_layout —
        # tight_layout still leaves noticeable padding, and for equal-aspect image
        # axes the inter-subplot gaps are the main source of wasted space.
        try:
            engine = fig.get_layout_engine()
            if engine is None or type(engine).__name__ == "PlaceholderLayoutEngine":
                fig.subplots_adjust(
                    left=0.02, right=0.98,
                    top=0.96, bottom=0.02,
                    wspace=0.03, hspace=0.03,
                )
        except Exception:
            pass

        self._canvas = FigureCanvasQTAgg(fig)
        # Let the canvas grow to fill whatever space Qt gives the tab.
        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        # Wire drag handler now that fig.canvas points to the Qt canvas
        if hasattr(fig, '_zoom_drag_data'):
            rects, zoom_axes, shape = fig._zoom_drag_data
            self._zoom_drag = _DraggableZoomRect(rects, zoom_axes, shape)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        self._lay.addWidget(self._toolbar)
        self._lay.addWidget(self._canvas)
        self._canvas.draw()

    def clear(self):
        """Discard the current figure and revert to the placeholder message."""
        if self._canvas is not None:
            self._lay.removeWidget(self._toolbar)
            self._lay.removeWidget(self._canvas)
            self._toolbar.deleteLater()
            self._canvas.deleteLater()
            self._toolbar = None
            self._canvas = None
        self._zoom_drag = None
        self._placeholder.setVisible(True)


class ResultsPanel(QWidget):
    """The right-hand results dock: one `_FigureTab` per pipeline stage
    (CMOS Calibration, Preview, Localisations, Statistics, Drift, FRC,
    Simulation, Unmixing, Nile Red), each showing a placeholder until its
    figure-setter method is called (e.g. `set_frc_figure`). `fov_requested`
    is emitted when the user navigates to a different FOV from the
    Localisations tab's controls."""

    fov_requested = pyqtSignal(int)   # emitted when user navigates to a different FOV

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)

        self._tabs = QTabWidget()
        lay.addWidget(self._tabs)

        self._calibration_tab = _FigureTab("Run CMOS calibration to see the computed maps.")
        self._tabs.addTab(self._calibration_tab, "CMOS Calibration")

        self._preview_tab = _FigureTab("Run fitting to see a preview.")
        self._tabs.addTab(self._preview_tab, "Preview")

        # Localisations tab: nav bar + figure
        locs_container = QWidget()
        locs_vbox = QVBoxLayout(locs_container)
        locs_vbox.setContentsMargins(0, 0, 0, 0)
        locs_vbox.setSpacing(2)

        self._locs_nav = QWidget()
        nav_h = QHBoxLayout(self._locs_nav)
        nav_h.setContentsMargins(4, 2, 4, 2)
        self._fov_prev_btn = QPushButton("←")
        self._fov_prev_btn.setFixedWidth(40)
        self._fov_label = QLabel("FOV 1 / 1")
        self._fov_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._fov_next_btn = QPushButton("→")
        self._fov_next_btn.setFixedWidth(40)
        nav_h.addWidget(self._fov_prev_btn)
        nav_h.addStretch()
        nav_h.addWidget(self._fov_label)
        nav_h.addStretch()
        nav_h.addWidget(self._fov_next_btn)
        self._locs_nav.setVisible(False)

        self._locs_tab = _FigureTab("Run fitting to see localisations.")
        locs_vbox.addWidget(self._locs_nav)
        locs_vbox.addWidget(self._locs_tab)
        self._tabs.addTab(locs_container, "Localisations")
        self._locs_container = locs_container

        self._stats_tab = _FigureTab("Run fitting to see photoelectron statistics.")
        self._tabs.addTab(self._stats_tab, "Statistics")

        self._drift_tab = _FigureTab("Run drift correction to see the drift trace.")
        self._tabs.addTab(self._drift_tab, "Drift")

        self._frc_tab = _FigureTab("Run FRC to see the resolution estimate.")
        self._tabs.addTab(self._frc_tab, "FRC")

        self._sim_tab = _FigureTab("Configure and run a simulation to see exemplar PSFs.")
        self._tabs.addTab(self._sim_tab, "Simulation")

        self._unmixing_tab = _FigureTab("Run channel unmixing to see the spectral/spatial overlay.")
        self._tabs.addTab(self._unmixing_tab, "Unmixing")

        self._nile_red_tab = _FigureTab("Run Nile Red fitting to see the wavelength map.")
        self._tabs.addTab(self._nile_red_tab, "Nile Red")

        self._n_fovs = 1
        self._current_fov_idx = 0
        self._fov_prev_btn.clicked.connect(self._on_fov_prev)
        self._fov_next_btn.clicked.connect(self._on_fov_next)

        self._tab_index = {
            "cmos_calibration": self._tabs.indexOf(self._calibration_tab),
            "preview": self._tabs.indexOf(self._preview_tab),
            "localisations": self._tabs.indexOf(self._locs_container),
            "statistics": self._tabs.indexOf(self._stats_tab),
            "drift": self._tabs.indexOf(self._drift_tab),
            "frc": self._tabs.indexOf(self._frc_tab),
            "simulation": self._tabs.indexOf(self._sim_tab),
            "channel_unmixing": self._tabs.indexOf(self._unmixing_tab),
            "nile_red": self._tabs.indexOf(self._nile_red_tab),
        }
        # Which result tabs are relevant for each controls-dock context. Contexts
        # not listed here (e.g. still-placeholder Nile Red tab) fall back to
        # showing everything, since they don't yet produce their own dedicated
        # result tab to switch to.
        self._context_tabs = {
            "cmos_calibration": {"cmos_calibration"},
            "analysis": {"preview", "localisations", "statistics", "drift"},
            "simulation": {"simulation"},
            "frc": {"frc"},
            # includes "frc": per-channel FRC (triggered from this tab once
            # Channel Unmixing has produced a `channel` column) lands there.
            "channel_unmixing": {"channel_unmixing", "frc"},
            "nile_red": {"nile_red"},
        }

    # ── FOV navigation ────────────────────────────────────────────────

    def set_fov_count(self, n: int):
        self._n_fovs = max(1, n)
        self._current_fov_idx = 0
        self._locs_nav.setVisible(self._n_fovs > 1)
        self._update_fov_nav()

    def set_current_fov_idx(self, idx: int):
        self._current_fov_idx = idx
        self._update_fov_nav()

    def _update_fov_nav(self):
        self._fov_label.setText(f"FOV {self._current_fov_idx + 1} / {self._n_fovs}")
        self._fov_prev_btn.setEnabled(self._current_fov_idx > 0)
        self._fov_next_btn.setEnabled(self._current_fov_idx < self._n_fovs - 1)

    def _on_fov_prev(self):
        if self._current_fov_idx > 0:
            self._current_fov_idx -= 1
            self._update_fov_nav()
            self.fov_requested.emit(self._current_fov_idx)

    def _on_fov_next(self):
        if self._current_fov_idx < self._n_fovs - 1:
            self._current_fov_idx += 1
            self._update_fov_nav()
            self.fov_requested.emit(self._current_fov_idx)

    # ── public figure setters ─────────────────────────────────────────

    def set_calibration_figure(self, fig: Figure):
        self._calibration_tab.set_figure(fig)
        self._tabs.setCurrentWidget(self._calibration_tab)

    def set_preview_figure(self, fig: Figure):
        self._preview_tab.set_figure(fig)
        self._tabs.setCurrentWidget(self._preview_tab)

    def set_localisations_figure(self, fig: Figure):
        self._locs_tab.set_figure(fig)
        self._tabs.setCurrentWidget(self._locs_container)

    def set_stats_figure(self, fig: Figure):
        self._stats_tab.set_figure(fig)

    def set_drift_figure(self, fig: Figure):
        self._drift_tab.set_figure(fig)
        self._tabs.setCurrentWidget(self._drift_tab)

    def set_frc_figure(self, fig: Figure):
        self._frc_tab.set_figure(fig)
        self._tabs.setCurrentWidget(self._frc_tab)

    def set_simulation_figure(self, fig: Figure):
        self._sim_tab.set_figure(fig)
        self._tabs.setCurrentWidget(self._sim_tab)

    def set_unmixing_figure(self, fig: Figure):
        self._unmixing_tab.set_figure(fig)
        self._tabs.setCurrentWidget(self._unmixing_tab)

    def set_nile_red_figure(self, fig: Figure):
        self._nile_red_tab.set_figure(fig)
        self._tabs.setCurrentWidget(self._nile_red_tab)

    # ── public figure clearers ────────────────────────────────────────
    # Mirror the setters above — used by MainWindow's "Clear Results" actions
    # so a discarded result doesn't linger as a stale figure in its tab.

    def clear_localisations_figure(self):
        self._locs_tab.clear()

    def clear_stats_figure(self):
        self._stats_tab.clear()

    def clear_drift_figure(self):
        self._drift_tab.clear()

    def clear_frc_figure(self):
        self._frc_tab.clear()

    def clear_simulation_figure(self):
        self._sim_tab.clear()

    def clear_unmixing_figure(self):
        self._unmixing_tab.clear()

    def clear_nile_red_figure(self):
        self._nile_red_tab.clear()

    # ── context-driven tab visibility ─────────────────────────────────

    def set_context(self, context: str):
        """Show only the result tabs relevant to *context* (a controls-dock
        tab key, e.g. "analysis"/"simulation"). Unknown contexts show all
        tabs, since not every controls-dock tab has a dedicated result view."""
        visible = self._context_tabs.get(context, set(self._tab_index))
        for name, idx in self._tab_index.items():
            self._tabs.setTabVisible(idx, name in visible)
        current = self._tabs.currentIndex()
        if current < 0 or not self._tabs.isTabVisible(current):
            for idx in self._tab_index.values():
                if self._tabs.isTabVisible(idx):
                    self._tabs.setCurrentIndex(idx)
                    break
