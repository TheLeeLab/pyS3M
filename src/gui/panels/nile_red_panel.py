from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QDoubleSpinBox, QSpinBox, QPushButton, QCheckBox, QLabel, QLineEdit,
    QFileDialog,
)
from PyQt6.QtCore import pyqtSignal

from pyS3M.gui.panels.simulation_panel import _CURATED_FILTERS, NILE_RED_DEFAULT_FILTERS

# Project root is four levels up from this file (src/gui/panels/nile_red_panel.py),
# same convention as fitting_panel.py's _PROJECT_ROOT.
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_NILE_RED_EXAMPLE_DIR = _PROJECT_ROOT / "test_tiffs" / "nile_red_example"


class NileRedPanel(QWidget):
    """Controls for pixelated Nile Red wavelength fitting
    (NileRed_Functions.fit_wavelengths_pixelated), run on the post-clustering
    per-localisation table (sf_db) by default — same "needs clustering to have
    produced real data first" gating as ChannelUnmixingPanel — or on an
    explicitly-loaded HDF5 file, which bypasses that gate entirely (useful for
    analysing a previously-saved localisation table without redoing fit+cluster
    in the current session). Always uses the pixelated (spatial-grid) fit
    rather than a plain per-localisation fit: it discretises localisations onto
    a regular pixel grid, pools each grid cell's photoelectrons per colour
    channel (far higher SNR than any single localisation), and fits one
    wavelength per cell, producing a genuine spatial wavelength map."""

    # filter_ids, wl_min, wl_max, pixel_size_nm, min_localisations
    fit_requested = pyqtSignal(list, float, float, float, int)
    clear_requested = pyqtSignal()
    load_locs_requested = pyqtSignal(str)  # path to .h5 file

    def __init__(self, parent=None):
        super().__init__(parent)
        self._enabled_by_state = False
        self._has_loaded_file = False
        self._filter_checks: list[tuple[QCheckBox, str]] = []
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        # --- Load Localisations (optional — overrides the current session's
        # clustering results with a previously-saved HDF5 file) ---
        lgrp = QGroupBox("Load Localisations (optional)")
        lform = QFormLayout(lgrp)

        h5_row = QWidget()
        h5_lay = QHBoxLayout(h5_row)
        h5_lay.setContentsMargins(0, 0, 0, 0)
        h5_lay.setSpacing(4)
        self._h5_path = QLineEdit()
        self._h5_path.setPlaceholderText("Select .h5 file…")
        self._h5_path.setReadOnly(True)
        self._h5_browse = QPushButton("Browse…")
        self._h5_browse.clicked.connect(self._on_h5_browse)
        h5_lay.addWidget(self._h5_path)
        h5_lay.addWidget(self._h5_browse)
        lform.addRow("H5 file:", h5_row)

        self._load_btn = QPushButton("Load")
        self._load_btn.setEnabled(False)
        self._load_btn.clicked.connect(self._on_load_clicked)
        lform.addRow(self._load_btn)

        self._load_status = QLabel("Using current clustering results.")
        self._load_status.setStyleSheet("color: gray;")
        self._load_status.setWordWrap(True)
        lform.addRow(self._load_status)

        outer.addWidget(lgrp)

        grp = QGroupBox("Nile Red — Pixelated Wavelength Fit")
        form = QFormLayout(grp)

        filter_box = QGroupBox("Filters")
        filter_lay = QVBoxLayout(filter_box)
        for label, filter_id in _CURATED_FILTERS:
            cb = QCheckBox(label)
            cb.setChecked(filter_id in NILE_RED_DEFAULT_FILTERS)
            filter_lay.addWidget(cb)
            self._filter_checks.append((cb, filter_id))
        form.addRow(filter_box)

        self._wl_min = QDoubleSpinBox()
        self._wl_min.setRange(300.0, 1000.0)
        self._wl_min.setDecimals(0)
        self._wl_min.setValue(500.0)
        self._wl_min.setSuffix(" nm")
        form.addRow("Min λ:", self._wl_min)

        self._wl_max = QDoubleSpinBox()
        self._wl_max.setRange(300.0, 1000.0)
        self._wl_max.setDecimals(0)
        self._wl_max.setValue(750.0)
        self._wl_max.setSuffix(" nm")
        form.addRow("Max λ:", self._wl_max)

        self._pixel_size = QDoubleSpinBox()
        self._pixel_size.setRange(5.0, 2000.0)
        self._pixel_size.setDecimals(0)
        self._pixel_size.setValue(50.0)
        self._pixel_size.setSuffix(" nm")
        self._pixel_size.setToolTip(
            "Spatial grid pixel size — trades spatial resolution against per-pixel SNR."
        )
        form.addRow("Grid pixel size:", self._pixel_size)

        self._min_locs = QSpinBox()
        self._min_locs.setRange(1, 1000)
        self._min_locs.setValue(3)
        self._min_locs.setToolTip(
            "Minimum localisations per grid pixel to attempt a fit; pixels below this "
            "fall back to their parent molecule's aggregate-level fit."
        )
        form.addRow("Min localisations/pixel:", self._min_locs)

        btn_row = QWidget()
        btn_lay = QHBoxLayout(btn_row)
        btn_lay.setContentsMargins(0, 0, 0, 0)
        btn_lay.setSpacing(6)
        self._run_btn = QPushButton("Run Nile Red Fit")
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._on_run_clicked)
        self._clear_btn = QPushButton("Clear Results")
        self._clear_btn.setEnabled(False)
        self._clear_btn.setToolTip("Discard the wavelength map so you can try again with different parameters")
        self._clear_btn.clicked.connect(self._on_clear_clicked)
        btn_lay.addWidget(self._run_btn)
        btn_lay.addWidget(self._clear_btn)
        form.addRow(btn_row)

        outer.addWidget(grp)
        outer.addStretch()

    def _on_h5_browse(self):
        start_dir = self._h5_path.text().strip() or (
            str(_NILE_RED_EXAMPLE_DIR) if _NILE_RED_EXAMPLE_DIR.is_dir() else ""
        )
        path, _ = QFileDialog.getOpenFileName(
            self, "Select localisation file", start_dir, "HDF5 files (*.h5 *.hdf5)"
        )
        if path:
            self._h5_path.setText(path)
            self._load_btn.setEnabled(True)

    def _on_load_clicked(self):
        path = self._h5_path.text().strip()
        if path:
            self.load_locs_requested.emit(path)

    def _on_run_clicked(self):
        filter_ids = [fid for cb, fid in self._filter_checks if cb.isChecked()]
        self.fit_requested.emit(
            filter_ids,
            self._wl_min.value(),
            self._wl_max.value(),
            self._pixel_size.value(),
            self._min_locs.value(),
        )

    def _on_clear_clicked(self):
        self._has_loaded_file = False
        self._h5_path.clear()
        self._load_btn.setEnabled(False)
        self._load_status.setText("Using current clustering results.")
        self._update_run_btn()
        self.clear_requested.emit()

    # ── public interface ──────────────────────────────────────────────

    def set_loaded(self, path: str, n_locs: int):
        """Called once a requested H5 file has been loaded successfully —
        bypasses the "clustered" state gate so the loaded file can be fitted
        regardless of what's happened in the current session."""
        self._has_loaded_file = True
        self._load_status.setText(f"Loaded {n_locs:,} localisations from {path}")
        self._update_run_btn()

    def _update_run_btn(self):
        if not self._run_btn.text().startswith("Running"):
            self._run_btn.setEnabled(self._enabled_by_state or self._has_loaded_file)

    def set_busy(self, busy: bool):
        self._run_btn.setEnabled(not busy and (self._enabled_by_state or self._has_loaded_file))
        self._run_btn.setText("Running…" if busy else "Run Nile Red Fit")
        if busy:
            self._clear_btn.setEnabled(False)

    def set_clear_enabled(self, enabled: bool):
        self._clear_btn.setEnabled(enabled)

    def on_state_changed(self, state: str):
        self._enabled_by_state = state == "clustered"
        self._update_run_btn()
