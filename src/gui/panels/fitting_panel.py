from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QComboBox, QDoubleSpinBox, QSpinBox, QPushButton, QLineEdit, QLabel,
)
from PyQt6.QtCore import pyqtSignal

from pyS3M.gui.widgets.folder_picker import FolderPicker

# Project root is four levels up from this file (src/gui/panels/fitting_panel.py),
# same convention as setup_panel.py's _PROJECT_ROOT.
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_TEST_TIFFS_DIR = _PROJECT_ROOT / "test_tiffs"

_FRET_QD_MODES = ("fret", "qd")

# GUI-only mode identifiers (combo box userData) -> the mode string AnalysisPipeline.fit
# actually understands. "smlm_single"/"smlm_multi" both dispatch to "smlm" — they differ
# only in the combined_output extra kwarg set below, not in which pipeline method runs.
_PIPELINE_MODE = {
    "smlm_single": "smlm",
    "smlm_multi": "smlm",
    "fret": "fret",
    "qd": "qd",
}

# Short description shown beneath the mode dropdown so the single-vs-multi-FOV distinction
# (and what each multi-file mode actually iterates over) doesn't have to be guessed from
# the combo box label alone.
_MODE_DESCRIPTIONS = {
    "smlm_single": (
        "Treats every TIFF in the data folder as one continuous field of view: frame "
        "numbers carry over between files and all results are written to a single "
        "Localisations.h5. Use this when a recording has been split into several TIFF "
        "parts (e.g. Micro-Manager's automatic split once a file hits its size limit)."
    ),
    "smlm_multi": (
        "Fits each TIFF in the data folder independently and writes one Localisations "
        "HDF5 per TIFF, alongside it. Use this when the folder holds several separate "
        "fields of view, not one FOV split across files."
    ),
    "fret": (
        "Change-point detection across every TIFF in the folder, each treated as its own "
        "FOV/trace source."
    ),
    "qd": (
        "Full time-series quantum-dot fitting across every TIFF in the folder, each "
        "treated as its own FOV."
    ),
}


class FittingPanel(QWidget):
    """Controls for running `AnalysisPipeline.fit` on a data folder: mode
    selection (single/multi-FOV SMLM, multi-FOV FRET change-point detection,
    multi-FOV quantum dot time series) plus the `FittingConfig` knobs.
    "Preview Fit" only applies to single-frame SMLM modes — it fits one
    frame without committing results, letting parameters be tuned before a
    full "Run Fitting"."""

    fit_requested          = pyqtSignal(str, str, object, object)  # data_dir, mode, FittingConfig, extra_kwargs
    preview_requested      = pyqtSignal(str, object)               # data_dir, FittingConfig
    stats_refresh_requested = pyqtSignal(tuple)                    # (min_photons, max_photons)
    clear_requested        = pyqtSignal()                          # discard fitting results, try again

    def __init__(self, parent=None):
        super().__init__(parent)
        self._enabled_by_state = False
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        grp = QGroupBox("Fitting")
        form = QFormLayout(grp)

        self._data_dir = FolderPicker(
            "Select data folder…",
            default_dir=str(_TEST_TIFFS_DIR) if _TEST_TIFFS_DIR.is_dir() else "",
        )
        self._data_dir.path_changed.connect(self._update_btns)
        form.addRow("Data folder:", self._data_dir)

        self._mode = QComboBox()
        self._mode.addItem("Single-FOV (SMLM)", userData="smlm_single")
        self._mode.addItem("Multi-FOV (SMLM, multiple files)", userData="smlm_multi")
        self._mode.addItem("Multi-FOV FRET (change-point detection, multi-file)", userData="fret")
        self._mode.addItem("Multi-FOV Quantum Dot (full time series, multi-file)", userData="qd")
        self._mode.currentIndexChanged.connect(self._on_mode_changed)
        form.addRow("Mode:", self._mode)

        self._mode_hint = QLabel()
        self._mode_hint.setWordWrap(True)
        self._mode_hint.setStyleSheet("color: palette(mid); font-size: 11px;")
        form.addRow("", self._mode_hint)

        self._pfa = QLineEdit("1e-3")
        self._pfa.setPlaceholderText("e.g. 1e-3")
        form.addRow("PFA:", self._pfa)

        self._roi_size = QSpinBox()
        self._roi_size.setRange(4, 64)
        self._roi_size.setValue(16)
        self._roi_size.setSuffix(" px")
        form.addRow("ROI size:", self._roi_size)

        self._wavelength = QDoubleSpinBox()
        self._wavelength.setRange(0.4, 1.0)
        self._wavelength.setDecimals(3)
        self._wavelength.setSingleStep(0.01)
        self._wavelength.setValue(0.638)
        self._wavelength.setSuffix(" µm")
        form.addRow("Peak λ:", self._wavelength)

        self._na = QDoubleSpinBox()
        self._na.setRange(0.1, 2.0)
        self._na.setDecimals(2)
        self._na.setSingleStep(0.01)
        self._na.setValue(1.49)
        form.addRow("NA:", self._na)

        # Preview + Run buttons side by side
        btn_row = QWidget()
        btn_lay = QHBoxLayout(btn_row)
        btn_lay.setContentsMargins(0, 0, 0, 0)
        btn_lay.setSpacing(6)
        self._preview_btn = QPushButton("Preview Fit")
        self._preview_btn.setEnabled(False)
        self._preview_btn.setToolTip("Run spot detection + fitting on one frame")
        self._preview_btn.clicked.connect(self._on_preview_clicked)
        self._run_btn = QPushButton("Run Fitting")
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._on_run_clicked)
        self._clear_btn = QPushButton("Clear Results")
        self._clear_btn.setEnabled(False)
        self._clear_btn.setToolTip("Discard fitting results so you can try again with different parameters")
        self._clear_btn.clicked.connect(self.clear_requested.emit)
        btn_lay.addWidget(self._preview_btn)
        btn_lay.addWidget(self._run_btn)
        btn_lay.addWidget(self._clear_btn)
        form.addRow(btn_row)

        outer.addWidget(grp)

        # ── Advanced detection options (FRET / QD only) ───────────────
        self._adv_grp = QGroupBox("Advanced Detection Options")
        adv_form = QFormLayout(self._adv_grp)
        self._adv_form = adv_form

        self._n_frames_sum = QSpinBox()
        self._n_frames_sum.setRange(1, 10_000)
        self._n_frames_sum.setValue(50)
        self._n_frames_sum.setToolTip("Number of frames summed for spot detection")
        adv_form.addRow("Frames to sum:", self._n_frames_sum)

        self._cp_penalty = QDoubleSpinBox()
        self._cp_penalty.setRange(0.01, 10.0)
        self._cp_penalty.setDecimals(2)
        self._cp_penalty.setSingleStep(0.1)
        self._cp_penalty.setValue(1.0)
        self._cp_penalty.setToolTip("Change-point detection penalty multiplier (FRET only)")
        adv_form.addRow("CP penalty:", self._cp_penalty)
        self._cp_row_idx = 1  # row index within adv_form

        self._chunk_size = QSpinBox()
        self._chunk_size.setRange(10, 10_000)
        self._chunk_size.setValue(500)
        self._chunk_size.setSingleStep(100)
        self._chunk_size.setToolTip("Frames loaded and fitted per chunk (QD only)")
        adv_form.addRow("Chunk size (frames):", self._chunk_size)
        self._chunk_row_idx = 2  # row index within adv_form

        self._adv_grp.setVisible(False)
        outer.addWidget(self._adv_grp)

        # ── Statistics filter ─────────────────────────────────────────
        flt_grp = QGroupBox("Statistics Filter")
        flt_form = QFormLayout(flt_grp)

        self._min_photons = QSpinBox()
        self._min_photons.setRange(0, 100_000_000)
        self._min_photons.setValue(200)
        self._min_photons.setSingleStep(100)
        flt_form.addRow("Min photons:", self._min_photons)

        self._max_photons = QSpinBox()
        self._max_photons.setRange(0, 100_000_000)
        self._max_photons.setValue(1_000_000)
        self._max_photons.setSingleStep(10_000)
        flt_form.addRow("Max photons:", self._max_photons)

        self._refresh_stats_btn = QPushButton("Refresh Stats")
        self._refresh_stats_btn.setEnabled(False)
        self._refresh_stats_btn.clicked.connect(self._on_refresh_stats_clicked)
        flt_form.addRow(self._refresh_stats_btn)

        outer.addWidget(flt_grp)

        # currentIndexChanged doesn't fire for the combo box's initial default item —
        # populate the hint text (and adv-group visibility) for it explicitly.
        self._on_mode_changed(self._mode.currentIndex())

    # ── helpers ──────────────────────────────────────────────────────

    def _make_fitting_config(self):
        from pyS3M.AnalysisPipeline import FittingConfig
        try:
            pfa = float(self._pfa.text())
        except ValueError:
            pfa = 1e-3
        return FittingConfig(
            pfa=pfa,
            ROI_size=self._roi_size.value(),
            peak_wavelength=self._wavelength.value(),
            NA=self._na.value(),
            fraction_true=0.0,
            use_variance_aware_demosaic=False,
        )

    def _extra_kwargs(self) -> dict:
        mode = self._mode.currentData()
        extra = {}
        if mode in _FRET_QD_MODES:
            extra["n_frames_sum"] = self._n_frames_sum.value()
        if mode == "fret":
            extra["cp_penalty_factor"] = self._cp_penalty.value()
        if mode == "qd":
            extra["chunk_size"] = self._chunk_size.value()
        if mode == "smlm_single":
            extra["combined_output"] = True
        return extra

    def _on_mode_changed(self, _idx: int):
        mode = self._mode.currentData()
        is_fret_qd = mode in _FRET_QD_MODES
        self._adv_grp.setVisible(is_fret_qd)
        self._adv_form.setRowVisible(self._cp_row_idx, mode == "fret")
        self._adv_form.setRowVisible(self._chunk_row_idx, mode == "qd")
        # Preview only applies to single-frame SMLM modes
        self._preview_btn.setVisible(not is_fret_qd)
        self._mode_hint.setText(_MODE_DESCRIPTIONS.get(mode, ""))
        self._update_btns()

    def _update_btns(self):
        ok = self._enabled_by_state and bool(self._data_dir.path)
        if not self._preview_btn.text().startswith("Preview…"):
            self._preview_btn.setEnabled(ok)
        if not self._run_btn.text().startswith("Running"):
            self._run_btn.setEnabled(ok)

    def _on_preview_clicked(self):
        self.preview_requested.emit(self._data_dir.path, self._make_fitting_config())

    def _on_refresh_stats_clicked(self):
        self.stats_refresh_requested.emit(self.photon_range)

    def _on_run_clicked(self):
        gui_mode = self._mode.currentData()
        self.fit_requested.emit(
            self._data_dir.path,
            _PIPELINE_MODE.get(gui_mode, gui_mode),
            self._make_fitting_config(),
            self._extra_kwargs(),
        )

    # ── public interface ──────────────────────────────────────────────

    @property
    def photon_range(self) -> tuple:
        return (self._min_photons.value(), self._max_photons.value())

    @property
    def data_dir(self) -> str:
        return self._data_dir.path

    def set_data_dir(self, path: str):
        self._data_dir.set_path(path)
        self._update_btns()

    def set_preview_busy(self, busy: bool):
        self._preview_btn.setEnabled(not busy)
        self._preview_btn.setText("Preview…" if busy else "Preview Fit")
        self._run_btn.setEnabled(not busy and self._enabled_by_state and bool(self._data_dir.path))

    def set_fit_busy(self, busy: bool):
        self._run_btn.setEnabled(not busy)
        self._run_btn.setText("Running…" if busy else "Run Fitting")
        self._preview_btn.setEnabled(not busy and self._enabled_by_state and bool(self._data_dir.path))
        if busy:
            self._clear_btn.setEnabled(False)

    def set_busy(self, busy: bool):
        """Reset both buttons (used by error handler)."""
        self._preview_btn.setEnabled(not busy)
        self._preview_btn.setText("Preview Fit")
        self._run_btn.setEnabled(not busy)
        self._run_btn.setText("Run Fitting")
        if busy:
            self._clear_btn.setEnabled(False)
        if not busy:
            self._update_btns()

    def set_clear_enabled(self, enabled: bool):
        self._clear_btn.setEnabled(enabled)

    def on_state_changed(self, state: str):
        self._enabled_by_state = state in ("calibrated", "fitted", "clustered")
        self._update_btns()
        self._refresh_stats_btn.setEnabled(state in ("fitted", "clustered"))
