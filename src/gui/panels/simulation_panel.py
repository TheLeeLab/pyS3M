from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QComboBox, QDoubleSpinBox, QSpinBox, QPushButton, QCheckBox, QLabel,
    QLineEdit,
)
from PyQt6.QtCore import pyqtSignal

from pyS3M.gui.widgets.image_picker import ImagePicker
from pyS3M.gui.widgets.folder_picker import FolderPicker

# Project root is four levels up from this file (src/gui/panels/simulation_panel.py),
# same convention as setup_panel.py's _PROJECT_ROOT.
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_EXAMPLE_PATTERNS_DIR = _PROJECT_ROOT / "test_tiffs" / "example_SR_patterns"

# Curated dyes commonly used in pyS3M notebooks
_CURATED_DYES = [
    "Cy3B",
    "Cy3",
    "Cy5",
    "Alexa Fluor 647",
    "ATTO 488",
    "ATTO 532",
    "ATTO 550",
    "ATTO 647N",
    "Nile Red",
    "Janelia Fluor JF549-HaloTag conjugate",
    "Janelia Fluor JF646-HaloTag conjugate",
    "SiR",
    "Abberior STAR RED",
]

# Curated filters: (display_label, filter_id)
_CURATED_FILTERS = [
    ("Notch 405/488/561/635", "semrock-nf03-405-488-561-635e"),
    ("BP 520/44",             "semrock-ff01-520-44"),
    ("BP 540/80",             "semrock-ff01-540-80"),
    ("BP 582/64",             "semrock-ff01-582-64"),
    ("BP 650/200",            "semrock-ff01-650-200"),
    ("LP 488",               "semrock-blp01-488r"),
    ("LP 561",               "semrock-blp02-561r"),
    ("LP 568",               "semrock-blp01-568r"),
    ("LP 635",               "semrock-blp01-635r"),
    ("LP 515",               "semrock-ff01-515-lp"),
    ("Dichroic R514",        "semrock-di03-r514-t1-25x36"),
]

# Fixed photon levels to sweep (rows in the PSF grid)
_PHOTON_LEVELS = [200, 500, 1_000, 5_000]


class SimulationPanel(QWidget):
    """Controls for the Simulation tab — exemplar PSF grid, plus (once
    calibration is loaded) image-driven STORM/PAINT acquisition simulation."""

    # dye_name, filter_ids, bg_photons_per_px, na, pixel_size_nm,
    # read_noise_e, peak_qe, n_replicates
    simulation_requested = pyqtSignal(str, list, float, float, float, float, float, int)

    # image_path, colour_to_dye {(r,g,b): dye_name}, n_frames, density_per_um2,
    # modality, on_rate, off_rate, bleach_after_cycles, photon_min, photon_max,
    # background_photons, na, output_dir, run_name
    pattern_simulation_requested = pyqtSignal(
        str, dict, int, float, str, float, float, int, float, float, float, float, str, str,
    )
    clear_pattern_requested = pyqtSignal()  # discard the simulation preview, try again

    def __init__(self, parent=None):
        super().__init__(parent)
        self._enabled_by_state = False
        self._colour_dye_rows: list[tuple[tuple, QComboBox]] = []
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)

        # --- Dye ---
        dye_grp = QGroupBox("Dye")
        dye_form = QFormLayout(dye_grp)
        self._dye = QComboBox()
        self._dye.addItems(_CURATED_DYES)
        dye_form.addRow("Dye:", self._dye)
        outer.addWidget(dye_grp)

        # --- Filters ---
        filt_grp = QGroupBox("Filters")
        filt_layout = QVBoxLayout(filt_grp)
        self._filter_checks: list[tuple[QCheckBox, str]] = []
        for label, fid in _CURATED_FILTERS:
            cb = QCheckBox(label)
            self._filter_checks.append((cb, fid))
            filt_layout.addWidget(cb)
        outer.addWidget(filt_grp)

        # --- Parameters ---
        param_grp = QGroupBox("Parameters")
        param_form = QFormLayout(param_grp)

        self._bg = QDoubleSpinBox()
        self._bg.setRange(0.0, 1000.0)
        self._bg.setValue(10.0)
        self._bg.setDecimals(1)
        self._bg.setSuffix(" ph/px")
        self._bg.setToolTip("Shared by both sections below.")
        param_form.addRow("Background:", self._bg)

        self._na = QDoubleSpinBox()
        self._na.setRange(0.5, 1.7)
        self._na.setValue(1.49)
        self._na.setDecimals(2)
        param_form.addRow("NA:", self._na)

        self._pixel_size = QDoubleSpinBox()
        self._pixel_size.setRange(10.0, 500.0)
        self._pixel_size.setValue(69.0)
        self._pixel_size.setDecimals(1)
        self._pixel_size.setSuffix(" nm")
        param_form.addRow("Pixel size:", self._pixel_size)

        self._read_noise = QDoubleSpinBox()
        self._read_noise.setRange(0.0, 20.0)
        self._read_noise.setValue(1.0)
        self._read_noise.setDecimals(2)
        self._read_noise.setSuffix(" e⁻ RMS")
        param_form.addRow("Read noise:", self._read_noise)

        self._peak_qe = QDoubleSpinBox()
        self._peak_qe.setRange(0.01, 1.0)
        self._peak_qe.setValue(0.7)
        self._peak_qe.setDecimals(2)
        param_form.addRow("Peak QE:", self._peak_qe)

        self._n_rep = QSpinBox()
        self._n_rep.setRange(1, 10)
        self._n_rep.setValue(5)
        param_form.addRow("Replicates:", self._n_rep)

        outer.addWidget(param_grp)

        # --- Run ---
        self._run_btn = QPushButton("Run Simulation")
        self._run_btn.clicked.connect(self._on_run)
        outer.addWidget(self._run_btn)

        # --- Simulate STORM/PAINT from Image ---
        pat_grp = QGroupBox("Simulate STORM/PAINT from Image")
        pat_outer = QVBoxLayout(pat_grp)

        self._calibration_hint = QLabel(
            "⚠ Requires a loaded calibration — load one on the Analysis tab, or "
            "compute one on the CMOS Calibration tab, first."
        )
        self._calibration_hint.setWordWrap(True)
        self._calibration_hint.setStyleSheet("color: #b36b00;")
        pat_outer.addWidget(self._calibration_hint)

        self._pattern_image = ImagePicker(
            "Select pattern image (e.g. a grid)…",
            default_dir=str(_EXAMPLE_PATTERNS_DIR) if _EXAMPLE_PATTERNS_DIR.is_dir() else "",
        )
        self._pattern_image.path_changed.connect(self._on_pattern_image_changed)
        pat_outer.addWidget(self._pattern_image)

        self._colour_dye_box = QGroupBox("Detected colours → dyes")
        self._colour_dye_layout = QVBoxLayout(self._colour_dye_box)
        self._no_colours_label = QLabel("Select a pattern image first.")
        self._no_colours_label.setStyleSheet("color: gray;")
        self._colour_dye_layout.addWidget(self._no_colours_label)
        pat_outer.addWidget(self._colour_dye_box)

        pat_form = QFormLayout()

        self._modality = QComboBox()
        self._modality.addItems(["STORM", "PAINT"])
        self._modality.setToolTip(
            "STORM: fixed dyes, photobleach after a few on/off cycles.\n"
            "PAINT: transient probes, continuously exchanged — no bleaching."
        )
        pat_form.addRow("Modality:", self._modality)

        self._pattern_n_frames = QSpinBox()
        self._pattern_n_frames.setRange(1, 100_000)
        self._pattern_n_frames.setValue(2000)
        pat_form.addRow("N frames:", self._pattern_n_frames)

        self._density = QDoubleSpinBox()
        self._density.setRange(0.01, 10_000.0)
        self._density.setValue(0.1)
        self._density.setDecimals(2)
        self._density.setSuffix(" /µm²")
        self._density.setToolTip(
            "Density of candidate molecule positions per µm² of each colour's masked "
            "structure area (not the whole field of view), per frame of the movie — any "
            "point of the mask could be sampled by a molecule in any frame, at this "
            "density, so the total candidate pool scales with density × area × N frames. "
            "0.2/µm² is a typical single-molecule SR density. Each candidate then blinks "
            "on/off independently (rates below); the number actually on in any given frame "
            "is far smaller than the pool. Solid/filled patterns will need a lower value "
            "than thin line patterns to keep the pool size (and render time) reasonable, "
            "since they cover far more masked area."
        )
        pat_form.addRow("Density:", self._density)

        self._on_rate = QDoubleSpinBox()
        self._on_rate.setRange(0.0001, 1.0)
        self._on_rate.setValue(0.01)
        self._on_rate.setDecimals(4)
        self._on_rate.setSingleStep(0.001)
        self._on_rate.setToolTip(
            "Per-frame probability an OFF candidate turns ON. Low by default so "
            "most molecules are off most of the time."
        )
        pat_form.addRow("On rate:", self._on_rate)

        self._off_rate = QDoubleSpinBox()
        self._off_rate.setRange(0.0001, 1.0)
        self._off_rate.setValue(0.5)
        self._off_rate.setDecimals(4)
        self._off_rate.setSingleStep(0.05)
        self._off_rate.setToolTip(
            "Per-frame probability an ON candidate turns OFF. Default gives a "
            "mean on-time of 1/off_rate ≈ 2 frames."
        )
        pat_form.addRow("Off rate:", self._off_rate)

        self._bleach_cycles = QSpinBox()
        self._bleach_cycles.setRange(1, 1000)
        self._bleach_cycles.setValue(5)
        self._bleach_cycles.setToolTip("STORM only: on/off cycles before a dye photobleaches.")
        pat_form.addRow("Bleach after cycles:", self._bleach_cycles)

        photon_row = QHBoxLayout()
        self._photon_min = QDoubleSpinBox()
        self._photon_min.setRange(1.0, 1_000_000.0)
        self._photon_min.setValue(1000.0)
        self._photon_min.setDecimals(0)
        self._photon_max = QDoubleSpinBox()
        self._photon_max.setRange(1.0, 1_000_000.0)
        self._photon_max.setValue(10_000.0)
        self._photon_max.setDecimals(0)
        photon_row.addWidget(self._photon_min)
        photon_row.addWidget(QLabel("–"))
        photon_row.addWidget(self._photon_max)
        pat_form.addRow("Photons/frame:", photon_row)

        pat_outer.addLayout(pat_form)

        self._pattern_output_dir = FolderPicker("Select output folder…")
        self._pattern_output_dir.path_changed.connect(self._update_pattern_run_btn)
        pat_outer.addWidget(self._pattern_output_dir)

        self._run_name = QLineEdit("simulation")
        self._run_name.setPlaceholderText("Run name (e.g. grid_test)")
        self._run_name.textChanged.connect(self._update_pattern_run_btn)
        pat_outer.addWidget(self._run_name)

        pattern_btn_row = QWidget()
        pattern_btn_lay = QHBoxLayout(pattern_btn_row)
        pattern_btn_lay.setContentsMargins(0, 0, 0, 0)
        pattern_btn_lay.setSpacing(6)
        self._pattern_run_btn = QPushButton("Simulate Acquisition")
        self._pattern_run_btn.setEnabled(False)
        self._pattern_run_btn.clicked.connect(self._on_run_pattern_simulation)
        self._pattern_clear_btn = QPushButton("Clear Preview")
        self._pattern_clear_btn.setEnabled(False)
        self._pattern_clear_btn.setToolTip("Clear the simulation preview so you can try again with different parameters")
        self._pattern_clear_btn.clicked.connect(self.clear_pattern_requested.emit)
        pattern_btn_lay.addWidget(self._pattern_run_btn)
        pattern_btn_lay.addWidget(self._pattern_clear_btn)
        pat_outer.addWidget(pattern_btn_row)

        outer.addWidget(pat_grp)

        outer.addStretch()

    def _on_pattern_image_changed(self, path: str):
        while self._colour_dye_layout.count():
            item = self._colour_dye_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._colour_dye_rows = []

        colours: list[tuple] = []
        if path:
            from pyS3M.simulation.pattern_source import detect_palette
            try:
                _, colours = detect_palette(path)
            except Exception:
                colours = []

        if not colours:
            label = QLabel("No distinct colours found." if path else "Select a pattern image first.")
            label.setStyleSheet("color: gray;")
            self._colour_dye_layout.addWidget(label)
        else:
            for colour in colours:
                row = QHBoxLayout()
                swatch = QLabel()
                swatch.setFixedSize(16, 16)
                swatch.setStyleSheet(
                    f"background-color: rgb({colour[0]},{colour[1]},{colour[2]}); "
                    "border: 1px solid gray;"
                )
                dye_combo = QComboBox()
                dye_combo.addItems(_CURATED_DYES)
                row.addWidget(swatch)
                row.addWidget(dye_combo)
                container = QWidget()
                container.setLayout(row)
                self._colour_dye_layout.addWidget(container)
                self._colour_dye_rows.append((colour, dye_combo))

        self._update_pattern_run_btn()

    def _update_pattern_run_btn(self):
        ready = (
            bool(self._colour_dye_rows)
            and bool(self._pattern_output_dir.path)
            and bool(self._run_name.text().strip())
            and self._enabled_by_state
        )
        if not self._pattern_run_btn.text().startswith("Running"):
            self._pattern_run_btn.setEnabled(ready)

    def _on_run_pattern_simulation(self):
        colour_to_dye = {
            colour: combo.currentText() for colour, combo in self._colour_dye_rows
        }
        self.pattern_simulation_requested.emit(
            self._pattern_image.path,
            colour_to_dye,
            self._pattern_n_frames.value(),
            self._density.value(),
            self._modality.currentText(),
            self._on_rate.value(),
            self._off_rate.value(),
            self._bleach_cycles.value(),
            self._photon_min.value(),
            self._photon_max.value(),
            self._bg.value(),
            self._na.value(),
            self._pattern_output_dir.path,
            self._run_name.text().strip(),
        )

    def _on_run(self):
        self.simulation_requested.emit(
            self._dye.currentText(),
            [fid for cb, fid in self._filter_checks if cb.isChecked()],
            self._bg.value(),
            self._na.value(),
            self._pixel_size.value(),
            self._read_noise.value(),
            self._peak_qe.value(),
            self._n_rep.value(),
        )

    def set_busy(self, busy: bool):
        self._run_btn.setEnabled(not busy)
        self._run_btn.setText("Running…" if busy else "Run Simulation")
        self._pattern_run_btn.setText("Running…" if busy else "Simulate Acquisition")
        if busy:
            self._pattern_run_btn.setEnabled(False)
            self._pattern_clear_btn.setEnabled(False)
        else:
            self._update_pattern_run_btn()

    def set_clear_enabled(self, enabled: bool):
        self._pattern_clear_btn.setEnabled(enabled)

    def on_state_changed(self, state: str):
        # Only the image-driven group gates on state (it needs a real loaded
        # calibration); the exemplar-grid section above stays always-available.
        self._enabled_by_state = state in ("calibrated", "fitted", "undrifted", "clustered")
        self._calibration_hint.setVisible(not self._enabled_by_state)
        self._update_pattern_run_btn()
