from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QComboBox, QDoubleSpinBox, QSpinBox, QPushButton, QLabel, QLineEdit,
    QFileDialog,
)
from PyQt6.QtCore import pyqtSignal


class PostProcPanel(QWidget):
    """Post-fitting controls: load a saved localisation `.h5` (skipping fitting
    entirely), filter + cluster it (HDBSCAN or DBSCAN, building `sm_db`/`sf_db`
    via `FilteringCriteria`/`ClusteringConfig`), and save the clustered result
    back out. The ε-multiplier row only applies to DBSCAN and is hidden
    whenever HDBSCAN is selected. "Clear Results" discards the current
    clustering (not the loaded localisations) so different filter/cluster
    parameters can be tried without reloading."""

    cluster_requested  = pyqtSignal(object, object)  # FilteringCriteria, ClusteringConfig
    load_locs_requested = pyqtSignal(str)             # path to .h5 file
    save_requested     = pyqtSignal()                 # triggered by Save Results button
    clear_requested    = pyqtSignal()                 # discard clustering results, try again

    def __init__(self, parent=None):
        super().__init__(parent)
        self._enabled_by_state = False
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        # Load localisations group
        lgrp = QGroupBox("Load Localisations")
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

        outer.addWidget(lgrp)

        # Filtering group
        fgrp = QGroupBox("Filtering")
        fform = QFormLayout(fgrp)

        self._min_photoelectrons = QSpinBox()
        self._min_photoelectrons.setRange(0, 1_000_000)
        self._min_photoelectrons.setValue(50)
        fform.addRow("Min photoelectrons:", self._min_photoelectrons)

        self._max_colour_err = QDoubleSpinBox()
        self._max_colour_err.setRange(0.0, 1.0)
        self._max_colour_err.setDecimals(3)
        self._max_colour_err.setSingleStep(0.01)
        self._max_colour_err.setValue(0.15)
        fform.addRow("Max colour error:", self._max_colour_err)

        self._max_loc_err = QDoubleSpinBox()
        self._max_loc_err.setRange(0.0, 10.0)
        self._max_loc_err.setDecimals(2)
        self._max_loc_err.setSingleStep(0.1)
        self._max_loc_err.setValue(1.0)
        self._max_loc_err.setSuffix(" px")
        fform.addRow("Max loc error:", self._max_loc_err)

        # Clustering group
        cgrp = QGroupBox("Clustering")
        cform = QFormLayout(cgrp)
        self._cform = cform

        self._method = QComboBox()
        self._method.addItems(["HDBSCAN", "DBSCAN"])
        self._method.currentIndexChanged.connect(self._on_method_changed)
        cform.addRow("Method:", self._method)

        self._min_cluster = QSpinBox()
        self._min_cluster.setRange(1, 1000)
        self._min_cluster.setValue(10)
        cform.addRow("Min cluster size:", self._min_cluster)

        self._eps_multiplier = QDoubleSpinBox()
        self._eps_multiplier.setRange(0.1, 20.0)
        self._eps_multiplier.setDecimals(2)
        self._eps_multiplier.setSingleStep(0.1)
        self._eps_multiplier.setValue(1.0)
        self._eps_multiplier.setToolTip(
            "Multiplier on median localisation precision to derive the DBSCAN ε radius.\n"
            "Increase if clusters are under-merged; decrease if over-merged."
        )
        self._eps_row_idx = cform.rowCount()
        cform.addRow("ε multiplier:", self._eps_multiplier)
        cform.setRowVisible(self._eps_row_idx, False)  # hidden until DBSCAN selected

        self._start_frame = QSpinBox()
        self._start_frame.setRange(0, 10_000_000)
        self._start_frame.setValue(0)
        cform.addRow("Start frame:", self._start_frame)

        self._cluster_btn = QPushButton("Filter & Cluster")
        self._cluster_btn.setEnabled(False)
        self._cluster_btn.clicked.connect(self._on_cluster_clicked)

        self._clear_btn = QPushButton("Clear Results")
        self._clear_btn.setEnabled(False)
        self._clear_btn.setToolTip("Discard clustering results so you can try again with different parameters")
        self._clear_btn.clicked.connect(self.clear_requested.emit)

        cluster_btn_row = QWidget()
        cluster_btn_lay = QHBoxLayout(cluster_btn_row)
        cluster_btn_lay.setContentsMargins(0, 0, 0, 0)
        cluster_btn_lay.setSpacing(6)
        cluster_btn_lay.addWidget(self._cluster_btn)
        cluster_btn_lay.addWidget(self._clear_btn)

        self._result_label = QLabel("—")

        self._save_btn = QPushButton("Save Results…")
        self._save_btn.setEnabled(False)
        self._save_btn.setToolTip("Save sm_db and sf_db to an HDF5 file")
        self._save_btn.clicked.connect(self.save_requested.emit)

        outer.addWidget(fgrp)
        outer.addWidget(cgrp)
        outer.addWidget(cluster_btn_row)
        outer.addWidget(self._result_label)
        outer.addWidget(self._save_btn)

    def _on_h5_browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select localisation file", "", "HDF5 files (*.h5 *.hdf5)"
        )
        if path:
            self._h5_path.setText(path)
            self._load_btn.setEnabled(True)

    def _on_load_clicked(self):
        path = self._h5_path.text().strip()
        if path:
            self.load_locs_requested.emit(path)

    def _on_method_changed(self):
        self._cform.setRowVisible(self._eps_row_idx, self._method.currentText() == "DBSCAN")

    def _on_cluster_clicked(self):
        from pyS3M.Constants import FilteringCriteria
        from pyS3M.clustering import ClusteringConfig
        filt = FilteringCriteria(
            min_photoelectrons=self._min_photoelectrons.value(),
            max_colour_error=self._max_colour_err.value(),
            max_localisation_error=self._max_loc_err.value(),
        )
        cc = ClusteringConfig(
            clustering_method=self._method.currentText(),
            min_cluster_size=self._min_cluster.value(),
            epsilon_multiplier=self._eps_multiplier.value(),
            start_frame=self._start_frame.value(),
        )
        self.cluster_requested.emit(filt, cc)

    # ── public interface ──────────────────────────────────────────────

    def set_busy(self, busy: bool):
        self._cluster_btn.setEnabled(not busy)
        self._cluster_btn.setText("Running…" if busy else "Filter & Cluster")
        if busy:
            self._clear_btn.setEnabled(False)

    def set_clear_enabled(self, enabled: bool):
        self._clear_btn.setEnabled(enabled)

    def show_result(self, n_sm: int, n_sf: int):
        self._result_label.setText(f"{n_sm} molecules from {n_sf} localisations")
        self._save_btn.setEnabled(True)

    def clear_result(self):
        """Reset the result label/save button back to their pre-clustering state."""
        self._result_label.setText("—")
        self._save_btn.setEnabled(False)

    def on_state_changed(self, state: str):
        self._enabled_by_state = state in ("fitted", "undrifted", "clustered")
        if not self._cluster_btn.text().startswith("Running"):
            self._cluster_btn.setEnabled(self._enabled_by_state)
