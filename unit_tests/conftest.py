import shutil
from pathlib import Path

import pytest

TEST_OUTPUT_DIR = Path(__file__).parent / "test_output"

# Delta-chi^2 fit gate: calibrate into a throw-away folder (never into the repo's
# Camera_Calibrations/) with few null fits, so the calibrations tests trigger are fast.
import os
import tempfile
_FIT_GATE_DIR = tempfile.mkdtemp(prefix="pys3m_fit_gate_")
os.environ["PYS3M_FIT_GATE_DIR"] = _FIT_GATE_DIR
os.environ["PYS3M_FIT_GATE_NNULL"] = "200"
PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture
def test_output_dir():
    """Directory for test-generated figures/artifacts.

    Unlike `tmp_path` (a fresh, hidden-away system temp directory per test,
    gone the moment that test finishes), this sits inside the repo so
    generated figures are easy to find and inspect during a test run --
    but it's wiped in one go once the whole session finishes
    (`pytest_sessionfinish` below), so nothing accumulates or gets committed.
    """
    TEST_OUTPUT_DIR.mkdir(exist_ok=True)
    return TEST_OUTPUT_DIR


@pytest.fixture(scope="session")
def real_fitted_drift_fixture(tmp_path_factory):
    """Real end-to-end fit of test_tiffs/drift_correction/ (gold-nanoparticle
    fiducials, injected 1000/500 nm drift), shared across the drift_correction/
    package's test files (fiducial.py + _facade.py's fiducial-detection tests)
    so the ~19s fit only happens once per test session, not once per test file.

    Mirrors notebooks/analyses/03_drift_correction.ipynb's exact fitting
    parameters. Returns a dict with the fitted pipeline, real localisations
    (both DataFrame and recarray form), and an `info` list using the real
    cropped ROI dimensions (145x145 -- NOT pipe.gain_map.shape, the full
    2064x1544 sensor, which would make any render()-based fiducial-detection
    call in a test try to allocate/scan a huge image for no benefit).
    """
    from pyS3M.AnalysisPipeline import AnalysisPipeline, FittingConfig
    from pyS3M.Constants import AnalysisConfig

    data_dir = PROJECT_ROOT / "test_tiffs" / "drift_correction"
    cal_dir = PROJECT_ROOT / "Camera_Calibrations" / "Ximea_Camera"
    assert data_dir.is_dir(), f"Bundled fixture missing: {data_dir}"
    assert cal_dir.is_dir(), f"Bundled calibration missing: {cal_dir}"

    work_dir = tmp_path_factory.mktemp("drift_correction_fit")
    for f in data_dir.iterdir():
        if f.is_file():
            shutil.copy(f, work_dir / f.name)

    cfg = AnalysisConfig(display=False)
    pipe = AnalysisPipeline(camera="ximea", config=cfg)
    pipe.load_calibration(cal_dir)
    fc = FittingConfig(peak_wavelength=0.561, pfa=1e-3)
    pipe.fit(work_dir, mode="smlm", fitting_config=fc)
    locs_df = pipe.load_localisations(work_dir)

    x0, y0, width, height = pipe.sr.helper.load_metadata_roi(
        work_dir, pipe.sr.io, use_fallback=False
    )
    pixel_size_nm = pipe.pixel_size * 1000.0
    n_frames = int(locs_df["frame"].max()) + 1
    info = [{
        "Width": int(width), "Height": int(height),
        "Frames": n_frames, "Pixelsize": pixel_size_nm,
    }]

    return {
        "pipe": pipe,
        "work_dir": work_dir,
        "locs_df": locs_df,
        "locs_rec": locs_df.to_records(index=False),
        "info": info,
        "pixel_size_nm": pixel_size_nm,
    }


def pytest_sessionfinish(session, exitstatus):
    """Remove all test-generated output once the full test session completes."""
    if TEST_OUTPUT_DIR.exists():
        shutil.rmtree(TEST_OUTPUT_DIR)
    shutil.rmtree(_FIT_GATE_DIR, ignore_errors=True)
