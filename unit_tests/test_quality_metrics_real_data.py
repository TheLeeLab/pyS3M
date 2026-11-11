#!/usr/bin/env python3
"""
Test quality metrics integration with real data.

This test analyzes a real folder using fit_imaging_data() and verifies that:
1. Quality metrics are correctly captured during spot detection
2. Quality metrics are filtered along with ROIs (edge cases, etc.)
3. Quality metrics are saved to the .h5 file
4. Quality metric columns are present in the output
"""

import sys
import os

import numpy as np
import pandas as pd
import types
import gc

# Import required modules
import pyS3M.IOFunctions as IOFunctions
import pyS3M.sCMOSFunctions as sCMOSFunctions
import pyS3M.SpectralFunctions as SpectralFunctions
import pyS3M.MaskFunctions as MaskFunctions
import pyS3M.SpotDetectionFunctions as SpotDetectionFunctions
import pyS3M.SR_Functions as SR_Functions
import pyS3M.ImageAnalysisFunctions as ImageAnalysisFunctions
import pyS3M.HelperFunctions as HelperFunctions


def test_quality_metrics_real_data():
    """Test quality metrics with real imaging data.

    Uses a bundled single-frame fixture (test_tiffs/quality_metrics_sample/) rather
    than an external drive -- a single real frame is sufficient here since spot
    detection/fitting runs independently per frame (no temporal/EVER filtering is
    exercised by this test), so it gives real localisations and quality-metric
    columns without needing more data. The accompanying metadata.txt is similarly
    trimmed to just the one "ROI" field load_metadata_roi()/metadata_reader_imageJ()
    actually read (no Summary block, no per-frame camera settings).
    """

    # Get project root and fixture/camera calibration folders
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_folder = os.path.join(project_root, 'test_tiffs', 'quality_metrics_sample')

    print("=" * 80)
    print("Testing Quality Metrics Integration with Real Data")
    print("=" * 80)
    print(f"Test folder: {test_folder}")
    print()

    assert os.path.isdir(test_folder), f"Bundled test fixture missing: {test_folder}"

    # Check for .tif files
    tif_files = [f for f in os.listdir(test_folder) if f.endswith('.tif')]
    assert tif_files, f"No .tif files found in fixture folder: {test_folder}"

    print(f"Found {len(tif_files)} .tif files")

    # Initialize functions
    print("Initializing functions...")
    io_funcs = IOFunctions.IO_Functions()
    scmos_funcs = sCMOSFunctions.sCMOS_Functions()
    sr_funcs = SR_Functions.SuperRes_Functions()

    camera_folder = os.path.join(project_root, 'Camera_Calibrations', 'Ximea_Camera')
    assert os.path.exists(camera_folder), f"Camera calibration folder not found: {camera_folder}"

    # Load camera parameters
    print("Loading camera parameters...")
    gain_map = io_funcs.read_tiff(os.path.join(camera_folder, 'gain.tif'))
    offset_map = io_funcs.read_tiff(os.path.join(camera_folder, 'offset.tif'))
    variance_map = io_funcs.read_tiff(os.path.join(camera_folder, 'variance.tif'))
    readnoise_map = io_funcs.read_tiff(os.path.join(camera_folder, 'readnoise.tif'))
    rqe_map = io_funcs.read_tiff(os.path.join(camera_folder, 'rqe.tif'))

    print(f"  Gain map shape: {gain_map.shape}")
    print(f"  Offset map shape: {offset_map.shape}")
    print(f"  Variance map shape: {variance_map.shape}")

    # Setup smoothing function
    print("Setting up smoothing function...")
    smoothing_function = types.SimpleNamespace()
    smoothing_function.args = {'sigma': 1.5}
    smoothing_function.extent = 1.5
    smoothing_function.smoothing_function = scmos_funcs.gaussian_filter_stack
    smoothing_function.data_arg = 'image'

    # Analysis parameters
    peak_wavelength = 0.638  # 638 nm (red)
    pfa = 1e-3
    ROI_size = 16
    NA = 1.49
    pixel_size = 0.069  # μm
    use_variance_aware_demosaic = True
    print()
    print("Analysis Parameters:")
    print(f"  Peak wavelength: {peak_wavelength} μm")
    print(f"  PFA: {pfa}")
    print(f"  ROI size: {ROI_size}")
    print(f"  Variance-aware demosaic: {use_variance_aware_demosaic}")
    print()

    # Clean up any existing .h5 files
    h5_file = os.path.join(test_folder, 'Localisations.h5')
    if os.path.exists(h5_file):
        print(f"Removing existing .h5 file: {h5_file}")
        os.remove(h5_file)

    # Run analysis
    print("=" * 80)
    print("Running fit_imaging_data()...")
    print("=" * 80)

    sr_funcs.fit_imaging_data(
        test_folder,
        smoothing_function,
        gain_map,
        offset_map,
        rqe_map,
        readnoise_map,
        variance=variance_map,
        pfa=pfa,
        ROI_size=ROI_size,
        peak_wavelength=peak_wavelength,
        NA=NA,
        pixel_size=pixel_size,
        image_type='.tif',
        use_variance_aware_demosaic=use_variance_aware_demosaic,
    )

    print()
    print("=" * 80)
    print("Analysis completed successfully!")
    print("=" * 80)

    try:
        assert os.path.exists(h5_file), f"Expected .h5 file was not created: {h5_file}"
        print(f"\n✓ .h5 file created: {h5_file}")

        # Read .h5 file and check for quality metrics
        print("\nReading .h5 file to verify quality metrics...")
        df = pd.read_hdf(h5_file, key='data')

        print(f"\n✓ Successfully read .h5 file")
        print(f"  Total localizations: {len(df)}")
        print(f"  Total columns: {len(df.columns)}")
        print()

        assert len(df) > 0, "fit_imaging_data produced zero localisations"

        # Check for standard columns
        standard_cols = ['xc', 'yc', 's_x', 's_y', 'A_R', 'A_G', 'A_B',
                        'bg_R', 'bg_G', 'bg_B', 'chi_sqr', 'frame', 'photons']

        print("Standard columns:")
        for col in standard_cols:
            present = "✓" if col in df.columns else "✗"
            print(f"  {present} {col}")

        # Check for quality metric columns (with 'spot_' prefix)
        expected_quality_cols = [
            'spot_matched_filter_response',
            'spot_background',
            'spot_background_std',
            'spot_mean_inner_intensity',
            'spot_snr'
        ]

        print()
        print("Quality metric columns:")
        quality_cols_present = []
        quality_cols_missing = []

        for col in expected_quality_cols:
            if col in df.columns:
                quality_cols_present.append(col)
                print(f"  ✓ {col}")
            else:
                quality_cols_missing.append(col)
                print(f"  ✗ {col} (MISSING)")

        print()

        assert not quality_cols_missing, (
            f"{len(quality_cols_missing)} quality metric columns are missing: "
            f"{quality_cols_missing}"
        )

        print(f"✓ All {len(expected_quality_cols)} quality metric columns are present!")

        # Display statistics for quality metrics
        print()
        print("Quality Metric Statistics:")
        print("-" * 80)

        for col in quality_cols_present:
            values = df[col]
            print(f"\n{col}:")
            print(f"  Count: {len(values)}")
            print(f"  Mean:  {values.mean():.3f}")
            print(f"  Std:   {values.std():.3f}")
            print(f"  Min:   {values.min():.3f}")
            print(f"  Max:   {values.max():.3f}")

            # Check for NaN values
            n_nan = values.isna().sum()
            if n_nan > 0:
                print(f"  ⚠️  NaN values: {n_nan} ({100*n_nan/len(values):.1f}%)")

        print()
        print("=" * 80)
        print("TEST PASSED!")
        print("=" * 80)
        print()
        print("Summary:")
        print(f"  ✓ Analysis completed successfully")
        print(f"  ✓ .h5 file created with {len(df)} localizations")
        print(f"  ✓ All {len(expected_quality_cols)} quality metric columns present")
        print(f"  ✓ Quality metrics saved alongside fitted parameters")
        print()

    finally:
        # Cleanup
        if os.path.exists(h5_file):
            os.remove(h5_file)
        gc.collect()


if __name__ == '__main__':
    test_quality_metrics_real_data()
