"""Integration test for quality metrics in the fitting pipeline.

This test verifies that quality metrics from spot detection are properly saved
in the final localization results, even when some ROIs are filtered out during
processing (e.g., spots too close to image edges).
"""
import sys
import os

import numpy as np
import tempfile
import shutil
import types


def test_quality_metrics_saved_with_roi_filtering():
    """Test that quality metrics are saved even when some ROIs are filtered."""

    print("Setting up test data...")

    # Import modules
    import pyS3M.IOFunctions as IOFunctions
    import pyS3M.sCMOSFunctions as sCMOSFunctions
    import pyS3M.SR_Functions as SR_Functions

    # Create temporary directory for test
    temp_dir = tempfile.mkdtemp(prefix='test_quality_metrics_')

    try:
        # Create synthetic test image with spots
        # Make it small but large enough for ROI_size=16
        width, height = 100, 100
        n_frames = 5

        # Create image stack with some bright spots
        # Place spots at various locations - some near edges (will be filtered),
        # some in center (will pass)
        image_stack = np.random.poisson(50, (n_frames, height, width)).astype(np.float32)

        # Add bright spots at known locations
        spot_locations = [
            (50, 50),   # Center - will pass
            (70, 70),   # Center-ish - will pass
            (5, 50),    # Near left edge - will be filtered (ROI_size=16 needs >8 pixels)
            (50, 95),   # Near right edge - will be filtered
            (30, 30),   # Center - will pass
        ]

        # Start from the Poisson background (not all-zero) -- the fitting
        # pipeline's _filter_fit_results rejects any fit with bg_R/bg_G/bg_B <= 0,
        # so an all-zero background causes every fit to be filtered out and no
        # .h5 file to be written, even though detection/fitting itself succeeds.
        bayer_image = image_stack.copy()
        for frame in range(n_frames):
            for y, x in spot_locations:
                # Add bright Gaussian spot
                yy, xx = np.meshgrid(
                    np.arange(max(0, y-5), min(height, y+5)),
                    np.arange(max(0, x-5), min(width, x+5)),
                    indexing='ij'
                )
                dist = np.sqrt((yy - y)**2 + (xx - x)**2)
                spot = 5000 * np.exp(-dist**2 / (2 * 1.5**2))

                # Apply to appropriate Bayer positions
                for dy in range(max(0, y-5), min(height, y+5)):
                    for dx in range(max(0, x-5), min(width, x+5)):
                        bayer_image[frame, dy, dx] += spot[dy - max(0, y-5), dx - max(0, x-5)]

        # Save as TIFF
        io = IOFunctions.IO_Functions()
        test_tif = os.path.join(temp_dir, 'test_data_001.tif')
        io.write_tiff(bayer_image.astype(np.uint16), test_tif)

        # _fit_files calls load_metadata_roi(..., use_fallback=False) unconditionally,
        # so a minimal ImageJ/MicroManager-style metadata sidecar is required even
        # though this synthetic image has no real ROI offset (full-frame: 0-0-width-height).
        metadata_path = os.path.join(temp_dir, 'test_data_001_metadata.txt')
        with open(metadata_path, 'w') as f:
            f.write(
                '{\n'
                '  "FrameKey-0-0-0": {\n'
                f'    "ROI": "0-0-{width}-{height}"\n'
                '  }\n'
                '}\n'
            )

        print(f"Created test TIFF: {test_tif}")
        print(f"Image shape: {bayer_image.shape}")
        print(f"Expected spots at: {spot_locations}")
        print(f"Expected ~3 valid spots after edge filtering (ROI_size=16)")

        # Set up camera parameters (simple defaults)
        gain = np.ones((height, width), dtype=np.float32)
        offset = np.zeros((height, width), dtype=np.float32)
        variance = 10.0 * np.ones((height, width), dtype=np.float32)
        readnoise = 10.0 * np.ones((height, width), dtype=np.float32)
        rqe = np.ones((height, width), dtype=np.float32)

        # Set up smoothing function (matching single_folder_analysis.py)
        scmos = sCMOSFunctions.sCMOS_Functions()
        smoothing_function = types.SimpleNamespace()
        smoothing_function.args = {"sigma": 1.5}
        smoothing_function.extent = 1.5
        smoothing_function.smoothing_function = scmos.gaussian_filter_stack
        smoothing_function.data_arg = "image"

        # Run fitting with quality metrics enabled
        print("\nRunning fit_SM_data with quality metrics...")
        supres = SR_Functions.SuperRes_Functions()

        supres.fit_SM_data(
            temp_dir,
            smoothing_function,
            gain,
            offset,
            rqe,
            readnoise,
            variance=variance,
            pfa=1e-4,
            ROI_size=16,
            peak_wavelength=0.55,
            NA=1.49,
            pixel_size=0.069,
            image_type=".tif",
            use_variance_aware_demosaic=False,
        )

        # Check that .h5 file was created
        h5_files = [f for f in os.listdir(temp_dir) if f.endswith('.h5')]
        print(f"\nCreated .h5 files: {h5_files}")

        assert h5_files, "Expected an .h5 file to be created (spots should have been detected and fitted)"

        # Read the results
        import pandas as pd
        h5_file = os.path.join(temp_dir, h5_files[0])
        results = pd.read_hdf(h5_file)

        print(f"\nResults shape: {results.shape}")
        print(f"Columns: {list(results.columns)}")

        assert len(results) > 0, "Expected at least one fitted localisation after ROI filtering"

        # Check for quality metrics columns
        quality_metric_cols = [col for col in results.columns if col.startswith('spot_')]
        assert quality_metric_cols, (
            f"Expected quality metric columns (prefixed 'spot_') in results, "
            f"got columns: {list(results.columns)}"
        )

        print(f"\n✓ Quality metrics found in results: {quality_metric_cols}")
        for col in quality_metric_cols:
            print(f"  {col}: {len(results[col])} values")
            print(f"    Range: [{results[col].min():.2f}, {results[col].max():.2f}]")

        # Verify all quality metrics have same length as fit results
        for col in quality_metric_cols:
            assert len(results[col]) == len(results), \
                f"Quality metric '{col}' length mismatch: {len(results[col])} vs {len(results)}"

        print("\n✓ All quality metrics have correct length matching fit results")
        print("✓ Quality metrics successfully saved despite ROI filtering!")

    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"\nCleaned up temp directory: {temp_dir}")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Quality Metrics Integration")
    print("=" * 60)
    print()

    test_quality_metrics_saved_with_roi_filtering()

    print()
    print("=" * 60)
    print("Test completed!")
    print("=" * 60)
