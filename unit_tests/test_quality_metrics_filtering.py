"""Test that quality metrics are correctly filtered when ROIs are removed during processing.

This test verifies the fix for the bug where quality metrics weren't being saved
because their length didn't match the final fit results (due to some ROIs being
filtered out for being too close to image edges).
"""
import sys
import os
import types

import numpy as np
from pyS3M.SR_Functions import SuperRes_Functions


def _identity_smoothing():
    """Identity smoothing_function object matching IOFunctions.apply_smoothing's
    expected interface (.args/.data_arg/.smoothing_function) -- a bare callable
    isn't enough since apply_smoothing does
    `smoothing_function.smoothing_function(**{**smoothing_function.args,
    smoothing_function.data_arg: data})` internally, not `smoothing_function(data)`.
    """
    sf = types.SimpleNamespace()
    sf.args = {}
    sf.data_arg = 'image'
    sf.smoothing_function = lambda image: image
    return sf


def test_quality_metrics_filtering():
    """Test that quality metrics are filtered when ROIs are removed."""

    # Create a mock SuperRes_Functions instance
    sr = SuperRes_Functions()

    # Create test data
    width, height = 100, 100
    ROI_size = 7

    # Create detected puncta - some will be too close to edges and filtered out
    # Format: [row, col, frame]
    detected_puncta = np.array([
        [50, 50, 0],  # Valid - center of image
        [2, 50, 0],   # Invalid - too close to top edge (ROI_size=7, needs 3 pixels)
        [50, 98, 0],  # Invalid - too close to right edge
        [70, 70, 0],  # Valid - far from edges
        [50, 2, 0],   # Invalid - too close to left edge
    ])

    # Create quality metrics dict with same length as detected_puncta
    quality_metrics = {
        'snr': np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
        'peak_intensity': np.array([100, 200, 300, 400, 500]),
        'background': np.array([5.0, 10.0, 15.0, 20.0, 25.0]),
    }

    # Create mock raw data
    raw_data = np.random.rand(height, width) * 100

    # Create mock masks (Bayer pattern)
    masks = np.zeros((height, width, 3))
    masks[::2, ::2, 0] = 1    # R
    masks[::2, 1::2, 1] = 1   # G
    masks[1::2, ::2, 1] = 1   # G
    masks[1::2, 1::2, 2] = 1  # B

    # Identity smoothing function (for testing)
    smoothing_function = _identity_smoothing()

    # Read noise
    read_noise = 10.0

    # Call _process_detected_puncta_batch with quality_metrics
    results = sr._process_detected_puncta_batch(
        raw_data=raw_data,
        detected_puncta=detected_puncta,
        width=width,
        height=height,
        ROI_size=ROI_size,
        smoothing_function=smoothing_function,
        read_noise=read_noise,
        masks=masks,
        gain_map=1.0,
        offset_map=0.0,
        rqe=1.0,
        quality_metrics=quality_metrics,
    )

    (puncta_tofit, smoothed_puncta_tofit, masks_tofit, weights_tofit,
     relative_coords, planes, filtered_quality_metrics, readnoise_tofit) = results

    # Check that we got 2 valid ROIs (indices 0 and 3)
    assert len(puncta_tofit) == 2, f"Expected 2 valid ROIs, got {len(puncta_tofit)}"

    # Check that filtered quality metrics match the number of valid ROIs
    if filtered_quality_metrics is not None:
        for key, values in filtered_quality_metrics.items():
            assert len(values) == 2, f"Expected quality metric '{key}' to have 2 values, got {len(values)}"

            # Check that the correct values were kept (indices 0 and 3)
            expected_values = quality_metrics[key][[0, 3]]
            np.testing.assert_array_equal(
                values, expected_values,
                err_msg=f"Quality metric '{key}' has incorrect values after filtering"
            )

        print("✓ Quality metrics correctly filtered to match processed ROIs")
        print(f"  Original detections: {len(detected_puncta)}")
        print(f"  Valid ROIs: {len(puncta_tofit)}")
        print(f"  Filtered quality metrics: {len(next(iter(filtered_quality_metrics.values())))}")
        print(f"  SNR values: {quality_metrics['snr']} -> {filtered_quality_metrics['snr']}")
    else:
        raise AssertionError("filtered_quality_metrics should not be None")


def test_no_quality_metrics():
    """Test that the function still works when no quality metrics are provided."""

    sr = SuperRes_Functions()

    width, height = 100, 100
    ROI_size = 7

    detected_puncta = np.array([
        [50, 50, 0],  # Valid
        [2, 50, 0],   # Invalid - too close to edge
    ])

    raw_data = np.random.rand(height, width) * 100

    masks = np.zeros((height, width, 3))
    masks[::2, ::2, 0] = 1
    masks[::2, 1::2, 1] = 1
    masks[1::2, ::2, 1] = 1
    masks[1::2, 1::2, 2] = 1

    smoothing_function = _identity_smoothing()

    # Call WITHOUT quality_metrics
    results = sr._process_detected_puncta_batch(
        raw_data=raw_data,
        detected_puncta=detected_puncta,
        width=width,
        height=height,
        ROI_size=ROI_size,
        smoothing_function=smoothing_function,
        read_noise=10.0,
        masks=masks,
        gain_map=1.0,
        offset_map=0.0,
        rqe=1.0,
        quality_metrics=None,  # No quality metrics
    )

    (puncta_tofit, smoothed_puncta_tofit, masks_tofit, weights_tofit,
     relative_coords, planes, filtered_quality_metrics, readnoise_tofit) = results

    # Check that we got 1 valid ROI
    assert len(puncta_tofit) == 1, f"Expected 1 valid ROI, got {len(puncta_tofit)}"

    # Check that filtered_quality_metrics is None when not provided
    assert filtered_quality_metrics is None, "filtered_quality_metrics should be None when not provided"

    print("✓ Function works correctly without quality metrics")


if __name__ == "__main__":
    print("Testing quality metrics filtering...")
    print()

    print("Test 1: Quality metrics filtering")
    test_quality_metrics_filtering()
    print()

    print("Test 2: No quality metrics provided")
    test_no_quality_metrics()
    print()

    print("All tests passed!")
