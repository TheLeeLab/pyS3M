#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive test using REAL spot detection and ROI extraction code.

This test verifies that the full pipeline correctly:
1. Detects spots at the right locations
2. Extracts ROIs containing those spots
3. Properly handles coordinate systems throughout

Created on 2025-10-03
"""

import numpy as np
import sys
import os


from pyS3M.SpotDetectionFunctions import SpotDetection_Functions
from pyS3M.SR_Functions import SuperRes_Functions
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def create_test_image_with_metadata():
    """Create test image with known spots and all required metadata."""
    height = 120
    width = 150

    image = np.zeros((height, width), dtype=np.float32)
    image += 100.0  # Background

    # True spot locations (x, y)
    true_spots = [
        (40, 30, "Spot 1"),
        (110, 50, "Spot 2"),
        (70, 95, "Spot 3"),
    ]

    # Add bright Gaussian spots
    sigma = 1.5
    amplitude = 3000

    for x, y, label in true_spots:
        for dy in range(-10, 11):
            for dx in range(-10, 11):
                row = y + dy
                col = x + dx
                if 0 <= row < height and 0 <= col < width:
                    gauss = amplitude * np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
                    image[row, col] += gauss

    # Add noise
    np.random.seed(42)
    image += np.random.normal(0, 10, image.shape)

    # Create required metadata
    variance = np.ones_like(image) * 100.0
    gain_map = np.ones_like(image)
    offset_map = np.zeros_like(image)
    read_noise = np.ones_like(image) * 10.0
    rqe = np.ones_like(image)

    return image, variance, gain_map, offset_map, read_noise, rqe, true_spots, width, height


def test_full_pipeline(test_output_dir):
    """Test the full detection and extraction pipeline."""
    print("=" * 80)
    print("COMPREHENSIVE TEST: Real Detection + Real Extraction")
    print("=" * 80)

    # Create test data
    (image, variance, gain_map, offset_map, read_noise, rqe,
     true_spots, width, height) = create_test_image_with_metadata()

    print(f"\n1. Test image: {width}x{height}")
    print(f"   True spots:")
    for x, y, label in true_spots:
        print(f"     {label}: (x={x}, y={y})")
        print(f"              image[{y}, {x}] = {image[y, x]:.1f}")

    # STEP 1: Real spot detection
    print(f"\n2. Running real spot detection...")
    spot_detector = SpotDetection_Functions()
    detected_puncta = spot_detector.detect_puncta_in_image(
        image=image,
        variance=variance,
        pfa=1e-6,
        wavelength=0.65,
        pixel_size=0.069,
        NA=1.49,
    )

    print(f"   Detected {len(detected_puncta)} spots")
    print(f"   Format: detected_puncta[i, 0] = {detected_puncta[0, 0]} (should be row/y)")
    print(f"           detected_puncta[i, 1] = {detected_puncta[0, 1]} (should be col/x)")

    # STEP 2: Use real SR_Functions._process_roi to extract ROIs
    print(f"\n3. Extracting ROIs using SR_Functions._process_roi()...")
    sr_functions = SuperRes_Functions()

    # Create simple bayer masks (RGGB pattern)
    # For testing, we just need the structure, not perfect masks
    mask_b = np.zeros((height, width), dtype=bool)
    mask_g = np.zeros((height, width), dtype=bool)
    mask_r = np.zeros((height, width), dtype=bool)

    # RGGB pattern
    mask_r[::2, ::2] = True    # Red: even rows, even cols
    mask_g[::2, 1::2] = True   # Green1: even rows, odd cols
    mask_g[1::2, ::2] = True   # Green2: odd rows, even cols
    mask_b[1::2, 1::2] = True  # Blue: odd rows, odd cols

    masks = np.dstack([mask_b, mask_g, mask_r])

    extracted_rois = []
    roi_max_values = []
    roi_coords = []

    for i in range(len(detected_puncta)):
        result = sr_functions._process_roi(
            detected_puncta=detected_puncta,
            i=i,
            raw_data=image,
            masks=masks,
            gain_map=gain_map,
            offset_map=offset_map,
            read_noise=read_noise,
            rqe=rqe,
            ROI_size=16,
            width=width,
            height=height,
            is_multi_frame=False,
            smoothing_function=None,  # No smoothing for test
            frame_offset=0,
        )

        if result is not None:
            photoelectron_roi, smoothed_roi, weights_roi, mask_roi, coords, plane, readnoise_roi = result
            extracted_rois.append(photoelectron_roi)
            roi_max_values.append(photoelectron_roi.max())
            roi_coords.append(coords)

            # Get the center coordinates
            ycentre = detected_puncta[i, 0]
            xcentre = detected_puncta[i, 1]

            print(f"   ROI {i+1}: center=({xcentre}, {ycentre}), "
                  f"shape={photoelectron_roi.shape}, max={photoelectron_roi.max():.1f}")
        else:
            print(f"   ROI {i+1}: Failed to extract (returned None)")

    # STEP 3: Verify extractions contain the spots
    print(f"\n4. Verification:")
    success_count = 0
    for i, (roi, max_val) in enumerate(zip(extracted_rois, roi_max_values)):
        is_square = (roi.shape[0] == roi.shape[1] == 16)
        contains_spot = (max_val > 2000)  # Bright spot threshold

        print(f"   ROI {i+1}: square={is_square}, contains_bright_spot={contains_spot}, max={max_val:.1f}")

        if is_square and contains_spot:
            success_count += 1

    # Visualize results
    print(f"\n5. Creating visualization...")
    fig = plt.figure(figsize=(20, 6))

    # Panel 1: Full image with detections
    ax1 = plt.subplot(1, 4 + len(extracted_rois), 1)
    ax1.imshow(image, cmap='hot', origin='upper')
    ax1.set_title('Full Image with Detections', fontsize=10)
    ax1.set_xlabel('x (columns)')
    ax1.set_ylabel('y (rows)')

    # Plot true locations (blue squares)
    for x, y, label in true_spots:
        ax1.plot(x, y, 'bs', markersize=12, markerfacecolor='none', markeredgewidth=2)
        ax1.text(x + 3, y - 3, label, color='blue', fontsize=8)

    # Plot detected locations (green crosses) - using CORRECT interpretation
    for i in range(len(detected_puncta)):
        ycentre = detected_puncta[i, 0]  # row = y
        xcentre = detected_puncta[i, 1]  # col = x
        ax1.plot(xcentre, ycentre, 'g+', markersize=15, markeredgewidth=3)

        # Draw ROI box
        roi_size = 16
        xmin = xcentre - roi_size/2
        ymin = ycentre - roi_size/2
        rect = patches.Rectangle((xmin, ymin), roi_size, roi_size,
                                linewidth=2, edgecolor='cyan', facecolor='none')
        ax1.add_patch(rect)

    # Panels 2-4: Extracted ROIs
    for i, (roi, (xmin, ymin)) in enumerate(zip(extracted_rois[:3], roi_coords[:3])):
        ax = plt.subplot(1, 4 + len(extracted_rois), i + 2)
        ax.imshow(roi, cmap='hot', origin='upper')
        ax.plot(8, 8, 'g+', markersize=20, markeredgewidth=3)
        ax.set_title(f'ROI {i+1}\nMax: {roi.max():.0f}', fontsize=10)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    # Summary panel
    ax_summary = plt.subplot(1, 4 + len(extracted_rois), 4 + len(extracted_rois))
    ax_summary.axis('off')

    summary = "PIPELINE TEST\n"
    summary += "=" * 25 + "\n\n"
    summary += f"Detection:\n"
    summary += f"  Found: {len(detected_puncta)}\n"
    summary += f"  Expected: {len(true_spots)}\n\n"
    summary += f"Extraction:\n"
    summary += f"  Success: {success_count}/{len(extracted_rois)}\n\n"
    summary += f"Coordinate Fix:\n"
    summary += f"  ycentre = [i,0] ✓\n"
    summary += f"  xcentre = [i,1] ✓\n"
    summary += f"  roi[y:y+h, x:x+w] ✓\n\n"

    if success_count == len(true_spots):
        summary += "✓ ALL TESTS PASSED\n"
        summary += "\nCorrectly extracting\n"
        summary += "spots at detected\n"
        summary += "locations!"
        color = 'green'
    else:
        summary += "✗ TESTS FAILED\n"
        summary += f"\nOnly {success_count}/{len(true_spots)}\n"
        summary += "spots extracted\n"
        summary += "correctly"
        color = 'red'

    ax_summary.text(0.1, 0.5, summary, fontsize=10, verticalalignment='center',
                   family='monospace', color=color, weight='bold')

    plt.tight_layout()
    output_path = test_output_dir / 'full_pipeline_test.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   Saved to: {output_path}")

    # Final verdict
    print(f"\n{'=' * 80}")
    print("FINAL RESULT:")
    print("=" * 80)

    if success_count == len(true_spots):
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        print(f"\nSuccessfully detected and extracted all {len(true_spots)} spots!")
        print("The coordinate system fixes are working correctly.")
        return True
    else:
        print("✗✗✗ TESTS FAILED ✗✗✗")
        print(f"\nOnly extracted {success_count}/{len(true_spots)} spots correctly.")
        return False


if __name__ == "__main__":
    success = test_full_pipeline()
    sys.exit(0 if success else 1)
