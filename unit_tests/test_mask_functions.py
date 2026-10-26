"""Full coverage tests for pyS3M.MaskFunctions -- Bayer/colour mask generation
(`get_masks`, `get_ROI_mask`, `get_stacked_masks`).

Small hand-built mosaic units and image sizes throughout (pure numeric code,
no I/O or database dependency).
"""
from __future__ import annotations

import numpy as np
import pytest

import pyS3M.MaskFunctions as MaskFunctions

Mask_Functions = MaskFunctions.Mask_Functions


class TestInit:
    def test_default_ximea(self):
        mf = Mask_Functions()
        assert mf.mosaic_unit.shape == (2, 2)
        assert set(np.unique(mf.mosaic_unit)) == {"B", "G", "R"}

    def test_zwo_camera(self):
        mf = Mask_Functions(camera="zwo")
        assert mf.mosaic_unit.shape == (2, 2)

    def test_explicit_mosaic_unit_overrides_camera(self):
        custom = np.array([["R", "G"], ["G", "B"]])
        mf = Mask_Functions(mosaic_unit=custom)
        np.testing.assert_array_equal(mf.mosaic_unit, custom)


class TestGetMasks:
    def test_tiled_branch_even_size(self):
        mf = Mask_Functions()
        masks = mf.get_masks(size_x=8, size_y=8)
        assert set(masks.keys()) == {"B", "G", "R"}
        for m in masks.values():
            assert m.shape == (8, 8)
            assert m.dtype == bool
        # Every pixel belongs to exactly one colour.
        stacked = np.stack(list(masks.values()))
        np.testing.assert_array_equal(stacked.sum(axis=0), np.ones((8, 8)))

    def test_tiled_branch_odd_size(self):
        mf = Mask_Functions()
        masks = mf.get_masks(size_x=7, size_y=9)
        for m in masks.values():
            assert m.shape == (7, 9)

    def test_exact_mosaic_shape_branch(self):
        mf = Mask_Functions()
        # size_x/size_y matching mosaic_unit.shape exactly triggers the
        # non-tiled equality branch.
        masks = mf.get_masks(size_x=2, size_y=2)
        for colour in ["B", "G", "R"]:
            assert masks[colour].shape == (2, 2)
        # B is at [0,0], R at [1,1], G at [0,1] and [1,0] for ximea BGGR.
        assert masks["B"][0, 0]
        assert masks["R"][1, 1]

    def test_custom_mosaic_unit_argument_overrides_instance_default(self):
        mf = Mask_Functions()
        custom = np.array([["W", "W"], ["W", "W"]])
        masks = mf.get_masks(size_x=4, size_y=4, mosaic_unit=custom)
        assert set(masks.keys()) == {"W"}
        assert masks["W"].all()

    def test_instance_default_used_when_arg_is_none(self):
        custom = np.array([["R", "G"], ["G", "B"]])
        mf = Mask_Functions(mosaic_unit=custom)
        masks = mf.get_masks(size_x=4, size_y=4)
        assert set(masks.keys()) == {"B", "G", "R"}


class TestGetROIMask:
    def test_full_frame_roi_matches_get_masks(self):
        mf = Mask_Functions()
        roi_masks = mf.get_ROI_mask(ROI_x_start=0, ROI_y_start=0, width=8, height=8)
        full_masks = mf.get_masks(size_x=8, size_y=8)
        for colour in roi_masks:
            np.testing.assert_array_equal(roi_masks[colour], full_masks[colour])

    def test_offset_roi_crops_correctly(self):
        mf = Mask_Functions()
        full_masks = mf.get_masks(size_x=10, size_y=10)
        roi_masks = mf.get_ROI_mask(ROI_x_start=2, ROI_y_start=3, width=4, height=5)
        for colour in roi_masks:
            assert roi_masks[colour].shape == (5, 4)
            np.testing.assert_array_equal(
                roi_masks[colour], full_masks[colour][3:8, 2:6]
            )

    def test_explicit_mosaic_unit_argument(self):
        mf = Mask_Functions()
        custom = np.array([["W", "W"], ["W", "W"]])
        roi_masks = mf.get_ROI_mask(
            ROI_x_start=0, ROI_y_start=0, width=4, height=4, mosaic_unit=custom
        )
        assert set(roi_masks.keys()) == {"W"}


class TestGetStackedMasks:
    def test_stack_shape_matches_colour_count(self):
        mf = Mask_Functions()
        stacked = mf.get_stacked_masks(ROI_x_start=0, ROI_y_start=0, width=6, height=6)
        assert stacked.shape == (6, 6, 3)
        assert stacked.dtype == bool

    def test_explicit_mosaic_unit_argument(self):
        mf = Mask_Functions()
        custom = np.array([["W", "W"], ["W", "W"]])
        stacked = mf.get_stacked_masks(
            ROI_x_start=0, ROI_y_start=0, width=4, height=4, mosaic_unit=custom
        )
        assert stacked.shape == (4, 4, 1)
        assert stacked.all()

    def test_instance_default_used_when_arg_is_none(self):
        custom = np.array([["R", "G"], ["G", "B"]])
        mf = Mask_Functions(mosaic_unit=custom)
        stacked = mf.get_stacked_masks(ROI_x_start=0, ROI_y_start=0, width=4, height=4)
        assert stacked.shape == (4, 4, 3)


class TestPixelFractions:
    @pytest.mark.parametrize("camera", ["ximea", "zwo"])
    def test_bayer_fractions(self, camera):
        assert Mask_Functions(camera=camera).pixel_fractions() == {"B": 0.25, "G": 0.5, "R": 0.25}

    def test_keys_follow_get_masks_order(self):
        mf = Mask_Functions(mosaic_unit=np.array([["R", "G"], ["G", "B"]]))
        assert list(mf.pixel_fractions()) == list(mf.get_masks(4, 4))

    def test_fractions_match_tiled_masks(self):
        custom = np.array([["G", "B", "G"], ["R", "G", "R"], ["G", "B", "G"]])
        mf = Mask_Functions(mosaic_unit=custom)
        masks = mf.get_masks(3, 3)
        fractions = mf.pixel_fractions()
        for colour, mask in masks.items():
            assert fractions[colour] == pytest.approx(mask.mean())
        assert sum(fractions.values()) == pytest.approx(1.0)

    def test_explicit_mosaic_unit_argument(self):
        mf = Mask_Functions()
        assert mf.pixel_fractions(np.array([["W", "W"], ["W", "W"]])) == {"W": 1.0}
