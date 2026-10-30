"""Tests for PlottingBase.add_bayer_key / bayer_pixel_type: the key shows the filters under the
displayed top-left corner of an image, for either imshow origin."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_hex

from pyS3M.PlottingBase import BAYER_COLOURS, add_bayer_key, bayer_pixel_type


def _masks(H=6, W=6):
    # BGGR: B at (even row, even col), R at (odd, odd), G elsewhere
    r, c = np.indices((H, W))
    B = (r % 2 == 0) & (c % 2 == 0)
    R = (r % 2 == 1) & (c % 2 == 1)
    return {"B": B, "G": ~(B | R), "R": R}


def test_pixel_type_follows_pixel_order():
    pt = bayer_pixel_type(_masks())
    assert pt[0, 0] == 0 and pt[0, 1] == 1 and pt[1, 1] == 2
    assert bayer_pixel_type(_masks(), pixel_order=("R", "G", "B"))[0, 0] == 2


@pytest.mark.parametrize("origin, rows", [("upper", [0, 1]), ("lower", [4, 5])])
def test_key_covers_displayed_top_left_with_its_filters(origin, rows):
    pt = bayer_pixel_type(_masks())
    fig, ax = plt.subplots()
    ax.imshow(np.zeros(pt.shape), origin=origin)
    add_bayer_key(ax, pt, origin=origin)
    cells = [p for p in ax.patches if p.get_fill()]
    assert len(cells) == 4
    for p in cells:
        col, row = int(round(p.get_x() + 0.5)), int(round(p.get_y() + 0.5))
        assert row in rows and col in (0, 1)
        assert to_hex(p.get_facecolor()) == to_hex(BAYER_COLOURS[pt[row, col]])
    # the displayed top-left pixel is under the key
    top_row = 0 if origin == "upper" else pt.shape[0] - 1
    assert any(int(round(p.get_y() + 0.5)) == top_row and int(round(p.get_x() + 0.5)) == 0 for p in cells)
    plt.close(fig)


def test_bad_origin_raises():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        add_bayer_key(ax, np.zeros((4, 4), int), origin="left")
    plt.close(fig)


def test_key_from_masks_at_roi_position():
    # a 4 x 4 crop of full-frame masks starting at an odd row and column: its top-left pixel is R
    full = _masks(10, 10)
    fig, ax = plt.subplots()
    ax.imshow(np.zeros((4, 4)), origin="upper")
    add_bayer_key(ax, full, row_min=3, col_min=5)
    cells = {(int(round(p.get_y() + 0.5)), int(round(p.get_x() + 0.5))): to_hex(p.get_facecolor())
             for p in ax.patches if p.get_fill()}
    pt = bayer_pixel_type(full)
    for (row, col), colour in cells.items():
        assert colour == to_hex(BAYER_COLOURS[pt[3 + row, 5 + col]])
    assert cells[(0, 0)] == to_hex(BAYER_COLOURS[2])
    plt.close(fig)


def test_origin_defaults_to_the_image():
    fig, ax = plt.subplots()
    ax.imshow(np.zeros((6, 6)), origin="lower")
    add_bayer_key(ax, _masks())
    assert {int(round(p.get_y() + 0.5)) for p in ax.patches if p.get_fill()} == {4, 5}
    plt.close(fig)


def test_masks_not_covering_the_image_raise():
    fig, ax = plt.subplots()
    ax.imshow(np.zeros((6, 6)), origin="upper")
    with pytest.raises(ValueError):
        add_bayer_key(ax, _masks(6, 6), row_min=2)  # crop runs off the masks
    with pytest.raises(ValueError):
        add_bayer_key(ax, _masks(6, 6), origin="lower")  # not how the image is shown
    plt.close(fig)
