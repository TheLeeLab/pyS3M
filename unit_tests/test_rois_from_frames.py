"""SuperRes_Functions._rois_from_frames (fit_FRET_data / fit_QD_data) builds each ROI from a block of
frames at a time; its output must be bit-identical to building every ROI from whole-chunk stacks, as the
fitters did before (photoelectrons, smoothing and weights of the full chunk, then cropped)."""

import types

import numpy as np
import pytest
import tifffile

from pyS3M import sCMOSFunctions
from pyS3M.Constants import AnalysisConfig
from pyS3M.SR_Functions import SuperRes_Functions


@pytest.fixture
def setup(tmp_path):
    rng = np.random.default_rng(3)
    n, h, w = 9, 40, 44
    stack = rng.integers(90, 2000, size=(n, h, w)).astype(np.uint16)
    path = tmp_path / "stack.tif"
    tifffile.imwrite(path, stack)
    cal = dict(gain_map=rng.uniform(0.3, 0.6, (h, w)).astype(np.float32),
               offset_map=rng.uniform(95, 105, (h, w)).astype(np.float32),
               rqe=rng.uniform(0.9, 1.1, (h, w)).astype(np.float32),
               read_noise=rng.uniform(1.0, 3.0, (h, w)).astype(np.float32))
    masks = rng.integers(0, 2, (h, w, 3)).astype(np.float32)
    smoothing = types.SimpleNamespace(args={"sigma": 1.5}, extent=1.5, data_arg="image",
                                      smoothing_function=sCMOSFunctions.sCMOS_Functions().gaussian_filter_stack)
    bounds = {0: (0, 10, 0, 10), 1: (30, 40, 25, 35), 2: (17, 27, 30, 40), 3: (34, 44, 0, 10)}
    return path, stack, cal, masks, smoothing, bounds


def _whole_chunk(sr, path, items, bounds, cal, masks, smoothing):
    """The fitters' former construction: whole-chunk stacks, then crops."""
    frames = list(range(min(f for _, f in items), max(f for _, f in items) + 1))
    raw = sr.io.read_tiff(path, dtype="float32", frame=frames)
    pe = sr.io.convert_to_photoelectrons(raw, gain_map=cal["gain_map"], offset_map=cal["offset_map"], rqe=cal["rqe"])
    args = dict(smoothing.args)
    args[smoothing.data_arg] = pe
    sm = smoothing.smoothing_function(**args)
    weights = 1.0 / (cal["read_noise"] ** 2 + np.maximum(pe, 0) / cal["gain_map"])
    out = [[], [], [], [], [], []]
    for pid, f in items:
        xmin, xmax, ymin, ymax = bounds[pid]
        j = f - frames[0]
        out[0].append(pe[j, ymin:ymax, xmin:xmax].copy())
        out[1].append(sm[j, ymin:ymax, xmin:xmax].copy())
        out[2].append(masks[ymin:ymax, xmin:xmax, :].copy())
        out[3].append(weights[j, ymin:ymax, xmin:xmax].copy())
        out[4].append(np.broadcast_to(np.asarray(cal["read_noise"], dtype=np.float32), pe.shape[1:])
                      [ymin:ymax, xmin:xmax].copy())
        out[5].append((xmin, ymin))
    return out


@pytest.mark.parametrize("frame_block", [1, 2, 4, 50])
def test_block_built_rois_match_whole_chunk(setup, frame_block):
    path, stack, cal, masks, smoothing, bounds = setup
    sr = SuperRes_Functions(camera="ximea", config=AnalysisConfig(display=False))
    # punctum-major (as fit_FRET_data) with gaps, then frame-major (as fit_QD_data)
    for items in ([(2, 1), (2, 2), (2, 5), (0, 3), (0, 1), (3, 8), (1, 2), (1, 7)],
                  [(p, f) for f in range(2, 9) for p in range(4)]):
        ref = _whole_chunk(sr, path, items, bounds, cal, masks, smoothing)
        got = sr._rois_from_frames(path, items, bounds, stack.shape[1:], cal["gain_map"], cal["offset_map"],
                                   cal["rqe"], cal["read_noise"], smoothing, masks, frame_block=frame_block)
        for name, a_list, b_list in zip(["puncta", "smoothed", "masks", "weights", "readnoise"], ref[:5], got[:5]):
            assert len(a_list) == len(b_list) == len(items)
            for a, b in zip(a_list, b_list):
                assert a.dtype == b.dtype, name
                np.testing.assert_array_equal(a, b, err_msg=name)
        assert ref[5] == got[5]
