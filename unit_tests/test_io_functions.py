"""Full coverage tests for pyS3M.IOFunctions -- file I/O for microscopy data
(HDF5 localisation databases, ImageJ/Thorlabs JSON metadata, TIFF stacks,
photoelectron conversion, simulation-result CSVs).

Tiny synthetic data throughout (small arrays, small HDF5/JSON/TIFF files built
directly in `tmp_path`), no large fixtures. `monkeypatch` is used to reach
branches no small real file can reach on its own: forced I/O exceptions
(corrupted TIFF pages, memmap failures, encoding errors), a >10MB file-size
gate, and HDF5 schema-mismatch dtype coercion paths.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import pytest
import tifffile

import pyS3M.IOFunctions as iof
from pyS3M.IOFunctions import IO_Functions


@pytest.fixture
def io():
    return IO_Functions()


def _loc_df(n=5, with_colour=True, with_frame=True):
    rng = np.random.default_rng(0)
    data = {}
    if with_frame:
        data["frame"] = np.arange(n)
    data["xc"] = rng.uniform(0, 40, n)
    data["yc"] = rng.uniform(0, 40, n)
    if with_colour:
        data["A_B"] = rng.uniform(100, 200, n)
        data["A_G"] = rng.uniform(100, 200, n)
        data["A_R"] = rng.uniform(100, 200, n)
        data["A_B_err"] = rng.uniform(1, 5, n)
        data["A_G_err"] = rng.uniform(1, 5, n)
        data["A_R_err"] = rng.uniform(1, 5, n)
        data["bg_B"] = rng.uniform(10, 20, n)
        data["bg_G"] = rng.uniform(10, 20, n)
        data["bg_R"] = rng.uniform(10, 20, n)
        data["bg_B_err"] = rng.uniform(0.1, 1, n)
        data["bg_G_err"] = rng.uniform(0.1, 1, n)
        data["bg_R_err"] = rng.uniform(0.1, 1, n)
    return pd.DataFrame(data)


def _imagej_metadata(roi="10-20-40-30", exposure_ms=None, n_frames=8):
    frame_entry = {"ROI": roi}
    if exposure_ms is not None:
        frame_entry["Exposure-ms"] = exposure_ms
    return {
        "FrameKey-0-0-0": frame_entry,
        "Summary": {"IntendedDimensions": {"time": n_frames}},
    }


class _FakeStat:
    def __init__(self, st_size):
        self.st_size = st_size


# ======================================================================
# _normalize_color_channels / _add_photon_columns
# ======================================================================

class TestNormalizeColorChannels:
    def test_normalizes_by_total(self, io):
        df = pd.DataFrame({"total": [10.0, 0.0], "a": [5.0, 3.0], "a_err": [1.0, 0.5]})
        out = io._normalize_color_channels(df, "total", ["a"])
        assert out["a"].iloc[0] == pytest.approx(0.5)
        # errors untouched: the fitter already reports errors of the normalised fractions
        assert out["a_err"].iloc[0] == pytest.approx(1.0)
        # zero-total row untouched (division-by-zero mask)
        assert out["a"].iloc[1] == pytest.approx(3.0)


class TestAddPhotonColumns:
    def test_adds_and_normalises_photon_columns(self, io):
        df = _loc_df(4)
        out = io._add_photon_columns(df, normalise=True)
        assert "photons" in out.columns
        assert "background_photons" in out.columns
        np.testing.assert_allclose(out["A_B"] + out["A_G"] + out["A_R"], 1.0)

    def test_no_normalise_keeps_raw_amplitudes(self, io):
        df = _loc_df(4)
        raw_sum = (df["A_B"] + df["A_G"] + df["A_R"]).to_numpy()
        out = io._add_photon_columns(df, normalise=False)
        np.testing.assert_allclose(out["photons"].to_numpy(), raw_sum)
        np.testing.assert_allclose(out["A_B"].to_numpy(), df["A_B"].to_numpy())

    def test_missing_colour_columns_skips_photon_columns(self, io):
        df = _loc_df(4, with_colour=False)
        out = io._add_photon_columns(df)
        assert "photons" not in out.columns
        assert "background_photons" not in out.columns
        assert "photoelectrons" not in out.columns

    @pytest.mark.parametrize("normalise", [True, False])
    def test_photoelectrons_weight_amplitudes_by_pixel_fraction(self, io, normalise):
        df = _loc_df(4)
        out = io._add_photon_columns(df, normalise=normalise)
        expected = df["A_B"] / 4 + df["A_G"] / 2 + df["A_R"] / 4
        np.testing.assert_allclose(out["photoelectrons"].to_numpy(), expected.to_numpy())
        expected_bg = df["bg_B"] / 4 + df["bg_G"] / 2 + df["bg_R"] / 4
        np.testing.assert_allclose(out["background_photoelectrons"].to_numpy(), expected_bg.to_numpy())


class TestAddPhotoelectronColumns:
    def test_matches_raw_amplitudes_after_normalisation(self, io):
        df = _loc_df(4)
        saved = io._add_photon_columns(df).drop(columns=["photoelectrons", "background_photoelectrons"])
        out = io.add_photoelectron_columns(saved)
        np.testing.assert_allclose(out["photoelectrons"].to_numpy(),
                                   (df["A_B"] / 4 + df["A_G"] / 2 + df["A_R"] / 4).to_numpy())
        np.testing.assert_allclose(out["background_photoelectrons"].to_numpy(),
                                   (df["bg_B"] / 4 + df["bg_G"] / 2 + df["bg_R"] / 4).to_numpy())

    def test_fractions_follow_mosaic_unit(self):
        io = IO_Functions(mosaic_unit=np.array([["B", "B"], ["G", "R"]]))
        df = pd.DataFrame({"photons": [100.0], "A_B": [0.2], "A_G": [0.3], "A_R": [0.5]})
        out = io.add_photoelectron_columns(df)
        assert out["photoelectrons"].iloc[0] == pytest.approx(100 * (0.2 / 2 + 0.3 / 4 + 0.5 / 4))

    def test_equals_signal_collected_on_the_mosaic(self, io):
        # A unit-integral spot of amplitude A_p on each pixel type: the image summed over the
        # sensor is the detected signal, sum_p fraction_p * A_p.
        from pyS3M.MaskFunctions import Mask_Functions
        masks = Mask_Functions().get_stacked_masks(0, 0, 64, 64)
        y, x = np.mgrid[:64, :64]
        spot = np.exp(-((x - 31.7) ** 2 + (y - 32.2) ** 2) / (2 * 1.5 ** 2)) / (2 * np.pi * 1.5 ** 2)
        amplitudes = np.array([300.0, 1200.0, 500.0])   # A_B, A_G, A_R
        image = (masks * amplitudes).sum(axis=2) * spot
        df = pd.DataFrame({f"A_{c}": [a] for c, a in zip("BGR", amplitudes)} | {f"A_{c}_err": [0.0] for c in "BGR"})
        out = io._add_photon_columns(df)
        assert out["photoelectrons"].iloc[0] == pytest.approx(image.sum(), rel=1e-3)
        assert out["photons"].iloc[0] > 2 * image.sum()

    def test_missing_columns_add_nothing(self, io):
        out = io.add_photoelectron_columns(pd.DataFrame({"photons": [1.0]}))
        assert list(out.columns) == ["photons"]


# ======================================================================
# read_h5_database / write_h5_database
# ======================================================================

class TestReadWriteH5Database:
    def test_write_then_read_roundtrip(self, io, tmp_path):
        df = _loc_df(6)
        path = tmp_path / "locs.h5"
        io.write_h5_database(df, path)
        out = io.read_h5_database(path)
        assert len(out) == 6
        assert "photons" in out.columns

    def test_empty_dataframe_writes_nothing(self, io, tmp_path):
        df = pd.DataFrame({"frame": [], "xc": []})
        path = tmp_path / "empty.h5"
        io.write_h5_database(df, path)
        assert not path.exists()

    def test_drops_all_nan_rows(self, io, tmp_path):
        df = _loc_df(4)
        df.loc[0, :] = np.nan
        path = tmp_path / "locs.h5"
        io.write_h5_database(df, path)
        out = io.read_h5_database(path)
        assert len(out) == 3

    def test_no_frame_column_is_tolerated(self, io, tmp_path):
        df = _loc_df(4, with_frame=False)
        path = tmp_path / "locs.h5"
        io.write_h5_database(df, path)
        out = io.read_h5_database(path)
        assert "frame" not in out.columns

    def test_photons_column_already_present_skips_recompute(self, io, tmp_path):
        df = _loc_df(3)
        df["photons"] = 999.0
        path = tmp_path / "locs.h5"
        io.write_h5_database(df, path)
        out = io.read_h5_database(path)
        np.testing.assert_allclose(out["photons"].to_numpy(), 999.0)

    def test_append_sorts_by_frame(self, io, tmp_path):
        path = tmp_path / "locs.h5"
        df1 = _loc_df(3)
        df1["frame"] = [5, 6, 7]
        io.write_h5_database(df1, path)

        df2 = _loc_df(3)
        df2["frame"] = [1, 2, 3]
        io.write_h5_database(df2, path, append=True, verbose=True)

        out = io.read_h5_database(path)
        assert len(out) == 6
        assert list(out["frame"]) == sorted(out["frame"])

    def test_append_to_nonexistent_file_writes_fresh(self, io, tmp_path):
        path = tmp_path / "new.h5"
        df = _loc_df(3)
        io.write_h5_database(df, path, append=True)
        assert path.exists()

    def test_write_h5_database_private_alias(self, io, tmp_path):
        path = tmp_path / "alias.h5"
        io._write_h5_database(_loc_df(2), path)
        assert path.exists()

    def test_custom_key_roundtrips_and_is_isolated_from_default_key(self, io, tmp_path):
        # Regression test: gui/main_window.py writes simulation ground truth
        # (columns xc_nm/yc_nm, not xc/yc) into the same folder as real fit
        # results, and AnalysisPipeline.load_localisations[_per_fov] glob that
        # folder for "*.h5" and read each under the default "data" key. Without
        # a distinct key, a later fit run's file-glob can pick up the ground
        # truth file instead of (or alongside) the real results and misread it
        # as localisations (KeyError: 'xc' downstream in the GUI's renderer).
        path = tmp_path / "ground_truth.h5"
        gt = pd.DataFrame({"dye": ["a", "b"], "xc_nm": [1.0, 2.0], "yc_nm": [3.0, 4.0]})
        io.write_h5_database(gt, path, verbose=False, key="ground_truth")

        out = io.read_h5_database(path, key="ground_truth")
        assert list(out["xc_nm"]) == [1.0, 2.0]

        # The default key ("data") -- what load_localisations[_per_fov] actually
        # read with -- must NOT find this file's contents, so the existing
        # KeyError-skip logic in AnalysisPipeline correctly skips it.
        with pytest.raises(KeyError):
            io.read_h5_database(path)

    def test_append_honours_custom_key(self, io, tmp_path):
        path = tmp_path / "custom_key.h5"
        df1 = _loc_df(3)
        df1["frame"] = [5, 6, 7]
        io.write_h5_database(df1, path, key="ground_truth")

        df2 = _loc_df(3)
        df2["frame"] = [1, 2, 3]
        io.write_h5_database(df2, path, append=True, key="ground_truth")

        out = io.read_h5_database(path, key="ground_truth")
        assert len(out) == 6
        assert list(out["frame"]) == sorted(out["frame"])
        with pytest.raises(KeyError):
            io.read_h5_database(path)


class TestSortH5ByFrame:
    def test_sorts_out_of_order_file_with_backup(self, io, tmp_path):
        path = tmp_path / "unsorted.h5"
        df = pd.DataFrame({"frame": [3, 1, 2], "xc": [1.0, 2.0, 3.0]})
        df.to_hdf(path, key="data", format="table", index=False)

        io.sort_h5_by_frame(str(path), backup=True)

        backup_path = str(path).replace(".h5", "_backup.h5")
        assert (tmp_path / "unsorted_backup.h5").exists()
        out = pd.read_hdf(path, key="data")
        assert list(out["frame"]) == [1, 2, 3]

    def test_no_backup(self, io, tmp_path):
        path = tmp_path / "unsorted2.h5"
        df = pd.DataFrame({"frame": [2, 1], "xc": [1.0, 2.0]})
        df.to_hdf(path, key="data", format="table", index=False)
        io.sort_h5_by_frame(str(path), backup=False)
        assert not (tmp_path / "unsorted2_backup.h5").exists()
        out = pd.read_hdf(path, key="data")
        assert list(out["frame"]) == [1, 2]

    def test_missing_file_logs_warning_and_does_not_raise(self, io, tmp_path, caplog):
        missing = str(tmp_path / "does_not_exist.h5")
        with caplog.at_level(logging.WARNING):
            io.sort_h5_by_frame(missing, backup=False)
        assert "Error sorting HDF5 file" in caplog.text

    def test_invalid_file_with_backup_logs_restore_hint(self, io, tmp_path, caplog):
        # A file that exists (so shutil.copy2 succeeds) but isn't valid HDF5,
        # so the failure happens inside the try, after backup_path is already set.
        path = tmp_path / "corrupt.h5"
        path.write_text("not an hdf5 file")
        with caplog.at_level(logging.INFO):
            io.sort_h5_by_frame(str(path), backup=True)
        assert "Restore from backup if needed" in caplog.text


class TestEnsureHdf5Compatibility:
    def test_empty_existing_table_returns_original(self, io, tmp_path):
        # to_hdf silently skips creating the "data" key for a genuinely empty
        # DataFrame, so the 0-row table must be built via append+remove instead
        # to actually reach the `len(existing_df) == 0` branch (not the outer
        # except, which a plain missing key would exercise instead).
        path = tmp_path / "existing.h5"
        seed_df = pd.DataFrame({"frame": [1, 2], "xc": [1.0, 2.0]})
        with pd.HDFStore(path, mode="w") as store:
            store.append("data", seed_df, format="table", index=False)
            store.remove("data", where="index >= 0")
        new_df = pd.DataFrame({"frame": [1], "xc": [1.0]})
        out = io._ensure_hdf5_compatibility(new_df, path)
        assert out is new_df

    def test_honours_custom_key(self, io, tmp_path):
        path = tmp_path / "existing.h5"
        existing = pd.DataFrame({"frame": np.array([1, 2], dtype="int16")})
        existing.to_hdf(path, key="ground_truth", format="table", index=False)
        new_df = pd.DataFrame({"frame": np.array([3, 4], dtype="int64")})
        out = io._ensure_hdf5_compatibility(new_df, path, key="ground_truth")
        assert out["frame"].dtype == np.dtype("int32")

    def test_no_common_columns_returns_original(self, io, tmp_path):
        path = tmp_path / "existing.h5"
        pd.DataFrame({"foo": [1]}).to_hdf(path, key="data", format="table", index=False)
        new_df = pd.DataFrame({"bar": [1.0]})
        out = io._ensure_hdf5_compatibility(new_df, path)
        assert list(out.columns) == ["bar"]

    def test_frame_column_int16_upgrades_to_int32(self, io, tmp_path, caplog):
        # dtypes must actually differ (existing int16, new int64) for the
        # conversion block to run at all -- same-dtype columns skip it entirely.
        path = tmp_path / "existing.h5"
        existing = pd.DataFrame({"frame": np.array([1, 2], dtype="int16")})
        existing.to_hdf(path, key="data", format="table", index=False)
        new_df = pd.DataFrame({"frame": np.array([3, 4], dtype="int64")})
        with caplog.at_level(logging.INFO):
            out = io._ensure_hdf5_compatibility(new_df, path)
        assert out["frame"].dtype == np.dtype("int32")
        assert "Frame column upgrading" in caplog.text

    def test_frame_column_both_int_non_int16_keeps_existing_dtype(self, io, tmp_path):
        path = tmp_path / "existing.h5"
        existing = pd.DataFrame({"frame": np.array([1, 2], dtype="int32")})
        existing.to_hdf(path, key="data", format="table", index=False)
        new_df = pd.DataFrame({"frame": np.array([3, 4], dtype="int64")})
        out = io._ensure_hdf5_compatibility(new_df, path)
        assert out["frame"].dtype == np.dtype("int32")

    def test_frame_column_non_int_existing_direct_astype(self, io, tmp_path):
        path = tmp_path / "existing.h5"
        existing = pd.DataFrame({"frame": np.array([1.0, 2.0], dtype="float64")})
        existing.to_hdf(path, key="data", format="table", index=False)
        new_df = pd.DataFrame({"frame": np.array([3, 4], dtype="int32")})
        out = io._ensure_hdf5_compatibility(new_df, path)
        assert out["frame"].dtype == np.dtype("float64")

    def test_int_to_int_numeric_column_uses_existing_dtype(self, io, tmp_path):
        path = tmp_path / "existing.h5"
        existing = pd.DataFrame({"count": np.array([1, 2], dtype="int64")})
        existing.to_hdf(path, key="data", format="table", index=False)
        new_df = pd.DataFrame({"count": np.array([3, 4], dtype="int32")})
        out = io._ensure_hdf5_compatibility(new_df, path)
        assert out["count"].dtype == np.dtype("int64")

    def test_float_to_int_all_integer_values_converts(self, io, tmp_path):
        path = tmp_path / "existing.h5"
        existing = pd.DataFrame({"val": np.array([1, 2], dtype="int32")})
        existing.to_hdf(path, key="data", format="table", index=False)
        new_df = pd.DataFrame({"val": np.array([3.0, 4.0], dtype="float64")})
        out = io._ensure_hdf5_compatibility(new_df, path)
        assert out["val"].dtype == np.dtype("int32")

    def test_float_to_int_decimal_values_kept_as_float(self, io, tmp_path):
        path = tmp_path / "existing.h5"
        existing = pd.DataFrame({"val": np.array([1, 2], dtype="int32")})
        existing.to_hdf(path, key="data", format="table", index=False)
        new_df = pd.DataFrame({"val": np.array([3.5, 4.2], dtype="float64")})
        out = io._ensure_hdf5_compatibility(new_df, path)
        assert out["val"].dtype == np.dtype("float64")

    def test_float32_to_int_converts_directly(self, io, tmp_path):
        # dtype != "float64" skips the integer-value inspection entirely and
        # falls straight to the plain astype(existing_dtype) branch.
        path = tmp_path / "existing.h5"
        existing = pd.DataFrame({"val": np.array([1, 2], dtype="int32")})
        existing.to_hdf(path, key="data", format="table", index=False)
        new_df = pd.DataFrame({"val": np.array([3.0, 4.0], dtype="float32")})
        out = io._ensure_hdf5_compatibility(new_df, path)
        assert out["val"].dtype == np.dtype("int32")

    def test_other_numeric_conversion(self, io, tmp_path):
        path = tmp_path / "existing.h5"
        existing = pd.DataFrame({"val": np.array([1.0, 2.0], dtype="float32")})
        existing.to_hdf(path, key="data", format="table", index=False)
        new_df = pd.DataFrame({"val": np.array([3.0, 4.0], dtype="float64")})
        out = io._ensure_hdf5_compatibility(new_df, path)
        assert out["val"].dtype == np.dtype("float32")

    def test_non_numeric_column_direct_conversion(self, io, tmp_path):
        path = tmp_path / "existing.h5"
        existing = pd.DataFrame({"label": pd.Series(["a", "b"], dtype="category")})
        existing.to_hdf(path, key="data", format="table", index=False)
        new_df = pd.DataFrame({"label": ["c", "d"]})
        out = io._ensure_hdf5_compatibility(new_df, path)
        assert out is not None

    def test_conversion_failure_keeps_original_dtype(self, io, tmp_path, caplog):
        path = tmp_path / "existing.h5"
        existing = pd.DataFrame({"val": np.array([1, 2], dtype="int32")})
        existing.to_hdf(path, key="data", format="table", index=False)
        new_df = pd.DataFrame({"val": ["not", "numeric"]})
        with caplog.at_level(logging.WARNING):
            out = io._ensure_hdf5_compatibility(new_df, path)
        assert list(out["val"]) == ["not", "numeric"]

    def test_missing_file_triggers_outer_exception_fallback(self, io, tmp_path, caplog):
        missing = tmp_path / "nope.h5"
        new_df = pd.DataFrame({"val": [1]})
        with caplog.at_level(logging.WARNING):
            out = io._ensure_hdf5_compatibility(new_df, missing)
        assert out is new_df


# ======================================================================
# read_json
# ======================================================================

class TestReadJson:
    def test_normal_read(self, io, tmp_path):
        path = tmp_path / "data.json"
        path.write_text(json.dumps({"a": 1}))
        assert io.read_json(path) == {"a": 1}

    def test_bad_encoding_falls_back_to_default(self, io, tmp_path):
        path = tmp_path / "data.json"
        path.write_text(json.dumps({"a": 1}))
        out = io.read_json(path, encoding="totally-bogus-encoding")
        assert out == {"a": 1}

    def test_truncated_json_salvaged(self, io, tmp_path):
        path = tmp_path / "truncated.json"
        path.write_text('{"a": 1, "b": 2')
        out = io.read_json(path)
        assert out == {"a": 1, "b": 2}

    def test_unsalvageable_json_raises(self, io, tmp_path):
        path = tmp_path / "garbage.json"
        # Closing the dangling brace still leaves invalid syntax ('{"a": }'),
        # unlike a bare truncated-string case where the salvage trivially succeeds.
        path.write_text('{"a": tru')
        with pytest.raises(json.JSONDecodeError):
            io.read_json(path)

    def test_pos_zero_skips_salvage(self, io, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text("")
        with pytest.raises(json.JSONDecodeError):
            io.read_json(path)


# ======================================================================
# read_json_streaming_first_framekey
# ======================================================================

class TestReadJsonStreamingFirstFramekey:
    def test_finds_first_framekey(self, io, tmp_path):
        path = tmp_path / "stream.json"
        path.write_text('{"FrameKey-0-0-0": {"a": 1, "b": {"c": 2}}, "ignored": "trailing"}')
        out = io.read_json_streaming_first_framekey(path)
        assert out == {"FrameKey-0-0-0": {"a": 1, "b": {"c": 2}}}

    def test_finds_framekey_across_chunk_boundary(self, io, tmp_path):
        path = tmp_path / "stream_big.json"
        padding = "x" * 20000
        content = '{"padding": "' + padding + '", "FrameKey-1-2-3": {"ROI": "0-0-10-10"}}'
        path.write_text(content)
        out = io.read_json_streaming_first_framekey(path)
        assert out == {"FrameKey-1-2-3": {"ROI": "0-0-10-10"}}

    def test_no_framekey_raises_value_error(self, io, tmp_path):
        path = tmp_path / "no_framekey.json"
        path.write_text('{"a": 1}')
        with pytest.raises(ValueError, match="No FrameKey found"):
            io.read_json_streaming_first_framekey(path)

    def test_no_framekey_in_large_file_truncates_buffer(self, io, tmp_path):
        # >2x chunk_size (8192) with no FrameKey anywhere exercises the
        # buffer-truncation branch taken while still searching for the start.
        path = tmp_path / "no_framekey_large.json"
        path.write_text('{"padding": "' + ("x" * 30000) + '"}')
        with pytest.raises(ValueError, match="No FrameKey found"):
            io.read_json_streaming_first_framekey(path)

    def test_malformed_framekey_json_raises(self, io, tmp_path):
        path = tmp_path / "malformed.json"
        path.write_text('{"FrameKey-0-0-0": {a: 1}, "trailer": 1}')
        with pytest.raises(json.JSONDecodeError):
            io.read_json_streaming_first_framekey(path)


# ======================================================================
# get_num_pages_in_TIF / metadata readers
# ======================================================================

class TestGetNumPagesInTIF:
    def test_counts_pages(self, io, tmp_path):
        path = tmp_path / "stack.tif"
        tifffile.imwrite(path, np.zeros((5, 8, 8), dtype=np.float32))
        assert io.get_num_pages_in_TIF(path) == 5


class TestMetadataReaderImageJ:
    def test_small_file_no_exposure(self, io, tmp_path):
        path = tmp_path / "meta.json"
        path.write_text(json.dumps(_imagej_metadata(roi="10-20-40-30")))
        x, y, w, h = io.metadata_reader_imageJ(path)
        assert (x, y, w, h) == (10, 20, 40, 30)

    def test_small_file_with_exposure(self, io, tmp_path):
        path = tmp_path / "meta.json"
        path.write_text(json.dumps(_imagej_metadata(roi="1-2-3-4", exposure_ms=50.0)))
        x, y, w, h, exposure = io.metadata_reader_imageJ(path, return_exposure=True)
        assert (x, y, w, h) == (1, 2, 3, 4)
        assert exposure == pytest.approx(50.0)

    def test_large_file_uses_streaming_parser(self, io, tmp_path, monkeypatch):
        path = tmp_path / "meta_large.json"
        path.write_text(json.dumps(_imagej_metadata(roi="5-6-7-8")))
        monkeypatch.setattr(iof.Path, "stat", lambda self: _FakeStat(20 * 1024 * 1024))
        x, y, w, h = io.metadata_reader_imageJ(path)
        assert (x, y, w, h) == (5, 6, 7, 8)


class TestMetadataNframesReaderImageJ:
    def test_reads_intended_time_dimension(self, io, tmp_path):
        path = tmp_path / "meta.json"
        path.write_text(json.dumps(_imagej_metadata(n_frames=42)))
        assert io.metadata_nframes_reader_imageJ(path) == 42


class TestMetadataReaderThorlabs:
    def test_reads_roi(self, io, tmp_path):
        path = tmp_path / "thorlabs.json"
        path.write_text(json.dumps({
            "ROIOriginX_pixels": 3, "ROIOriginY_pixels": 4,
            "ROIWidth_pixels": 50, "ROIHeight_pixels": 60,
        }))
        assert io.metadata_reader_Thorlabs(path) == (3, 4, 50, 60)


# ======================================================================
# _read_tiff_robust
# ======================================================================

class TestReadTiffRobust:
    def test_reads_all_frames_cleanly(self, io, tmp_path):
        path = tmp_path / "clean.tif"
        arr = np.arange(2 * 6 * 6, dtype=np.float32).reshape(2, 6, 6)
        tifffile.imwrite(path, arr)
        out = io._read_tiff_robust(path, None)
        assert out.shape == (2, 6, 6)

    def test_single_frame_request_returns_unwrapped_array(self, io, tmp_path):
        path = tmp_path / "clean.tif"
        arr = np.arange(2 * 6 * 6, dtype=np.float32).reshape(2, 6, 6)
        tifffile.imwrite(path, arr)
        out = io._read_tiff_robust(path, [0])
        assert out.shape == (6, 6)

    def test_corrupted_middle_frame_filled_with_zeros(self, io, tmp_path, monkeypatch):
        path = tmp_path / "clean.tif"
        # Avoid exactly 3/4 frames -- tifffile ambiguously stores those as a
        # single RGB/RGBA page instead of a genuine multi-page frame stack.
        arr = np.ones((5, 6, 6), dtype=np.float32)
        tifffile.imwrite(path, arr)

        from tifffile import TiffPage
        real_asarray = TiffPage.asarray

        def flaky_asarray(self, *a, **kw):
            if self.index == 1:
                raise RuntimeError("simulated corruption")
            return real_asarray(self, *a, **kw)

        monkeypatch.setattr(TiffPage, "asarray", flaky_asarray)
        out = io._read_tiff_robust(path, None)
        assert out.shape == (5, 6, 6)
        np.testing.assert_allclose(out[1], 0.0)
        np.testing.assert_allclose(out[0], 1.0)

    def test_first_frame_corrupted_falls_through_to_second(self, io, tmp_path, monkeypatch):
        path = tmp_path / "clean.tif"
        arr = np.stack([np.zeros((6, 6), dtype=np.float32), np.ones((6, 6), dtype=np.float32)])
        tifffile.imwrite(path, arr)

        from tifffile import TiffPage
        real_asarray = TiffPage.asarray

        def flaky_first(self, *a, **kw):
            if self.index == 0:
                raise RuntimeError("simulated corruption")
            return real_asarray(self, *a, **kw)

        monkeypatch.setattr(TiffPage, "asarray", flaky_first)
        out = io._read_tiff_robust(path, None)
        assert out.shape == (2, 6, 6)
        np.testing.assert_allclose(out[0], 0.0)
        np.testing.assert_allclose(out[1], 1.0)

    def test_all_frames_corrupted_raises_runtime_error(self, io, tmp_path, monkeypatch):
        path = tmp_path / "clean.tif"
        arr = np.ones((2, 6, 6), dtype=np.float32)
        tifffile.imwrite(path, arr)

        from tifffile import TiffPage

        def always_fail(self, *a, **kw):
            raise RuntimeError("simulated corruption")

        monkeypatch.setattr(TiffPage, "asarray", always_fail)
        with pytest.raises(RuntimeError, match="Could not load any frames"):
            io._read_tiff_robust(path, None)


# ======================================================================
# read_hyperstack
# ======================================================================

class TestReadHyperstack:
    def test_reads_stack_with_axes(self, io, tmp_path):
        path = tmp_path / "hyper.tif"
        arr = np.zeros((2, 6, 6), dtype=np.float32)
        tifffile.imwrite(path, arr)
        data, axes = io.read_hyperstack(path)
        assert data.shape == (2, 6, 6)


# ======================================================================
# read_tiff
# ======================================================================

class TestReadTiff:
    def _make_stack(self, tmp_path, n_frames=5, h=6, w=6, name="stack.tif"):
        path = tmp_path / name
        arr = np.arange(n_frames * h * w, dtype=np.float32).reshape(n_frames, h, w)
        tifffile.imwrite(path, arr)
        return path, arr

    def test_full_stack_memmap(self, io, tmp_path):
        path, arr = self._make_stack(tmp_path)
        out = io.read_tiff(path)
        assert out.shape == arr.shape

    def test_full_stack_no_memmap(self, io, tmp_path):
        path, arr = self._make_stack(tmp_path)
        out = io.read_tiff(path, memmap=False)
        assert out.shape == arr.shape

    def test_single_frame_memmap(self, io, tmp_path):
        path, arr = self._make_stack(tmp_path)
        out = io.read_tiff(path, frame=2)
        np.testing.assert_allclose(out, arr[2])

    def test_single_frame_no_memmap(self, io, tmp_path):
        path, arr = self._make_stack(tmp_path)
        out = io.read_tiff(path, frame=2, memmap=False)
        np.testing.assert_allclose(out, arr[2])

    def test_frame_list_memmap(self, io, tmp_path):
        path, arr = self._make_stack(tmp_path)
        out = io.read_tiff(path, frame=[0, 2])
        assert out.shape == (2, 6, 6)

    def test_frame_list_no_memmap(self, io, tmp_path):
        path, arr = self._make_stack(tmp_path)
        out = io.read_tiff(path, frame=[0, 2], memmap=False)
        assert out.shape == (2, 6, 6)

    def test_out_of_range_frame_reraises_index_error(self, io, tmp_path):
        path, arr = self._make_stack(tmp_path)
        with pytest.raises(IndexError):
            io.read_tiff(path, frame=999)

    def test_memmap_failure_falls_back_to_standard_none_frames(self, io, tmp_path, monkeypatch):
        path, arr = self._make_stack(tmp_path)
        real_asarray = tifffile.TiffFile.asarray

        def flaky(self, *a, **kw):
            if kw.get("out") == "memmap":
                raise RuntimeError("simulated memmap failure")
            return real_asarray(self, *a, **kw)

        monkeypatch.setattr(tifffile.TiffFile, "asarray", flaky)
        out = io.read_tiff(path)
        assert out.shape == arr.shape

    def test_memmap_failure_falls_back_to_standard_frame_list(self, io, tmp_path, monkeypatch):
        path, arr = self._make_stack(tmp_path)
        real_asarray = tifffile.TiffFile.asarray

        def flaky(self, *a, **kw):
            if kw.get("out") == "memmap":
                raise RuntimeError("simulated memmap failure")
            return real_asarray(self, *a, **kw)

        monkeypatch.setattr(tifffile.TiffFile, "asarray", flaky)
        out = io.read_tiff(path, frame=[0, 1])
        assert out.shape == (2, 6, 6)

    def test_memmap_failure_falls_back_to_standard_single_frame(self, io, tmp_path, monkeypatch):
        path, arr = self._make_stack(tmp_path)
        real_asarray = tifffile.TiffFile.asarray

        def flaky(self, *a, **kw):
            if kw.get("out") == "memmap":
                raise RuntimeError("simulated memmap failure")
            return real_asarray(self, *a, **kw)

        monkeypatch.setattr(tifffile.TiffFile, "asarray", flaky)
        out = io.read_tiff(path, frame=1)
        np.testing.assert_allclose(out, arr[1])

    def test_single_frame_fallback_index_error_reraises(self, io, tmp_path, monkeypatch):
        path, arr = self._make_stack(tmp_path)

        def always_fail(self, *a, **kw):
            raise RuntimeError("simulated memmap failure")

        monkeypatch.setattr(tifffile.TiffFile, "asarray", always_fail)

        def imread_raises_index_error(*a, **kw):
            raise IndexError("simulated out of range")

        monkeypatch.setattr(iof, "imread", imread_raises_index_error)
        with pytest.raises(IndexError):
            io.read_tiff(path, frame=1)

    def test_all_methods_fail_triggers_robust_recovery_none_frames(self, io, tmp_path, monkeypatch):
        path, arr = self._make_stack(tmp_path)

        def always_fail(self, *a, **kw):
            raise RuntimeError("simulated memmap failure")

        monkeypatch.setattr(tifffile.TiffFile, "asarray", always_fail)

        def imread_fails(*a, **kw):
            raise RuntimeError("simulated standard-load failure")

        monkeypatch.setattr(iof, "imread", imread_fails)
        out = io.read_tiff(path)
        assert out.shape == arr.shape

    def test_all_methods_fail_triggers_robust_recovery_frame_list(self, io, tmp_path, monkeypatch):
        path, arr = self._make_stack(tmp_path)

        def always_fail(self, *a, **kw):
            raise RuntimeError("simulated memmap failure")

        monkeypatch.setattr(tifffile.TiffFile, "asarray", always_fail)

        def imread_fails(*a, **kw):
            raise RuntimeError("simulated standard-load failure")

        monkeypatch.setattr(iof, "imread", imread_fails)
        out = io.read_tiff(path, frame=[0, 1])
        assert out.shape == (2, 6, 6)

    def test_all_methods_fail_triggers_robust_recovery_single_frame(self, io, tmp_path, monkeypatch):
        path, arr = self._make_stack(tmp_path)

        def always_fail(self, *a, **kw):
            raise RuntimeError("simulated memmap failure")

        monkeypatch.setattr(tifffile.TiffFile, "asarray", always_fail)

        def imread_fails(*a, **kw):
            raise RuntimeError("simulated standard-load failure")

        monkeypatch.setattr(iof, "imread", imread_fails)
        out = io.read_tiff(path, frame=1)
        np.testing.assert_allclose(out, arr[1])


class TestGetNFrames:
    def test_counts_frames(self, io, tmp_path):
        path = tmp_path / "stack.tif"
        tifffile.imwrite(path, np.zeros((7, 6, 6), dtype=np.float32))
        assert io.get_n_frames(path) == 7


# ======================================================================
# read_tiff_tophotoelectrons / convert_to_photoelectrons
# ======================================================================

class TestReadTiffTophotoelectrons:
    def test_scalar_gain_offset(self, io, tmp_path):
        path = tmp_path / "raw.tif"
        arr = np.full((6, 6), 100.0, dtype=np.float32)
        tifffile.imwrite(path, arr)
        out = io.read_tiff_tophotoelectrons(path, gain_map=2.0, offset_map=10.0, rqe=1.0)
        np.testing.assert_allclose(out, (100.0 - 10.0) / 2.0)

    def test_array_gain_matching_shape_single_frame(self, io, tmp_path):
        path = tmp_path / "raw.tif"
        arr = np.full((6, 6), 100.0, dtype=np.float32)
        tifffile.imwrite(path, arr)
        gain = np.full((6, 6), 2.0, dtype=np.float32)
        offset = np.full((6, 6), 5.0, dtype=np.float32)
        out = io.read_tiff_tophotoelectrons(path, gain_map=gain, offset_map=offset, rqe=1.0)
        np.testing.assert_allclose(out, (100.0 - 5.0) / 2.0)

    def test_array_gain_stack_scalar_rqe(self, io, tmp_path):
        # Regression test: array gain/offset + scalar (default) rqe on stack
        # data used to crash with `TypeError: 'float' object is not subscriptable`
        # since the 3D branch assumed rqe was always array-shaped alongside gain_map.
        path = tmp_path / "raw_stack.tif"
        arr = np.full((5, 6, 6), 100.0, dtype=np.float32)
        tifffile.imwrite(path, arr)
        gain = np.full((6, 6), 2.0, dtype=np.float32)
        offset = np.full((6, 6), 5.0, dtype=np.float32)
        out = io.read_tiff_tophotoelectrons(path, gain_map=gain, offset_map=offset, rqe=1.0)
        assert out.shape == (5, 6, 6)
        np.testing.assert_allclose(out[0], (100.0 - 5.0) / 2.0)

    def test_array_gain_stack_array_rqe(self, io, tmp_path):
        path = tmp_path / "raw_stack.tif"
        arr = np.full((5, 6, 6), 100.0, dtype=np.float32)
        tifffile.imwrite(path, arr)
        gain = np.full((6, 6), 2.0, dtype=np.float32)
        offset = np.full((6, 6), 5.0, dtype=np.float32)
        rqe = np.full((6, 6), 0.5, dtype=np.float32)
        out = io.read_tiff_tophotoelectrons(path, gain_map=gain, offset_map=offset, rqe=rqe)
        assert out.shape == (5, 6, 6)
        np.testing.assert_allclose(out[0], ((100.0 - 5.0) / 2.0) / 0.5)

    def test_array_gain_mismatched_shape_falls_back(self, io, tmp_path):
        path = tmp_path / "raw.tif"
        arr = np.full((6, 6), 100.0, dtype=np.float32)
        tifffile.imwrite(path, arr)
        gain = np.full((4, 4), 2.0, dtype=np.float32)
        out = io.read_tiff_tophotoelectrons(path, gain_map=gain, offset_map=0.0, rqe=1.0)
        np.testing.assert_allclose(out, 100.0)


class TestConvertToPhotoelectrons:
    def test_all_scalar(self, io):
        raw = np.full((6, 6), 100.0)
        out = io.convert_to_photoelectrons(raw, gain_map=2.0, offset_map=10.0, rqe=1.0)
        np.testing.assert_allclose(out, 45.0)

    def test_array_maps_2d(self, io):
        raw = np.full((6, 6), 100.0)
        gain = np.full((6, 6), 2.0)
        offset = np.full((6, 6), 10.0)
        rqe = np.full((6, 6), 0.5)
        out = io.convert_to_photoelectrons(raw, gain_map=gain, offset_map=offset, rqe=rqe)
        np.testing.assert_allclose(out, ((100.0 - 10.0) / 2.0) / 0.5)

    def test_array_maps_3d_stack(self, io):
        raw = np.full((3, 6, 6), 100.0)
        gain = np.full((6, 6), 2.0)
        offset = np.full((6, 6), 10.0)
        rqe = np.full((6, 6), 0.5)
        out = io.convert_to_photoelectrons(raw, gain_map=gain, offset_map=offset, rqe=rqe)
        assert out.shape == (3, 6, 6)
        np.testing.assert_allclose(out[0], ((100.0 - 10.0) / 2.0) / 0.5)

    def test_mismatched_gain_shape_falls_back_to_scalar(self, io):
        raw = np.full((6, 6), 100.0)
        gain = np.full((4, 4), 2.0)
        out = io.convert_to_photoelectrons(raw, gain_map=gain, offset_map=99.0, rqe=1.0)
        np.testing.assert_allclose(out, 100.0)


# ======================================================================
# apply_smoothing / generate_weights
# ======================================================================

class TestApplySmoothing:
    def test_none_smoothing_function_returns_data_as_is(self, io):
        data = np.arange(9, dtype=np.float32).reshape(3, 3)
        out = io.apply_smoothing(data, None)
        np.testing.assert_allclose(out, data)

    def test_applies_smoothing_function(self, io):
        import types

        def double_it(data):
            return data * 2

        sf = types.SimpleNamespace(args={}, data_arg="data", smoothing_function=double_it)
        data = np.ones((3, 3), dtype=np.float32)
        out = io.apply_smoothing(data, sf)
        np.testing.assert_allclose(out, 2.0)


class TestGenerateWeights:
    def test_scalar_read_noise(self, io):
        data = np.full((4, 4), 3.0, dtype=np.float32)
        out = io.generate_weights(data, read_noise=1.0)
        expected = 1.0 / (4.0 + 1.0)
        np.testing.assert_allclose(out, expected)

    def test_negative_values_clipped_to_zero(self, io):
        data = np.full((4, 4), -5.0, dtype=np.float32)
        out = io.generate_weights(data, read_noise=1.0)
        expected = 1.0 / (0.0 + 1.0 + 1.0)
        np.testing.assert_allclose(out, expected)

    def test_array_read_noise_2d(self, io):
        data = np.full((4, 4), 3.0, dtype=np.float32)
        read_noise = np.full((4, 4), 2.0, dtype=np.float32)
        out = io.generate_weights(data, read_noise=read_noise)
        expected = 1.0 / (4.0 + 4.0)
        np.testing.assert_allclose(out, expected)

    def test_array_read_noise_3d_stack(self, io):
        data = np.full((2, 4, 4), 3.0, dtype=np.float32)
        read_noise = np.full((4, 4), 2.0, dtype=np.float32)
        out = io.generate_weights(data, read_noise=read_noise)
        assert out.shape == (2, 4, 4)

    def test_hot_pixels_get_tiny_weight(self, io):
        data = np.full((4, 4), 3.0, dtype=np.float32)
        read_noise = np.full((4, 4), 1.0, dtype=np.float32)
        read_noise[0, 0] = 100.0
        out = io.generate_weights(data, read_noise=read_noise, hot_pixel_threshold=20)
        assert out[0, 0] == pytest.approx(1e-8)
        assert out[1, 1] != pytest.approx(1e-8)

    def test_hot_pixels_3d_stack(self, io):
        data = np.full((2, 4, 4), 3.0, dtype=np.float32)
        read_noise = np.full((4, 4), 1.0, dtype=np.float32)
        read_noise[0, 0] = 100.0
        out = io.generate_weights(data, read_noise=read_noise, hot_pixel_threshold=20)
        assert out[0, 0, 0] == pytest.approx(1e-8)
        assert out[1, 0, 0] == pytest.approx(1e-8)


# ======================================================================
# write_tiff
# ======================================================================

class TestWriteTiff:
    def test_grayscale_roundtrip(self, io, tmp_path):
        path = tmp_path / "gray.tif"
        volume = np.arange(2 * 6 * 6, dtype=np.float64).reshape(2, 6, 6)
        io.write_tiff(volume, path, bit="double", pixel_size=0.1)
        out = tifffile.imread(path)
        np.testing.assert_allclose(out, volume)

    def test_rgb_roundtrip(self, io, tmp_path):
        path = tmp_path / "rgb.tif"
        volume = (np.random.default_rng(1).uniform(0, 255, (6, 6, 3))).astype(np.uint8)
        io.write_tiff(volume, path, bit="uint8", pixel_size=0.1, photometric="rgb")
        out = tifffile.imread(path)
        assert out.shape == (6, 6, 3)


# ======================================================================
# save_simulation_results
# ======================================================================

class TestSaveSimulationResults:
    def _call(self, io, tmp_path, pixel_size=0.1, NA=1.4, background_photons=10.0,
               smoothing_function_extent=3.0):
        io.save_simulation_results(
            save_folder=tmp_path,
            starting_flag="test_",
            default_params=np.array(["param1"]),
            n_photon_space=np.array([100.0, 200.0]),
            fit_RMSE_mean=np.array([[1.0, 2.0]]),
            fit_RMSE_std=np.array([[0.1, 0.2]]),
            pixel_size=pixel_size,
            NA=NA,
            background_photons=background_photons,
            fit_function_name="gauss",
            smoothing_function_name="box",
            smoothing_function_extent=smoothing_function_extent,
            dye="Alexa/647",
        )

    def test_integer_valued_params_produce_files(self, io, tmp_path):
        self._call(io, tmp_path, pixel_size=1.0, NA=1.0, background_photons=10.0,
                    smoothing_function_extent=3.0)
        csvs = list(tmp_path.glob("*.csv"))
        assert len(csvs) == 2
        assert all("Alexa-647" in f.name for f in csvs)

    def test_fractional_valued_params_produce_files(self, io, tmp_path):
        self._call(io, tmp_path, pixel_size=0.107, NA=1.42, background_photons=10.5,
                    smoothing_function_extent=3.25)
        csvs = list(tmp_path.glob("*.csv"))
        assert len(csvs) == 2
