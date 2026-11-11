"""Shared helper for the SI "example puncta" panels (raw / weights / fit / residual /
spectral fingerprint), used by `SI/FigureSI_ExampleFitting_QDots.ipynb`.

Picks single-frame localisations from a fit result (`fit_SM_data` / `fit_imaging_data`), crops
the raw frame each one came from, recomputes the photoelectron image exactly as the fitter does
(`SR_Functions._build_roi`) and the weights of the fit's final pass, and rebuilds the fitted PSF
model from the stored parameters (Gaussian x Bayer mask).

Weights: STANDARD_DATA fits three times (smoothed-data weights, then model weights, then raw-data
weights; `ImageAnalysisFunctions.StandardDataFittingProcessor`). The stored parameters, errors and
chi^2 come from the last pass, weighted 1 / (max(pe, 0) + 1 + read_noise^2) per pixel, so those are
the weights shown.

Units: the fit databases store normalised `A_*`/`bg_*` fractions plus the detected signal
`photoelectrons` and the background per pixel `background_photoelectrons`; each pixel type's
fitted amplitude and background follow from those (`IO_Functions.channel_amplitudes`, with the
Ximea/ZWO Bayer mosaic of `IO`). The model rebuilt here is in photoelectrons per pixel, like the
data it is compared to; numbers printed on the figure are the detected photoelectrons.
Ported from developer:notebooks/figures/SI/FigureSI_ExampleAnalysisWorkflow_*.ipynb.
"""
import numpy as np
import tifffile

import pyS3M.gaussoptfuncs as gaussoptfuncs
from pyS3M import IOFunctions, PlottingBase

IO = IOFunctions.IO_Functions()
CHANNEL_COLOURS = ["#5555ff", "#53b03b", "#ff2a2a"]  # B, G, R
ROW_LABELS = ["raw", "weights", "fit", "residuals", "spectral\nfingerprint"]


class FrameReader:
    """Read single frames by *global* frame index from a multi-file acquisition.

    `fit_imaging_data` numbers frames continuously across the files it fits, in
    `file_search` order (`…_Default.ome.tif`, `…_Default_1.ome.tif`, …), so the same
    ordered file list maps a global index back to (file, page). `fit_SM_data` writes one
    `.h5` per TIFF, numbered from 0: give it that one TIFF.
    """

    def __init__(self, image_files):
        self.files = [str(f) for f in image_files]
        self._n_pages = {}

    def n_pages(self, path):
        if path not in self._n_pages:
            with tifffile.TiffFile(path, is_ome=False) as tf:
                self._n_pages[path] = len(tf.pages)
        return self._n_pages[path]

    def read(self, global_frame, dtype=np.float32):
        offset = 0
        for path in self.files:
            n = self.n_pages(path)
            if global_frame < offset + n:
                with tifffile.TiffFile(path, is_ome=False) as tf:
                    return tf.pages[global_frame - offset].asarray().astype(dtype)
            offset += n
        raise IndexError(f"frame {global_frame} out of range ({offset} frames in {len(self.files)} file(s))")


def choose_puncta(fits_df, width, height, roi_size, n_puncta, seed, query=None, spread_by=None):
    """Pick `n_puncta` fits whose full ROI lies inside the frame.

    Args:
        query: optional pandas query string for dataset-specific quality cuts.
        spread_by: optional column (e.g. ``"A_R"``). The candidates are split into `n_puncta`
            equal-count bins of this column and one fit is drawn from each, so the examples span
            its range (for QDots: from the bluest to the reddest). Without it, `n_puncta` fits
            are drawn at random.
        seed: seed of the random draws.

    Returns:
        DataFrame of the chosen fits, sorted by `spread_by` if given.
    """
    margin = roi_size / 2
    valid = fits_df[
        (fits_df["xc"] > margin) & (fits_df["xc"] < width - margin)
        & (fits_df["yc"] > margin) & (fits_df["yc"] < height - margin)
    ]
    if query:
        valid = valid.query(query)
    valid = valid.reset_index(drop=True)
    print(f"{len(valid):,}/{len(fits_df):,} fits pass the quality cuts with a full {roi_size}x{roi_size} ROI")
    if len(valid) < n_puncta:
        raise ValueError(f"only {len(valid)} candidate fits for {n_puncta} puncta -- loosen the quality cuts")
    rng = np.random.default_rng(seed)
    if spread_by is None:
        return valid.iloc[rng.choice(len(valid), size=n_puncta, replace=False)].reset_index(drop=True)
    order = valid[spread_by].to_numpy().argsort(kind="stable")
    picks = [rng.choice(b) for b in np.array_split(order, n_puncta)]
    return valid.iloc[picks].reset_index(drop=True)


def build_example_puncta(rows, reader, cropped_maps, masks, roi_size):
    """Crop, convert and re-model each chosen fit.

    Args:
        rows: DataFrame of chosen fits (`choose_puncta` output).
        reader: FrameReader over the acquisition's files.
        cropped_maps: dict of gain_map/offset_map/rqe/read_noise already cropped to the
            acquisition ROI.
        masks: stacked Bayer masks for the acquisition ROI (`Mask_Functions.get_stacked_masks`).
    Returns:
        list of dicts with raw, weights, model, residual (photoelectrons per pixel; weights in
        photoelectrons^-2), spectrum and its error (normalised A_B, A_G, A_R), xy (ROI-local),
        frame, photoelectrons (detected signal), background_photoelectrons (per pixel), and
        pixel_type (the ROI's Bayer filter per pixel, 0/1/2 = B/G/R, for ``add_bayer_key``).
    """
    height, width = masks.shape[:2]
    x_arr = np.arange(roi_size, dtype=np.float32)
    out = []
    for _, row in rows.iterrows():
        xcentre, ycentre = int(round(row["xc"])), int(round(row["yc"]))
        xmin = int(np.clip(xcentre - roi_size // 2, 0, width - roi_size))
        ymin = int(np.clip(ycentre - roi_size // 2, 0, height - roi_size))
        sl = (slice(ymin, ymin + roi_size), slice(xmin, xmin + roi_size))

        frame = reader.read(int(row["frame"]))
        pe = IO.convert_to_photoelectrons(
            frame[sl], gain_map=cropped_maps["gain_map"][sl], offset_map=cropped_maps["offset_map"][sl],
            rqe=cropped_maps["rqe"][sl],
        )
        # Final-pass (raw-data) weights, as StandardDataFittingProcessor._raw_data_weights
        weights = 1.0 / (np.maximum(pe, 0) + 1.0 + np.square(cropped_maps["read_noise"][sl])).astype(np.float32)

        # Mask/h5 channel order is B, G, R. A_*/bg_* are normalised fractions; channel_amplitudes
        # turns them into the fitted amplitude and background of each pixel type, in
        # photoelectrons, i.e. the model the fitter compared to the data.
        # The model works in the ROI's [row, column] frame: row centre and width first (yc, s_y),
        # while the database reports xc = column, yc = row.
        xc_local, yc_local = row["xc"] - xmin, row["yc"] - ymin
        gauss2d = gaussoptfuncs.gaussian_unscaled_model(
            np.zeros((roi_size, roi_size), dtype=np.float32), x_arr, roi_size, yc_local, xc_local, row["s_y"], row["s_x"],
        )
        A_frac = np.array([row["A_B"], row["A_G"], row["A_R"]])
        A_err = np.array([row["A_B_err"], row["A_G_err"], row["A_R_err"]])
        A_pe = np.array([IO.channel_amplitudes(row, "A_")[c] for c in "BGR"])
        bg_pe = np.array([IO.channel_amplitudes(row, "bg_")[c] for c in "BGR"])
        model = np.zeros((roi_size, roi_size), dtype=np.float32)
        mask_roi = masks[sl[0], sl[1], :]
        for c in range(3):
            m = mask_roi[:, :, c].astype(bool)
            model[m] = A_pe[c] * gauss2d[m] + bg_pe[c]

        out.append(dict(raw=pe, weights=weights, model=model, residual=pe - model, spectrum=A_frac,
                        spectrum_err=A_err, xy=(xc_local, yc_local), frame=int(row["frame"]),
                        pixel_type=np.argmax(mask_roi, axis=2), photoelectrons=float(row["photoelectrons"]),
                        background_photoelectrons=float(row["background_photoelectrons"])))
    return out


def plot_example_puncta(plotter, puncta, pixel_size_nm, scalebar_nm=500):
    """5-row grid (raw / weights / fit / residuals / spectral fingerprint), one column per punctum.

    Raw and fit share one photoelectron colour scale, residuals another. Each raw punctum carries
    the 2 x 2 Bayer key of its own corner (the filters under its top-left pixels).
    """
    n = len(puncta)
    fig, axs = plotter.two_column_plot(nrows=5, ncols=n, width=6.69, height=6.5)
    raws = np.array([p["raw"] for p in puncta])
    wts = np.array([p["weights"] for p in puncta])
    vmin, vmax = np.percentile(raws, 1), np.percentile(raws, 99.5)
    wmin, wmax = np.percentile(wts, 1), np.percentile(wts, 99.5)
    rmax = max(np.max(np.abs(p["residual"])) for p in puncta)
    last = n - 1
    for k, p in enumerate(puncta):
        axs[0, k], _ = plotter.image_plot(
            axs[0, k], p["raw"], vmin=vmin, vmax=vmax, origin="upper", pixelsize=pixel_size_nm,
            scalebar=(k == 0), scalebarsize=float(scalebar_nm), scalebarlabel=f"{scalebar_nm} nm",
            colorbar=(k == last), colorbar_label="photoelectrons",
        )
        PlottingBase.add_bayer_key(axs[0, k], p["pixel_type"], origin="upper")
        axs[1, k], _ = plotter.image_plot(axs[1, k], p["weights"], vmin=wmin, vmax=wmax, cmap="gray", origin="upper",
                                          colorbar=(k == last), colorbar_label="weight (pe$^{-2}$)")
        axs[2, k], _ = plotter.image_plot(axs[2, k], p["model"], vmin=vmin, vmax=vmax, origin="upper",
                                          colorbar=(k == last), colorbar_label="photoelectrons")
        axs[2, k].plot(*p["xy"], "rx", markersize=8, markeredgewidth=2)
        axs[3, k], _ = plotter.image_plot(axs[3, k], p["residual"], cmap="RdBu_r", vmin=-rmax, vmax=rmax,
                                          origin="upper", colorbar=(k == last),
                                          colorbar_label="residual (pe)")
        for r in (1, 3):
            axs[r, k].set_xticks([])
            axs[r, k].set_yticks([])
        axs[4, k].bar(np.arange(3), p["spectrum"], yerr=p["spectrum_err"], color=CHANNEL_COLOURS,
                      error_kw=dict(lw=0.6, capsize=1.5))
        axs[4, k].set_ylim([0, 1])
        axs[4, k].set_xticks([])
        axs[4, k].set_yticks([0, 1])
        if k != 0:
            axs[4, k].set_yticklabels("")
        axs[4, k].grid(False)
    for r, label in enumerate(ROW_LABELS):
        axs[r, 0].text(-0.35, 0.5, label, transform=axs[r, 0].transAxes, fontsize=8, ha="right", va="center", rotation=90)
    return fig, axs
