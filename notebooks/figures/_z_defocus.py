"""TetraSpeck bead z-stack (SI "Effect of Z Defocus on Extraction of Spectral Signature").

Nine fields of view of TetraSpeck beads in 1 % PVA on the Ximea, 488 nm excitation, Perfect Focus
System on, 21 z-steps from -1,000 to +1,000 nm (100 nm apart) x 100 frames per step. MicroManager
split each field into a main ``.ome.tif`` and a ``_1.ome.tif`` continuation file; ``zlog.csv`` is the
stage log (measured z-drive position of every plane).

Pipeline (as the developer notebook ``z_defocus_data/z_defocus_analysis``):

* fit: ``fit_QD_data`` detects spots on the sum of all 2,100 frames of a field, then fits every frame
  at every spot (``fit_fields``; the split files are stitched by page count, ``_multipart_patches``);
* select (``select_beads``): fit QC (colour-fraction errors < 1), >= ``MIN_PHOTOELECTRONS`` detected
  photoelectrons and pixel 2 share >= ``MIN_A_G`` per localisation, then beads present in >= 90 % of
  the z-steps;
* z = 0 is the z-step with the narrowest mean PSF over all beads (``global_focus``);
* per z-step (``precision_vs_z``): sigma_xy = RMS of each localisation's offset from its own bead's
  mean position at that step; sigma_colour of pixel p = SD of A_p about its own bead's mean A_p at
  that step, in units of relative pixel QE. (Not divided by the mean as in Fig. 2d: at 488 nm the
  TetraSpeck pixel-3 share is ~0.01, so the relative SD of pixel 3 is 50-160 % and says nothing.)
"""

import json
import os
import re
import shutil
import types
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

REMOTE_FOLDER = "2026_Multicolour_Paper/Ximea/Z_Defocus/PFS_zstack_488"
FIRST_FIELD = "PFS_zstack_488_489_MMStack_Pos-0-0"   # FAST_MODE: this field only

N_Z_STEPS = 21
N_REPEATS = 100
Z_STEP_NM = 100
Z_NM = np.arange(-1000, 1000 + Z_STEP_NM, Z_STEP_NM)
PIXEL_NM = 69.0

# fit_QD_data settings of the 2026-11-09 fits (detection on the sum of all frames; no upper PSF
# width bound, so defocused PSFs are kept).
FIT_PARAMS = dict(n_frames_sum=N_Z_STEPS * N_REPEATS, pfa=1e-4, ROI_size=40, peak_wavelength=0.488,
                  NA=1.49, image_type=".ome.tif", fit_max_sigma=None)

MAX_COLOUR_ERROR = 1.0
MIN_PHOTOELECTRONS = 700
MIN_A_G = 0.5
MIN_PRESENCE_FRACTION = 0.9
MIN_PLANES = int(np.ceil(MIN_PRESENCE_FRACTION * N_Z_STEPS))

CHANNELS = ["B", "G", "R"]   # pixels 1, 2, 3
PIXEL_NAMES = {"B": "pixel 1", "G": "pixel 2", "R": "pixel 3"}
PIXEL_COLOURS = {"B": "tab:blue", "G": "tab:green", "R": "tab:red"}

_PART_RE = re.compile(r"^(?P<base>.+)_(?P<part>\d+)\.ome\.tif$")
_PAGES = {}


# ── split OME-TIFFs ───────────────────────────────────────────────────────────

def field_groups(folder):
    """``{field name: [main file, continuation files...]}`` (frame order) for a folder of
    MicroManager OME-TIFFs split at ~4 GB."""
    names = sorted(f for f in os.listdir(folder) if f.endswith(".ome.tif"))
    parts, mains = {}, []
    for f in names:
        m = _PART_RE.match(f)
        if m:
            parts.setdefault(m.group("base"), []).append((int(m.group("part")), f))
        else:
            mains.append(f)
    return {m[:-len(".ome.tif")]: [str(Path(folder) / f) for f in
                                   [m] + [p for _, p in sorted(parts.get(m[:-len(".ome.tif")], []))]]
            for m in mains}


def n_pages(path):
    key = str(Path(path).resolve())
    if key not in _PAGES:
        with tifffile.TiffFile(path, is_ome=False) as tf:
            _PAGES[key] = len(tf.pages)
    return _PAGES[key]


def read_frames(group, frames):
    """Frames (global indices over the field's files) as a ``(len(frames), H, W)`` float array."""
    bounds = np.cumsum([0] + [n_pages(p) for p in group])
    out = []
    for f in frames:
        i = int(np.searchsorted(bounds, f, side="right") - 1)
        with tifffile.TiffFile(group[i], is_ome=False) as tf:
            out.append(tf.pages[int(f - bounds[i])].asarray().astype(float))
    return np.stack(out)


def _multipart_patches(fitter, groups):
    """Make ``fitter`` (a ``SuperRes_Functions``) see each split field as one file:
    ``file_search`` drops the continuation files, ``get_num_pages_in_TIF`` and ``read_tiff``
    span all of a field's files. Paths outside ``groups`` use the original methods."""
    by_main = {str(Path(g[0]).resolve()): g for g in groups.values()}
    helper_cls, io_cls = type(fitter.helper), type(fitter.io)
    orig_search, orig_pages, orig_read = helper_cls.file_search, io_cls.get_num_pages_in_TIF, io_cls.read_tiff

    def file_search(self, folder, string1, string2):
        files = orig_search(self, folder, string1, string2)
        return [f for f in files if not _PART_RE.match(os.path.basename(f))] if string1 == ".ome.tif" else files

    def get_num_pages(self, filename):
        g = by_main.get(str(Path(filename).resolve()))
        return orig_pages(self, filename) if g is None else sum(n_pages(p) for p in g)

    def read_tiff(self, file_path, frame=None, dtype="float32", memmap=True):
        g = by_main.get(str(Path(file_path).resolve()))
        if g is None:
            return orig_read(self, file_path, frame=frame, dtype=dtype, memmap=memmap)
        if frame is None:
            return np.concatenate([orig_read(self, p, frame=None, dtype=dtype, memmap=memmap) for p in g])
        bounds = np.cumsum([0] + [n_pages(p) for p in g])
        single = not hasattr(frame, "__len__")
        fr = np.asarray([frame] if single else frame)
        out = None
        for p, lo, hi in zip(g, bounds[:-1], bounds[1:]):
            sel = (fr >= lo) & (fr < hi)
            if not sel.any():
                continue
            local = (fr[sel] - lo).tolist()
            data = orig_read(self, p, frame=local if len(local) > 1 else local[0], dtype=dtype, memmap=memmap)
            data = data[np.newaxis] if data.ndim == 2 else data
            if out is None:
                out = np.empty((len(fr),) + data.shape[1:], dtype=data.dtype)
            out[np.where(sel)[0]] = data
        return out[0] if single else out

    fitter.helper.file_search = types.MethodType(file_search, fitter.helper)
    fitter.io.get_num_pages_in_TIF = types.MethodType(get_num_pages, fitter.io)
    fitter.io.read_tiff = types.MethodType(read_tiff, fitter.io)


def fit_fields(fitter, folder, calib, smoothing_function, refit=False):
    """Fit every field in ``folder`` with ``fit_QD_data`` (one ``<field>.h5`` beside its TIFFs).

    Existing fits are reused only if a ``fit_record.json`` beside them names the same fields,
    ``FIT_PARAMS`` and fitter stopping settings (``fitter_settings``), and ``refit`` is False. Fits
    without a record (e.g. the 2026-11-09 fits, made with the loose large-ROI tolerances) are
    refitted. A refit moves the old ``.h5`` to ``old_fits/<folder>_<date>/`` next to the folder
    first (``fit_QD_data`` appends to existing files).
    """
    folder = Path(folder)
    groups = field_groups(folder)
    if not groups:
        raise FileNotFoundError(f"no .ome.tif in {folder}")
    record_file = folder / "fit_record.json"
    record = {"fields": sorted(groups), **FIT_PARAMS, "fitter": fitter_settings()}
    previous = json.loads(record_file.read_text()) if record_file.exists() else None
    fits = [folder / f"{name}.h5" for name in groups]
    if not refit and all(f.exists() for f in fits) and previous == record:
        print(f"Reusing {len(fits)} fit(s) in {folder}")
        return fits
    if previous != record and any(f.exists() for f in fits):
        print("Refitting: " + ("no fit_record.json beside the fits" if previous is None
                               else "the fit record names other settings"))
    old = [f for f in folder.glob("*.h5")]
    if old:
        dest = folder.parent / "old_fits" / f"{folder.name}_{datetime.now():%Y%m%d_%H%M}"
        dest.mkdir(parents=True)
        for f in old:
            shutil.move(str(f), str(dest / f.name))
        print(f"Moved {len(old)} earlier fit(s) to {dest}")
    _multipart_patches(fitter, groups)
    print(f"Fitting {len(groups)} field(s) in {folder}; ~10 min each")
    fitter.fit_QD_data(image_folder=str(folder), smoothing_function=smoothing_function, **calib, **FIT_PARAMS)
    record_file.write_text(json.dumps(record, indent=1))
    return fits


def fitter_settings():
    """The Levenberg-Marquardt stopping settings the fits of ``FIT_PARAMS["ROI_size"]`` boxes use
    (``ImageAnalysisFunctions._lm_kwargs``), recorded with the fits."""
    from pyS3M import ImageAnalysisFunctions as IAF
    n_params = 10   # STANDARD_DATA: row, col, s_row, s_col, 3 backgrounds, 3 amplitudes
    return {k: float(v) for k, v in IAF._lm_kwargs(FIT_PARAMS["ROI_size"], n_params).items()}


# ── analysis ──────────────────────────────────────────────────────────────────

def load_fits(folder, io):
    """All fields' fits, with ``z_step_idx``, ``repeat_idx``, nominal ``z_nm`` and a ``bead`` id
    unique over fields. Frames are ordered z-step by z-step: frame = z_step * N_REPEATS + repeat."""
    tables = []
    for name in sorted(field_groups(folder)):
        d = io.add_photoelectron_columns(pd.read_hdf(Path(folder) / f"{name}.h5"))
        d["field"] = name
        d["z_step_idx"] = (d["frame"] // N_REPEATS).astype(int).clip(0, N_Z_STEPS - 1)
        d["repeat_idx"] = (d["frame"] % N_REPEATS).astype(int)
        d["z_nm"] = Z_NM[d["z_step_idx"].to_numpy()]
        d["bead"] = name + "_" + d["puncta_id"].astype(str)
        tables.append(d)
    return pd.concat(tables, ignore_index=True)


def select_beads(fits, sm_extractor, pixel_nm=PIXEL_NM):
    """Fit QC (colour-fraction errors < ``MAX_COLOUR_ERROR``, no other cut: defocused PSFs are wide
    and imprecise by design), then localisations with >= ``MIN_PHOTOELECTRONS`` and A_G >=
    ``MIN_A_G``, then beads present in >= ``MIN_PLANES`` z-steps. Returns (table, counts)."""
    counts = {"fits": len(fits), "beads fitted": fits["bead"].nunique()}
    d = sm_extractor.filter_quality_localisations(
        fits, chi_val=None, max_localisation_error=np.inf, max_colour_error=MAX_COLOUR_ERROR,
        max_background_error=None, min_sigma=0.0, max_sigma=np.inf, max_sigma_error=np.inf,
        min_photoelectrons=-np.inf, max_photoelectrons=np.inf)
    counts["after fit QC"] = len(d)
    d = d[(d["photoelectrons"] >= MIN_PHOTOELECTRONS) & (d["A_G"] >= MIN_A_G)]
    counts[f">= {MIN_PHOTOELECTRONS} pe, A_G >= {MIN_A_G}"] = len(d)
    planes = d.groupby("bead")["z_step_idx"].nunique()
    d = d[d["bead"].isin(planes.index[planes >= MIN_PLANES])].copy()
    counts["localisations kept"] = len(d)
    counts[f"beads in >= {MIN_PLANES}/{N_Z_STEPS} z-steps"] = d["bead"].nunique()
    return d, counts


def global_focus(d):
    """z = 0: the z-step with the smallest mean PSF width (all beads; with the PFS on, the beads
    share one focal plane). Returns (focus z-step index, nominal z of it, mean sigma per step, px)."""
    s = ((d["s_x"] + d["s_y"]) / 2).groupby(d["z_step_idx"]).mean()
    idx = int(s.idxmin())
    return idx, float(Z_NM[idx]), s


def precision_vs_z(d, pixel_nm=PIXEL_NM, min_locs=5):
    """Per z-step: ``sigma_xy_nm`` (RMS of the offsets from each bead's mean position at that
    step), ``sigma_colour_<p>`` (SD of A_p - bead mean A_p, relative pixel QE) and
    ``rel_sigma_colour_<p>`` (SD of A_p / bead mean A_p - 1) for p in B, G, R,
    the median detected photoelectrons and the number of localisations and beads."""
    rows = []
    for z, g in d.groupby("z_rel_nm"):
        if len(g) < min_locs:
            continue
        by = g.groupby("bead")
        ex = g["xc"] - by["xc"].transform("mean")
        ey = g["yc"] - by["yc"].transform("mean")
        row = {"z_rel_nm": z, "sigma_xy_nm": np.sqrt((np.nanstd(ex) ** 2 + np.nanstd(ey) ** 2) / 2) * pixel_nm,
               "median_pe": g["photoelectrons"].median(), "n_locs": len(g), "n_beads": g["bead"].nunique()}
        for ch in CHANNELS:
            a = g["A_" + ch]
            mean = by["A_" + ch].transform("mean")
            row["sigma_colour_" + ch] = np.nanstd(a - mean)
            row["rel_sigma_colour_" + ch] = np.nanstd(a / mean - 1)
        rows.append(row)
    return pd.DataFrame(rows)


# ── figures ───────────────────────────────────────────────────────────────────

def grid(ax):
    ax.grid(color="gray", lw=0.5, ls="--", alpha=0.5)
    ax.set_axisbelow(True)


def plot_precision(axs, prec):
    """Left: sigma_xy against z. Right: sigma_colour of each pixel type against z."""
    ax_xy, ax_c = axs
    ax_xy.plot(prec["z_rel_nm"], prec["sigma_xy_nm"], "o-", color="k", ms=3, lw=1)
    ax_xy.set_ylabel(r"$\sigma_{xy}$/nm")
    for ch in CHANNELS:
        ax_c.plot(prec["z_rel_nm"], prec["sigma_colour_" + ch], "o-", color=PIXEL_COLOURS[ch],
                  ms=3, lw=1, label=PIXEL_NAMES[ch])
    ax_c.set_ylabel(r"$\sigma_{colour}$ (relative pixel QE)")
    ax_c.legend(fontsize=6)
    for ax in axs:
        ax.axvline(0, color="gray", lw=0.8, ls="--")
        ax.set_xlabel("z relative to focus/nm")
        ax.set_ylim(bottom=0)
        grid(ax)


def plot_ridge(fig, d, z_positions, x_range=(0.0, 1.0), bw=0.15):
    """One row per z (nearest step to each of ``z_positions``, relative to focus): KDE of each pixel
    type's relative pixel QE, every curve scaled to the same peak height; dashed lines = the means
    at focus. Returns the axes."""
    from scipy.stats import gaussian_kde
    x = np.linspace(*x_range, 400)
    avail = np.sort(d["z_rel_nm"].unique())
    used = [float(avail[np.argmin(np.abs(avail - z))]) for z in z_positions]
    at_focus = {ch: d.loc[d["z_rel_nm"] == 0, "A_" + ch].mean() for ch in CHANNELS}
    axs = fig.subplots(len(used), 1, sharex=True, gridspec_kw={"hspace": 0})
    for ax, z in zip(np.atleast_1d(axs), used):
        sub = d[d["z_rel_nm"] == z]
        for ch in CHANNELS:
            v = sub["A_" + ch].to_numpy(float)
            v = v[np.isfinite(v)]
            y = gaussian_kde(v, bw_method=bw)(x)
            y = y / y.max()
            ax.fill_between(x, 0, y, color=PIXEL_COLOURS[ch], alpha=0.55, lw=0)
            ax.plot(x, y, color=PIXEL_COLOURS[ch], lw=1.2)
            ax.axvline(at_focus[ch], color=PIXEL_COLOURS[ch], lw=1.2, ls="--", alpha=0.8)
        ax.set_ylim(0, 1.1)
        ax.set_yticks([])
        ax.axhline(0, color="k", lw=0.8)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.text(-0.02, 0.05, f"z = {z:+.0f} nm", fontsize=7, ha="right", va="bottom", transform=ax.transAxes)
        ax.grid(axis="x", color="gray", lw=0.5, ls="--", alpha=0.5)
        ax.set_axisbelow(True)
    axs = np.atleast_1d(axs)
    axs[-1].set_xlim(*x_range)
    axs[-1].set_xlabel("relative pixel QE")
    from matplotlib.lines import Line2D
    axs[0].legend(handles=[Line2D([], [], color=PIXEL_COLOURS[c], lw=3, label=PIXEL_NAMES[c]) for c in CHANNELS],
                  fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    return axs
