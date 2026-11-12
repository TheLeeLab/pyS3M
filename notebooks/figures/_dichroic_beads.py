"""Shared code for FigureSI_DichroicBeads (dichroic bead control, Fig. S78).

Yellow-green (YG, FluoSpheres F8803) and red (R, F8801) 100 nm beads on one coverslip, ND 3.5,
imaged on the Ximea camera. Each field of view (FOV) is one ``.ome.tif`` of 5 timepoints x 3
channels, interleaved (frame = 3 x timepoint + channel):

========  =================  ==============================  ==========================
channel   excitation         emission filters                sees
========  =================  ==============================  ==========================
0         488 nm             Di03-R488/561 + FF01-520/44     YG beads (reference)
1         561 nm             Di03-R488/561 + BLP01-568R      R beads (reference)
                             + FF01-582/64
2         488 + 561 nm       Di03-R488/561 + NF03 quad       all beads: the S3M measurement
                             notch                           ("combined channel")
========  =================  ==============================  ==========================

Three power sets (488 nm at 2, 4, 8 mW with 561 nm at 5, 10, 20 %), 12 FOVs each.

Pipeline (the notebook's sections):

1. fit        -- ``AnalysisPipeline.fit`` (mode ``smlm``) on every frame of every FOV.
2. filter     -- quality cuts as in the Methods, HDBSCAN per FOV across all 15 frames, A_B cut
                 (:func:`load_set`).
3. calls      -- per power set, a two-component GMM on (A_R, A_G) of every combined-channel
                 localisation, with no reference information (:func:`unmix`).
4. references -- persistent spots in each filter channel, followed through every pipeline stage in
                 their own channel and in the combined channel (:func:`track_fov`), and the S3M
                 outcome of each (:func:`score`).
5. figure     -- spectra, example FOV, confusion matrices by own-channel photoelectron range and
                 the photoelectron histogram (:func:`plot_figure`).

Signal is in detected photoelectrons (``photoelectrons`` = A_B/4 + A_G/2 + A_R/4). Fitted ``xc`` is
the image column and ``yc`` the row (``Constants.ResultColumns``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter, median_filter
from scipy.spatial import cKDTree

from pyS3M.Constants import FilteringConstants

MM = 1 / 25.4
N_CHANNELS = 3
CHANNEL_NAMES = {0: "YG (488)", 1: "R (561)", 2: "All"}
SET_LABELS = {"488_2mW_561_5p": "2 mW / 5 %", "488_4mW_561_10p": "4 mW / 10 %",
              "488_8mW_561_20p": "8 mW / 20 %"}
SETS = list(SET_LABELS)

# Bead colours; YG and R also differ in line style (red/green is a colour-blind confusion pair)
STYLE = {"YG": dict(color="#3a9a2a", ls="-"), "R": dict(color="#c8102e", ls="--")}

# Filter stacks of the three channels (Semrock part numbers, as in Spectra/spectral_data.duckdb)
FILTERS = {
    "YG only": ["semrock-ff01-520-44", "semrock-di03-r488-561-t1-25x36"],
    "R only": ["semrock-di03-r488-561-t1-25x36", "semrock-blp01-568r", "semrock-ff01-582-64"],
    "both": ["semrock-di03-r488-561-t1-25x36", "semrock-nf03-405-488-561-635e"],
}
BEAD_SPECTRA = {"YG": "FluoSpheres yellow-green", "R": "FluoSpheres Red"}

OUTCOMES = ["called YG", "called R", "unclassified", "missed"]

ROW_MM = {"spectra": 34.0, "g": 58.0, "ij": 42.0}   # figure row heights; the FOV row fits its images
FOV_GAP_MM = 1.5


@dataclass
class Settings:
    """Every threshold of the analysis (see the notebook's methods cell)."""
    pixel_nm: float = 69.0
    min_pe: float = 50.0            # quality cut, detected photoelectrons
    max_pe: float = 5000.0          # scored range: the Gaussian fitter is less reliable above
    chi2_max: float = 3.0           # reduced chi^2
    loc_err_max_px: float = 1.0
    sigma_bounds_nm: tuple = (75.0, 250.0)
    a_b_max: float = 0.3
    min_cluster_size: int = 2
    gmm_confidence: float = 0.6
    persist: int = 3                # reference: detected in >= persist of the 5 timepoints
    match_px: float = 2.0           # same bead within this distance (138 nm)
    stages: list = field(default_factory=lambda: ["detected", "fitted", "passed filter",
                                                  "clustered", "A_B cut"])

    def filtering_criteria(self):
        from pyS3M.Constants import FilteringCriteria
        # No colour-error cut (10,000): it would select localisations on the quantity being tested.
        return FilteringCriteria(
            min_photoelectrons=self.min_pe, max_photoelectrons=None, max_colour_error=10000,
            chi_val=self.chi2_max, max_localisation_error=self.loc_err_max_px,
            min_sigma=self.sigma_bounds_nm[0] / self.pixel_nm,
            max_sigma=self.sigma_bounds_nm[1] / self.pixel_nm)

    @property
    def missed_labels(self):
        return {f"lost: {s}": f"missed: {m}" for s, m in zip(
            self.stages, ["not detected", "no fit", "quality filter", "not clustered",
                          f"A_B >= {self.a_b_max:g}"])}


# --------------------------------------------------------------------------- spectra (a-c)
def channel_spectra(S_F, wavelength):
    """Per channel: the filter stack's transmission and each bead's emission through it, each
    scaled to its own maximum (Brendan's BeadSpectra notebook). ``{channel: {"filters", "YG",
    "R"}}``."""
    from pyS3M import SpectralFunctions
    beads = {k: S_F.get_spectral_data(v, wavelength,
                                      data_type=SpectralFunctions.SpectralDataType.DYE)[0]
             for k, v in BEAD_SPECTRA.items()}
    out = {}
    for ch, names in FILTERS.items():
        t = np.prod(S_F.get_dye_or_filter_data(names=names, wavelength=wavelength,
                                               dye_or_filter=False), axis=0)
        t = t / t.max()
        out[ch] = {"filters": t, **{k: t * b / (t * b).max() if (t * b).max() > 0 else t * b
                                    for k, b in beads.items()}}
        # As in the original panels: in a single-colour channel the other bead is scaled to the
        # reference bead's maximum, so its bleed-through is shown to scale.
        if ch == "YG only":
            out[ch]["R"] = t * beads["R"] / (t * beads["YG"]).max()
        elif ch == "R only":
            out[ch]["YG"] = t * beads["YG"] / (t * beads["R"]).max()
    return out


def plot_spectra(axs, wavelength, spectra, xlim=(450, 650)):
    for ax, (ch, s) in zip(axs, spectra.items()):
        ax.plot(wavelength, s["filters"], color="k", lw=1, label="filters")
        for bead, name in (("YG", "FluoSpheres YG"), ("R", "FluoSpheres R")):
            ax.plot(wavelength, s[bead], lw=1, label=name, **STYLE[bead])
        ax.set(xlim=xlim, ylim=(0, 1.05))
        ax.set_xlabel("wavelength/nm", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, ls="--", lw=0.6, alpha=0.6, color="grey")
        ax.set_axisbelow(True)
    for ax in axs[1:]:
        ax.set_yticklabels([])
    axs[0].set_ylabel("transmission,\nemission (norm.)", fontsize=7)
    axs[-1].legend(fontsize=5, loc="upper left", frameon=False)


# --------------------------------------------------------------------------- filter, cluster, calls
def load_set(pipe, fov_dirs, st: Settings):
    """Localisations of every FOV of one power set: ``fov_data`` [(locs, tif)] and the filtered,
    clustered table (all FOVs, ``fov_index``, global ``molecular_index``, A_B cut applied)."""
    from pyS3M.clustering import ClusteringConfig
    cc, filt = ClusteringConfig(min_cluster_size=st.min_cluster_size), st.filtering_criteria()
    fov_data, parts, offset = [], [], 0
    for fov_idx, d in enumerate(fov_dirs):
        (locs, tif), = pipe.load_localisations_per_fov(d)
        locs = pipe.sm.io.add_photoelectron_columns(locs)
        fov_data.append((locs, tif))
        _, sf = pipe.filter_and_cluster(locs, clustering_config=cc, criteria=filt)
        if sf.empty:
            continue
        parts.append(sf.assign(fov_index=fov_idx, fov_name=d.name,
                               molecular_index=sf["molecular_index"] + offset))
        offset = int(parts[-1]["molecular_index"].max()) + 1
    sf = pd.concat(parts, ignore_index=True)
    return fov_data, sf[sf["A_B"] < st.a_b_max]


def unmix(pipe, locs, st: Settings):
    """GMM colour calls on (A_R, A_G): adds ``call`` (YG / R / unclassified) and ``p_R``."""
    out, meta = pipe.sm.unmix_channels(
        locs, n_channels=2, channels_to_use=["A_R", "A_G"], confidence_threshold=st.gmm_confidence,
        outlier_rejection="none", verbose=False, plot_results=False)
    r = int(np.argmax(np.asarray(meta["means"])[:, 0]))   # more pixel-3 signal (A_R) = R beads
    out["call"] = np.select([out["channel"] < 0, out["channel"] == r], ["unclassified", "R"], "YG")
    out["p_R"] = out[f"channel_probability_{r}"]
    return out


# --------------------------------------------------------------------------- reference tracking
def _nearest(ref_rc, points_rc, r):
    """Index of the nearest point within r of each reference position, else -1."""
    if len(points_rc) == 0 or len(ref_rc) == 0:
        return np.full(len(ref_rc), -1)
    d, i = cKDTree(points_rc).query(ref_rc)
    return np.where(d <= r, i, -1)


def _dedupe(points, r):
    keep = []
    for p in points:
        if not keep or cKDTree(keep).query(p)[0] > r:
            keep.append(p)
    return np.array(keep).reshape(-1, 2)


def _loc_rc(df):
    """(row, col) of localisations: fitted xc is the column, yc the row."""
    return np.column_stack([df["yc"].to_numpy(), df["xc"].to_numpy()]) if len(df) else np.zeros((0, 2))


def _bead_signal(mean_pe, rc):
    """Smoothed peak signal (pe above local background) around each position."""
    sig = gaussian_filter(mean_pe - median_filter(mean_pe, 25), 1.3)
    return np.array([sig[max(i - 2, 0):i + 3, max(j - 2, 0):j + 3].max()
                     for i, j in np.round(rc).astype(int)])


def track_fov(pipe, fc, st: Settings, set_name, fov_idx, fov_dir, locs, tif):
    """One row per (reference spot, tracked channel, timepoint): the pipeline stages it reached.

    Reference spots: spot detection alone (the same call and settings as the fit) on each filter
    channel's 5 timepoints; detections within ``match_px`` merged; a spot seen in >= ``persist``
    timepoints is a persistent reference. Each is followed in its own channel and in the
    combined channel: detected -> fitted -> passed the quality filter -> clustered -> A_B cut,
    each a match within ``match_px`` of the reference position in that frame.
    """
    from pyS3M.clustering import ClusteringConfig
    cc, filt = ClusteringConfig(min_cluster_size=st.min_cluster_size), st.filtering_criteria()
    sd, io = pipe.sr.spot_detection, pipe.sr.io

    def detect(img, variance):
        return sd.detect_puncta_in_image(
            img, pfa=fc.pfa, variance=variance, wavelength=fc.peak_wavelength,
            pixel_size=pipe.pixel_size, NA=fc.NA).astype(float)   # (row, col), as the fit does

    x0, y0, _, _ = pipe.sr.helper.load_metadata_roi(fov_dir, io, use_fallback=True)
    stack = io.read_tiff(str(tif), dtype="float32")
    n_t = stack.shape[0] // N_CHANNELS
    H, W = stack.shape[1:]
    crop = (slice(y0, y0 + H), slice(x0, x0 + W))
    variance, gain, offset = pipe.variance[crop], pipe.gain_map[crop], pipe.offset_map[crop]
    pe = (stack - offset) / gain

    ref = []
    for ch in (0, 1):
        frames = [detect(stack[N_CHANNELS * t + ch], variance) for t in range(n_t)]
        cand = _dedupe(np.vstack(frames), st.match_px)
        n_seen = sum(_nearest(cand, f, st.match_px) >= 0 for f in frames)
        ref += [(ch, rc) for rc in cand[n_seen >= st.persist]]
    ref_type = np.array([c for c, _ in ref], dtype=int)
    ref_rc = np.array([p for _, p in ref]).reshape(-1, 2)

    filtered = pipe.sm.filter_quality_localisations(locs, criteria=filt)
    _, clustered = pipe.filter_and_cluster(locs, clustering_config=cc, criteria=filt)

    rows = []
    for track_ch in (0, 1, 2):
        sel = (ref_type == track_ch) if track_ch < 2 else np.ones(len(ref_rc), bool)
        if not sel.any():
            continue
        signal = _bead_signal(pe[track_ch::N_CHANNELS].mean(0), ref_rc[sel])
        for t in range(n_t):
            frame = N_CHANNELS * t + track_ch
            fl = locs[locs["frame"] == frame]
            ff = filtered[filtered["frame"] == frame]
            fcl = clustered[clustered["frame"] == frame] if len(clustered) else clustered
            i_det = _nearest(ref_rc[sel], detect(stack[frame], variance), st.match_px)
            i_fit = _nearest(ref_rc[sel], _loc_rc(fl), st.match_px)
            i_flt = _nearest(ref_rc[sel], _loc_rc(ff), st.match_px)
            i_cl = _nearest(ref_rc[sel], _loc_rc(fcl), st.match_px)
            for k, b in enumerate(np.flatnonzero(sel)):
                fit = fl.iloc[i_fit[k]] if i_fit[k] >= 0 else None
                ok = [i_det[k] >= 0, i_fit[k] >= 0, i_flt[k] >= 0, i_cl[k] >= 0,
                      i_cl[k] >= 0 and fcl.iloc[i_cl[k]]["A_B"] < st.a_b_max]
                rows.append({
                    "set": set_name, "fov_index": fov_idx, "fov_name": fov_dir.name, "bead": b,
                    "bead_type": "YG" if ref_type[b] == 0 else "R",
                    "row": ref_rc[b, 0], "col": ref_rc[b, 1],
                    "tracked_channel": track_ch, "timepoint": t, "frame": frame,
                    "signal_pe": signal[k],
                    "lost_at": next((f"lost: {s}" for s, o in zip(st.stages, ok) if not o), "kept"),
                    "photoelectrons": fit["photoelectrons"] if fit is not None else np.nan,
                })
    return pd.DataFrame(rows)


def score(track, s3m, st: Settings):
    """Every reference spot x timepoint in the combined channel with its S3M outcome.

    A spot kept in the combined channel takes the call of the nearest S3M localisation within
    ``match_px`` (called YG / called R / unclassified); otherwise it is ``missed: <stage>``. Spots
    persistent in both filter channels have no single true colour and are flagged ``in_both``.
    Adds ``own_pe`` / ``own_kept`` (the spot's fit and pipeline outcome in its own filter channel,
    same timepoint) and ``pe_s3m``. Labels ``s3m["reference"]`` (YG / R / both / no reference).
    Returns the scored table (``in_both`` removed) and the number of ``in_both`` spots.
    """
    s3m["reference"] = "no reference"
    ref = track[track["tracked_channel"] == 2].copy()
    ref["outcome"] = ref["lost_at"].map(st.missed_labels)
    ref["pe_s3m"] = np.nan
    for (s, fov, frame), rows in ref[ref["lost_at"] == "kept"].groupby(["set", "fov_index", "frame"]):
        cand = s3m[(s3m["set"] == s) & (s3m["fov_index"] == fov) & (s3m["frame"] == frame)]
        hit = _nearest(rows[["row", "col"]].to_numpy(), _loc_rc(cand), st.match_px)
        matched = cand.index[hit[hit >= 0]]
        ref.loc[rows.index[hit >= 0], "outcome"] = [
            c if c == "unclassified" else f"called {c}" for c in cand.loc[matched, "call"]]
        ref.loc[rows.index[hit >= 0], "pe_s3m"] = cand.loc[matched, "photoelectrons"].to_numpy()
        s3m.loc[matched, "reference"] = rows["bead_type"].to_numpy()[hit >= 0]
    if ref["outcome"].isna().any():
        raise RuntimeError("a reference spot kept by the pipeline has no S3M localisation")

    both = set()
    for (s, fov), g in ref.groupby(["set", "fov_index"]):
        yg = g[g["bead_type"] == "YG"].drop_duplicates("bead")
        r = g[g["bead_type"] == "R"].drop_duplicates("bead")
        if len(yg) and len(r):
            hit = _nearest(yg[["row", "col"]].to_numpy(), r[["row", "col"]].to_numpy(), st.match_px)
            both |= {(s, fov, b) for b in yg["bead"].to_numpy()[hit >= 0]}
            both |= {(s, fov, b) for b in r["bead"].to_numpy()[hit[hit >= 0]]}
    ref["in_both"] = [k in both for k in zip(ref["set"], ref["fov_index"], ref["bead"])]
    for (s, fov, frame), rows in ref[ref["in_both"] & (ref["lost_at"] == "kept")].groupby(
            ["set", "fov_index", "frame"]):
        cand = s3m[(s3m["set"] == s) & (s3m["fov_index"] == fov) & (s3m["frame"] == frame)]
        hit = _nearest(rows[["row", "col"]].to_numpy(), _loc_rc(cand), st.match_px)
        s3m.loc[cand.index[hit[hit >= 0]], "reference"] = "both"

    own = track[track["tracked_channel"] != 2].set_index(["set", "fov_index", "bead", "timepoint"])
    key = pd.MultiIndex.from_frame(ref[["set", "fov_index", "bead", "timepoint"]])
    ref["own_pe"] = own["photoelectrons"].reindex(key).to_numpy()
    ref["own_kept"] = (own["lost_at"].reindex(key) == "kept").to_numpy()
    return ref[~ref["in_both"]].copy(), len(both)


def cached(path, record, compute):
    """``compute()`` -> {name: DataFrame}, saved to the h5 ``path`` with ``record`` (a JSON-able dict
    of everything the result depends on) beside it; reused while the record is unchanged."""
    import json
    from pathlib import Path
    path, rec_path = Path(path), Path(path).with_suffix(".json")
    record = json.loads(json.dumps(record, default=str))   # tuples -> lists, as read back
    if path.exists() and rec_path.exists() and json.loads(rec_path.read_text()) == record:
        with pd.HDFStore(path, mode="r") as store:
            print(f"Reusing {path} (same settings and fits)")
            return {k.strip("/"): store[k] for k in store.keys()}
    out = compute()
    with pd.HDFStore(path, mode="w") as store:
        for k, v in out.items():
            store.put(k, v)
    rec_path.write_text(json.dumps(record, indent=1))
    return out


# --------------------------------------------------------------------------- confusion
def wilson(k, n, z=1.96):
    """Wilson score 95 % interval for k successes out of n."""
    if n == 0:
        return np.nan, np.nan
    p, d = k / n, 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return max(centre - half, 0.0), min(centre + half, 1.0)


def confusion_counts(events, calls=None, missed_labels=None):
    """Rows YG / R reference (+ "no reference" S3M calls if ``calls`` is given), columns
    ``OUTCOMES``; and the missed references by stage ({colour: [count per stage]})."""
    counts, stages = {}, {}
    for colour in ["YG", "R"]:
        o = events.loc[events["bead_type"] == colour, "outcome"]
        counts[f"{colour} reference"] = [(o == "called YG").sum(), (o == "called R").sum(),
                                         (o == "unclassified").sum(), o.str.startswith("missed").sum()]
        if missed_labels:
            stages[colour] = [(o == m).sum() for m in missed_labels.values()]
    if calls is not None:
        counts["no reference"] = [(calls["call"] == "YG").sum(), (calls["call"] == "R").sum(),
                                  (calls["call"] == "unclassified").sum(), np.nan]
    return pd.DataFrame(counts, index=OUTCOMES).T.astype("Int64"), stages


def confusion_table(counts):
    """Counts with row totals and 95 % Wilson intervals (for the SI table / CSV)."""
    out = counts.assign(total=counts.sum(axis=1))
    for col in OUTCOMES:
        out[f"{col}, 95 % CI"] = [
            "" if pd.isna(k) or t == 0 else "[{:.1%}, {:.1%}]".format(*wilson(int(k), int(t)))
            for k, t in zip(counts[col], out["total"])]
    return out


def draw_confusion(ax, counts, show_ci=True, fontsize=5.5):
    """Confusion matrix: each cell its share of the row, k/n and (``show_ci``) the 95 % Wilson
    interval. The "no reference" row has no "missed" cell."""
    n = counts.sum(axis=1)
    frac = counts.div(n.replace(0, np.nan), axis=0)
    cmap = plt.get_cmap("gist_yarg").copy()
    cmap.set_bad("0.93")
    ax.imshow(np.ma.masked_invalid(frac.to_numpy(float, na_value=np.nan)), cmap=cmap, vmin=0,
              vmax=1, aspect="auto")
    for i, row in enumerate(counts.index):
        for j, col in enumerate(OUTCOMES):
            k = counts.loc[row, col]
            if pd.isna(k) or n[row] == 0:
                ax.text(j, i, "n/a" if pd.isna(k) else "–", ha="center", va="center",
                        fontsize=fontsize, color="0.5")
                continue
            v = frac.loc[row, col]
            text = f"{v:.1%}\n{int(k)}/{int(n[row])}"
            if show_ci:
                lo, hi = wilson(int(k), int(n[row]))
                text += f"\n[{lo:.1%}, {hi:.1%}]"
            ax.text(j, i, text, ha="center", va="center", fontsize=fontsize,
                    color="white" if v > 0.5 else "black")
    labels = {"YG reference": "YG beads", "R reference": "R beads",
              "no reference": "no reference\n(S³M calls)"}
    ax.set_xticks(range(len(OUTCOMES)), ["YG", "R", "uncl.", "missed"], fontsize=fontsize + 1)
    ax.set_yticks(range(len(counts)), [labels[r] for r in counts.index], fontsize=fontsize + 1)
    ax.tick_params(length=0)


def draw_missed_stages(ax, counts, stages, missed_labels, fontsize=5.5):
    """Missed references by the stage where they were lost, as a fraction of the row."""
    greys = plt.get_cmap("Greys")(np.linspace(0.9, 0.25, len(missed_labels)))
    left = np.zeros(2)
    for st_i, g in enumerate(greys):
        w = np.array([stages[c][st_i] / counts.loc[f"{c} reference"].sum()
                      if counts.loc[f"{c} reference"].sum() else 0 for c in ["YG", "R"]])
        ax.barh([0, 1], w, left=left, color=g, edgecolor="k", lw=0.3,
                label=list(missed_labels.values())[st_i].removeprefix("missed: "))
        left += w
    ax.set_ylim(len(counts) - 0.5, -0.5)   # rows line up with the matrix
    ax.set_yticks([])
    ax.set_xlim(0, max(left.max() * 1.25, 0.05))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.tick_params(labelsize=fontsize)
    ax.set_xlabel("missed,\nby stage", fontsize=fontsize + 1)   # two lines, as the matrix's label
    ax.grid(True, axis="x", ls="--", lw=0.6, alpha=0.6, color="grey")
    ax.set_axisbelow(True)


def range_label(lo, hi):
    return f"{lo:,.0f}–{hi:,.0f} pe"


# --------------------------------------------------------------------------- figure
def plot_figure(wavelength, spectra, fov_frames, fov_events, fov_noref, scale_bar_px,
                events, noref_calls, st: Settings, sm_range, high_ranges, hist_bins=None,
                fov_note=""):
    """Fig. S78.

    a)-c) filter transmission and bead emission per channel; d)-f) example FOV in the YG, R and
    combined channels: only what enters the confusion matrices -- d), e) the scored reference
    beads of that FOV, f) the same beads coloured by their S3M outcome plus the S3M calls with
    no reference;
    g) confusion matrix for references with ``sm_range`` photoelectrons in their own filter
    channel (single-molecule-like), with the no-reference S3M calls in the same combined-channel
    photoelectron range, and the missed references by stage; h) own-channel photoelectrons of the
    scored references, stacked by power set, with the ranges marked; i), j) confusion matrices for
    the ``high_ranges``.

    ``events``: scored references at timepoint 0 (definition b); ``noref_calls``: S3M calls at
    timepoint 0 with no reference; ``fov_events`` / ``fov_noref``: the rows of those two tables
    from the example FOV. Returns (fig, {range label: counts}).
    """
    # Fixed row heights (mm). The FOV row has no layout engine: the three images are placed edge
    # to edge (FOV_GAP_MM apart) at the full figure width, and the row is exactly as tall as they are.
    width_mm, gap_mm = 167.8, FOV_GAP_MM
    img_h, img_w = fov_frames[0].shape
    w_img = (width_mm - 2 * gap_mm) / 3
    h_img = w_img * img_h / img_w
    heights = [ROW_MM["spectra"], h_img + 1.0, ROW_MM["g"], ROW_MM["ij"]]
    fig = plt.figure(figsize=(width_mm * MM, sum(heights) * MM), layout="constrained")
    sf_spec, sf_fov, sf_g, sf_ij = fig.subfigures(4, 1, height_ratios=heights, hspace=0)
    axs_spec = list(sf_spec.subplots(1, 3))
    axs_fov = [sf_fov.add_axes([i * (w_img + gap_mm) / width_mm, 0.5 / heights[1],
                                w_img / width_mm, h_img / heights[1]]) for i in range(3)]
    gs = sf_g.add_gridspec(1, 6)
    ax_g = sf_g.add_subplot(gs[0, 0:3])
    ax_gm = sf_g.add_subplot(gs[0, 3])
    ax_h = sf_g.add_subplot(gs[0, 4:6])
    ax_i, ax_j = sf_ij.subplots(1, 2)

    plot_spectra(axs_spec, wavelength, spectra)

    for ax, img, cmap in zip(axs_fov, fov_frames, ["Greens", "Reds", "gray"]):
        ax.imshow(img, cmap=cmap, origin="lower", vmin=np.percentile(img, 1),
                  vmax=np.percentile(img, 99.9), interpolation="none")
        ax.set_axis_off()
        h, w = img.shape
        ax.plot([w * 0.95 - scale_bar_px, w * 0.95], [h * 0.05] * 2, color="white", lw=2)
        ax.text(w * 0.95 - scale_bar_px / 2, h * 0.07, "10 µm", color="white", fontsize=6,
                ha="center", va="bottom")
    # Every circle is one entry of g)-j): d), e) the scored references; f) their S3M outcome.
    for ax, b in zip(axs_fov[:2], ("YG", "R")):
        e = fov_events[fov_events["bead_type"] == b]
        ax.scatter(e["col"], e["row"], s=40, facecolor="none", edgecolor="k", lw=0.6)
    missed = fov_events["outcome"].str.startswith("missed")
    for sel, colour, ls, label in (
            (fov_events["outcome"] == "called YG", "chartreuse", "-", "called YG"),
            (fov_events["outcome"] == "called R", "red", "-", "called R"),
            (fov_events["outcome"] == "unclassified", "white", "-", "unclassified"),
            (missed, "0.6", (0, (2, 1.5)), "missed")):
        e = fov_events[sel]
        axs_fov[2].scatter(e["col"], e["row"], s=40, facecolor="none", edgecolor=colour, lw=0.6,
                           linestyle=ls, label=f"{label} ({len(e)})")
    axs_fov[2].scatter(fov_noref["xc"], fov_noref["yc"], s=40, marker="s", facecolor="none",
                       edgecolor="yellow", lw=0.6, label=f"no reference ({len(fov_noref)})")
    axs_fov[2].legend(loc="upper left", bbox_to_anchor=(0.0, 0.9), fontsize=4.5, frameon=True, framealpha=0.85,
                      handletextpad=0.2, borderpad=0.3, labelspacing=0.2)
    if fov_note:
        axs_fov[0].text(0.02, 0.88, fov_note, transform=axs_fov[0].transAxes, fontsize=5.5,
                        va="top", ha="left", color="k")

    def in_range(df, col, lo, hi):
        return df[(df[col] >= lo) & (df[col] < hi)]

    out = {}
    lo, hi = sm_range
    ev = in_range(events, "own_pe", lo, hi)
    counts, stages = confusion_counts(ev, in_range(noref_calls, "photoelectrons", lo, hi),
                                      st.missed_labels)
    draw_confusion(ax_g, counts, show_ci=True)
    sets_in = ", ".join(SET_LABELS[s] for s in SETS if (ev["set"] == s).any())
    ax_g.set_xlabel(f"S³M outcome; beads with {range_label(lo, hi)} in their own filter channel\n"
                    f"(timepoint 0; power sets {sets_in})", fontsize=6)
    draw_missed_stages(ax_gm, counts, stages, st.missed_labels)
    ax_gm.legend(fontsize=4.5, loc="lower right", frameon=False)
    out[range_label(lo, hi)] = counts

    for ax, (lo2, hi2) in zip((ax_i, ax_j), high_ranges):
        ev2 = in_range(events, "own_pe", lo2, hi2)
        c2, _ = confusion_counts(ev2, in_range(noref_calls, "photoelectrons", lo2, hi2))
        draw_confusion(ax, c2, show_ci=False)
        sets2 = ", ".join(SET_LABELS[s] for s in SETS if (ev2["set"] == s).any())
        ax.set_xlabel(f"S³M outcome; beads with {range_label(lo2, hi2)}\n(power sets {sets2})",
                      fontsize=6)
        out[range_label(lo2, hi2)] = c2

    bins = hist_bins if hist_bins is not None else np.geomspace(st.min_pe, st.max_pe, 31)
    data = [events.loc[events["set"] == s, "own_pe"].dropna() for s in SETS]
    ax_h.hist(data, bins, stacked=True, color=["0.2", "0.55", "0.85"], edgecolor="k", lw=0.3,
              label=[SET_LABELS[s] for s in SETS])
    ax_h.axvspan(*sm_range, color=STYLE["YG"]["color"], alpha=0.12, lw=0)
    for edge in sorted({*sm_range, *np.ravel(high_ranges)}):
        ax_h.axvline(edge, color="k", lw=0.6, ls=":")
    ax_h.set_xscale("log")
    ax_h.set_xlabel("photoelectrons in own filter channel", fontsize=6)
    ax_h.set_ylabel("reference beads", fontsize=6)
    ax_h.tick_params(labelsize=5.5)
    ax_h.legend(fontsize=4.5, frameon=False, loc="upper left")
    ax_h.grid(True, ls="--", lw=0.6, alpha=0.6, color="grey")
    ax_h.set_axisbelow(True)

    for ax, letter in zip(axs_spec + [ax_g, ax_h, ax_i, ax_j], "abcghij"):
        ax.text(-0.02, 1.02, f"{letter})", transform=ax.transAxes, fontsize=8, fontweight="bold",
                va="bottom", ha="right")
    for ax, letter in zip(axs_fov, "def"):   # inside the image (the images fill the row), on a box
        ax.text(0.015, 0.985, f"{letter})", transform=ax.transAxes, fontsize=8, fontweight="bold",
                va="top", ha="left", color="k",
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
    return fig, out
