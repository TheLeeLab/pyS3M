"""Shared code for ``Figure_4_Tracking_FRET.ipynb``.

Figure 4 b-c: three-colour single-protein tracking on a supported lipid bilayer (Ximea, 50 ms).
Figure 4 e-h: single-molecule FRET of a DNA Holliday junction, Cy3/Cy5 (ZWO, 10 ms).

Pipelines (as the analysis notebooks they replace):

- tracking (``tracking/20260331_Dan_RawData_Analysis`` + ``20260331_Dan_Track_Analysis``):
  ``fit_tracking_data`` -> spectral LAP linking -> KMeans on (A_R, A_G, A_B) into the three
  proteins -> OLSF MSD fit per track (D);
- FRET (``fret/DNA_HJ_Analysis_ZWO`` -> ``DNA_HJ_PostHoc_Changepoints_AllFiles`` ->
  ``FRET_FigureMaking_btc29_zwo``): ``fit_FRET_data`` -> joint (A_R, A_G) change points per
  punctum (PELT, BIC penalty) -> segment means -> apparent E by projection onto the Cy3-Cy5
  line, gamma-corrected -> two-state Gaussian mixture on the segment E -> dwell times of the
  mixture states -> exponential rates.

Fits are written beside the TIFFs; earlier fits are moved to a sibling ``old_fits/`` first.

Data: BioStudies S-BIAD3210, ``TRACK_REMOTE`` (31 fields of view, one TIFF each) and ``FRET_REMOTE``
(25 positions).
"""

from __future__ import annotations

import multiprocessing
import shutil
from concurrent import futures
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

FIT_FIX = datetime(2026, 10, 28)   # x/y ROI-origin fix: fits made earlier are refitted


# --------------------------------------------------------------------------- fits on disk
def tiff_fit_pairs(root: Path, pattern: str = "*.tif"):
    """[(tiff, fit .h5 beside it)] for every TIFF under ``root`` (recursive), sorted."""
    tifs = sorted(p for p in Path(root).rglob(pattern) if p.suffix == ".tif")
    return [(t, t.with_name(t.name.split(".")[0] + ".h5")) for t in tifs]


def fit_is_current(fit: Path) -> bool:
    return fit.exists() and datetime.fromtimestamp(fit.stat().st_mtime) > FIT_FIX


def move_old_fits(root: Path, dry_run: bool = False):
    """Move every ``.h5`` under ``root`` made before the x/y fix to ``<root>/../old_fits/<root name>/``
    (same relative path). Outside ``root``, so recursive searches of ``root`` no longer see them.
    Returns the list of (source, destination)."""
    root = Path(root)
    dest_root = root.parent / "old_fits" / root.name
    moved = []
    for p in sorted(root.rglob("*.h5")):
        if datetime.fromtimestamp(p.stat().st_mtime) > FIT_FIX:
            continue
        dest = dest_root / p.relative_to(root)
        moved.append((p, dest))
        if not dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(dest))
    return moved


# --------------------------------------------------------------------------- panel sizes
# Axes (plot area) sizes as placed in the paper's Figure_4.svg at 100 %, mm (w, h); g and h are stacks
# of equal axes, with the gap between them.
PANEL_MM = {
    "b": (45.47, 45.46),
    "c": (62.64, 38.20),
    "e": (52.68, 52.68),
    "f": (52.67, 52.67),
    "g": (110.76, 13.21, 5.65),   # three traces
    "h": (42.69, 21.97, 2.16),    # low-, high-FRET dwells
}


def mm_stack(w_mm, h_mm, n=1, gap_mm=0.0, margins_mm=(10.0, 8.0, 2.0, 2.0)):
    """A figure with ``n`` axes of exactly ``w_mm`` x ``h_mm``, stacked top to bottom ``gap_mm`` apart,
    with margins (left, bottom, right, top; mm) for labels. No layout engine, so the axes keep their
    size; saving with ``bbox_inches="tight"`` only trims the margins. Returns (fig, [axes, top first])."""
    import matplotlib.pyplot as plt
    mm = 1 / 25.4
    left, bottom, right, top = margins_mm
    W, H = w_mm + left + right, n * h_mm + (n - 1) * gap_mm + bottom + top
    fig = plt.figure(figsize=(W * mm, H * mm))
    fig.set_layout_engine("none")
    axs = [fig.add_axes([left / W, (bottom + (n - 1 - i) * (h_mm + gap_mm)) / H, w_mm / W, h_mm / H])
           for i in range(n)]
    return fig, axs


def side_colorbar(fig, ax, vmin, vmax, label, cmap="gray", width_mm=1.2, gap_mm=0.6, tick_fontsize=6,
                  label_fontsize=7):
    """A colour bar as tall as ``ax``, ``gap_mm`` to its right (the axes keeps its size)."""
    import matplotlib.pyplot as plt
    W, H = fig.get_size_inches() * 25.4
    p = ax.get_position()
    cax = fig.add_axes([p.x1 + gap_mm / W, p.y0, width_mm / W, p.height])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cb = fig.colorbar(sm, cax=cax)
    cb.ax.tick_params(labelsize=tick_fontsize, length=2, width=0.5, pad=1)
    cb.outline.set_linewidth(0.5)
    cb.set_label(label, fontsize=label_fontsize, labelpad=2)
    return cb


# --------------------------------------------------------------------------- tracking
CLUSTER_NAMES = ["ICAM-1", "CD58", "UCHT1"]        # AF488, AF555, JF646: ordered by A_R
CLUSTER_COLOURS = ["#2196F3", "#4CAF50", "#F44336"]
CLUSTER_ALPHAS = [1.0, 0.8, 0.6]                    # as Fig. 4c; 4b uses the same, so b and c match


def colour_clusters(sm: pd.DataFrame, seed: int = 42) -> np.ndarray:
    """KMeans (k = 3) on the normalised (A_R, A_G, A_B) of each molecule; labels ordered by
    mean A_R, so 0 = AF488 (ICAM-1), 1 = AF555 (CD58), 2 = JF646 (UCHT1)."""
    from sklearn.cluster import KMeans
    x = sm[["A_R", "A_G", "A_B"]].to_numpy(float)
    x = x / np.where(x.sum(1, keepdims=True) > 0, x.sum(1, keepdims=True), 1.0)
    lab = KMeans(n_clusters=3, random_state=seed, n_init=10).fit_predict(x)
    order = np.argsort([x[lab == k, 0].mean() for k in range(3)])
    remap = np.empty(3, int)
    remap[order] = np.arange(3)
    return remap[lab]


def diffusion_per_track(sm, sf, msd, pixel_nm, dt, R=1 / 6, min_frames=5):
    """OLSF MSD fit per molecule: D (um^2/s) and localisation variance sigma^2 (nm^2)."""
    D = np.full(len(sm), np.nan)
    s2 = np.full(len(sm), np.nan)
    groups = {k: g.sort_values("frame") for k, g in sf.groupby("molecular_index")}
    for i, mid in enumerate(sm["molecular_index"].to_numpy()):
        tr = groups.get(mid)
        if tr is None or len(tr) < min_frames:
            continue
        xy = tr[["xc", "yc"]].to_numpy(float) * pixel_nm
        d, sig2 = msd.DSigma2_OLSF(xy, dt, R=R, n_d=2, min_points=min_frames)
        D[i], s2[i] = d / 1e6, sig2
    return D, s2


# --------------------------------------------------------------------------- deposition
TRACK_REMOTE = "2026_Multicolour_Paper/Ximea/LipidBilayers/AF488ICAM1_AF555CD58_JF646UCHT1"
TRACK_FOVS = ([f"1nM_Total/Well{w}/4banddichroic_NF_{i}" for w in (1, 2, 3) for i in range(1, 7)]
              + ["250pM_Total/Well1/4banddichroic_NF_1"]
              + [f"500pM_Total/Well{w}/4banddichroic_NF_{i}" for w in (1, 2, 3, 4) for i in (1, 2, 3)])
FRET_REMOTE = "2026_Multicolour_Paper/ZWO/20260402_HJs/50mM_MgCl_20pMHJ_515LP650200BP_100perc515_2"
FRET_FAST_STEM = "50mM_MgCl_20pMHJ_515LP650200BP_100perc515_2_MMStack_Pos-0-0"   # the pictured position

# Fig. 4b: one track per protein, as {cluster: (field of view, first frame, first (xc, yc) in px)}.
# Picked from the refit as the tracks closest in shape to the paper's exemplars; UCHT1 is the
# paper's own molecule.
EXEMPLARS = {
    0: ("500pM_Total/Well1/4banddichroic_NF_2", 267, (570.5, 455.5)),   # ICAM-1
    1: ("500pM_Total/Well4/4banddichroic_NF_3", 862, (619.4, 221.9)),   # CD58
    2: ("1nM_Total/Well1/4banddichroic_NF_1", 43, (221.5, 86.4)),       # UCHT1
}
TRACK_FAST_FOVS = sorted({fov for fov, _, _ in EXEMPLARS.values()})   # FAST_MODE: the exemplars' fields


def find_exemplar(sm: pd.DataFrame, sf: pd.DataFrame, track_root: Path, fov: str, first_frame: int,
                  xy, max_frames: int = 3, max_px: float = 2.0) -> int:
    """``molecular_index`` of the track in field of view ``fov`` (relative to ``track_root``) that
    starts within ``max_frames`` frames and ``max_px`` px of (``first_frame``, ``xy``)."""
    folder = Path(track_root) / fov
    ids = sm.loc[sm["fov_name"].map(lambda f: Path(f).parent == folder), "molecular_index"]
    first = sf[sf["molecular_index"].isin(ids)].sort_values("frame").groupby("molecular_index").first()
    d = np.hypot(first["xc"] - xy[0], first["yc"] - xy[1])
    ok = (first["frame"] - first_frame).abs().le(max_frames) & d.le(max_px)
    if not ok.any():
        raise LookupError(f"no track in {fov} starts near frame {first_frame}, ({xy[0]}, {xy[1]})")
    return int(d[ok].idxmin())


# --------------------------------------------------------------------------- FRET: dyes
def cy3_cy5_fingerprints(S_F, filters, qy_cy3=0.16, qy_cy5=0.29):
    """Pixel fractions (B, G, R) of Cy3 and Cy5 through ``filters`` on this camera, their
    detection efficiencies and gamma = (eta_Cy5 QY_Cy5) / (eta_Cy3 QY_Cy3)."""
    R, G, B, wl = S_F.getpixelefficiency()
    qys = np.vstack([B, G, R])
    filt = np.prod(S_F.get_dye_or_filter_data(names=filters, wavelength=wl, dye_or_filter=False), axis=0)
    filt = filt * S_F.getobjectiveefficiency(wl)
    out = {}
    for dye in ("Cy3", "Cy5"):
        _, eff = S_F.get_pixel_fractions_dye_and_filters(dye, filters, wl, qys)
        spec = S_F.get_dye_or_filter_data(dye, wavelength=wl)
        at_det = np.array(np.multiply(spec, filt).T / np.sum(spec, axis=1)).T
        out[dye] = dict(fractions=eff / np.sum(eff), detection=float(np.sum(at_det * qys)))
    gamma = (out["Cy5"]["detection"] * qy_cy5) / (out["Cy3"]["detection"] * qy_cy3)
    return out["Cy3"]["fractions"], out["Cy5"]["fractions"], gamma


def apparent_E(abg: np.ndarray, f_cy3: np.ndarray, f_cy5: np.ndarray):
    """Projection of (A_B, A_G, A_R) rows onto the Cy3-Cy5 line: E = 0 at Cy3, 1 at Cy5.
    Returns (E, squared residual off the line)."""
    d = f_cy3 - f_cy5
    E = 1 - (abg - f_cy5) @ d / (d @ d)          # Brendan's projection (alpha = 1 - E)
    recon = (1 - E)[:, None] * f_cy3 + E[:, None] * f_cy5
    return E, np.sum((abg - recon) ** 2, axis=1)


def gamma_correct(E_app, gamma):
    return E_app / (E_app + gamma * (1 - E_app))


# --------------------------------------------------------------------------- FRET: change points
def filter_initial_guess(df, tol=0.01):
    """Drop rows whose A_R, A_G, A_B are all ~0.33 (a fit that never left its guess)."""
    m = (np.isclose(df["A_R"], 0.33, atol=tol) & np.isclose(df["A_G"], 0.33, atol=tol)
         & np.isclose(df["A_B"], 0.33, atol=tol))
    return df[~m].copy(), int(m.sum())


def find_colour_CPs(A_R, A_G, model="l2", min_size=25, jump=1):
    """PELT change points jointly in (A_R, A_G), BIC penalty log(n) * dim * sigma^2.
    Returns indices; the last is always len(signal)."""
    import ruptures as rpt
    n = len(A_R)
    if n < min_size:
        return [n]
    signal = np.column_stack([A_R, A_G])
    sigma2 = np.nanmean([np.nanvar(A_R), np.nanvar(A_G)])
    if sigma2 == 0:
        return [n]
    return rpt.Pelt(model=model, min_size=min_size, jump=jump).fit(signal).predict(
        pen=np.log(n) * signal.shape[1] * sigma2)


def _cp_worker(args):
    pid, A_R, A_G, model, min_size, jump = args
    return pid, find_colour_CPs(A_R, A_G, model, min_size, jump)


def change_points(df, model="l2", min_size=25, jump=1, n_workers=None):
    """{puncta_id: change-point indices} for every punctum with at least one change point."""
    tasks = [(pid, g["A_R"].to_numpy(float), g["A_G"].to_numpy(float), model, min_size, jump)
             for pid, g in df.sort_values("frame").groupby("puncta_id")]
    n_workers = n_workers or min(60, max(1, int(0.9 * multiprocessing.cpu_count())))
    out = {}
    with futures.ProcessPoolExecutor(n_workers) as ex:
        for pid, cps in ex.map(_cp_worker, tasks, chunksize=64):
            if len(cps) > 1:
                out[pid] = cps
    return out


def segment_table(df, cps):
    """One row per segment between change points (frames, means and s.d. of the colours)."""
    rows = []
    groups = {k: g.sort_values("frame").reset_index(drop=True) for k, g in df.groupby("puncta_id")}
    for pid, c in cps.items():
        sub = groups[pid]
        bounds = [0] + list(c[:-1]) + [len(sub)]
        for k, (s, e) in enumerate(zip(bounds[:-1], bounds[1:]), start=1):
            seg = sub.iloc[s:e]
            if not len(seg):
                continue
            rows.append(dict(puncta_id=pid, segment_number=k, frame_start=int(seg["frame"].min()),
                             frame_end=int(seg["frame"].max()), n_points=len(seg),
                             **{f"{col}_mean": seg[col].mean() for col in ("A_R", "A_G", "A_B", "photons")},
                             **{f"{col}_std": seg[col].std() for col in ("A_R", "A_G", "A_B", "photons")}))
    return pd.DataFrame(rows)


def changepoint_table(df, cps):
    rows = []
    groups = {k: g.sort_values("frame").reset_index(drop=True) for k, g in df.groupby("puncta_id")}
    for pid, c in cps.items():
        f = groups[pid]["frame"].to_numpy()
        rows += [dict(puncta_id=pid, cp_number=i + 1, cp_index=int(ci), frame=int(f[ci]))
                 for i, ci in enumerate(c[:-1]) if ci < len(f)]
    return pd.DataFrame(rows, columns=["puncta_id", "cp_number", "cp_index", "frame"])


def save_changepoints(path, df, cps):
    df.to_hdf(path, key="data", mode="w", complevel=5, complib="blosc")
    changepoint_table(df, cps).to_hdf(path, key="changepoints", mode="a", complevel=5, complib="blosc")
    segment_table(df, cps).to_hdf(path, key="segments", mode="a", complevel=5, complib="blosc")


def load_changepoints(path):
    return (pd.read_hdf(path, key="data"), pd.read_hdf(path, key="changepoints"),
            pd.read_hdf(path, key="segments"))


# --------------------------------------------------------------------------- FRET: states and dwells
def two_state_gmm(E, seed=0):
    """Two-component Gaussian mixture on the segment E values in [0, 1]. Returns the fitted
    mixture and the index of the high-FRET component (higher mean)."""
    from sklearn.mixture import GaussianMixture
    x = np.asarray(E, float)
    x = x[(x >= 0) & (x <= 1)].reshape(-1, 1)
    gmm = GaussianMixture(2, covariance_type="full", random_state=seed).fit(x)
    return gmm, int(np.argmax(gmm.means_.ravel()))


def state_dwells(seg, high_label, dt):
    """Dwell times (s) of the mixture states.

    ``seg`` needs ``puncta_id``, ``segment_number``, ``frame_start`` and ``state`` (mixture label).
    Consecutive segments in the same state are merged into one dwell (a change point between them
    is not a transition). A dwell lasts from its first frame to the first frame of the next
    dwell; the first and last dwell of each trace are cut by the start/end of the observation and
    are dropped. Returns a DataFrame (puncta_id, state 'high'/'low', dwell_s).
    """
    rows = []
    for pid, g in seg.sort_values(["puncta_id", "segment_number"]).groupby("puncta_id"):
        st = g["state"].to_numpy()
        start = g["frame_start"].to_numpy()
        new = np.r_[True, st[1:] != st[:-1]]           # first segment of each run
        run_state, run_start = st[new], start[new]
        for k in range(1, len(run_state) - 1):          # interior runs only
            rows.append(dict(puncta_id=pid, state="high" if run_state[k] == high_label else "low",
                             dwell_s=(run_start[k + 1] - run_start[k]) * dt))
    return pd.DataFrame(rows, columns=["puncta_id", "state", "dwell_s"])


def exp_rate(dwells, t_min, bins="sqrt"):
    """Exit rate of one state from its dwell times (>= t_min, the minimum segment length).

    Two estimates: the paper's binned fit of a * exp(-k t) to the histogram (Poisson weights,
    sqrt(counts)), and the maximum-likelihood rate of an exponential truncated at t_min,
    k = 1 / mean(t - t_min), with s.e. k / sqrt(n).
    Returns dict(k_fit, k_fit_err, a, edges, counts, k_mle, k_mle_err, n).
    """
    from scipy.optimize import curve_fit
    t = np.asarray(dwells, float)
    t = t[t >= t_min]
    counts, edges = np.histogram(t, bins=np.histogram_bin_edges(t, bins))
    centres = 0.5 * (edges[:-1] + edges[1:])
    ok = counts > 0
    f = lambda x, a, k: a * np.exp(-k * x)
    popt, pcov = curve_fit(f, centres[ok], counts[ok], p0=[counts.max() * np.exp(centres[0]), 1 / t.mean()],
                           sigma=np.sqrt(counts[ok]), absolute_sigma=True, maxfev=10000)
    k_mle = 1 / np.mean(t - t_min)
    return dict(k_fit=popt[1], k_fit_err=float(np.sqrt(pcov[1, 1])), a=popt[0], edges=edges,
                counts=counts, k_mle=k_mle, k_mle_err=k_mle / np.sqrt(len(t)), n=len(t))
