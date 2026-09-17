"""
within_basin_gpcca.py
─────────────────────
Re-runs GPCCA independently on each M=2 basin ("arm") from the global run
and generates notebook-02-style diagnostics for each parent basin:

  • Eigenvalue spectra of the within-basin operator
  • 3-D and 2-D GPCCA sub-arm scatter (native within-basin eigenspace)
  • 3-D and 2-D GPCCA sub-arm scatter (parent shared eigenspace)
  • Per-sub-arm species cloud plots
  • Cosine-similarity matrix of sub-arm centroids
  • UMAP density and log₂ enrichment panels (per species & per sub-arm)
  • MoSeq syllable probability & enrichment heatmaps per sub-arm

Usage
─────
python within_basin_gpcca.py [--view VIEW] [--m-min M_MIN] [--m-max M_MAX]
                              [--chi-threshold CHI_THRESH]
                              [--save-figs] [--fig-dir FIG_DIR]

All paths are derived automatically from the RUN_ROOT constant below.
Edit that one line if your outputs live elsewhere.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from scipy.ndimage import gaussian_filter
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# 0.  CONFIGURE  ── edit this one line ────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
RUN_ROOT = Path(
    "/Users/meganbishop/slowmodeevo/outputs/"
    "multispecies_slow_modes/global_clustering_modes/"
    "global_outputs/XY_EvenSampled_SlowModes"
)

EXPORT_ROOT = RUN_ROOT / "saved_slow_mode_arrays"

# ──────────────────────────────────────────────────────────────────────────────
# 1.  DEFAULTS (overridden by CLI)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_VIEW          = "with_low_occupancy_cutoff"
DEFAULT_M_MIN         = 2          # minimum sub-arm count to try
DEFAULT_M_MAX         = 6          # maximum sub-arm count to try
DEFAULT_CHI_THRESH    = 0.5        # chi membership threshold for hard assignment
DEFAULT_MIN_CLUSTERS  = 10         # skip a basin if fewer than this many clusters
DENSITY_BINS          = 90
DENSITY_SIGMA         = 1.25
DENSITY_VMAX_P        = 99.5
ENRICH_VMAX           = 2.5
EPS                   = 1e-10
MOSEQ_TOP_N           = 30
N_PARENT_BASINS       = 2          # M from the global run (adjusted automatically)

# ──────────────────────────────────────────────────────────────────────────────
# 2.  ARGUMENT PARSING
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--view",          default=DEFAULT_VIEW,
                   choices=["with_low_occupancy_cutoff", "without_low_occupancy_cutoff"])
    p.add_argument("--m-min",         type=int,   default=DEFAULT_M_MIN)
    p.add_argument("--m-max",         type=int,   default=DEFAULT_M_MAX)
    p.add_argument("--chi-threshold", type=float, default=DEFAULT_CHI_THRESH,
                   help="Chi membership threshold for hard basin assignment")
    p.add_argument("--save-figs",     action="store_true",
                   help="Save each figure instead of (or in addition to) showing it")
    p.add_argument("--fig-format",    default="jpeg",
                   choices=["jpeg", "jpg", "png", "pdf", "svg"],
                   help="Image format for saved figures (default: jpeg)")
    p.add_argument("--fig-dpi",       type=int, default=150,
                   help="DPI for saved raster figures (default: 150)")
    p.add_argument("--fig-dir",       type=Path,
                   default=Path.home() / "Desktop" / "within_basin_gpcca_figs",
                   help="Directory for saved figures (default: ~/Desktop/within_basin_gpcca_figs)")
    p.add_argument("--m-inner",       type=int,   default=None,
                   help="Fix M for within-basin GPCCA (overrides spectral-gap selection). "
                        "If omitted, each basin picks M by majority vote of per-species "
                        "spectral gaps.")
    p.add_argument("--no-show",       action="store_true",
                   help="Do not call plt.show() (useful in batch mode with --save-figs)")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# 3.  DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_species_arrays(view: str) -> dict:
    """Load all saved macro arrays for one view.  Returns {species: {name: array}}."""
    view_dir = EXPORT_ROOT / view
    if not view_dir.exists():
        raise FileNotFoundError(f"View directory not found: {view_dir}")
    out = {}
    for sp_dir in sorted(view_dir.iterdir()):
        if not sp_dir.is_dir():
            continue
        sp = sp_dir.name
        arrays = {}
        for f in sp_dir.glob("*.npy"):
            arrays[f.stem] = np.load(f, allow_pickle=False)
        out[sp] = arrays
    return out


def load_reference_axes() -> dict:
    ref_dir = EXPORT_ROOT / "pooled_full_reference_axes"
    return {f.stem: np.load(f, allow_pickle=False) for f in sorted(ref_dir.glob("*.npy"))}


# ──────────────────────────────────────────────────────────────────────────────
# 4.  WITHIN-BASIN GPCCA
# ──────────────────────────────────────────────────────────────────────────────

def stationary_from_T(T: np.ndarray) -> np.ndarray:
    """Stationary distribution of a row-stochastic T via left eigenvector."""
    vals, vecs = np.linalg.eig(T.T)
    idx = int(np.argmin(np.abs(vals - 1.0)))
    pi = np.abs(vecs[:, idx].real)
    pi = np.maximum(pi, 1e-14)
    return pi / pi.sum()


def leading_eigvals(T: np.ndarray, k: int = 10) -> np.ndarray:
    """Top-k eigenvalue magnitudes of T (excluding lambda=1)."""
    k = min(k, T.shape[0] - 1)
    vals, _ = np.linalg.eig(T)
    mags = np.sort(np.abs(vals))[::-1]
    # drop the stationary eigenvalue (magnitude ≈ 1)
    mags = mags[mags < 0.9999]
    return mags[:k]


def select_spectral_gap_M(eigvals: np.ndarray, m_min: int, m_max: int) -> tuple[int, np.ndarray]:
    """Pick M by the largest |λ_M|/|λ_{M+1}| ratio gap."""
    e = np.asarray(eigvals, dtype=float)
    m_max = min(m_max, len(e))
    Ms = np.arange(m_min, m_max + 1)
    ratios = np.array([e[m - 2] / max(e[m - 1], 1e-12) for m in Ms])
    best = int(Ms[np.argmax(ratios)])
    return best, ratios


def subset_and_renorm_T(T: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Subset T to `mask` rows/cols and re-normalise to be row-stochastic.

    Rows that sum to zero after subsetting are given a self-loop to keep the
    matrix valid (these clusters had no transitions within the arm during the
    original estimation window).
    """
    T_sub = T[np.ix_(mask, mask)].copy()
    row_sums = T_sub.sum(axis=1, keepdims=True)
    zero_rows = (row_sums.ravel() < 1e-12)
    # self-loop for isolated rows
    for i in np.where(zero_rows)[0]:
        T_sub[i, i] = 1.0
        row_sums[i, 0] = 1.0
    T_sub /= row_sums
    return T_sub


def _prepare_basin(
    T_macro: np.ndarray,
    pi_macro: np.ndarray,
    chi_macro: np.ndarray,
    arm_idx: int,
    chi_threshold: float,
    m_min: int,
    m_max: int,
) -> dict | None:
    """Subset T to the arm and compute eigenvalues + spectral-gap M suggestion.

    Returns a 'probe' dict (no GPCCA fit yet), or None if the basin is too small.
    Used in the first pass so a consensus M can be chosen across species before
    running GPCCA at that shared M.
    """
    arm_hard  = (np.argmax(chi_macro, axis=1) == arm_idx)
    soft_mask = (chi_macro[:, arm_idx] > chi_threshold) | arm_hard
    mask      = soft_mask
    n_in      = int(mask.sum())

    if n_in < DEFAULT_MIN_CLUSTERS:
        return None

    T_sub  = subset_and_renorm_T(T_macro, mask)
    pi_sub = stationary_from_T(T_sub)

    k_ev    = min(m_max + 4, n_in - 1)
    eigvals = leading_eigvals(T_sub, k=k_ev)
    if len(eigvals) < m_min:
        return None

    M_suggest, gap_ratios = select_spectral_gap_M(
        eigvals, m_min, min(m_max, len(eigvals))
    )
    return {
        "arm_idx":    arm_idx,
        "mask":       mask,
        "n_clusters": n_in,
        "T_sub":      T_sub,
        "pi_sub":     pi_sub,
        "eigvals":    eigvals,
        "gap_ratios": gap_ratios,
        "M_suggest":  M_suggest,
    }


def _fit_gpcca(probe: dict, M_inner: int) -> dict | None:
    """Run GPCCA at the given M on a pre-prepared probe dict.

    Returns the full result dict (probe fields + GPCCA outputs), or None on failure.
    """
    arm_idx = probe["arm_idx"]
    T_sub   = probe["T_sub"]
    pi_sub  = probe["pi_sub"]
    n_in    = probe["n_clusters"]

    if M_inner > n_in - 1:
        print(f"  Basin {arm_idx}: M_inner={M_inner} too large for {n_in} clusters — skip")
        return None

    try:
        from gpcca_utils import run_gpcca
        gpcca_out = run_gpcca(T_sub, M_inner, eta=pi_sub)
    except Exception as exc:
        print(f"  Basin {arm_idx}: GPCCA failed at M={M_inner} — {exc}")
        return None

    chi_inner  = gpcca_out["chi"]
    asgn_inner = gpcca_out["assignments"]
    crispness  = gpcca_out["crispness"]

    vals, vecs = np.linalg.eig(T_sub)
    order      = np.argsort(-np.abs(vals))
    phi_native = vecs[:, order].real[:, 1:4]    # (n_in, 3) non-trivial eigenvectors

    return {
        **probe,
        "M_inner":    M_inner,
        "chi_inner":  chi_inner,
        "asgn_inner": asgn_inner,
        "crispness":  crispness,
        "phi_native": phi_native,
    }


def run_gpcca_on_basin(
    T_macro: np.ndarray,
    pi_macro: np.ndarray,
    chi_macro: np.ndarray,
    arm_idx: int,
    chi_threshold: float,
    m_min: int,
    m_max: int,
    M_inner: int | None = None,
) -> dict | None:
    """Convenience wrapper: probe + fit in one call (used for the summary table)."""
    probe = _prepare_basin(T_macro, pi_macro, chi_macro, arm_idx,
                           chi_threshold, m_min, m_max)
    if probe is None:
        print(f"  Basin {arm_idx}: too few clusters — skip")
        return None
    M = M_inner if M_inner is not None else probe["M_suggest"]
    return _fit_gpcca(probe, M)


# ──────────────────────────────────────────────────────────────────────────────
# 5.  PLOT HELPERS (adapted from notebook 02)
# ──────────────────────────────────────────────────────────────────────────────

def normalise_rows(X):
    X = np.asarray(X, dtype=float)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def arm_centroid_in_space(phi, chi, pi):
    """π-weighted chi centroid for each sub-arm in phi-space."""
    phi = np.asarray(phi, dtype=float)
    chi = np.asarray(chi, dtype=float)
    pi  = np.asarray(pi,  dtype=float)
    pi  = pi / max(pi.sum(), 1e-12)
    centroids = np.zeros((chi.shape[1], phi.shape[1]))
    for arm in range(chi.shape[1]):
        w = chi[:, arm] * pi
        w_sum = w.sum()
        centroids[arm] = (phi * w[:, None]).sum(0) / max(w_sum, 1e-12)
    return centroids


def _show_or_save(fig, title: str, args, subfolder: str = ""):
    if args.save_figs:
        fig_dir = args.fig_dir / subfolder if subfolder else args.fig_dir
        fig_dir.mkdir(parents=True, exist_ok=True)
        safe = title.replace(" ", "_").replace("/", "-").replace(":", "")
        fmt  = args.fig_format.lstrip(".")   # normalise "jpg" → "jpeg" for matplotlib
        if fmt == "jpg":
            fmt = "jpeg"
        ext  = "jpg" if fmt == "jpeg" else fmt
        fpath = fig_dir / f"{safe}.{ext}"
        fig.savefig(fpath, dpi=args.fig_dpi, bbox_inches="tight", format=fmt)
        print(f"  Saved: {fpath}")
    if not args.no_show:
        plt.show()
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 6.  CROSS-SPECIES HUNGARIAN ALIGNMENT OF SUB-ARMS
# ──────────────────────────────────────────────────────────────────────────────

def align_sub_arms(valid: dict, species_arrays: dict) -> dict:
    """Align sub-arm labels across species using Hungarian matching in the
    parent shared eigenspace.

    Strategy
    --------
    1. For each species, compute the π-weighted centroid of each sub-arm in
       ``phi_macro_shared`` (the parent-run shared eigenspace that is common
       to all species).
    2. Choose the species with the most in-basin clusters as the reference.
    3. For every other species, build a cosine-similarity matrix between its
       sub-arm centroids and the reference centroids, then solve the linear
       assignment problem (maximise total cosine similarity) to find the
       optimal column permutation.
    4. Permute ``chi_inner`` columns and re-derive ``asgn_inner`` accordingly.

    Returns the same ``valid`` dict with updated ``chi_inner``, ``asgn_inner``,
    and a new ``perm_from_ref`` key recording the permutation applied.
    """
    # ── Compute centroids in shared eigenspace for every species ──────────────
    def shared_centroids(sp, r):
        phi_full = species_arrays[sp].get("phi_macro_shared")
        if phi_full is None or phi_full.shape[1] < 3:
            return None
        phi = phi_full[r["mask"], :3]
        return arm_centroid_in_space(phi, r["chi_inner"], r["pi_sub"])  # (M, 3)

    centroids = {}
    for sp, r in valid.items():
        c = shared_centroids(sp, r)
        if c is not None:
            centroids[sp] = c

    if len(centroids) < 2:
        print("  Alignment: fewer than 2 species have shared coords — skipping.")
        for r in valid.values():
            r["perm_from_ref"] = list(range(r["M_inner"]))
        return valid

    # ── Pick reference: species with most in-basin clusters ───────────────────
    ref_sp = max(centroids, key=lambda sp: valid[sp]["n_clusters"])
    ref_c  = centroids[ref_sp]                     # (M, 3)
    ref_unit = normalise_rows(ref_c)
    print(f"  Alignment reference species: {ref_sp}")

    # ── Align every species to the reference ──────────────────────────────────
    for sp, r in valid.items():
        if sp not in centroids:
            r["perm_from_ref"] = list(range(r["M_inner"]))
            continue

        c    = centroids[sp]                        # (M, 3)
        unit = normalise_rows(c)
        # cosine similarity matrix: (M_ref × M_sp) → we want to match rows
        cos_sim = ref_unit @ unit.T                 # (M, M)
        # linear_sum_assignment minimises cost → negate for maximisation
        row_ind, col_ind = linear_sum_assignment(-cos_sim)
        # col_ind[i] = which of this species' sub-arms maps to reference arm i
        perm = col_ind.tolist()

        if sp == ref_sp:
            r["perm_from_ref"] = list(range(r["M_inner"]))
            continue

        # Permute chi columns so column j → reference sub-arm j
        r["chi_inner"]     = r["chi_inner"][:, perm]
        r["asgn_inner"]    = np.argmax(r["chi_inner"], axis=1)
        r["perm_from_ref"] = perm
        print(f"  {sp}: permutation {perm}  "
              f"(cos-sim diag after align: "
              f"{cos_sim[row_ind, col_ind].round(3).tolist()})")

    return valid


# ──────────────────────────────────────────────────────────────────────────────
# 7.  PER-BASIN ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def analyse_basin(
    parent_arm_idx: int,
    species_arrays: dict,
    species_names: list,
    cluster_layout_2d: np.ndarray | None,
    moseq_probs: pd.DataFrame | None,
    args,
):
    """Run within-basin GPCCA for all species and produce all plots."""

    arm_label = f"Parent basin {parent_arm_idx + 1}"
    print(f"\n{'='*70}")
    print(f"  {arm_label}")
    print(f"{'='*70}")

    # ── Pass 1: probe all species — compute sub-T and spectral-gap M suggestions ─
    probes = {}
    for sp in species_names:
        d   = species_arrays[sp]
        T   = d.get("transfer_operator_macro")
        chi = d.get("chi_macro_aligned")
        pi  = d.get("pi_macro")
        if T is None or chi is None or pi is None:
            print(f"  {sp}: missing arrays — skip")
            continue
        if parent_arm_idx >= chi.shape[1]:
            print(f"  {sp}: parent_arm_idx {parent_arm_idx} out of range — skip")
            continue
        probe = _prepare_basin(T, pi, chi, parent_arm_idx,
                               args.chi_threshold, args.m_min, args.m_max)
        if probe is None:
            print(f"  {sp}: too few clusters in basin {parent_arm_idx} — skip")
        probes[sp] = probe   # None entries stay so we can report skipped species

    valid_probes = {sp: p for sp, p in probes.items() if p is not None}
    if not valid_probes:
        print(f"  No valid species for {arm_label} — skipping plots.")
        return

    # ── Choose a shared M for this basin ─────────────────────────────────────
    if args.m_inner is not None:
        # User forced a specific M
        M_shared = args.m_inner
        print(f"\n  M_inner forced by --m-inner: M = {M_shared}")
    else:
        # Majority vote across per-species spectral-gap suggestions
        suggestions = [p["M_suggest"] for p in valid_probes.values()]
        counts      = {}
        for m in suggestions:
            counts[m] = counts.get(m, 0) + 1
        M_shared = max(counts, key=counts.get)
        suggestion_str = "  ".join(
            f"{sp}→M={p['M_suggest']}" for sp, p in valid_probes.items()
        )
        print(f"\n  Per-species spectral-gap suggestions: {suggestion_str}")
        print(f"  Shared M (majority vote): M = {M_shared}  "
              f"(votes: {dict(sorted(counts.items()))})")
        print(f"  Override with --m-inner N to force a different value.\n")

    # ── Pass 2: fit GPCCA at the shared M for every valid species ────────────
    results = {}
    for sp, probe in valid_probes.items():
        print(f"  Fitting {sp}  (n_clusters={probe['n_clusters']}, M={M_shared}) …")
        res = _fit_gpcca(probe, M_shared)
        if res is not None:
            print(f"    crispness = {res['crispness']:.3f}")
        results[sp] = res

    valid = {sp: r for sp, r in results.items() if r is not None}
    if not valid:
        print(f"  All GPCCA fits failed for {arm_label} — skipping plots.")
        return

    # ── Pass 3: Hungarian alignment of sub-arms across species ───────────────
    valid = align_sub_arms(valid, species_arrays)

    sub_arm_cmap   = plt.get_cmap("Set2")
    sub_arm_colors = [sub_arm_cmap(i / max(M_shared - 1, 1)) for i in range(M_shared)]

    sp_cmap   = plt.get_cmap("tab10")
    sp_colors = {sp: sp_cmap(i / max(len(species_names) - 1, 1))
                 for i, sp in enumerate(species_names)}

    folder = f"basin_{parent_arm_idx + 1}"

    # ── Plot A: Eigenvalue spectra ────────────────────────────────────────────
    plot_eigenvalue_spectra(valid, arm_label, sub_arm_colors, args, folder)

    # ── Plot B: Native within-basin eigenspace (3-D + 2-D) ───────────────────
    plot_native_eigenspace(valid, arm_label, sub_arm_colors, species_names, args, folder)

    # ── Plot C: Parent shared eigenspace (3-D + 2-D) ─────────────────────────
    plot_shared_eigenspace(valid, arm_label, sub_arm_colors, species_names,
                           species_arrays, args, folder)

    # ── Plot D: Per-sub-arm species clouds (shared eigenspace) ────────────────
    plot_sub_arm_clouds(valid, arm_label, species_names, sp_colors,
                        species_arrays, args, folder)

    # ── Plot E: Cosine-similarity matrix ─────────────────────────────────────
    plot_cosine_similarity(valid, arm_label, species_arrays, species_names, args, folder)

    # ── Plot F: UMAP density & enrichment ────────────────────────────────────
    if cluster_layout_2d is not None:
        plot_umap_panels(valid, arm_label, species_names, sp_colors,
                         species_arrays, cluster_layout_2d, args, folder)

    # ── Plot G: MoSeq syllable heatmaps (parent arm context) ─────────────────
    if moseq_probs is not None:
        plot_moseq_heatmaps(valid, parent_arm_idx, arm_label,
                            species_arrays, moseq_probs, args, folder)

    # ── Plot H: Sub-basin syllable bias  (notebook-01 Cell-20 style) ─────────
    # Only meaningful for the large basin (basin 1 / arm_idx=0).  Basin 2 is
    # typically too small for reliable sub-arm syllable statistics.
    if moseq_probs is not None and parent_arm_idx == 0:
        plot_subbasin_syllable_bias(valid, parent_arm_idx, arm_label,
                                    moseq_probs, args, folder)


# ──────────────────────────────────────────────────────────────────────────────
# 7.  INDIVIDUAL PLOT FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def plot_eigenvalue_spectra(valid, arm_label, sub_arm_colors, args, folder):
    n_sp   = len(valid)
    n_cols = min(4, n_sp)
    n_rows = max(1, int(np.ceil(n_sp / n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5 * n_cols, 2.8 * n_rows),
                             constrained_layout=True, squeeze=False)
    for ax, (sp, r) in zip(axes.flat, valid.items()):
        ev = r["eigvals"]
        ax.stem(np.arange(1, len(ev) + 1), ev, basefmt=" ")
        ax.axhline(1.0, color="k", lw=0.6, ls="--")
        M_inner = r["M_inner"]
        ax.axvline(M_inner, color="crimson", lw=1.2, ls="--",
                   label=f"Selected M={M_inner}")
        ax.set_title(f"{sp.replace('_', ' ')}", fontsize=8)
        ax.set_xlabel("λ rank"); ax.set_ylabel("|λ|")
        ax.legend(fontsize=7, frameon=False)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(valid):]:
        ax.set_visible(False)
    title = f"{arm_label} — within-basin eigenvalue spectra"
    fig.suptitle(title)
    _show_or_save(fig, title, args, folder)


def _common_limits(phi_list, pad=0.06):
    """Compute common x/y/z axis limits across multiple point clouds."""
    all_pts = np.vstack([p for p in phi_list if p is not None and len(p)])
    all_pts = all_pts[np.isfinite(all_pts).all(1)]
    if len(all_pts) == 0:
        return [(-1, 1)] * 3
    lo, hi = all_pts.min(0), all_pts.max(0)
    p = pad * np.maximum(hi - lo, 1e-12)
    return [(lo[d] - p[d], hi[d] + p[d]) for d in range(min(3, all_pts.shape[1]))]


def _scatter_arm(ax3, ax2, phi, chi, pi, asgn, occ, sub_arm_colors, M_inner):
    """Draw 3-D + 2-D scatter coloured by sub-arm."""
    occ = occ / max(occ.sum(), 1e-12)
    sizes = 6 + 44 * np.sqrt(occ / max(occ.max(), 1e-12))
    for arm in range(M_inner):
        mask = (asgn == arm) & np.isfinite(phi[:, :3]).all(1) & (occ > 0)
        if not mask.any():
            continue
        col = sub_arm_colors[arm] if arm < len(sub_arm_colors) else "grey"
        ax3.scatter(*phi[mask, :3].T, s=sizes[mask], color=col, alpha=0.45,
                    linewidths=0, label=f"Sub-arm {arm+1}")
        ax2.scatter(phi[mask, 0], phi[mask, 1], s=sizes[mask], color=col,
                    alpha=0.6, linewidths=0, label=f"Sub-arm {arm+1}")
    # centroids
    ctrs = arm_centroid_in_space(phi[:, :3], chi, pi)
    for arm, ctr in enumerate(ctrs):
        col = sub_arm_colors[arm] if arm < len(sub_arm_colors) else "grey"
        ax3.scatter(*ctr, s=90, marker="X", color=col, edgecolors="k",
                    linewidths=0.7, zorder=6)
        ax2.scatter(ctr[0], ctr[1], s=90, marker="X", color=col,
                    edgecolors="k", linewidths=0.7, zorder=6)


def plot_native_eigenspace(valid, arm_label, sub_arm_colors, species_names, args, folder):
    """3-D and 2-D scatter in the within-basin native eigenspace."""
    n_sp   = len(valid)
    n_cols = min(4, n_sp)
    n_rows = max(1, int(np.ceil(n_sp / n_cols)))

    # Common limits
    phi_list = [r["phi_native"][:, :3] for r in valid.values() if r["phi_native"].shape[1] >= 3]
    lims = _common_limits(phi_list)

    fig3 = plt.figure(figsize=(5 * n_cols, 4.5 * n_rows), constrained_layout=True)
    fig2, axes2 = plt.subplots(n_rows, n_cols,
                               figsize=(4.2 * n_cols, 3.5 * n_rows),
                               constrained_layout=True, squeeze=False)

    for idx, (sp, r) in enumerate(valid.items(), 1):
        phi   = r["phi_native"]
        chi   = r["chi_inner"]
        pi    = r["pi_sub"]
        asgn  = r["asgn_inner"]
        occ   = pi.copy()           # use pi as point size proxy within arm
        M_i   = r["M_inner"]
        ax3   = fig3.add_subplot(n_rows, n_cols, idx, projection="3d")
        ax2   = axes2.flat[idx - 1]
        if phi.shape[1] < 3:
            ax3.set_title(f"{sp}\n< 3 dims"); ax2.set_title(f"{sp}"); continue
        _scatter_arm(ax3, ax2, phi, chi, pi, asgn, occ,
                     sub_arm_colors[:M_i], M_i)
        for d, (ax, lim) in enumerate(zip([ax3, ax2], lims[:2])):
            pass
        if len(lims) >= 3:
            ax3.set_xlim(*lims[0]); ax3.set_ylim(*lims[1]); ax3.set_zlim(*lims[2])
        ax2.set_xlim(*lims[0]); ax2.set_ylim(*lims[1])
        ax3.set_xlabel("ψ₂"); ax3.set_ylabel("ψ₃"); ax3.set_zlabel("ψ₄")
        ax2.set_xlabel("ψ₂"); ax2.set_ylabel("ψ₃")
        ax3.set_title(f"{sp.replace('_', ' ')}\nM={M_i}  crisp={r['crispness']:.3f}",
                      fontsize=8)
        ax2.set_title(f"{sp.replace('_', ' ')}", fontsize=8)
        if idx == 1:
            ax2.legend(fontsize=7, frameon=False)
    for ax in list(axes2.flat)[len(valid):]:
        ax.set_visible(False)
    title3 = f"{arm_label} — native within-basin eigenspace (3-D)"
    title2 = f"{arm_label} — native within-basin eigenspace (2-D)"
    fig3.suptitle(title3)
    fig2.suptitle(title2)
    _show_or_save(fig3, title3, args, folder)
    _show_or_save(fig2, title2, args, folder)


def plot_shared_eigenspace(valid, arm_label, sub_arm_colors, species_names,
                           species_arrays, args, folder):
    """3-D and 2-D scatter projected back into the PARENT shared eigenspace."""
    n_sp   = len(valid)
    n_cols = min(4, n_sp)
    n_rows = max(1, int(np.ceil(n_sp / n_cols)))

    # Collect shared coords for the in-basin clusters
    phi_list = []
    for sp, r in valid.items():
        phi_s = species_arrays[sp].get("phi_macro_shared")
        if phi_s is not None and phi_s.shape[1] >= 3:
            phi_list.append(phi_s[r["mask"], :3])
    lims = _common_limits(phi_list)

    fig3 = plt.figure(figsize=(5 * n_cols, 4.5 * n_rows), constrained_layout=True)
    fig2, axes2 = plt.subplots(n_rows, n_cols,
                               figsize=(4.2 * n_cols, 3.5 * n_rows),
                               constrained_layout=True, squeeze=False)

    for idx, (sp, r) in enumerate(valid.items(), 1):
        phi_full = species_arrays[sp].get("phi_macro_shared")
        ax3 = fig3.add_subplot(n_rows, n_cols, idx, projection="3d")
        ax2 = axes2.flat[idx - 1]
        if phi_full is None or phi_full.shape[1] < 3:
            ax3.set_title(f"{sp}\nno shared coords"); ax2.set_title(f"{sp}"); continue
        phi   = phi_full[r["mask"]]
        chi   = r["chi_inner"]
        pi    = r["pi_sub"]
        asgn  = r["asgn_inner"]
        occ   = pi.copy()
        M_i   = r["M_inner"]
        _scatter_arm(ax3, ax2, phi, chi, pi, asgn, occ,
                     sub_arm_colors[:M_i], M_i)
        if len(lims) >= 3:
            ax3.set_xlim(*lims[0]); ax3.set_ylim(*lims[1]); ax3.set_zlim(*lims[2])
        ax2.set_xlim(*lims[0]); ax2.set_ylim(*lims[1])
        ax3.set_xlabel("shared φ₂"); ax3.set_ylabel("shared φ₃"); ax3.set_zlabel("shared φ₄")
        ax2.set_xlabel("shared φ₂"); ax2.set_ylabel("shared φ₃")
        ax3.set_title(f"{sp.replace('_', ' ')}\nM={M_i}", fontsize=8)
        ax2.set_title(f"{sp.replace('_', ' ')}", fontsize=8)
        if idx == 1:
            ax2.legend(fontsize=7, frameon=False)
    for ax in list(axes2.flat)[len(valid):]:
        ax.set_visible(False)
    title3 = f"{arm_label} — parent shared eigenspace (3-D)"
    title2 = f"{arm_label} — parent shared eigenspace (2-D)"
    fig3.suptitle(title3)
    fig2.suptitle(title2)
    _show_or_save(fig3, title3, args, folder)
    _show_or_save(fig2, title2, args, folder)


def plot_sub_arm_clouds(valid, arm_label, species_names, sp_colors,
                        species_arrays, args, folder):
    """One 3-D + 2-D panel per sub-arm, coloured by species (shared eigenspace)."""

    # Collect all shared-space points for global axis limits
    all_pts = []
    for sp, r in valid.items():
        phi_s = species_arrays[sp].get("phi_macro_shared")
        if phi_s is not None:
            all_pts.append(phi_s[r["mask"], :3])
    lims = _common_limits(all_pts)

    # Find the max M_inner
    max_M = max(r["M_inner"] for r in valid.values())

    for sub_arm in range(max_M):
        fig3 = plt.figure(figsize=(8, 6.5), constrained_layout=True)
        ax3  = fig3.add_subplot(111, projection="3d")
        fig2, ax2 = plt.subplots(figsize=(7, 5.8), constrained_layout=True)

        any_plotted = False
        for sp, r in valid.items():
            if sub_arm >= r["M_inner"]:
                continue
            phi_full = species_arrays[sp].get("phi_macro_shared")
            if phi_full is None or phi_full.shape[1] < 3:
                continue
            phi   = phi_full[r["mask"]]
            asgn  = r["asgn_inner"]
            pi    = r["pi_sub"]
            mask  = (asgn == sub_arm) & np.isfinite(phi[:, :3]).all(1) & (pi > 0)
            if not mask.any():
                continue
            sizes = 6 + 44 * np.sqrt(pi[mask] / max(pi.max(), 1e-12))
            col   = sp_colors[sp]
            ctr   = np.average(phi[mask, :3], axis=0, weights=pi[mask])
            ax3.scatter(*phi[mask, :3].T, s=sizes, color=col, alpha=0.6,
                        linewidths=0, label=sp.replace("_", " "))
            ax3.scatter(*ctr, s=95, marker="X", color=col, edgecolors="k", linewidths=0.6)
            ax2.scatter(phi[mask, 0], phi[mask, 1], s=sizes, color=col, alpha=0.65,
                        linewidths=0, label=sp.replace("_", " "))
            ax2.scatter(ctr[0], ctr[1], s=95, marker="X", color=col, edgecolors="k", linewidths=0.6)
            any_plotted = True

        if not any_plotted:
            plt.close(fig3); plt.close(fig2); continue

        if len(lims) >= 3:
            ax3.set_xlim(*lims[0]); ax3.set_ylim(*lims[1]); ax3.set_zlim(*lims[2])
        ax2.set_xlim(*lims[0]); ax2.set_ylim(*lims[1])
        ax3.set_xlabel("shared φ₂"); ax3.set_ylabel("shared φ₃"); ax3.set_zlabel("shared φ₄")
        ax2.set_xlabel("shared φ₂"); ax2.set_ylabel("shared φ₃")
        title3 = f"{arm_label} — sub-arm {sub_arm+1} species cloud (3-D)"
        title2 = f"{arm_label} — sub-arm {sub_arm+1} species cloud (2-D)"
        ax3.set_title(title3, fontsize=9)
        ax3.legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
        ax2.set_title(title2, fontsize=9)
        ax2.legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
        _show_or_save(fig3, title3, args, folder)
        _show_or_save(fig2, title2, args, folder)


def plot_cosine_similarity(valid, arm_label, species_arrays, species_names, args, folder):
    """Cosine-similarity matrix of all (species × sub-arm) centroid vectors
    in the parent shared eigenspace."""
    rows, vecs = [], []
    for sp, r in valid.items():
        phi_full = species_arrays[sp].get("phi_macro_shared")
        if phi_full is None or phi_full.shape[1] < 3:
            continue
        phi = phi_full[r["mask"]]
        ctrs = arm_centroid_in_space(phi[:, :3], r["chi_inner"], r["pi_sub"])
        for arm, ctr in enumerate(ctrs):
            rows.append({"label": f"{sp.replace('_', ' ')} sub-{arm+1}", "arm": arm})
            vecs.append(ctr)

    if len(vecs) < 2:
        print(f"  Not enough sub-arm vectors for cosine-similarity plot")
        return

    vecs  = np.asarray(vecs, dtype=float)
    unit  = normalise_rows(vecs)
    cosim = unit @ unit.T
    dist  = np.clip(1 - cosim, 0, 2)
    link  = linkage(squareform(dist, checks=False), method="average")
    order = leaves_list(link)
    ordered_sim    = cosim[np.ix_(order, order)]
    ordered_labels = [rows[i]["label"] for i in order]

    fig, axes = plt.subplots(1, 2,
                             figsize=(14, max(7, 0.28 * len(ordered_labels))),
                             constrained_layout=True,
                             gridspec_kw={"width_ratios": [0.9, 1.6]})
    dendrogram(link, labels=[r["label"] for r in rows],
               orientation="left", ax=axes[0], leaf_font_size=7, color_threshold=0)
    im = axes[1].imshow(ordered_sim, cmap="coolwarm", vmin=-1, vmax=1,
                        aspect="auto", interpolation="nearest")
    axes[1].set_xticks(range(len(ordered_labels)))
    axes[1].set_xticklabels(ordered_labels, rotation=90, fontsize=6.5)
    axes[1].set_yticks(range(len(ordered_labels)))
    axes[1].set_yticklabels(ordered_labels, fontsize=6.5)
    fig.colorbar(im, ax=axes[1], label="cosine similarity", fraction=0.025, pad=0.02)
    title = f"{arm_label} — sub-arm cosine similarity"
    fig.suptitle(title)
    _show_or_save(fig, title, args, folder)


def plot_umap_panels(valid, arm_label, species_names, sp_colors,
                     species_arrays, cluster_layout_2d, args, folder):
    """UMAP density and log₂ enrichment — occupancy weighted by within-arm chi."""
    x = cluster_layout_2d[:, 0].astype(float)
    y = cluster_layout_2d[:, 1].astype(float)
    x_rng = x.max() - x.min(); y_rng = y.max() - y.min()
    x_edges = np.linspace(x.min() - 0.04*x_rng, x.max() + 0.04*x_rng, DENSITY_BINS + 1)
    y_edges = np.linspace(y.min() - 0.04*y_rng, y.max() + 0.04*y_rng, DENSITY_BINS + 1)
    extent  = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

    N_global = len(x)

    def density_grid(weights, normalise=False):
        w = np.asarray(weights, dtype=float)
        H, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=w)
        H = gaussian_filter(H.T, sigma=DENSITY_SIGMA, mode="constant")
        if normalise:
            H /= max(H.sum(), EPS)
        return H

    def log2_enrich(obs_w, ref_w):
        obs = density_grid(obs_w, True)
        ref = density_grid(ref_w, True)
        return np.log2((obs + EPS) / (ref + EPS))

    def umap_figure(panel_grids, title, cmap, cb_label, vmin=None, vmax=None):
        sps_with_data = [sp for sp in species_names if sp in panel_grids]
        n_sp   = len(sps_with_data)
        n_cols = min(4, n_sp)
        n_rows = max(1, int(np.ceil(n_sp / n_cols)))
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(4.0 * n_cols, 3.4 * n_rows),
                                 sharex=True, sharey=True,
                                 constrained_layout=True, squeeze=False)
        im = None
        for ax, sp in zip(axes.flat, sps_with_data):
            g = panel_grids.get(sp)
            if g is None:
                ax.set_visible(False); continue
            im = ax.imshow(g, origin="lower", extent=extent, cmap=cmap,
                           vmin=vmin, vmax=vmax, interpolation="bilinear", aspect="equal")
            ax.set_title(sp.replace("_", " "), fontsize=8)
            ax.tick_params(labelsize=7, length=2)
        for ax in axes.flat[n_sp:]:
            ax.remove()
        if im is not None:
            fig.colorbar(im, ax=axes.flat[:n_sp], location="right",
                         fraction=0.025, pad=0.02, label=cb_label)
        fig.supxlabel("UMAP 1"); fig.supylabel("UMAP 2")
        fig.suptitle(title)
        _show_or_save(fig, title, args, folder)

    # ── build occupancy grids for each species, restricted to parent arm ──────
    occ_grids = {}
    pooled_w  = np.zeros(N_global)

    for sp, r in valid.items():
        occ_global = species_arrays[sp].get("occupancy_global")
        chi_global = species_arrays[sp].get("chi_global_aligned")
        if occ_global is None or len(occ_global) != N_global:
            continue
        # Weight global occupancy by chi of the parent arm
        if chi_global is not None and chi_global.shape[1] > r["arm_idx"]:
            w = chi_global[:, r["arm_idx"]] * occ_global
        else:
            # Fall-back: use hard-assigned macro clusters projected to global
            mask_macro = r["mask"]          # (N_macro,) bool
            # map macro mask to global using chi_global_aligned (N_global, N_basins)
            w = occ_global * 0.0
            # simple approximation: zero out clusters not in this arm
            # (we don't have a direct macro→global mapping here; use chi_global)
            w = (chi_global[:, r["arm_idx"]] if chi_global is not None
                 else np.ones(N_global)) * occ_global
        w = np.asarray(w, dtype=float)
        w /= max(w.sum(), EPS)
        occ_grids[sp] = density_grid(w)
        pooled_w += w

    if not occ_grids:
        print("  Skipping UMAP panels — no global occupancy data available.")
        return

    pooled_w /= max(pooled_w.sum(), EPS)
    flat = np.concatenate([g.ravel() for g in occ_grids.values()])
    vmax_occ = np.percentile(flat[flat > 0], DENSITY_VMAX_P) if flat.any() else 1.0

    umap_figure(occ_grids,
                f"{arm_label} — species occupancy density (UMAP)",
                "magma", "density", vmin=0, vmax=vmax_occ)

    # ── log₂ species enrichment vs pooled ────────────────────────────────────
    enrich_grids = {}
    for sp, r in valid.items():
        occ_global = species_arrays[sp].get("occupancy_global")
        chi_global = species_arrays[sp].get("chi_global_aligned")
        if occ_global is None or len(occ_global) != N_global:
            continue
        if chi_global is not None and chi_global.shape[1] > r["arm_idx"]:
            w = chi_global[:, r["arm_idx"]] * occ_global
        else:
            w = occ_global
        enrich_grids[sp] = log2_enrich(w, pooled_w)

    umap_figure(enrich_grids,
                f"{arm_label} — species enrichment log₂(species/pooled) (UMAP)",
                "bwr", "log₂ enrichment", vmin=-ENRICH_VMAX, vmax=ENRICH_VMAX)

    # ── per-sub-arm enrichment ────────────────────────────────────────────────
    max_M = max(r["M_inner"] for r in valid.values())
    for sub_arm in range(max_M):
        sub_enrich = {}
        for sp, r in valid.items():
            if sub_arm >= r["M_inner"]:
                continue
            occ_global = species_arrays[sp].get("occupancy_global")
            chi_global = species_arrays[sp].get("chi_global_aligned")
            if occ_global is None or len(occ_global) != N_global:
                continue

            # Project within-arm sub-arm chi back to global clusters:
            # global chi for parent arm × within-arm sub-arm chi (approximated
            # by mapping macro → global via parent chi_global_aligned)
            parent_chi_g = (chi_global[:, r["arm_idx"]]
                            if chi_global is not None and chi_global.shape[1] > r["arm_idx"]
                            else np.ones(N_global))

            # Within-arm chi for this sub-arm lives at macro resolution;
            # map macro clusters back to global via chi_macro_aligned / chi_global_aligned
            # We approximate: global sub-arm weight = parent_chi_global * (sub-arm chi
            # interpolated from macro-level assignments)
            # Use a simple per-macro assignment: broadcast chi_inner back via chi_global_aligned
            chi_inner_col = r["chi_inner"][:, sub_arm]  # (n_in_macro,)
            # Build a length-N_macro vector of within-arm chi for this sub-arm
            n_macro = species_arrays[sp]["chi_macro_aligned"].shape[0]
            chi_inner_full = np.zeros(n_macro)
            chi_inner_full[r["mask"]] = chi_inner_col

            # Lift to global: use chi_global_aligned to project macro chi to global
            # chi_global_aligned shape: (N_global, N_parent_basins) — not macro×sub
            # Best approximation: use chi_macro_aligned as proxy
            chi_mac = species_arrays[sp].get("chi_macro_aligned")
            if chi_mac is not None:
                # chi_global_aligned gives global→parent arm; we need global→macro
                # Use occupancy-weighted projection: not available directly.
                # Fallback: weight global clusters by parent arm chi × uniform sub-arm
                w_sub = parent_chi_g * occ_global
            else:
                w_sub = occ_global

            w_basin = chi_global[:, r["arm_idx"]] * occ_global if chi_global is not None and chi_global.shape[1] > r["arm_idx"] else occ_global
            sub_enrich[sp] = log2_enrich(w_sub, w_basin + EPS)

        if sub_enrich:
            umap_figure(sub_enrich,
                        f"{arm_label} — sub-arm {sub_arm+1} enrichment log₂(sub-arm/parent) (UMAP)",
                        "bwr", "log₂ enrichment", vmin=-ENRICH_VMAX, vmax=ENRICH_VMAX)


def plot_moseq_heatmaps(valid, parent_arm_idx: int, arm_label: str,
                         species_arrays, moseq_probs: pd.DataFrame, args, folder):
    """MoSeq syllable heatmaps for the parent arm, with sub-arm context overlay.

    Because within-basin sub-mode syllable probabilities require raw frame data
    that is not saved in the slow-mode arrays, this panel shows the parent-arm
    syllable landscape as context.  A separate sub-arm-weighted enrichment panel
    is produced using the macro-level chi_inner to soft-weight the global syllable
    probabilities.
    """
    try:
        import seaborn as sns
    except ImportError:
        print("  seaborn not installed — skipping syllable heatmaps")
        return

    parent_arm_1indexed = parent_arm_idx + 1   # CSV uses 1-indexed arms
    df = moseq_probs[moseq_probs["aligned_arm"] == parent_arm_1indexed].copy()
    if df.empty:
        print(f"  No MoSeq data for parent arm {parent_arm_1indexed} — skipping")
        return

    required = {"species", "moseq_cluster", "p_syllable_given_species_arm"}
    if not required.issubset(df.columns):
        print(f"  Unexpected columns: {list(df.columns)} — skipping syllable heatmaps")
        return

    has_labels = "moseq_label" in df.columns
    if has_labels:
        label_map = (df.dropna(subset=["moseq_label"])
                       .drop_duplicates("moseq_cluster")
                       .set_index("moseq_cluster")["moseq_label"]
                       .to_dict())
    else:
        label_map = {}

    def syl_label(cid):
        return label_map.get(cid, str(cid))

    # ── A: Parent-arm syllable probability heatmap ────────────────────────────
    top_syl = (df.groupby("moseq_cluster")["p_syllable_given_species_arm"]
                 .mean().nlargest(MOSEQ_TOP_N).index.tolist())
    sub_p = df[df["moseq_cluster"].isin(top_syl)]
    pivot_p = sub_p.pivot_table(index="moseq_cluster", columns="species",
                                values="p_syllable_given_species_arm", aggfunc="mean"
                               ).reindex(top_syl)
    pivot_p.index   = [syl_label(c) for c in pivot_p.index]
    pivot_p.columns = [c.replace("_", " ") for c in pivot_p.columns]

    n_syl = len(pivot_p.index); n_sp = len(pivot_p.columns)
    fig, ax = plt.subplots(figsize=(max(10, n_syl * 0.55), max(4, n_sp * 0.7)),
                           constrained_layout=True)
    sns.heatmap(pivot_p.T, ax=ax, cmap="magma", linewidths=0.3,
                linecolor="white", xticklabels=False,
                cbar_kws={"label": "p(syllable|species,arm)"})
    ax.set_xticks(np.arange(n_syl) + 0.5)
    ax.set_xticklabels(list(pivot_p.index), rotation=45, ha="right", fontsize=7)
    title = f"{arm_label} — syllable probabilities (top {MOSEQ_TOP_N})"
    ax.set_title(title); ax.set_xlabel("MoSeq syllable"); ax.set_ylabel("Species")
    ax.tick_params(axis="y", labelsize=8)
    _show_or_save(fig, title, args, folder)

    # ── B: Parent-arm syllable enrichment vs. repertoire ────────────────────
    if "log2_enrichment_vs_repertoire" in df.columns:
        top_enr = (df.groupby("moseq_cluster")["log2_enrichment_vs_repertoire"]
                     .mean().nlargest(MOSEQ_TOP_N).index.tolist())
        sub_e = df[df["moseq_cluster"].isin(top_enr)]
        pivot_e = sub_e.pivot_table(index="moseq_cluster", columns="species",
                                    values="log2_enrichment_vs_repertoire", aggfunc="mean"
                                   ).reindex(top_enr)
        pivot_e.index   = [syl_label(c) for c in pivot_e.index]
        pivot_e.columns = [c.replace("_", " ") for c in pivot_e.columns]

        n_syl_e = len(pivot_e.index); n_sp_e = len(pivot_e.columns)
        fig, ax = plt.subplots(figsize=(max(10, n_syl_e * 0.55), max(4, n_sp_e * 0.7)),
                               constrained_layout=True)
        finite_vals = pivot_e.values[np.isfinite(pivot_e.values)]
        vmax = np.nanpercentile(np.abs(finite_vals), 95) if len(finite_vals) else 2.0
        sns.heatmap(pivot_e.T, ax=ax, cmap="bwr", center=0,
                    vmin=-vmax, vmax=vmax, linewidths=0.3, linecolor="white",
                    xticklabels=False,
                    cbar_kws={"label": "log₂ enrichment vs. repertoire"})
        ax.set_xticks(np.arange(n_syl_e) + 0.5)
        ax.set_xticklabels(list(pivot_e.index), rotation=45, ha="right", fontsize=7)
        title = f"{arm_label} — syllable enrichment vs. repertoire (top {MOSEQ_TOP_N})"
        ax.set_title(title); ax.set_xlabel("MoSeq syllable"); ax.set_ylabel("Species")
        ax.tick_params(axis="y", labelsize=8)
        _show_or_save(fig, title, args, folder)

    # ── C: Sub-arm enrichment note ────────────────────────────────────────────
    print(f"\n  NOTE: Within-arm sub-mode syllable enrichment requires raw frame-level "
          f"syllable assignments (not saved in the slow-mode arrays).  The heatmaps above "
          f"show parent-arm-level syllable content as context.  To get per-sub-arm "
          f"enrichment, pass the frame-by-frame cluster sequences and MoSeq syllable "
          f"labels through _conditional_gpcca_for_clusters() from pooled_user_pipeline.py, "
          f"using chi_inner to define sub-mode membership.")


# ──────────────────────────────────────────────────────────────────────────────
# 7b.  SUB-BASIN SYLLABLE BIAS  (notebook 01 Cell 20 style)
# ──────────────────────────────────────────────────────────────────────────────

def plot_subbasin_syllable_bias(
    valid: dict,
    parent_arm_idx: int,
    arm_label: str,
    moseq_probs: pd.DataFrame,
    args,
    folder: str,
):
    """Notebook-01 Cell-20 style syllable bias between the two sub-arms within a
    parent basin.

    Strategy
    --------
    For each species the within-basin GPCCA produced ``chi_inner`` (shape
    ``n_in × M_inner``), which gives soft sub-arm membership of every in-basin
    macro cluster.  We compute a pi-weighted sub-arm syllable probability:

        p(syllable k | sub-arm j) ∝ Σ_i  chi_inner[i, j] · π_sub[i]  ·  δ(k, cluster_i)

    where δ(k, cluster_i) = 1 when macro cluster index i corresponds to MoSeq
    cluster k in the CSV (tested as both 0-indexed and 1-indexed).

    The result is a horizontal diverging bar chart of
    ``log₂(p_sub1 / p_sub0)`` for the top-N syllables by |bias|, one panel per
    species — matching the notebook 01 style exactly.
    """
    if moseq_probs is None:
        print("  No MoSeq data — skipping sub-basin syllable bias plot.")
        return

    parent_arm_1indexed = parent_arm_idx + 1
    df_parent = moseq_probs[moseq_probs["aligned_arm"] == parent_arm_1indexed].copy()
    if df_parent.empty:
        print(f"  No MoSeq data for parent arm {parent_arm_1indexed} — skipping sub-basin bias.")
        return

    M_inner = next(iter(valid.values()))["M_inner"]
    if M_inner < 2:
        print("  Only 1 sub-arm — no bias plot possible.")
        return

    TOP_N       = 20
    eps         = 1e-9
    COLOR_SUB0  = "#2471a3"   # blue  → sub-arm 1
    COLOR_SUB1  = "#c0392b"   # red   → sub-arm 2

    # Build label map from any species (shared syllable vocabulary)
    label_map: dict = {}
    if "moseq_label" in df_parent.columns:
        label_map = (
            df_parent.dropna(subset=["moseq_label"])
            .drop_duplicates("moseq_cluster")
            .set_index("moseq_cluster")["moseq_label"]
            .to_dict()
        )

    species_list = list(valid.keys())
    n_sp   = len(species_list)
    n_cols = min(2, n_sp)
    n_rows = max(1, int(np.ceil(n_sp / n_cols)))
    h_per  = max(5.5, TOP_N * 0.40)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(7.5 * n_cols, h_per * n_rows),
        constrained_layout=True,
        squeeze=False,
    )

    any_plotted = False

    for idx, sp in enumerate(species_list):
        ax = axes.flat[idx]
        r  = valid[sp]

        df_sp = df_parent[df_parent["species"] == sp]
        if df_sp.empty:
            ax.text(0.5, 0.5, f"{sp.replace('_', ' ')}\nNo MoSeq data in CSV",
                    ha="center", va="center", transform=ax.transAxes, fontsize=8)
            ax.set_title(sp.replace("_", " "), fontsize=9)
            continue

        mask         = r["mask"]                     # (N_macro,) bool
        in_basin_idx = np.where(mask)[0]             # positions in macro space
        chi_inner    = r["chi_inner"]                # (n_in, M_inner)
        pi_sub       = r["pi_sub"]                   # (n_in,) stationary within arm

        # ── Build p(syllable k | sub-arm j) using chi_inner · pi_sub ──────────
        # We try to match the CSV's moseq_cluster column to macro cluster indices.
        # Try 0-indexed first, then 1-indexed.
        def _build_sub_weights(index_offset: int) -> dict | None:
            """Returns {sub_arm: pd.Series(moseq_cluster → probability)} or None."""
            sub_w: dict[int, dict] = {j: {} for j in range(min(2, M_inner))}
            matched = 0
            for i_local, macro_idx in enumerate(in_basin_idx):
                k = int(macro_idx) + index_offset    # candidate moseq_cluster key
                if k not in df_sp["moseq_cluster"].values:
                    continue
                matched += 1
                for j in range(min(2, M_inner)):
                    w = float(chi_inner[i_local, j]) * float(pi_sub[i_local])
                    sub_w[j][k] = sub_w[j].get(k, 0.0) + w
            if matched == 0:
                return None
            out = {}
            for j, wd in sub_w.items():
                s   = pd.Series(wd)
                tot = s.sum()
                out[j] = s / max(tot, eps)
            return out

        p_subarm = _build_sub_weights(0)             # try 0-indexed
        if p_subarm is None:
            p_subarm = _build_sub_weights(1)         # try 1-indexed

        if p_subarm is None or len(p_subarm) < 2:
            ax.text(
                0.5, 0.5,
                f"{sp.replace('_', ' ')}\n"
                "moseq_cluster values don't match\n"
                "macro cluster indices.\n"
                "Per-sub-arm bias needs raw frames.",
                ha="center", va="center", transform=ax.transAxes, fontsize=8,
            )
            ax.set_title(sp.replace("_", " "), fontsize=9)
            continue

        p0 = p_subarm[0]
        p1 = p_subarm[1]
        common = p0.index.intersection(p1.index)
        if len(common) == 0:
            ax.text(0.5, 0.5, f"{sp.replace('_', ' ')}\nNo common syllables",
                    ha="center", va="center", transform=ax.transAxes, fontsize=8)
            ax.set_title(sp.replace("_", " "), fontsize=9)
            continue

        delta   = np.log2((p1[common] + eps) / (p0[common] + eps))
        top_idx = delta.abs().nlargest(TOP_N).index
        delta   = delta[top_idx].sort_values(ascending=True)

        ytick_labels = [label_map.get(k, str(k)) for k in delta.index]
        colors       = [COLOR_SUB1 if d > 0 else COLOR_SUB0 for d in delta.values]

        ax.barh(range(len(delta)), delta.values, color=colors, height=0.75,
                edgecolor="none")

        xlim_max = max(float(delta.abs().max()) * 1.25, 0.5)
        ax.axvspan(-xlim_max, 0, color=COLOR_SUB0, alpha=0.07)
        ax.axvspan(0, xlim_max, color=COLOR_SUB1, alpha=0.07)
        ax.axvline(0, color="k", lw=0.9, zorder=5)
        ax.set_xlim(-xlim_max, xlim_max)
        ax.set_yticks(range(len(delta)))
        ax.set_yticklabels(ytick_labels, fontsize=7)
        ax.set_xlabel("log₂(sub-arm 2 / sub-arm 1)", fontsize=8)
        ax.set_title(sp.replace("_", " "), fontsize=9)
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", length=0)

        from matplotlib.patches import Patch
        leg_handles = [
            Patch(facecolor=COLOR_SUB0, label="Sub-arm 1"),
            Patch(facecolor=COLOR_SUB1, label="Sub-arm 2"),
        ]
        ax.legend(handles=leg_handles, fontsize=7, frameon=False,
                  loc="lower right")
        any_plotted = True

    for ax in axes.flat[n_sp:]:
        ax.set_visible(False)

    title = f"{arm_label} — sub-arm syllable bias  log₂(sub-arm 2 / sub-arm 1)"
    fig.suptitle(title, fontsize=10)

    if any_plotted:
        _show_or_save(fig, title, args, folder)
    else:
        plt.close(fig)
        print("  Sub-basin syllable bias: no per-sub-arm data could be built from the CSV. "
              "Provide raw frame-level syllable sequences to compute true sub-arm probs.")


# ──────────────────────────────────────────────────────────────────────────────
# 8.  SUMMARY TABLE
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(all_results: dict):
    """Print a summary table of within-basin GPCCA results."""
    rows = []
    for (view, parent_arm, sp), r in all_results.items():
        if r is None:
            rows.append({
                "view": view, "parent_arm": parent_arm+1, "species": sp,
                "n_clusters": "—", "M_inner": "—", "crispness": "—", "note": "skipped"
            })
        else:
            rows.append({
                "view": view, "parent_arm": parent_arm+1, "species": sp,
                "n_clusters": r["n_clusters"], "M_inner": r["M_inner"],
                "crispness": f"{r['crispness']:.3f}", "note": "ok"
            })
    if rows:
        df = pd.DataFrame(rows)
        print("\n" + "="*70)
        print("WITHIN-BASIN GPCCA SUMMARY")
        print("="*70)
        print(df.to_string(index=False))


# ──────────────────────────────────────────────────────────────────────────────
# 9.  MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.save_figs:
        args.fig_dir.mkdir(parents=True, exist_ok=True)
        print(f"Figures will be saved to: {args.fig_dir}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\nLoading arrays — view: {args.view}")
    species_arrays = load_species_arrays(args.view)
    species_names  = sorted(species_arrays.keys())
    print(f"Species ({len(species_names)}): {species_names}")

    # Detect N parent basins from saved chi
    n_basins = int(
        next(iter(species_arrays.values()))["chi_macro_aligned"].shape[1]
    )
    print(f"Parent N_BASINS = {n_basins}")

    # Cluster embedding for UMAP panels
    embedding_csv = RUN_ROOT / "global_cluster_embedding.csv"
    cluster_layout_2d = None
    if embedding_csv.exists():
        emb = pd.read_csv(embedding_csv)
        xy = [c for c in emb.columns
              if c.lower() in ("x", "y", "umap1", "umap2", "umap_1", "umap_2",
                               "dim1", "dim2", "0", "1")]
        if len(xy) >= 2:
            cluster_layout_2d = emb[xy[:2]].values
        elif len(emb.columns) >= 2:
            cluster_layout_2d = emb.iloc[:, :2].values
        print(f"Cluster layout loaded: {cluster_layout_2d.shape}")
    else:
        print("global_cluster_embedding.csv not found — UMAP panels skipped.")

    # MoSeq syllable data
    moseq_csv = RUN_ROOT / "matched_arm_syllable_probs.csv"
    moseq_probs = pd.read_csv(moseq_csv) if moseq_csv.exists() else None
    print(f"MoSeq syllable probs: {'loaded' if moseq_probs is not None else 'not found'}")

    # ── Per-basin analysis ────────────────────────────────────────────────────
    all_results = {}
    for parent_arm_idx in range(n_basins):
        analyse_basin(
            parent_arm_idx=parent_arm_idx,
            species_arrays=species_arrays,
            species_names=species_names,
            cluster_layout_2d=cluster_layout_2d,
            moseq_probs=moseq_probs,
            args=args,
        )
        # Collect summary info
        for sp in species_names:
            d = species_arrays[sp]
            T    = d.get("transfer_operator_macro")
            chi  = d.get("chi_macro_aligned")
            pi   = d.get("pi_macro")
            if T is not None and chi is not None and pi is not None:
                if parent_arm_idx < chi.shape[1]:
                    r = run_gpcca_on_basin(T, pi, chi, parent_arm_idx,
                                           args.chi_threshold, args.m_min, args.m_max)
                    all_results[(args.view, parent_arm_idx, sp)] = r

    print_summary(all_results)
    print("\nDone.")


if __name__ == "__main__":
    main()
