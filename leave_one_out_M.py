"""
Leave-one-out cross-validation for the number of metastable basins M.

Replicates the analysis in Fig. S7H (flies) / Fig. S3E (worms) of:
  "Using timescale as a state coordinate reveals the metastable geometry of behavior"
  Kaur, Jain, Berman (2025), arXiv:2605.24135

WHAT THIS DOES
--------------
G-PCCA is fit to *all* individuals pooled to select M (the number of metastable
basins). That selection is validated here by a leave-one-individual-out
cross-validation of the *held-out predictive information* (PI):

For each candidate M and each individual i:
  1. Estimate the N×N cluster-level transfer matrix T(τ) from the training set
     (all individuals except i).
  2. Refit G-PCCA at basin count M on that training T(τ) to obtain M Schur vectors
     and soft membership matrix χ_train ∈ [0,1]^{N×M}.
  3. Align the M recovered basin directions to the full-data (pooled) reference arms
     via cosine-similarity + Hungarian matching.
  4. Project the held-out individual's cluster sequence through χ_train to obtain
     a soft membership trace; take argmax to get a hard basin label sequence.
  5. Estimate the M×M lumped transition matrix T_M from the *training* basin
     label sequences (not the held-out individual's).
  6. Evaluate held-out PI per transition:
       PI_i(M) = (1 / N_trans) Σ_t  log2[ T_M[ arm_t, arm_{t+1} ] ]
     (log-likelihood of consecutive basin transitions under the training model,
     evaluated on the held-out individual's basin sequence).
  7. Compute a random-coloring null: randomly permute the cluster→basin assignment
     (preserving basin sizes), repeat step 4–6, and average over n_null draws.

OUTPUT
------
A plot of mean ± SEM of PI_i(M) vs M, overlaid with the null, showing a sharp
elbow at the true M (as in Fig. S7H of the paper).

ADAPTING TO YOUR DATA
---------------------
The script assumes you have, for each individual, an array of *cluster indices*
at each time frame — the N-state discrete cluster sequence produced by k-means.
It also needs the pooled full-data G-PCCA result (chi_global) for alignment.

Edit the DATA LOADING section below to match your file paths.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import permutations
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — no window, saves to disk only
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── optional: G-PCCA from pyGPCCA ──────────────────────────────────────────
try:
    from pygpcca import GPCCA
    HAS_GPCCA = True
except ImportError:
    HAS_GPCCA = False
    print("pyGPCCA not found. Install with:  pip install pygpcca")

# ── optional: progress bar ──────────────────────────────────────────────────
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kw: x  # no-op fallback


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  DATA LOADING — project-specific paths                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

RUN_ROOT = Path(
    "/Users/meganbishop/slowmodeevo/outputs/multispecies_slow_modes/"
    "global_clustering_modes/global_outputs/XY_EvenSampled_SlowModes"
)

SAVE_ROOT = RUN_ROOT / "saved_slow_mode_arrays"

# All 8 species in the dataset
ALL_SPECIES = [
    "Mus_caroli",
    "Mus_musculus",
    "Mus_spretus",
    "Peromyscus_californicus",
    "Peromyscus_gossypinus",
    "Peromyscus_leucopus",
    "Peromyscus_maniculatus",
    "Peromyscus_polionotus",
]

# Number of discrete clusters (N_states from k-means)
N_CLUSTERS = 1000

# Working lag τ (in frames): TAU_SECONDS=1.0, FS_HZ=120.0  →  τ = 120 frames
TAU_FRAMES = 120

# Range of M values to test
M_RANGE = list(range(2, 9))

# Number of random-coloring null draws per (M, individual)
N_NULL = 200

# ── Load individual cluster sequences ───────────────────────────────────────
# States are stored in a FLAT directory:
#   {RUN_ROOT}/states/{Species}__subject__{ID}_states.npy
# All species share the same flat folder; we glob by species prefix.

def load_individual_cluster_sequences(run_root: Path, species: str) -> list[np.ndarray]:
    """
    Load per-individual cluster-label sequences for one species.

    Files live in a single flat directory:
        {run_root}/states/{species}__subject__*_states.npy
    """
    states_dir = run_root / "states"
    pattern = f"{species}__subject__*_states.npy"
    files = sorted(states_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No state files matching '{pattern}' in {states_dir}."
        )
    seqs = []
    for f in files:
        seq = np.load(f)
        seqs.append(seq.astype(int))
    return seqs


# ── Load full-data (pooled) G-PCCA membership matrix ────────────────────────
# Use the saved chi_macro_aligned from the with_low_occupancy_cutoff view.
# Shape: (N_retained, M_saved) — used only for arm-direction alignment;
# the LOO refits chi from scratch for each training fold.

def load_global_chi(run_root: Path, species: str, M: int) -> np.ndarray:
    """
    Return the saved chi_global_aligned array (1000 × N_basins) as the
    reference for Hungarian arm-direction alignment.

    chi_global_aligned is the membership matrix lifted back to all 1000
    global clusters (via absorption weights), so its row count always matches
    the chi_train produced by the LOO G-PCCA refit on the full 1000-cluster
    transition matrix.

    (chi_macro_aligned has only N_retained rows — the active clusters after
    the occupancy cutoff — and would cause a shape mismatch with chi_train.)
    """
    chi_path = (
        SAVE_ROOT / "with_low_occupancy_cutoff" / species / "chi_global_aligned.npy"
    )
    if not chi_path.exists():
        raise FileNotFoundError(
            f"Could not find saved chi_global_aligned at {chi_path}."
        )
    return np.load(chi_path)   # (1000, M_saved)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CORE UTILITIES                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def build_transition_matrix(sequences: list[np.ndarray],
                             n_states: int,
                             lag: int,
                             pseudocount: float = 1e-6) -> np.ndarray:
    """
    Estimate an N×N row-stochastic transition matrix at the given lag
    from a list of discrete-state sequences.

    T[i, j] = P(state_{t+lag} = j | state_t = i)

    A small pseudocount (Laplace smoothing) is added to every entry so that:
      - all rows sum to 1 (pyGPCCA requirement), and
      - the matrix is irreducible (no structural zeros), which G-PCCA needs
        for a well-defined Perron eigenvalue.
    The pseudocount is tiny relative to typical counts (~1e5 frames/individual)
    and does not meaningfully affect the estimated dynamics.
    """
    counts = np.zeros((n_states, n_states), dtype=np.float64)
    for seq in sequences:
        src = seq[:-lag]
        dst = seq[lag:]
        np.add.at(counts, (src, dst), 1)
    # Add small pseudocount to every entry to guarantee row-stochasticity
    # and irreducibility even for clusters absent from the training fold.
    counts += pseudocount
    row_sums = counts.sum(axis=1, keepdims=True)
    return counts / row_sums


def stationary_distribution(T: np.ndarray) -> np.ndarray:
    """
    Compute the stationary distribution of a row-stochastic matrix T via
    the left eigenvector at eigenvalue 1 (power iteration fallback).
    """
    from scipy.sparse.linalg import eigs
    import scipy.sparse as sp
    n = T.shape[0]
    vals, vecs = eigs(T.T, k=1, which='LM')
    pi = vecs[:, 0].real
    pi = np.abs(pi)
    pi /= pi.sum()
    return pi


def run_gpcca(T: np.ndarray, M: int) -> np.ndarray:
    """
    Run G-PCCA on the N×N transition matrix T and return the chi matrix
    (shape N × M, rows sum to 1).

    Mirrors the gpcca_utils.run_gpcca() used in notebook 00:
      - computes stationary distribution eta from T
      - floors eta above 0 (pyGPCCA rejects zero entries)
      - calls g.optimize(M) to fit at exactly M basins
      - retries at M+1 and drops smallest-mass basin if a complex-conjugate
        eigenvalue pair is split (standard recipe from Reuter & Weber 2018)
    """
    if not HAS_GPCCA:
        raise RuntimeError("pyGPCCA is required. Install with: pip install pygpcca")

    eta = stationary_distribution(T)
    eta = np.maximum(eta, 1e-12)
    eta /= eta.sum()

    g = GPCCA(T, eta=eta, z='LM', method='brandts')
    try:
        g.optimize(M)
    except ValueError as exc:
        if 'complex conjugate' in str(exc).lower():
            g = GPCCA(T, eta=eta, z='LM', method='brandts')
            g.optimize(M + 1)
            chi = g.memberships
            mass = chi.T @ eta
            keep = np.argsort(-mass)[:M]
            chi = chi[:, keep]
            chi /= chi.sum(axis=1, keepdims=True)
            return chi
        raise

    return g.memberships   # shape (N, M), rows sum to 1


def hungarian_align(chi_train: np.ndarray, chi_ref: np.ndarray) -> np.ndarray:
    """
    Align the M columns of chi_train to the M columns of chi_ref via
    Hungarian matching on cosine similarity.

    Returns chi_train reordered so that column k corresponds to the k-th
    reference arm.
    """
    # Normalise columns to unit vectors for cosine similarity
    def col_norm(X):
        norms = np.linalg.norm(X, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms

    C = col_norm(chi_ref).T @ col_norm(chi_train)  # (M, M) cosine similarities
    # Hungarian: maximise similarity → minimise negative similarity
    row_ind, col_ind = linear_sum_assignment(-C)
    return chi_train[:, col_ind]


def basin_sequence_from_chi(cluster_seq: np.ndarray,
                              chi: np.ndarray) -> np.ndarray:
    """
    Project a cluster index sequence through chi to obtain hard basin labels.

    cluster_seq : (T,) integer array of cluster indices
    chi         : (N_clusters, M) soft membership matrix
    Returns     : (T,) integer array of basin indices (0-based)
    """
    soft = chi[cluster_seq]          # (T, M)
    return np.argmax(soft, axis=1)   # (T,)


def lumped_transition_matrix(basin_seqs: list[np.ndarray],
                              M: int,
                              lag: int = 1) -> np.ndarray:
    """
    Estimate the M×M row-stochastic lumped transition matrix from a list of
    hard basin label sequences (each entry 0 … M-1).
    """
    return build_transition_matrix(basin_seqs, M, lag)


def held_out_log_likelihood(T_M: np.ndarray,
                             held_out_basin_seq: np.ndarray,
                             lag: int = 1,
                             eps: float = 1e-30) -> float:
    """
    Compute held-out predictive information per transition (bits):

        PI = (1/N_trans) Σ_t  log2[ T_M[ arm_t, arm_{t+lag} ] ]

    This is the cross-entropy of the held-out basin transitions under the
    lumped Markov model estimated from the training set.
    """
    src = held_out_basin_seq[:-lag]
    dst = held_out_basin_seq[lag:]
    log_probs = np.log2(T_M[src, dst] + eps)
    return log_probs.mean()


def random_coloring_null(cluster_seq: np.ndarray,
                          chi_ref: np.ndarray,
                          T_M: np.ndarray,
                          n_null: int,
                          lag: int = 1,
                          rng: np.random.Generator | None = None) -> float:
    """
    Compute the mean held-out PI under random-coloring of the cluster→basin map.

    Procedure (matches the paper's null):
      - Compute the basin sizes (number of clusters assigned to each arm) from chi_ref
      - Randomly permute the cluster → basin assignment while preserving those sizes
      - Recompute the held-out basin sequence and its log-likelihood under T_M
    """
    if rng is None:
        rng = np.random.default_rng()

    M = chi_ref.shape[1]
    N = chi_ref.shape[0]
    # Hard basin assignment for each cluster (from reference chi)
    ref_arm = np.argmax(chi_ref, axis=1)       # (N,)
    basin_sizes = np.bincount(ref_arm, minlength=M)

    pi_nulls = []
    for _ in range(n_null):
        # Build a random cluster → basin map preserving basin sizes
        perm_arm = np.empty(N, dtype=int)
        clusters_shuffled = rng.permutation(N)
        start = 0
        for arm_id, size in enumerate(basin_sizes):
            perm_arm[clusters_shuffled[start:start + size]] = arm_id
            start += size
        held_basin = perm_arm[cluster_seq]
        pi_nulls.append(held_out_log_likelihood(T_M, held_basin, lag=lag))

    return float(np.mean(pi_nulls))


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN LEAVE-ONE-OUT LOOP                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def leave_one_out_M(sequences: list[np.ndarray],
                    n_clusters: int,
                    tau: int,
                    M_range: list[int],
                    n_null: int = 200,
                    global_chi_fn=None,
                    seed: int = 0) -> dict:
    """
    Run leave-one-individual-out cross-validation across M values.

    Parameters
    ----------
    sequences   : list of (T_i,) integer arrays — per-individual cluster sequences
    n_clusters  : total number of discrete clusters N
    tau         : lag in frames for T(τ)
    M_range     : list of candidate basin counts to evaluate
    n_null      : number of random-coloring null draws per (M, individual)
    global_chi_fn : callable(M) → chi_global (N × M); if None, full-data chi is
                    re-estimated from all sequences (slower but self-contained)
    seed        : RNG seed for reproducibility

    Returns
    -------
    dict with keys:
      'M_range'     : list of M values
      'held_out_pi' : (len(M_range), n_individuals) array of held-out PIs [bits]
      'null_pi'     : (len(M_range), n_individuals) array of null PIs [bits]
    """
    rng = np.random.default_rng(seed)
    n_ind = len(sequences)
    held_out_pi = np.full((len(M_range), n_ind), np.nan)
    null_pi     = np.full((len(M_range), n_ind), np.nan)

    # ── Pooled full-data T for reference G-PCCA ──────────────────────────
    print("Building pooled (full-data) transition matrix …")
    T_full = build_transition_matrix(sequences, n_clusters, tau)

    for m_idx, M in enumerate(M_range):
        print(f"\n── M = {M} ──────────────────────────────────")

        # Reference chi from full data (for alignment)
        if global_chi_fn is not None:
            chi_global = global_chi_fn(M)
        else:
            print(f"  Fitting full-data G-PCCA at M={M} (for alignment reference) …")
            chi_global = run_gpcca(T_full, M)

        for i in tqdm(range(n_ind), desc=f"  LOO M={M}", leave=False):
            # ── Training set: all individuals except i ────────────────────
            train_seqs = [sequences[j] for j in range(n_ind) if j != i]

            # Estimate training transfer matrix
            T_train = build_transition_matrix(train_seqs, n_clusters, tau)

            # Refit G-PCCA on training T
            chi_train = run_gpcca(T_train, M)

            # Align training chi columns to global reference via Hungarian
            chi_aligned = hungarian_align(chi_train, chi_global)

            # Basin sequences for training individuals (under training chi)
            train_basin_seqs = [
                basin_sequence_from_chi(sequences[j], chi_aligned)
                for j in range(n_ind) if j != i
            ]

            # Lumped M×M transition matrix from training basins
            T_M = lumped_transition_matrix(train_basin_seqs, M, lag=1)

            # Held-out basin sequence
            held_basin_seq = basin_sequence_from_chi(sequences[i], chi_aligned)

            # Held-out predictive information
            held_out_pi[m_idx, i] = held_out_log_likelihood(
                T_M, held_basin_seq, lag=1
            )

            # Random-coloring null
            null_pi[m_idx, i] = random_coloring_null(
                sequences[i], chi_global, T_M, n_null, lag=1, rng=rng
            )

    return {
        "M_range":     M_range,
        "held_out_pi": held_out_pi,
        "null_pi":     null_pi,
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PLOTTING                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def plot_loo_results(results: dict,
                     species: str = "",
                     optimal_M: int | None = None,
                     save_path: Path | None = None):
    """
    Replicate Fig. S7H: held-out PI ± SEM vs M, with random-coloring null.
    """
    M_range     = np.array(results["M_range"])
    held_out_pi = results["held_out_pi"]   # (n_M, n_ind)
    null_pi     = results["null_pi"]        # (n_M, n_ind)

    n_ind = held_out_pi.shape[1]

    mean_pi   = np.nanmean(held_out_pi, axis=1)
    sem_pi    = np.nanstd(held_out_pi, axis=1, ddof=1) / np.sqrt(n_ind)
    mean_null = np.nanmean(null_pi, axis=1)
    sem_null  = np.nanstd(null_pi, axis=1, ddof=1) / np.sqrt(n_ind)

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)

    # Individual held-out traces (thin, semi-transparent)
    for i in range(n_ind):
        ax.plot(M_range, held_out_pi[:, i],
                color="#1f77b4", alpha=0.25, lw=0.9, zorder=1)

    # Mean ± SEM held-out
    ax.plot(M_range, mean_pi, color="#1f77b4", lw=2.0, zorder=3,
            label="Held-out (data)")
    ax.fill_between(M_range, mean_pi - sem_pi, mean_pi + sem_pi,
                    color="#1f77b4", alpha=0.25, zorder=2)

    # Null
    ax.plot(M_range, mean_null, color="#7f7f7f", lw=1.5, ls="--", zorder=3,
            label="Random coloring (null)")
    ax.fill_between(M_range, mean_null - sem_null, mean_null + sem_null,
                    color="#7f7f7f", alpha=0.2, zorder=2)

    # Elbow marker
    if optimal_M is not None:
        ax.axvline(optimal_M, color="goldenrod", lw=1.5, ls="--", zorder=4,
                   label=f"M = {optimal_M} (elbow)")

    ax.set_xlabel("Number of basins M", fontsize=12)
    ax.set_ylabel("Held-out PI per transition (bits)", fontsize=12)
    title = "Leave-one-out cross-validation for M"
    if species:
        title += f"\n{species.replace('_', ' ')}"
    ax.set_title(title, fontsize=12)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.legend(fontsize=10)
    ax.grid(axis="y", lw=0.4, alpha=0.5)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    # plt.show()  # disabled for non-interactive runs; plots saved to disk
    return fig, ax


def detect_elbow(results: dict, min_gain: float = 0.05) -> int:
    """
    Find the M at which the held-out PI curve elbows — the first M beyond which
    each additional basin adds < min_gain × (total gain from M_min to M_max).

    Returns the detected optimal M.
    """
    M_range     = np.array(results["M_range"])
    mean_pi     = np.nanmean(results["held_out_pi"], axis=1)
    gains       = np.diff(mean_pi)                    # PI gain per +1 basin
    total_gain  = mean_pi[-1] - mean_pi[0]
    threshold   = min_gain * total_gain
    # First M where *subsequent* gain falls below threshold
    for k, g in enumerate(gains):
        if g < threshold:
            return int(M_range[k])
    return int(M_range[np.argmax(gains)])            # fallback: largest gain


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ENTRY POINT                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Leave-one-out M cross-validation for all species.")
    parser.add_argument(
        "--species", nargs="+", default=ALL_SPECIES,
        help="Species to run (default: all 8)."
    )
    parser.add_argument(
        "--m-range", nargs="+", type=int, default=M_RANGE,
        help="M values to test (default: 2 3 4 5 6 7 8)."
    )
    parser.add_argument(
        "--n-null", type=int, default=N_NULL,
        help="Number of random-coloring null draws (default: 200)."
    )
    parser.add_argument(
        "--tau", type=int, default=TAU_FRAMES,
        help="Lag in frames (default: 120 = 1 s at 120 Hz)."
    )
    args = parser.parse_args()

    out_dir = RUN_ROOT / "loo_M_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}\n")

    summary = {}

    for species in args.species:
        print(f"\n{'='*60}")
        print(f"  Species: {species}")
        print(f"{'='*60}")

        # Skip if results already saved
        out_npz = out_dir / f"loo_M_results_{species}.npz"
        if out_npz.exists():
            print(f"  [SKIP] results already exist: {out_npz.name}")
            data = np.load(out_npz)
            results_cached = {"M_range": list(data["M_range"]),
                              "held_out_pi": data["held_out_pi"],
                              "null_pi": data["null_pi"]}
            optimal_M = detect_elbow(results_cached)
            summary[species] = optimal_M
            print(f"  Detected optimal M = {optimal_M}  (from cached results)")
            continue

        print("  Loading individual cluster sequences …")
        try:
            sequences = load_individual_cluster_sequences(RUN_ROOT, species)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue
        print(f"  {len(sequences)} individuals, "
              f"lengths: min={min(len(s) for s in sequences)}, "
              f"max={max(len(s) for s in sequences)}")

        def global_chi_fn(M, _sp=species):
            return load_global_chi(RUN_ROOT, _sp, M)

        results = leave_one_out_M(
            sequences     = sequences,
            n_clusters    = N_CLUSTERS,
            tau           = args.tau,
            M_range       = args.m_range,
            n_null        = args.n_null,
            global_chi_fn = global_chi_fn,
            seed          = 42,
        )

        # Save raw results
        np.savez(out_npz,
                 M_range     = np.array(results["M_range"]),
                 held_out_pi = results["held_out_pi"],
                 null_pi     = results["null_pi"])
        print(f"  Results saved → {out_npz}")

        # Detect optimal M
        optimal_M = detect_elbow(results)
        print(f"  Detected optimal M = {optimal_M}  (elbow criterion)")
        summary[species] = optimal_M

        # Plot
        plot_loo_results(
            results,
            species   = species,
            optimal_M = optimal_M,
            save_path = out_dir / f"loo_M_{species}.png",
        )
        plt.close("all")

    print(f"\n{'='*60}")
    print("  SUMMARY — Optimal M per species")
    print(f"{'='*60}")
    for sp, m in summary.items():
        print(f"  {sp:<40s}  M = {m}")
