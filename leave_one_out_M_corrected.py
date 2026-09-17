"""
Leave-one-out cross-validation for the number of metastable basins M.
CORRECTED implementation (supersedes leave_one_out_M.py).

Replicates Fig. S7H (flies) / Fig. S3E (worms) of
  Kaur, Jain & Berman (2025), "Using timescale as a state coordinate reveals
  the metastable geometry of behavior", arXiv:2605.24135


WHAT WAS WRONG WITH leave_one_out_M.py
======================================

(1) THE PI FORMULA WAS MISSING ITS NULL TERM  -->  this is why PI was negative.

    The old `held_out_log_likelihood` returned

        (1/N) sum_t log2 T_M[b_t, b_{t+1}]

    which is *minus the cross-entropy* of the held-out sequence under the
    training model. Since every T_M entry is a probability <= 1, every log2 is
    <= 0, so this quantity is <= 0 BY CONSTRUCTION. It is a log-likelihood, not
    an information.

    Predictive information is the log-likelihood *ratio* against the memoryless
    (marginal) model:

        PI = (1/N) sum_t [ log2 T_M[b_t, b_{t+tau}]  -  log2 p_next[b_{t+tau}] ]

    which estimates I(b_t ; b_{t+tau}), is >= 0 whenever the Markov model beats
    the basin marginals, and is bounded above by log2(M).

    The repo already had this right in diagnostics.py::crossval_vs_markov_null;
    leave_one_out_M.py simply dropped the second term.

(2) EVERY M WAS SILENTLY COLLAPSED TO M = 2.

    `load_global_chi(run_root, species, M)` ignored its `M` argument and always
    returned `chi_global_aligned.npy`, which is (1000, 2) for all 8 species.
    `hungarian_align` then built a rectangular (2, M) cost matrix, so
    `linear_sum_assignment` returned only 2 column indices and
    `chi_train[:, col_ind]` came back with 2 columns. Every fold at every M was
    therefore scored as a 2-basin model, with rows 2..M-1 of T_M containing
    nothing but pseudocount. That is why the old curves were flat and had no
    elbow.

    Fix: drop the alignment step entirely. Hungarian alignment only fixes basin
    *identity* (which arm is called 0 vs 1) for interpretability. PI is
    invariant under relabelling as long as the training model and the held-out
    sequence use the same chi, which they do. No reference chi is needed.

(3) THE LUMPED MODEL WAS ESTIMATED AND SCORED AT lag = 1 FRAME.

    Basins are defined by tau = 120-frame (1 s at 120 Hz) dynamics, but T_M and
    the held-out likelihood used lag = 1 frame (~8 ms). At that lag essentially
    every transition is a self-transition, so PI is pinned near zero regardless
    of M. Fixed: T_M and the held-out score both use lag = tau.

(4) THE NULL USED THE WRONG PARTITION AND THE WRONG MODEL.

    `random_coloring_null` drew its basin sizes from the (2-column) reference
    chi, but scored the shuffled sequence against a T_M estimated from the
    *real* partition. A matched random-coloring control has to be pushed through
    the whole pipeline: recolour the clusters, re-estimate T_M and p_next from
    the training folds under that recolouring, and only then score the held-out
    individual. Fixed here.


IMPLEMENTATION NOTE
===================
Counts are accumulated once per individual as a 1000x1000 cluster-level matrix.
A training fold is then C_total - C_i (no re-scan of the sequences), and both
the data and the null lumpings are M x M reductions S^T C S with S the sparse
cluster->basin indicator. This makes 200 null draws essentially free and lets
one Schur decomposition per fold be shared across all M.
"""

from __future__ import annotations

import os

# Keep BLAS single-threaded; we parallelise over folds instead. This MUST run
# before numpy/scipy are imported — setting it afterwards has no effect, and the
# resulting 4-workers x 4-BLAS-threads oversubscription costs ~25x in wall time.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import argparse  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from pygpcca import GPCCA  # noqa: E402


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

# Resolved relative to this file so the script works from any checkout location
# (override with the SLOWMODE_REPO environment variable if needed).
REPO_ROOT = Path(os.environ.get("SLOWMODE_REPO", Path(__file__).resolve().parent))
RUN_ROOT = (
    REPO_ROOT
    / "outputs/multispecies_slow_modes/global_clustering_modes"
    / "global_outputs/XY_EvenSampled_SlowModes"
)

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

N_CLUSTERS = 1000
TAU_FRAMES = 120          # 1 s at 120 Hz
M_RANGE = list(range(2, 9))
N_NULL = 200
SMOOTHING = 1.0           # Laplace smoothing at the BASIN level (M x M)
CLUSTER_PSEUDOCOUNT = 1e-6  # only to keep the 1000x1000 matrix irreducible


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

def load_individual_cluster_sequences(run_root: Path, species: str) -> list[np.ndarray]:
    states_dir = run_root / "states"
    files = sorted(states_dir.glob(f"{species}__subject__*_states.npy"))
    if not files:
        raise FileNotFoundError(
            f"No state files matching '{species}__subject__*_states.npy' in {states_dir}."
        )
    return [np.load(f).astype(np.int64) for f in files]


# ---------------------------------------------------------------------------
# counting / lumping
# ---------------------------------------------------------------------------

def cluster_count_matrix(seq: np.ndarray, n_states: int, lag: int) -> sp.csr_matrix:
    """Sparse N x N lag-tau transition COUNT matrix for one sequence."""
    if seq.size <= lag:
        return sp.csr_matrix((n_states, n_states), dtype=np.float64)
    src, dst = seq[:-lag], seq[lag:]
    data = np.ones(src.size, dtype=np.float64)
    return sp.coo_matrix(
        (data, (src, dst)), shape=(n_states, n_states)
    ).tocsr()


def indicator(coloring: np.ndarray, M: int) -> sp.csr_matrix:
    """Sparse N x M cluster -> basin indicator."""
    n = coloring.size
    return sp.csr_matrix(
        (np.ones(n), (np.arange(n), coloring)), shape=(n, M)
    )


def lump(C, S) -> np.ndarray:
    """S^T C S : reduce an N x N count matrix to M x M under a colouring."""
    return np.asarray((S.T @ C @ S).todense() if sp.issparse(C) else (S.T @ (C @ S)))


# ---------------------------------------------------------------------------
# predictive information
# ---------------------------------------------------------------------------

def predictive_information(counts_train_M: np.ndarray,
                           counts_held_M: np.ndarray,
                           smoothing: float = SMOOTHING) -> float:
    """
    Held-out predictive information in bits per transition.

        PI = sum_{a,b} q[a,b] * ( log2 T[a,b] - log2 p_next[b] )

    where q is the held-out individual's empirical joint over (b_t, b_{t+tau}),
    and T / p_next are the conditional and marginal estimated from the TRAINING
    folds only. This is the held-out estimate of I(b_t ; b_{t+tau}); it is 0 for
    a model that does no better than the basin marginals and is bounded above by
    log2(M).
    """
    total_held = counts_held_M.sum()
    if total_held <= 0:
        return np.nan

    c = counts_train_M + smoothing
    T = c / c.sum(axis=1, keepdims=True)
    p_next = c.sum(axis=0) / c.sum()

    q = counts_held_M / total_held
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.log2(T) - np.log2(p_next)[None, :]
    return float(np.sum(q * log_ratio))


def size_preserving_recolouring(coloring: np.ndarray, M: int,
                                rng: np.random.Generator) -> np.ndarray:
    """Random cluster -> basin map with exactly the same basin sizes."""
    return coloring[rng.permutation(coloring.size)]


# ---------------------------------------------------------------------------
# G-PCCA
# ---------------------------------------------------------------------------

def stationary_distribution(T: np.ndarray, tol: float = 1e-12,
                            max_iter: int = 100_000) -> np.ndarray:
    """Left Perron vector by power iteration (robust for dense row-stochastic T)."""
    n = T.shape[0]
    pi = np.full(n, 1.0 / n)
    Tt = T.T
    for _ in range(max_iter):
        new = Tt @ pi
        new /= new.sum()
        if np.abs(new - pi).sum() < tol:
            return new
        pi = new
    return pi


class FoldGPCCA:
    """One Schur decomposition per fold, reused across all M."""

    def __init__(self, T: np.ndarray):
        eta = stationary_distribution(T)
        eta = np.maximum(eta, 1e-12)
        eta /= eta.sum()
        self._T, self._eta = T, eta
        self._g = GPCCA(T, eta=eta, z="LM", method="brandts")

    def colouring(self, M: int) -> np.ndarray:
        """Hard cluster -> basin assignment at M basins."""
        try:
            self._g.optimize(M)
            chi = self._g.memberships
        except ValueError as exc:
            if "complex conjugate" not in str(exc).lower():
                raise
            # standard Reuter & Weber recipe: fit M+1, drop lowest-mass basin
            g2 = GPCCA(self._T, eta=self._eta, z="LM", method="brandts")
            g2.optimize(M + 1)
            chi = g2.memberships
            keep = np.argsort(-(chi.T @ self._eta))[:M]
            chi = chi[:, keep]
        return np.asarray(np.argmax(chi, axis=1), dtype=np.int64)


# ---------------------------------------------------------------------------
# one leave-one-out fold (all M)
# ---------------------------------------------------------------------------

# Shared read-only state for worker processes. Set in the parent before the Pool
# is forked, so workers inherit it copy-on-write instead of pickling ~GB of
# sparse count matrices into every job.
_SHARED: dict = {}


def run_fold(args):
    (held_idx, M_range, n_null, seed) = args
    per_ind_counts = _SHARED["per_ind_counts"]
    C_total = _SHARED["C_total"]
    rng = np.random.default_rng(seed)

    C_held = per_ind_counts[held_idx]
    C_train = C_total - C_held                       # sparse
    C_train_dense = np.asarray(C_train.todense())

    # row-stochastic training transfer matrix for G-PCCA
    counts = C_train_dense + CLUSTER_PSEUDOCOUNT
    T_train = counts / counts.sum(axis=1, keepdims=True)

    fold = FoldGPCCA(T_train)

    pi_out = np.full(len(M_range), np.nan)
    null_out = np.full(len(M_range), np.nan)

    # descending M so the largest Schur basis is computed first and reused
    for m_idx in sorted(range(len(M_range)), key=lambda k: -M_range[k]):
        M = M_range[m_idx]
        try:
            colouring = fold.colouring(M)
        except Exception as exc:                      # noqa: BLE001
            print(f"    [warn] fold {held_idx} M={M}: {type(exc).__name__}: {exc}")
            continue

        S = indicator(colouring, M)
        pi_out[m_idx] = predictive_information(
            lump(C_train, S), lump(C_held, S)
        )

        nulls = np.empty(n_null)
        for k in range(n_null):
            S_null = indicator(size_preserving_recolouring(colouring, M, rng), M)
            nulls[k] = predictive_information(
                lump(C_train, S_null), lump(C_held, S_null)
            )
        null_out[m_idx] = float(np.nanmean(nulls))

    return held_idx, pi_out, null_out


# ---------------------------------------------------------------------------
# species driver
# ---------------------------------------------------------------------------

def leave_one_out_M(sequences, n_clusters, tau, M_range, n_null=N_NULL,
                    seed=42, n_jobs=1, ckpt_path: Path | None = None,
                    time_budget: float | None = None):
    """
    Run all leave-one-individual-out folds.

    Checkpointing: completed folds are written to `ckpt_path` after each fold, so
    an interrupted run resumes where it left off. If `time_budget` seconds elapse
    the run stops cleanly and returns with complete=False; just invoke again.
    """
    n_ind = len(sequences)
    n_M = len(M_range)

    held_out_pi = np.full((n_M, n_ind), np.nan)
    null_pi = np.full((n_M, n_ind), np.nan)
    done = np.zeros(n_ind, dtype=bool)

    if ckpt_path is not None and ckpt_path.exists():
        ck = np.load(ckpt_path)
        if list(ck["M_range"]) == list(M_range) and ck["done"].size == n_ind:
            held_out_pi, null_pi, done = ck["held_out_pi"], ck["null_pi"], ck["done"]
            print(f"  resuming: {int(done.sum())}/{n_ind} folds already done",
                  flush=True)

    todo = [i for i in range(n_ind) if not done[i]]
    if not todo:
        return {"M_range": M_range, "held_out_pi": held_out_pi,
                "null_pi": null_pi, "complete": True}

    print("  Accumulating per-individual count matrices ...", flush=True)
    per_ind_counts = [cluster_count_matrix(s, n_clusters, tau) for s in sequences]
    C_total = per_ind_counts[0].copy()
    for C in per_ind_counts[1:]:
        C_total = C_total + C

    _SHARED["per_ind_counts"] = per_ind_counts
    _SHARED["C_total"] = C_total

    jobs = [(i, M_range, n_null, seed + 1000 * i) for i in todo]
    t0 = time.time()
    out_of_time = False

    def record(i, pi_col, null_col):
        held_out_pi[:, i] = pi_col
        null_pi[:, i] = null_col
        done[i] = True
        if ckpt_path is not None:
            tmp = ckpt_path.with_suffix(".tmp.npz")
            np.savez(tmp, M_range=np.array(M_range), held_out_pi=held_out_pi,
                     null_pi=null_pi, done=done)
            tmp.replace(ckpt_path)
        print(f"    fold {int(done.sum())}/{n_ind}  ({time.time()-t0:.0f}s)",
              flush=True)

    if n_jobs > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(n_jobs) as pool:
            it = pool.imap_unordered(run_fold, jobs)
            for i, pi_col, null_col in it:
                record(i, pi_col, null_col)
                if time_budget is not None and time.time() - t0 > time_budget:
                    out_of_time = True
                    pool.terminate()
                    break
    else:
        for job in jobs:
            record(*run_fold(job))
            if time_budget is not None and time.time() - t0 > time_budget:
                out_of_time = True
                break

    complete = bool(done.all())
    if out_of_time and not complete:
        print(f"  [time budget reached] {int(done.sum())}/{n_ind} folds done; "
              f"re-run to continue", flush=True)

    return {"M_range": M_range, "held_out_pi": held_out_pi,
            "null_pi": null_pi, "complete": complete}


# ---------------------------------------------------------------------------
# sanity checks
# ---------------------------------------------------------------------------

def sanity_check(results) -> list[str]:
    """
    Checks that must hold for a valid predictive-information estimate.

    On the log2(M) bound: the held-out score decomposes as

        PI = I_q(b_t; b_{t+tau})  +  KL(q_next || p_next)
                                  -  E_a KL( q(.|a) || T(.|a) )

    where q is the held-out individual's empirical joint and p_next/T come from
    the training folds. The first term is bounded by log2(M), but the second is
    not: an individual whose basin occupancy differs from the population mean
    can score above log2(M). That is a property of a held-out likelihood ratio,
    not an error, so it is reported as a note rather than a failure. A LARGE
    fraction of folds above the ceiling would instead indicate real trouble.
    """
    M_range = np.asarray(results["M_range"])
    pi = results["held_out_pi"]
    null = results["null_pi"]
    finite = np.isfinite(pi)
    problems = []

    over = pi - np.log2(M_range)[:, None]
    n_over = int(np.sum(over[finite] > 1e-9))
    if n_over:
        frac = n_over / max(int(finite.sum()), 1)
        msg = (f"note: {n_over} fold-M cells ({100*frac:.1f}%) above log2(M) "
               f"(max excess {np.nanmax(over):.3f} bits) — occupancy shift")
        problems.append("SEVERE " + msg if frac > 0.10 else msg)

    frac_neg = float(np.mean(pi[finite] < -1e-9))
    if frac_neg > 0.05:
        problems.append(f"{100*frac_neg:.1f}% of folds have PI < 0")
    if np.nanmean(np.abs(null)) > 0.05:
        problems.append("random-colouring null is not close to 0")
    if np.nanmean(pi) <= np.nanmean(null):
        problems.append("data PI does not exceed the null")
    if np.isnan(pi).any():
        problems.append(f"{int(np.isnan(pi).sum())} fold-M cells failed (NaN)")
    return problems


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--species", nargs="+", default=ALL_SPECIES)
    p.add_argument("--m-range", nargs="+", type=int, default=M_RANGE)
    p.add_argument("--n-null", type=int, default=N_NULL)
    p.add_argument("--tau", type=int, default=TAU_FRAMES)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--time-budget", type=float, default=None,
                   help="Stop cleanly after this many seconds; re-run to resume.")
    p.add_argument("--ckpt-dir", default=None,
                   help="Where to keep resumable per-fold checkpoints "
                        "(default: alongside the results).")
    args = p.parse_args()

    started = time.time()
    all_complete = True

    out_dir = Path(args.out_dir) if args.out_dir else RUN_ROOT / "loo_M_results_corrected"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else out_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")
    print(f"tau = {args.tau} frames | M = {args.m_range} | n_null = {args.n_null}\n")

    for species in args.species:
        print(f"\n{'='*64}\n  {species}\n{'='*64}", flush=True)
        out_npz = out_dir / f"loo_M_results_{species}.npz"
        if out_npz.exists() and not args.overwrite:
            print(f"  [skip] {out_npz.name} already exists")
            continue

        try:
            sequences = load_individual_cluster_sequences(RUN_ROOT, species)
        except FileNotFoundError as e:
            print(f"  [skip] {e}")
            continue
        print(f"  {len(sequences)} individuals "
              f"(lengths {min(map(len, sequences))}-{max(map(len, sequences))})",
              flush=True)

        remaining = None
        if args.time_budget is not None:
            remaining = args.time_budget - (time.time() - started)
            if remaining <= 10:
                print("  [time budget reached] stopping before this species")
                all_complete = False
                break

        results = leave_one_out_M(
            sequences, N_CLUSTERS, args.tau, args.m_range,
            n_null=args.n_null, seed=42, n_jobs=args.n_jobs,
            ckpt_path=ckpt_dir / f"_ckpt_{species}.npz",
            time_budget=remaining,
        )

        if not results["complete"]:
            all_complete = False
            break

        np.savez(
            out_npz,
            M_range=np.array(results["M_range"]),
            held_out_pi=results["held_out_pi"],
            null_pi=results["null_pi"],
            tau_frames=args.tau,
            n_null=args.n_null,
        )

        mean_pi = np.nanmean(results["held_out_pi"], axis=1)
        mean_null = np.nanmean(results["null_pi"], axis=1)
        print(f"\n  M        : {np.array(results['M_range'])}")
        print(f"  PI (bits): {mean_pi.round(4)}")
        print(f"  null     : {mean_null.round(4)}")
        problems = sanity_check(results)
        print("  sanity   : " + ("OK" if not problems else "; ".join(problems)))
        print(f"  saved -> {out_npz}", flush=True)
        try:
            (ckpt_dir / f"_ckpt_{species}.npz").unlink(missing_ok=True)
        except OSError:
            pass   # read-only / no-delete filesystem; harmless to leave behind

    if all_complete:
        print("\nALL_COMPLETE")
    else:
        print("\nINCOMPLETE — re-run to resume")
        raise SystemExit(2)
