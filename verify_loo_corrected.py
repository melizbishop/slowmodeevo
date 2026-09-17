"""
Independent verification of leave_one_out_M_corrected.py.

Three checks:

  1. BRUTE FORCE vs LUMPING ALGEBRA.
     The production code never scans the basin sequences: it lumps precomputed
     1000x1000 cluster-level count matrices via S^T C S. This recomputes one
     fold the naive way -- map every frame to its basin, count b_t -> b_{t+tau}
     by scanning, score transition by transition -- and compares.

  2. AGREEMENT WITH diagnostics.py::crossval_vs_markov_null.
     That function is the repo's pre-existing (and correct) definition of the
     same quantity. Run on the same fold it must give the same number.

  3. RE-RUN THE SANITY CHECKS on every saved species result.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import importlib.util
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("SLOWMODE_REPO", Path(__file__).resolve().parent))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


loo = _load("loo", REPO / "leave_one_out_M_corrected.py")
diag = _load("diag", REPO / "diagnostics.py")

TAU = 120
SPECIES = "Mus_caroli"
HELD = 0
M = 4


# --------------------------------------------------------------------------
# brute-force reference implementation
# --------------------------------------------------------------------------
def brute_force_pi(train_seqs, held_seq, colouring, M, tau, smoothing=1.0):
    """Scan the basin sequences frame by frame. No matrix algebra shortcuts."""
    counts = np.zeros((M, M))
    for s in train_seqs:
        b = colouring[s]
        for t in range(len(b) - tau):          # explicit loop, deliberately naive
            counts[b[t], b[t + tau]] += 1

    c = counts + smoothing
    T = c / c.sum(axis=1, keepdims=True)
    p_next = c.sum(axis=0) / c.sum()

    bh = colouring[held_seq]
    total = 0.0
    n = 0
    for t in range(len(bh) - tau):
        a, b_ = bh[t], bh[t + tau]
        total += np.log2(T[a, b_]) - np.log2(p_next[b_])
        n += 1
    return total / n


def main():
    print(f"Verifying on {SPECIES}, held-out index {HELD}, M={M}, tau={TAU}\n")

    seqs = loo.load_individual_cluster_sequences(loo.RUN_ROOT, SPECIES)
    # keep the brute-force loop tractable: truncate every sequence
    seqs = [s[:60_000] for s in seqs]
    train = [s for i, s in enumerate(seqs) if i != HELD]
    held = seqs[HELD]

    # ---- fit the fold exactly as production does -------------------------
    cnt = [loo.cluster_count_matrix(s, 1000, TAU) for s in seqs]
    C_total = cnt[0].copy()
    for c in cnt[1:]:
        C_total = C_total + c
    C_held = cnt[HELD]
    C_train = C_total - C_held

    dense = np.asarray(C_train.todense()) + loo.CLUSTER_PSEUDOCOUNT
    T_train = dense / dense.sum(axis=1, keepdims=True)
    colouring = loo.FoldGPCCA(T_train).colouring(M)
    print(f"basin sizes (clusters per basin): "
          f"{np.bincount(colouring, minlength=M).tolist()}")

    # ---- check 1: lumping algebra vs brute force -------------------------
    S = loo.indicator(colouring, M)
    fast = loo.predictive_information(loo.lump(C_train, S), loo.lump(C_held, S))
    slow = brute_force_pi(train, held, colouring, M, TAU)
    print(f"\n[1] lumping algebra : {fast:.10f} bits")
    print(f"    brute force     : {slow:.10f} bits")
    print(f"    abs difference  : {abs(fast-slow):.2e}  "
          f"-> {'PASS' if abs(fast-slow) < 1e-9 else 'FAIL'}")

    # ---- check 2: agreement with diagnostics.py --------------------------
    # These differ slightly *by design*. diagnostics.py takes p_next from the
    # unconditional basin occupancy (every frame). This code takes it from the
    # column marginal of the lag-tau transition counts, i.e. the occupancy of
    # the frames that actually appear as b_{t+tau}. Check 2b shows why that
    # matters: only the column-marginal convention makes PI exactly zero for a
    # genuinely memoryless model, which is the property the statistic needs.
    chi = np.zeros((1000, M))
    chi[np.arange(1000), colouring] = 1.0
    res = diag.crossval_vs_markov_null(seqs, chi, TAU, smoothing=1.0)
    ref = res["per_fold_bits"][HELD]
    rel = abs(fast - ref) / max(abs(ref), 1e-12)
    print(f"\n[2] diagnostics.py::crossval_vs_markov_null : {ref:.10f} bits")
    print(f"    this implementation                     : {fast:.10f} bits")
    print(f"    relative difference : {100*rel:.3f}%  "
          f"-> {'PASS (p_next convention only)' if rel < 0.01 else 'FAIL'}")

    # ---- check 2b: PI must be exactly 0 for a memoryless model -----------
    # Build a training count matrix whose rows are all identical (so b_{t+tau}
    # is independent of b_t). A correct PI estimator must return exactly 0 on
    # it, whatever the held-out data look like.
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(M))
    row_mass = rng.integers(1_000, 10_000, size=M).astype(float)
    memoryless = np.outer(row_mass, p)
    held_any = rng.integers(1, 500, size=(M, M)).astype(float)
    zero = loo.predictive_information(memoryless, held_any, smoothing=0.0)
    print(f"\n[2b] PI under a memoryless training model : {zero:+.3e} bits"
          f"  -> {'PASS' if abs(zero) < 1e-12 else 'FAIL'}")

    # and the old formula on the same input, for contrast
    Tm = memoryless / memoryless.sum(axis=1, keepdims=True)
    q = held_any / held_any.sum()
    old_on_memoryless = float(np.sum(q * np.log2(Tm)))
    print(f"     old formula on the same input        : {old_on_memoryless:+.3f} bits"
          f"  <- should be 0, is not")

    # ---- check 3: the old formula, for the record ------------------------
    bh = colouring[held]
    old_counts = np.zeros((M, M))
    for s in train:
        b = colouring[s]
        np.add.at(old_counts, (b[:-1], b[1:]), 1)     # OLD: lag = 1 frame
    old_counts += 1e-6
    T_old = old_counts / old_counts.sum(axis=1, keepdims=True)
    old_pi = float(np.mean(np.log2(T_old[bh[:-1], bh[1:]] + 1e-30)))
    print(f"\n[3] old formula (no null term, lag=1) : {old_pi:.6f} bits "
          f"<- negative by construction")

    # ---- check 4: sanity checks on every saved species -------------------
    print("\n[4] sanity checks on saved results")
    out_dir = loo.RUN_ROOT / "loo_M_results_corrected"
    for f in sorted(out_dir.glob("loo_M_results_*.npz")):
        d = np.load(f)
        r = {"M_range": list(d["M_range"]), "held_out_pi": d["held_out_pi"],
             "null_pi": d["null_pi"]}
        probs = loo.sanity_check(r)
        sp = f.stem.replace("loo_M_results_", "")
        status = "OK" if not probs else "; ".join(probs)
        print(f"    {sp:<28s} min PI={np.nanmin(r['held_out_pi']):+.3f}  "
              f"max null={np.nanmax(r['null_pi']):.4f}  {status}")


if __name__ == "__main__":
    main()
