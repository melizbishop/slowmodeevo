"""
Plot the CORRECTED leave-one-out M results (see leave_one_out_M_corrected.py).

Produces, in <RUN_ROOT>/loo_M_results_corrected/ :
  loo_M_<species>.png       one panel per species (matches the old figure style)
  loo_M_all_species.png     8-panel overview
  loo_M_summary.csv         mean +/- SEM per (species, M), plus the null
"""

import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

REPO_ROOT = Path(os.environ.get("SLOWMODE_REPO", Path(__file__).resolve().parent))
LOO_DIR = (
    REPO_ROOT
    / "outputs/multispecies_slow_modes/global_clustering_modes"
    / "global_outputs/XY_EvenSampled_SlowModes/loo_M_results_corrected"
)

DATA_C = "#1f77b4"
NULL_C = "#7f7f7f"


def load(npz_path):
    d = np.load(npz_path)
    M = np.asarray(d["M_range"])
    pi, null = d["held_out_pi"], d["null_pi"]
    n = pi.shape[1]
    return dict(
        M=M, pi=pi, null=null, n=n,
        mean=np.nanmean(pi, axis=1),
        sem=np.nanstd(pi, axis=1, ddof=1) / np.sqrt(n),
        mean_null=np.nanmean(null, axis=1),
        sem_null=np.nanstd(null, axis=1, ddof=1) / np.sqrt(n),
        tau=int(d["tau_frames"]) if "tau_frames" in d else None,
    )


def draw(ax, r, title, show_traces=True, legend=True):
    if show_traces:
        for i in range(r["n"]):
            ax.plot(r["M"], r["pi"][:, i], color=DATA_C, alpha=0.18, lw=0.8, zorder=1)

    ax.plot(r["M"], r["mean"], color=DATA_C, lw=2.0, zorder=4, label="Held-out (data)")
    ax.fill_between(r["M"], r["mean"] - r["sem"], r["mean"] + r["sem"],
                    color=DATA_C, alpha=0.25, zorder=3)

    ax.plot(r["M"], r["mean_null"], color=NULL_C, lw=1.5, ls="--", zorder=4,
            label="Random colouring (null)")
    ax.fill_between(r["M"], r["mean_null"] - r["sem_null"],
                    r["mean_null"] + r["sem_null"], color=NULL_C, alpha=0.2, zorder=3)

    ax.plot(r["M"], np.log2(r["M"]), color="#c0392b", lw=1.0, ls=":", zorder=2,
            label=r"$\log_2 M$ ceiling")

    ax.axhline(0.0, color="k", lw=0.6, alpha=0.5, zorder=2)
    ax.set_title(title, fontsize=11)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    if legend:
        ax.legend(fontsize=9, frameon=False)


def main():
    files = sorted(LOO_DIR.glob("loo_M_results_*.npz"))
    if not files:
        raise SystemExit(f"No corrected results found in {LOO_DIR}")

    rows = ["species,n_individuals,M,held_out_pi_mean_bits,held_out_pi_sem_bits,"
            "null_mean_bits,null_sem_bits"]
    loaded = []

    for f in files:
        species = f.stem.replace("loo_M_results_", "")
        r = load(f)
        loaded.append((species, r))

        fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        tau = f" (τ = {r['tau']} frames)" if r["tau"] else ""
        draw(ax, r, f"Leave-one-out cross-validation for M{tau}\n"
                   f"{species.replace('_', ' ')}  (n = {r['n']})")
        ax.set_xlabel("Number of basins M", fontsize=12)
        ax.set_ylabel("Held-out PI per transition (bits)", fontsize=12)
        out = LOO_DIR / f"loo_M_{species}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out.name}")

        for k, M in enumerate(r["M"]):
            rows.append(f"{species},{r['n']},{M},{r['mean'][k]:.6f},"
                        f"{r['sem'][k]:.6f},{r['mean_null'][k]:.6f},"
                        f"{r['sem_null'][k]:.6f}")

    ncol = 4
    nrow = int(np.ceil(len(loaded) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.2 * nrow),
                             constrained_layout=True, sharex=True)
    for ax, (species, r) in zip(axes.ravel(), loaded):
        draw(ax, r, f"{species.replace('_', ' ')} (n={r['n']})",
             show_traces=False, legend=False)
    for ax in axes.ravel()[len(loaded):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Number of basins M")
    for ax in axes[:, 0]:
        ax.set_ylabel("Held-out PI (bits)")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Leave-one-individual-out cross-validation for the basin count M "
                 "(corrected predictive information)", fontsize=13)
    out = LOO_DIR / "loo_M_all_species.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.name}")

    (LOO_DIR / "loo_M_summary.csv").write_text("\n".join(rows) + "\n")
    print(f"Saved: loo_M_summary.csv")


if __name__ == "__main__":
    main()
