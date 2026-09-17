"""
Replot leave-one-out M results from saved .npz files, without the elbow marker.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

LOO_DIR = Path(
    "/Users/meganbishop/slowmodeevo/outputs/multispecies_slow_modes/"
    "global_clustering_modes/global_outputs/XY_EvenSampled_SlowModes/loo_M_results"
)

for npz_path in sorted(LOO_DIR.glob("loo_M_results_*.npz")):
    species = npz_path.stem.replace("loo_M_results_", "")
    data = np.load(npz_path)
    M_range     = data["M_range"]
    held_out_pi = data["held_out_pi"]   # (n_M, n_ind)
    null_pi     = data["null_pi"]        # (n_M, n_ind)

    n_ind     = held_out_pi.shape[1]
    mean_pi   = np.nanmean(held_out_pi, axis=1)
    sem_pi    = np.nanstd(held_out_pi,  axis=1, ddof=1) / np.sqrt(n_ind)
    mean_null = np.nanmean(null_pi,     axis=1)
    sem_null  = np.nanstd(null_pi,      axis=1, ddof=1) / np.sqrt(n_ind)

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)

    # Individual held-out traces
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

    ax.set_xlabel("Number of basins M", fontsize=12)
    ax.set_ylabel("Held-out PI per transition (bits)", fontsize=12)
    ax.set_title(f"Leave-one-out cross-validation for M\n"
                 f"{species.replace('_', ' ')}", fontsize=12)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.legend(fontsize=10)
    ax.grid(axis="y", lw=0.4, alpha=0.5)

    out_path = LOO_DIR / f"loo_M_{species}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

print("\nDone.")
