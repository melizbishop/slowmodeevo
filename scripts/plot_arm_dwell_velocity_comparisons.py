from pathlib import Path
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d


REPO_ROOT = Path("/Users/meganbishop/slowmodeevo").resolve()
KMEANS_ROOT = (
    REPO_ROOT
    / "outputs/single_species_distance_comparison"
    / "pca_multispecies_evensamp"
    / "global_kmeans_N1000_balanced_species"
)
OUT_DIR = KMEANS_ROOT / "species_identity_by_cluster"
ARM_DIR = OUT_DIR / "slow_mode_arm_enrichment_occupancy_threshold"
SPECIES_RUN_DIR = ARM_DIR / "species_operators"
PLOT_DIR = OUT_DIR / "arm_dwell_velocity_comparisons_occupancy_threshold"
MANIFEST_CSV = KMEANS_ROOT / "global_kmeans_manifest.csv"

FS_HZ = 120.0
DWELL_DELTA_SECONDS = 2.0
DRIFT_MAX_ARROWS = 260
ARM_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2"]


def infer_lag_frames():
    summary_path = ARM_DIR / "slow_mode_run_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        if "lag_frames" in summary.columns and summary["lag_frames"].notna().any():
            return int(summary["lag_frames"].dropna().iloc[0])
    params_csv = KMEANS_ROOT.parent / "final_outputs" / "parameters.csv"
    if params_csv.exists():
        params = pd.read_csv(params_csv)
        vals = dict(zip(params["parameter"], params["value"]))
        if "lag_frames" in vals:
            return int(float(vals["lag_frames"]))
        if {"tau_seconds", "fs"}.issubset(vals):
            return int(float(vals["tau_seconds"]) * float(vals["fs"]))
    return 240


def load_species_run(species):
    species_dir = SPECIES_RUN_DIR / species
    run_dir = species_dir / "macro_operator"
    if not run_dir.exists():
        run_dir = species_dir
    run = {
        "species": species,
        "phi": np.load(run_dir / "phi.npy"),
        "chi": np.load(run_dir / "chi.npy"),
        "pi": np.load(run_dir / "stationary_distribution.npy"),
        "T": np.load(run_dir / "transfer_operator.npy"),
        "active_clusters": np.load(run_dir / "active_clusters.npy"),
        "assignments": np.load(run_dir / "arm_assignments.npy"),
    }
    expanded_chi_path = species_dir / "expanded_global_chi.npy"
    if expanded_chi_path.exists():
        run["global_chi"] = np.load(expanded_chi_path)
    else:
        run["global_chi"] = run["chi"]
    return run


def load_species_states(manifest, species):
    rows = manifest.loc[manifest["species"].astype(str) == species]
    arrays = []
    for path in rows["state_file"].astype(str):
        states = np.asarray(np.load(path, mmap_mode="r"), dtype=np.int64)
        arrays.append(states[states >= 0])
    return arrays


def residence_times_by_arm(states_list, chi, fs, delta_seconds):
    chi = np.asarray(chi, dtype=float)
    n_clusters, n_arms = chi.shape
    win = max(1, int(round(delta_seconds * fs)))
    pooled = [[] for _ in range(n_arms)]
    per_individual_rows = []
    for individual_index, states in enumerate(states_list):
        states = np.asarray(states, dtype=np.int64)
        states = states[(states >= 0) & (states < n_clusters)]
        if states.size == 0:
            continue
        memberships = chi[states]
        if delta_seconds > 0:
            memberships = uniform_filter1d(memberships, size=win, axis=0, mode="nearest")
        labels = np.argmax(memberships, axis=1).astype(int)
        change = np.r_[0, np.flatnonzero(np.diff(labels)) + 1, labels.size]
        lengths = np.diff(change) / fs
        run_labels = labels[change[:-1]]
        for dwell, arm_label in zip(lengths, run_labels):
            pooled[int(arm_label)].append(float(dwell))
        for arm in range(n_arms):
            vals = lengths[run_labels == arm]
            per_individual_rows.append(
                {
                    "individual_index": individual_index,
                    "arm": arm + 1,
                    "n_dwell_events": int(vals.size),
                    "median_dwell_seconds": float(np.median(vals)) if vals.size else np.nan,
                    "mean_dwell_seconds": float(np.mean(vals)) if vals.size else np.nan,
                }
            )
    return [np.asarray(x, dtype=float) for x in pooled], per_individual_rows


def build_dwell_tables(manifest, species_names):
    all_dwell = {}
    summary_rows = []
    individual_rows = []
    for species in species_names:
        run = load_species_run(species)
        states = load_species_states(manifest, species)
        dwell_by_arm, indiv_rows = residence_times_by_arm(
            states, run["global_chi"], FS_HZ, DWELL_DELTA_SECONDS
        )
        all_dwell[species] = dwell_by_arm
        for arm, vals in enumerate(dwell_by_arm, start=1):
            summary_rows.append(
                {
                    "species": species,
                    "arm": arm,
                    "n_dwell_events": int(vals.size),
                    "median_dwell_seconds": float(np.median(vals)) if vals.size else np.nan,
                    "mean_dwell_seconds": float(np.mean(vals)) if vals.size else np.nan,
                    "q25_dwell_seconds": float(np.quantile(vals, 0.25)) if vals.size else np.nan,
                    "q75_dwell_seconds": float(np.quantile(vals, 0.75)) if vals.size else np.nan,
                    "q90_dwell_seconds": float(np.quantile(vals, 0.90)) if vals.size else np.nan,
                    "q99_dwell_seconds": float(np.quantile(vals, 0.99)) if vals.size else np.nan,
                }
            )
        for row in indiv_rows:
            row["species"] = species
            individual_rows.append(row)
    return all_dwell, pd.DataFrame(summary_rows), pd.DataFrame(individual_rows)


def plot_dwell_violin(all_dwell, species_names):
    ncols = 4
    nrows = math.ceil(len(species_names) / ncols)
    all_positive = [
        vals for per_species in all_dwell.values() for vals in per_species
        if vals.size and np.any(vals > 0)
    ]
    ymax = max((np.quantile(vals[vals > 0], 0.995) for vals in all_positive), default=1.0)
    ymax = max(ymax, 1.0)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 3.3 * nrows),
        squeeze=False, constrained_layout=True,
    )
    for ax, species in zip(axes.ravel(), species_names):
        raw_data = [vals[vals > 0] for vals in all_dwell[species]]
        positions = np.arange(1, len(raw_data) + 1)
        nonempty = [(pos, vals) for pos, vals in zip(positions, raw_data) if vals.size]
        if nonempty:
            plot_positions = [x[0] for x in nonempty]
            data = [x[1] for x in nonempty]
            parts = ax.violinplot(
                data, positions=plot_positions, showmeans=False,
                showmedians=True, showextrema=False,
            )
            for pos, body in zip(plot_positions, parts["bodies"]):
                body.set_facecolor(ARM_COLORS[(pos - 1) % len(ARM_COLORS)])
                body.set_edgecolor("0.25")
                body.set_alpha(0.65)
            parts["cmedians"].set_color("black")
            parts["cmedians"].set_linewidth(1.2)
            medians = [np.median(vals) if vals.size else np.nan for vals in raw_data]
            ax.plot(positions, medians, color="black", marker="o", lw=1.0, ms=3)
        for pos, vals in zip(positions, raw_data):
            if vals.size == 0:
                ax.text(pos, 0.02, "no events", rotation=90, ha="center", va="bottom", fontsize=7)
        ax.set_title(species.replace("_", " "), fontsize=10)
        ax.set_xticks(positions)
        ax.set_xticklabels([f"arm {i}" for i in positions], rotation=35, ha="right")
        ax.set_yscale("log")
        ax.set_ylim(max(1 / FS_HZ, 1e-2), ymax * 1.2)
        ax.grid(True, axis="y", which="both", alpha=0.25)
    for ax in axes.ravel()[len(species_names):]:
        ax.axis("off")
    fig.suptitle(
        f"Dwell times by species-specific arm, all species (chi smoothing {DWELL_DELTA_SECONDS:g} s)",
        fontsize=14, fontweight="bold",
    )
    fig.supxlabel("species-specific arm")
    fig.supylabel("dwell time (s, log scale)")
    return fig


def plot_residence_ccdf(all_dwell, species_names):
    ncols = 4
    nrows = math.ceil(len(species_names) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 3.3 * nrows),
        squeeze=False, constrained_layout=True,
    )
    for ax, species in zip(axes.ravel(), species_names):
        for arm, vals in enumerate(all_dwell[species], start=1):
            vals = np.sort(vals[vals > 0])
            if vals.size == 0:
                continue
            ccdf = 1.0 - np.arange(vals.size) / vals.size
            ax.loglog(
                vals, ccdf, color=ARM_COLORS[(arm - 1) % len(ARM_COLORS)],
                lw=1.4, label=f"arm {arm}",
            )
        ax.set_title(species.replace("_", " "), fontsize=10)
        ax.set_xlabel("residence time (s)")
        ax.set_ylabel("CCDF")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7, frameon=False)
    for ax in axes.ravel()[len(species_names):]:
        ax.axis("off")
    fig.suptitle(
        f"Residence-time distributions per species (all arms; chi smoothing {DWELL_DELTA_SECONDS:g} s)",
        fontsize=14, fontweight="bold",
    )
    return fig


def plot_drift_velocity(species_names, lag_frames):
    tau_seconds = lag_frames / FS_HZ
    ncols = 4
    nrows = math.ceil(len(species_names) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 3.7 * nrows),
        squeeze=False, constrained_layout=True,
    )
    velocity_rows = []
    for ax, species in zip(axes.ravel(), species_names):
        run = load_species_run(species)
        phi = np.asarray(run["phi"], dtype=float)
        if phi.shape[1] < 2:
            phi = np.pad(phi, ((0, 0), (0, 2 - phi.shape[1])), constant_values=0.0)
        coords = phi[:, :2]
        T = np.asarray(run["T"], dtype=float)
        pi = np.asarray(run["pi"], dtype=float)
        active = np.asarray(run["active_clusters"], dtype=int)
        valid = active[(active >= 0) & (active < coords.shape[0])]
        expected_next = T @ coords
        velocity = (expected_next - coords) / max(tau_seconds, 1e-12)
        speed = np.linalg.norm(velocity, axis=1)
        if valid.size > DRIFT_MAX_ARROWS:
            order = valid[np.argsort(pi[valid])[::-1]]
            plot_clusters = np.sort(order[:DRIFT_MAX_ARROWS])
        else:
            plot_clusters = valid
        arms = np.asarray(run["assignments"], dtype=int)
        colors = [
            ARM_COLORS[int(arms[c]) % len(ARM_COLORS)] if 0 <= arms[c] else "0.7"
            for c in plot_clusters
        ]
        bg = valid
        bg_sizes = 6 + 55 * np.sqrt(pi[bg] / max(pi[bg].max(), 1e-12))
        ax.scatter(coords[bg, 0], coords[bg, 1], s=bg_sizes, c="0.88", alpha=0.65, linewidths=0)
        ax.quiver(
            coords[plot_clusters, 0], coords[plot_clusters, 1],
            velocity[plot_clusters, 0], velocity[plot_clusters, 1],
            color=colors, angles="xy", scale_units="xy", scale=None,
            width=0.0032, alpha=0.82,
        )
        ax.set_title(species.replace("_", " "), fontsize=10)
        ax.set_xlabel(r"$\phi_2$")
        ax.set_ylabel(r"$\phi_3$")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.18)
        for c in plot_clusters:
            velocity_rows.append(
                {
                    "species": species,
                    "cluster": int(c),
                    "phi_2": float(coords[c, 0]),
                    "phi_3": float(coords[c, 1]),
                    "drift_phi_2_per_second": float(velocity[c, 0]),
                    "drift_phi_3_per_second": float(velocity[c, 1]),
                    "drift_speed_per_second": float(speed[c]),
                    "stationary_probability": float(pi[c]),
                    "arm": int(arms[c] + 1) if arms[c] >= 0 else -1,
                }
            )
    for ax in axes.ravel()[len(species_names):]:
        ax.axis("off")
    fig.suptitle(
        f"Drift-velocity fields in species-specific slow-mode space (lag {lag_frames} frames, tau={tau_seconds:g} s)",
        fontsize=14, fontweight="bold",
    )
    return fig, pd.DataFrame(velocity_rows)


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST_CSV)
    species_names = sorted(manifest["species"].astype(str).unique())
    lag_frames = infer_lag_frames()

    all_dwell, dwell_summary, individual_summary = build_dwell_tables(manifest, species_names)
    dwell_summary.to_csv(PLOT_DIR / "species_arm_dwell_time_summary.csv", index=False)
    individual_summary.to_csv(PLOT_DIR / "species_arm_dwell_time_per_individual_summary.csv", index=False)

    fig = plot_dwell_violin(all_dwell, species_names)
    fig.savefig(PLOT_DIR / "species_arm_dwell_time_violin_subplots.png", dpi=260)
    fig.savefig(PLOT_DIR / "species_arm_dwell_time_violin_subplots.pdf")
    plt.close(fig)

    fig = plot_residence_ccdf(all_dwell, species_names)
    fig.savefig(PLOT_DIR / "species_residence_time_ccdf_subplots.png", dpi=260)
    fig.savefig(PLOT_DIR / "species_residence_time_ccdf_subplots.pdf")
    plt.close(fig)

    fig, velocity_vectors = plot_drift_velocity(species_names, lag_frames)
    fig.savefig(PLOT_DIR / "species_drift_velocity_field_phi2_phi3_subplots.png", dpi=260)
    fig.savefig(PLOT_DIR / "species_drift_velocity_field_phi2_phi3_subplots.pdf")
    plt.close(fig)
    velocity_vectors.to_csv(PLOT_DIR / "species_drift_velocity_field_phi2_phi3_vectors.csv", index=False)

    print(f"Wrote dwell/residence/drift plots to {PLOT_DIR}")
    return PLOT_DIR


if __name__ == "__main__":
    main()
