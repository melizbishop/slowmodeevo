from pathlib import Path
import math
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment


REPO_ROOT = Path("/Users/meganbishop/slowmodeevo").resolve()
KMEANS_ROOT = (
    REPO_ROOT
    / "outputs/single_species_distance_comparison"
    / "pca_multispecies_evensamp"
    / "global_kmeans_N1000_balanced_species"
)
OUT_DIR = KMEANS_ROOT / "species_identity_by_cluster"
EMBEDDING_DIR = OUT_DIR / "cluster_space_embedding"
ARM_DIR = OUT_DIR / "slow_mode_arm_enrichment_occupancy_threshold"
SPECIES_RUN_DIR = ARM_DIR / "species_operators"
PLOT_DIR = OUT_DIR / "species_operator_arm_umap_enrichment_occupancy_threshold"
MANIFEST_CSV = KMEANS_ROOT / "global_kmeans_manifest.csv"
CLUSTER_DISTRIBUTION_CSV = OUT_DIR / "cluster_distribution_within_species.csv"
MOSEQ_ENRICHMENT_CSV = ARM_DIR / "combined_arm_moseq_enrichment.csv"
MOSEQ_LABELS_CSV = (
    REPO_ROOT
    / "outputs/single_species_distance_comparison/tuning/egocentered_and_normalized_distances"
    / "setM_basin_syllable_matching_phylo/setM_moseq_syllable_labels.csv"
)

ARM_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2"]
POINT_SIZE = 12
BACKGROUND_SIZE = 5
REFERENCE_SPECIES = "Mus_caroli"
OCCUPIED_ARM_MASS_THRESHOLD = 1e-6
ARM_EPS = 1e-12
DENSITY_BINS = 90
DENSITY_SMOOTH_SIGMA = 1.25
DENSITY_EPS = 1e-12
MOSEQ_TOP_N_LABELS = 3
DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "matched_arm_density",
    ["#f7fbff", "#cfe8f3", "#8bc2dd", "#2f8fbd", "#045275"],
)


def species_names_from_manifest():
    manifest = pd.read_csv(MANIFEST_CSV)
    return sorted(manifest["species"].astype(str).unique())


def load_cluster_embedding():
    embedding = pd.read_csv(EMBEDDING_DIR / "global_cluster_embedding.csv").sort_values("cluster")
    return embedding


def load_species_operator(species):
    species_dir = SPECIES_RUN_DIR / species
    run_dir = species_dir / "macro_operator"
    if not run_dir.exists():
        run_dir = species_dir
    chi = np.load(run_dir / "chi.npy")
    assignments = np.load(run_dir / "arm_assignments.npy")
    pi = np.load(run_dir / "stationary_distribution.npy")
    active_clusters = np.load(run_dir / "active_clusters.npy")

    expanded_chi_path = species_dir / "expanded_global_chi.npy"
    expanded_assignments_path = species_dir / "expanded_global_arm_assignments.npy"
    macro_mapping_path = species_dir / "global_cluster_to_macro_cluster.npy"
    if expanded_chi_path.exists() and expanded_assignments_path.exists() and macro_mapping_path.exists():
        old_to_macro = np.asarray(np.load(macro_mapping_path), dtype=int)
        macro_member_counts = np.bincount(old_to_macro, minlength=len(pi)).clip(min=1)
        chi = np.load(expanded_chi_path)
        assignments = np.load(expanded_assignments_path)
        pi = pi[old_to_macro] / macro_member_counts[old_to_macro]
        active_clusters = np.arange(old_to_macro.size, dtype=int)

    return {
        "chi": chi,
        "assignments": assignments,
        "pi": pi,
        "active_clusters": active_clusters,
    }


def normalize_arm_vectors(chi, pi):
    chi = np.asarray(chi, dtype=float)
    weights = np.sqrt(np.maximum(np.asarray(pi, dtype=float), 0.0))
    vectors = chi * weights[:, None]
    norms = np.linalg.norm(vectors, axis=0)
    return vectors / np.maximum(norms[None, :], ARM_EPS)


def build_reference_arm_matching(species_names, reference_species=REFERENCE_SPECIES):
    if reference_species not in species_names:
        reference_species = species_names[0]
    reference = load_species_operator(reference_species)
    reference_vectors = normalize_arm_vectors(reference["chi"], reference["pi"])
    reference_masses = np.asarray(reference["chi"], dtype=float).T @ np.asarray(reference["pi"], dtype=float)
    reference_arms = np.flatnonzero(reference_masses > OCCUPIED_ARM_MASS_THRESHOLD)
    if reference_arms.size == 0:
        reference_arms = np.arange(reference["chi"].shape[1], dtype=int)

    provisional_mapping = {}
    rows = []
    for species in species_names:
        run = load_species_operator(species)
        vectors = normalize_arm_vectors(run["chi"], run["pi"])
        masses = np.asarray(run["chi"], dtype=float).T @ np.asarray(run["pi"], dtype=float)
        occupied_arms = np.flatnonzero(masses > OCCUPIED_ARM_MASS_THRESHOLD)
        if occupied_arms.size == 0:
            occupied_arms = np.arange(run["chi"].shape[1], dtype=int)
        similarity = vectors[:, occupied_arms].T @ reference_vectors[:, reference_arms]
        row_ind, col_ind = linear_sum_assignment(-similarity)
        native_to_reference = np.full(run["chi"].shape[1], -1, dtype=int)
        for local_native, local_reference in zip(row_ind, col_ind):
            native_arm = int(occupied_arms[local_native])
            reference_arm = int(reference_arms[local_reference])
            native_to_reference[native_arm] = reference_arm
            rows.append(
                {
                    "species": species,
                    "native_arm": native_arm + 1,
                    "matched_reference_species": reference_species,
                    "matched_reference_arm": reference_arm + 1,
                    "cosine_similarity": float(similarity[local_native, local_reference]),
                    "native_arm_pi_mass": float(masses[native_arm]),
                    "reference_arm_pi_mass": float(reference_masses[reference_arm]),
                }
            )
        for native_arm in range(run["chi"].shape[1]):
            if native_to_reference[native_arm] >= 0:
                continue
            rows.append(
                {
                    "species": species,
                    "native_arm": native_arm + 1,
                    "matched_reference_species": reference_species,
                    "matched_reference_arm": -1,
                    "cosine_similarity": np.nan,
                    "native_arm_pi_mass": float(masses[native_arm]),
                    "reference_arm_pi_mass": np.nan,
                }
            )
        provisional_mapping[species] = {
            "native_to_reference": native_to_reference,
            "native_arm_pi_mass": masses,
            "reference_species": reference_species,
            "n_reference_arms": int(reference["chi"].shape[1]),
        }

    matched_mass_by_reference = np.zeros(reference["chi"].shape[1], dtype=float)
    for entry in provisional_mapping.values():
        for native_arm, reference_arm in enumerate(entry["native_to_reference"]):
            if reference_arm >= 0:
                matched_mass_by_reference[int(reference_arm)] += float(entry["native_arm_pi_mass"][native_arm])

    reference_order = np.lexsort((np.arange(reference["chi"].shape[1]), -matched_mass_by_reference))
    reference_to_matched = np.full(reference["chi"].shape[1], -1, dtype=int)
    for matched_arm, reference_arm in enumerate(reference_order):
        reference_to_matched[int(reference_arm)] = int(matched_arm)

    mapping = {}
    for species, entry in provisional_mapping.items():
        native_to_matched = np.full(entry["native_to_reference"].shape, -1, dtype=int)
        for native_arm, reference_arm in enumerate(entry["native_to_reference"]):
            if reference_arm >= 0:
                native_to_matched[native_arm] = int(reference_to_matched[int(reference_arm)])
        mapping[species] = {
            "native_to_reference": entry["native_to_reference"],
            "native_to_matched": native_to_matched,
            "reference_species": reference_species,
            "reference_to_matched": reference_to_matched,
            "matched_to_reference": reference_order,
            "matched_arm_total_pi_mass": matched_mass_by_reference[reference_order],
            "n_reference_arms": int(reference["chi"].shape[1]),
            "n_matched_arms": int(reference["chi"].shape[1]),
        }

    mapping_table = pd.DataFrame(rows)
    if not mapping_table.empty:
        matched_arms = []
        matched_masses = []
        for reference_arm in mapping_table["matched_reference_arm"].to_numpy():
            if int(reference_arm) > 0:
                matched_arm = int(reference_to_matched[int(reference_arm) - 1]) + 1
                total_mass = float(matched_mass_by_reference[int(reference_arm) - 1])
            else:
                matched_arm = -1
                total_mass = np.nan
            matched_arms.append(matched_arm)
            matched_masses.append(total_mass)
        mapping_table["matched_arm"] = matched_arms
        mapping_table["matched_arm_total_pi_mass"] = matched_masses
        mapping_table["matched_arm_ordering"] = "descending_total_native_pi_mass"
    return mapping, mapping_table


def matched_chi(run, mapping_entry):
    chi = np.asarray(run["chi"], dtype=float)
    native_to_matched = mapping_entry.get("native_to_matched", mapping_entry["native_to_reference"])
    remapped = np.zeros((chi.shape[0], int(mapping_entry.get("n_matched_arms", mapping_entry["n_reference_arms"]))), dtype=float)
    for native_arm, matched_arm in enumerate(native_to_matched):
        if matched_arm >= 0:
            remapped[:, int(matched_arm)] = chi[:, native_arm]
    return remapped


def matched_assignments(run, mapping_entry):
    assignments = np.asarray(run["assignments"], dtype=int)
    remapped = np.full(assignments.shape, -1, dtype=int)
    native_to_matched = mapping_entry.get("native_to_matched", mapping_entry["native_to_reference"])
    for native_arm, matched_arm in enumerate(native_to_matched):
        if matched_arm >= 0:
            remapped[assignments == native_arm] = int(matched_arm)
    return remapped


def load_cluster_given_species(species_names):
    table = pd.read_csv(CLUSTER_DISTRIBUTION_CSV).sort_values("cluster")
    return {
        species: table[species].to_numpy(float)
        for species in species_names
        if species in table.columns
    }


def load_moseq_label_map():
    if not MOSEQ_LABELS_CSV.exists():
        return {}
    labels = pd.read_csv(MOSEQ_LABELS_CSV)
    if "moseq_cluster" not in labels.columns:
        return {}
    label_col = "syllable_label" if "syllable_label" in labels.columns else None
    if label_col is None:
        for candidate in ("moseq_label", "moseq_syllable_name"):
            if candidate in labels.columns:
                label_col = candidate
                break
    if label_col is None:
        return {}
    return {
        int(row["moseq_cluster"]): str(row[label_col])
        for _, row in labels.dropna(subset=["moseq_cluster"]).iterrows()
        if pd.notna(row.get(label_col))
    }


def matched_arm_moseq_labels(species_names, arm_mapping):
    if not MOSEQ_ENRICHMENT_CSV.exists():
        return pd.DataFrame(), {}
    label_map = load_moseq_label_map()
    moseq = pd.read_csv(MOSEQ_ENRICHMENT_CSV)
    moseq = moseq[
        (moseq["operator_scope"] == "species_operator")
        & (moseq["species"].isin(species_names))
        & (moseq["operator_species"] == moseq["species"])
    ].copy()
    rows = []
    for species, group in moseq.groupby("species", sort=False):
        native_to_matched = arm_mapping[species]["native_to_matched"]
        group = group.copy()
        group["native_arm"] = group["arm"].astype(int)
        group["matched_arm"] = group["native_arm"].map(
            lambda arm: int(native_to_matched[arm]) + 1
            if 0 <= int(arm) < len(native_to_matched) and native_to_matched[int(arm)] >= 0
            else -1
        )
        group = group[group["matched_arm"] > 0]
        for matched_arm, arm_group in group.groupby("matched_arm", sort=True):
            top = arm_group.sort_values(
                ["p_moseq_given_arm_species", "weighted_frames"],
                ascending=False,
            ).head(MOSEQ_TOP_N_LABELS)
            for rank, (_, row) in enumerate(top.iterrows(), start=1):
                syllable = int(row["moseq_cluster"])
                rows.append(
                    {
                        "species": species,
                        "matched_arm": int(matched_arm),
                        "native_arm": int(row["native_arm"]) + 1,
                        "rank": rank,
                        "moseq_cluster": syllable,
                        "moseq_label": label_map.get(syllable, str(syllable)),
                        "p_moseq_given_arm_species": float(row["p_moseq_given_arm_species"]),
                        "log2_enrichment": float(row["log2_enrichment"]),
                        "weighted_frames": float(row["weighted_frames"]),
                    }
                )
    table = pd.DataFrame(rows)
    label_text = {}
    if not table.empty:
        for (species, matched_arm), group in table.groupby(["species", "matched_arm"], sort=False):
            lines = []
            for _, row in group.sort_values("rank").iterrows():
                label = str(row["moseq_label"])
                label = label.replace("_", " ")
                label = "\n".join(textwrap.wrap(label, width=22)) if len(label) > 22 else label
                lines.append(
                    f"{int(row['rank'])}. {label} "
                    f"({100 * row['p_moseq_given_arm_species']:.1f}%)"
                )
            label_text[(species, int(matched_arm) - 1)] = "\n".join(lines)
    return table, label_text


def point_sizes(weights, *, min_size=POINT_SIZE, max_size=90):
    weights = np.asarray(weights, dtype=float)
    if weights.size == 0 or np.nanmax(weights) <= 0:
        return np.full(weights.shape, min_size, dtype=float)
    scaled = np.sqrt(weights / np.nanmax(weights))
    return min_size + (max_size - min_size) * scaled


def add_shared_axes_labels(fig, embedding_method):
    x_label = "UMAP 1" if embedding_method == "umap" else "PC 1"
    y_label = "UMAP 2" if embedding_method == "umap" else "PC 2"
    fig.supxlabel(x_label)
    fig.supylabel(y_label)


def shared_umap_edges(embedding, bins=DENSITY_BINS):
    x = embedding["x"].to_numpy(float)
    y = embedding["y"].to_numpy(float)
    x_padding = 0.03 * max(np.ptp(x), 1e-12)
    y_padding = 0.03 * max(np.ptp(y), 1e-12)
    x_edges = np.linspace(x.min() - x_padding, x.max() + x_padding, int(bins) + 1)
    y_edges = np.linspace(y.min() - y_padding, y.max() + y_padding, int(bins) + 1)
    return x_edges, y_edges


def add_umap_outline(ax, embedding):
    ax.scatter(
        embedding["x"].to_numpy(float),
        embedding["y"].to_numpy(float),
        s=4,
        c="#b9c0c8",
        alpha=0.5,
        linewidths=0,
        zorder=1,
    )


def matched_arm_density_grids(embedding, species_names, cluster_given_species, arm_mapping):
    x = embedding["x"].to_numpy(float)
    y = embedding["y"].to_numpy(float)
    x_edges, y_edges = shared_umap_edges(embedding)
    n_matched_arms = arm_mapping[species_names[0]]["n_matched_arms"]
    density = {}
    raw_density = {}
    grid_rows = []
    for species in species_names:
        run = load_species_operator(species)
        chi = matched_chi(run, arm_mapping[species])
        occupancy = cluster_given_species.get(species, np.zeros(len(x), dtype=float))
        active = np.asarray(run["active_clusters"], dtype=int)
        active = active[(active >= 0) & (active < len(x))]
        density[species] = {}
        raw_density[species] = {}
        for arm in range(n_matched_arms):
            weights = np.zeros(len(x), dtype=float)
            weights[active] = np.maximum(occupancy[active], 0.0) * np.maximum(chi[active, arm], 0.0)
            H, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=weights)
            H = H.T
            H_smooth = gaussian_filter(H, sigma=DENSITY_SMOOTH_SIGMA, mode="constant")
            total = float(H_smooth.sum())
            H_norm = H_smooth / total if total > 0 else H_smooth
            raw_density[species][arm] = H
            density[species][arm] = H_norm
            np.save(PLOT_DIR / f"{species}_matched_arm_{arm + 1}_usage_density_grid.npy", H_norm)
            np.save(PLOT_DIR / f"{species}_matched_arm_{arm + 1}_raw_usage_density_grid.npy", H)
            grid_rows.append(
                {
                    "species": species,
                    "matched_arm": arm + 1,
                    "raw_weight_sum": float(weights.sum()),
                    "smoothed_grid_sum": total,
                    "normalized_grid_sum": float(H_norm.sum()),
                    "n_active_clusters": int(active.size),
                }
            )
    return density, x_edges, y_edges, pd.DataFrame(grid_rows)


def plot_matched_arm_density_by_species(
    embedding, density, species_names, arm, x_edges, y_edges
):
    ncols = 4
    nrows = math.ceil(len(species_names) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 3.7 * nrows),
        squeeze=False, constrained_layout=True,
    )
    values = np.concatenate([
        density[species][arm].ravel()
        for species in species_names
        if np.any(density[species][arm] > 0)
    ])
    vmax = max(float(np.nanpercentile(values[values > 0], 99.5)) if np.any(values > 0) else 1.0, 1e-12)
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    last = None
    for ax, species in zip(axes.ravel(), species_names):
        grid = np.clip(density[species][arm], 0, vmax)
        denom = max(float(np.nanmax(grid)), DENSITY_EPS)
        alpha = 0.9 * np.clip(grid / denom, 0.0, 1.0) ** 0.55
        add_umap_outline(ax, embedding)
        last = ax.imshow(
            grid,
            origin="lower",
            extent=extent,
            cmap=DENSITY_CMAP,
            vmin=0,
            vmax=vmax,
            alpha=alpha,
            aspect="auto",
            zorder=2,
        )
        ax.set_title(species.replace("_", " "), fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[len(species_names):]:
        ax.axis("off")
    fig.suptitle(
        f"Matched arm {arm + 1}: per-species normalized density",
        fontsize=14,
        fontweight="bold",
    )
    fig.supxlabel("UMAP 1")
    fig.supylabel("UMAP 2")
    if last is not None:
        fig.colorbar(last, ax=axes.ravel().tolist(), label="normalized matched-arm density")
    return fig


def plot_moseq_top_syllables_by_species(moseq_label_table, species_names, arm):
    arm_table = moseq_label_table[moseq_label_table["matched_arm"] == arm + 1].copy()
    ncols = 4
    nrows = math.ceil(len(species_names) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.7 * ncols, 3.2 * nrows),
        squeeze=False, constrained_layout=True,
    )
    xmax = max(float(arm_table["p_moseq_given_arm_species"].max()) if not arm_table.empty else 0.01, 0.01)
    for ax, species in zip(axes.ravel(), species_names):
        sub = arm_table[arm_table["species"] == species].sort_values("rank", ascending=False)
        if sub.empty:
            ax.text(0.5, 0.5, "no MoSeq labels", transform=ax.transAxes, ha="center", va="center")
            ax.set_axis_off()
            continue
        labels = [
            "\n".join(textwrap.wrap(str(label).replace("_", " "), width=24))
            for label in sub["moseq_label"]
        ]
        values = sub["p_moseq_given_arm_species"].to_numpy(float)
        colors = plt.cm.Blues(np.linspace(0.45, 0.85, len(values)))
        ax.barh(np.arange(len(values)), values, color=colors, edgecolor="white", linewidth=0.6)
        ax.set_yticks(np.arange(len(values)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlim(0, xmax * 1.08)
        ax.set_title(species.replace("_", " "), fontsize=10)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="x", color="0.88", linewidth=0.6)
        for y_pos, value in enumerate(values):
            ax.text(
                value + xmax * 0.015,
                y_pos,
                f"{100 * value:.1f}%",
                va="center",
                fontsize=7,
                color="0.25",
            )
    for ax in axes.ravel()[len(species_names):]:
        ax.axis("off")
    fig.suptitle(
        f"Matched arm {arm + 1}: top MoSeq syllables by species",
        fontsize=14,
        fontweight="bold",
    )
    fig.supxlabel("p(MoSeq syllable | matched arm, species)")
    return fig


def plot_moseq_probability_heatmap_by_arm(moseq_label_table, species_names, arm):
    arm_table = moseq_label_table[moseq_label_table["matched_arm"] == arm + 1].copy()
    if arm_table.empty:
        return None
    selected = (
        arm_table.sort_values(["rank", "p_moseq_given_arm_species"], ascending=[True, False])
        ["moseq_cluster"]
        .drop_duplicates()
        .head(16)
        .astype(int)
        .tolist()
    )
    matrix = (
        arm_table[arm_table["moseq_cluster"].isin(selected)]
        .pivot_table(
            index="species",
            columns="moseq_cluster",
            values="p_moseq_given_arm_species",
            aggfunc="max",
            fill_value=0.0,
        )
        .reindex(index=species_names, columns=selected, fill_value=0.0)
    )
    label_by_cluster = (
        arm_table.drop_duplicates("moseq_cluster")
        .set_index("moseq_cluster")["moseq_label"]
        .to_dict()
    )
    xlabels = [
        "\n".join(textwrap.wrap(str(label_by_cluster.get(cluster, cluster)).replace("_", " "), width=14))
        for cluster in selected
    ]
    fig, ax = plt.subplots(
        figsize=(max(9.5, 0.75 * len(selected)), 4.8),
        constrained_layout=True,
    )
    im = ax.imshow(matrix.to_numpy(float), cmap="Blues", aspect="auto")
    ax.set_yticks(np.arange(len(species_names)))
    ax.set_yticklabels([species.replace("_", " ") for species in species_names], fontsize=8)
    ax.set_xticks(np.arange(len(selected)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=7)
    ax.set_title(f"Matched arm {arm + 1}: MoSeq syllable probabilities", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, label="p(MoSeq syllable | matched arm, species)")
    return fig


def plot_species_matched_arm_density_panel(
    embedding, density, species, n_matched_arms, x_edges, y_edges
):
    ncols = min(4, n_matched_arms)
    nrows = math.ceil(n_matched_arms / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 3.7 * nrows),
        squeeze=False, constrained_layout=True,
    )
    values = np.concatenate([
        density[species][arm].ravel()
        for arm in range(n_matched_arms)
        if np.any(density[species][arm] > 0)
    ])
    vmax = max(float(np.nanpercentile(values[values > 0], 99.5)) if np.any(values > 0) else 1.0, 1e-12)
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    last = None
    for ax, arm in zip(axes.ravel(), range(n_matched_arms)):
        grid = np.clip(density[species][arm], 0, vmax)
        denom = max(float(np.nanmax(grid)), DENSITY_EPS)
        alpha = 0.9 * np.clip(grid / denom, 0.0, 1.0) ** 0.55
        add_umap_outline(ax, embedding)
        last = ax.imshow(
            grid,
            origin="lower",
            extent=extent,
            cmap=DENSITY_CMAP,
            vmin=0,
            vmax=vmax,
            alpha=alpha,
            aspect="auto",
            zorder=2,
        )
        ax.set_title(f"matched arm {arm + 1}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[n_matched_arms:]:
        ax.axis("off")
    fig.suptitle(
        f"{species.replace('_', ' ')}: matched-arm normalized density",
        fontsize=13,
        fontweight="bold",
    )
    fig.supxlabel("UMAP 1")
    fig.supylabel("UMAP 2")
    if last is not None:
        fig.colorbar(last, ax=axes.ravel().tolist(), label="normalized matched-arm density")
    return fig


def plot_all_species_hard_arm_umap(embedding, species_names, cluster_given_species):
    x = embedding["x"].to_numpy(float)
    y = embedding["y"].to_numpy(float)
    embedding_method = str(embedding.get("embedding_method", pd.Series(["umap"])).iloc[0])
    ncols = 4
    nrows = math.ceil(len(species_names) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows),
        squeeze=False, constrained_layout=True,
    )
    rows = []
    for ax, species in zip(axes.ravel(), species_names):
        run = load_species_operator(species)
        assignments = np.asarray(run["assignments"], dtype=int)
        active = np.asarray(run["active_clusters"], dtype=int)
        active = active[(active >= 0) & (active < len(x))]
        occupancy = cluster_given_species.get(species, np.zeros(len(x), dtype=float))
        ax.scatter(x, y, s=BACKGROUND_SIZE, c="0.90", alpha=0.45, linewidths=0)
        for arm in range(run["chi"].shape[1]):
            mask = active[assignments[active] == arm]
            if mask.size == 0:
                continue
            sizes = point_sizes(occupancy[mask], min_size=POINT_SIZE, max_size=76)
            ax.scatter(
                x[mask], y[mask], s=sizes,
                color=ARM_COLORS[arm % len(ARM_COLORS)],
                alpha=0.82, linewidths=0, label=f"arm {arm + 1}",
            )
        ax.set_title(species.replace("_", " "), fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(fontsize=6, frameon=False, loc="best")
        for cluster in active:
            rows.append(
                {
                    "species": species,
                    "cluster": int(cluster),
                    "umap_x": float(x[cluster]),
                    "umap_y": float(y[cluster]),
                    "hard_arm": int(assignments[cluster] + 1) if assignments[cluster] >= 0 else -1,
                    "species_cluster_frequency": float(occupancy[cluster]),
                    "stationary_probability": float(run["pi"][cluster]),
                }
            )
    for ax in axes.ravel()[len(species_names):]:
        ax.axis("off")
    fig.suptitle(
        "Species-specific transfer-operator hard arms on shared delay-embedded UMAP",
        fontsize=14, fontweight="bold",
    )
    add_shared_axes_labels(fig, embedding_method)
    return fig, pd.DataFrame(rows)


def plot_all_species_hungarian_matched_hard_arm_umap(
    embedding, species_names, cluster_given_species, arm_mapping
):
    x = embedding["x"].to_numpy(float)
    y = embedding["y"].to_numpy(float)
    embedding_method = str(embedding.get("embedding_method", pd.Series(["umap"])).iloc[0])
    n_matched_arms = arm_mapping[species_names[0]]["n_matched_arms"]
    ncols = 4
    nrows = math.ceil(len(species_names) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows),
        squeeze=False, constrained_layout=True,
    )
    rows = []
    for ax, species in zip(axes.ravel(), species_names):
        run = load_species_operator(species)
        assignments = matched_assignments(run, arm_mapping[species])
        active = np.asarray(run["active_clusters"], dtype=int)
        active = active[(active >= 0) & (active < len(x))]
        occupancy = cluster_given_species.get(species, np.zeros(len(x), dtype=float))
        ax.scatter(x, y, s=BACKGROUND_SIZE, c="0.90", alpha=0.45, linewidths=0)
        for arm in range(n_matched_arms):
            mask = active[assignments[active] == arm]
            if mask.size == 0:
                continue
            sizes = point_sizes(occupancy[mask], min_size=POINT_SIZE, max_size=76)
            ax.scatter(
                x[mask], y[mask], s=sizes,
                color=ARM_COLORS[arm % len(ARM_COLORS)],
                alpha=0.82, linewidths=0, label=f"matched arm {arm + 1}",
            )
        unmatched = active[assignments[active] < 0]
        if unmatched.size:
            ax.scatter(
                x[unmatched], y[unmatched],
                s=point_sizes(occupancy[unmatched], min_size=8, max_size=42),
                c="0.35", alpha=0.45, linewidths=0, label="unmatched",
            )
        ax.set_title(species.replace("_", " "), fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(fontsize=6, frameon=False, loc="best")
        native_assignments = np.asarray(run["assignments"], dtype=int)
        for cluster in active:
            rows.append(
                {
                    "species": species,
                    "cluster": int(cluster),
                    "umap_x": float(x[cluster]),
                    "umap_y": float(y[cluster]),
                    "native_arm": int(native_assignments[cluster] + 1)
                    if native_assignments[cluster] >= 0 else -1,
                    "matched_arm": int(assignments[cluster] + 1)
                    if assignments[cluster] >= 0 else -1,
                    "species_cluster_frequency": float(occupancy[cluster]),
                    "stationary_probability": float(run["pi"][cluster]),
                }
            )
    for ax in axes.ravel()[len(species_names):]:
        ax.axis("off")
    fig.suptitle(
        "Hungarian-matched species-specific arms on shared delay-embedded UMAP",
        fontsize=14, fontweight="bold",
    )
    add_shared_axes_labels(fig, embedding_method)
    return fig, pd.DataFrame(rows)


def plot_species_soft_chi_umap(embedding, species, cluster_given_species, arm_mapping=None):
    x = embedding["x"].to_numpy(float)
    y = embedding["y"].to_numpy(float)
    embedding_method = str(embedding.get("embedding_method", pd.Series(["umap"])).iloc[0])
    run = load_species_operator(species)
    chi = matched_chi(run, arm_mapping[species]) if arm_mapping is not None else np.asarray(run["chi"], dtype=float)
    active = np.asarray(run["active_clusters"], dtype=int)
    active = active[(active >= 0) & (active < len(x))]
    occupancy = cluster_given_species.get(species, np.zeros(len(x), dtype=float))
    n_arms = chi.shape[1]
    ncols = min(4, n_arms)
    nrows = math.ceil(n_arms / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows),
        squeeze=False, constrained_layout=True,
    )
    last = None
    sizes = point_sizes(occupancy[active], min_size=POINT_SIZE, max_size=82)
    for ax, arm in zip(axes.ravel(), range(n_arms)):
        ax.scatter(x, y, s=BACKGROUND_SIZE, c="0.92", alpha=0.42, linewidths=0)
        last = ax.scatter(
            x[active], y[active],
            c=chi[active, arm], s=sizes, cmap="magma", vmin=0, vmax=1,
            alpha=0.88, linewidths=0,
        )
        title = f"matched arm {arm + 1}" if arm_mapping is not None else f"arm {arm + 1}"
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[n_arms:]:
        ax.axis("off")
    fig.suptitle(
        (
            f"{species.replace('_', ' ')}: Hungarian-matched soft chi arms on shared delay-embedded UMAP"
            if arm_mapping is not None
            else f"{species.replace('_', ' ')}: soft chi arms on shared delay-embedded UMAP"
        ),
        fontsize=13, fontweight="bold",
    )
    add_shared_axes_labels(fig, embedding_method)
    if last is not None:
        fig.colorbar(last, ax=axes.ravel().tolist(), label="species-specific chi membership")
    return fig


def plot_species_arm_enrichment_umap(embedding, species, cluster_given_species, arm_mapping=None):
    x = embedding["x"].to_numpy(float)
    y = embedding["y"].to_numpy(float)
    embedding_method = str(embedding.get("embedding_method", pd.Series(["umap"])).iloc[0])
    run = load_species_operator(species)
    chi = matched_chi(run, arm_mapping[species]) if arm_mapping is not None else np.asarray(run["chi"], dtype=float)
    active = np.asarray(run["active_clusters"], dtype=int)
    active = active[(active >= 0) & (active < len(x))]
    occupancy = cluster_given_species.get(species, np.zeros(len(x), dtype=float))
    n_arms = chi.shape[1]
    ncols = min(4, n_arms)
    nrows = math.ceil(n_arms / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows),
        squeeze=False, constrained_layout=True,
    )
    rows = []
    baseline = np.maximum(np.asarray(run["pi"], dtype=float), 1e-12)
    species_frequency = np.maximum(occupancy, 0.0)
    # Enrichment asks where this species actually visits an arm more than the
    # operator stationary mass alone would suggest.
    score = np.log2((species_frequency[:, None] * chi + 1e-12) / (baseline[:, None] * chi.sum(axis=0)[None, :] + 1e-12))
    finite = score[np.isfinite(score)]
    vmax = max(float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 1.0, 1e-6)
    sizes = point_sizes(occupancy[active], min_size=POINT_SIZE, max_size=82)
    last = None
    for ax, arm in zip(axes.ravel(), range(n_arms)):
        ax.scatter(x, y, s=BACKGROUND_SIZE, c="0.92", alpha=0.42, linewidths=0)
        values = np.clip(score[active, arm], -vmax, vmax)
        last = ax.scatter(
            x[active], y[active],
            c=values, s=sizes, cmap="coolwarm", vmin=-vmax, vmax=vmax,
            alpha=0.88, linewidths=0,
        )
        title = f"matched arm {arm + 1}" if arm_mapping is not None else f"arm {arm + 1}"
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for cluster, value in zip(active, score[active, arm]):
            rows.append(
                {
                    "species": species,
                    "arm": arm + 1,
                    "arm_label_type": "hungarian_matched"
                    if arm_mapping is not None else "native",
                    "cluster": int(cluster),
                    "umap_x": float(x[cluster]),
                    "umap_y": float(y[cluster]),
                    "chi": float(chi[cluster, arm]),
                    "species_cluster_frequency": float(occupancy[cluster]),
                    "stationary_probability": float(run["pi"][cluster]),
                    "log2_species_arm_enrichment": float(value),
                }
            )
    for ax in axes.ravel()[n_arms:]:
        ax.axis("off")
    fig.suptitle(
        (
            f"{species.replace('_', ' ')}: Hungarian-matched arm-enriched usage on shared delay-embedded UMAP"
            if arm_mapping is not None
            else f"{species.replace('_', ' ')}: arm-enriched usage on shared delay-embedded UMAP"
        ),
        fontsize=13, fontweight="bold",
    )
    add_shared_axes_labels(fig, embedding_method)
    if last is not None:
        fig.colorbar(last, ax=axes.ravel().tolist(), label="log2 species arm enrichment")
    return fig, pd.DataFrame(rows)


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    species_names = species_names_from_manifest()
    embedding = load_cluster_embedding()
    cluster_given_species = load_cluster_given_species(species_names)
    arm_mapping, mapping_table = build_reference_arm_matching(species_names)
    mapping_table.to_csv(PLOT_DIR / "hungarian_reference_arm_mapping.csv", index=False)
    mapping_table.to_csv(PLOT_DIR / "hungarian_matched_arm_mapping.csv", index=False)
    n_matched_arms = arm_mapping[species_names[0]]["n_matched_arms"]

    fig, hard_rows = plot_all_species_hard_arm_umap(embedding, species_names, cluster_given_species)
    fig.savefig(PLOT_DIR / "all_species_species_operator_hard_arms_shared_delay_umap.png", dpi=260)
    fig.savefig(PLOT_DIR / "all_species_species_operator_hard_arms_shared_delay_umap.pdf")
    plt.close(fig)
    hard_rows.to_csv(PLOT_DIR / "all_species_species_operator_hard_arms_shared_delay_umap.csv", index=False)

    fig, matched_hard_rows = plot_all_species_hungarian_matched_hard_arm_umap(
        embedding, species_names, cluster_given_species, arm_mapping
    )
    fig.savefig(
        PLOT_DIR / "all_species_hungarian_matched_species_operator_hard_arms_shared_delay_umap.png",
        dpi=260,
    )
    fig.savefig(PLOT_DIR / "all_species_hungarian_matched_species_operator_hard_arms_shared_delay_umap.pdf")
    plt.close(fig)
    matched_hard_rows.to_csv(
        PLOT_DIR / "all_species_hungarian_matched_species_operator_hard_arms_shared_delay_umap.csv",
        index=False,
    )

    enrichment_tables = []
    matched_enrichment_tables = []
    for species in species_names:
        fig = plot_species_soft_chi_umap(embedding, species, cluster_given_species)
        fig.savefig(PLOT_DIR / f"{species}_species_operator_soft_chi_arms_shared_delay_umap.png", dpi=260)
        fig.savefig(PLOT_DIR / f"{species}_species_operator_soft_chi_arms_shared_delay_umap.pdf")
        plt.close(fig)

        fig, enrichment = plot_species_arm_enrichment_umap(embedding, species, cluster_given_species)
        fig.savefig(PLOT_DIR / f"{species}_species_operator_arm_enrichment_shared_delay_umap.png", dpi=260)
        fig.savefig(PLOT_DIR / f"{species}_species_operator_arm_enrichment_shared_delay_umap.pdf")
        plt.close(fig)
        enrichment.to_csv(PLOT_DIR / f"{species}_species_operator_arm_enrichment_shared_delay_umap.csv", index=False)
        enrichment_tables.append(enrichment)

        fig = plot_species_soft_chi_umap(embedding, species, cluster_given_species, arm_mapping=arm_mapping)
        fig.savefig(PLOT_DIR / f"{species}_hungarian_matched_species_operator_soft_chi_arms_shared_delay_umap.png", dpi=260)
        fig.savefig(PLOT_DIR / f"{species}_hungarian_matched_species_operator_soft_chi_arms_shared_delay_umap.pdf")
        plt.close(fig)

        fig, matched_enrichment = plot_species_arm_enrichment_umap(
            embedding, species, cluster_given_species, arm_mapping=arm_mapping
        )
        fig.savefig(PLOT_DIR / f"{species}_hungarian_matched_species_operator_arm_enrichment_shared_delay_umap.png", dpi=260)
        fig.savefig(PLOT_DIR / f"{species}_hungarian_matched_species_operator_arm_enrichment_shared_delay_umap.pdf")
        plt.close(fig)
        matched_enrichment.to_csv(
            PLOT_DIR / f"{species}_hungarian_matched_species_operator_arm_enrichment_shared_delay_umap.csv",
            index=False,
        )
        matched_enrichment_tables.append(matched_enrichment)

    if enrichment_tables:
        pd.concat(enrichment_tables, ignore_index=True).to_csv(
            PLOT_DIR / "all_species_species_operator_arm_enrichment_shared_delay_umap.csv",
            index=False,
        )
    if matched_enrichment_tables:
        pd.concat(matched_enrichment_tables, ignore_index=True).to_csv(
            PLOT_DIR / "all_species_hungarian_matched_species_operator_arm_enrichment_shared_delay_umap.csv",
            index=False,
        )

    density, x_edges, y_edges, density_summary = (
        matched_arm_density_grids(embedding, species_names, cluster_given_species, arm_mapping)
    )
    density_summary.to_csv(
        PLOT_DIR / "hungarian_matched_arm_density_grid_summary.csv",
        index=False,
    )
    moseq_label_table, _ = matched_arm_moseq_labels(species_names, arm_mapping)
    if not moseq_label_table.empty:
        moseq_label_table.to_csv(
            PLOT_DIR / "hungarian_matched_arm_top_moseq_labels.csv",
            index=False,
        )

    for arm in range(n_matched_arms):
        fig = plot_matched_arm_density_by_species(
            embedding,
            density,
            species_names,
            arm,
            x_edges,
            y_edges,
        )
        fig.savefig(
            PLOT_DIR / f"hungarian_matched_arm_{arm + 1}_density_by_species.png",
            dpi=260,
        )
        fig.savefig(PLOT_DIR / f"hungarian_matched_arm_{arm + 1}_density_by_species.pdf")
        plt.close(fig)

        if not moseq_label_table.empty:
            fig = plot_moseq_top_syllables_by_species(
                moseq_label_table,
                species_names,
                arm,
            )
            fig.savefig(
                PLOT_DIR / f"hungarian_matched_arm_{arm + 1}_moseq_top_syllables_by_species.png",
                dpi=260,
            )
            fig.savefig(PLOT_DIR / f"hungarian_matched_arm_{arm + 1}_moseq_top_syllables_by_species.pdf")
            plt.close(fig)

            fig = plot_moseq_probability_heatmap_by_arm(
                moseq_label_table,
                species_names,
                arm,
            )
            if fig is not None:
                fig.savefig(
                    PLOT_DIR / f"hungarian_matched_arm_{arm + 1}_moseq_probability_heatmap.png",
                    dpi=260,
                )
                fig.savefig(PLOT_DIR / f"hungarian_matched_arm_{arm + 1}_moseq_probability_heatmap.pdf")
                plt.close(fig)

    for species in species_names:
        fig = plot_species_matched_arm_density_panel(
            embedding,
            density,
            species,
            n_matched_arms,
            x_edges,
            y_edges,
        )
        fig.savefig(
            PLOT_DIR / f"{species}_hungarian_matched_arm_density_panel.png",
            dpi=260,
        )
        fig.savefig(PLOT_DIR / f"{species}_hungarian_matched_arm_density_panel.pdf")
        plt.close(fig)

    print(f"Wrote species-operator arm UMAP enrichment plots to {PLOT_DIR}")
    return PLOT_DIR


if __name__ == "__main__":
    main()
