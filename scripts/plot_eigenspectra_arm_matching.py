from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


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
PLOT_DIR = OUT_DIR / "eigenspectra_arm_matching_occupancy_threshold"
MANIFEST_CSV = KMEANS_ROOT / "global_kmeans_manifest.csv"

N_EIGENVALUES_TO_PLOT = 30
ARM_EPS = 1e-12
OCCUPIED_ARM_MASS_THRESHOLD = 1e-6


def species_names_from_manifest():
    manifest = pd.read_csv(MANIFEST_CSV)
    return sorted(manifest["species"].astype(str).unique())


def load_species_arrays(species):
    species_dir = SPECIES_RUN_DIR / species
    run_dir = species_dir / "macro_operator"
    if not run_dir.exists():
        run_dir = species_dir
    chi = np.load(run_dir / "chi.npy")
    pi = np.load(run_dir / "stationary_distribution.npy")

    expanded_chi_path = species_dir / "expanded_global_chi.npy"
    macro_mapping_path = species_dir / "global_cluster_to_macro_cluster.npy"
    if expanded_chi_path.exists() and macro_mapping_path.exists():
        old_to_macro = np.asarray(np.load(macro_mapping_path), dtype=int)
        macro_member_counts = np.bincount(old_to_macro, minlength=len(pi)).clip(min=1)
        chi = np.load(expanded_chi_path)
        pi = pi[old_to_macro] / macro_member_counts[old_to_macro]

    return {
        "T_fit": np.load(run_dir / "gpcca_fit_transfer_operator.npy"),
        "chi": chi,
        "pi": pi,
    }


def compute_eigenspectrum(T):
    eigvals = np.linalg.eigvals(np.asarray(T, dtype=float))
    eigvals = np.asarray(eigvals)
    order = np.argsort(np.abs(eigvals))[::-1]
    return eigvals[order]


def plot_eigenspectra(species_names):
    rows = []
    fig, ax = plt.subplots(figsize=(8.4, 5.6), constrained_layout=True)
    for species in species_names:
        T = load_species_arrays(species)["T_fit"]
        eigvals = compute_eigenspectrum(T)
        n_plot = min(N_EIGENVALUES_TO_PLOT, eigvals.size)
        x = np.arange(1, n_plot + 1)
        ax.plot(x, np.abs(eigvals[:n_plot]), marker="o", ms=3, lw=1.2, label=species)
        for rank, value in enumerate(eigvals, start=1):
            rows.append(
                {
                    "species": species,
                    "rank": rank,
                    "eigenvalue_real": float(np.real(value)),
                    "eigenvalue_imag": float(np.imag(value)),
                    "eigenvalue_abs": float(np.abs(value)),
                }
            )
    ax.axhline(1.0, color="0.4", lw=0.9, ls="--")
    ax.set_xlabel("eigenvalue rank")
    ax.set_ylabel("absolute eigenvalue")
    ax.set_title("Species-specific transfer-operator eigenspectra")
    ax.set_ylim(0, 1.04)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2, frameon=False)
    return fig, pd.DataFrame(rows)


def normalize_arm_vectors(chi, pi=None):
    chi = np.asarray(chi, dtype=float)
    if pi is None:
        weights = np.ones(chi.shape[0], dtype=float)
    else:
        weights = np.sqrt(np.maximum(np.asarray(pi, dtype=float), 0.0))
    vectors = chi * weights[:, None]
    norms = np.linalg.norm(vectors, axis=0)
    return vectors / np.maximum(norms[None, :], ARM_EPS)


def cosine_similarity_matrix(vectors_a, vectors_b):
    return np.asarray(vectors_a.T @ vectors_b, dtype=float)


def hungarian_match_arms(similarity):
    row_ind, col_ind = linear_sum_assignment(-similarity)
    order = np.argsort(row_ind)
    row_ind = row_ind[order]
    col_ind = col_ind[order]
    matched = similarity[row_ind, col_ind]
    return row_ind, col_ind, matched


def compute_arm_matching(species_names):
    arrays = {species: load_species_arrays(species) for species in species_names}
    arm_vectors = {
        species: normalize_arm_vectors(arrays[species]["chi"], arrays[species]["pi"])
        for species in species_names
    }
    arm_masses = {
        species: np.asarray(arrays[species]["chi"], dtype=float).T
        @ np.asarray(arrays[species]["pi"], dtype=float)
        for species in species_names
    }

    mean_matrix = pd.DataFrame(index=species_names, columns=species_names, dtype=float)
    min_matrix = pd.DataFrame(index=species_names, columns=species_names, dtype=float)
    occupied_mean_matrix = pd.DataFrame(index=species_names, columns=species_names, dtype=float)
    occupied_min_matrix = pd.DataFrame(index=species_names, columns=species_names, dtype=float)
    detail_rows = []
    occupied_detail_rows = []
    full_similarity_rows = []
    arm_mass_rows = []
    for species in species_names:
        for arm, mass in enumerate(arm_masses[species], start=1):
            arm_mass_rows.append(
                {
                    "species": species,
                    "arm": arm,
                    "pi_mass": float(mass),
                    "occupied_for_matching": bool(mass > OCCUPIED_ARM_MASS_THRESHOLD),
                }
            )
    for species_a in species_names:
        for species_b in species_names:
            sim = cosine_similarity_matrix(arm_vectors[species_a], arm_vectors[species_b])
            row_ind, col_ind, matched = hungarian_match_arms(sim)
            mean_matrix.loc[species_a, species_b] = float(np.mean(matched))
            min_matrix.loc[species_a, species_b] = float(np.min(matched))
            for arm_a in range(sim.shape[0]):
                for arm_b in range(sim.shape[1]):
                    full_similarity_rows.append(
                        {
                            "species_a": species_a,
                            "species_b": species_b,
                            "arm_a": arm_a + 1,
                            "arm_b": arm_b + 1,
                            "cosine_similarity": float(sim[arm_a, arm_b]),
                        }
                    )
            for arm_a, arm_b, score in zip(row_ind, col_ind, matched):
                detail_rows.append(
                    {
                        "species_a": species_a,
                        "species_b": species_b,
                        "arm_a": int(arm_a + 1),
                        "matched_arm_b": int(arm_b + 1),
                        "cosine_similarity": float(score),
                    }
                )

            occupied_a = np.flatnonzero(arm_masses[species_a] > OCCUPIED_ARM_MASS_THRESHOLD)
            occupied_b = np.flatnonzero(arm_masses[species_b] > OCCUPIED_ARM_MASS_THRESHOLD)
            if occupied_a.size and occupied_b.size:
                occupied_sim = sim[np.ix_(occupied_a, occupied_b)]
                occ_row, occ_col, occ_matched = hungarian_match_arms(occupied_sim)
                occupied_mean_matrix.loc[species_a, species_b] = float(np.mean(occ_matched))
                occupied_min_matrix.loc[species_a, species_b] = float(np.min(occ_matched))
                for local_a, local_b, score in zip(occ_row, occ_col, occ_matched):
                    occupied_detail_rows.append(
                        {
                            "species_a": species_a,
                            "species_b": species_b,
                            "arm_a": int(occupied_a[local_a] + 1),
                            "matched_arm_b": int(occupied_b[local_b] + 1),
                            "cosine_similarity": float(score),
                            "arm_a_pi_mass": float(arm_masses[species_a][occupied_a[local_a]]),
                            "arm_b_pi_mass": float(arm_masses[species_b][occupied_b[local_b]]),
                        }
                    )
            else:
                occupied_mean_matrix.loc[species_a, species_b] = np.nan
                occupied_min_matrix.loc[species_a, species_b] = np.nan
    return (
        mean_matrix,
        min_matrix,
        occupied_mean_matrix,
        occupied_min_matrix,
        pd.DataFrame(detail_rows),
        pd.DataFrame(occupied_detail_rows),
        pd.DataFrame(full_similarity_rows),
        pd.DataFrame(arm_mass_rows),
    )


def plot_matching_heatmap(matrix, *, title, colorbar_label):
    fig, ax = plt.subplots(figsize=(7.8, 6.6), constrained_layout=True)
    values = matrix.to_numpy(float)
    im = ax.imshow(values, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index, fontsize=8)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if values[i, j] < 0.55 else "black")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=colorbar_label)
    return fig


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    species_names = species_names_from_manifest()

    fig, eigenspectrum = plot_eigenspectra(species_names)
    fig.savefig(PLOT_DIR / "species_transfer_operator_eigenspectra.png", dpi=260)
    fig.savefig(PLOT_DIR / "species_transfer_operator_eigenspectra.pdf")
    plt.close(fig)
    eigenspectrum.to_csv(PLOT_DIR / "species_transfer_operator_eigenspectra.csv", index=False)

    (
        mean_matrix,
        min_matrix,
        occupied_mean_matrix,
        occupied_min_matrix,
        match_detail,
        occupied_match_detail,
        full_similarity,
        arm_masses,
    ) = compute_arm_matching(species_names)
    mean_matrix.to_csv(PLOT_DIR / "species_pair_hungarian_matched_arm_mean_cosine.csv")
    min_matrix.to_csv(PLOT_DIR / "species_pair_hungarian_matched_arm_min_cosine.csv")
    occupied_mean_matrix.to_csv(
        PLOT_DIR / "species_pair_hungarian_matched_occupied_arm_mean_cosine.csv"
    )
    occupied_min_matrix.to_csv(
        PLOT_DIR / "species_pair_hungarian_matched_occupied_arm_min_cosine.csv"
    )
    match_detail.to_csv(PLOT_DIR / "species_pair_hungarian_matched_arm_details.csv", index=False)
    occupied_match_detail.to_csv(
        PLOT_DIR / "species_pair_hungarian_matched_occupied_arm_details.csv", index=False
    )
    full_similarity.to_csv(PLOT_DIR / "species_pair_all_arm_cosine_similarity.csv", index=False)
    arm_masses.to_csv(PLOT_DIR / "species_arm_pi_masses_for_matching.csv", index=False)

    fig = plot_matching_heatmap(
        mean_matrix,
        title="Hungarian-matched species-arm cosine similarity",
        colorbar_label="mean cosine across matched arms",
    )
    fig.savefig(PLOT_DIR / "species_pair_hungarian_matched_arm_mean_cosine_heatmap.png", dpi=260)
    fig.savefig(PLOT_DIR / "species_pair_hungarian_matched_arm_mean_cosine_heatmap.pdf")
    plt.close(fig)

    fig = plot_matching_heatmap(
        min_matrix,
        title="Weakest Hungarian-matched arm cosine per species pair",
        colorbar_label="minimum matched-arm cosine",
    )
    fig.savefig(PLOT_DIR / "species_pair_hungarian_matched_arm_min_cosine_heatmap.png", dpi=260)
    fig.savefig(PLOT_DIR / "species_pair_hungarian_matched_arm_min_cosine_heatmap.pdf")
    plt.close(fig)

    fig = plot_matching_heatmap(
        occupied_mean_matrix,
        title="Hungarian-matched occupied-arm cosine similarity",
        colorbar_label="mean cosine across occupied matched arms",
    )
    fig.savefig(
        PLOT_DIR / "species_pair_hungarian_matched_occupied_arm_mean_cosine_heatmap.png",
        dpi=260,
    )
    fig.savefig(PLOT_DIR / "species_pair_hungarian_matched_occupied_arm_mean_cosine_heatmap.pdf")
    plt.close(fig)

    print(f"Wrote eigenspectra and arm matching outputs to {PLOT_DIR}")
    return PLOT_DIR


if __name__ == "__main__":
    main()
