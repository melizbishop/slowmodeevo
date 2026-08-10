from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path("/Users/meganbishop/slowmodeevo")
TUNING_REPRESENTATION = "egocentered_and_normalized_distances"
SETM_ROOT = (
    REPO_ROOT
    / "outputs"
    / "single_species_distance_comparison"
    / "tuning"
    / TUNING_REPRESENTATION
)
SOURCE_OUT_DIR = SETM_ROOT / "setM_basin_syllable_matching_phylo"
OUT_DIR = SOURCE_OUT_DIR / "species_chi_trajectory_classifier_logreg_1000"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPS = 1e-12
FS = 120.0
WINDOW_SECONDS = (5, 10, 20, 40, 60)
STRIDE_FRACTION = 0.5
MAX_WINDOWS_PER_RECORDING = 80
MAX_WINDOWS_PER_SPECIES = 1000
N_SPLITS = 10
TARGET_WINDOW_SECONDS = 20
SEED = 14
REFERENCE_SPECIES = "Peromyscus_polionotus"


def setm_arm_path(species):
    species_dir = SETM_ROOT / species
    preferred = species_dir / f"{species}_arms_setM.npz"
    if preferred.exists():
        return preferred
    matches = sorted(species_dir.glob("*_arms_setM.npz"))
    if not matches:
        raise FileNotFoundError(f"No setM arm file found for {species} under {species_dir}")
    return matches[0]


def load_setm_chi(species):
    with np.load(setm_arm_path(species), allow_pickle=True) as result:
        chi = np.asarray(result["chi"], dtype=float)
    chi = np.clip(chi, 0.0, None)
    return chi / np.maximum(chi.sum(axis=1, keepdims=True), EPS)


def state_files_for_species(species):
    states_dir = SETM_ROOT / species / "states"
    return sorted(states_dir.glob("*_states.npy")) if states_dir.exists() else []


def load_reference_alignment():
    path = SOURCE_OUT_DIR / f"setM_enrichment_reference_arm_alignment_to_{REFERENCE_SPECIES}.csv"
    alignment = pd.read_csv(path)
    species_order = sorted(alignment["species"].unique())
    reference_arms = sorted(alignment["reference_arm"].unique())
    mapping = {}
    for species, sub in alignment.groupby("species"):
        by_reference = dict(zip(sub["reference_arm"].astype(int), sub["species_arm"].astype(int)))
        missing = set(reference_arms) - set(by_reference)
        if missing:
            raise ValueError(f"{species} missing reference arms: {sorted(missing)}")
        mapping[species] = by_reference
    return species_order, reference_arms, mapping


SPECIES_ORDER, REFERENCE_ARMS, SPECIES_TO_REFERENCE_ALIGNMENT = load_reference_alignment()


def reference_aligned_chi(species):
    chi = load_setm_chi(species)
    mapping = SPECIES_TO_REFERENCE_ALIGNMENT[species]
    ordered_species_arms = [mapping[int(ref_arm)] for ref_arm in REFERENCE_ARMS]
    return chi[:, ordered_species_arms]


def sample_window_starts(n_frames, window_frames, stride_frames, max_windows, rng):
    if n_frames < window_frames:
        return np.array([], dtype=int)
    starts = np.arange(0, n_frames - window_frames + 1, stride_frames, dtype=int)
    if starts.size > max_windows:
        starts = np.sort(rng.choice(starts, size=max_windows, replace=False))
    return starts


def run_lengths(labels):
    labels = np.asarray(labels)
    if labels.size == 0:
        return np.array([], dtype=float)
    changes = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.r_[0, changes]
    stops = np.r_[changes, labels.size]
    return (stops - starts).astype(float)


def chi_window_features(window_chi):
    window_chi = np.asarray(window_chi, dtype=float)
    hard = np.argmax(window_chi, axis=1)
    n_arms = window_chi.shape[1]
    entropy = -(window_chi * np.log(np.maximum(window_chi, EPS))).sum(axis=1) / np.log(max(n_arms, 2))
    max_chi = window_chi.max(axis=1)
    arm_mass = window_chi.mean(axis=0)
    arm_std = window_chi.std(axis=0)
    hard_mass = np.bincount(hard, minlength=n_arms).astype(float)
    hard_mass = hard_mass / max(hard_mass.sum(), EPS)
    if window_chi.shape[0] > 1:
        soft_velocity_by_arm = np.abs(np.diff(window_chi, axis=0)).mean(axis=0)
        soft_step = np.abs(np.diff(window_chi, axis=0)).sum(axis=1)
        hard_switch = hard[1:] != hard[:-1]
        switch_rate = float(hard_switch.mean())
        transition = np.zeros((n_arms, n_arms), dtype=float)
        np.add.at(transition, (hard[:-1], hard[1:]), 1.0)
        transition = transition / max(transition.sum(), EPS)
    else:
        soft_velocity_by_arm = np.zeros(n_arms, dtype=float)
        soft_step = np.array([0.0])
        switch_rate = 0.0
        transition = np.zeros((n_arms, n_arms), dtype=float)
    dwell = run_lengths(hard)
    dwell = dwell if dwell.size else np.array([0.0])
    scalar = np.array(
        [
            entropy.mean(),
            entropy.std(),
            max_chi.mean(),
            max_chi.std(),
            soft_step.mean(),
            soft_step.std(),
            switch_rate,
            dwell.mean(),
            np.median(dwell),
            dwell.max(),
        ],
        dtype=float,
    )
    return np.concatenate([arm_mass, arm_std, hard_mass, soft_velocity_by_arm, transition.ravel(), scalar])


def build_species_window_table():
    rng = np.random.default_rng(SEED)
    rows = []
    for species in SPECIES_ORDER:
        chi = reference_aligned_chi(species)
        for state_path in state_files_for_species(species):
            states = np.load(state_path, mmap_mode="r")
            states = np.asarray(states, dtype=int)
            valid = (states >= 0) & (states < chi.shape[0])
            valid_fraction = float(valid.mean()) if valid.size else 0.0
            states = states[valid]
            if states.size == 0:
                continue
            trajectory = chi[states]
            recording_id = f"{species}/{state_path.stem}"
            for window_seconds in WINDOW_SECONDS:
                window_frames = max(2, int(round(window_seconds * FS)))
                stride_frames = max(1, int(round(window_frames * STRIDE_FRACTION)))
                starts = sample_window_starts(
                    len(states),
                    window_frames,
                    stride_frames,
                    MAX_WINDOWS_PER_RECORDING,
                    rng,
                )
                for start in starts:
                    stop = start + window_frames
                    rows.append(
                        {
                            "window_seconds": float(window_seconds),
                            "species": species,
                            "genus": species.split("_")[0],
                            "recording_id": recording_id,
                            "state_file": str(state_path),
                            "window_start": int(start),
                            "window_stop": int(stop),
                            "valid_state_fraction": valid_fraction,
                            "n_reference_arms": len(REFERENCE_ARMS),
                            "features": chi_window_features(trajectory[start:stop]),
                        }
                    )
    return pd.DataFrame(rows)


def balance_species_windows(window_df):
    balanced = []
    for (window_seconds, species), sub in window_df.groupby(["window_seconds", "species"], sort=True):
        if len(sub) > MAX_WINDOWS_PER_SPECIES:
            sub = sub.sample(n=MAX_WINDOWS_PER_SPECIES, random_state=SEED)
        balanced.append(sub)
    return pd.concat(balanced, ignore_index=True) if balanced else window_df


def species_recording_split(meta_df, split_idx):
    rng = np.random.default_rng(SEED + 1009 * split_idx)
    train_groups = set()
    test_groups = set()
    for species, sub in meta_df[["species", "recording_id"]].drop_duplicates().groupby("species", sort=True):
        groups = rng.permutation(sub["recording_id"].to_numpy())
        if groups.size < 2:
            train_groups.update(groups)
            continue
        n_test = max(1, int(round(0.25 * groups.size)))
        test_groups.update(groups[:n_test])
        train_groups.update(groups[n_test:])
    return train_groups, test_groups


def species_feature_matrix(mode_window_df):
    return np.vstack(mode_window_df["features"].to_numpy())


def main():
    species_window_df = balance_species_windows(build_species_window_table())
    if species_window_df.empty:
        raise RuntimeError("No setM chi trajectory windows could be built for species classification.")

    species_window_summary = (
        species_window_df.groupby(["window_seconds", "species"], sort=True)
        .agg(
            n_windows=("species", "size"),
            n_recordings=("recording_id", "nunique"),
            median_valid_state_fraction=("valid_state_fraction", "median"),
        )
        .reset_index()
    )
    species_window_summary.to_csv(OUT_DIR / "setM_species_logreg_1000_window_summary.csv", index=False)

    species_accuracy_rows = []
    target_predictions = []
    for window_seconds, sub in species_window_df.groupby("window_seconds", sort=True):
        sub = sub.reset_index(drop=True)
        X = species_feature_matrix(sub)
        y = sub["species"].to_numpy()
        for split_idx in range(N_SPLITS):
            train_groups, test_groups = species_recording_split(sub, split_idx)
            train_mask = sub["recording_id"].isin(train_groups).to_numpy()
            test_mask = sub["recording_id"].isin(test_groups).to_numpy()
            if train_mask.sum() == 0 or test_mask.sum() == 0:
                continue
            if len(set(y[train_mask])) < len(SPECIES_ORDER) or len(set(y[test_mask])) < 2:
                continue
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=SEED,
                    solver="liblinear",
                ),
            )
            clf.fit(X[train_mask], y[train_mask])
            y_pred = clf.predict(X[test_mask])
            species_accuracy_rows.append(
                {
                    "window_seconds": float(window_seconds),
                    "split": split_idx,
                    "n_train_windows": int(train_mask.sum()),
                    "n_test_windows": int(test_mask.sum()),
                    "n_train_recordings": int(sub.loc[train_mask, "recording_id"].nunique()),
                    "n_test_recordings": int(sub.loc[test_mask, "recording_id"].nunique()),
                    "accuracy": float(np.mean(y_pred == y[test_mask])),
                    "balanced_accuracy": float(balanced_accuracy_score(y[test_mask], y_pred)),
                }
            )
            if float(window_seconds) == float(TARGET_WINDOW_SECONDS):
                target_predictions.append(
                    pd.DataFrame(
                        {
                            "split": split_idx,
                            "window_seconds": float(window_seconds),
                            "true_species": y[test_mask],
                            "predicted_species": y_pred,
                        }
                    )
                )

    species_accuracy_df = pd.DataFrame(species_accuracy_rows)
    if species_accuracy_df.empty:
        raise RuntimeError("No valid species classifier train/test splits were available.")
    species_accuracy_df.to_csv(OUT_DIR / "setM_species_logreg_1000_accuracy_by_split.csv", index=False)

    species_accuracy_summary = (
        species_accuracy_df.groupby("window_seconds", sort=True)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            sem_accuracy=("accuracy", "sem"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            sem_balanced_accuracy=("balanced_accuracy", "sem"),
            n_splits=("split", "nunique"),
            mean_train_windows=("n_train_windows", "mean"),
            mean_test_windows=("n_test_windows", "mean"),
            mean_train_recordings=("n_train_recordings", "mean"),
            mean_test_recordings=("n_test_recordings", "mean"),
        )
        .reset_index()
    )
    species_accuracy_summary.to_csv(OUT_DIR / "setM_species_logreg_1000_accuracy_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    summary = species_accuracy_summary.sort_values("window_seconds")
    ax.errorbar(
        summary["window_seconds"],
        summary["mean_accuracy"],
        yerr=summary["sem_accuracy"].fillna(0.0),
        marker="s",
        capsize=3,
        label="accuracy",
        alpha=0.85,
    )
    ax.errorbar(
        summary["window_seconds"],
        summary["mean_balanced_accuracy"],
        yerr=summary["sem_balanced_accuracy"].fillna(0.0),
        marker="o",
        capsize=3,
        label="balanced accuracy",
        alpha=0.85,
    )
    chance = 1.0 / max(len(SPECIES_ORDER), 1)
    ax.axhline(chance, color="black", linestyle="--", linewidth=1, label=f"chance = {chance:.3f}")
    ax.set_xticks(WINDOW_SECONDS)
    ax.set_xticklabels([str(int(x)) for x in WINDOW_SECONDS])
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("window length (seconds)")
    ax.set_ylabel("held-out recording accuracy")
    ax.set_title("Species identity from chi trajectory windows: logistic regression, 1000/species")
    ax.legend(frameon=False)
    accuracy_png = OUT_DIR / "setM_species_logreg_1000_accuracy.png"
    accuracy_pdf = OUT_DIR / "setM_species_logreg_1000_accuracy.pdf"
    fig.savefig(accuracy_png, dpi=220)
    fig.savefig(accuracy_pdf)
    plt.close(fig)

    if target_predictions:
        target_prediction_df = pd.concat(target_predictions, ignore_index=True)
        target_prediction_df.to_csv(
            OUT_DIR / f"setM_species_logreg_1000_{int(TARGET_WINDOW_SECONDS)}s_predictions.csv",
            index=False,
        )
        cm = confusion_matrix(
            target_prediction_df["true_species"],
            target_prediction_df["predicted_species"],
            labels=SPECIES_ORDER,
            normalize="true",
        )
        cm_df = pd.DataFrame(cm, index=SPECIES_ORDER, columns=SPECIES_ORDER)
        cm_df.to_csv(
            OUT_DIR / f"setM_species_logreg_1000_{int(TARGET_WINDOW_SECONDS)}s_confusion_matrix_normalized.csv"
        )
        fig, ax = plt.subplots(figsize=(7.8, 7.2), constrained_layout=True)
        ConfusionMatrixDisplay(cm, display_labels=[s.replace("_", " ") for s in SPECIES_ORDER]).plot(
            ax=ax,
            xticks_rotation=45,
            cmap="magma",
            colorbar=True,
            values_format=".2f",
        )
        ax.set_title(f"Species logistic regression at {int(TARGET_WINDOW_SECONDS)} s windows")
        cm_png = OUT_DIR / f"setM_species_logreg_1000_{int(TARGET_WINDOW_SECONDS)}s_confusion_matrix_normalized.png"
        cm_pdf = OUT_DIR / f"setM_species_logreg_1000_{int(TARGET_WINDOW_SECONDS)}s_confusion_matrix_normalized.pdf"
        fig.savefig(cm_png, dpi=220)
        fig.savefig(cm_pdf)
        plt.close(fig)

    print(species_window_summary.to_string(index=False))
    print()
    print(species_accuracy_summary.to_string(index=False))
    print()
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
