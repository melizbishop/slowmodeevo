from __future__ import annotations

import csv
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import parallel_backend
from scipy.optimize import linear_sum_assignment
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

N_CLUSTERS = 1000
N_BASINS = 4
ARM_EPS = 1e-12
TOP_SYLLABLES = 80
RANDOM_STATE = 14

RESULTS_DIR = (
    REPO_ROOT
    / "outputs/multispecies_slow_modes/global_clustering_modes/species_outputs/results"
)
THRESHOLD_DIR = RESULTS_DIR / "slow_mode_arm_enrichment_occupancy_threshold"
SPECIES_OPERATOR_DIR = THRESHOLD_DIR / "species_operators"
GLOBAL_OUTPUT_DIR = REPO_ROOT / "outputs/multispecies_slow_modes/global_clustering_modes/global_outputs"
STATE_DIR = GLOBAL_OUTPUT_DIR / "states"
PROJECTION_RESULT = GLOBAL_OUTPUT_DIR / "projection_result.pkl"
REFERENCE_CHI = RESULTS_DIR / "slow_mode_arm_enrichment/all_species_operator/chi.npy"
MOSEQ_TIMESERIES_CSV = Path("/Users/meganbishop/moseq_dataverse/multispecies_moseq_timeseries.csv")

OUT_DIR = THRESHOLD_DIR / "absorbed_cluster_syllable_classifier_diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def genus_from_species(species: str) -> str:
    return str(species).split("_")[0]


def load_projection_metadata() -> list[dict]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        with open(PROJECTION_RESULT, "rb") as handle:
            result = pickle.load(handle)
    rows = []
    for meta in result["metadata"]:
        individual_id = str(meta["individual_id"])
        state_path = STATE_DIR / f"{individual_id}_states.npy"
        if not state_path.exists():
            continue
        row = dict(meta)
        row["individual_id"] = individual_id
        row["species"] = str(meta.get("species", individual_id.split("__subject__")[0]))
        row["state_path"] = str(state_path)
        rows.append(row)
    return rows


def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    denom = np.maximum(
        np.linalg.norm(A, axis=0)[:, None] * np.linalg.norm(B, axis=0)[None, :],
        ARM_EPS,
    )
    return (A.T @ B) / denom


def align_chi_to_reference(chi: np.ndarray, reference_chi: np.ndarray) -> np.ndarray:
    sim = cosine_similarity_matrix(chi, reference_chi)
    row_ind, col_ind = linear_sum_assignment(-sim)
    permutation = np.zeros(chi.shape[1], dtype=int)
    permutation[col_ind] = row_ind
    return chi[:, permutation]


def full_aligned_chi_by_species(species_names: list[str], reference_chi: np.ndarray) -> dict[str, np.ndarray]:
    aligned = {}
    for species in species_names:
        species_dir = SPECIES_OPERATOR_DIR / species
        chi = np.asarray(np.load(species_dir / "expanded_global_chi.npy"), dtype=np.float64)
        aligned[species] = align_chi_to_reference(chi, reference_chi)
    return aligned


def representative_mask(species: str) -> np.ndarray:
    species_dir = SPECIES_OPERATOR_DIR / species
    reps = set(np.asarray(np.load(species_dir / "macro_representative_global_clusters.npy"), dtype=int).tolist())
    return np.array([cluster in reps for cluster in range(N_CLUSTERS)], dtype=bool)


def load_moseq_sequences(recording_ids: set[str]) -> dict[str, np.ndarray]:
    sequences = {}
    with open(MOSEQ_TIMESERIES_CSV, newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        recording_col = header.index("recording_id")
        time_cols = [i for i, name in enumerate(header) if name.startswith("t_")]
        for row in reader:
            recording_id = str(row[recording_col])
            if recording_id not in recording_ids:
                continue
            sequences[recording_id] = np.asarray([int(float(row[i])) for i in time_cols], dtype=np.int16)
            if len(sequences) == len(recording_ids):
                break
    missing = sorted(recording_ids - set(sequences))
    if missing:
        print(f"Warning: missing {len(missing)} recording IDs from MoSeq CSV; first few: {missing[:5]}")
    return sequences


def iter_matched_frames(meta: dict, syllables_by_recording: dict[str, np.ndarray]):
    states = np.asarray(np.load(meta["state_path"], mmap_mode="r"), dtype=np.int64)
    segment_lengths = np.asarray(meta.get("segment_lengths", []), dtype=int)
    rec_ids = [str(value) for value in meta.get("recording_ids", [])]
    if len(segment_lengths) == 0 or len(rec_ids) == 0:
        segment_lengths = np.array([len(states)], dtype=int)
        rec_ids = [meta["individual_id"]]

    segment_ends = np.cumsum(segment_lengths)
    starts = np.r_[0, segment_ends[:-1]]
    embedding_window = max(1, int(meta.get("n_frames", len(states))) - len(states) + 1)
    frame_index = np.arange(len(states), dtype=int)
    segment_index = np.searchsorted(segment_ends, frame_index, side="right")
    valid_segment = segment_index < len(segment_ends)
    crosses = np.ones(len(states), dtype=bool)
    crosses[valid_segment] = frame_index[valid_segment] + embedding_window - 1 >= segment_ends[segment_index[valid_segment]]

    for segment, recording_id in enumerate(rec_ids):
        syllable_sequence = syllables_by_recording.get(recording_id)
        if syllable_sequence is None:
            continue
        mask = valid_segment & (segment_index == segment) & ~crosses
        if not mask.any():
            continue
        local_states = states[mask]
        raw_frames = frame_index[mask] - starts[segment]
        valid = (
            (local_states >= 0)
            & (local_states < N_CLUSTERS)
            & (raw_frames >= 0)
            & (raw_frames < len(syllable_sequence))
        )
        if np.any(valid):
            yield local_states[valid].astype(np.int64), np.asarray(syllable_sequence[raw_frames[valid]], dtype=np.int64)


def choose_top_syllables(metadata: list[dict], syllables_by_recording: dict[str, np.ndarray]) -> np.ndarray:
    counts = {}
    for idx, meta in enumerate(metadata, start=1):
        for _, syllables in iter_matched_frames(meta, syllables_by_recording):
            values, freqs = np.unique(syllables, return_counts=True)
            for value, freq in zip(values, freqs):
                counts[int(value)] = counts.get(int(value), 0) + int(freq)
        if idx % 50 == 0:
            print(f"top-syllable pass: {idx}/{len(metadata)} individuals")
    table = pd.DataFrame(
        [{"moseq_cluster": key, "frames": value} for key, value in counts.items()]
    ).sort_values("frames", ascending=False)
    table.to_csv(OUT_DIR / "matched_moseq_syllable_totals.csv", index=False)
    return table.head(TOP_SYLLABLES)["moseq_cluster"].astype(int).to_numpy()


def build_syllable_features(
    metadata: list[dict],
    syllables_by_recording: dict[str, np.ndarray],
    aligned_chi: dict[str, np.ndarray],
    rep_masks: dict[str, np.ndarray],
    classifier_syllables: np.ndarray,
    *,
    drop_absorbed: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    syllable_index = {int(value): idx for idx, value in enumerate(classifier_syllables)}
    rows = []
    evidence_rows = []
    for idx, meta in enumerate(metadata, start=1):
        species = meta["species"]
        chi = aligned_chi[species]
        rep_mask = rep_masks[species]
        arm_syllable_weights = np.zeros((N_BASINS, len(classifier_syllables)), dtype=np.float64)
        arm_total_weights = np.zeros(N_BASINS, dtype=np.float64)
        absorbed_arm_weights = np.zeros(N_BASINS, dtype=np.float64)
        absorbed_frame_count = 0
        total_frame_count = 0

        for local_states, syllables in iter_matched_frames(meta, syllables_by_recording):
            keep_syllable = np.array([int(value) in syllable_index for value in syllables], dtype=bool)
            if not np.any(keep_syllable):
                continue
            local_states = local_states[keep_syllable]
            syllables = syllables[keep_syllable]
            is_representative = rep_mask[local_states]
            memberships = chi[local_states]
            total_frame_count += int(local_states.size)
            absorbed_frame_count += int((~is_representative).sum())
            absorbed_arm_weights += memberships[~is_representative].sum(axis=0) if np.any(~is_representative) else 0.0

            if drop_absorbed:
                local_states = local_states[is_representative]
                syllables = syllables[is_representative]
                memberships = memberships[is_representative]
                if local_states.size == 0:
                    continue
            syllable_cols = np.array([syllable_index[int(value)] for value in syllables], dtype=int)
            for arm in range(N_BASINS):
                weights = memberships[:, arm]
                if weights.sum() <= 0:
                    continue
                np.add.at(arm_syllable_weights[arm], syllable_cols, weights)
                arm_total_weights[arm] += float(weights.sum())

        for arm in range(N_BASINS):
            distribution = (
                arm_syllable_weights[arm] / arm_total_weights[arm]
                if arm_total_weights[arm] > 0
                else np.zeros(len(classifier_syllables), dtype=np.float64)
            )
            row = {
                "global_id": meta["individual_id"],
                "species": species,
                "genus": genus_from_species(species),
                "aligned_arm": int(arm),
                "matched_weighted_frames": float(arm_total_weights[arm]),
                "drop_absorbed": bool(drop_absorbed),
            }
            for syllable, value in zip(classifier_syllables, distribution):
                row[f"syllable_{int(syllable)}"] = float(value)
            rows.append(row)

        evidence = {
            "global_id": meta["individual_id"],
            "species": species,
            "drop_absorbed": bool(drop_absorbed),
            "top_syllable_matched_frames": int(total_frame_count),
            "absorbed_top_syllable_frames": int(absorbed_frame_count),
            "absorbed_frame_fraction": float(absorbed_frame_count / total_frame_count) if total_frame_count else np.nan,
        }
        all_arm_weight = absorbed_arm_weights + arm_total_weights if drop_absorbed else arm_total_weights
        for arm in range(N_BASINS):
            denominator = all_arm_weight[arm]
            evidence[f"absorbed_weight_arm_{arm + 1}"] = float(absorbed_arm_weights[arm])
            evidence[f"total_weight_arm_{arm + 1}"] = float(denominator)
            evidence[f"absorbed_fraction_arm_{arm + 1}"] = (
                float(absorbed_arm_weights[arm] / denominator) if denominator > 0 else np.nan
            )
        evidence_rows.append(evidence)

        if idx % 25 == 0:
            label = "representative-only" if drop_absorbed else "full"
            print(f"feature pass ({label}): {idx}/{len(metadata)} individuals")

    return pd.DataFrame(rows), pd.DataFrame(evidence_rows)


def classify_arm_features(df: pd.DataFrame, label_column: str, arm: int) -> tuple[dict, pd.DataFrame]:
    work = df.loc[(df["aligned_arm"].astype(int) == int(arm)) & (df["matched_weighted_frames"] > 0)].copy()
    feature_cols = [col for col in work.columns if col.startswith("syllable_")]
    y = work[label_column].astype(str).to_numpy()
    groups = work["global_id"].astype(str).to_numpy()
    X = work[feature_cols].to_numpy(float)
    labels = np.unique(y)
    if len(labels) < 2 or len(np.unique(groups)) < 2:
        summary = {
            "target": label_column,
            "aligned_arm": int(arm),
            "status": "skipped_insufficient_classes_or_groups",
        }
        return summary, pd.DataFrame()
    cv = LeaveOneGroupOut()
    clf = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(
            solver="liblinear",
            max_iter=500,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    )
    baseline = DummyClassifier(strategy="most_frequent")
    with parallel_backend("threading"):
        y_pred = cross_val_predict(clf, X, y, cv=cv, groups=groups, n_jobs=-1)
        y_base = cross_val_predict(baseline, X, y, cv=cv, groups=groups, n_jobs=-1)
    summary = {
        "target": label_column,
        "aligned_arm": int(arm),
        "status": "ok",
        "n_individual_arm_rows": int(len(work)),
        "n_individuals": int(len(np.unique(groups))),
        "n_classes": int(len(labels)),
        "accuracy": float(accuracy_score(y, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "baseline_accuracy": float(accuracy_score(y, y_base)),
        "baseline_balanced_accuracy": float(balanced_accuracy_score(y, y_base)),
    }
    cm = pd.DataFrame(confusion_matrix(y, y_pred, labels=labels), index=labels, columns=labels)
    return summary, cm


def plot_weight_matrix(matrix_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    values = matrix_df[[f"arm_{i}" for i in range(1, N_BASINS + 1)]].to_numpy(float)
    vmax = max(0.01, float(np.nanmax(values)))
    im = ax.imshow(values, cmap="magma", vmin=0, vmax=vmax)
    ax.set_yticks(np.arange(len(matrix_df)))
    ax.set_yticklabels(matrix_df["species"])
    ax.set_xticks(np.arange(N_BASINS))
    ax.set_xticklabels([f"arm {i}" for i in range(1, N_BASINS + 1)])
    ax.set_xlabel("Hungarian-aligned arm")
    ax.set_ylabel("species")
    ax.set_title("Syllable-classifier chi weight from absorbed clusters")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{100 * values[i, j]:.1f}%", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, label="absorbed chi weight / total chi weight")
    fig.savefig(out_path, dpi=260)
    plt.close(fig)


def plot_loo_results(summary_df: pd.DataFrame, out_path: Path) -> None:
    ok = summary_df.loc[summary_df["status"] == "ok"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True, constrained_layout=True)
    for ax, target in zip(axes, ["genus", "species"]):
        sub = ok.loc[ok["target"] == target].sort_values(["drop_absorbed", "aligned_arm"])
        arms = np.arange(N_BASINS)
        full = sub.loc[sub["drop_absorbed"] == False].set_index("aligned_arm")
        reps = sub.loc[sub["drop_absorbed"] == True].set_index("aligned_arm")
        ax.bar(arms - 0.22, full.loc[arms, "balanced_accuracy"], width=0.22, label="full")
        ax.bar(arms, reps.loc[arms, "balanced_accuracy"], width=0.22, label="representative only")
        ax.bar(arms + 0.22, full.loc[arms, "baseline_balanced_accuracy"], width=0.22, label="baseline")
        ax.set_xticks(arms)
        ax.set_xticklabels([f"arm {i + 1}" for i in arms])
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{target} LOO")
        ax.set_ylabel("balanced accuracy")
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle("Syllable classifier leave-one-individual-out CV")
    fig.savefig(out_path, dpi=260)
    plt.close(fig)


def main() -> None:
    feature_path = OUT_DIR / "syllable_classifier_features_full_vs_representative_only.csv"
    evidence_path = OUT_DIR / "syllable_classifier_absorbed_evidence_by_individual.csv"
    matrix_path = OUT_DIR / "syllable_classifier_absorbed_weight_matrix.csv"

    if feature_path.exists() and evidence_path.exists() and matrix_path.exists():
        print("Reusing cached syllable feature/evidence tables.")
        features = pd.read_csv(feature_path)
        matrix_df = pd.read_csv(matrix_path)
        plot_weight_matrix(matrix_df, OUT_DIR / "syllable_classifier_absorbed_weight_matrix.png")
    else:
        metadata = load_projection_metadata()
        species_names = sorted({meta["species"] for meta in metadata})
        recording_ids = {
            str(recording_id)
            for meta in metadata
            for recording_id in meta.get("recording_ids", [])
        }
        print(f"Loaded {len(metadata)} individuals and {len(recording_ids)} recording IDs.")
        syllables_by_recording = load_moseq_sequences(recording_ids)
        print(f"Loaded {len(syllables_by_recording)} MoSeq recording sequences.")

        reference_chi = np.asarray(np.load(REFERENCE_CHI), dtype=np.float64)
        aligned_chi = full_aligned_chi_by_species(species_names, reference_chi)
        rep_masks = {species: representative_mask(species) for species in species_names}

        classifier_syllables = choose_top_syllables(metadata, syllables_by_recording)
        pd.DataFrame({"moseq_cluster": classifier_syllables}).to_csv(
            OUT_DIR / "classifier_top_80_moseq_syllables.csv",
            index=False,
        )

        full_features, full_evidence = build_syllable_features(
            metadata,
            syllables_by_recording,
            aligned_chi,
            rep_masks,
            classifier_syllables,
            drop_absorbed=False,
        )
        rep_features, rep_evidence = build_syllable_features(
            metadata,
            syllables_by_recording,
            aligned_chi,
            rep_masks,
            classifier_syllables,
            drop_absorbed=True,
        )

        features = pd.concat([full_features, rep_features], ignore_index=True)
        evidence = pd.concat([full_evidence, rep_evidence], ignore_index=True)
        features.to_csv(feature_path, index=False)
        evidence.to_csv(evidence_path, index=False)

        matrix_rows = []
        for species, sub in full_evidence.groupby("species", sort=True):
            row = {"species": species}
            for arm in range(1, N_BASINS + 1):
                absorbed = sub[f"absorbed_weight_arm_{arm}"].sum()
                total = sub[f"total_weight_arm_{arm}"].sum()
                row[f"arm_{arm}"] = float(absorbed / total) if total > 0 else np.nan
            row["absorbed_frame_fraction"] = float(
                sub["absorbed_top_syllable_frames"].sum() / max(sub["top_syllable_matched_frames"].sum(), 1)
            )
            matrix_rows.append(row)
        matrix_df = pd.DataFrame(matrix_rows)
        matrix_df.to_csv(matrix_path, index=False)
        plot_weight_matrix(matrix_df, OUT_DIR / "syllable_classifier_absorbed_weight_matrix.png")

    summaries = []
    for drop_absorbed, label in [(False, "full"), (True, "representative_only")]:
        df = features.loc[features["drop_absorbed"] == drop_absorbed].copy()
        for target in ["genus", "species"]:
            for arm in range(N_BASINS):
                summary, cm = classify_arm_features(df, target, arm)
                summary["feature_set"] = label
                summary["drop_absorbed"] = bool(drop_absorbed)
                summaries.append(summary)
                if not cm.empty:
                    cm.to_csv(OUT_DIR / f"{label}_{target}_arm_{arm + 1}_loo_confusion_matrix.csv")
                print(f"finished {label} {target} arm {arm + 1}: {summary.get('balanced_accuracy', np.nan):.3f}")
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUT_DIR / "syllable_classifier_leave_one_individual_out_summary.csv", index=False)
    plot_loo_results(summary_df, OUT_DIR / "syllable_classifier_leave_one_individual_out_balanced_accuracy.png")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
