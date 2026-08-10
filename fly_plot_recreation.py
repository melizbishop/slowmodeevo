"""Fly-analysis plot recreations for outputs from pooled_user_pipeline.

The functions mirror the fly-only analyses in bermanlabemory/slowmode while
accepting this repository's per-individual projection and state files.
"""

from pathlib import Path
import csv
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.special import gamma as gamma_fn
from scipy import stats

import gpcca_utils as gu
import pipeline as pp
import pooled_user_pipeline as pup


ARM_PALETTE = ["#D55E00", "#CC79A7", "#0072B2", "#009E73"]


def cluster_total_time_seconds(run):
    """Return pooled empirical occupancy time for every cluster."""
    counts = np.zeros(run["n_clusters"], dtype=np.int64)
    for states in run["states"]:
        states = np.asarray(states, dtype=int)
        valid = (states >= 0) & (states < run["n_clusters"])
        np.add.at(counts, states[valid], 1)
    return counts / float(run["fs"])


def cluster_time_mask(run, min_cluster_time_seconds=0.0):
    """Select clusters meeting a pooled cumulative-time threshold."""
    threshold = max(0.0, float(min_cluster_time_seconds))
    return cluster_total_time_seconds(run) >= threshold


def cluster_fraction_threshold(run, drop_fraction):
    """Return a time cutoff that drops approximately the lowest fraction."""
    fraction = float(drop_fraction)
    if not 0 <= fraction < 1:
        raise ValueError("drop_fraction must be in [0, 1)")
    times = np.sort(cluster_total_time_seconds(run))
    n_drop = int(np.floor(fraction * len(times)))
    if n_drop == 0:
        return 0.0
    return float((times[n_drop - 1] + times[n_drop]) / 2)


def export_cluster_time_filter_report(run, output_path, min_cluster_time_seconds):
    """Export per-cluster keep/drop decisions and return summary counts."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    times = cluster_total_time_seconds(run)
    keep = times >= max(0.0, float(min_cluster_time_seconds))
    with open(output_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "cluster_id",
            "pooled_time_seconds",
            "minimum_time_seconds",
            "kept",
        ])
        for cluster_id, total_time, is_kept in zip(
            np.arange(len(times)), times, keep
        ):
            writer.writerow([
                int(cluster_id),
                float(total_time),
                float(min_cluster_time_seconds),
                bool(is_kept),
            ])
    return {
        "path": str(output_path),
        "total_clusters": int(len(keep)),
        "kept_clusters": int(keep.sum()),
        "dropped_clusters": int((~keep).sum()),
    }


def plot_cluster_time_distribution(run, min_cluster_time_seconds=0.0):
    """Plot pooled cluster occupancy times and the active display threshold."""
    times = cluster_total_time_seconds(run)
    positive = times[times > 0]
    threshold = max(0.0, float(min_cluster_time_seconds))
    keep = times >= threshold
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    if len(positive):
        low = max(positive.min(), np.finfo(float).tiny)
        bins = np.logspace(np.log10(low), np.log10(positive.max()), 35)
        axes[0].hist(positive, bins=bins, color="#4C78A8", edgecolor="white")
        axes[0].set_xscale("log")
    if threshold > 0:
        axes[0].axvline(threshold, color="red", ls="--", label="threshold")
        axes[0].legend()
    axes[0].set(
        title="Cluster occupancy-time distribution",
        xlabel="pooled time per cluster (s)",
        ylabel="number of clusters",
    )

    ranked = np.sort(times)[::-1]
    axes[1].plot(np.arange(1, len(ranked) + 1), ranked, "o-", ms=3)
    axes[1].set_yscale("log")
    if threshold > 0:
        axes[1].axhline(threshold, color="red", ls="--")
    axes[1].set(
        title=f"{keep.sum()} kept, {(~keep).sum()} dropped",
        xlabel="cluster rank",
        ylabel="pooled time (s)",
    )
    return fig


def _load_moseq_sequences(path, recording_ids):
    """Stream only requested recording rows from the wide MoSeq CSV."""
    wanted = {str(value) for value in recording_ids}
    sequences = {}
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        time_columns = [i for i, name in enumerate(header) if name.startswith("t_")]
        for row in reader:
            recording_id = str(row[0])
            if recording_id in wanted:
                sequences[recording_id] = np.asarray(
                    [int(float(row[i])) for i in time_columns], dtype=np.int16
                )
                if len(sequences) == len(wanted):
                    break
    return sequences


def moseq_arm_dominance(run, moseq_timeseries_csv):
    """Join per-frame MoSeq clusters to soft arm memberships."""
    metadata = {
        str(item["individual_id"]): item
        for item in run["projection"]["metadata"]
    }
    recording_ids = {
        str(recording_id)
        for item in metadata.values()
        for recording_id in item.get("recording_ids", [])
    }
    sequences = _load_moseq_sequences(moseq_timeseries_csv, recording_ids)
    weighted = {}
    matched_frames = 0
    excluded_boundary_frames = 0
    missing_recordings = set()
    individual_counts = {}

    for individual_id, states in zip(run["individual_ids"], run["states"]):
        item = metadata[str(individual_id)]
        segment_lengths = np.asarray(item.get("segment_lengths", []), dtype=int)
        rec_ids = [str(value) for value in item.get("recording_ids", [])]
        if not len(segment_lengths):
            segment_lengths = np.array([len(states)], dtype=int)
            rec_ids = [str(individual_id)]
        segment_ends = np.cumsum(segment_lengths)
        starts = np.r_[0, segment_ends[:-1]]
        embedding_window = max(
            1, int(item.get("n_frames", len(states))) - len(states) + 1
        )
        frame_index = np.arange(len(states), dtype=int)
        segment_index = np.searchsorted(segment_ends, frame_index, side="right")
        valid_segment = segment_index < len(segment_ends)
        crosses = np.ones(len(states), dtype=bool)
        crosses[valid_segment] = (
            frame_index[valid_segment] + embedding_window - 1
            >= segment_ends[segment_index[valid_segment]]
        )
        excluded_boundary_frames += int(crosses.sum())

        for segment, recording_id in enumerate(rec_ids):
            syllable_sequence = sequences.get(recording_id)
            if syllable_sequence is None:
                missing_recordings.add(recording_id)
                continue
            mask = valid_segment & (segment_index == segment) & ~crosses
            local_states = np.asarray(states, dtype=int)[mask]
            raw_frames = frame_index[mask] - starts[segment]
            valid = (
                (local_states >= 0)
                & (local_states < run["chi"].shape[0])
                & (raw_frames >= 0)
                & (raw_frames < len(syllable_sequence))
            )
            if not np.any(valid):
                continue
            local_states = local_states[valid]
            syllables = syllable_sequence[raw_frames[valid]]
            memberships = run["chi"][local_states]
            matched_frames += int(len(syllables))
            unique, inverse = np.unique(syllables, return_inverse=True)
            frame_counts = np.bincount(inverse, minlength=len(unique))
            for syllable, count in zip(unique, frame_counts):
                key = (str(individual_id), int(syllable))
                individual_counts[key] = individual_counts.get(key, 0) + int(count)
            for arm in range(run["n_basins"]):
                totals = np.bincount(
                    inverse, weights=memberships[:, arm], minlength=len(unique)
                )
                for syllable, value in zip(unique, totals):
                    key = (int(syllable), int(arm))
                    weighted[key] = weighted.get(key, 0.0) + float(value)

    rows = [
        {"moseq_cluster": syllable, "arm": arm, "weighted_frames": value}
        for (syllable, arm), value in weighted.items()
    ]
    long = pd.DataFrame(rows)
    if long.empty:
        raise ValueError("No MoSeq frames matched the slow-mode state sequences")
    matrix = long.pivot_table(
        index="moseq_cluster", columns="arm", values="weighted_frames",
        aggfunc="sum", fill_value=0.0,
    ).reindex(columns=range(run["n_basins"]), fill_value=0.0)
    totals = matrix.sum(axis=1)
    fractions = matrix.div(totals.replace(0, np.nan), axis=0).fillna(0.0)
    dominant_arm = fractions.to_numpy().argmax(axis=1)
    summary = pd.DataFrame({
        "moseq_cluster": matrix.index.astype(int),
        "dominant_arm": dominant_arm.astype(int),
        "dominant_arm_label": [f"arm {value + 1}" for value in dominant_arm],
        "dominance_fraction": fractions.to_numpy().max(axis=1),
        "matched_frames": totals.to_numpy(),
    })
    for arm in range(run["n_basins"]):
        summary[f"arm_{arm + 1}_weighted_frames"] = matrix[arm].to_numpy()
        summary[f"arm_{arm + 1}_fraction"] = fractions[arm].to_numpy()
    summary = summary.sort_values(
        ["dominant_arm", "dominance_fraction", "matched_frames"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    all_syllables = sorted(summary["moseq_cluster"].astype(int).unique())
    full_index = pd.MultiIndex.from_product(
        [[str(value) for value in run["individual_ids"]], all_syllables],
        names=["individual_id", "moseq_cluster"],
    )
    individual_time = pd.Series(
        individual_counts, name="matched_frames", dtype=float
    ).reindex(full_index, fill_value=0).reset_index()
    individual_time["matched_frames"] = individual_time[
        "matched_frames"
    ].astype(int)
    individual_time["seconds"] = (
        individual_time["matched_frames"] / float(run["fs"])
    )
    totals_by_individual = individual_time.groupby(
        "individual_id"
    )["seconds"].transform("sum")
    individual_time["fraction_of_individual_time"] = np.divide(
        individual_time["seconds"],
        totals_by_individual,
        out=np.zeros(len(individual_time), dtype=float),
        where=totals_by_individual.to_numpy() > 0,
    )
    stats_out = {
        "requested_recordings": int(len(recording_ids)),
        "matched_recordings": int(len(sequences)),
        "missing_recordings": sorted(missing_recordings),
        "matched_frames": int(matched_frames),
        "excluded_boundary_frames": int(excluded_boundary_frames),
    }
    return summary, individual_time, stats_out


def add_moseq_labels(frame, labels_csv):
    """Attach canonical MoSeq names/categories to a cluster-indexed table."""
    labels = pd.read_csv(labels_csv)
    id_column = next(
        (
            name for name in [
                "number_id", "moseq_syllable", "moseq_cluster"
            ] if name in labels
        ),
        None,
    )
    if id_column is None:
        raise ValueError("MoSeq label table has no syllable/cluster ID column")
    keep = [id_column]
    for name in ["name", "category", "moseq_syllable_name", "moseq_category"]:
        if name in labels:
            keep.append(name)
    labels = labels[keep].rename(columns={
        id_column: "moseq_cluster",
        "name": "moseq_syllable_name",
        "category": "moseq_category",
    })
    labels["moseq_cluster"] = pd.to_numeric(
        labels["moseq_cluster"], errors="coerce"
    )
    labels = labels.dropna(subset=["moseq_cluster"])
    labels["moseq_cluster"] = labels["moseq_cluster"].astype(int)
    labels = labels.drop_duplicates("moseq_cluster")
    result = frame.merge(labels, on="moseq_cluster", how="left")
    if "moseq_syllable_name" not in result:
        result["moseq_syllable_name"] = np.nan
    result["moseq_syllable_name"] = result["moseq_syllable_name"].fillna(
        "unlabeled"
    )
    result["moseq_label"] = (
        result["moseq_cluster"].astype(int).astype(str)
        + ": "
        + result["moseq_syllable_name"].astype(str)
    )
    return result


def exclude_moseq_categories(frame, categories=("noise",)):
    """Remove labeled MoSeq categories and renormalize individual fractions."""
    if "moseq_category" not in frame:
        raise ValueError("Attach MoSeq labels before filtering categories")
    excluded = {str(value).strip().lower() for value in categories}
    category = frame["moseq_category"].fillna("unlabeled").str.lower()
    result = frame.loc[~category.isin(excluded)].copy()
    if {"individual_id", "seconds"}.issubset(result.columns):
        totals = result.groupby("individual_id")["seconds"].transform("sum")
        result["fraction_of_individual_time"] = np.divide(
            result["seconds"],
            totals,
            out=np.zeros(len(result), dtype=float),
            where=totals.to_numpy() > 0,
        )
    return result


def plot_moseq_arm_dominance(summary, *, min_matched_frames=100, max_clusters=80):
    """Plot row-normalized arm weights for dominantly assigned MoSeq clusters."""
    summary = summary[summary["matched_frames"] >= min_matched_frames].copy()
    if max_clusters is not None:
        summary = summary.head(int(max_clusters))
    if summary.empty:
        raise ValueError("No MoSeq clusters pass min_matched_frames")
    fraction_columns = [
        column for column in summary.columns
        if column.startswith("arm_") and column.endswith("_fraction")
    ]
    values = summary[fraction_columns].to_numpy(dtype=float)
    fig, ax = plt.subplots(
        figsize=(7.5, max(4.5, 0.22 * len(summary))),
        constrained_layout=True,
    )
    image = ax.imshow(values, aspect="auto", vmin=0, vmax=1, cmap="magma")
    ax.set_xticks(np.arange(len(fraction_columns)))
    ax.set_xticklabels([f"arm {i + 1}" for i in range(len(fraction_columns))])
    ax.set_yticks(np.arange(len(summary)))
    labels = (
        summary["moseq_label"]
        if "moseq_label" in summary
        else summary["moseq_cluster"].astype(str)
    )
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("slow-mode arm")
    ax.set_ylabel("MoSeq cluster")
    ax.set_title("Dominant slow-mode arm for each MoSeq cluster")
    fig.colorbar(image, ax=ax, label="fraction of arm membership")
    return fig


def dominant_moseq_clusters_by_arm(
    summary,
    *,
    top_n=10,
    min_matched_frames=0,
    min_dominance_fraction=0.0,
):
    """Rank MoSeq clusters whose strongest membership is each slow-mode arm."""
    summary = summary.copy()
    required = {"moseq_cluster", "dominant_arm", "dominance_fraction",
                "matched_frames"}
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"MoSeq summary is missing columns: {sorted(missing)}")

    rows = []
    n_arms = int(summary["dominant_arm"].max()) + 1
    for arm in range(n_arms):
        arm_rows = summary.loc[
            (summary["dominant_arm"] == arm)
            & (summary["matched_frames"] >= int(min_matched_frames))
            & (summary["dominance_fraction"] >= float(min_dominance_fraction))
        ].copy()
        arm_rows = arm_rows.sort_values(
            ["dominance_fraction", "matched_frames"],
            ascending=[False, False],
        )
        if top_n is not None:
            arm_rows = arm_rows.head(int(top_n))
        if arm_rows.empty:
            continue
        arm_rows.insert(0, "arm_rank", np.arange(1, len(arm_rows) + 1))
        arm_rows.insert(0, "nested_arm_label", f"nested arm {arm + 1}")
        rows.append(arm_rows)
    if not rows:
        return summary.iloc[0:0].copy()
    return pd.concat(rows, ignore_index=True)


def plot_dominant_moseq_by_arm(
    ranked,
    *,
    value_column="dominance_fraction",
    label_column="moseq_label",
):
    """Plot top dominant MoSeq clusters separately for each nested arm."""
    if ranked.empty:
        raise ValueError("No dominant MoSeq clusters are available to plot")
    ranked = ranked.copy()
    if label_column not in ranked:
        label_column = "moseq_cluster"
    arms = list(ranked["dominant_arm"].drop_duplicates())
    ncols = min(3, len(arms))
    nrows = int(np.ceil(len(arms) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.4 * ncols, max(3.2, 2.8 * nrows)),
        constrained_layout=True,
        squeeze=False,
    )
    for panel, arm in enumerate(arms):
        ax = axes.flat[panel]
        arm_rows = ranked.loc[ranked["dominant_arm"] == arm].copy()
        arm_rows = arm_rows.sort_values("arm_rank", ascending=False)
        values = arm_rows[value_column].to_numpy(dtype=float)
        labels = arm_rows[label_column].astype(str).to_numpy()
        y = np.arange(len(arm_rows))
        ax.barh(y, values, color=ARM_PALETTE[int(arm) % len(ARM_PALETTE)])
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlim(0, max(1.0, float(np.nanmax(values)) if len(values) else 1.0))
        ax.set_xlabel(value_column.replace("_", " "))
        ax.set_title(f"nested arm {int(arm) + 1}")
    for ax in axes.flat[len(arms):]:
        ax.remove()
    return fig


def moseq_density_by_arm(summary):
    """Return per-arm MoSeq density from arm-weighted frame totals."""
    weighted_columns = [
        column for column in summary.columns
        if column.startswith("arm_") and column.endswith("_weighted_frames")
    ]
    if not weighted_columns:
        raise ValueError("MoSeq summary has no arm weighted-frame columns")
    rows = []
    label_map = {}
    if "moseq_label" in summary:
        label_map = dict(zip(summary["moseq_cluster"], summary["moseq_label"]))
    for arm_index, column in enumerate(weighted_columns):
        weights = summary[column].to_numpy(dtype=float)
        total = float(np.sum(weights))
        density = np.divide(
            weights,
            total,
            out=np.zeros_like(weights, dtype=float),
            where=total > 0,
        )
        for cluster, value, raw in zip(
            summary["moseq_cluster"].astype(int), density, weights,
        ):
            rows.append({
                "arm": int(arm_index),
                "arm_label": f"arm {arm_index + 1}",
                "moseq_cluster": int(cluster),
                "moseq_label": label_map.get(cluster, str(int(cluster))),
                "weighted_frames": float(raw),
                "density": float(value),
            })
    return pd.DataFrame(rows)


def plot_moseq_density_by_arm(
    density,
    *,
    top_n=None,
    label_column="moseq_label",
):
    """Plot the distribution over MoSeq syllables within each subarm."""
    if density.empty:
        raise ValueError("No MoSeq density rows are available to plot")
    density = density.copy()
    if label_column not in density:
        label_column = "moseq_cluster"
    arms = list(density["arm"].drop_duplicates())
    ncols = min(3, len(arms))
    nrows = int(np.ceil(len(arms) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.8 * ncols, max(3.4, 3.0 * nrows)),
        constrained_layout=True,
        squeeze=False,
    )
    for panel, arm in enumerate(arms):
        ax = axes.flat[panel]
        arm_rows = density.loc[density["arm"] == arm].copy()
        arm_rows = arm_rows.sort_values("density", ascending=False)
        if top_n is not None:
            arm_rows = arm_rows.head(int(top_n))
        arm_rows = arm_rows.sort_values("density", ascending=True)
        y = np.arange(len(arm_rows))
        ax.barh(
            y, arm_rows["density"], color=ARM_PALETTE[int(arm) % len(ARM_PALETTE)]
        )
        ax.set_yticks(y)
        ax.set_yticklabels(arm_rows[label_column].astype(str), fontsize=7)
        ax.set_xlim(0, max(0.01, float(arm_rows["density"].max()) * 1.08))
        ax.set_xlabel("within-arm MoSeq density")
        ax.set_title(f"arm {int(arm) + 1}")
    for ax in axes.flat[len(arms):]:
        ax.remove()
    return fig


def plot_individual_moseq_cluster_time(individual_time, *, max_clusters=60):
    """Plot seconds each individual spends in each MoSeq cluster."""
    pooled = (
        individual_time.groupby("moseq_cluster")["seconds"].sum()
        .sort_values(ascending=False)
    )
    if max_clusters is not None:
        pooled = pooled.head(int(max_clusters))
    selected = individual_time[
        individual_time["moseq_cluster"].isin(pooled.index)
    ]
    matrix = selected.pivot(
        index="individual_id", columns="moseq_cluster", values="seconds"
    ).reindex(columns=pooled.index, fill_value=0.0)
    fig, ax = plt.subplots(
        figsize=(max(9, 0.25 * matrix.shape[1]), max(4, 0.35 * matrix.shape[0])),
        constrained_layout=True,
    )
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(matrix.shape[1]))
    if "moseq_label" in selected:
        label_map = (
            selected[["moseq_cluster", "moseq_label"]]
            .drop_duplicates("moseq_cluster")
            .set_index("moseq_cluster")["moseq_label"]
        )
        x_labels = [label_map.get(value, str(value)) for value in matrix.columns]
    else:
        x_labels = matrix.columns.astype(str)
    ax.set_xticklabels(x_labels, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index.astype(str), fontsize=8)
    ax.set_xlabel("MoSeq cluster")
    ax.set_ylabel("individual")
    ax.set_title("Time spent in each MoSeq cluster")
    fig.colorbar(image, ax=ax, label="seconds")
    return fig


def load_run(
    root,
    *,
    fs,
    tau_seconds=3.0,
    n_basins=4,
    species=None,
    representation_name=None,
):
    """Load projections/states and reconstruct the pooled slow-mode result."""
    root = pup.resolve_pooled_run_root(
        root,
        species=species,
        representation_name=representation_name,
    )
    projection = pup.load_projection_result(root / "projection_result.pkl")
    ids = [str(value) for value in projection["individual_ids"]]
    states, state_files = pup.load_ordered_states_for_projection(
        projection,
        root=root,
        mmap_mode="r",
    )
    n_clusters = max(int(np.max(state)) for state in states) + 1
    lag = max(1, int(round(float(tau_seconds) * float(fs))))
    transfer = pp.make_transition_matrix(states, lag=lag, n_states=n_clusters)
    pi = pp.stationary_distribution(transfer)
    evals, evecs = pp.leading_eigvecs(transfer, k=min(10, n_clusters - 1))
    gpcca = gu.run_gpcca(transfer, M=n_basins, eta=pi)
    # Geometry uses non-stationary modes phi_2, ..., phi_M.
    # Figure 4 uses phi_2, phi_3, and phi_4 even when GPCCA uses M=3.
    phi = np.asarray(evecs[:, 1:4].real)
    geometry = gu.compute_hub_arms(phi, pi, gpcca["chi"])
    return {
        "root": root,
        "fs": float(fs),
        "tau_seconds": float(tau_seconds),
        "lag": lag,
        "n_basins": int(n_basins),
        "n_clusters": n_clusters,
        "projection": projection,
        "proj_files": [Path(path) for path in projection["proj_files"]],
        "individual_ids": ids,
        "state_files": state_files,
        "states": states,
        "T": transfer,
        "pi": pi,
        "evals": evals,
        "evecs": evecs,
        "phi": phi,
        "chi": gpcca["chi"],
        "assignments": gpcca["assignments"],
        "gpcca": gpcca,
        "geometry": geometry,
    }


def refit_after_dropping_clusters(run, min_cluster_time_seconds):
    """Refit spectral/GPCCA quantities on clusters above an occupancy cutoff."""
    keep = cluster_time_mask(run, min_cluster_time_seconds)
    retained = np.flatnonzero(keep)
    if len(retained) < run["n_basins"]:
        raise ValueError(
            f"Only {len(retained)} clusters remain for "
            f"{run['n_basins']} requested basins"
        )

    transfer = np.asarray(run["T"], dtype=float)[np.ix_(keep, keep)].copy()
    row_sums = transfer.sum(axis=1)
    empty = np.flatnonzero(row_sums <= 0)
    transfer[empty, empty] = 1.0
    transfer /= transfer.sum(axis=1, keepdims=True)

    pi = pp.stationary_distribution(transfer)
    evals, evecs = pp.leading_eigvecs(
        transfer, k=min(10, transfer.shape[0] - 1)
    )
    gpcca = gu.run_gpcca(transfer, M=run["n_basins"], eta=pi)
    # Keep three non-stationary coordinates for the Figure 4 projections.
    phi = np.asarray(evecs[:, 1:4].real)
    geometry = gu.compute_hub_arms(phi, pi, gpcca["chi"])

    old_to_new = np.full(run["n_clusters"], -1, dtype=int)
    old_to_new[retained] = np.arange(len(retained))
    mapped_states = [
        old_to_new[np.asarray(states, dtype=int)] for states in run["states"]
    ]
    return {
        **run,
        "n_clusters": int(len(retained)),
        "T": transfer,
        "pi": pi,
        "evals": evals,
        "evecs": evecs,
        "phi": phi,
        "chi": gpcca["chi"],
        "assignments": gpcca["assignments"],
        "gpcca": gpcca,
        "geometry": geometry,
        "states": mapped_states,
        "retained_original_cluster_ids": retained,
        "dropped_original_cluster_ids": np.flatnonzero(~keep),
        "cluster_filter_threshold_seconds": float(min_cluster_time_seconds),
    }


def conditional_nested_gpcca(
    run,
    *,
    parent_basin=None,
    n_nested_basins=3,
    membership_threshold=None,
    min_within_probability=1e-12,
):
    """Run a second GPCCA pass conditioned on one global basin.

    The global model is not changed.  This constructs a transition operator
    among only the selected parent-basin clusters by conditioning each row on
    transitions that remain inside that parent basin.
    """
    assignments = np.asarray(run["assignments"], dtype=int)
    chi = np.asarray(run["chi"], dtype=float)
    pi = np.asarray(run["pi"], dtype=float)
    T = np.asarray(run["T"], dtype=float)

    if parent_basin is None:
        n_global = int(run["n_basins"])
        basin_mass = np.bincount(assignments, weights=pi, minlength=n_global)
        parent_basin = int(np.argmax(basin_mass))
    parent_basin = int(parent_basin)

    if membership_threshold is None:
        parent_mask = assignments == parent_basin
    else:
        parent_mask = chi[:, parent_basin] >= float(membership_threshold)
    parent_clusters = np.flatnonzero(parent_mask)
    if parent_clusters.size < int(n_nested_basins):
        raise ValueError(
            f"Parent basin {parent_basin} contains {parent_clusters.size} "
            f"clusters, fewer than n_nested_basins={int(n_nested_basins)}"
        )

    T_sub = T[np.ix_(parent_clusters, parent_clusters)].copy()
    within_probability = T_sub.sum(axis=1)
    pi_sub = pi[parent_clusters].astype(float, copy=True)
    pi_sub_sum = pi_sub.sum()
    if pi_sub_sum <= 0:
        pi_sub = np.full(parent_clusters.size, 1.0 / parent_clusters.size)
    else:
        pi_sub /= pi_sub_sum

    T_cond = np.zeros_like(T_sub, dtype=float)
    valid_rows = within_probability > float(min_within_probability)
    T_cond[valid_rows] = T_sub[valid_rows] / within_probability[valid_rows, None]
    if np.any(~valid_rows):
        T_cond[~valid_rows] = pi_sub

    pi_cond = pp.stationary_distribution(T_cond)
    evals, evecs = pp.leading_eigvecs(
        T_cond, k=min(10, max(1, parent_clusters.size - 1)),
    )
    gpcca = gu.run_gpcca(T_cond, M=int(n_nested_basins), eta=pi_cond)
    phi = np.asarray(evecs[:, 1:4].real)
    if phi.shape[1] < 3:
        phi = np.pad(phi, ((0, 0), (0, 3 - phi.shape[1])))
    geometry = gu.compute_hub_arms(phi, pi_cond, gpcca["chi"])

    nested_assignments_full = np.full(run["n_clusters"], -1, dtype=int)
    nested_assignments_full[parent_clusters] = gpcca["assignments"]
    nested_chi_full = np.full(
        (run["n_clusters"], int(n_nested_basins)), np.nan, dtype=float,
    )
    nested_chi_full[parent_clusters] = gpcca["chi"]

    original_ids = run.get("retained_original_cluster_ids")
    if original_ids is None:
        original_ids = np.arange(run["n_clusters"], dtype=int)
    original_ids = np.asarray(original_ids, dtype=int)

    cluster_table = pd.DataFrame({
        "cluster_id": parent_clusters.astype(int),
        "original_cluster_id": original_ids[parent_clusters].astype(int),
        "parent_basin": parent_basin,
        "global_pi": pi[parent_clusters],
        "global_chi_parent": chi[parent_clusters, parent_basin],
        "within_parent_transition_probability": within_probability,
        "conditional_pi": pi_cond,
        "nested_arm": np.asarray(gpcca["assignments"], dtype=int),
        "nested_confidence": np.asarray(gpcca["chi"]).max(axis=1),
    })
    for arm in range(int(n_nested_basins)):
        cluster_table[f"nested_chi_{arm}"] = gpcca["chi"][:, arm]

    return {
        "parent_basin": parent_basin,
        "n_nested_basins": int(n_nested_basins),
        "membership_threshold": membership_threshold,
        "parent_clusters": parent_clusters,
        "T": T_cond,
        "pi": pi_cond,
        "evals": evals,
        "evecs": evecs,
        "phi": phi,
        "gpcca": gpcca,
        "chi": gpcca["chi"],
        "assignments": gpcca["assignments"],
        "geometry": geometry,
        "nested_assignments_full": nested_assignments_full,
        "nested_chi_full": nested_chi_full,
        "cluster_table": cluster_table,
        "parent_global_pi_mass": float(pi[parent_clusters].sum()),
        "mean_within_parent_transition_probability": float(
            np.mean(within_probability)
        ),
        "min_within_parent_transition_probability": float(
            np.min(within_probability)
        ),
    }


def build_recursive_conditional_arm_tree(
    run,
    *,
    min_parent_clusters_to_split=20,
    min_child_balance=0.05,
    min_parent_operator_gap_ratio=1.15,
    min_child_global_pi=0.0,
    max_depth=3,
    nested_basins=2,
    membership_threshold=None,
):
    """Recursively split arms with balanced children and a separated slow mode.

    This is intentionally additive: it returns a tree and does not modify the
    supplied run.  The terminal leaves can be converted into a derived run with
    ``apply_recursive_arm_tree``.
    """
    assignments = np.asarray(run["assignments"], dtype=int)
    pi = np.asarray(run["pi"], dtype=float)
    n_global = int(run["n_basins"])
    nodes = []
    leaves = []
    next_id = 0

    def add_node(parent_id, depth, label, clusters, path):
        nonlocal next_id
        node_id = next_id
        next_id += 1
        clusters = np.asarray(clusters, dtype=int)
        node = {
            "node_id": node_id,
            "parent_id": parent_id,
            "depth": int(depth),
            "label": str(label),
            "path": tuple(path),
            "clusters": clusters,
            "n_clusters": int(len(clusters)),
            "global_pi_mass": float(pi[clusters].sum()) if len(clusters) else 0.0,
            "split": False,
            "split_reason": "not_evaluated",
            "children": [],
        }
        nodes.append(node)
        return node

    def split_node(node):
        if node["n_clusters"] == 0:
            node["split_reason"] = "empty_arm"
            return
        if node["depth"] >= int(max_depth):
            node["split_reason"] = "max_depth"
            leaves.append(node)
            return
        if node["n_clusters"] < int(min_parent_clusters_to_split):
            node["split_reason"] = "parent_cluster_count_below_threshold"
            leaves.append(node)
            return

        try:
            sub = _conditional_gpcca_for_clusters(
                run,
                node["clusters"],
                n_nested_basins=int(nested_basins),
            )
        except ValueError as exc:
            node["split_reason"] = f"segment_operator_unavailable: {exc}"
            leaves.append(node)
            return
        node["conditional"] = sub
        abs_evals = np.abs(np.asarray(sub["evals"]))
        nested_lambda2 = float(abs_evals[1]) if len(abs_evals) > 1 else 0.0
        nested_gap_ratio = (
            float(abs_evals[1] / max(abs_evals[2], 1e-12))
            if len(abs_evals) > 2 else np.inf
        )
        nested_crispness = float(sub["gpcca"].get("crispness", 0.0))
        node["candidate_lambda2"] = nested_lambda2
        node["candidate_gap_ratio"] = nested_gap_ratio
        node["candidate_crispness"] = nested_crispness
        child_assignments = np.asarray(sub["assignments"], dtype=int)
        child_counts = np.bincount(
            child_assignments, minlength=int(nested_basins),
        )
        child_pi = np.asarray(sub["gpcca"]["pi_basin"], dtype=float)
        child_global_pi = np.array([
            float(pi[node["clusters"][child_assignments == child_arm]].sum())
            for child_arm in range(int(nested_basins))
        ])
        child_balance = (
            float(np.min(child_pi) / max(np.sum(child_pi), 1e-12))
            if child_pi.size else 0.0
        )
        node["candidate_child_counts"] = child_counts.tolist()
        node["candidate_child_conditional_pi"] = child_pi.tolist()
        node["candidate_child_global_pi"] = child_global_pi.tolist()
        node["candidate_child_balance"] = child_balance
        if np.any(child_counts < 1):
            node["split_reason"] = "empty_child"
            leaves.append(node)
            return
        if child_balance < float(min_child_balance):
            node["split_reason"] = "child_balance_below_threshold"
            leaves.append(node)
            return
        if nested_gap_ratio < float(min_parent_operator_gap_ratio):
            node["split_reason"] = "parent_operator_gap_below_threshold"
            leaves.append(node)
            return
        if np.min(child_global_pi) < float(min_child_global_pi):
            node["split_reason"] = "child_global_pi_below_threshold"
            leaves.append(node)
            return

        node["split"] = True
        node["split_reason"] = "accepted"
        for child_arm in range(int(nested_basins)):
            child_clusters = node["clusters"][child_assignments == child_arm]
            child = add_node(
                node["node_id"],
                node["depth"] + 1,
                f"{node['label']}.{child_arm}",
                child_clusters,
                node["path"] + (int(child_arm),),
            )
            node["children"].append(child["node_id"])
            split_node(child)

    roots = []
    for arm in range(n_global):
        root = add_node(
            None, 0, f"arm_{arm}",
            np.flatnonzero(assignments == arm),
            (int(arm),),
        )
        roots.append(root["node_id"])
        split_node(root)

    rows = []
    for node in nodes:
        rows.append({
            "node_id": node["node_id"],
            "parent_id": node["parent_id"],
            "depth": node["depth"],
            "label": node["label"],
            "path": ".".join(map(str, node["path"])),
            "n_clusters": node["n_clusters"],
            "global_pi_mass": node["global_pi_mass"],
            "split": node["split"],
            "split_reason": node["split_reason"],
            "children": ";".join(map(str, node["children"])),
            "candidate_child_counts": node.get("candidate_child_counts"),
            "candidate_child_conditional_pi": node.get(
                "candidate_child_conditional_pi"
            ),
            "candidate_child_global_pi": node.get("candidate_child_global_pi"),
            "candidate_child_balance": node.get("candidate_child_balance"),
            "candidate_lambda2": node.get("candidate_lambda2"),
            "candidate_gap_ratio": node.get("candidate_gap_ratio"),
            "candidate_crispness": node.get("candidate_crispness"),
        })

    cluster_rows = []
    for leaf_index, leaf in enumerate(leaves):
        for cluster in leaf["clusters"]:
            cluster_rows.append({
                "cluster_id": int(cluster),
                "leaf_arm": int(leaf_index),
                "leaf_label": leaf["label"],
                "leaf_node_id": int(leaf["node_id"]),
                "global_pi": float(pi[cluster]),
            })

    return {
        "nodes": nodes,
        "roots": roots,
        "leaves": leaves,
        "node_table": pd.DataFrame(rows),
        "cluster_table": pd.DataFrame(cluster_rows),
        "parameters": {
            "min_parent_clusters_to_split": int(min_parent_clusters_to_split),
            "min_child_balance": float(min_child_balance),
            "min_parent_operator_gap_ratio": float(
                min_parent_operator_gap_ratio
            ),
            "min_child_global_pi": float(min_child_global_pi),
            "max_depth": int(max_depth),
            "nested_basins": int(nested_basins),
            "membership_threshold": membership_threshold,
        },
    }


def _conditional_gpcca_for_clusters(
    run,
    parent_clusters,
    *,
    n_nested_basins=2,
):
    """Nested GPCCA from transition counts within contiguous parent-arm bouts."""
    parent_clusters = np.asarray(parent_clusters, dtype=int)
    if parent_clusters.size < int(n_nested_basins):
        raise ValueError(
            f"Need at least {int(n_nested_basins)} clusters for conditional GPCCA"
        )

    local_index = np.full(int(run["n_clusters"]), -1, dtype=int)
    local_index[parent_clusters] = np.arange(len(parent_clusters), dtype=int)

    segments = []
    segment_rows = []
    lag = int(run["lag"])
    for individual_id, states in zip(run["individual_ids"], run["states"]):
        states = np.asarray(states, dtype=int)
        in_parent = np.isin(states, parent_clusters)
        start = None
        for index, keep in enumerate(in_parent):
            if keep and start is None:
                start = index
            elif not keep and start is not None:
                stop = index
                local_segment = local_index[states[start:stop]]
                segments.append(local_segment.astype(int, copy=False))
                segment_rows.append({
                    "individual_id": str(individual_id),
                    "start_frame": int(start),
                    "stop_frame": int(stop),
                    "n_frames": int(stop - start),
                    "usable_for_lag": bool((stop - start) > lag),
                })
                start = None
        if start is not None:
            stop = len(states)
            local_segment = local_index[states[start:stop]]
            segments.append(local_segment.astype(int, copy=False))
            segment_rows.append({
                "individual_id": str(individual_id),
                "start_frame": int(start),
                "stop_frame": int(stop),
                "n_frames": int(stop - start),
                "usable_for_lag": bool((stop - start) > lag),
            })

    usable_segments = [segment for segment in segments if len(segment) > lag]
    if not usable_segments:
        raise ValueError(
            "No within-arm segments are longer than the transition lag"
        )

    T_cond = pp.make_transition_matrix(
        usable_segments, lag=lag, n_states=len(parent_clusters),
    )
    pi_cond = pp.stationary_distribution(T_cond)
    evals, evecs = pp.leading_eigvecs(
        T_cond, k=min(10, max(1, parent_clusters.size - 1)),
    )
    gpcca = gu.run_gpcca(T_cond, M=int(n_nested_basins), eta=pi_cond)
    phi = np.asarray(evecs[:, 1:4].real)
    if phi.shape[1] < 3:
        phi = np.pad(phi, ((0, 0), (0, 3 - phi.shape[1])))
    geometry = gu.compute_hub_arms(phi, pi_cond, gpcca["chi"])
    return {
        "parent_clusters": parent_clusters,
        "T": T_cond,
        "pi": pi_cond,
        "evals": evals,
        "evecs": evecs,
        "phi": phi,
        "gpcca": gpcca,
        "chi": gpcca["chi"],
        "assignments": gpcca["assignments"],
        "geometry": geometry,
        "segments": segments,
        "segment_table": pd.DataFrame(segment_rows),
        "n_segments": int(len(segments)),
        "n_usable_segments": int(len(usable_segments)),
        "n_segment_frames": int(sum(len(segment) for segment in segments)),
        "n_usable_segment_frames": int(
            sum(len(segment) for segment in usable_segments)
        ),
    }


def apply_recursive_arm_tree(run, arm_tree):
    """Return a derived run whose arms are the terminal recursive leaves."""
    leaves = arm_tree["leaves"]
    n_leaves = len(leaves)
    n_clusters = int(run["n_clusters"])
    leaf_chi = np.zeros((n_clusters, n_leaves), dtype=float)
    leaf_assignments = np.full(n_clusters, -1, dtype=int)

    cluster_to_leaf = {}
    for leaf_index, leaf in enumerate(leaves):
        clusters = np.asarray(leaf["clusters"], dtype=int)
        leaf_assignments[clusters] = leaf_index
        for cluster in clusters:
            cluster_to_leaf[int(cluster)] = leaf_index
        _fill_leaf_membership(run, arm_tree, leaf, leaf_index, leaf_chi)

    row_sum = leaf_chi.sum(axis=1, keepdims=True)
    missing = row_sum[:, 0] <= 0
    if np.any(missing):
        for cluster in np.flatnonzero(missing):
            leaf = cluster_to_leaf.get(int(cluster))
            if leaf is not None:
                leaf_chi[cluster, leaf] = 1.0
        row_sum = leaf_chi.sum(axis=1, keepdims=True)
    leaf_chi = np.divide(
        leaf_chi, row_sum, out=np.zeros_like(leaf_chi), where=row_sum > 0,
    )

    phi = np.asarray(run["phi"])
    pi = np.asarray(run["pi"])
    geometry = gu.compute_hub_arms(phi, pi, leaf_chi)
    leaf_run = dict(run)
    leaf_run.update({
        "n_basins": int(n_leaves),
        "chi": leaf_chi,
        "assignments": leaf_assignments,
        "geometry": geometry,
        "arm_tree": arm_tree,
        "leaf_table": arm_tree["cluster_table"],
        "is_recursive_leaf_run": True,
        "original_run": run,
    })
    return leaf_run


def _fill_leaf_membership(run, arm_tree, leaf, leaf_index, leaf_chi):
    """Fill one terminal leaf column with hard membership."""
    leaf_clusters = np.asarray(leaf["clusters"], dtype=int)
    leaf_chi[leaf_clusters, leaf_index] = 1.0


def export_recursive_arm_tree(arm_tree, output_dir):
    """Save recursive split decisions and terminal cluster assignments."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    node_path = output_dir / "recursive_arm_tree_nodes.csv"
    cluster_path = output_dir / "recursive_arm_tree_clusters.csv"
    arm_tree["node_table"].to_csv(node_path, index=False)
    arm_tree["cluster_table"].to_csv(cluster_path, index=False)
    return {"nodes": str(node_path), "clusters": str(cluster_path)}


def _compact_int_list(values, max_items=12):
    values = [int(value) for value in values]
    if len(values) <= int(max_items):
        return ", ".join(map(str, values))
    head = ", ".join(map(str, values[:int(max_items)]))
    return f"{head}, ..."


def recursive_leaf_summary_table(arm_tree, *, max_cluster_preview=12):
    """Summarize terminal recursive leaf arms in a compact display table."""
    rows = []
    for leaf_arm, leaf in enumerate(arm_tree.get("leaves", [])):
        clusters = np.asarray(leaf["clusters"], dtype=int)
        path = tuple(leaf.get("path", ()))
        rows.append({
            "leaf_arm": int(leaf_arm),
            "leaf_label": leaf["label"],
            "root_arm": int(path[0]) if len(path) else np.nan,
            "depth": int(leaf["depth"]),
            "path": ".".join(map(str, path)),
            "n_clusters": int(len(clusters)),
            "global_pi_mass": float(leaf["global_pi_mass"]),
            "stop_reason": leaf["split_reason"],
            "clusters_preview": _compact_int_list(
                clusters, max_items=max_cluster_preview
            ),
        })
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values(
            ["root_arm", "path", "leaf_arm"], ascending=[True, True, True]
        ).reset_index(drop=True)
    return table


def recursive_split_summary_table(arm_tree):
    """Summarize only accepted recursive splits and their proposed children."""
    node_table = arm_tree.get("node_table", pd.DataFrame()).copy()
    if node_table.empty:
        return node_table
    keep = [
        "node_id", "depth", "label", "n_clusters", "global_pi_mass",
        "candidate_child_counts", "candidate_child_conditional_pi",
        "candidate_child_global_pi", "candidate_child_balance",
        "candidate_gap_ratio", "candidate_lambda2", "candidate_crispness",
        "children",
    ]
    keep = [column for column in keep if column in node_table]
    return (
        node_table.loc[node_table["split"].astype(bool), keep]
        .sort_values(["depth", "label"])
        .reset_index(drop=True)
    )


def plot_recursive_leaf_summary(leaf_summary):
    """Plot terminal recursive leaves as stationary-mass bars by lineage."""
    if leaf_summary.empty:
        raise ValueError("No recursive leaf arms are available to plot")
    data = leaf_summary.sort_values(
        ["root_arm", "path", "leaf_arm"], ascending=[True, True, True]
    ).reset_index(drop=True)
    labels = [
        f"{row.leaf_arm}: {row.leaf_label}\n"
        f"N={row.n_clusters}, pi={row.global_pi_mass:.3f}"
        for row in data.itertuples()
    ]
    colors = [
        ARM_PALETTE[int(root) % len(ARM_PALETTE)]
        for root in data["root_arm"].fillna(0).astype(int)
    ]
    fig, ax = plt.subplots(
        figsize=(8.0, max(3.5, 0.42 * len(data))),
        constrained_layout=True,
    )
    y = np.arange(len(data))
    ax.barh(y, data["global_pi_mass"], color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("global stationary mass")
    ax.set_title("Terminal recursive leaf arms")
    return fig


def recursive_arm_tree_eigenvalue_table(arm_tree, *, max_eigenvalues=10):
    """Tabulate eigenvalues for every recursive node with a rebuilt operator."""
    rows = []
    for node in arm_tree.get("nodes", []):
        conditional = node.get("conditional")
        if conditional is None:
            continue
        evals = np.abs(np.asarray(conditional["evals"], dtype=complex))
        row = {
            "node_id": int(node["node_id"]),
            "parent_id": node["parent_id"],
            "depth": int(node["depth"]),
            "label": node["label"],
            "path": ".".join(map(str, node["path"])),
            "n_clusters": int(node["n_clusters"]),
            "global_pi_mass": float(node["global_pi_mass"]),
            "split": bool(node["split"]),
            "split_reason": node["split_reason"],
            "candidate_child_balance": node.get("candidate_child_balance"),
            "candidate_child_counts": node.get("candidate_child_counts"),
            "candidate_child_conditional_pi": node.get(
                "candidate_child_conditional_pi"
            ),
            "candidate_child_global_pi": node.get("candidate_child_global_pi"),
        }
        for index, value in enumerate(evals[:int(max_eigenvalues)], start=1):
            row[f"lambda_{index}"] = float(value)
        if len(evals) > 2:
            row["lambda2_over_lambda3"] = float(evals[1] / max(evals[2], 1e-12))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_recursive_arm_tree_eigenvalues(
    arm_tree,
    *,
    max_eigenvalues=8,
    gap_threshold=None,
):
    """Plot spectra for each node whose within-arm transfer operator was rebuilt."""
    nodes = [node for node in arm_tree.get("nodes", []) if node.get("conditional")]
    if not nodes:
        raise ValueError("No recursive tree nodes contain rebuilt operators")
    ncols = min(3, len(nodes))
    nrows = int(np.ceil(len(nodes) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.4 * ncols, max(3.2, 3.0 * nrows)),
        constrained_layout=True,
        squeeze=False,
    )
    for panel, node in enumerate(nodes):
        ax = axes.flat[panel]
        evals = np.abs(np.asarray(node["conditional"]["evals"], dtype=complex))
        evals = evals[:int(max_eigenvalues)]
        x = np.arange(1, len(evals) + 1)
        ax.plot(x, evals, "o-", color="0.2")
        ax.set_ylim(0, 1.02)
        ax.set_xticks(x)
        ax.set_xlabel("eigenvalue index")
        ax.set_ylabel(r"$|\lambda|$")
        if len(evals) > 2:
            gap = float(evals[1] / max(evals[2], 1e-12))
            title = f"{node['label']} gap={gap:.3f}"
            if gap_threshold is not None:
                title += f" min={float(gap_threshold):.2f}"
        else:
            title = f"{node['label']} gap=n/a"
        title += f" ({node['split_reason']})"
        ax.set_title(title, fontsize=9)
    for ax in axes.flat[len(nodes):]:
        ax.remove()
    return fig


def recursive_split_diagnostics(run, *, nested_basins=2, min_clusters=2):
    """Compute the conditional operator/eigendecomposition tested per chi arm."""
    rows = []
    diagnostics = []
    assignments = np.asarray(run["assignments"], dtype=int)
    pi = np.asarray(run["pi"], dtype=float)
    for arm in range(int(run["n_basins"])):
        clusters = np.flatnonzero(assignments == arm)
        row = {
            "arm": int(arm),
            "n_clusters": int(len(clusters)),
            "global_pi": float(pi[clusters].sum()) if len(clusters) else 0.0,
        }
        if len(clusters) < int(min_clusters):
            row["status"] = "too_few_clusters"
            rows.append(row)
            continue
        try:
            diag = _conditional_gpcca_for_clusters(
                run, clusters, n_nested_basins=int(nested_basins),
            )
        except ValueError as exc:
            row["status"] = f"segment_operator_unavailable: {exc}"
            rows.append(row)
            continue
        abs_evals = np.abs(np.asarray(diag["evals"]))
        lambda2 = float(abs_evals[1]) if len(abs_evals) > 1 else np.nan
        lambda3 = float(abs_evals[2]) if len(abs_evals) > 2 else np.nan
        row.update({
            "status": "ok",
            "lambda2": lambda2,
            "lambda3": lambda3,
            "lambda2_over_lambda3": (
                float(lambda2 / max(lambda3, 1e-12))
                if np.isfinite(lambda2) and np.isfinite(lambda3) else np.nan
            ),
            "nested_crispness": float(diag["gpcca"].get("crispness", np.nan)),
            "nested_counts": diag["gpcca"]["basin_counts"].tolist(),
            "nested_pi": diag["gpcca"]["pi_basin"].tolist(),
            "nested_child_balance": float(
                np.min(diag["gpcca"]["pi_basin"])
                / max(np.sum(diag["gpcca"]["pi_basin"]), 1e-12)
            ),
            "n_segments": int(diag["n_segments"]),
            "n_usable_segments": int(diag["n_usable_segments"]),
            "n_segment_frames": int(diag["n_segment_frames"]),
            "n_usable_segment_frames": int(diag["n_usable_segment_frames"]),
        })
        diag["arm"] = int(arm)
        diagnostics.append(diag)
        rows.append(row)
    return {"summary": pd.DataFrame(rows), "diagnostics": diagnostics}


def plot_recursive_split_diagnostics(
    diagnostics, *, child_balance_threshold=None,
):
    """Plot conditional transfer operators and GPCCA eigenspaces per arm."""
    n = len(diagnostics)
    if n == 0:
        raise ValueError("No recursive split diagnostics to plot")
    fig = plt.figure(figsize=(14, max(3.8, 3.4 * n)), constrained_layout=True)
    gs = fig.add_gridspec(n, 3)
    for row, diag in enumerate(diagnostics):
        arm = int(diag["arm"])
        T = np.asarray(diag["T"], dtype=float)
        evals = np.abs(np.asarray(diag["evals"]))
        phi = np.asarray(diag["phi"])
        chi = np.asarray(diag["chi"])
        pi = np.asarray(diag["pi"])
        gpcca = diag["gpcca"]
        geometry = diag["geometry"]

        ax_T = fig.add_subplot(gs[row, 0])
        image = ax_T.imshow(
            np.log10(np.maximum(T, 1e-12)), aspect="auto", cmap="viridis"
        )
        ax_T.set_title(f"arm {arm}: segment-built T")
        ax_T.set_xlabel("cluster at t + lag")
        ax_T.set_ylabel("cluster at t")
        fig.colorbar(image, ax=ax_T, fraction=0.046, pad=0.04)

        ax_eval = fig.add_subplot(gs[row, 1])
        k = np.arange(1, len(evals) + 1)
        ax_eval.plot(k, evals, "o-", color="0.2")
        ax_eval.set_ylim(0, 1.02)
        ax_eval.set_xlabel("conditional eigenvalue index")
        ax_eval.set_ylabel(r"$|\lambda|$")
        child_pi = np.asarray(gpcca["pi_basin"], dtype=float)
        balance = float(np.min(child_pi) / max(np.sum(child_pi), 1e-12))
        title = f"lambda2={evals[1]:.3f}" if len(evals) > 1 else "lambda2=n/a"
        title += f", balance={balance:.3f}"
        if child_balance_threshold is not None:
            title += f" (min {float(child_balance_threshold):.3f})"
        ax_eval.set_title(title)

        ax_phi = fig.add_subplot(gs[row, 2], projection="3d")
        fg_kwargs = dict(
            pi=pi,
            chi=chi,
            hub=geometry["hub"],
            arm_dirs=geometry["arm_dirs"],
            arm_centroids=geometry["arm_centroids"],
        )
        try:
            import figures as fg
            fg.plot_eigenspace_3d(ax_phi, phi, **fg_kwargs)
        except Exception:
            colors = plt.get_cmap("tab10")(
                np.asarray(gpcca["assignments"]) % 10
            )
            ax_phi.scatter(phi[:, 0], phi[:, 1], phi[:, 2], s=12, c=colors)
        ax_phi.set_title(
            f"nested GPCCA counts {gpcca['basin_counts'].tolist()}"
        )
    return fig


def export_recursive_split_diagnostics(diagnostics, output_dir):
    """Save one CSV per arm for the conditional GPCCA cluster memberships."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for diag in diagnostics:
        arm = int(diag["arm"])
        table = pd.DataFrame({
            "local_cluster_index": np.arange(len(diag["parent_clusters"])),
            "cluster_id": np.asarray(diag["parent_clusters"], dtype=int),
            "conditional_pi": np.asarray(diag["pi"], dtype=float),
            "nested_assignment": np.asarray(diag["assignments"], dtype=int),
            "nested_confidence": np.asarray(diag["chi"]).max(axis=1),
        })
        for j in range(diag["chi"].shape[1]):
            table[f"nested_chi_{j}"] = diag["chi"][:, j]
        path = output_dir / f"recursive_split_diagnostics_arm_{arm}.csv"
        table.to_csv(path, index=False)
        paths.append(str(path))
        segment_path = output_dir / f"recursive_split_segments_arm_{arm}.csv"
        diag["segment_table"].to_csv(segment_path, index=False)
        paths.append(str(segment_path))
    return paths


def export_conditional_nested_clusters(nested, output_path):
    """Save the conditional nested-cluster membership table."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nested["cluster_table"].to_csv(output_path, index=False)
    return str(output_path)


def plot_conditional_nested_gpcca(run, nested):
    """Plot global parent clusters and their conditional nested GPCCA arms."""
    parent_clusters = np.asarray(nested["parent_clusters"], dtype=int)
    nested_assignments = np.asarray(nested["assignments"], dtype=int)
    n_nested = int(nested["n_nested_basins"])
    colors = plt.get_cmap("tab10")(np.arange(max(n_nested, 1)) % 10)

    fig = plt.figure(figsize=(14, 4.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 3)

    ax_global = fig.add_subplot(gs[0, 0], projection="3d")
    phi_global = np.asarray(run["phi"])
    non_parent = np.ones(run["n_clusters"], dtype=bool)
    non_parent[parent_clusters] = False
    ax_global.scatter(
        phi_global[non_parent, 1], phi_global[non_parent, 2],
        phi_global[non_parent, 0], s=4, c="0.85", alpha=0.25,
    )
    for arm in range(n_nested):
        local = nested_assignments == arm
        clusters = parent_clusters[local]
        ax_global.scatter(
            phi_global[clusters, 1], phi_global[clusters, 2],
            phi_global[clusters, 0], s=14, color=colors[arm],
            label=f"nested {arm}",
        )
    ax_global.set_title(
        f"Global eigenspace: parent basin {nested['parent_basin']}"
    )
    ax_global.set_xlabel(r"$\phi_3$")
    ax_global.set_ylabel(r"$\phi_4$")
    ax_global.set_zlabel(r"$\phi_2$")
    ax_global.legend(fontsize=8, frameon=False)

    ax_nested = fig.add_subplot(gs[0, 1], projection="3d")
    phi_nested = np.asarray(nested["phi"])
    geo = nested["geometry"]
    for arm in range(n_nested):
        mask = nested_assignments == arm
        ax_nested.scatter(
            phi_nested[mask, 1], phi_nested[mask, 2], phi_nested[mask, 0],
            s=18, color=colors[arm], label=f"nested {arm}",
        )
        centroid = geo["arm_centroids"][arm]
        hub = geo["hub"]
        ax_nested.plot(
            [hub[1], centroid[1]], [hub[2], centroid[2]],
            [hub[0], centroid[0]], color=colors[arm], lw=2.0,
        )
    ax_nested.scatter(
        [geo["hub"][1]], [geo["hub"][2]], [geo["hub"][0]],
        marker="*", s=120, color="red", edgecolor="black",
    )
    ax_nested.set_title("Conditional eigenspace")
    ax_nested.set_xlabel(r"$\psi_3$")
    ax_nested.set_ylabel(r"$\psi_4$")
    ax_nested.set_zlabel(r"$\psi_2$")

    ax_mass = fig.add_subplot(gs[0, 2])
    counts = np.bincount(nested_assignments, minlength=n_nested)
    mass = np.bincount(
        nested_assignments, weights=nested["pi"], minlength=n_nested,
    )
    x = np.arange(n_nested)
    ax_mass.bar(x - 0.18, counts, width=0.36, color=colors[:n_nested],
                alpha=0.8, label="clusters")
    ax_mass_2 = ax_mass.twinx()
    ax_mass_2.bar(x + 0.18, mass, width=0.36, color="0.25",
                  alpha=0.55, label="conditional mass")
    ax_mass.set_xticks(x)
    ax_mass.set_xlabel("nested arm")
    ax_mass.set_ylabel("microstate count")
    ax_mass_2.set_ylabel("conditional stationary mass")
    ax_mass.set_title("Nested arm support")
    return fig


def maybe_save(fig, output_dir, name, save=False):
    if save:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f"{name}.png", dpi=200, bbox_inches="tight")


def participation_ratio(phi):
    phi = np.asarray(phi)
    return np.sum(phi ** 2, axis=0) ** 2 / np.maximum(
        np.sum(phi ** 4, axis=0), 1e-15
    )


def _transfer_order(transfer):
    """Order states once from pooled incoming and outgoing transition profiles."""
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import pdist

    transfer = np.asarray(transfer, dtype=float)
    profiles = np.concatenate([transfer, transfer.T], axis=1)
    distances = pdist(profiles, metric="euclidean")
    if len(distances) == 0 or np.allclose(distances, 0):
        return np.arange(transfer.shape[0])
    return leaves_list(linkage(distances, method="average"))


def transfer_operator_figures(
    run,
    *,
    individuals_per_page=6,
    order=None,
    log_floor=1e-6,
    visited_only=True,
    min_cluster_time_seconds=0.0,
):
    """Plot pooled and per-individual operators with one ordering/color scale."""
    order = _transfer_order(run["T"]) if order is None else np.asarray(order)
    keep = cluster_time_mask(run, min_cluster_time_seconds)
    order = order[keep[order]]
    if len(order) == 0:
        raise ValueError("Cluster-time threshold removed every cluster")
    vmin = np.log10(float(log_floor))
    vmax = 0.0

    def ordered_log(transfer, state_order=order):
        values = np.asarray(transfer)[np.ix_(state_order, state_order)]
        return np.log10(np.maximum(values, log_floor))

    pooled_fig, pooled_ax = plt.subplots(
        figsize=(6.5, 5.8), constrained_layout=True
    )
    pooled_image = pooled_ax.imshow(
        ordered_log(run["T"]), aspect="auto", cmap="viridis",
        vmin=vmin, vmax=vmax, interpolation="nearest",
    )
    pooled_ax.set(
        title=f"Pooled transfer operator (lag={run['lag']} frames)",
        xlabel="cluster at t + lag",
        ylabel="cluster at t",
    )
    pooled_fig.colorbar(
        pooled_image, ax=pooled_ax, label="log10 transition probability"
    )

    individual_figures = []
    per_page = max(1, int(individuals_per_page))
    for page_start in range(0, len(run["states"]), per_page):
        page_stop = min(page_start + per_page, len(run["states"]))
        count = page_stop - page_start
        ncols = min(3, count)
        nrows = int(np.ceil(count / ncols))
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows),
            constrained_layout=True, squeeze=False,
        )
        image = None
        for panel, index in enumerate(range(page_start, page_stop)):
            ax = axes.flat[panel]
            transfer = pp.make_transition_matrix(
                [run["states"][index]],
                lag=run["lag"],
                n_states=run["n_clusters"],
            )
            individual_order = order
            if visited_only:
                visited = np.bincount(
                    np.asarray(run["states"][index], dtype=int),
                    minlength=run["n_clusters"],
                ) > 0
                individual_order = order[visited[order]]
            image = ax.imshow(
                ordered_log(transfer, individual_order),
                aspect="auto", cmap="viridis",
                vmin=vmin, vmax=vmax, interpolation="nearest",
            )
            ax.set_title(
                f"{run['individual_ids'][index]} "
                f"(N={len(individual_order)} visited)"
            )
            ax.set_xlabel("cluster at t + lag")
            ax.set_ylabel("cluster at t")
        for ax in axes.flat[count:]:
            ax.remove()
        if image is not None:
            fig.colorbar(
                image, ax=list(axes.flat[:count]),
                label="log10 transition probability", shrink=0.8,
            )
        individual_figures.append(fig)
    return pooled_fig, individual_figures, order


def _smooth_chi(chi_seq, width):
    return uniform_filter1d(
        np.asarray(chi_seq, dtype=float),
        size=max(1, int(width)),
        axis=0,
        mode="nearest",
    )


def figure_4(run, *, fixed=None, min_cluster_time_seconds=0.0):
    """Recreate the fly Figure 4 geometry, spectrum, PR, and chi panels."""
    keep = cluster_time_mask(run, min_cluster_time_seconds)
    if not np.any(keep):
        raise ValueError("Cluster-time threshold removed every cluster")
    phi = run["phi"][keep]
    chi = run["chi"][keep]
    pi = run["pi"][keep]
    assignments = run["assignments"][keep]
    geo = gu.compute_hub_arms(phi, pi, chi)
    m = run["n_basins"]
    fig = plt.figure(figsize=(14, 11), constrained_layout=True)
    gs = fig.add_gridspec(3, 3)

    ax_a = fig.add_subplot(gs[0, 0], projection="3d")
    if fixed is None:
        ax_a.text2D(
            0.5, 0.5, "Optional fixed-timescale eigenvectors\nnot supplied",
            transform=ax_a.transAxes, ha="center", va="center",
        )
    else:
        pf = np.asarray(fixed["phi"])[:, :3]
        ax_a.scatter(pf[:, 1], pf[:, 2], pf[:, 0], s=6, c="0.25", alpha=0.5)
    ax_a.set_title("A  fixed-timescale control")

    ax_b = fig.add_subplot(gs[0, 1], projection="3d")
    fuzzy = chi.max(axis=1) < 0.5
    ax_b.scatter(
        phi[fuzzy, 1], phi[fuzzy, 2], phi[fuzzy, 0],
        s=5, c="0.8", alpha=0.4,
    )
    for j in range(m):
        mask = (assignments == j) & ~fuzzy
        ax_b.scatter(
            phi[mask, 1], phi[mask, 2], phi[mask, 0],
            s=10, color=ARM_PALETTE[j % len(ARM_PALETTE)],
        )
        end = geo["arm_centroids"][j]
        hub = geo["hub"]
        ax_b.plot(
            [hub[1], end[1]], [hub[2], end[2]], [hub[0], end[0]],
            color=ARM_PALETTE[j % len(ARM_PALETTE)], lw=2.5,
        )
    ax_b.scatter(
        [geo["hub"][1]], [geo["hub"][2]], [geo["hub"][0]],
        marker="*", s=140, color="red", edgecolor="black",
    )
    ax_b.set_title(
        f"B  multi-timescale arms ({keep.sum()}/{len(keep)} clusters shown)"
    )
    ax_b.set_xlabel(r"$\phi_3$")
    ax_b.set_ylabel(r"$\phi_4$")
    ax_b.set_zlabel(r"$\phi_2$")

    pair_specs = [
        (0, 1, r"$\phi_2$", r"$\phi_3$"),
        (0, 2, r"$\phi_2$", r"$\phi_4$"),
        (1, 2, r"$\phi_3$", r"$\phi_4$"),
    ]
    for panel, (x_idx, y_idx, x_label, y_label) in enumerate(pair_specs):
        ax_pair = fig.add_subplot(gs[1, panel])
        for j in range(m):
            mask = assignments == j
            ax_pair.scatter(
                phi[mask, x_idx], phi[mask, y_idx], s=10,
                color=ARM_PALETTE[j % len(ARM_PALETTE)], alpha=0.8,
            )
            centroid = geo["arm_centroids"][j]
            ax_pair.plot(
                [geo["hub"][x_idx], centroid[x_idx]],
                [geo["hub"][y_idx], centroid[y_idx]],
                color=ARM_PALETTE[j % len(ARM_PALETTE)], lw=2,
            )
        ax_pair.scatter(
            geo["hub"][x_idx], geo["hub"][y_idx], marker="*", s=140,
            color="red", edgecolor="black", zorder=10,
        )
        ax_pair.set_title(f"B{panel + 1}  {x_label} vs {y_label}")
        ax_pair.set_xlabel(x_label)
        ax_pair.set_ylabel(y_label)
        ax_pair.set_aspect("equal", adjustable="datalim")

    ax_c = fig.add_subplot(gs[0, 2])
    values = np.sort(np.abs(np.linalg.eigvals(run["T"])))[::-1]
    ax_c.plot(np.arange(1, min(20, len(values)) + 1), values[:20], "o-")
    ax_c.axvline(m, color="red", ls="--")
    ax_c.set(xlabel="eigenvalue index", ylabel=r"$|\lambda_k|$",
             title="C  spectrum")

    ax_d = fig.add_subplot(gs[2, 0])
    pr = participation_ratio(phi)
    ax_d.plot(np.arange(2, len(pr) + 2), pr, "o-", color="0.15",
              label="multi-timescale")
    if fixed is not None:
        pr_fixed = participation_ratio(np.asarray(fixed["phi"])[:, : len(pr)])
        ax_d.plot(np.arange(2, len(pr_fixed) + 2), pr_fixed, "o-",
                  color="firebrick", label="fixed-timescale")
    ax_d.set_yscale("log")
    ax_d.set(xlabel="eigenvector index", ylabel="participation ratio",
             title="D  collective support")
    ax_d.legend()

    ax_e = fig.add_subplot(gs[2, 1:])
    seq = np.asarray(run["states"][0])
    limit = min(len(seq), int(20 * 60 * run["fs"]))
    seq = seq[:limit]
    valid = seq >= 0
    safe_seq = np.where(valid, seq, 0)
    smooth = _smooth_chi(run["chi"][safe_seq], 5 * run["fs"])
    smooth[~valid] = np.nan
    t_min = np.arange(limit) / run["fs"] / 60
    for j in range(m):
        ax_e.plot(t_min, smooth[:, j], color=ARM_PALETTE[j % len(ARM_PALETTE)], lw=1,
                  label=f"arm {j + 1}")
    ax_e.set(xlabel="time (min)", ylabel=r"$\chi_j(t)$",
             title="E  representative individual")
    ax_e.legend(ncol=2, fontsize=8)
    return fig


def predictive_mi(sequences, lag, n_states):
    counts = np.zeros((n_states, n_states), dtype=float)
    for seq in sequences:
        seq = np.asarray(seq, dtype=int)
        if len(seq) > lag:
            np.add.at(counts, (seq[:-lag], seq[lag:]), 1)
    p = counts / max(counts.sum(), 1)
    left = p.sum(axis=1)
    right = p.sum(axis=0)
    expected = left[:, None] * right[None, :]
    mask = (p > 0) & (expected > 0)
    return float(np.sum(p[mask] * np.log2(p[mask] / expected[mask])))


def residence_times(run, delta=2.0):
    m = run["n_basins"]
    pooled = [[] for _ in range(m)]
    for seq in run["states"]:
        labels = np.argmax(
            _smooth_chi(run["chi"][np.asarray(seq)], delta * run["fs"]),
            axis=1,
        )
        changes = np.r_[0, np.flatnonzero(np.diff(labels)) + 1, len(labels)]
        for start, stop in zip(changes[:-1], changes[1:]):
            pooled[labels[start]].append((stop - start) / run["fs"])
    return [np.asarray(values, dtype=float) for values in pooled]


def figure_5(run, *, behavior_density=None):
    """Recreate fly Figure 5 using imported behavior density when available."""
    m = run["n_basins"]
    fs = run["fs"]
    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    top = gs[0, :].subgridspec(1, m)
    if behavior_density is None:
        ax = fig.add_subplot(gs[0, :])
        ax.text(
            0.5, 0.5,
            "A  Optional behavior_density_chi matrix not supplied",
            ha="center", va="center",
        )
        ax.axis("off")
    else:
        cond = np.asarray(behavior_density["cond"])
        dev = cond - np.nanmean(cond, axis=0, keepdims=True)
        vmax = np.nanpercentile(np.abs(dev), 99)
        for j in range(m):
            ax = fig.add_subplot(top[0, j])
            ax.pcolormesh(
                behavior_density["x_edges"], behavior_density["y_edges"],
                dev[j], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                shading="auto",
            )
            ax.set_title(f"Arm {j + 1}", color=ARM_PALETTE[j])
            ax.set_aspect("equal")
            ax.axis("off")

    lag_seconds = np.array([0.1, 0.5, 1, 2, 5, 10, 30, 60, 120])
    lag_frames = np.maximum(1, np.rint(lag_seconds * fs).astype(int))
    decay = np.full((len(lag_frames), m - 1), np.nan)
    for row, lag in enumerate(lag_frames):
        transfer = pp.make_transition_matrix(
            run["states"], lag=int(lag), n_states=run["n_clusters"]
        )
        evals, _ = pp.leading_eigvecs(transfer, k=m)
        decay[row] = -np.log(np.maximum(np.abs(evals[1:m]), 1e-12)) / (
            lag / fs
        )
    ax_b = fig.add_subplot(gs[1, 0])
    for k in range(m - 1):
        ax_b.loglog(lag_seconds, decay[:, k], "o-", label=rf"$r_{k + 2}$")
    ax_b.set(xlabel=r"lag $\tau$ (s)", ylabel="apparent decay rate",
             title="B  slow-mode decay")
    ax_b.legend()

    basin_map = run["assignments"]
    basin_sequences = [basin_map[np.asarray(seq)] for seq in run["states"]]
    mi_emp = np.array(
        [predictive_mi(basin_sequences, int(lag), m) for lag in lag_frames]
    )
    basin_t = pp.make_transition_matrix(
        basin_sequences, lag=run["lag"], n_states=m
    )
    pi_b = pp.stationary_distribution(basin_t)
    mi_markov = []
    for lag in lag_frames:
        steps = max(1, int(round(lag / run["lag"])))
        matrix = np.linalg.matrix_power(basin_t, steps)
        joint = pi_b[:, None] * matrix
        expected = joint.sum(axis=1)[:, None] * joint.sum(axis=0)[None, :]
        mask = (joint > 0) & (expected > 0)
        mi_markov.append(
            np.sum(joint[mask] * np.log2(joint[mask] / expected[mask]))
        )
    ax_c = fig.add_subplot(gs[1, 1])
    ax_c.semilogx(lag_seconds, mi_emp, "o-", label="data")
    ax_c.semilogx(lag_seconds, mi_markov, "s--", label="Markov")
    ax_c.set(xlabel=r"lag $\tau$ (s)", ylabel="predictive MI (bits)",
             title="C  predictive information")
    ax_c.legend()

    ax_d = fig.add_subplot(gs[1, 2])
    for j, values in enumerate(residence_times(run)):
        values = np.sort(values)
        ccdf = 1 - np.arange(len(values)) / max(len(values), 1)
        ax_d.loglog(values, ccdf, color=ARM_PALETTE[j], label=f"arm {j + 1}")
    ax_d.set(xlabel="metastable residence (s)", ylabel="CCDF",
             title="D  residence times")
    ax_d.legend()
    return fig


def supplement_7(run, *, cao_e1=None, entropy_gap=None, cv_pi=None):
    """Fly methodological diagnostics corresponding to Supplement S7."""
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
    eigvals = np.asarray(run["projection"]["eigvals_pca"])
    axes[0, 0].plot(np.arange(1, len(eigvals) + 1),
                    np.cumsum(eigvals) / eigvals.sum(), "o-")
    axes[0, 0].axvline(run["projection"]["n_kept"], color="red", ls="--")
    axes[0, 0].set(title="A  PCA variance", xlabel="PC", ylabel="cumulative")

    if cao_e1 is None:
        axes[0, 1].text(0.5, 0.5, "Optional Cao E1 array", ha="center")
    else:
        axes[0, 1].plot(np.arange(1, len(cao_e1) + 1), cao_e1, "o-")
    axes[0, 1].set(title="B  Cao embedding criterion", xlabel="d",
                   ylabel=r"$E_1(d)$")

    if entropy_gap is None:
        axes[0, 2].text(0.5, 0.5, "Optional entropy-gap arrays", ha="center")
    else:
        axes[0, 2].plot(entropy_gap["N"], entropy_gap["gap"], "o-")
    axes[0, 2].set(title="C  entropy gap", xlabel="N", ylabel=r"$\Delta h$")

    tau_values = np.array([0.5, 1, 2, 3, 5, 10])
    spectra = []
    crispness = []
    basin_sizes = []
    for tau in tau_values:
        lag = max(1, int(round(tau * run["fs"])))
        matrix = pp.make_transition_matrix(
            run["states"], lag=lag, n_states=run["n_clusters"]
        )
        evals, _ = pp.leading_eigvecs(matrix, k=7)
        spectra.append(np.abs(evals))
        out = gu.run_gpcca(matrix, M=run["n_basins"],
                           eta=pp.stationary_distribution(matrix))
        crispness.append(out["crispness"])
        basin_sizes.append(np.sort(out["basin_counts"])[::-1])
    spectra = np.asarray(spectra)
    for k in range(1, min(6, spectra.shape[1])):
        axes[0, 3].plot(tau_values, spectra[:, k], "o-", label=rf"$\lambda_{k+1}$")
    axes[0, 3].set_xscale("log")
    axes[0, 3].set(title="D  eigenvalues vs lag", xlabel="tau (s)",
                   ylabel=r"$|\lambda|$")

    axes[1, 0].plot(tau_values, crispness, "o-")
    axes[1, 0].set(title="E  GPCCA stability", xlabel="tau (s)",
                   ylabel="crispness")
    for j in range(run["n_basins"]):
        axes[1, 1].plot(tau_values, np.asarray(basin_sizes)[:, j], "o-")
    axes[1, 1].set(title="F  basin sizes vs lag", xlabel="tau (s)",
                   ylabel="clusters")

    m_values = np.arange(2, min(9, run["n_clusters"]))
    sizes = []
    for m in m_values:
        out = gu.run_gpcca(run["T"], M=int(m), eta=run["pi"])
        row = np.full(m_values.max(), np.nan)
        row[:m] = np.sort(out["basin_counts"])[::-1]
        sizes.append(row)
    for j in range(np.asarray(sizes).shape[1]):
        axes[1, 2].plot(m_values, np.asarray(sizes)[:, j], "o-", alpha=0.7)
    axes[1, 2].set(title="G  basin sizes vs M", xlabel="M", ylabel="clusters")

    if cv_pi is None:
        axes[1, 3].text(0.5, 0.5, "Optional held-out PI matrix", ha="center")
    else:
        mean = np.nanmean(cv_pi, axis=0)
        sem = np.nanstd(cv_pi, axis=0, ddof=1) / np.sqrt(cv_pi.shape[0])
        axes[1, 3].errorbar(m_values[:len(mean)], mean, yerr=sem, fmt="o-")
    axes[1, 3].set(title="H  held-out prediction", xlabel="M",
                   ylabel="PI/transition")
    return fig


def supplement_8(run):
    """Fly residence model comparison and slow-membership shape (S8)."""
    dwells = residence_times(run)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for j, values in enumerate(dwells):
        values = np.sort(values)
        axes[0].loglog(
            values, 1 - np.arange(len(values)) / len(values),
            color=ARM_PALETTE[j], label=f"arm {j + 1}",
        )
    axes[0].set(title="A  residence CCDF", xlabel="seconds", ylabel="CCDF")
    axes[0].legend()

    comparisons = []
    try:
        import powerlaw
        for values in dwells:
            fit = powerlaw.Fit(values, discrete=False, verbose=False)
            comparisons.append([
                fit.distribution_compare("power_law", "lognormal",
                                         normalized_ratio=True)[0],
                fit.distribution_compare("power_law", "truncated_power_law",
                                         normalized_ratio=True)[0],
            ])
        axes[1].bar(np.arange(len(dwells)) - 0.18,
                    np.asarray(comparisons)[:, 0], 0.36, label="PL vs lognormal")
        axes[1].bar(np.arange(len(dwells)) + 0.18,
                    np.asarray(comparisons)[:, 1], 0.36, label="PL vs truncated")
    except ImportError:
        axes[1].text(0.5, 0.5, "Install powerlaw for model comparisons",
                     ha="center")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set(title="B  residence model comparison", xlabel="arm",
                ylabel="normalized likelihood ratio")
    axes[1].legend()

    for j in range(run["n_basins"]):
        values = []
        for seq in run["states"]:
            smooth = _smooth_chi(run["chi"][np.asarray(seq)], 2 * run["fs"])
            z = np.log(np.clip(smooth[:, j], 1e-6, 1 - 1e-6) /
                       np.clip(1 - smooth[:, j], 1e-6, 1))
            values.append(z[::max(1, int(run["fs"]))])
        values = np.concatenate(values)
        axes[2].hist(values, bins=80, density=True, histtype="step",
                     color=ARM_PALETTE[j], label=f"arm {j + 1}")
    axes[2].set_yscale("log")
    axes[2].set(title="C  slow-membership shape", xlabel=r"logit($\bar\chi$)",
                ylabel="density")
    axes[2].legend()
    return fig


def supplement_9(run, *, arm_cosines=None):
    """Fly per-individual arm reproducibility and occupancy (S9)."""
    m = run["n_basins"]
    occupancy = np.zeros((len(run["states"]), m))
    for i, seq in enumerate(run["states"]):
        labels = run["assignments"][np.asarray(seq)]
        occupancy[i] = np.bincount(labels, minlength=m) / len(labels)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    if arm_cosines is None:
        axes[0].text(0.5, 0.5, "Optional per-individual refit arm cosines",
                     ha="center")
    else:
        for j in range(m):
            axes[0].scatter(
                np.full(len(arm_cosines), j), np.asarray(arm_cosines)[:, j],
                color=ARM_PALETTE[j], alpha=0.7,
            )
    axes[0].set(title="A  individual vs pooled arms", ylabel="cosine similarity")
    order = np.argsort(occupancy[:, 0])
    bottom = np.zeros(len(order))
    for j in range(m):
        axes[1].bar(
            np.arange(len(order)), occupancy[order, j], bottom=bottom,
            color=ARM_PALETTE[j], width=1, label=f"arm {j + 1}",
        )
        bottom += occupancy[order, j]
    axes[1].set(title="B  individual occupancy", xlabel="individual",
                ylabel="fraction")
    axes[1].legend()
    return fig


def _simulate_markov(transition, length, start, rng):
    cdf = np.cumsum(transition, axis=1)
    result = np.empty(length, dtype=np.int32)
    result[0] = int(start)
    for i in range(1, length):
        result[i] = np.searchsorted(cdf[result[i - 1]], rng.random())
    return result


def supplement_10(run, *, seed=0):
    """Fly-only Markov-surrogate residence controls from S10."""
    rng = np.random.default_rng(seed)
    lag1 = pp.make_transition_matrix(
        run["states"], lag=1, n_states=run["n_clusters"]
    )
    data = residence_times(run)
    surrogate_states = [
        _simulate_markov(lag1, len(seq), int(seq[0]), rng)
        for seq in run["states"]
    ]
    surrogate_run = {**run, "states": surrogate_states}
    surrogate = residence_times(surrogate_run)
    fig, axes = plt.subplots(1, run["n_basins"], figsize=(14, 3.5),
                             constrained_layout=True)
    for j, ax in enumerate(np.atleast_1d(axes)):
        for values, style, label in [
            (data[j], "-", "data"), (surrogate[j], "--", "one-step Markov")
        ]:
            values = np.sort(values)
            ax.loglog(values, 1 - np.arange(len(values)) / len(values),
                      style, label=label)
        ax.set(title=f"Arm {j + 1}", xlabel="residence (s)", ylabel="CCDF")
        ax.legend(fontsize=8)
    return fig


def supplement_11(run, *, deltas=(0, 0.5, 1, 2, 5)):
    """Fly-only smoothing-scale residence sensitivity from S11."""
    medians = np.zeros((len(deltas), run["n_basins"]))
    tail_mass = np.zeros_like(medians)
    for row, delta in enumerate(deltas):
        values = residence_times(run, delta=float(delta))
        for j, dwell in enumerate(values):
            medians[row, j] = np.median(dwell)
            tail_mass[row, j] = np.mean(dwell >= 30)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for j in range(run["n_basins"]):
        axes[0].plot(deltas, medians[:, j], "o-", color=ARM_PALETTE[j],
                     label=f"arm {j + 1}")
        axes[1].plot(deltas, tail_mass[:, j], "o-", color=ARM_PALETTE[j])
    axes[0].set(title="A  residence scale vs smoothing", xlabel="Delta (s)",
                ylabel="median residence (s)")
    axes[1].set(title="B  long-tail robustness", xlabel="Delta (s)",
                ylabel="fraction >= 30 s")
    axes[0].legend()
    return fig
