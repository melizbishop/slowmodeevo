"""Data access helpers for all-to-all keypoint-distance slow-mode analyses."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


@dataclass
class AllPairDistanceData:
    """Manifest-backed loader for all unique pairwise keypoint distances."""

    dataverse_root: Path
    manifest_path: Path | None = None
    metadata_path: Path | None = None
    source_root: Path | None = None
    confidence_dataset: str = "confidences"
    distance_chunk_frames: int = 20_000
    default_samplerate: float = 120.0

    def __post_init__(self):
        import json

        self.dataverse_root = Path(self.dataverse_root)
        keypoint_root = self.dataverse_root / "keypoint_moseq"
        self.manifest_path = Path(
            self.manifest_path or keypoint_root / "manifest_by_species.csv"
        )
        self.metadata_path = Path(
            self.metadata_path or keypoint_root / "metadata.json"
        )
        self.source_root = Path(
            self.source_root or self.dataverse_root / "data_files"
        )
        self.manifest = pd.read_csv(self.manifest_path)
        self.manifest = self.manifest[
            self.manifest["status"].isin(["ok", "exists"])
        ].copy()
        self.manifest["subject"] = self.manifest["subject"].astype(str)
        self.manifest = self.manifest.sort_values(
            ["species_folder", "subject", "recording_id"]
        )
        with open(self.metadata_path) as handle:
            self.metadata = json.load(handle)
        self.available_species = sorted(
            self.manifest["species_folder"].dropna().astype(str).unique()
        )
        self.left_keypoint, self.right_keypoint = np.triu_indices(
            len(self.metadata["bodyparts"]), 1
        )

    @property
    def n_distance_features(self):
        return len(self.left_keypoint)

    def validate_species(self, species):
        species = [species] if isinstance(species, str) else list(species)
        missing = sorted(set(species) - set(self.available_species))
        if missing:
            raise ValueError(
                f"Unknown species {missing}; available: {self.available_species}"
            )
        return species

    def source_path(self, row):
        output_file = row.get("output_file")
        if pd.notna(output_file):
            return self.dataverse_root / output_file
        return self.source_root / f"{row['recording_id']}.hdf5"

    def samplerate(self, row):
        with h5py.File(self.source_path(row), "r") as handle:
            if "recording" in handle and "samplerate" in handle["recording"].attrs:
                return float(handle["recording"].attrs["samplerate"])
            if "samplerate" in handle.attrs:
                return float(handle.attrs["samplerate"])
        return float(self.default_samplerate)

    def _dataset_aliases(self, dataset_name):
        aliases = [dataset_name]
        if dataset_name.startswith("recording/"):
            leaf_name = dataset_name.rsplit("/", 1)[-1]
            aliases.append(leaf_name)
            aliases.extend({
                "egocentric_coordinates_rigid": ["coordinates", "distances"],
                "size_normalized_gimbal_keypoints": [
                    "size_normalized_coordinates",
                    "size_normalized_distances",
                ],
                "confidences_3d": ["confidences"],
            }.get(leaf_name, []))
        aliases.extend({
            "egocentered_and_normalized_distances": ["distances", "coordinates"],
            "normalized_distances": [
                "size_normalized_distances",
                "size_normalized_coordinates",
            ],
        }.get(dataset_name, []))
        return list(dict.fromkeys(aliases))

    def _read_dataset(self, handle, dataset_name, frame_slice):
        for candidate in self._dataset_aliases(dataset_name):
            if candidate in handle:
                return np.asarray(handle[candidate][frame_slice], dtype=np.float32), candidate
        raise KeyError(
            f"Missing {dataset_name!r}; tried {self._dataset_aliases(dataset_name)}"
        )

    def species_subjects(self, species_name):
        rows = self.manifest[self.manifest["species_folder"].eq(species_name)]
        if rows.empty:
            raise ValueError(f"Unknown species {species_name!r}")
        return sorted(rows["subject"].dropna().astype(str).unique())

    def select_manifest(self, species_name, individual_ids=None):
        rows = self.manifest[
            self.manifest["species_folder"].eq(species_name)
        ].copy()
        subjects = self.species_subjects(species_name)
        if individual_ids is None:
            keep = subjects
        elif isinstance(individual_ids, (str, int, float)):
            keep = [str(individual_ids)]
        else:
            keep = [str(value) for value in individual_ids]
        missing = sorted(set(keep) - set(subjects))
        if missing:
            raise ValueError(f"{species_name}: unknown individuals {missing}")
        return rows[
            rows["subject"].astype(str).isin(keep)
        ].sort_values(["subject", "recording_id"])

    def coordinates_to_distances(self, coordinates, chunk_frames=None):
        """Convert ``(frames, keypoints, xyz)`` coordinates to unique distances."""
        coordinates = np.asarray(coordinates)
        chunk_frames = int(chunk_frames or self.distance_chunk_frames)
        distances = np.empty(
            (len(coordinates), self.n_distance_features), dtype=np.float32
        )
        for start in range(0, len(coordinates), chunk_frames):
            stop = min(start + chunk_frames, len(coordinates))
            block = np.asarray(coordinates[start:stop], dtype=np.float32)
            difference = (
                block[:, self.left_keypoint] - block[:, self.right_keypoint]
            )
            distances[start:stop] = np.sqrt(
                np.sum(difference * difference, axis=2, dtype=np.float32)
            )
        return distances

    def make_loader(self, coordinate_dataset, representation_name):
        """Return a loader compatible with ``pooled_user_pipeline``."""

        def load_individual_trace(
            species_name,
            individual_id=None,
            dataset_name=representation_name,
            dtype=np.float32,
        ):
            if dataset_name != representation_name:
                raise ValueError(
                    f"Expected {representation_name!r}; got {dataset_name!r}"
                )
            subjects = self.species_subjects(species_name)
            individual_id = str(
                subjects[0] if individual_id is None else individual_id
            )
            rows = self.select_manifest(species_name, individual_id)
            distance_segments = []
            confidence_segments = []
            recording_ids = []
            segment_lengths = []
            samplerates = []
            frame_slice = slice(None)
            loaded_dataset = None
            for _, row in rows.iterrows():
                path = self.source_path(row)
                samplerates.append(self.samplerate(row))
                with h5py.File(path, "r") as handle:
                    values, loaded_dataset = self._read_dataset(
                        handle, coordinate_dataset, frame_slice
                    )
                    confidences, _ = self._read_dataset(
                        handle, self.confidence_dataset, frame_slice
                    )
                if values.ndim == 2 and values.shape[1] == self.n_distance_features:
                    distances = values
                elif values.ndim == 3:
                    distances = self.coordinates_to_distances(values)
                else:
                    raise ValueError(
                        f"{path}: expected coordinates or distances, got {values.shape}"
                    )
                distance_segments.append(distances.astype(dtype, copy=False))
                confidence_segments.append(confidences)
                recording_ids.append(str(row["recording_id"]))
                segment_lengths.append(len(distances))
            if len(set(samplerates)) != 1:
                raise ValueError(
                    f"{species_name}/{individual_id}: samplerates differ: "
                    f"{samplerates}"
                )
            raw_fs = samplerates[0]
            return {
                "species_folder": species_name,
                "individual_id": individual_id,
                "dataset_name": representation_name,
                "dataset": loaded_dataset or coordinate_dataset,
                "X": np.concatenate(distance_segments),
                "confidences": np.concatenate(confidence_segments),
                "recording_ids": recording_ids,
                "segment_lengths": np.asarray(segment_lengths, dtype=int),
                "raw_fs": raw_fs,
                "fs": raw_fs,
                "n_recordings": len(recording_ids),
            }

        return load_individual_trace

    def representation_preflight(
        self,
        species,
        representations,
        max_source_frames=20_000,
        frame_step=20,
    ):
        """Compare distance scales for the first recording of each species."""
        names = list(representations)
        if len(names) < 2:
            raise ValueError("At least two representations are required")
        rows = []
        samples = {}
        for species_name in self.validate_species(species):
            row = self.manifest[
                self.manifest["species_folder"].eq(species_name)
            ].iloc[0]
            path = self.source_path(row)
            with h5py.File(path, "r") as handle:
                missing = [
                    dataset for dataset in representations.values()
                    if dataset not in handle
                ]
                if missing:
                    raise KeyError(f"{path}: missing {missing}")
                n_frames = min(
                    int(max_source_frames),
                    *(handle[dataset].shape[0] for dataset in representations.values()),
                )
                frame_index = np.arange(0, n_frames, int(frame_step), dtype=int)
                distances = {
                    name: self.coordinates_to_distances(
                        np.asarray(
                            handle[dataset][frame_index], dtype=np.float32
                        )
                    )
                    for name, dataset in representations.items()
                }
            left, right = names[:2]
            ratio = distances[left] / np.maximum(distances[right], 1e-8)
            rows.append({
                "species": species_name,
                "recording_id": str(row["recording_id"]),
                "left_representation": left,
                "right_representation": right,
                "sampled_frames": len(frame_index),
                "distance_features": distances[left].shape[1],
                "left_distance_median": float(np.median(distances[left])),
                "right_distance_median": float(np.median(distances[right])),
                "median_left_to_right_ratio": float(np.median(ratio)),
                "scale_ratio_cv": float(
                    np.std(ratio) / np.maximum(np.mean(ratio), 1e-12)
                ),
                "distance_correlation": float(
                    np.corrcoef(
                        distances[left].ravel(), distances[right].ravel()
                    )[0, 1]
                ),
            })
            samples[species_name] = distances
        return pd.DataFrame(rows), samples


def select_individuals(subjects, n=None, seed=0):
    """Deterministically select all or a random subset of individual IDs."""
    subjects = [str(value) for value in subjects]
    if n is None or int(n) >= len(subjects):
        return subjects
    rng = np.random.default_rng(seed)
    return rng.choice(subjects, size=int(n), replace=False).tolist()
