"""Memory-safe pooled slow-mode pipeline for user_data.ipynb.

The workflow is:

1. Compute checkpointed, subsampled wavelet rows for one individual at a time.
2. Reuse those per-individual checkpoints to fit one shared PCA basis.
3. Stream wavelet feature blocks and project every frame onto that basis.
4. Fit shared k-means clusters from subsampled embedded projections.
5. Assign every embedded frame, pool per-individual state sequences, and fit
   the slow modes from one pooled transition matrix.

This preserves full-frame wavelets without ever holding an individual's full
wavelet-amplitude matrix in RAM at once.
"""
from __future__ import annotations

import csv
from pathlib import Path
import gc
import pickle
import shutil
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.exceptions import InconsistentVersionWarning
from numpy.lib.format import open_memmap

import pipeline as pp
import gpcca_utils as gu


class CovariancePCA:
    """PCA fitted from a streamed full covariance matrix."""

    def __init__(
        self,
        *,
        components,
        mean,
        explained_variance,
        explained_variance_ratio,
        singular_values,
        n_samples,
        n_features,
    ):
        self.components_ = np.asarray(components, dtype=np.float64)
        self.mean_ = np.asarray(mean, dtype=np.float64)
        self.explained_variance_ = np.asarray(explained_variance, dtype=np.float64)
        self.explained_variance_ratio_ = np.asarray(
            explained_variance_ratio, dtype=np.float64
        )
        self.singular_values_ = np.asarray(singular_values, dtype=np.float64)
        self.n_samples_seen_ = int(n_samples)
        self.n_features_in_ = int(n_features)
        self.n_components_ = int(self.components_.shape[0])
        self.fit_method = "streaming_covariance"

    def transform(self, X):
        X = np.asarray(X)
        return (X - self.mean_) @ self.components_.T


def _remove_suffix(text, suffix):
    """Python 3.8-compatible ``str.removesuffix``."""
    text = str(text)
    return text[:-len(suffix)] if suffix and text.endswith(suffix) else text


def _subsample_rows(A, factor):
    """Return every ``factor``-th row; factor <= 1 keeps all rows."""
    factor = 1 if factor is None else int(factor)
    if factor <= 1:
        return np.asarray(A)
    return np.asarray(A[::factor])


def _wavelet_frequencies(fmin, fmax, n_freqs):
    """Return the dyadically spaced Morlet center frequencies used by pipeline.py."""
    Tmin, Tmax = 1.0 / fmax, 1.0 / fmin
    Ts = Tmin * (2 ** ((np.arange(n_freqs) * np.log(Tmax / Tmin))
                       / (np.log(2) * (n_freqs - 1))))
    return (1.0 / Ts)[::-1]


def _wavelet_feature_blocks(x, *, fs, fmin, fmax, n_freqs, omega0=5.0,
                            dtype=np.float32):
    """Yield ``(feature_start, amplitudes)`` without materializing all features."""
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    freqs = _wavelet_frequencies(fmin, fmax, n_freqs)
    dt = 1.0 / fs
    for c in range(x.shape[1]):
        block = pp._wavelet_one_channel(x[:, c], freqs, dt, omega0).T
        yield c * n_freqs, block.astype(dtype, copy=False)


def _checkpoint_meta_path(path):
    path = Path(path)
    return path.with_suffix(path.suffix + ".pkl")


def _configs_match(saved_config, expected_config):
    """Compare checkpoint configs while tolerating removed legacy keys."""
    saved_config = dict(saved_config or {})
    expected_config = dict(expected_config or {})
    for legacy_key in ("stride", "max_frames_per_recording"):
        saved_config.pop(legacy_key, None)
        expected_config.pop(legacy_key, None)
    return saved_config == expected_config


def _sample_file_matches(path, *, n_rows, n_features, config=None):
    """Return True when a saved PCA-training wavelet sample can be reused."""
    path = Path(path)
    if not path.exists():
        return False
    if config is not None:
        meta_path = _checkpoint_meta_path(path)
        if not meta_path.exists():
            return False
        try:
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
        except (OSError, pickle.PickleError, EOFError):
            return False
        if not _configs_match(meta.get("config"), config):
            return False
    try:
        arr = np.load(path, mmap_mode="r")
    except (OSError, ValueError):
        return False
    return arr.shape == (int(n_rows), int(n_features)) and arr.dtype == np.float32


def _load_pca_checkpoint(path, config):
    """Load a fitted PCA checkpoint when its config matches the current run."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            checkpoint = pickle.load(f)
    except (OSError, pickle.PickleError, EOFError):
        return None
    if not _configs_match(checkpoint.get("config"), config):
        return None
    required = {"pca", "n_kept", "eigvals_pca", "shuffle_threshold"}
    if not required.issubset(checkpoint):
        return None
    return checkpoint


def _save_pca_checkpoint_atomic(path, checkpoint):
    """Atomically save the fitted PCA stage so projection restarts skip PCA."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.stem}.tmp{path.suffix}")
    try:
        with open(tmp_path, "wb") as f:
            pickle.dump(checkpoint, f)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _delay_embed_subsample(X, d, factor, tau=1):
    """Build only the delay-embedded rows used to fit shared k-means."""
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:, None]
    d = int(d)
    tau = int(tau)
    factor = max(1, int(factor))
    n_rows = X.shape[0] - (d - 1) * tau
    if n_rows <= 0:
        raise ValueError(
            f"Projection has {X.shape[0]} rows, too short for d={d}, tau={tau}"
        )
    row_idx = np.arange(0, n_rows, factor)
    sample = np.empty((len(row_idx), d * X.shape[1]), dtype=X.dtype)
    for k in range(d):
        sample[:, k * X.shape[1]:(k + 1) * X.shape[1]] = X[row_idx + k * tau]
    return sample


def _predict_delay_embedded_in_chunks(km, X, d, tau=1, chunk_rows=250_000):
    """Predict cluster states without materializing a full embedded recording."""
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:, None]
    d = int(d)
    tau = int(tau)
    n_rows = X.shape[0] - (d - 1) * tau
    if n_rows <= 0:
        raise ValueError(
            f"Projection has {X.shape[0]} rows, too short for d={d}, tau={tau}"
        )
    states = np.empty(n_rows, dtype=np.int32)
    chunk_rows = max(1, int(chunk_rows))
    context = (d - 1) * tau
    for start in range(0, n_rows, chunk_rows):
        stop = min(start + chunk_rows, n_rows)
        block = pp.delay_embed(X[start:stop + context], d=d, tau=tau)
        states[start:stop] = km.predict(block).astype(np.int32, copy=False)
        del block
    return states


def fit_pca_with_shuffle_threshold_sample(A, n_components=20, n_shuffles=10,
                                          seed=0, percentile=None):
    """Fit PCA and choose kept PCs using the notebook shuffle threshold."""
    rng = np.random.default_rng(seed)
    A = np.asarray(A, dtype=np.float32)
    pca = PCA(n_components=min(n_components, A.shape[1]), svd_solver="randomized",
              random_state=seed)
    pca.fit(A)
    eigvals = pca.explained_variance_

    shuf_lambdas = np.zeros(n_shuffles, dtype=float)
    for s in range(n_shuffles):
        Ash = A.copy()
        for j in range(Ash.shape[1]):
            rng.shuffle(Ash[:, j])
        pca_sh = PCA(n_components=1, svd_solver="randomized", random_state=seed + s + 1)
        pca_sh.fit(Ash)
        shuf_lambdas[s] = pca_sh.explained_variance_[0]

    if percentile is None:
        threshold = float(shuf_lambdas.mean())
    else:
        threshold = float(np.percentile(shuf_lambdas, percentile))
    n_kept = int(np.sum(eigvals > threshold))
    n_kept = max(n_kept, 1)
    return pca, n_kept, eigvals, threshold


def _fit_pca_only_sample(A, n_components=20, seed=0):
    """Fit the real-data PCA before the shuffle-threshold stage."""
    A = np.asarray(A, dtype=np.float32)
    pca = PCA(n_components=min(n_components, A.shape[1]), svd_solver="randomized",
              random_state=seed)
    pca.fit(A)
    return pca, pca.explained_variance_


def _shuffle_threshold_sample(A, n_shuffles=10, seed=0, percentile=None):
    """Compute the leading shuffled-PC threshold for an already-fit PCA."""
    rng = np.random.default_rng(seed)
    A = np.asarray(A, dtype=np.float32)
    shuf_lambdas = np.zeros(n_shuffles, dtype=float)
    for s in range(n_shuffles):
        Ash = A.copy()
        for j in range(Ash.shape[1]):
            rng.shuffle(Ash[:, j])
        pca_sh = PCA(n_components=1, svd_solver="randomized", random_state=seed + s + 1)
        pca_sh.fit(Ash)
        shuf_lambdas[s] = pca_sh.explained_variance_[0]
        del Ash, pca_sh
        gc.collect()
    if percentile is None:
        return float(shuf_lambdas.mean()), shuf_lambdas
    return float(np.percentile(shuf_lambdas, percentile)), shuf_lambdas


def _n_components_for_variance(pca, target):
    """Return components needed to explain ``target`` cumulative PCA variance."""
    target = float(target)
    if not (0 < target <= 1):
        raise ValueError(f"pca_variance_target must be in (0, 1], got {target}")
    cumulative = np.cumsum(np.asarray(pca.explained_variance_ratio_, dtype=float))
    n_kept = int(np.searchsorted(cumulative, target, side="left")) + 1
    n_kept = min(max(n_kept, 1), len(cumulative))
    return n_kept, bool(cumulative[n_kept - 1] >= target), cumulative


def _merge_covariance_stats(n_a, mean_a, M_a, n_b, mean_b, M_b):
    """Merge unnormalized covariance sums with a batch-mean correction."""
    if n_a == 0:
        return int(n_b), mean_b, M_b
    if n_b == 0:
        return int(n_a), mean_a, M_a
    n_total = int(n_a) + int(n_b)
    delta = mean_b - mean_a
    mean = mean_a + delta * (float(n_b) / float(n_total))
    M = M_a + M_b + (float(n_a) * float(n_b) / float(n_total)) * np.outer(delta, delta)
    return n_total, mean, M


def _fit_streaming_covariance_pca(
    batches,
    *,
    n_components,
    verbose=False,
    label="PCA",
):
    """Fit exact PCA from streamed batches by accumulating covariance stats."""
    n_seen = 0
    mean = None
    M = None
    n_features = None
    for batch_index, X in enumerate(batches, start=1):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"{label} batch {batch_index} must be 2D, got {X.shape}")
        if X.shape[0] == 0:
            continue
        if n_features is None:
            n_features = int(X.shape[1])
        elif X.shape[1] != n_features:
            raise ValueError(
                f"{label} batch {batch_index}: expected {n_features} features, "
                f"got {X.shape[1]}"
            )
        batch_n = int(X.shape[0])
        batch_mean = X.mean(axis=0)
        centered = X - batch_mean
        batch_M = centered.T @ centered
        n_seen, mean, M = _merge_covariance_stats(
            n_seen,
            mean,
            M,
            batch_n,
            batch_mean,
            batch_M,
        )
        if verbose:
            print(f"    covariance batch {batch_index}: {X.shape}, rows seen={n_seen:,}")
        del X, centered, batch_M
        gc.collect()
    if n_seen < 2 or n_features is None:
        raise ValueError(f"{label} covariance PCA needs at least two sampled rows")
    n_components = min(int(n_components), int(n_features), int(n_seen))
    covariance = M / float(n_seen - 1)
    eigvals, eigvecs = np.linalg.eigh(covariance)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]
    components = eigvecs[:, :n_components].T
    explained = eigvals[:n_components]
    total_variance = float(np.maximum(eigvals.sum(), np.finfo(float).eps))
    explained_ratio = explained / total_variance
    singular_values = np.sqrt(explained * float(n_seen - 1))
    if verbose:
        print(
            f"  {label} streaming covariance rows = {n_seen:,}, "
            f"features = {n_features}, components = {n_components}"
        )
    return CovariancePCA(
        components=components,
        mean=mean,
        explained_variance=explained,
        explained_variance_ratio=explained_ratio,
        singular_values=singular_values,
        n_samples=n_seen,
        n_features=n_features,
    )


def fit_streaming_covariance_pca(
    batches,
    *,
    n_components,
    verbose=False,
    label="PCA",
):
    """Public wrapper for streamed full-covariance PCA fitting."""
    return _fit_streaming_covariance_pca(
        batches,
        n_components=n_components,
        verbose=verbose,
        label=label,
    )


def _fit_wavelet_normalizer(A, eps=1e-8):
    """Configure per-frame L1 normalization of wavelet amplitudes."""
    return {"mode": "frame_l1", "eps": float(eps)}


def _apply_wavelet_normalizer(A, normalizer):
    """Normalize each frame by its total wavelet amplitude."""
    if normalizer is None:
        return A
    A = np.asarray(A, dtype=np.float32)
    if normalizer.get("mode") != "frame_l1":
        raise ValueError(f"Unsupported wavelet normalization: {normalizer!r}")
    scale = np.nansum(np.abs(A), axis=1, keepdims=True)
    scale = np.where(scale > normalizer["eps"], scale, 1.0)
    return A / scale


def _projection_file_matches(path, *, n_rows, n_components):
    """Return True when a saved projection can be reused for this run."""
    path = Path(path)
    if not path.exists():
        return False
    try:
        arr = np.load(path, mmap_mode="r")
    except (OSError, ValueError):
        return False
    return arr.shape == (int(n_rows), int(n_components)) and arr.dtype == np.float32


def _require_free_space(path, required_bytes):
    """Raise a clear error before starting a write that cannot fit."""
    free_bytes = shutil.disk_usage(Path(path)).free
    if free_bytes < required_bytes:
        required_gb = required_bytes / 1024 ** 3
        free_gb = free_bytes / 1024 ** 3
        raise OSError(
            f"Not enough free disk space under {path}: need at least "
            f"{required_gb:.2f} GiB, have {free_gb:.2f} GiB"
        )


def _save_projection_atomic(path, proj):
    """Save a projection via a same-directory temp file, then replace."""
    path = Path(path)
    tmp_path = path.with_name(f".{path.stem}.tmp.npy")
    _require_free_space(path.parent, int(proj.nbytes + 64 * 1024 ** 2))
    try:
        np.save(tmp_path, proj)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _write_wavelet_sample_checkpoint(path, x, *, fs, fmin, fmax, n_freqs,
                                     wavelet_subsample_factor, config,
                                     dtype=np.float32):
    """Save every Nth wavelet row for one individual without a full wavelet matrix."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.stem}.tmp.npy")
    tmp_meta_path = _checkpoint_meta_path(tmp_path)
    x = np.asarray(x)
    row_idx = np.arange(
        0, x.shape[0], max(1, int(wavelet_subsample_factor)), dtype=int
    )
    n_features = x.shape[1] * int(n_freqs)
    required_bytes = len(row_idx) * n_features * np.dtype(dtype).itemsize
    _require_free_space(path.parent, int(required_bytes + 64 * 1024 ** 2))
    try:
        sample = open_memmap(
            tmp_path, mode="w+", dtype=dtype, shape=(len(row_idx), n_features)
        )
        for feature_start, block in _wavelet_feature_blocks(
            x, fs=fs, fmin=fmin, fmax=fmax, n_freqs=n_freqs, dtype=dtype
        ):
            sample[:, feature_start:feature_start + n_freqs] = block[row_idx]
            del block
        sample.flush()
        del sample
        with open(tmp_meta_path, "wb") as f:
            pickle.dump({"config": config}, f)
        tmp_path.replace(path)
        tmp_meta_path.replace(_checkpoint_meta_path(path))
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
        if tmp_meta_path.exists():
            tmp_meta_path.unlink()
    return path


def _project_wavelets_streaming(x, pca, n_kept, *, fs, fmin, fmax, n_freqs,
                                wavelet_normalizer=None, dtype=np.float32):
    """Project wavelets without materializing the full wavelet matrix."""
    x = np.asarray(x)
    n_rows, n_channels = x.shape
    components = pca.components_[:n_kept].astype(dtype, copy=False)
    mean = pca.mean_.astype(dtype, copy=False)
    proj = np.zeros((n_rows, n_kept), dtype=dtype)

    if wavelet_normalizer is not None:
        if wavelet_normalizer.get("mode") != "frame_l1":
            raise ValueError(f"Unsupported wavelet normalization: {wavelet_normalizer!r}")
        scale = np.zeros((n_rows, 1), dtype=dtype)
        for _, block in _wavelet_feature_blocks(
            x, fs=fs, fmin=fmin, fmax=fmax, n_freqs=n_freqs, dtype=dtype
        ):
            scale[:, 0] += np.nansum(np.abs(block), axis=1, dtype=dtype)
            del block
        scale = np.where(scale > wavelet_normalizer["eps"], scale, 1.0).astype(
            dtype, copy=False
        )
    else:
        scale = None

    for feature_start, block in _wavelet_feature_blocks(
        x, fs=fs, fmin=fmin, fmax=fmax, n_freqs=n_freqs, dtype=dtype
    ):
        stop = feature_start + n_freqs
        if scale is not None:
            block = block / scale
        block -= mean[feature_start:stop]
        proj += block @ components[:, feature_start:stop].T
        del block

    # The sklearn PCA transform adds no additional offset after centering.
    if n_channels * int(n_freqs) != pca.components_.shape[1]:
        raise ValueError(
            "PCA feature count does not match streamed wavelet feature count: "
            f"{pca.components_.shape[1]} vs {n_channels * int(n_freqs)}"
        )
    return proj


def _wavelet_features_streaming(x, *, fs, fmin, fmax, n_freqs,
                                wavelet_normalizer=None, dtype=np.float32):
    """Compute wavelet amplitudes as the saved projection representation."""
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    n_rows, n_channels = x.shape
    n_features = n_channels * int(n_freqs)
    proj = np.empty((n_rows, n_features), dtype=dtype)

    if wavelet_normalizer is not None:
        if wavelet_normalizer.get("mode") != "frame_l1":
            raise ValueError(f"Unsupported wavelet normalization: {wavelet_normalizer!r}")
        scale = np.zeros((n_rows, 1), dtype=dtype)
        for _, block in _wavelet_feature_blocks(
            x, fs=fs, fmin=fmin, fmax=fmax, n_freqs=n_freqs, dtype=dtype
        ):
            scale[:, 0] += np.nansum(np.abs(block), axis=1, dtype=dtype)
            del block
        scale = np.where(scale > wavelet_normalizer["eps"], scale, 1.0).astype(
            dtype, copy=False
        )
    else:
        scale = None

    for feature_start, block in _wavelet_feature_blocks(
        x, fs=fs, fmin=fmin, fmax=fmax, n_freqs=n_freqs, dtype=dtype
    ):
        stop = feature_start + n_freqs
        if scale is not None:
            block = block / scale
        proj[:, feature_start:stop] = block
        del block
    return proj


def _collect_wavelet_samples_and_metadata(
    *,
    pipeline_species,
    individual_ids,
    load_individual_trace_fn,
    dataset_name,
    fs,
    fmin,
    fmax,
    n_freqs,
    wavelet_subsample_factor,
    wavelet_dtype,
    sample_dir,
    wavelet_sample_config,
    reuse_existing_wavelet_samples,
    save_wavelet_samples,
    collect_samples,
    verbose,
):
    """Collect per-individual metadata and optional sampled wavelet rows."""
    sample_files = []
    in_memory_samples = []
    metadata = []
    for i, individual_id in enumerate(individual_ids, start=1):
        bundle = load_individual_trace_fn(
            pipeline_species,
            individual_id=individual_id,
            dataset_name=dataset_name,
        )
        if verbose:
            print(f"  [{i}/{len(individual_ids)}] {individual_id}: X={bundle['X'].shape}")
        n_sample_rows = len(
            np.arange(
                0,
                int(bundle["X"].shape[0]),
                max(1, int(wavelet_subsample_factor)),
                dtype=int,
            )
        )
        n_features = int(bundle["X"].shape[1]) * int(n_freqs)
        sample_file = sample_dir / f"{individual_id}_wavelet_sample.npy"
        if collect_samples:
            if save_wavelet_samples and reuse_existing_wavelet_samples and _sample_file_matches(
                sample_file,
                n_rows=n_sample_rows,
                n_features=n_features,
                config={**wavelet_sample_config, "individual_id": individual_id},
            ):
                if verbose:
                    print(f"    reusing sampled-wavelet checkpoint: {sample_file.name}")
            elif save_wavelet_samples:
                _write_wavelet_sample_checkpoint(
                    sample_file,
                    bundle["X"],
                    fs=fs,
                    fmin=fmin,
                    fmax=fmax,
                    n_freqs=n_freqs,
                    wavelet_subsample_factor=wavelet_subsample_factor,
                    config={**wavelet_sample_config, "individual_id": individual_id},
                    dtype=wavelet_dtype,
                )
            else:
                sample = np.empty((n_sample_rows, n_features), dtype=wavelet_dtype)
                row_idx = np.arange(
                    0,
                    int(bundle["X"].shape[0]),
                    max(1, int(wavelet_subsample_factor)),
                    dtype=int,
                )
                for feature_start, block in _wavelet_feature_blocks(
                    bundle["X"],
                    fs=fs,
                    fmin=fmin,
                    fmax=fmax,
                    n_freqs=n_freqs,
                    dtype=wavelet_dtype,
                ):
                    sample[:, feature_start:feature_start + n_freqs] = block[row_idx]
                    del block
                in_memory_samples.append(sample)
        if collect_samples and save_wavelet_samples:
            sample_files.append(str(sample_file))
        metadata.append({
            "individual_id": individual_id,
            "recording_ids": bundle["recording_ids"],
            "segment_lengths": bundle["segment_lengths"],
            "n_frames": int(bundle["X"].shape[0]),
            "n_wavelet_sample_rows": int(n_sample_rows),
            "n_wavelet_features": int(n_features),
        })
        del bundle
        gc.collect()
    return metadata, sample_files, in_memory_samples


def _balanced_pca_sample_plan(metadata, species_labels, *, rows_per_species=None):
    """Return per-individual PCA sample rows with equal total rows per species."""
    if species_labels is None:
        return None
    if len(species_labels) != len(metadata):
        raise ValueError(
            "species_labels must have one entry per individual: "
            f"{len(species_labels)} labels for {len(metadata)} metadata rows"
        )
    plan = pd.DataFrame({
        "individual_id": [str(meta["individual_id"]) for meta in metadata],
        "species": [str(value) for value in species_labels],
        "available_sample_rows": [int(meta.get("n_wavelet_sample_rows", 0)) for meta in metadata],
    })
    species_available = plan.groupby("species")["available_sample_rows"].sum().sort_index()
    if species_available.empty:
        raise ValueError("No species are available for balanced PCA sampling")
    if (species_available <= 0).any():
        empty = species_available[species_available <= 0].index.tolist()
        raise ValueError(f"Species with no PCA sample rows: {empty}")
    target_rows = int(species_available.min() if rows_per_species is None else rows_per_species)
    target_rows = max(1, target_rows)
    plan["species_sample_budget"] = plan["species"].map(
        lambda species: min(target_rows, int(species_available.loc[species]))
    ).astype(int)
    plan["pca_fit_sample_rows"] = 0
    for species, sub in plan.groupby("species", sort=True):
        species_total = int(sub["available_sample_rows"].sum())
        species_budget = min(target_rows, species_total)
        exact = sub["available_sample_rows"].to_numpy(float) * species_budget / species_total
        take = np.floor(exact).astype(int)
        shortfall = int(species_budget - take.sum())
        if shortfall > 0:
            order = np.argsort(-(exact - take), kind="mergesort")[:shortfall]
            take[order] += 1
        take = np.minimum(take, sub["available_sample_rows"].to_numpy(int))
        shortfall = int(species_budget - take.sum())
        if shortfall > 0:
            spare = sub["available_sample_rows"].to_numpy(int) - take
            for local_idx in np.argsort(-spare, kind="mergesort"):
                if shortfall <= 0 or spare[local_idx] <= 0:
                    break
                add = min(shortfall, int(spare[local_idx]))
                take[local_idx] += add
                shortfall -= add
        plan.loc[sub.index, "pca_fit_sample_rows"] = take
    return plan


def _row_limit_values(pca_sample_plan):
    if pca_sample_plan is None:
        return None
    return pca_sample_plan["pca_fit_sample_rows"].astype(int).to_numpy()


def _select_pca_sample_rows(A, row_limit, rng):
    A = np.asarray(A, dtype=np.float32)
    if row_limit is None:
        return A
    row_limit = int(row_limit)
    if row_limit >= A.shape[0]:
        return A
    if row_limit <= 0:
        return np.empty((0, A.shape[1]), dtype=np.float32)
    rows = np.sort(rng.choice(A.shape[0], size=row_limit, replace=False))
    return np.asarray(A[rows], dtype=np.float32)


def _load_wavelet_pca_sample(
    sample_files,
    in_memory_samples,
    *,
    save_wavelet_samples,
    row_limits=None,
    seed=0,
):
    """Materialize the pooled sampled-wavelet matrix for fitting PCA."""
    rng = np.random.default_rng(int(seed))
    sources = sample_files if save_wavelet_samples else in_memory_samples
    if row_limits is None:
        row_limits = [None] * len(sources)
    amp_samples = []
    for source, row_limit in zip(sources, row_limits):
        A = np.load(source, mmap_mode="r") if save_wavelet_samples else source
        sample = _select_pca_sample_rows(A, row_limit, rng)
        if sample.shape[0]:
            amp_samples.append(sample)
    if not amp_samples:
        raise ValueError("No wavelet PCA sample rows selected")
    A_sample = np.concatenate(amp_samples, axis=0).astype(np.float32, copy=False)
    del amp_samples
    return A_sample


def _iter_wavelet_pca_sample_batches(sample_files, in_memory_samples, *,
                                     save_wavelet_samples, normalizer=None,
                                     min_batch_rows=1, row_limits=None, seed=0):
    """Yield sampled wavelet rows without materializing the pooled sample."""
    pending = []
    pending_rows = 0
    rng = np.random.default_rng(int(seed))
    sources = sample_files if save_wavelet_samples else in_memory_samples
    if row_limits is None:
        row_limits = [None] * len(sources)
    for source, row_limit in zip(sources, row_limits):
        A = np.load(source, mmap_mode="r") if save_wavelet_samples else source
        A = _select_pca_sample_rows(A, row_limit, rng)
        if A.shape[0] == 0:
            continue
        if normalizer is not None:
            A = _apply_wavelet_normalizer(A, normalizer)
        pending.append(A)
        pending_rows += A.shape[0]
        if pending_rows >= int(min_batch_rows):
            batch = np.concatenate(pending, axis=0).astype(np.float32, copy=False)
            pending.clear()
            pending_rows = 0
            yield batch
            del batch
            gc.collect()
    if pending:
        batch = np.concatenate(pending, axis=0).astype(np.float32, copy=False)
        pending.clear()
        yield batch
        del batch
        gc.collect()


def _fit_covariance_pca_wavelet_samples(
    *,
    sample_files,
    in_memory_samples,
    save_wavelet_samples,
    n_components,
    wavelet_normalizer,
    pca_sample_plan,
    seed,
    verbose,
):
    """Fit PCA on sampled wavelet rows by streaming exact covariance stats."""
    n_features = None
    n_rows = 0
    row_limits = _row_limit_values(pca_sample_plan)
    sources = sample_files if save_wavelet_samples else in_memory_samples
    for index, source in enumerate(sources):
        A = np.load(source, mmap_mode="r") if save_wavelet_samples else source
        rows_available = int(A.shape[0])
        rows_used = rows_available if row_limits is None else min(int(row_limits[index]), rows_available)
        n_rows += rows_used
        n_features = int(A.shape[1]) if n_features is None else n_features
    if n_features is None:
        raise ValueError("No wavelet PCA samples are available")
    n_components = min(int(n_components), int(n_features), int(n_rows))
    if n_components < 1:
        raise ValueError("Wavelet PCA needs at least one sampled row")
    min_batch_rows = max(n_components, 2 * n_components)
    if verbose:
        print(
            f"  covariance PCA sample rows = {n_rows}, "
            f"features = {n_features}, components = {n_components}"
        )
    batches = _iter_wavelet_pca_sample_batches(
        sample_files,
        in_memory_samples,
        save_wavelet_samples=save_wavelet_samples,
        normalizer=wavelet_normalizer,
        min_batch_rows=min_batch_rows,
        row_limits=row_limits,
        seed=seed,
    )
    pca = _fit_streaming_covariance_pca(
        batches,
        n_components=n_components,
        verbose=verbose,
        label="post-wavelet PCA",
    )
    return pca, pca.explained_variance_, int(n_rows)


def _fit_or_load_wavelet_pca(
    *,
    A_sample,
    in_memory_samples,
    pca_checkpoint,
    pca_checkpoint_path,
    pca_fit_checkpoint_path,
    pca_checkpoint_config,
    pipeline_species,
    dataset_name,
    individual_ids,
    metadata,
    freqs,
    output_dir,
    sample_files,
    save_wavelet_samples,
    pca_sample_plan,
    balance_pca_samples_by_species,
    pca_rows_per_species,
    wavelet_subsample_factor,
    normalize_wavelets_for_pca,
    pca_max_components,
    pca_variance_target,
    n_shuffles,
    seed,
    verbose,
):
    """Fit/reuse the post-wavelet PCA stage and choose retained components."""
    if pca_checkpoint is not None:
        pca = pca_checkpoint["pca"]
        n_kept = int(pca_checkpoint["n_kept"])
        eigvals_pca = pca_checkpoint["eigvals_pca"]
        thresh = float(pca_checkpoint["shuffle_threshold"])
        pca_selection_mode = pca_checkpoint.get("pca_selection_mode", "shuffle_threshold")
        pca_variance_target_reached = pca_checkpoint.get("pca_variance_target_reached")
        pca_cumulative_variance = pca_checkpoint.get("pca_cumulative_variance")
        wavelet_normalizer = pca_checkpoint.get("wavelet_normalizer")
        if verbose:
            print(f"  reusing PCA checkpoint: {pca_checkpoint_path.name}")
            _print_pca_selection_summary(
                pca_selection_mode,
                n_kept,
                thresh,
                pca_variance_target,
                pca_variance_target_reached,
                pca_cumulative_variance,
            )
        return {
            "pca": pca,
            "n_kept": n_kept,
            "eigvals_pca": eigvals_pca,
            "shuffle_threshold": thresh,
            "shuffle_lambdas": pca_checkpoint.get("shuffle_lambdas"),
            "pca_selection_mode": pca_selection_mode,
            "pca_variance_target_reached": pca_variance_target_reached,
            "pca_cumulative_variance": pca_cumulative_variance,
            "wavelet_normalizer": wavelet_normalizer,
        }

    wavelet_normalizer = (
        _fit_wavelet_normalizer(None) if normalize_wavelets_for_pca else None
    )

    pca_fit_checkpoint = _load_pca_checkpoint(
        pca_fit_checkpoint_path, pca_checkpoint_config
    )
    if pca_fit_checkpoint is not None:
        pca = pca_fit_checkpoint["pca"]
        eigvals_pca = pca_fit_checkpoint["eigvals_pca"]
        pca_fit_rows = pca_fit_checkpoint.get("pca_fit_rows")
        if verbose:
            print(f"  reusing PCA fit checkpoint: {pca_fit_checkpoint_path.name}")
    else:
        pca_fit_rows = None

    if pca_fit_checkpoint is None:
        if pca_variance_target is not None:
            pca, eigvals_pca, pca_fit_rows = (
                _fit_covariance_pca_wavelet_samples(
                    sample_files=sample_files,
                    in_memory_samples=in_memory_samples,
                    save_wavelet_samples=save_wavelet_samples,
                    n_components=pca_max_components,
                    wavelet_normalizer=wavelet_normalizer,
                    pca_sample_plan=pca_sample_plan,
                    seed=seed,
                    verbose=verbose,
                )
            )
        else:
            if A_sample is None:
                A_sample = _load_wavelet_pca_sample(
                    sample_files,
                    in_memory_samples,
                    save_wavelet_samples=save_wavelet_samples,
                    row_limits=_row_limit_values(pca_sample_plan),
                    seed=seed,
                )
            if normalize_wavelets_for_pca:
                A_sample = _apply_wavelet_normalizer(A_sample, wavelet_normalizer)
            if verbose:
                suffix = " frame-normalized" if normalize_wavelets_for_pca else ""
                print(f"  PCA sample matrix = {A_sample.shape}{suffix}")
            pca, eigvals_pca = _fit_pca_only_sample(
                A_sample,
                n_components=pca_max_components,
                seed=seed,
            )
            pca_fit_rows = int(A_sample.shape[0])
        _save_pca_checkpoint_atomic(
            pca_fit_checkpoint_path,
            {
                "config": pca_checkpoint_config,
                "pipeline_species": pipeline_species,
                "dataset_name": dataset_name,
                "individual_ids": individual_ids,
                "metadata": metadata,
                "frequencies": freqs,
                "pca": pca,
                "n_kept": 1,
                "eigvals_pca": eigvals_pca,
                "shuffle_threshold": np.nan,
                "pca_selection_mode": (
                    "variance_target" if pca_variance_target is not None
                    else "shuffle_threshold"
                ),
                "pca_variance_target": (
                    float(pca_variance_target)
                    if pca_variance_target is not None else None
                ),
                "pca_variance_target_reached": None,
                "pca_cumulative_variance": np.cumsum(
                    np.asarray(pca.explained_variance_ratio_, dtype=float)
                ),
                "wavelet_subsample_factor": int(wavelet_subsample_factor),
                "normalize_wavelets_for_pca": bool(normalize_wavelets_for_pca),
                "wavelet_normalization_mode": (
                    "frame_l1" if normalize_wavelets_for_pca else None
                ),
                "wavelet_normalizer": wavelet_normalizer,
                "wavelet_sample_files": sample_files,
                "save_wavelet_samples": bool(save_wavelet_samples),
                "balance_pca_samples_by_species": bool(balance_pca_samples_by_species),
                "pca_rows_per_species": (
                    None if pca_rows_per_species is None else int(pca_rows_per_species)
                ),
                "pca_sample_plan": (
                    None if pca_sample_plan is None else pca_sample_plan.to_dict("records")
                ),
                "pca_fit_method": getattr(pca, "fit_method", "full_sample_pca"),
                "pca_fit_rows": pca_fit_rows,
                "output_dir": str(output_dir),
            },
        )
        if verbose:
            print(f"  saved PCA fit checkpoint: {pca_fit_checkpoint_path}")

    pca_cumulative_variance = np.cumsum(
        np.asarray(pca.explained_variance_ratio_, dtype=float)
    )
    if pca_variance_target is not None:
        n_kept, pca_variance_target_reached, pca_cumulative_variance = (
            _n_components_for_variance(pca, pca_variance_target)
        )
        thresh = np.nan
        shuf_lambdas = None
        pca_selection_mode = "variance_target"
    else:
        if A_sample is None:
            A_sample = _load_wavelet_pca_sample(
                sample_files,
                in_memory_samples,
                save_wavelet_samples=save_wavelet_samples,
                row_limits=_row_limit_values(pca_sample_plan),
                seed=seed,
            )
            if normalize_wavelets_for_pca:
                A_sample = _apply_wavelet_normalizer(A_sample, wavelet_normalizer)
        thresh, shuf_lambdas = _shuffle_threshold_sample(
            A_sample,
            n_shuffles=n_shuffles,
            seed=seed,
        )
        n_kept = int(np.sum(eigvals_pca > thresh))
        n_kept = max(n_kept, 1)
        pca_variance_target_reached = None
        pca_selection_mode = "shuffle_threshold"
    if A_sample is not None:
        del A_sample
    gc.collect()

    _save_pca_checkpoint_atomic(
        pca_checkpoint_path,
        {
            "config": pca_checkpoint_config,
            "pipeline_species": pipeline_species,
            "dataset_name": dataset_name,
            "individual_ids": individual_ids,
            "metadata": metadata,
            "frequencies": freqs,
            "pca": pca,
            "n_kept": n_kept,
            "eigvals_pca": eigvals_pca,
            "shuffle_threshold": thresh,
            "shuffle_lambdas": shuf_lambdas,
            "pca_selection_mode": pca_selection_mode,
            "pca_variance_target": (
                float(pca_variance_target)
                if pca_variance_target is not None else None
            ),
            "pca_variance_target_reached": pca_variance_target_reached,
            "pca_cumulative_variance": pca_cumulative_variance,
            "wavelet_subsample_factor": int(wavelet_subsample_factor),
            "normalize_wavelets_for_pca": bool(normalize_wavelets_for_pca),
            "wavelet_normalization_mode": (
                "frame_l1" if normalize_wavelets_for_pca else None
            ),
            "wavelet_normalizer": wavelet_normalizer,
            "wavelet_sample_files": sample_files,
            "save_wavelet_samples": bool(save_wavelet_samples),
            "balance_pca_samples_by_species": bool(balance_pca_samples_by_species),
            "pca_rows_per_species": (
                None if pca_rows_per_species is None else int(pca_rows_per_species)
            ),
            "pca_sample_plan": (
                None if pca_sample_plan is None else pca_sample_plan.to_dict("records")
            ),
            "pca_fit_method": getattr(pca, "fit_method", "full_sample_pca"),
            "pca_fit_rows": (
                pca_fit_checkpoint.get("pca_fit_rows")
                if pca_fit_checkpoint is not None
                else pca_fit_rows
            ),
            "output_dir": str(output_dir),
        },
    )
    if verbose:
        _print_pca_selection_summary(
            pca_selection_mode,
            n_kept,
            thresh,
            pca_variance_target,
            pca_variance_target_reached,
            pca_cumulative_variance,
        )
        print(f"  saved PCA checkpoint: {pca_checkpoint_path}")
    return {
        "pca": pca,
        "n_kept": n_kept,
        "eigvals_pca": eigvals_pca,
        "shuffle_threshold": thresh,
        "shuffle_lambdas": shuf_lambdas,
        "pca_selection_mode": pca_selection_mode,
        "pca_variance_target_reached": pca_variance_target_reached,
        "pca_cumulative_variance": pca_cumulative_variance,
        "wavelet_normalizer": wavelet_normalizer,
    }


def _print_pca_selection_summary(
    pca_selection_mode,
    n_kept,
    thresh,
    pca_variance_target,
    pca_variance_target_reached,
    pca_cumulative_variance,
):
    """Print a concise summary of the selected wavelet-PCA dimensionality."""
    if pca_selection_mode == "variance_target":
        explained = (
            pca_cumulative_variance[n_kept - 1]
            if pca_cumulative_variance is not None else np.nan
        )
        status = "reached" if pca_variance_target_reached else "not reached"
        print(f"  kept {n_kept} PCs for {float(pca_variance_target):.1%} variance ({explained:.1%}, {status})")
    elif pca_selection_mode == "none":
        print(f"  kept raw wavelet features ({n_kept} dimensions); no post-wavelet PCA")
    else:
        print(f"  kept {n_kept} PCs above threshold {thresh:.3g}")


def _write_projection_files(
    *,
    individual_ids,
    metadata,
    proj_dir,
    load_individual_trace_fn,
    pipeline_species,
    dataset_name,
    fs,
    fmin,
    fmax,
    n_freqs,
    n_kept,
    pca,
    wavelet_normalizer,
    wavelet_dtype,
    wavelet_projection_mode,
    reuse_existing_projections,
    verbose,
):
    """Write per-individual projection files for the selected wavelet mode."""
    if verbose:
        if wavelet_projection_mode == "raw_wavelets":
            print("Pass 2/2: stream wavelet blocks and save raw wavelet features")
        else:
            print("Pass 2/2: stream wavelet blocks and project every frame onto shared PCA")
    proj_files = []
    metadata_by_id = {m["individual_id"]: m for m in metadata}
    for i, individual_id in enumerate(individual_ids, start=1):
        proj_file = proj_dir / f"{individual_id}_proj.npy"
        expected_rows = metadata_by_id[individual_id]["n_frames"]
        if reuse_existing_projections and _projection_file_matches(
            proj_file, n_rows=expected_rows, n_components=n_kept
        ):
            if verbose:
                print(f"  [{i}/{len(individual_ids)}] {individual_id}: reusing saved projection")
            proj_files.append(str(proj_file))
            continue

        bundle = load_individual_trace_fn(
            pipeline_species,
            individual_id=individual_id,
            dataset_name=dataset_name,
        )
        if verbose:
            print(f"  [{i}/{len(individual_ids)}] {individual_id}: projecting")
        if wavelet_projection_mode == "raw_wavelets":
            proj = _wavelet_features_streaming(
                bundle["X"],
                fs=fs,
                fmin=fmin,
                fmax=fmax,
                n_freqs=n_freqs,
                wavelet_normalizer=wavelet_normalizer,
                dtype=wavelet_dtype,
            )
        else:
            proj = _project_wavelets_streaming(
                bundle["X"],
                pca,
                n_kept,
                fs=fs,
                fmin=fmin,
                fmax=fmax,
                n_freqs=n_freqs,
                wavelet_normalizer=wavelet_normalizer,
                dtype=wavelet_dtype,
            )
        _save_projection_atomic(proj_file, proj)
        proj_files.append(str(proj_file))
        del proj, bundle
        gc.collect()
    return proj_files


def prepare_pooled_individual_projections(
    *,
    pipeline_species,
    individual_ids,
    load_individual_trace_fn,
    dataset_name,
    fs,
    fmin,
    fmax,
    n_freqs,
    output_dir="outputs/pooled_by_individual",
    wavelet_subsample_factor=20,
    pca_max_components=20,
    n_shuffles=10,
    pca_variance_target=None,
    wavelet_dtype=np.float32,
    normalize_wavelets_for_pca=False,
    balance_pca_samples_by_species=False,
    pca_sample_species_labels=None,
    pca_rows_per_species=None,
    reuse_existing_wavelet_samples=True,
    reuse_existing_projections=True,
    save_wavelet_samples=True,
    wavelet_projection_mode="pca",
    seed=0,
    verbose=True,
):
    """Compute shared PCA basis and per-individual projected time series.

    This is the expensive wavelet stage.  It does not choose ``d_embed``, ``N``,
    ``tau_seconds``, or ``M``; those are tuned later from the saved projections
    and state sequences.
    """
    wavelet_projection_mode = str(wavelet_projection_mode)
    valid_modes = {"pca", "raw_wavelets"}
    if wavelet_projection_mode not in valid_modes:
        raise ValueError(
            f"wavelet_projection_mode must be one of {sorted(valid_modes)}, "
            f"got {wavelet_projection_mode!r}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    proj_dir = output_dir / "projections"
    proj_dir.mkdir(exist_ok=True)
    sample_dir = output_dir / "wavelet_samples"
    collect_samples = wavelet_projection_mode == "pca"
    if save_wavelet_samples and collect_samples:
        sample_dir.mkdir(exist_ok=True)

    individual_ids = [str(x) for x in individual_ids]
    wavelet_sample_config = {
        "pipeline_species": str(pipeline_species),
        "dataset_name": str(dataset_name),
        "fs": float(fs),
        "fmin": float(fmin),
        "fmax": float(fmax),
        "n_freqs": int(n_freqs),
        "wavelet_subsample_factor": max(1, int(wavelet_subsample_factor)),
        "wavelet_dtype": str(np.dtype(wavelet_dtype)),
    }
    if pca_sample_species_labels is not None:
        pca_sample_species_labels = [str(value) for value in pca_sample_species_labels]
    if balance_pca_samples_by_species and pca_sample_species_labels is None:
        pca_sample_species_labels = [
            str(individual_id).split("__subject__", 1)[0]
            for individual_id in individual_ids
        ]
    pca_checkpoint_config = {
        **wavelet_sample_config,
        "individual_ids": individual_ids,
        "pca_max_components": int(pca_max_components),
        "n_shuffles": int(n_shuffles),
        "normalize_wavelets_for_pca": bool(normalize_wavelets_for_pca),
        "balance_pca_samples_by_species": bool(balance_pca_samples_by_species),
        "pca_sample_species_labels": (
            pca_sample_species_labels if balance_pca_samples_by_species else None
        ),
        "pca_rows_per_species": (
            None if pca_rows_per_species is None else int(pca_rows_per_species)
        ),
        "pca_fit_method": "streaming_covariance",
        "save_wavelet_samples": bool(save_wavelet_samples),
        "seed": int(seed),
    }
    if pca_variance_target is not None:
        pca_checkpoint_config["pca_variance_target"] = float(pca_variance_target)
    if wavelet_projection_mode != "pca":
        pca_checkpoint_config["wavelet_projection_mode"] = wavelet_projection_mode
    pca_checkpoint_path = output_dir / "pca_checkpoint.pkl"
    pca_fit_checkpoint_path = output_dir / "pca_fit_checkpoint.pkl"
    pca_checkpoint = None
    if wavelet_projection_mode == "pca":
        pca_checkpoint = _load_pca_checkpoint(
            pca_checkpoint_path, pca_checkpoint_config
        )

    if verbose:
        if wavelet_projection_mode == "raw_wavelets":
            print("Pass 1/2: collect metadata; wavelets will be saved without "
                  "post-wavelet PCA")
        elif pca_checkpoint is not None and not save_wavelet_samples:
            print(f"Pass 1/3: reusing PCA checkpoint; no wavelet sample files saved")
        elif save_wavelet_samples:
            print("Pass 1/3: checkpoint each individual's sampled wavelet rows "
                  f"(every {wavelet_subsample_factor} row)")
        else:
            print("Pass 1/3: stream each individual's sampled wavelet rows into "
                  f"the PCA sample (every {wavelet_subsample_factor} row; no "
                  "wavelet sample files saved)")
    freqs = _wavelet_frequencies(fmin, fmax, n_freqs)
    in_memory_samples = []
    if pca_checkpoint is not None and not save_wavelet_samples:
        metadata = list(pca_checkpoint.get("metadata", []))
        sample_files = list(pca_checkpoint.get("wavelet_sample_files", []))
    else:
        metadata, sample_files, in_memory_samples = _collect_wavelet_samples_and_metadata(
            pipeline_species=pipeline_species,
            individual_ids=individual_ids,
            load_individual_trace_fn=load_individual_trace_fn,
            dataset_name=dataset_name,
            fs=fs,
            fmin=fmin,
            fmax=fmax,
            n_freqs=n_freqs,
            wavelet_subsample_factor=wavelet_subsample_factor,
            wavelet_dtype=wavelet_dtype,
            sample_dir=sample_dir,
            wavelet_sample_config=wavelet_sample_config,
            reuse_existing_wavelet_samples=reuse_existing_wavelet_samples,
            save_wavelet_samples=save_wavelet_samples,
            collect_samples=collect_samples,
            verbose=verbose,
        )
    for meta in metadata:
        if "n_wavelet_sample_rows" not in meta:
            meta["n_wavelet_sample_rows"] = len(
                np.arange(
                    0,
                    int(meta["n_frames"]),
                    max(1, int(wavelet_subsample_factor)),
                    dtype=int,
                )
            )

    pca_sample_plan = None
    if collect_samples and balance_pca_samples_by_species:
        pca_sample_plan = _balanced_pca_sample_plan(
            metadata,
            pca_sample_species_labels,
            rows_per_species=pca_rows_per_species,
        )
        pca_sample_plan.to_csv(output_dir / "post_wavelet_pca_balanced_sample_plan.csv", index=False)
        if verbose:
            summary = (
                pca_sample_plan.groupby("species")[["available_sample_rows", "pca_fit_sample_rows"]]
                .sum()
                .reset_index()
            )
            print("  post-wavelet PCA balanced sample rows by species:")
            print(summary.to_string(index=False))

    if wavelet_projection_mode == "raw_wavelets":
        pca = None
        n_kept = int(metadata[0]["n_wavelet_features"]) if metadata else 0
        eigvals_pca = np.asarray([], dtype=float)
        thresh = np.nan
        pca_selection_mode = "none"
        pca_variance_target_reached = None
        pca_cumulative_variance = None
        wavelet_normalizer = (
            _fit_wavelet_normalizer(None) if normalize_wavelets_for_pca else None
        )
        if verbose:
            _print_pca_selection_summary(
                pca_selection_mode,
                n_kept,
                thresh,
                pca_variance_target,
                pca_variance_target_reached,
                pca_cumulative_variance,
            )
    else:
        A_sample = None
        if pca_checkpoint is None and pca_variance_target is None:
            A_sample = _load_wavelet_pca_sample(
                sample_files,
                in_memory_samples,
                save_wavelet_samples=save_wavelet_samples,
                row_limits=_row_limit_values(pca_sample_plan),
                seed=seed,
            )
        pca_stage = _fit_or_load_wavelet_pca(
            A_sample=A_sample,
            in_memory_samples=in_memory_samples,
            pca_checkpoint=pca_checkpoint,
            pca_checkpoint_path=pca_checkpoint_path,
            pca_fit_checkpoint_path=pca_fit_checkpoint_path,
            pca_checkpoint_config=pca_checkpoint_config,
            pipeline_species=pipeline_species,
            dataset_name=dataset_name,
            individual_ids=individual_ids,
            metadata=metadata,
            freqs=freqs,
            output_dir=output_dir,
            sample_files=sample_files,
            save_wavelet_samples=save_wavelet_samples,
            pca_sample_plan=pca_sample_plan,
            balance_pca_samples_by_species=balance_pca_samples_by_species,
            pca_rows_per_species=pca_rows_per_species,
            wavelet_subsample_factor=wavelet_subsample_factor,
            normalize_wavelets_for_pca=normalize_wavelets_for_pca,
            pca_max_components=pca_max_components,
            pca_variance_target=pca_variance_target,
            n_shuffles=n_shuffles,
            seed=seed,
            verbose=verbose,
        )
        pca = pca_stage["pca"]
        n_kept = pca_stage["n_kept"]
        eigvals_pca = pca_stage["eigvals_pca"]
        thresh = pca_stage["shuffle_threshold"]
        pca_selection_mode = pca_stage["pca_selection_mode"]
        pca_variance_target_reached = pca_stage["pca_variance_target_reached"]
        pca_cumulative_variance = pca_stage["pca_cumulative_variance"]
        wavelet_normalizer = pca_stage["wavelet_normalizer"]

    proj_files = _write_projection_files(
        individual_ids=individual_ids,
        metadata=metadata,
        proj_dir=proj_dir,
        load_individual_trace_fn=load_individual_trace_fn,
        pipeline_species=pipeline_species,
        dataset_name=dataset_name,
        fs=fs,
        fmin=fmin,
        fmax=fmax,
        n_freqs=n_freqs,
        n_kept=n_kept,
        pca=pca,
        wavelet_normalizer=wavelet_normalizer,
        wavelet_dtype=wavelet_dtype,
        wavelet_projection_mode=wavelet_projection_mode,
        reuse_existing_projections=reuse_existing_projections,
        verbose=verbose,
    )

    pca_variance_target_value = (
        float(pca_variance_target)
        if wavelet_projection_mode == "pca" and pca_variance_target is not None
        else None
    )
    wavelet_normalization_mode = (
        "frame_l1" if normalize_wavelets_for_pca else None
    )
    pca_checkpoint_file = (
        str(pca_checkpoint_path) if wavelet_projection_mode == "pca" else None
    )
    saved_wavelet_samples = sample_files if collect_samples else []
    result = {
        "pipeline_species": pipeline_species,
        "dataset_name": dataset_name,
        "individual_ids": individual_ids,
        "metadata": metadata,
        "frequencies": freqs,
        "pca": pca,
        "n_kept": n_kept,
        "eigvals_pca": eigvals_pca,
        "shuffle_threshold": thresh,
        "pca_selection_mode": pca_selection_mode,
        "pca_variance_target": pca_variance_target_value,
        "pca_variance_target_reached": pca_variance_target_reached,
        "pca_cumulative_variance": pca_cumulative_variance,
        "wavelet_projection_mode": wavelet_projection_mode,
        "wavelet_subsample_factor": int(wavelet_subsample_factor),
        "normalize_wavelets_for_pca": bool(normalize_wavelets_for_pca),
        "balance_pca_samples_by_species": bool(balance_pca_samples_by_species and collect_samples),
        "pca_rows_per_species": (
            None if pca_rows_per_species is None else int(pca_rows_per_species)
        ),
        "pca_sample_plan": (
            None if pca_sample_plan is None else pca_sample_plan.to_dict("records")
        ),
        "wavelet_normalization_mode": wavelet_normalization_mode,
        "wavelet_normalizer": wavelet_normalizer,
        "reuse_existing_wavelet_samples": bool(reuse_existing_wavelet_samples),
        "reuse_existing_projections": bool(reuse_existing_projections),
        "save_wavelet_samples": bool(save_wavelet_samples and collect_samples),
        "pca_checkpoint_file": pca_checkpoint_file,
        "wavelet_sample_files": saved_wavelet_samples,
        "proj_files": proj_files,
        "output_dir": str(output_dir),
    }
    with open(output_dir / "projection_result.pkl", "wb") as f:
        pickle.dump(result, f)
    if verbose:
        label = "raw wavelet feature projections" if wavelet_projection_mode == "raw_wavelets" else "shared PCA projections"
        print(f"Saved {label} under {proj_dir}")
    return result


def projection_sample_for_tuning(proj_files, subsample_factor=20):
    """Load every Nth row from saved projection files for tuning diagnostics."""
    samples = []
    for proj_file in proj_files:
        proj = np.load(proj_file, mmap_mode="r")
        samples.append(_subsample_rows(proj, subsample_factor))
    return np.concatenate(samples, axis=0).astype(np.float32, copy=False)


def resolve_repo_path(path, *, repo_root=None):
    """Resolve saved relative paths against the repository root."""
    path = Path(path)
    if path.is_absolute():
        return path
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent
    return Path(repo_root) / path


def load_projection_result(path, *, repo_root=None, normalize_paths=True):
    """Load a saved ``projection_result.pkl`` with sklearn-version warnings muted."""
    path = Path(path)
    if path.is_dir():
        path = path / "projection_result.pkl"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
        with open(path, "rb") as handle:
            projection = pickle.load(handle)
    if normalize_paths and "proj_files" in projection:
        projection = dict(projection)
        projection["proj_files"] = [
            str(resolve_repo_path(value, repo_root=repo_root))
            for value in projection["proj_files"]
        ]
        if "wavelet_sample_files" in projection:
            projection["wavelet_sample_files"] = [
                str(resolve_repo_path(value, repo_root=repo_root))
                for value in projection.get("wavelet_sample_files", [])
            ]
    return projection


def resolve_pooled_run_root(root, *, species=None, representation_name=None):
    """Find a saved pooled run root across the historical notebook layouts.

    Accepted inputs include a direct run folder, a ``projection_result.pkl`` file,
    an output root containing ``<species>``, and the older
    ``<representation>/<species>/distance_pca_wavelet_pca`` tuning layout.
    """
    root = Path(root)
    if root.is_file() and root.name == "projection_result.pkl":
        return root.parent
    candidates = [root]
    if species is not None:
        species = str(species)
        candidates.extend([
            root / species,
            root / species / "distance_pca_wavelet_pca",
        ])
        if representation_name is not None:
            representation_name = str(representation_name)
            candidates.extend([
                root / representation_name / species,
                root / representation_name / species / "distance_pca_wavelet_pca",
            ])
            if root.name == representation_name:
                candidates.extend([
                    root.parent / species,
                    root.parent / species / "distance_pca_wavelet_pca",
                ])
    for candidate in candidates:
        if (candidate / "projection_result.pkl").exists():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No projection_result.pkl found in: {searched}")


def ordered_state_files_for_projection(projection, root=None):
    """Return state files ordered to match ``projection['individual_ids']``."""
    if root is None:
        root = projection.get("output_dir")
    if root is None:
        raise ValueError("Pass root when projection has no output_dir")
    root = Path(root)
    state_dir = root / "states"
    if not state_dir.exists():
        raise FileNotFoundError(f"No states directory found at {state_dir}")

    state_files_by_id = {
        _remove_suffix(path.stem, "_states"): path
        for path in sorted(state_dir.glob("*_states.npy"))
    }
    ids = [str(value) for value in projection["individual_ids"]]
    missing = [value for value in ids if value not in state_files_by_id]
    if missing:
        preview = ", ".join(missing[:10])
        raise KeyError(
            f"Missing state files for {len(missing)} projection individual IDs "
            f"under {state_dir}: {preview}"
        )
    return [state_files_by_id[value] for value in ids]


def load_ordered_states_for_projection(projection, root=None, *, mmap_mode="r"):
    """Load state arrays ordered to match ``projection['individual_ids']``."""
    state_files = ordered_state_files_for_projection(projection, root=root)
    return [np.load(path, mmap_mode=mmap_mode) for path in state_files], state_files


def assign_pooled_states_from_projections(
    *,
    proj_files,
    d_embed,
    N,
    output_dir="outputs/pooled_by_individual",
    kmeans_subsample_factor=20,
    seed=0,
    kmeans_n_init=20,
    save_states=True,
    verbose=True,
):
    """Fit shared k-means from saved projections and assign every frame."""
    output_dir = Path(output_dir)
    state_dir = output_dir / "states"
    if save_states:
        state_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Fitting shared k-means: d_embed={d_embed}, N={N}, "
              f"sample every {kmeans_subsample_factor} embedded rows")

    embed_samples = []
    for proj_file in proj_files:
        proj = np.load(proj_file, mmap_mode="r")
        embed_samples.append(
            _delay_embed_subsample(
                proj, d=d_embed, tau=1, factor=kmeans_subsample_factor
            ).astype(np.float32, copy=False)
        )
        del proj

    X_emb_sample = np.concatenate(embed_samples, axis=0)
    del embed_samples
    if verbose:
        print(f"  k-means sample matrix = {X_emb_sample.shape}")
    km = MiniBatchKMeans(
        n_clusters=N,
        batch_size=max(N * 5, 1000),
        n_init=kmeans_n_init,
        random_state=seed,
        init="random",
    )
    km.fit(X_emb_sample)
    del X_emb_sample

    if verbose:
        print("Assigning every embedded frame to shared clusters")
    states_list = []
    state_files = []
    for i, proj_file in enumerate(proj_files, start=1):
        individual_id = _remove_suffix(Path(proj_file).stem, "_proj")
        if verbose:
            print(f"  [{i}/{len(proj_files)}] {individual_id}: assigning states")
        proj = np.load(proj_file, mmap_mode="r")
        states = _predict_delay_embedded_in_chunks(
            km,
            proj,
            d=d_embed,
            tau=1,
        )
        states_list.append(states)
        if save_states:
            state_file = state_dir / f"{individual_id}_states.npy"
            np.save(state_file, states)
            state_files.append(str(state_file))
        del proj
        gc.collect()

    return {
        "kmeans": km,
        "states_list": states_list,
        "state_files": state_files,
        "d_embed": int(d_embed),
        "N": int(N),
        "kmeans_subsample_factor": int(kmeans_subsample_factor),
        "save_states": bool(save_states),
    }


def entropy_gap_pooled_projections(
    *,
    proj_files,
    d_embed,
    N_values,
    lag,
    framerate=1.0,
    kmeans_subsample_factor=20,
    seed=0,
    n_init=5,
    verbose=True,
):
    """Compute Delta H(N) using pooled per-individual state sequences."""
    H = np.zeros(len(N_values))
    H_shuf = np.zeros(len(N_values))
    for k, N in enumerate(N_values):
        out = assign_pooled_states_from_projections(
            proj_files=proj_files,
            d_embed=d_embed,
            N=int(N),
            output_dir=Path(proj_files[0]).parents[1],
            kmeans_subsample_factor=kmeans_subsample_factor,
            seed=seed,
            kmeans_n_init=n_init,
            save_states=False,
            verbose=False,
        )
        labels = out["states_list"]
        H[k] = pp.markov_entropy(labels, lag, framerate=framerate)
        labels_shuf = [
            pp.shannon_shuffle(
                s,
                seed=None if seed is None else int(seed) + sequence_index,
            )
            for sequence_index, s in enumerate(labels)
        ]
        H_shuf[k] = pp.markov_entropy(labels_shuf, lag, framerate=framerate)
        if verbose:
            print(f"  N={N}: H={H[k]:.3f}  H_shuf={H_shuf[k]:.3f}  "
                  f"DeltaH={H_shuf[k] - H[k]:.3f}")
    return np.array(N_values), H, H_shuf


def _conditional_gpcca_for_clusters(
    states_list,
    lag,
    n_states,
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

    local_index = np.full(int(n_states), -1, dtype=int)
    local_index[parent_clusters] = np.arange(len(parent_clusters), dtype=int)
    segments = []
    segment_rows = []
    for sequence_index, states in enumerate(states_list):
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
                    "sequence_index": int(sequence_index),
                    "start_frame": int(start),
                    "stop_frame": int(stop),
                    "n_frames": int(stop - start),
                    "usable_for_lag": bool((stop - start) > int(lag)),
                })
                start = None
        if start is not None:
            stop = len(states)
            local_segment = local_index[states[start:stop]]
            segments.append(local_segment.astype(int, copy=False))
            segment_rows.append({
                "sequence_index": int(sequence_index),
                "start_frame": int(start),
                "stop_frame": int(stop),
                "n_frames": int(stop - start),
                "usable_for_lag": bool((stop - start) > int(lag)),
            })

    usable_segments = [segment for segment in segments if len(segment) > int(lag)]
    if not usable_segments:
        raise ValueError(
            "No within-arm segments are longer than the transition lag"
        )

    T_cond = pp.make_transition_matrix(
        usable_segments, lag=int(lag), n_states=len(parent_clusters),
    )
    pi_cond = pp.stationary_distribution(T_cond)
    evals, _ = pp.leading_eigvecs(
        T_cond, k=min(10, max(1, parent_clusters.size - 1)),
    )
    gpcca = gu.run_gpcca(T_cond, M=int(n_nested_basins), eta=pi_cond)
    return {
        "parent_clusters": parent_clusters,
        "T": T_cond,
        "pi": pi_cond,
        "evals": evals,
        "gpcca": gpcca,
        "chi": gpcca["chi"],
        "assignments": gpcca["assignments"],
        "segments": segments,
        "segment_table": pd.DataFrame(segment_rows),
        "n_segments": int(len(segments)),
        "n_usable_segments": int(len(usable_segments)),
        "n_segment_frames": int(sum(len(segment) for segment in segments)),
        "n_usable_segment_frames": int(
            sum(len(segment) for segment in usable_segments)
        ),
    }


def build_recursive_gpcca_arm_tree(
    T,
    pi,
    base_gpcca,
    states_list,
    lag,
    *,
    min_parent_clusters_to_split=20,
    min_child_balance=0.05,
    min_parent_operator_gap_ratio=1.15,
    min_child_global_pi=0.0,
    max_depth=3,
    nested_basins=2,
):
    """Recursively split arms with balanced children and a separated slow mode."""
    assignments = np.asarray(base_gpcca["assignments"], dtype=int)
    pi = np.asarray(pi, dtype=float)
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
                states_list,
                lag,
                T.shape[0],
                node["clusters"],
                n_nested_basins=int(nested_basins),
            )
        except ValueError as exc:
            node["split_reason"] = f"segment_operator_unavailable: {exc}"
            leaves.append(node)
            return
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
        node["n_segments"] = sub["n_segments"]
        node["n_usable_segments"] = sub["n_usable_segments"]
        node["n_segment_frames"] = sub["n_segment_frames"]
        node["n_usable_segment_frames"] = sub["n_usable_segment_frames"]
        child_assignments = np.asarray(sub["assignments"], dtype=int)
        child_counts = np.bincount(child_assignments, minlength=int(nested_basins))
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
        node["conditional"] = sub
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

    n_base = base_gpcca["chi"].shape[1]
    roots = []
    for arm in range(n_base):
        root = add_node(
            None, 0, f"arm_{arm}",
            np.flatnonzero(assignments == arm),
            (int(arm),),
        )
        roots.append(root["node_id"])
        split_node(root)

    node_rows = []
    for node in nodes:
        node_rows.append({
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
            "n_segments": node.get("n_segments"),
            "n_usable_segments": node.get("n_usable_segments"),
            "n_segment_frames": node.get("n_segment_frames"),
            "n_usable_segment_frames": node.get("n_usable_segment_frames"),
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
        "node_table": pd.DataFrame(node_rows),
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
        },
    }


def _fill_recursive_leaf_membership(base_gpcca, arm_tree, leaf, leaf_index, leaf_chi):
    """Fill one terminal leaf column with hard membership."""
    leaf_clusters = np.asarray(leaf["clusters"], dtype=int)
    leaf_chi[leaf_clusters, leaf_index] = 1.0


def recursive_gpcca_from_tree(base_gpcca, pi, arm_tree):
    """Convert a recursive arm tree into a GPCCA-like leaf-arm result."""
    n_clusters = base_gpcca["chi"].shape[0]
    leaves = arm_tree["leaves"]
    n_leaves = len(leaves)
    leaf_chi = np.zeros((n_clusters, n_leaves), dtype=float)
    leaf_assignments = np.full(n_clusters, -1, dtype=int)
    for leaf_index, leaf in enumerate(leaves):
        clusters = np.asarray(leaf["clusters"], dtype=int)
        leaf_assignments[clusters] = leaf_index
        _fill_recursive_leaf_membership(
            base_gpcca, arm_tree, leaf, leaf_index, leaf_chi,
        )

    row_sum = leaf_chi.sum(axis=1, keepdims=True)
    for cluster in np.flatnonzero(row_sum[:, 0] <= 0):
        hard_leaf = leaf_assignments[cluster]
        if hard_leaf >= 0:
            leaf_chi[cluster, hard_leaf] = 1.0
    row_sum = leaf_chi.sum(axis=1, keepdims=True)
    leaf_chi = np.divide(
        leaf_chi, row_sum, out=np.zeros_like(leaf_chi), where=row_sum > 0,
    )

    pi = np.asarray(pi, dtype=float)
    pi_basin = leaf_chi.T @ pi
    basin_counts = np.array([
        int(np.sum(leaf_assignments == j)) for j in range(n_leaves)
    ])
    return {
        "chi": leaf_chi,
        "assignments": leaf_assignments,
        "crispness": float(np.mean(np.max(leaf_chi, axis=1))),
        "basin_counts": basin_counts,
        "pi_basin": pi_basin,
        "pi": pi,
        "arm_tree": arm_tree,
        "is_recursive_leaf_gpcca": True,
        "base_gpcca": base_gpcca,
    }


def censored_segment_gpcca(
    states_list,
    chi,
    *,
    lag,
    n_states,
    M,
    censor_arm=1,
    chi_threshold=0.1,
    store_original_indices=False,
):
    """Rebuild T after censoring one chi arm and splitting on censored bouts.

    This follows the labmate-style workflow: states with membership in one
    censor arm above a threshold are merged into one censor state, the original
    state sequences are split on that merged state, and transitions are counted
    only within the remaining contiguous segments.
    """
    chi = np.asarray(chi, dtype=float)
    censor_arm = int(censor_arm)
    if chi.ndim != 2:
        raise ValueError("chi must be a 2D matrix of shape clusters x arms")
    if not 0 <= censor_arm < chi.shape[1]:
        raise ValueError(
            f"censor_arm={censor_arm} is outside chi with {chi.shape[1]} arms"
        )
    if chi.shape[0] != int(n_states):
        raise ValueError("chi must have one row per microstate")

    censor_clusters = np.flatnonzero(chi[:, censor_arm] > float(chi_threshold))
    keep_clusters = np.setdiff1d(
        np.arange(int(n_states), dtype=int), censor_clusters, assume_unique=True,
    )
    if len(keep_clusters) < int(M):
        raise ValueError(
            f"Only {len(keep_clusters)} uncensored states remain for M={int(M)}"
        )

    original_to_local = np.full(int(n_states), -1, dtype=int)
    original_to_local[keep_clusters] = np.arange(len(keep_clusters), dtype=int)

    segments = []
    segment_rows = []
    original_indices = [] if store_original_indices else None
    for sequence_index, states in enumerate(states_list):
        states = np.asarray(states, dtype=int)
        censored = np.isin(states, censor_clusters)
        start = None
        for index, is_censored in enumerate(censored):
            if not is_censored and start is None:
                start = index
            elif is_censored and start is not None:
                stop = index
                local_segment = original_to_local[states[start:stop]]
                if np.any(local_segment < 0):
                    raise ValueError("Internal censoring map produced -1 states")
                segments.append(local_segment.astype(int, copy=False))
                if store_original_indices:
                    original_indices.append(np.arange(start, stop, dtype=int))
                segment_rows.append({
                    "sequence_index": int(sequence_index),
                    "start_frame": int(start),
                    "stop_frame": int(stop),
                    "n_frames": int(stop - start),
                    "usable_for_lag": bool((stop - start) > int(lag)),
                })
                start = None
        if start is not None:
            stop = len(states)
            local_segment = original_to_local[states[start:stop]]
            if np.any(local_segment < 0):
                raise ValueError("Internal censoring map produced -1 states")
            segments.append(local_segment.astype(int, copy=False))
            if store_original_indices:
                original_indices.append(np.arange(start, stop, dtype=int))
            segment_rows.append({
                "sequence_index": int(sequence_index),
                "start_frame": int(start),
                "stop_frame": int(stop),
                "n_frames": int(stop - start),
                "usable_for_lag": bool((stop - start) > int(lag)),
            })

    usable_segments = [segment for segment in segments if len(segment) > int(lag)]
    if not usable_segments:
        raise ValueError("No uncensored segments are longer than the transition lag")

    T = pp.make_transition_matrix(
        usable_segments, lag=int(lag), n_states=len(keep_clusters),
    )
    pi = pp.stationary_distribution(T)
    evals, evecs = pp.leading_eigvecs(T, k=min(10, max(1, len(keep_clusters) - 1)))
    gpcca = gu.run_gpcca(T, M=int(M), eta=pi)
    phi = np.asarray(evecs[:, 1:4].real)
    if phi.shape[1] < 3:
        phi = np.pad(phi, ((0, 0), (0, 3 - phi.shape[1])))
    geometry = gu.compute_hub_arms(phi, pi, gpcca["chi"])

    split_censored_states = (
        np.concatenate(segments) if segments else np.array([], dtype=int)
    )
    result = {
        "lag": int(lag),
        "n_states": int(len(keep_clusters)),
        "original_n_states": int(n_states),
        "censor_arm": censor_arm,
        "chi_threshold": float(chi_threshold),
        "censor_clusters": censor_clusters,
        "keep_clusters": keep_clusters,
        "original_to_local": original_to_local,
        "local_to_original": keep_clusters,
        "segments": segments,
        "usable_segments": usable_segments,
        "split_censored_states": split_censored_states,
        "segment_table": pd.DataFrame(segment_rows),
        "n_segments": int(len(segments)),
        "n_usable_segments": int(len(usable_segments)),
        "n_segment_frames": int(sum(len(segment) for segment in segments)),
        "n_usable_segment_frames": int(
            sum(len(segment) for segment in usable_segments)
        ),
        "T": T,
        "pi": pi,
        "evals": evals,
        "evecs": evecs,
        "phi": phi,
        "gpcca": gpcca,
        "chi": gpcca["chi"],
        "assignments": gpcca["assignments"],
        "geometry": geometry,
    }
    if store_original_indices:
        result["original_indices"] = original_indices
        result["original_idx"] = (
            np.concatenate(original_indices)
            if original_indices else np.array([], dtype=int)
        )
    return result


def compute_pooled_slow_modes(
    states_list,
    *,
    fs,
    tau_seconds,
    N,
    M,
    use_recursive_arms=True,
    recursive_min_parent_clusters_to_split=20,
    recursive_min_child_balance=0.05,
    recursive_min_parent_operator_gap_ratio=1.15,
    recursive_min_child_global_pi=0.0,
    recursive_max_depth=3,
    recursive_nested_basins=2,
):
    """Build pooled T(tau), eigendecompose, and run the default arm model."""
    lag = int(tau_seconds * fs)
    T = pp.make_transition_matrix(states_list, lag=lag, n_states=N)
    pi = pp.stationary_distribution(T)
    evals, evecs = pp.leading_eigvecs(T, k=10)
    base_gpcca = gu.run_gpcca(T, M=M, eta=pi)
    gpcca = base_gpcca
    arm_tree = None
    if use_recursive_arms:
        arm_tree = build_recursive_gpcca_arm_tree(
            T,
            pi,
            base_gpcca,
            states_list,
            lag,
            min_parent_clusters_to_split=recursive_min_parent_clusters_to_split,
            min_child_balance=recursive_min_child_balance,
            min_parent_operator_gap_ratio=recursive_min_parent_operator_gap_ratio,
            min_child_global_pi=recursive_min_child_global_pi,
            max_depth=recursive_max_depth,
            nested_basins=recursive_nested_basins,
        )
        if len(arm_tree["leaves"]):
            gpcca = recursive_gpcca_from_tree(base_gpcca, pi, arm_tree)
    return {
        "lag": lag,
        "T": T,
        "pi": pi,
        "evals": evals,
        "evecs": evecs,
        "gpcca": gpcca,
        "base_gpcca": base_gpcca,
        "arm_tree": arm_tree,
        "use_recursive_arms": bool(use_recursive_arms),
    }


def lagged_mutual_information(states, lag, *, n_states=None):
    """Empirical I(s_t; s_{t+lag}) in bits for one discrete sequence."""
    states = np.asarray(states, dtype=int)
    lag = int(lag)
    if lag < 1:
        raise ValueError("lag must be at least 1")
    if states.size <= lag:
        return np.nan
    if n_states is None:
        n_states = int(states.max()) + 1

    joint = np.zeros((int(n_states), int(n_states)), dtype=float)
    np.add.at(joint, (states[:-lag], states[lag:]), 1.0)
    joint /= joint.sum()
    p_t = joint.sum(axis=1)
    p_lag = joint.sum(axis=0)
    independent = p_t[:, None] * p_lag[None, :]
    occupied = joint > 0
    return float(np.sum(
        joint[occupied] * np.log2(joint[occupied] / independent[occupied])
    ))


def leave_one_individual_out_basin_validation(
    states_list,
    *,
    lag,
    n_states,
    M_values,
    n_random=200,
    seed=0,
    individual_ids=None,
):
    """Re-fit G-PCCA per held-out individual and evaluate basin-level MI.

    Each random control permutes the fitted hard basin labels across clusters,
    preserving exactly the number of clusters assigned to every basin.
    """
    import pandas as pd

    sequences = [np.asarray(states, dtype=int) for states in states_list]
    if len(sequences) < 2:
        raise ValueError("Leave-one-individual-out validation needs at least 2 individuals")
    if individual_ids is None:
        individual_ids = [f"individual_{index + 1}" for index in range(len(sequences))]
    if len(individual_ids) != len(sequences):
        raise ValueError("individual_ids and states_list must have the same length")

    rng = np.random.default_rng(seed)
    rows = []
    for M in [int(value) for value in M_values]:
        if not 2 <= M <= int(n_states):
            raise ValueError(f"M={M} must be between 2 and n_states={n_states}")
        for held_out, held_out_states in enumerate(sequences):
            training_states = [
                states for index, states in enumerate(sequences) if index != held_out
            ]
            T_train = pp.make_transition_matrix(
                training_states, lag=lag, n_states=n_states,
            )
            pi_train = pp.stationary_distribution(T_train)
            gpcca = gu.run_gpcca(T_train, M=M, eta=pi_train)
            cluster_to_basin = np.asarray(gpcca["assignments"], dtype=int)
            held_out_basins = cluster_to_basin[held_out_states]
            observed_mi = lagged_mutual_information(
                held_out_basins, lag, n_states=M,
            )

            random_mi = np.empty(int(n_random), dtype=float)
            for shuffle_index in range(int(n_random)):
                random_assignment = rng.permutation(cluster_to_basin)
                random_mi[shuffle_index] = lagged_mutual_information(
                    random_assignment[held_out_states], lag, n_states=M,
                )

            finite_random = random_mi[np.isfinite(random_mi)]
            random_mean = (
                float(finite_random.mean()) if finite_random.size else np.nan
            )
            random_std = (
                float(finite_random.std(ddof=1)) if finite_random.size > 1 else np.nan
            )
            p_value = (
                float((1 + np.sum(finite_random >= observed_mi))
                      / (1 + finite_random.size))
                if finite_random.size and np.isfinite(observed_mi) else np.nan
            )
            rows.append({
                "M": M,
                "held_out_index": held_out,
                "held_out_individual": str(individual_ids[held_out]),
                "held_out_mi_bits": observed_mi,
                "random_mi_mean_bits": random_mean,
                "random_mi_std_bits": random_std,
                "mi_above_random_bits": observed_mi - random_mean,
                "randomization_p_value": p_value,
                "n_random": int(n_random),
                "lag_frames": int(lag),
                "training_gpcca_crispness": float(gpcca["crispness"]),
                "training_basin_cluster_counts": str(
                    np.bincount(cluster_to_basin, minlength=M).tolist()
                ),
            })

    folds = pd.DataFrame(rows)
    summary = (
        folds.groupby("M", as_index=False)
        .agg(
            held_out_mi_mean_bits=("held_out_mi_bits", "mean"),
            held_out_mi_sem_bits=("held_out_mi_bits", "sem"),
            random_mi_mean_bits=("random_mi_mean_bits", "mean"),
            random_mi_sem_bits=("random_mi_mean_bits", "sem"),
            mi_above_random_mean_bits=("mi_above_random_bits", "mean"),
            n_folds=("held_out_index", "size"),
        )
    )
    return {"folds": folds, "summary": summary}


def plot_leave_one_individual_out_basin_validation(
    folds,
    *,
    title="Leave-one-individual-out validation of basin count",
    output_prefix=None,
):
    """Plot held-out basin MI and matched-size random-coloring controls."""
    import matplotlib.pyplot as plt

    M_values = np.sort(folds["M"].unique())
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    for M in M_values:
        subset = folds.loc[folds["M"] == M].sort_values("held_out_index")
        offsets = np.linspace(-0.13, 0.13, len(subset)) if len(subset) > 1 else [0]
        x = M + np.asarray(offsets)
        ax.scatter(
            x, subset["held_out_mi_bits"], color="#E07B39", s=38,
            label="held-out G-PCCA partition" if M == M_values[0] else None,
            zorder=3,
        )
        ax.scatter(
            x, subset["random_mi_mean_bits"], marker="s", color="#777777", s=30,
            label="matched-size random colorings" if M == M_values[0] else None,
            zorder=2,
        )
        for x_value, observed, random_mean in zip(
            x, subset["held_out_mi_bits"], subset["random_mi_mean_bits"],
        ):
            ax.plot(
                [x_value, x_value], [random_mean, observed],
                color="#C8C8C8", linewidth=0.7, zorder=1,
            )

    ax.set_xticks(M_values)
    ax.set_xlabel("number of basins M")
    ax.set_ylabel(r"held-out $I(b_t; b_{t+\tau})$ (bits)")
    ax.set_title(title)
    ax.legend(frameon=False)
    if output_prefix is not None:
        output_prefix = Path(output_prefix)
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_prefix.with_suffix(".png"), dpi=260)
        fig.savefig(output_prefix.with_suffix(".pdf"))
    return fig, ax


def plot_fixed_M_leave_one_individual_out_validation(
    folds,
    *,
    M,
    title=None,
    output_prefix=None,
):
    """Plot paired observed/control MI across held-out individuals at fixed M."""
    import matplotlib.pyplot as plt

    subset = (
        folds.loc[folds["M"] == int(M)]
        .sort_values("held_out_index")
        .reset_index(drop=True)
    )
    if subset.empty:
        raise ValueError(f"No leave-one-out folds are available for M={M}")

    x = np.arange(len(subset))
    fig_width = max(7.2, 0.55 * len(subset))
    fig, ax = plt.subplots(
        figsize=(fig_width, 4.6), constrained_layout=True,
    )
    for x_value, observed, random_mean in zip(
        x, subset["held_out_mi_bits"], subset["random_mi_mean_bits"],
    ):
        ax.plot(
            [x_value, x_value], [random_mean, observed],
            color="#C8C8C8", linewidth=0.8, zorder=1,
        )
    ax.scatter(
        x, subset["held_out_mi_bits"], color="#E07B39", s=42,
        label="held-out G-PCCA partition", zorder=3,
    )
    ax.scatter(
        x, subset["random_mi_mean_bits"], marker="s", color="#777777", s=34,
        label="matched-size random colorings", zorder=2,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        subset["held_out_individual"], rotation=45, ha="right",
    )
    ax.set_xlabel("held-out individual")
    ax.set_ylabel(r"held-out $I(b_t; b_{t+\tau})$ (bits)")
    ax.set_title(
        title or f"Leave-one-individual-out basin validation at M={int(M)}"
    )
    ax.legend(frameon=False)

    if output_prefix is not None:
        output_prefix = Path(output_prefix)
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_prefix.with_suffix(".png"), dpi=260)
        fig.savefig(output_prefix.with_suffix(".pdf"))
    return fig, ax


def _transfer_operator_profiles(T):
    """Incoming-plus-outgoing transition profiles used for cluster distances."""
    T = np.asarray(T, dtype=float)
    return np.concatenate([T, T.T], axis=1)


def clustered_transfer_operator_order(T, *, method="average"):
    """Return a hierarchical-clustering order for transfer-operator states."""
    T = np.asarray(T, dtype=float)
    if T.shape[0] <= 2:
        return np.arange(T.shape[0])
    profiles = _transfer_operator_profiles(T)
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import pdist

        distances = pdist(profiles, metric="euclidean")
        if np.allclose(distances, 0):
            return np.arange(T.shape[0])
        return leaves_list(linkage(distances, method=method))
    except ImportError:
        distances = np.linalg.norm(
            profiles[:, None, :] - profiles[None, :, :], axis=2,
        )
        if np.allclose(distances, 0):
            return np.arange(T.shape[0])
        remaining = set(range(T.shape[0]))
        current = int(np.argmax(np.linalg.norm(profiles, axis=1)))
        order = [current]
        remaining.remove(current)
        while remaining:
            candidates = np.array(sorted(remaining), dtype=int)
            nearest = candidates[np.argmin(distances[current, candidates])]
            current = int(nearest)
            order.append(current)
            remaining.remove(current)
        return np.asarray(order, dtype=int)


def plot_clustered_transfer_operator(
    T,
    *,
    pi=None,
    title="Clustered transfer operator",
    output_prefix=None,
    order=None,
    method="average",
    log_scale=True,
    cmap="viridis",
):
    """Plot a transfer operator after hierarchical row/column ordering."""
    import matplotlib.pyplot as plt

    T = np.asarray(T, dtype=float)
    if order is None:
        order = clustered_transfer_operator_order(T, method=method)
    ordered = T[np.ix_(order, order)]
    if log_scale:
        image_values = np.log10(np.maximum(ordered, 1e-12))
        colorbar_label = "log10 transition probability"
    else:
        image_values = ordered
        colorbar_label = "transition probability"

    has_pi = pi is not None
    if has_pi:
        pi = np.asarray(pi, dtype=float)
        fig, axes = plt.subplots(
            2, 1, figsize=(6.2, 6.8),
            gridspec_kw={"height_ratios": [1, 12]},
            constrained_layout=True,
        )
        ax_pi, ax = axes
        ax_pi.bar(np.arange(len(order)), pi[order], width=1.0, color="#4C78A8")
        ax_pi.set_ylabel("pi")
        ax_pi.set_xticks([])
    else:
        fig, ax = plt.subplots(figsize=(6.2, 5.8), constrained_layout=True)

    image = ax.imshow(image_values, aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("cluster at t + lag")
    ax.set_ylabel("cluster at t")
    fig.colorbar(image, ax=ax, label=colorbar_label)

    if output_prefix is not None:
        output_prefix = Path(output_prefix)
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_prefix.with_suffix(".png"), dpi=260)
        fig.savefig(output_prefix.with_suffix(".pdf"))
    return fig, ax, np.asarray(order)


def coarse_grain_low_occupancy_clusters(T, pi, *, min_pi=1e-4):
    """Merge low-occupancy clusters into nearest retained transition-profile neighbor.

    Distances are computed from each cluster's outgoing and incoming transition
    probabilities.  The coarse-grained operator is built by aggregating
    stationary flux, ``pi_i * T_ij``, then row-normalizing by coarse occupancy.
    """
    T = np.asarray(T, dtype=float)
    pi = np.asarray(pi, dtype=float)
    min_pi = float(min_pi)
    if T.shape[0] != T.shape[1]:
        raise ValueError("T must be square.")
    if T.shape[0] != pi.shape[0]:
        raise ValueError("pi must have one entry per transfer-operator row.")

    low = pi < min_pi
    keep = ~low
    if not np.any(keep):
        keep[int(np.argmax(pi))] = True
        low = ~keep

    kept_states = np.flatnonzero(keep)
    profiles = _transfer_operator_profiles(T)
    assignment = np.full(T.shape[0], -1, dtype=int)
    assignment[kept_states] = np.arange(len(kept_states))

    rows = []
    for state in np.flatnonzero(low):
        deltas = profiles[kept_states] - profiles[state]
        distances = np.linalg.norm(deltas, axis=1)
        nearest_pos = int(np.argmin(distances))
        nearest_state = int(kept_states[nearest_pos])
        assignment[state] = nearest_pos
        rows.append({
            "merged_cluster": int(state),
            "nearest_cluster": nearest_state,
            "coarse_cluster": nearest_pos,
            "pi": float(pi[state]),
            "nearest_pi": float(pi[nearest_state]),
            "profile_distance": float(distances[nearest_pos]),
        })

    n_coarse = len(kept_states)
    coarse_pi = np.bincount(assignment, weights=pi, minlength=n_coarse)
    flux = pi[:, None] * T
    coarse_flux = np.zeros((n_coarse, n_coarse), dtype=float)
    for source_coarse in range(n_coarse):
        source_mask = assignment == source_coarse
        source_flux = flux[source_mask].sum(axis=0)
        coarse_flux[source_coarse] = np.bincount(
            assignment, weights=source_flux, minlength=n_coarse,
        )

    coarse_T = np.zeros_like(coarse_flux)
    nonempty = coarse_pi > 0
    coarse_T[nonempty] = coarse_flux[nonempty] / coarse_pi[nonempty, None]
    if np.any(~nonempty):
        coarse_T[~nonempty] = 1.0 / max(n_coarse, 1)

    merge_table = pd.DataFrame(rows)
    map_table = pd.DataFrame({
        "original_cluster": np.arange(T.shape[0], dtype=int),
        "coarse_cluster": assignment,
        "representative_cluster": kept_states[assignment],
        "is_merged": low,
        "pi": pi,
    })
    return {
        "T": coarse_T,
        "pi": coarse_pi,
        "assignment": assignment,
        "kept_states": kept_states,
        "merge_table": merge_table,
        "map_table": map_table,
    }


def save_frame_cluster_assignments_csv(
    *,
    output_csv,
    species,
    individual_ids,
    state_files,
    metadata=None,
    d_embed=1,
    chunksize=100000,
):
    """Stream per-embedded-frame cluster assignments to CSV.

    State arrays are read from disk with memory mapping, so this does not create
    a large per-species DataFrame in RAM.  Frame indices refer to the
    concatenated per-individual trace used for assignment.  Because each
    embedded frame spans ``d_embed`` raw frames, ``raw_frame_start`` and
    ``raw_frame_end`` give the covered raw-frame window.
    """
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    metadata_by_id = {}
    if metadata is not None:
        metadata_by_id = {str(m["individual_id"]): m for m in metadata}

    window = max(int(d_embed), 1)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "species",
            "individual_id",
            "embedded_frame_index",
            "raw_frame_start",
            "raw_frame_end",
            "recording_id",
            "recording_frame_start",
            "recording_frame_end",
            "crosses_recording_boundary",
            "cluster",
        ])

        for individual_id, state_file in zip(individual_ids, state_files):
            individual_id = str(individual_id)
            states = np.load(state_file, mmap_mode="r")
            meta = metadata_by_id.get(individual_id, {})
            segment_lengths = np.asarray(meta.get("segment_lengths", []), dtype=int)
            recording_ids = [str(r) for r in meta.get("recording_ids", [])]
            segment_ends = np.cumsum(segment_lengths) if len(segment_lengths) else np.array([])

            for start in range(0, len(states), int(chunksize)):
                stop = min(start + int(chunksize), len(states))
                rows = []
                for embedded_idx in range(start, stop):
                    raw_start = embedded_idx
                    raw_end = embedded_idx + window - 1
                    recording_id = ""
                    rec_start = raw_start
                    rec_end = raw_end
                    crosses_boundary = False

                    if len(segment_ends):
                        seg_idx = int(np.searchsorted(segment_ends, raw_start, side="right"))
                        seg_start = 0 if seg_idx == 0 else int(segment_ends[seg_idx - 1])
                        seg_end = int(segment_ends[seg_idx])
                        recording_id = (
                            recording_ids[seg_idx]
                            if seg_idx < len(recording_ids)
                            else str(seg_idx)
                        )
                        rec_start = raw_start - seg_start
                        rec_end = raw_end - seg_start
                        crosses_boundary = raw_end >= seg_end

                    rows.append([
                        species,
                        individual_id,
                        int(embedded_idx),
                        int(raw_start),
                        int(raw_end),
                        recording_id,
                        int(rec_start),
                        int(rec_end),
                        bool(crosses_boundary),
                        int(states[embedded_idx]),
                    ])
                writer.writerows(rows)

    return str(output_csv)


def save_pooled_outputs(
    *,
    output_dir,
    species,
    individual_ids,
    state_files,
    states_list,
    T,
    pi,
    evals,
    evecs,
    chi,
    parameters,
    metadata=None,
    arm_tree=None,
    save_csv_transition=True,
    save_frame_assignments=True,
    frame_assignments_csv=None,
):
    """Save pooled slow-mode outputs and a 3D arms/eigenspace plot.

    Large per-frame cluster allocations are kept as one ``.npy`` file per
    individual; the manifest records which file belongs to which individual.
    """
    import matplotlib.pyplot as plt
    import figures as fg

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Transfer operator.
    np.save(output_dir / "transfer_operator.npy", T)
    if save_csv_transition:
        pd.DataFrame(T).to_csv(output_dir / "transfer_operator.csv", index=False)
    np.save(output_dir / "stationary_distribution.npy", pi)
    np.save(output_dir / "evals.npy", evals)
    np.save(output_dir / "evecs.npy", evecs)
    np.save(output_dir / "chi.npy", chi)

    # State allocation manifest.  The state arrays themselves are already saved
    # by assign_pooled_states_from_projections; keep CSV compact and readable.
    rows = []
    metadata_by_id = {}
    if metadata is not None:
        metadata_by_id = {str(m["individual_id"]): m for m in metadata}
    for individual_id, state_file, states in zip(individual_ids, state_files, states_list):
        m = metadata_by_id.get(str(individual_id), {})
        rows.append({
            "species": species,
            "individual_id": str(individual_id),
            "state_file": str(state_file),
            "n_embedded_frames": int(len(states)),
            "n_raw_frames": int(m.get("n_frames", len(states))),
            "n_recordings": len(m.get("recording_ids", [])),
            "recording_ids": ";".join(m.get("recording_ids", [])),
        })
    allocation_df = pd.DataFrame(rows)
    allocation_df.to_csv(output_dir / "cluster_allocations_manifest.csv", index=False)

    # Parameters table.
    param_rows = [{"parameter": k, "value": v} for k, v in parameters.items()]
    param_df = pd.DataFrame(param_rows)
    param_df.to_csv(output_dir / "parameters.csv", index=False)

    arm_tree_nodes_path = None
    arm_tree_clusters_path = None
    if arm_tree is not None:
        arm_tree_nodes_path = output_dir / "recursive_arm_tree_nodes.csv"
        arm_tree_clusters_path = output_dir / "recursive_arm_tree_clusters.csv"
        arm_tree["node_table"].to_csv(arm_tree_nodes_path, index=False)
        arm_tree["cluster_table"].to_csv(arm_tree_clusters_path, index=False)

    frame_assignments_path = None
    if save_frame_assignments and state_files:
        if frame_assignments_csv is None:
            frame_assignments_csv = output_dir / "frame_cluster_assignments.csv"
        frame_assignments_path = save_frame_cluster_assignments_csv(
            output_csv=frame_assignments_csv,
            species=species,
            individual_ids=individual_ids,
            state_files=state_files,
            metadata=metadata,
            d_embed=parameters.get("d_embed", 1),
        )

    # Arms / eigenspace plot.
    # Exclude phi_1, the stationary mode, from eigenspace geometry.
    phi = np.asarray(evecs[:, 1:4].real)
    if phi.shape[1] < 3:
        phi = np.pad(phi, ((0, 0), (0, 3 - phi.shape[1])))
    geo = gu.compute_hub_arms(phi, pi, chi)
    fig = plt.figure(figsize=(4.8, 4.1))
    ax = fig.add_subplot(111, projection="3d")
    fg.plot_eigenspace_3d(
        ax, phi, pi=pi, chi=chi,
        hub=geo["hub"], arm_dirs=geo["arm_dirs"],
        arm_centroids=geo["arm_centroids"],
    )
    ax.set_title(str(species))
    fig.tight_layout()
    fig.savefig(output_dir / "arms_eigenspace.png", dpi=260)
    fig.savefig(output_dir / "arms_eigenspace.pdf")
    plt.close(fig)

    return {
        "output_dir": str(output_dir),
        "parameters_csv": str(output_dir / "parameters.csv"),
        "allocations_csv": str(output_dir / "cluster_allocations_manifest.csv"),
        "transfer_operator_npy": str(output_dir / "transfer_operator.npy"),
        "arms_plot_png": str(output_dir / "arms_eigenspace.png"),
        "frame_assignments_csv": frame_assignments_path,
        "recursive_arm_tree_nodes_csv": (
            str(arm_tree_nodes_path) if arm_tree_nodes_path is not None else None
        ),
        "recursive_arm_tree_clusters_csv": (
            str(arm_tree_clusters_path)
            if arm_tree_clusters_path is not None else None
        ),
    }


def run_pooled_individual_pipeline(
    *,
    pipeline_species,
    individual_ids,
    load_individual_trace_fn,
    dataset_name,
    fs,
    fmin,
    fmax,
    n_freqs,
    d_embed,
    N,
    tau_seconds,
    M,
    output_dir="outputs/pooled_by_individual",
    wavelet_subsample_factor=20,
    kmeans_subsample_factor=None,
    pca_max_components=20,
    n_shuffles=10,
    wavelet_dtype=np.float32,
    normalize_wavelets_for_pca=False,
    seed=0,
    kmeans_n_init=20,
    use_recursive_arms=True,
    recursive_min_parent_clusters_to_split=20,
    recursive_min_child_balance=0.05,
    recursive_min_parent_operator_gap_ratio=1.15,
    recursive_min_child_global_pi=0.0,
    recursive_max_depth=3,
    recursive_nested_basins=2,
    verbose=True,
):
    """Convenience wrapper that runs all pooled stages with fixed parameters."""
    if kmeans_subsample_factor is None:
        kmeans_subsample_factor = wavelet_subsample_factor
    projection_result = prepare_pooled_individual_projections(
        pipeline_species=pipeline_species,
        individual_ids=individual_ids,
        load_individual_trace_fn=load_individual_trace_fn,
        dataset_name=dataset_name,
        fs=fs,
        fmin=fmin,
        fmax=fmax,
        n_freqs=n_freqs,
        output_dir=output_dir,
        wavelet_subsample_factor=wavelet_subsample_factor,
        pca_max_components=pca_max_components,
        n_shuffles=n_shuffles,
        wavelet_dtype=wavelet_dtype,
        normalize_wavelets_for_pca=normalize_wavelets_for_pca,
        seed=seed,
        verbose=verbose,
    )
    state_result = assign_pooled_states_from_projections(
        proj_files=projection_result["proj_files"],
        d_embed=d_embed,
        N=N,
        output_dir=output_dir,
        kmeans_subsample_factor=kmeans_subsample_factor,
        seed=seed,
        kmeans_n_init=kmeans_n_init,
        verbose=verbose,
    )
    slow_result = compute_pooled_slow_modes(
        state_result["states_list"],
        fs=fs,
        tau_seconds=tau_seconds,
        N=N,
        M=M,
        use_recursive_arms=use_recursive_arms,
        recursive_min_parent_clusters_to_split=recursive_min_parent_clusters_to_split,
        recursive_min_child_balance=recursive_min_child_balance,
        recursive_min_parent_operator_gap_ratio=recursive_min_parent_operator_gap_ratio,
        recursive_min_child_global_pi=recursive_min_child_global_pi,
        recursive_max_depth=recursive_max_depth,
        recursive_nested_basins=recursive_nested_basins,
    )

    result = {
        **projection_result,
        **state_result,
        **slow_result,
        "microstate_files": state_result["state_files"],
    }
    output_dir = Path(output_dir)
    with open(output_dir / "pooled_result.pkl", "wb") as f:
        pickle.dump(
            {k: v for k, v in result.items() if k not in {"states_list"}},
            f,
        )
    if verbose:
        print(f"Saved projections/states/results under {output_dir}")
    return result


def _choose_individual_ids(all_ids, *, random_n_individuals=None, seed=0):
    """Return all IDs or a reproducible random subset."""
    all_ids = [str(x) for x in all_ids]
    if random_n_individuals is None:
        return all_ids
    rng = np.random.default_rng(seed)
    n_sample = min(int(random_n_individuals), len(all_ids))
    return rng.choice(all_ids, size=n_sample, replace=False).tolist()


def run_species_batch(
    *,
    species_names,
    species_subjects_fn,
    select_manifest_fn,
    samplerate_fn,
    load_individual_trace_fn,
    dataset_name,
    n_freqs,
    d_embed,
    N,
    tau_seconds,
    M,
    output_root="outputs/pooled_by_individual",
    random_n_individuals=None,
    seed=0,
    wavelet_subsample_factor=20,
    kmeans_subsample_factor=None,
    pca_max_components=20,
    n_shuffles=10,
    wavelet_dtype=np.float32,
    normalize_wavelets_for_pca=False,
    kmeans_n_init=20,
    use_recursive_arms=True,
    recursive_min_parent_clusters_to_split=20,
    recursive_min_child_balance=0.05,
    recursive_min_parent_operator_gap_ratio=1.15,
    recursive_min_child_global_pi=0.0,
    recursive_max_depth=3,
    recursive_nested_basins=2,
    fmin=0.1,
    fmax_cap=60.0,
    summary_csv=None,
    verbose=True,
):
    """Run and save the pooled-by-individual pipeline for each species.

    Each species is fully saved before the next one starts.  The summary CSV is
    rewritten after every completed species so progress survives interruptions.
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if summary_csv is None:
        summary_csv = output_root / "batch_summary.csv"
    summary_csv = Path(summary_csv)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(species_names, str):
        species_names = [species_names]
    species_names = [str(s) for s in species_names]
    if kmeans_subsample_factor is None:
        kmeans_subsample_factor = wavelet_subsample_factor

    summaries = []
    for species_name in species_names:
        selected_ids = _choose_individual_ids(
            species_subjects_fn(species_name),
            random_n_individuals=random_n_individuals,
            seed=seed,
        )
        if not selected_ids:
            raise ValueError(f"No individuals selected for {species_name!r}")

        first_row = select_manifest_fn(species_name, selected_ids[0]).iloc[0]
        fs = samplerate_fn(first_row)
        species_fmin = float(fmin)
        species_fmax = min(float(fmax_cap), 0.45 * fs)
        species_out = output_root / species_name

        if verbose:
            print(f"\n=== {species_name}: {len(selected_ids)} individuals ===")
            print(f"fs={fs:g} Hz, f range=[{species_fmin:g}, {species_fmax:g}] Hz")

        result = run_pooled_individual_pipeline(
            pipeline_species=species_name,
            individual_ids=selected_ids,
            load_individual_trace_fn=load_individual_trace_fn,
            dataset_name=dataset_name,
            fs=fs,
            fmin=species_fmin,
            fmax=species_fmax,
            n_freqs=n_freqs,
            d_embed=d_embed,
            N=N,
            tau_seconds=tau_seconds,
            M=M,
            output_dir=species_out,
            wavelet_subsample_factor=wavelet_subsample_factor,
            kmeans_subsample_factor=kmeans_subsample_factor,
            pca_max_components=pca_max_components,
            n_shuffles=n_shuffles,
            wavelet_dtype=wavelet_dtype,
            normalize_wavelets_for_pca=normalize_wavelets_for_pca,
            seed=seed,
            kmeans_n_init=kmeans_n_init,
            use_recursive_arms=use_recursive_arms,
            recursive_min_parent_clusters_to_split=recursive_min_parent_clusters_to_split,
            recursive_min_child_balance=recursive_min_child_balance,
            recursive_min_parent_operator_gap_ratio=recursive_min_parent_operator_gap_ratio,
            recursive_min_child_global_pi=recursive_min_child_global_pi,
            recursive_max_depth=recursive_max_depth,
            recursive_nested_basins=recursive_nested_basins,
            verbose=verbose,
        )

        gpcca = result["gpcca"]
        run_parameters = {
            "species": species_name,
            "dataset": dataset_name,
            "n_individuals": len(selected_ids),
            "individual_selection_seed": seed,
            "fs": fs,
            "fmin": species_fmin,
            "fmax": species_fmax,
            "n_freqs": int(n_freqs),
            "wavelet_subsample_factor": int(wavelet_subsample_factor),
            "normalize_wavelets_for_pca": bool(normalize_wavelets_for_pca),
            "kmeans_subsample_factor": int(kmeans_subsample_factor),
            "n_kept_pcs": result["n_kept"],
            "pca_shuffle_threshold": result["shuffle_threshold"],
            "d_embed": int(d_embed),
            "N": int(N),
            "tau_seconds": float(tau_seconds),
            "lag_frames": result["lag"],
            "M": int(M),
            "use_recursive_arms": bool(use_recursive_arms),
            "recursive_min_parent_clusters_to_split": int(
                recursive_min_parent_clusters_to_split
            ),
            "recursive_min_child_balance": float(recursive_min_child_balance),
            "recursive_min_parent_operator_gap_ratio": float(
                recursive_min_parent_operator_gap_ratio
            ),
            "recursive_min_child_global_pi": float(recursive_min_child_global_pi),
            "recursive_max_depth": int(recursive_max_depth),
            "recursive_nested_basins": int(recursive_nested_basins),
            "n_leaf_arms": int(gpcca["chi"].shape[1]),
            "gpcca_crispness": gpcca["crispness"],
            "basin_counts": gpcca["basin_counts"].tolist(),
            "pi_basin": gpcca["pi_basin"].round(6).tolist(),
        }
        saved = save_pooled_outputs(
            output_dir=species_out / "final_outputs",
            species=species_name,
            individual_ids=selected_ids,
            state_files=result["state_files"],
            states_list=result["states_list"],
            T=result["T"],
            pi=result["pi"],
            evals=result["evals"],
            evecs=result["evecs"],
            chi=gpcca["chi"],
            parameters=run_parameters,
            metadata=result["metadata"],
            arm_tree=result.get("arm_tree"),
        )

        summaries.append({**run_parameters, **saved})
        summary_df = pd.DataFrame(summaries)
        summary_df.to_csv(summary_csv, index=False)
        if verbose:
            print(f"Saved {species_name} and updated {summary_csv}")

        try:
            import matplotlib.pyplot as plt
            plt.close("all")
        except Exception:
            pass
        del result, gpcca, saved
        gc.collect()

    return pd.DataFrame(summaries)
