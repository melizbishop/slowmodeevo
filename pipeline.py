"""Multi-timescale transfer-operator pipeline.

The functions in this module implement the analysis pipeline described in
Kaur, Jain, & Berman, "Using timescale as a state coordinate reveals the
metastable geometry of behavior" (2026):

    raw signal -> wavelet amplitudes -> PCA -> delay embedding ->
    k-means clustering -> transition matrix T(tau) -> eigendecomposition.

Downstream basin identification (G-PCCA, hub/arm geometry) lives in
gpcca_utils.py.

References to the manuscript are given in each function's docstring.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Morlet wavelet transform (Berman 2014 MotionMapper convention; Methods Sec.
# "Wavelet decomposition").  Identical numerical recipe to find_wavelets.py
# in the original code repository, refactored for clarity and made GPU-free.
# ---------------------------------------------------------------------------
def morlet_wavelet_amplitudes(x, fs, fmin, fmax, n_freqs, omega0=5.0,
                              dtype=None):
    """Morlet wavelet amplitudes of a multivariate time series.

    Parameters
    ----------
    x : (T, d) ndarray
        Multivariate time series.  Columns are channels (e.g. joint angles).
    fs : float
        Sampling frequency in Hz.
    fmin, fmax : float
        Lowest and highest wavelet centre frequencies in Hz.
    n_freqs : int
        Number of dyadically-spaced frequency channels.  Methods uses 25.
    omega0 : float
        Dimensionless Morlet parameter (controls time-frequency tradeoff).
        We use omega0 = 5 throughout.
    dtype : numpy dtype or None
        Optional dtype for the returned amplitude matrix.  ``None`` preserves
        the historical float64 output; ``np.float32`` is useful for large
        pooled runs where memory is the limiting factor.

    Returns
    -------
    amplitudes : (T, d * n_freqs) ndarray
        Wavelet amplitudes |W(t,f)|, channel-major: amplitudes[:, c*n_freqs:(c+1)*n_freqs]
        is the spectrogram of channel c.
    f : (n_freqs,) ndarray
        Centre frequencies (low to high).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    T, d = x.shape
    dt = 1.0 / fs

    # Dyadic frequency spacing (low -> high).
    Tmin, Tmax = 1.0 / fmax, 1.0 / fmin
    Ts = Tmin * (2 ** ((np.arange(n_freqs) * np.log(Tmax / Tmin))
                       / (np.log(2) * (n_freqs - 1))))
    f = (1.0 / Ts)[::-1]

    out_dtype = float if dtype is None else dtype
    out = np.zeros((T, d * n_freqs), dtype=out_dtype)
    for c in range(d):
        out[:, c * n_freqs:(c + 1) * n_freqs] = _wavelet_one_channel(
            x[:, c], f, dt, omega0).T
    return out, f


def _wavelet_one_channel(x, f, dt, omega0):
    """Per-channel Morlet wavelet amplitudes (frequency x time)."""
    N0 = len(x)
    if N0 % 2 == 1:
        x = np.concatenate([x, [0.0]])
        wasodd = True
    else:
        wasodd = False
    M = len(x)
    # Symmetric zero-padding to reduce edge effects.
    x_pad = np.concatenate([np.zeros(M // 2), x, np.zeros(M // 2)])
    Npad = len(x_pad)
    scales = (omega0 + np.sqrt(2 + omega0 ** 2)) / (4 * np.pi * f)
    Omega = 2 * np.pi * np.arange(-Npad / 2, Npad / 2) / (Npad * dt)

    xhat = np.fft.fftshift(np.fft.fft(x_pad))
    L = len(f)
    amp = np.zeros((L, M))
    idx = np.arange(M // 2, M // 2 + M).astype(int)
    norm = (np.pi ** -0.25) * np.exp(0.25 * (omega0
                                             - np.sqrt(omega0 ** 2 + 2)) ** 2)
    for i in range(L):
        m = (np.pi ** -0.25) * np.exp(-0.5 * (-Omega * scales[i] - omega0) ** 2)
        q = np.fft.ifft(m * xhat) * np.sqrt(scales[i])
        q = q[idx]
        amp[i, :] = np.abs(q) * norm / np.sqrt(2 * scales[i])
    if wasodd:
        amp = amp[:, :N0]
    return amp[:, :N0]


# ---------------------------------------------------------------------------
# PCA on wavelet amplitudes with a temporal-shuffle threshold (Methods Sec.
# "PCA on wavelet amplitudes").  We keep the leading components whose
# eigenvalues exceed the average of the leading eigenvalue across
# temporally-shuffled copies of the wavelet matrix (each feature permuted
# independently in time -- not a phase randomization).
# ---------------------------------------------------------------------------
def pca_with_shuffle_threshold(amplitudes, n_shuffles=10, max_keep=50, seed=0,
                               percentile=None):
    """PCA with shuffle-based component selection.

    Parameters
    ----------
    amplitudes : (T, F) ndarray
        Wavelet amplitudes from morlet_wavelet_amplitudes (or any non-negative
        spectrogram-like matrix).
    n_shuffles : int
        Number of independent temporal shuffles for the threshold estimate.
    max_keep : int
        Maximum number of components to keep regardless of threshold.
    percentile : float or None
        How to reduce the `n_shuffles` shuffled leading-eigenvalues to a single
        threshold.  None (default) uses their mean (the manuscript recipe); a
        value in (0, 100] uses that percentile instead -- e.g. 95 is more
        conservative (keeps fewer PCs), 50 is the median.

    Returns
    -------
    projections : (T, k) ndarray
        Projection of `amplitudes` onto the leading k principal components.
    n_kept : int
        Number of components above threshold.
    eigvals : (max_keep,) ndarray
        Top-`max_keep` eigenvalues of the data covariance.
    threshold : float
        The lambda_1-mean of the shuffled covariance.
    """
    rng = np.random.default_rng(seed)
    A = np.asarray(amplitudes, dtype=float)
    pca = PCA(n_components=min(max_keep, A.shape[1]))
    pca.fit(A)
    eigvals = pca.explained_variance_

    # Threshold: mean of lambda_1 across phase-shuffled wavelet matrices.
    shuf_lambdas = np.zeros(n_shuffles)
    for s in range(n_shuffles):
        # Shuffle each column independently across time: preserves each
        # feature's marginal while destroying temporal and cross-feature
        # structure (a plain permutation, not a phase randomization).
        Ash = A.copy()
        for j in range(Ash.shape[1]):
            Ash[:, j] = rng.permutation(Ash[:, j])
        pca_sh = PCA(n_components=1)
        pca_sh.fit(Ash)
        shuf_lambdas[s] = pca_sh.explained_variance_[0]
    if percentile is None:
        threshold = float(shuf_lambdas.mean())
    else:
        threshold = float(np.percentile(shuf_lambdas, percentile))
    n_kept = int(np.sum(eigvals > threshold))
    projections = pca.transform(A)[:, :n_kept]
    return projections, n_kept, eigvals, threshold


# ---------------------------------------------------------------------------
# Cao's E_1(d) saturation criterion for embedding dimension (Cao 1997).
# Used to choose d in Sec. "Delay embedding and state space construction".
# ---------------------------------------------------------------------------
def cao_e1(
    X,
    max_d=20,
    tau=1,
    n_samples=20000,
    seed=0,
    n_reference_samples=None,
    verbose=False,
):
    """Cao's E_1(d) statistic for delay embedding.

    Parameters
    ----------
    X : (T, p) ndarray
        Multivariate time series in which to embed (e.g. PCA projections).
    max_d : int
        Largest embedding dimension to try.
    tau : int
        Delay (in frames).  Default 1, matching the manuscript.
    n_samples : int
        Number of query points used to estimate Cao's statistic.
    n_reference_samples : int, optional
        Maximum number of valid embedded points in the nearest-neighbour
        reference pool. The default, ``None``, preserves the original exact
        calculation against every embedded point. Set this for very long
        recordings to retain Cao's E1 calculation with a deterministic,
        bounded nearest-neighbour pool.
    verbose : bool, default False
        If True, print progress before and after each embedding dimension.

    Returns
    -------
    E1 : (max_d-1,) ndarray
        E_1(d) for d = 1..max_d-1.  Saturation indicates the embedding
        dimension; first d at which E_1 plateaus is the manuscript's choice.
    """
    import time
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2:
        raise ValueError(f"Expected a 1-D or 2-D time series; got shape {X.shape}")
    T, p = X.shape
    max_d = int(max_d)
    tau = int(tau)
    n_valid = T - max_d * tau - 2
    if n_valid <= 1:
        raise ValueError(
            f"Time series has {T} rows, too short for max_d={max_d}, tau={tau}"
        )

    bounded_reference = n_reference_samples is not None
    if bounded_reference and n_valid > int(n_reference_samples):
        reference_idx = np.sort(
            rng.choice(n_valid, size=int(n_reference_samples), replace=False)
        )
    elif bounded_reference:
        reference_idx = np.arange(n_valid)
    else:
        reference_idx = None

    if n_valid > int(n_samples):
        query_idx = rng.choice(n_valid, size=int(n_samples), replace=False)
    else:
        query_idx = np.arange(n_valid)
    if bounded_reference:
        if len(reference_idx) > int(n_samples):
            query_positions = np.sort(
                rng.choice(len(reference_idx), size=int(n_samples), replace=False)
            )
        else:
            query_positions = np.arange(len(reference_idx))
        query_idx = reference_idx[query_positions]

    if verbose:
        reference_rows = len(reference_idx) if bounded_reference else n_valid
        print(
            "Cao E1 scan: "
            f"T={T:,}, features={p}, max_d={max_d}, tau={tau}, "
            f"query_rows={len(query_idx):,}, reference_rows={reference_rows:,}",
            flush=True,
        )

    def embed_at(indices, d):
        E = np.empty((len(indices), d * p), dtype=X.dtype)
        for k in range(d):
            E[:, k * p:(k + 1) * p] = X[indices + k * tau]
        return E

    def embed_all(d):
        n_rows = T - (d - 1) * tau
        E = np.empty((n_rows, d * p), dtype=X.dtype)
        for k in range(d):
            E[:, k * p:(k + 1) * p] = X[k * tau:k * tau + n_rows]
        return E

    E1 = np.zeros(max_d - 1)
    a_prev = None
    for d in range(1, max_d):
        t0 = time.perf_counter()
        if verbose:
            print(
                f"[Cao E1] d={d}/{max_d - 1}: building embeddings "
                f"({d * p} -> {(d + 1) * p} dimensions)",
                flush=True,
            )
        if bounded_reference:
            Ed = embed_at(reference_idx, d)
            Edp = embed_at(reference_idx, d + 1)
            sub_d = Ed[query_positions]
            sub_dp = Edp[query_positions]
        else:
            Ed = embed_all(d)
            Edp = embed_all(d + 1)
            sub_d = Ed[query_idx]
            sub_dp = Edp[query_idx]
        if verbose:
            print(
                f"[Cao E1] d={d}/{max_d - 1}: querying nearest neighbors "
                f"against {Ed.shape[0]:,} reference rows",
                flush=True,
            )
        tree = cKDTree(Ed)
        dists, nbrs = tree.query(sub_d, k=2)
        nn = nbrs[:, 1]
        valid = nn < Edp.shape[0]
        dnn = np.maximum(dists[valid, 1], 1e-12)
        dnp = np.linalg.norm(sub_dp[valid] - Edp[nn[valid]], axis=1)
        a_d = np.mean(dnp / dnn)
        if a_prev is not None:
            E1[d - 2] = a_d / a_prev
            if verbose:
                print(
                    f"[Cao E1] d={d}/{max_d - 1}: E1({d - 1})={E1[d - 2]:.6g} "
                    f"completed in {time.perf_counter() - t0:.1f}s",
                    flush=True,
                )
        elif verbose:
            print(
                f"[Cao E1] d={d}/{max_d - 1}: baseline a_d={a_d:.6g} "
                f"completed in {time.perf_counter() - t0:.1f}s",
                flush=True,
            )
        a_prev = a_d
    return E1


# ---------------------------------------------------------------------------
# Delay embedding (Methods Sec. "Delay embedding").
# ---------------------------------------------------------------------------
def delay_embed(X, d, tau=1):
    """Delay-embed a (T, p) time series into (T - (d-1)*tau, d*p)."""
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2:
        raise ValueError(f"Expected a 1-D or 2-D time series; got shape {X.shape}")
    d = int(d)
    tau = int(tau)
    if d < 1 or tau < 1:
        raise ValueError(f"d and tau must be positive integers; got d={d}, tau={tau}")
    T, p = X.shape
    l = T - (d - 1) * tau
    if l <= 0:
        raise ValueError(
            f"Time series has {T} rows, too short for d={d}, tau={tau}"
        )
    # Preserve float32 projections. The previous implicit float64 conversion
    # doubled peak memory and was immediately copied back to float32 by callers.
    out = np.empty((l, d * p), dtype=X.dtype)
    for k in range(d):
        out[:, k * p:(k + 1) * p] = X[k * tau:k * tau + l]
    return out


# ---------------------------------------------------------------------------
# K-means partition (Methods Sec. "K-means partitioning").  Uses
# MiniBatchKMeans with n_init=20 to match the analyses in the paper.
# ---------------------------------------------------------------------------
def kmeans_partition(X, N, batch_size=None, n_init=20, seed=None,
                     return_centers=False):
    """MiniBatchKMeans partition into N clusters.

    Returns the (T,) sequence of integer cluster labels, and optionally the
    (N, d) array of cluster centers.
    """
    if batch_size is None:
        batch_size = N * 5
    km = MiniBatchKMeans(n_clusters=N, batch_size=batch_size, n_init=n_init,
                         random_state=seed, init='random').fit(X)
    if return_centers:
        return km.labels_, km.cluster_centers_
    return km.labels_


# ---------------------------------------------------------------------------
# Entropy-gap criterion for choosing N (Methods Sec. "Choosing the number of
# clusters").  We compare the per-step Markov entropy of the true sequence to
# that of a Shannon-shuffled surrogate; the optimal N maximises the gap.
# ---------------------------------------------------------------------------
def markov_entropy(states, lag, framerate=1.0):
    """Shannon entropy rate (nats / time-unit) of the lag-tau Markov model
    estimated from a single sequence.  (Natural log, matching the entropy-gap
    figures, which are labelled in nats.)"""
    T = make_transition_matrix(states, lag)
    pi = stationary_distribution(T)
    h = 0.0
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            if T[i, j] > 0:
                h -= pi[i] * T[i, j] * np.log(T[i, j])
    return h * framerate


def shannon_shuffle(states, seed=None):
    """Shuffle labels in time while preserving empirical state occupancy.

    This destroys temporal correlations and gives the high-entropy surrogate
    used by the entropy-gap criterion for choosing ``N``.
    """
    rng = np.random.default_rng(seed)
    states = np.asarray(states, dtype=int)
    return rng.permutation(states)


def entropy_gap(X_embedded, N_values, lag, framerate=1.0, seed=0,
                n_init=20, verbose=False):
    """Compute Delta H(N) = H_shuf(N) - H(N) for each N in N_values, on a
    single delay-embedded time series.

    Returns N_values (ndarray), H (ndarray), H_shuf (ndarray)."""
    rng_seed = seed
    H = np.zeros(len(N_values))
    H_shuf = np.zeros(len(N_values))
    for k, N in enumerate(N_values):
        labels = kmeans_partition(X_embedded, N, n_init=n_init, seed=rng_seed)
        H[k] = markov_entropy(labels, lag, framerate=framerate)
        labels_shuf = shannon_shuffle(labels, seed=rng_seed)
        H_shuf[k] = markov_entropy(labels_shuf, lag, framerate=framerate)
        if verbose:
            print(f'  N={N}: H={H[k]:.3f}  H_shuf={H_shuf[k]:.3f}  '
                  f'DeltaH={H_shuf[k] - H[k]:.3f}')
    return np.array(N_values), H, H_shuf


# ---------------------------------------------------------------------------
# Transition matrix and eigendecomposition (Methods Sec. "Transfer operator
# estimation").  Returns the row-stochastic T(tau).
# ---------------------------------------------------------------------------
def make_transition_matrix(states, lag, n_states=None):
    """Row-stochastic transition matrix at the given lag.

    The element T[i, j] is the empirical probability of being in state j at
    time t + lag conditional on being in state i at time t.

    `states` may be a single 1-D integer sequence, OR a list/tuple of
    per-individual sequences.  When a list is given, transitions are counted
    *within* each sequence only and never across the boundary between
    individuals (or recording segments) -- pooling with ``np.concatenate``
    instead would inject ``lag`` spurious splice transitions per boundary, so
    always pass a list when you have more than one recording.

    Rows for states that never appear are set to a uniform distribution to
    keep T well-defined (downstream G-PCCA rejects all-zero rows).
    """
    if isinstance(states, (list, tuple)):
        seqs = [np.asarray(s, dtype=int) for s in states]
    else:
        seqs = [np.asarray(states, dtype=int)]
    if n_states is None:
        n_states = int(max(int(s.max()) for s in seqs if s.size)) + 1
    a = np.arange(n_states + 1)
    F = np.zeros((n_states, n_states))
    for s in seqs:
        if s.size > lag:
            Fi, _, _ = np.histogram2d(s[:-lag], s[lag:], bins=[a, a])
            F += Fi
    sums = F.sum(axis=1, keepdims=True)
    # Empty rows (state never appears as a "from") get a uniform distribution
    # rather than all zeros, so T remains a valid (row-stochastic) transition
    # matrix.  Without this, downstream G-PCCA rejects T with "not a transition
    # matrix" when running on short per-individual sub-sequences.
    empty = (sums.ravel() == 0)
    sums[empty, 0] = 1.0
    F[empty] = 1.0 / n_states
    return F / sums


def stationary_distribution(T):
    """Stationary distribution of a row-stochastic T (left eigenvector at 1)."""
    vals, vecs = np.linalg.eig(T.T)
    idx = int(np.argmin(np.abs(vals - 1.0)))
    pi = np.abs(vecs[:, idx].real)
    pi /= pi.sum()
    return pi


def leading_eigvecs(T, k=10, drop_stationary=True):
    """Leading right eigenvectors of T, ordered by descending |lambda|.

    Returns
    -------
    eigvals : (k,) complex ndarray
        Eigenvalues, ordered by descending magnitude (with the stationary
        eigenvalue at 1 dropped if drop_stationary).
    eigvecs : (N, k) complex ndarray
        Right eigenvectors, columns matching `eigvals`.
    """
    vals, vecs = np.linalg.eig(T)
    order = np.argsort(np.abs(vals))[::-1]
    keep_v, keep_idx = [], []
    for idx in order:
        v = vals[idx]
        if drop_stationary and np.isclose(v, 1.0):
            continue
        keep_idx.append(idx); keep_v.append(v)
        if len(keep_idx) >= k:
            break
    return np.array(keep_v), vecs[:, keep_idx]


def phi2_correlation_preet(states, h, fs=100.0, savgol_window=150,
                           savgol_polyorder=5):
    """Pearson |r(phi_2(t), h(t))| with the per-seed characteristic-timescale
    lag and Savitzky-Golay smoothing used in the manuscript Fig. 2D sweep.

    Recipe (Methods Sec. "Lorenz panel D"; equivalent to Preet's
    `compute_lorenz_corr_seeds.py` in the original repo):

        1. Build T(lag = 1) and compute t_c = -1 / log|lambda_2| (in frames).
        2. Build T(lag = round(t_c)).
        3. Sort the eigenvalues of T by descending real part; the leading
           non-trivial eigenvector is at index 1 (index 0 is the trivial
           lambda = 1 mode).
        4. Project per-frame as `phi_2(t) = phi_2_cluster[states]` and smooth
           with `savgol_filter(window_length=150, polyorder=5)`.
        5. Return |Pearson r| against the (length-aligned) hidden driver h.

    Returns (r_pearson, lag_used_in_frames, t_c_in_frames).
    Returns (np.nan, np.nan, np.nan) if t_c is not finite.
    """
    from scipy.signal import savgol_filter
    from scipy.stats import pearsonr
    states = np.asarray(states, dtype=int)
    n_states_ = int(states.max()) + 1

    # Step 1: characteristic timescale from a lag=1 transition matrix.
    T1 = make_transition_matrix(states, lag=1, n_states=n_states_)
    eigvals1 = np.linalg.eig(T1)[0]
    abs_sorted = np.sort(np.abs(eigvals1))[::-1]
    lam2 = abs_sorted[1] if len(abs_sorted) > 1 else np.nan
    if not (0 < lam2 < 1):
        return np.nan, np.nan, np.nan
    t_c = -1.0 / np.log(lam2)
    lag = max(int(round(t_c)), 1)

    # Step 2-3: eigenvectors at lag = round(t_c), sort by real eigvalue.
    Tlag = make_transition_matrix(states, lag=lag, n_states=n_states_)
    eigvals, eigvecs = np.linalg.eig(Tlag)
    sorted_idx = np.argsort(eigvals.real)[::-1]
    phi2_cluster = eigvecs[:, sorted_idx[1]].real

    # Step 4-5: project, smooth, correlate.
    phi2_t = phi2_cluster[states]
    h_trim = h[: len(phi2_t)]
    phi2_sm = savgol_filter(phi2_t, savgol_window, savgol_polyorder)
    r, _ = pearsonr(phi2_sm, h_trim)
    return abs(r), lag, t_c


def participation_ratio(phi):
    """Participation ratio of one or more eigenvectors.

    PR = (sum_i phi_i^2)^2 / sum_i phi_i^4.

    Near 1 when phi is concentrated on a single state, near N when uniform.
    """
    phi = np.asarray(phi)
    if phi.ndim == 1:
        phi = phi[:, None]
    s2 = (phi.real ** 2).sum(axis=0)
    s4 = (phi.real ** 4).sum(axis=0)
    return s2 ** 2 / np.maximum(s4, 1e-30)


# ---------------------------------------------------------------------------
# Multi-timescale wavelet pipeline as a single convenience call.
# ---------------------------------------------------------------------------
def multi_timescale_pipeline(x, fs, fmin, fmax, n_freqs, n_pcs=None,
                             d=None, tau_embed=1, N=None, lag=None,
                             omega0=5.0, seed=None, verbose=False):
    """Run the full multi-timescale pipeline on a multivariate signal.

    Returns a dict with intermediate products: amplitudes, projections,
    n_pcs_kept, X_embedded, states, T, eigvals, eigvecs, pi.
    Each of these is also obtainable individually via the lower-level
    functions above.
    """
    if verbose:
        print(f'  computing wavelets ({n_freqs} bands {fmin:.2f}-{fmax:.2f} Hz)')
    amps, f = morlet_wavelet_amplitudes(x, fs, fmin, fmax, n_freqs, omega0)
    if verbose:
        print(f'  PCA on amplitudes (shape {amps.shape})')
    if n_pcs is None:
        proj, n_kept, eigvals, thresh = pca_with_shuffle_threshold(amps, seed=seed or 0)
    else:
        pca = PCA(n_components=n_pcs).fit(amps)
        proj = pca.transform(amps)
        n_kept = n_pcs; eigvals = pca.explained_variance_; thresh = None
    if verbose:
        print(f'  kept {n_kept} PCs')
    if d is None:
        raise ValueError('d (embedding dimension) must be provided')
    X_embed = delay_embed(proj, d, tau_embed)
    if verbose:
        print(f'  delay embedding -> {X_embed.shape}')
    if N is not None:
        states = kmeans_partition(X_embed, N, seed=seed)
    else:
        states = None
    out = dict(amplitudes=amps, frequencies=f, projections=proj,
               n_pcs_kept=n_kept, eigvals_pca=eigvals, shuffle_threshold=thresh,
               X_embedded=X_embed, states=states)
    if states is not None and lag is not None:
        T = make_transition_matrix(states, lag)
        evals, evecs = leading_eigvecs(T, k=10)
        pi = stationary_distribution(T)
        out.update(T=T, eigvals=evals, eigvecs=evecs, pi=pi)
    return out


# ---------------------------------------------------------------------------
# Metastable residence times (Methods Sec. "Dwell-time distributions").
# Smooth the per-frame soft membership chi[states] with a Delta-second moving
# average, assign each frame to its dominant basin (argmax), and take the
# durations of maximal same-basin runs.  This single definition is used for
# every dwell-time figure (main Fig. 5D; Supp. Figs. S4, S5, S8, S10, S11),
# so they are mutually consistent.
# ---------------------------------------------------------------------------
def residences_per_individual(states, chi, framerate, delta=2.0):
    """Per-individual metastable residence times, in seconds.

    Parameters
    ----------
    states : list of (T_i,) int arrays
        Per-individual cluster-index sequences (one per worm/fly).
    chi : (N_clusters, M) ndarray
        Soft basin memberships from G-PCCA.
    framerate : float
        Sampling rate in Hz.
    delta : float
        Membership smoothing window in seconds (boxcar moving average).
        delta = 0 uses the raw per-frame argmax (sub-second flicker); the
        manuscript working value is 2 s.

    Returns
    -------
    list (len = n_individuals) of lists (len = M) of float arrays
        residences[i][b] = basin-b residence durations (s) for individual i.
    """
    from scipy.ndimage import uniform_filter1d
    chi = np.asarray(chi)
    M = chi.shape[1]
    w = max(1, int(round(delta * framerate)))
    per_indiv = []
    for s in states:
        s = np.asarray(s, dtype=int)
        m = chi[s]
        if delta > 0:
            m = uniform_filter1d(m, size=w, axis=0, mode='nearest')
        lab = np.argmax(m, axis=1)
        runs = [[] for _ in range(M)]
        cur = int(lab[0]); n = 1
        for a in lab[1:]:
            if a == cur:
                n += 1
            else:
                runs[cur].append(n); cur = int(a); n = 1
        runs[cur].append(n)
        per_indiv.append([np.asarray(r, float) / framerate for r in runs])
    return per_indiv


def metastable_residences(states, chi, framerate, delta=2.0):
    """Per-basin metastable residence times (seconds), pooled over individuals.

    Thin wrapper over `residences_per_individual` that concatenates each
    basin's residences across individuals.  Returns a list of M float arrays.
    """
    M = np.asarray(chi).shape[1]
    per_indiv = residences_per_individual(states, chi, framerate, delta=delta)
    pooled = []
    for b in range(M):
        parts = [pi[b] for pi in per_indiv if len(pi[b])]
        pooled.append(np.concatenate(parts) if parts else np.array([], float))
    return pooled
