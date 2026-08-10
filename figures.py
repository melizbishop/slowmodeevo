"""Shared figure styling and plotting helpers.

The aesthetics match those used in the published manuscript: Helvetica/Arial
sans-serif, no top/right spines, Okabe-Ito categorical palette for arms,
RdBu_r for diverging enrichment maps.

Usage::

    from figures import setup_style, ARM_PALETTE, plot_density_map, \
        plot_eigenspace_3d, plot_chi_timeseries, plot_eigenvalue_dots
    setup_style()
"""
from __future__ import annotations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# Okabe-Ito categorical palette (used for the four fly arms).
ARM_PALETTE = ['#E69F00', '#56B4E9', '#009E73', '#D55E00']
ARM_PALETTE_2 = ['#0072B2', '#D55E00']  # two-basin palette for worms


def _palette_for_n(n, palette=None):
    """Return at least n categorical colors."""
    if palette is None:
        palette = ARM_PALETTE
    palette = list(palette)
    if len(palette) >= n:
        return palette
    extra = list(plt.get_cmap('tab20').colors)
    extra = [mpl.colors.to_hex(c) for c in extra]
    colors = palette[:]
    for color in extra:
        if color not in colors:
            colors.append(color)
        if len(colors) >= n:
            break
    if len(colors) < n:
        reps = int(np.ceil(n / len(colors)))
        colors = (colors * reps)[:n]
    return colors


def setup_style():
    """Set Matplotlib rcParams to match the manuscript aesthetic."""
    # fontTools complains about Apple-specific TrueType tables (Zapf, feat,
    # morx) it can't subset when matplotlib embeds Type 42 fonts in PDFs.
    # The tables are dropped harmlessly; silence the noise.
    import logging
    logging.getLogger('fontTools').setLevel(logging.ERROR)
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.8,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'legend.fontsize': 8, 'legend.frameon': False,
        'figure.dpi': 110, 'savefig.dpi': 260, 'savefig.bbox': 'tight',
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })


# ---------------------------------------------------------------------------
# RdBu_r enrichment density map (used for Fig 2B Lorenz, Fig 3C/D worms,
# Fig 5A flies).  Bins (x, y) into a regular grid and renders the per-bin
# average of c with a diverging colormap, masking bins with too few visits.
# ---------------------------------------------------------------------------
def plot_density_map(ax, x, y, c, n_bins=80, min_count=20, vmax=None,
                     cmap='RdBu_r', show_frame=True, pad_frac=0.04):
    """Per-bin mean-c heatmap, RdBu_r centered at 0.

    Parameters
    ----------
    ax : matplotlib axes
    x, y : (T,) ndarray
        Coordinates for each frame.
    c : (T,) ndarray
        Quantity whose per-bin mean we display (e.g. a soft membership chi_j,
        or an eigenvector coordinate phi_2).
    n_bins : int
    min_count : int
        Minimum number of frames in a bin for it to be drawn (else masked).
    vmax : float, optional
        Color scale bound (-vmax, +vmax).  Default: 99th pct of |mean_c|.
    show_frame : bool
        Draw a thin rectangle around the data extent (matches Fig 3 style).
    """
    x = np.asarray(x); y = np.asarray(y); c = np.asarray(c)
    H_count, xe, ye = np.histogram2d(x, y, bins=n_bins)
    H_sum, _, _ = np.histogram2d(x, y, bins=[xe, ye], weights=c)
    with np.errstate(invalid='ignore', divide='ignore'):
        H_mean = np.where(H_count >= min_count, H_sum / H_count, np.nan)
    if vmax is None:
        vals = H_mean[~np.isnan(H_mean)]
        vmax = float(np.nanpercentile(np.abs(vals), 99)) if vals.size else 1.0
    pcm = ax.pcolormesh(xe, ye, H_mean.T, cmap=cmap, vmin=-vmax, vmax=vmax,
                        shading='auto', rasterized=True)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ('top', 'right', 'bottom', 'left'):
        ax.spines[s].set_visible(False)
    if show_frame:
        xlo, xhi = xe[0], xe[-1]; ylo, yhi = ye[0], ye[-1]
        dx = (xhi - xlo) * pad_frac; dy = (yhi - ylo) * pad_frac
        rect = Rectangle((xlo - dx, ylo - dy), (xhi - xlo) + 2 * dx,
                         (yhi - ylo) + 2 * dy,
                         fill=False, lw=0.7, ec='0.4',
                         transform=ax.transData)
        ax.add_patch(rect)
        ax.set_xlim(xlo - dx, xhi + dx); ax.set_ylim(ylo - dy, yhi + dy)
    return pcm


# ---------------------------------------------------------------------------
# Cluster scatter in a 3D eigenspace, with arms drawn from a hub.
# ---------------------------------------------------------------------------
def plot_eigenspace_3d(ax, phi, pi=None, chi=None, hub=None, arm_dirs=None,
                       arm_centroids=None, palette=None, marker_scale=18,
                       arm_extension=1.0, view=(28, -30), axis_order=(0, 1, 2),
                       axis_labels=(r'$\phi_2$', r'$\phi_3$', r'$\phi_4$')):
    """3D scatter plot of cluster positions colored by basin.

    Parameters
    ----------
    ax : 3D matplotlib axes (projection='3d')
    phi : (N, 3) ndarray
        Cluster positions in eigenspace (columns = phi_2, phi_3, phi_4 by
        convention).
    axis_order : (i, j, k) of column indices
        Which columns of `phi` go on the (x, y, z) plot axes.  Default
        (0, 1, 2) plots phi_2 on x, phi_3 on y, phi_4 on z.  The published
        Fig 4B uses (1, 2, 0) -- phi_3 on x, phi_4 on y, phi_2 on z.
    axis_labels : tuple of three strings
        Axis label LaTeX (paired with axis_order so the labels stay correct
        when the columns are permuted).
    pi, chi, hub, arm_dirs, arm_centroids, palette, marker_scale,
    arm_extension, view : see manuscript Methods Sec. "Arm geometry".
    """
    if chi is not None:
        n_colors = chi.shape[1]
    elif arm_dirs is not None:
        n_colors = len(arm_dirs)
    else:
        n_colors = len(ARM_PALETTE if palette is None else palette)
    palette = _palette_for_n(n_colors, palette)
    if marker_scale is None:
        marker_scale = 18
    if pi is None:
        sizes = np.full(phi.shape[0], marker_scale)
    else:
        s = np.sqrt(pi / pi.max() + 1e-9)
        sizes = marker_scale * s / s.max()
    if chi is not None:
        assignments = chi.argmax(axis=1)
        colors = [palette[a] for a in assignments]
    else:
        colors = '0.4'
    i, j, k = axis_order
    ax.scatter(phi[:, i], phi[:, j], phi[:, k], s=sizes, c=colors,
               alpha=0.65, edgecolors='none', depthshade=False)
    if hub is not None:
        ax.scatter([hub[i]], [hub[j]], [hub[k]], marker='*',
                   s=180, c='red', edgecolors='k', linewidths=0.7,
                   zorder=10)
        if arm_dirs is not None:
            for jj, d in enumerate(arm_dirs):
                if arm_centroids is not None:
                    L = np.linalg.norm(arm_centroids[jj] - hub) * arm_extension
                else:
                    L = 1.0
                end = hub + L * d
                ax.plot([hub[i], end[i]], [hub[j], end[j]], [hub[k], end[k]],
                        color=palette[jj], lw=2.0, alpha=0.95)
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_xlabel(axis_labels[0]); ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])


# ---------------------------------------------------------------------------
# Chi_j(t) time series, smoothed for visual clarity.
# ---------------------------------------------------------------------------
def plot_chi_timeseries(ax, t, chi_traces, palette=None, smoothing_window=None,
                        labels=None):
    """Soft basin memberships chi_j(t) (Fig 4D, 3D-equivalent panel).

    Parameters
    ----------
    ax : axes
    t : (T,) ndarray
        Time values.
    chi_traces : (T, M) ndarray
        Memberships per frame.
    palette : list of M colors
    smoothing_window : int, optional
        Boxcar smoothing window in frames (e.g. 5 * fs for a 5-s window).
    """
    if palette is None:
        palette = ARM_PALETTE
    M = chi_traces.shape[1]
    palette = _palette_for_n(M, palette)
    if labels is None:
        labels = [fr'$\chi_{{{j+1}}}$' for j in range(M)]
    if smoothing_window is not None and smoothing_window > 1:
        kernel = np.ones(smoothing_window) / smoothing_window
        smoothed = np.stack([np.convolve(chi_traces[:, j], kernel, mode='same')
                             for j in range(M)], axis=1)
    else:
        smoothed = chi_traces
    for j in range(M):
        ax.plot(t, smoothed[:, j], color=palette[j], lw=1.0, label=labels[j])
    ax.set_ylim(0, 1)
    ax.set_xlabel('time (s)')
    ax.set_ylabel(r'$\chi_j(t)$')
    ax.legend(loc='upper right', ncol=M)


# ---------------------------------------------------------------------------
# Eigenvalue bar plot with a dashed line marking the spectral-gap-picked M.
# ---------------------------------------------------------------------------
def plot_eigenvalue_bars(ax, eigvals, M=None, color='0.25', dashed_color='0.6'):
    """Bars of |lambda_k| with the M+1 boundary marked (Fig 3B / Fig 4C)."""
    e = np.abs(np.asarray(eigvals))
    k = np.arange(1, len(e) + 1)
    ax.bar(k, e, width=0.7, color=color, edgecolor='none')
    if M is not None:
        ax.axvline(M + 0.5, ls='--', color=dashed_color, lw=0.9)
    ax.set_xlabel(r'eigenvalue index $k$')
    ax.set_ylabel(r'$|\lambda_k|$')
    ax.set_xticks(k[::1] if len(k) <= 10 else k[::2])
    ax.set_ylim(0, 1.02)


def plot_eigenvalue_dots(ax, eigvals, M=None, color='0.25', dashed_color='0.6',
                         marker_size=36):
    """Same as plot_eigenvalue_bars, but with markers instead of bars."""
    e = np.abs(np.asarray(eigvals))
    k = np.arange(1, len(e) + 1)
    ax.scatter(k, e, s=marker_size, color=color, edgecolors='none', zorder=3)
    if M is not None:
        ax.axvline(M + 0.5, ls='--', color=dashed_color, lw=0.9)
    ax.set_xlabel(r'eigenvalue index $k$')
    ax.set_ylabel(r'$|\lambda_k|$')
    ax.set_xticks(k[::1] if len(k) <= 10 else k[::2])
    ax.set_ylim(0, 1.02)


# ---------------------------------------------------------------------------
# Quick utility: save a figure to outputs/ as both PNG and PDF.
# ---------------------------------------------------------------------------
def save_panel(fig, name, out_dir='outputs', formats=('png', 'pdf')):
    """Save `fig` to out_dir/name.{png,pdf}.  Creates out_dir if absent."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    for fmt in formats:
        fig.savefig(f'{out_dir}/{name}.{fmt}')


# ---------------------------------------------------------------------------
# Power spectral density, to choose the wavelet frequency band [fmin, fmax]
# (the one system-specific physical input of the pipeline).
# ---------------------------------------------------------------------------
def plot_psd(X, fs, fmin=None, fmax=None, nperseg=None, ax=None, labels=None):
    """Welch PSD of each channel (log-log), as an aid to picking the wavelet band.

    Manuscript rule of thumb (Methods, "Wavelet decomposition"): set ``fmin``
    at the frequency where the PSD leaves the broadband noise floor and rises
    into a clear spectral band, and ``fmax`` where the PSD returns to baseline
    (or the Nyquist frequency if there is no clear high-frequency rolloff).

    Parameters
    ----------
    X : (T,) or (T, n_channels) ndarray   Measurement time series.
    fs : float                            Sampling rate (Hz).
    fmin, fmax : float, optional          Candidate band; drawn as dashed lines.
    nperseg : int, optional               Welch segment length (default min(4096, T)).
    ax : matplotlib axes, optional        Axes to draw into (created if None).
    labels : list of str, optional        Per-channel legend labels.

    Returns
    -------
    freqs, psd : ndarrays                 Frequencies (Hz) and per-channel PSD.
    """
    from scipy.signal import welch
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    T, n_ch = X.shape
    if nperseg is None:
        nperseg = min(4096, T)
    if ax is None:
        ax = plt.subplots(figsize=(4.6, 3.0))[1]
    freqs, P = welch(X, fs=fs, nperseg=nperseg, axis=0)
    for c in range(n_ch):
        lab = labels[c] if labels is not None and c < len(labels) else None
        ax.loglog(freqs[1:], P[1:, c], lw=0.9, alpha=0.75, label=lab)
    if fmin is not None:
        ax.axvline(fmin, color='r', ls='--', lw=0.8)
    if fmax is not None:
        ax.axvline(fmax, color='r', ls='--', lw=0.8)
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel('PSD')
    if labels is not None:
        ax.legend(fontsize=7, frameon=False)
    return freqs, P
