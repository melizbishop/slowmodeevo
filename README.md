# Multi-timescale transfer-operator pipeline — supplementary code

This directory contains the supplementary code accompanying

> Kaur, Jain, & Berman, *Using timescale as a state coordinate reveals the
> metastable geometry of behavior* (2026).

The repository has two layers:

1. **`figures/`** — One Python script per published figure (Fig. 2 panels
   B–E, Fig. 3, Fig. 4, Fig. 5, and Supp. Figs. S1–S11).  Each script loads
   precomputed inputs from `data/` and writes PNG and PDF outputs to
   `outputs/`.  Running `python figures/run_all.py` regenerates every panel
   in 5–10 minutes.
2. **`*.ipynb` notebooks** — End-to-end pipeline demonstrations (wavelet
   decomposition → PCA → delay embedding → clustering → transition matrix
   → G-PCCA) for each system.  The notebooks show how the cached
   `data/` files were produced and let you re-run the upstream steps on
   your own data via `user_data.ipynb`.

If you only want to reproduce the published figures, you can ignore the
notebooks and use `figures/` directly.

## Pipeline at a glance

For every system, the pipeline runs:

1. **Wavelet decomposition** (Morlet, $\omega_0 = 5$) of each measurement
   channel into 25 dyadically spaced frequency bands.
2. **PCA** on the wavelet amplitudes, keeping the components that exceed a
   temporal-shuffle threshold (each feature permuted in time; not a phase
   randomization).
3. **Delay embedding** at dimension $d$, chosen by Cao's $E_1(d)$
   saturation criterion.
4. **K-means partition** into $N$ clusters, with $N$ chosen by the entropy
   gap $\Delta H(N)$.
5. **Transition matrix** $T(\tau)$ at a working lag $\tau$ chosen by the
   implied-timescales plateau.
6. **Eigendecomposition** of $T$ and selection of the number of basins $M$
   from the largest spectral-gap ratio $|\lambda_M| / |\lambda_{M+1}|$.
7. **G-PCCA** (Reuter–Weber 2018) for soft basin memberships
   $\chi_j(i) \in [0, 1]$.
8. **Hub-and-arm geometry** in the leading non-trivial eigenvectors:
   $\pi$-weighted hub + per-basin arm vectors as centroid$-$hub.

## All-pair keypoint-distance workflow

Use `all_pair_distance_pipeline.ipynb` as the single entry point for the
multispecies mouse analysis. It:

- generates all unique pairwise keypoint distances from the source HDF5 data;
- supports projection reuse and single-species tuning of $d$, $N$, $\tau$, and
  basin count $M$;
- runs one species or the complete multispecies batch;
- compares slow-mode eigenspace arms, transfer dynamics, basin occupancy, and
  hierarchical species structure; and
- joins frame assignments to named MoSeq syllables to describe behavior within
  each basin.

Reusable implementation lives in `all_pair_distance_data.py`,
`pooled_user_pipeline.py`, and `multi_species_comparison.py`. Heavy notebook
stages are guarded by explicit `RUN_*` flags, and existing saved batch runs are
resumed from `outputs/single_species_distance_comparison/`.

To check a candidate representation on your own data, `diagnostics.run_diagnostics`
evaluates the paper's four falsifiable criteria (spectral gap, participation
ratio, simplex/arms geometry, and held-out prediction beating a memoryless
null) and prints a pass/fail table; `diagnostics.stationarity_drift` flags
violations of the single-stationary-distribution assumption. When pooling
multiple recordings, pass a **list** of per-individual cluster sequences to
`pipeline.make_transition_matrix` so transitions are never counted across the
splice between individuals.

## Repository layout

```
slowmode/
├── pipeline.py             # core algorithms (wavelets, PCA, embed, kmeans, T)
├── gpcca_utils.py          # G-PCCA wrapper, hub/arm geometry
├── diagnostics.py          # the four falsifiable criteria + run_diagnostics()
├── figures.py              # shared matplotlib styling, ARM_PALETTE, plot_psd
├── lorenz_simulation.py    # standalone driven-Lorenz simulator
├── requirements.txt
│
├── figures/                # one script per published figure
│   ├── _paths.py           # repo path setup + setup() style helper
│   ├── figure_2B.py
│   ├── figure_2C.py
│   ├── figure_2D.py
│   ├── figure_2E.py
│   ├── figure_3.py
│   ├── figure_4.py
│   ├── figure_5.py
│   ├── supp_figure_1.py
│   ├── supp_figure_2.py
│   ├── supp_figure_3.py
│   ├── supp_figure_4.py
│   ├── supp_figure_5.py
│   ├── supp_figure_6.py
│   ├── supp_figure_7.py
│   ├── supp_figure_8.py
│   ├── supp_figure_9.py
│   ├── supp_figure_10.py
│   ├── supp_figure_11.py
│   └── run_all.py          # convenience: run every script in order
│
├── data/                   # precomputed inputs shipped with the repo (~120 MB)
│   ├── lorenz/
│   ├── worms/
│   └── flies/
│
├── outputs/                # PNG + PDF written by figure scripts
│
├── lorenz.ipynb            # pipeline demo (Lorenz)
├── worms.ipynb             # pipeline demo (C. elegans)
├── flies.ipynb             # pipeline demo (Drosophila)
├── user_data.ipynb         # pipeline applied to your own time series
│
└── README.md
```

## Figure-to-script map

The numbering matches the published manuscript.  Fig. 1 is a hand-drawn
pipeline schematic and has no Python source.  Fig. 2 panel A (driver
double-well sketch + sample $h(t)$ trace) is generated inside
`lorenz.ipynb`; panels B–E are mirrored as separate scripts so they can be
inspected and re-rendered independently (the published figure is assembled
in Illustrator from these per-panel outputs).

| Figure | Script | Notes |
|--------|--------|-------|
| Fig. 2A | `lorenz.ipynb` | Driver double-well + sample $h(t)$ |
| Fig. 2B | `figures/figure_2B.py` | 3D attractor + (x,y),(y,z),(x,z) projections coloured by $\phi_2$ |
| Fig. 2C | `figures/figure_2C.py` | $|\lambda_k(\tau)|$ multi vs fixed at $\beta=0.5$ |
| Fig. 2D | `figures/figure_2D.py` | $h(t)$, multi $\phi_2$, fixed $\phi_2$ time series |
| Fig. 2E | `figures/figure_2E.py` | Pearson $|r(\phi_2, h)|$ vs mean dwell time, multi vs fixed, $\beta$ sweep |
| Fig. 3 | `figures/figure_3.py` | Worms (panels A–E) |
| Fig. 4 | `figures/figure_4.py` | Flies, methodology (panels A–E) |
| Fig. 5 | `figures/figure_5.py` | Flies, biology (panels A–D) |
| S1 | `figures/supp_figure_1.py` | Lorenz: Cao $E_1$ fixed + multi; mean dwell vs $\beta$; entropy gap vs $N$ |
| S2 | `figures/supp_figure_2.py` | Worm operator diagnostics |
| S3 | `figures/supp_figure_3.py` | Worm biological correspondence + leave-one-worm-out CV |
| S4 | `figures/supp_figure_4.py` | 12-worm grid of dwell-time CCDFs |
| S5 | `figures/supp_figure_5.py` | Worm residence-time model selection + slow-mode shape |
| S6 | `figures/supp_figure_6.py` | Worm time-evolving landscape $V(\phi_2, t)$ |
| S7 | `figures/supp_figure_7.py` | Fly parameter selection + basin-count justification |
| S8 | `figures/supp_figure_8.py` | Fly residence-time model selection + slow-mode shape |
| S9 | `figures/supp_figure_9.py` | Fly reproducibility / individuality |
| S10 | `figures/supp_figure_10.py` | One-step-Markov surrogate dwell null (flies + worms) |
| S11 | `figures/supp_figure_11.py` | Residence-time robustness to smoothing scale $\Delta$ |

## Quick start: regenerate all paper figures

```bash
pip install -r requirements.txt          # installs numpy/scipy/matplotlib/pygpcca/powerlaw/umap-learn
python figures/run_all.py                # ~5--10 min, writes everything to outputs/
```

To render a single panel:

```bash
python figures/figure_3.py
```

## Installation

Python 3.10 or newer.  Required packages:

```
numpy
scipy
scikit-learn
matplotlib
pygpcca         # G-PCCA implementation (Reuter & Weber 2018)
powerlaw        # MLE power-law fits (Clauset et al. 2009)
umap-learn      # 2D embedding for worm/fly behavior maps
```

```bash
pip install numpy scipy scikit-learn matplotlib pygpcca powerlaw umap-learn
```

## Running on Google Colab

Each notebook starts with a cell that, when uncommented, will
`pip install` the missing dependencies and clone this repository:

```python
# !pip install pygpcca powerlaw umap-learn
# !git clone https://github.com/bermanlabemory/slowmode.git
# %cd slowmode
```

After that, `python figures/run_all.py` (or any individual script) runs
top-to-bottom.

## Data shipped in the repository

We ship enough precomputed input data in `data/` for every figure script to
run end-to-end.  Total size: ~125 MB.  The dwell-time figures (Fig. 5D and
Supp. Figs. S5, S8, S10, S11) recompute metastable residences on the fly from
the shipped cluster sequences (via `pipeline.metastable_residences`), so they
depend only on `states_*.pkl` and the G-PCCA memberships.

| File | Size | What |
|------|------|------|
| **Lorenz (`data/lorenz/`, 4.7 MB)** |||
| `lorenz_panel2B_data.npz` | 4.9 MB | Per-frame + per-cluster $\phi_2$ for Fig 2B |
| `lorenz_supp_data.npz` | 48 KB | Cao $E_1$, $\Delta h(N)$, dwell vs $\beta$, eigenvalue spectra |
| `lorenz_corr_sweep.npz` | 2 KB | Single-seed $\beta$ sweep for Fig 2D |
| `lorenz_corr_seeds.npz` | 4 KB | Multi-seed $\beta$ sweep with SEMs for Fig 2E |
| **Worms (`data/worms/`, 33 MB)** |||
| `states_worms.pkl` | 1.5 MB | Cluster sequences at $N = 250$, per worm × segment |
| `all_valid_segments_worms.pkl` | 17 MB | Raw eigenworm coefficients (Stephens 2008) for Fig 3A power spectrum |
| `worm_eigs_tau3s.npz` | 9 KB | Multi-timescale eigenvectors at $\tau = 3$ s |
| `gpcca_worms_M2_tau3s.npz` | 16 KB | G-PCCA basin membership $\chi$, $\pi$, eigvals |
| `gpcca_worms_M3_M4_tau3s.npz` | 18 KB | G-PCCA at deeper hierarchies $M = 3, 4$ |
| `worms_costa_per_cluster.npz` | 34 KB | Per-cluster $(\bar\omega, |\theta|)$ |
| `worms_umap_canonical_full.npz` | 15 MB | 2D UMAP grid for Fig 3 C/D, S3 A/B |
| `arm_dynamics_worms_tau3s.npz` | 25 KB | Basin-level MI and apparent decay rate |
| `worms_supp_data.npz` | 7 KB | Cao $E_1$, entropy gap, eigenvalue ladders |
| `M_eq_1_vs_M_eq_2_test.npz` | 10 KB | Leave-one-worm-out held-out MI vs null |
| `lognormal_reanalysis_worms.npz` | 89 KB | Slow-mode (GED) shape for S5C |
| **Flies (`data/flies/`, 88 MB)** |||
| `states_flies.pkl` | 43 MB | Multi-timescale cluster sequences at $N = 1000$ |
| `states_flies_fixed.pkl` | 43 MB | Fixed-timescale cluster sequences at $N = 3000$ |
| `fly_eigs_tau2s.npz` | 130 KB | Leading eigenvectors at $\tau = 2$ s |
| `gpcca_flies_M4_tau2s.npz` | 85 KB | G-PCCA basin membership at $M = 4$ |
| `arm_dynamics_results_tau2s.npz` | 871 KB | $r_k(\tau)$, predictive MI, per-fly stats |
| `behavior_density_chi_tau2s.npz` | 2.9 MB | $\chi$-weighted Berman 2014 behavior map |
| `pr_leave_one_out.npz` | 3 KB | Leave-one-fly-out PR contrast (Fig 4D) |
| `cv_flies_tau2s.npz` | 9 KB | Leave-one-fly-out CV of basin count |
| `per_fly_pcca_refit_tau2s.npz` | 10 KB | Per-fly G-PCCA refit cosines |
| `lognormal_reanalysis.npz` | 878 KB | Slow-mode (GED) shape for S8C |
| `flies_supp_method_data.npz` | 8 KB | PCA shuffle, Cao $E_1$, $\Delta h(N)$ |
| `flies_supp_method_v2_data.npz` | 3 KB | S7 $\tau$/$M$ sweeps + basin-count CV |

**Files not shipped (regenerable from raw recordings):**

| File | Approx. size | Where to get it |
|------|----|----|
| `flies_wlets_pca.pkl` | 612 MB | Recompute from joint-angle CSVs (`pipeline.morlet_wavelet_amplitudes` + `pipeline.pca_with_shuffle_threshold`); raw recordings are in the [Berman 2014 dataset](https://doi.org/...) |
| `flies_zvals.pkl` | 163 MB | Recompute from the same raw recordings using `MotionMapper` |
| `worms_wlets_pca.pkl` | 15 MB | Recompute from the [Stephens 2008 / Broekmans 2016](https://doi.org/...) eigenworm coefficients shipped in `all_valid_segments_worms.pkl` |

Recomputing these large intermediate files is what each `*.ipynb` notebook
shows; they're not needed if you just want to regenerate the figures.

## Pipeline demonstration notebooks

The notebooks document how the cached files in `data/` were produced and
serve as worked examples for applying the pipeline to your own data.
They do not generate the published figures themselves (the
`figures/*.py` scripts do); for figure regeneration you can skip the
notebooks entirely.

| Notebook | What it shows |
|----------|---------------|
| `lorenz.ipynb` | Driven-Lorenz simulation, wavelet pipeline, multi- vs fixed-timescale eigendecomposition, $\beta$ sweep |
| `worms.ipynb` | Eigenworm → wavelet PCA → cluster → transition matrix → G-PCCA at $M=2$ |
| `flies.ipynb` | Joint-angle → wavelet PCA → cluster → transition matrix → G-PCCA at $M=4$, plus the Berman 2014 behavior-map projection |
| `user_data.ipynb` | Applies the same pipeline to a generic $(T, d)$ time series of your choosing |

## Per-notebook runtime estimates

Rough estimates on a 2024 MacBook Pro without a GPU.

| Workflow | Time |
|----------|------|
| `python figures/run_all.py` (just regenerate all paper figures from `data/`) | 5–10 min |
| `lorenz.ipynb` (single $\beta$, $N=200$) | ~5 min |
| `lorenz.ipynb` ($\beta$ sweep + $N=1300$, multi seeds) | ~30–60 min |
| `worms.ipynb` (cached cluster sequences) | ~5–10 min |
| `worms.ipynb` (re-cluster from scratch at $N=250$) | ~20 min |
| `flies.ipynb` (cached cluster sequences) | ~10–15 min |
| `flies.ipynb` (re-cluster at $N=1000$ on all 30 flies) | ~1–2 h |
| `user_data.ipynb` | depends on $T$, $N$ |

## Citing

If this code is useful in your work, please cite the manuscript:

```bibtex
@article{KaurJainBerman2026,
  title   = {Using timescale as a state coordinate reveals the metastable
             geometry of behavior},
  author  = {Kaur, R. and Jain, K. and Berman, G. J.},
  year    = {2026},
}
```

The G-PCCA implementation we wrap is `pygpcca` (Reuter & Weber 2018); the
power-law fitting routine is the `powerlaw` package (Alstott et al.
2014); UMAP via McInnes et al. (2018).
