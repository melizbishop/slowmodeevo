# Handoff — arm syllable enrichment / PCA (session of 2026-09-17)

Context for continuing on another machine. Everything below is in this repo
unless stated. Companion write-ups live in the Claude project `slowmodeevo`
(`claude/syllable-enrichment-pca-arms.md`, `claude/arm-transition-structure.md`,
`claude/arm-support-coverage.md`, `claude/loo-predictive-information-correction.md`).

---

## 1. What this analysis is

Notebook `notebooks/02_slowmode_arrays_plots.ipynb`, sections **13-15** (new).

Each species has two G-PCCA "arms" (macrostates of its transfer operator, tau =
120 frames at 120 Hz = 1 s; implied dwell 2.2-6.6 s). The question was whether
the arms differ in MoSeq syllable composition, and whether species differ in how
they use that contrast.

Core quantity, per species, over 49-50 syllables:

    L[sp, arm](s) = log2( p(s | sp, arm) / p(s | sp) )        # own-repertoire reference
    delta_sp(s)   = log2( p(s | sp, arm2) / p(s | sp, arm1) ) # arm contrast, reference-free

PCA is run on the 8 per-species `delta` vectors (and on the 16 species-arm rows).

## 2. Three data problems found and fixed — read this before trusting old figures

**(a) The `moseq_label` column is stale.** It carries placeholder `syllable_<n>`
for clusters 12, 15, 25, 30, 43, 46, 49 — really `leave_rear_turn_right`,
`leave_rear_turn_left`, `mid_rear`, `mid_rear_4`, `very_fast`, `groom_9`,
`groom_12`. Five are rear/locomotion, i.e. exactly what loads on the arm axis, so
any name-based filter biases against the signal.
→ Use **`moseq_syllable_labels.csv`** (repo root; transcribed from
kpms_syll_names-022525). Named syllables are clusters 0-49, which are exactly the
50 present in every species-arm. Clusters >= 50 are the unnamed noise tail.
Other repo CSVs still carry stale labels: `syllable_rank_table.csv` (736 rows),
`basin_moseq_syllable_profiles.csv` (296), `syllable_species_enrichment_by_arm.csv` (168).

**(b) `matched_arm_syllable_probs.csv` uses only the FIRST recording per subject.**
70.3M soft frames instead of 147.7M. Peromyscus are mostly single-recording and
were unaffected; Mus average 3-4 recordings/subject and were running on a quarter
to a third of their data — an asymmetry along the exact contrast being studied.
→ Rebuilt as **`matched_arm_syllable_probs_full_join.csv`** (and `_nocut.csv`) by
`scripts/join.py`. Validated: reproduces the reference at ratio 1.0000 when
restricted to the reference's scope.

**(c) The low-occupancy cut covers 45-99.8% of frames depending on species.**
In `with_low_occupancy_cutoff`, frames in cut clusters get chi ~ 0 in BOTH arms
and drop out of per-arm totals. Coverage: P. gossypinus 99.8%, P. leucopus 45.2%.
→ §13-15 now default to `ARM_VIEW = "without_low_occupancy_cutoff"` (100% for all).

**Still unexplained:** what sets `active_global_clusters` (325 for P. leucopus,
975 for P. gossypinus). It tracks neither frame count nor repertoire breadth.
Also `soft_grouped_transfer_operator_summary.csv` reports
`n_retained_clusters = 1000` / `supported_occupancy_fraction = 1.0` for all eight
species, which contradicts `active_global_clusters.npy`. That column is
mislabelled and currently hides problem (c).

**The arms themselves are fine.** `n_states == projection n_frames - (d_embed-1)`
for all 354 subjects (159.6M frames), so operators/G-PCCA used all recordings.
Only the syllable readout was truncated. No need to refit the operators.

## 3. Toggles in §13 (first code cell of section 13)

```python
SYLLABLE_SUBSET = "labeled"                    # 50 named (clusters 0-49) | "all" (89)
DROP_CLUSTERS   = {26}                         # still_2, flagged "possible noise"
ENRICH_REFERENCE = "species"                   # "pooled" reproduces the old CSV convention
ARM_VIEW        = "without_low_occupancy_cutoff"
USE_FULL_JOIN   = True
```

Sensitivity already checked: syllable subset (rho of PC1 >= 0.95 across 49/50/89),
pseudocount swept 200x (rho >= 0.976), leave-one-species-out (|r| >= 0.983),
dropping cluster 26 (loading cosine 0.92; only P. leucopus moves, -21%).

## 4. What the result actually is, after the fixes

**Survived everything: a single shared arm axis.** Arm 1 = rearing + locomotion,
arm 2 = grooming. Every species rides it; R² on a leave-one-species-out consensus
is 0.69-0.97 (M. spretus 0.69 the weakest). PC1 47.6%, PC2 31.9%.

**Did NOT survive: the genus split.** Originally reported as Peromyscus separating
their arms ~5x more strongly than Mus, non-overlapping. Gains through the fixes:

| species | original | + full join | + no cut |
|---|---|---|---|
| M. caroli | -0.09 | 3.66 | 6.05 |
| M. spretus | 1.33 | 5.68 | 6.19 |
| M. musculus | 2.04 | 8.98 | 8.95 |
| P. gossypinus | 9.16 | 9.28 | 9.11 |
| P. polionotus | 9.49 | 10.47 | 9.51 |
| P. leucopus | 11.15 | 11.22 | 10.00 |
| P. maniculatus | 9.28 | 10.35 | 10.12 |
| P. californicus | 9.03 | 10.46 | 10.33 |

M. musculus (8.95) now sits inside the Peromyscus range. M. caroli, previously
reported as a null at the noise floor, is not a null.

**Composition, not transitions.** Bout-to-bout transition kernels are not
arm-invariant (JSD 2.6-8.2x a count-matched permutation null in 5 of 7 testable
species), but the effect is small (0.017-0.039 bits of a possible 1), and once
availability is removed the grammars are near-identical: transition *lift*
correlates r = 0.90-0.98 between arms in every species. The slow mode partitions
*what* occurs, not *how it chains*. See `scripts/trans.py`, `scripts/trans_test.py`.

**M=2 is under-resolved.** Your corrected LOO PI is still rising at M=8; M=2
captures 10-48% of it. lambda2-lambda3 gaps are thin for several species. Of 1000
clusters, only 4-51 have chi_arm2 > 0.5. Treat "two arms" as a coarse slice.

## 5. The per-animal result — the most solid thing here

`scripts/per_animal.py` builds per-animal cluster occupancy (1000-d) and per-animal
arm contrast (50-d) for 348 animals. Balanced leave-one-animal-out nearest-centroid:

| representation | species (8-way, chance 12.5%) | genus (chance 50%) | species eta² |
|---|---|---|---|
| cluster occupancy | **81.4%** | 82.2% | 0.266 |
| arm contrast delta | **63.3%** | 79.7% | 0.216 |

Permutation nulls sit at chance; p = 0.000 for all four. **Species identity is
strongly encoded** — a single animal's behaviour identifies its species 4 times
in 5. So the representation is not empty; what dissolved was one fragile summary.

With animals as replicates, per-animal ||delta|| medians: Mus 9.20, Peromyscus
10.50. With species as the unit (the correct one for a genus claim, n = 3 vs 5),
U-test p = 0.036 — real but weak, IQRs overlapping.

**Within-species SD of ||delta|| = 2.08; between-species SD of means = 1.80.**
Individual variation is comparable to species variation. That is why every
species-mean analysis here was fragile.

## 6. Open lead: M. spretus

On measures computed from raw state histograms only (no chi, no arms, no MoSeq
join, no cut — none of the machinery that produced the artifacts above):

| species | effective clusters used | top-50 clusters hold | arm-2 occupancy |
|---|---|---|---|
| seven others | 743-850 | 13.4-18.8% | 0.06-0.21 |
| **M. spretus** | **345** | **43.9%** | **0.402** |

Not one bad cluster (top cluster 2.5% of frames), not one animal (per-animal
median 291 vs 570-730), not sample size (M. caroli has matched n and sits at 803).
M. spretus is also the weakest fit to the shared axis and the PC2 extreme. Its
||delta|| is ordinary, so the unusual thing is repertoire concentration and arm
balance, not arm contrast.

## 7. Suggested next steps

1. Write up §per-animal properly — it is what makes everything else trustworthy.
2. Chase M. spretus with per-animal error bars on repertoire breadth / arm balance.
3. Find what sets `active_global_clusters`; fix the summary CSV columns.
4. Regenerate the stale-label and truncated-join CSVs listed in §2.
5. Extend M past 8 to find where PI saturates; consider PI per added basin.
6. Persist tau and framerate in run outputs (both had to be inferred).

## 8. Files added this session

```
moseq_syllable_labels.csv                     curated names + categories (clusters 0-49)
HANDOFF_arm_syllable_analysis.md              this file
scripts/cache_moseq.py                        cache per-frame syllables from the wide CSV
scripts/join.py                               rebuild soft_frames from ALL recordings
scripts/per_animal.py                         per-animal occupancy + arm contrast
scripts/trans.py, scripts/trans_test.py       bout transition kernels + arm-invariance test
notebooks/02_slowmode_arrays_plots.ipynb      sections 13, 14, 15 added
notebooks/02_slowmode_arrays_plots.ipynb.bak  pre-edit backup
outputs/.../matched_arm_syllable_probs_full_join{,_nocut}.csv
outputs/.../syllable_enrichment_pca/          scores, loadings, variance, delta matrices
outputs/.../arm_transition_structure/         arm_transition_invariance_summary.csv
Claude outputs/syllable_enrichment_pca_figures/   all §13-15 figures
```

**External dependency:** per-frame syllables are at
`~/moseq_dataverse/multispecies_moseq_timeseries.csv` (NOT in this repo).
Layout gotcha: `recording_id` is column 0, then `t_0 … t_215994`, then
`subject, strain, genus, species, subspecies` at the END. 740 recordings.
Some recordings are 216,008 frames but the CSV is 215,995 wide, so their tails
are truncated — `scripts/join.py` handles this.
