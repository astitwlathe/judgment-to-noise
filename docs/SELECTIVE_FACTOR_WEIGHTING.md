# Selective factor weighting: schematic adherence vs factor collapse

**Status:** investigative notes, 2026-04-13. Not yet in the NeurIPS draft;
captured here for archival reference while drafting.

## Question

Does a well-separated rubric (low factor collapse) make LLM judges "ignore"
some factors when producing the overall verdict, driving down schematic
adherence? In other words, is there an inverse relationship between factor
collapse and schematic adherence that reflects a judge behavior (selective
attention) rather than a mathematical artifact of collapsed rubrics?

## Summary

- The observational inverse relationship holds across our three benchmarks
  (Arena-Hard low collapse / low adherence, FLASK high collapse / high
  adherence), but the **simpler mechanism** is that collapsed rubrics make
  schematic adherence trivially achievable because a single factor already
  predicts the overall.
- A stronger finding emerged from the investigation: **selective factor
  weighting is visible on well-separated rubrics**. On Arena-Hard across
  11 (judge × setting) cells, judges systematically underweight safety
  (per-unit coefficient in $[-0.10, +0.13]$) and overweight style
  ($[+0.20, +0.61]$), even though the rubric states equal weighting.
- The widely-cited $-0.76$ conciseness loading (from the older
  `fig:elo_collapse` figure / `results.tex:19`) is **ELO-specific** and
  does **not reproduce** on any of our 11 (judge × setting) cells.
  The closest match in the full ELO coefficient matrix is QwQ-32B
  setting 4 style $= -0.74$, not conciseness for GPT-4o-mini. For
  GPT-4o-mini setting 1 specifically, conciseness under ELO is $+0.29$.
  On raw 1--5 factor scores, conciseness has small *positive*
  coefficients (0.03--0.20) in every Arena-Hard cell.
- The within-Arena-Hard data **cannot fully disentangle** the user's
  directional hypothesis ("more separation $\to$ more ignoring") because
  factor collapse is nearly constant across Arena-Hard settings
  (mean $|r| = 0.30$--$0.43$). We can only show that selective weighting
  is *visible* when separation is non-trivial.

## Method

### Decomposition of schematic adherence

For each (judge, benchmark) cell, we compute:

- `best_single_r2`: $R^2$ of `overall ~ (single best factor)` maximized over
  factors. Captures "how much of the overall verdict can be explained by
  one factor alone."
- `pc1_r2`: $R^2$ of `overall ~ PC1(factors)`. If factors are collapsed,
  PC1 explains as much as all factors together; the "marginal value of
  using multiple factors" $=$ `all_r2 - pc1_r2`.
- `all_r2`: $R^2$ of `overall ~ all factors` (the published schematic
  adherence measure; poly term ignored here because the linear component
  dominates).
- `gain = all_r2 - best_single_r2`: gain from using all factors beyond the
  best single one.
- `mean |r|`: mean absolute pairwise Spearman correlation between factors,
  as a summary of factor collapse.
- Per-factor per-unit coefficient $\beta_i$ from
  `LinearRegression().fit(X, y).coef_` (raw, not standardized; captures
  "by how much does the overall verdict shift per 1-point shift in
  factor $i$, holding other factors fixed").

### Cross-benchmark summary

Data: setting 1 / single ajudge job per benchmark, averaged across the
4 judges DeepSeek-R1-32B, GPT-4o-mini, GPT-3.5-Turbo, QwQ-32B.

| Benchmark | Judge | `best_single_r2` | `all_r2` | gain | mean $\lvert r \rvert$ |
|---|---|---:|---:|---:|---:|
| Arena-Hard | DeepSeek-R1-32B | 0.47 | 0.50 | 0.02 | 0.43 |
| Arena-Hard | GPT-4o-mini    | 0.80 | 0.87 | 0.08 | 0.40 |
| Arena-Hard | GPT-3.5-Turbo  | 0.71 | 0.78 | 0.06 | 0.30 |
| Arena-Hard | QwQ-32B        | 0.70 | 0.74 | 0.05 | 0.43 |
| MT-Bench   | DeepSeek-R1-32B | 0.82 | 0.94 | 0.12 | 0.51 |
| MT-Bench   | GPT-4o-mini    | 0.90 | 0.97 | 0.07 | 0.69 |
| MT-Bench   | GPT-3.5-Turbo  | 0.72 | 0.90 | **0.18** | 0.41 |
| MT-Bench   | QwQ-32B        | 0.89 | 0.94 | 0.05 | 0.52 |
| FLASK      | DeepSeek-R1-32B | 0.80 | 0.92 | 0.13 | 0.70 |
| FLASK      | GPT-4o-mini    | 0.92 | 0.96 | 0.04 | 0.84 |
| FLASK      | GPT-3.5-Turbo  | 0.86 | 0.96 | 0.10 | 0.86 |
| FLASK      | QwQ-32B        | 0.86 | 0.95 | 0.09 | 0.81 |

Key takeaways:

1. **FLASK's high schematic adherence is substantially an artifact.** The
   best single factor alone already gives $R^2 = 0.80$--$0.92$. The
   "judge follows the rubric" interpretation is mathematically
   indistinguishable from "any one factor predicts the overall because
   all factors are redundant."

2. **Arena-Hard gains from adding factors are smaller (2--8%) than on
   MT-Bench (5--18%)**, but that's because single-factor $R^2$ on
   Arena-Hard is also much lower (DeepSeek: 0.47 best-single vs 0.82
   on MT-Bench). This is consistent with *the verdict being detached
   from any single factor*, not with selective attention.

3. **MT-Bench GPT-3.5-Turbo is the clearest example of multi-factor
   integration** (gain 0.18): using all six factors adds 18 points of
   $R^2$ beyond the best single one.

### Per-setting decomposition within Arena-Hard

Using all 11 (judge × setting) cells available under the
`InDepthAnalysis/` tree. `|r|` reports the mean absolute pairwise Spearman
correlation; `R^2` is the linear multi-factor fit; gain is
`all - best_single`.

| Setting | n | $\lvert r \rvert$ | best1 | all | gain |
|---|---:|---:|---:|---:|---:|
| DeepSeek-R1-32B setting1 | 12000 | 0.43 | 0.47 | 0.50 | 0.02 |
| DeepSeek-R1-32B setting2 | 10000 | 0.36 | 0.28 | 0.33 | 0.05 |
| DeepSeek-R1-32B setting3 (reasoning) | 10000 | 0.40 | 0.55 | 0.62 | 0.07 |
| DeepSeek-R1-32B setting4 | 12000 | 0.35 | 0.14 | 0.15 | 0.01 |
| DeepSeek-R1-32B setting5 (reasoning) | 12000 | 0.35 | 0.06 | 0.06 | 0.00 |
| GPT-3.5-Turbo-0125 setting1 | 12000 | 0.30 | 0.71 | 0.78 | 0.06 |
| GPT-4o-mini-0718 setting1 | 12000 | 0.40 | 0.80 | 0.87 | 0.08 |
| QwQ-32B setting1 | 12000 | 0.43 | 0.70 | 0.74 | 0.05 |
| QwQ-32B setting2 | 9000 | 0.31 | 0.59 | 0.71 | 0.12 |
| QwQ-32B setting4 | 12000 | 0.42 | 0.66 | 0.73 | 0.07 |
| QwQ-32B setting5 (reasoning) | 12000 | 0.39 | 0.68 | 0.75 | 0.07 |

Within-Arena-Hard, mean $|r|$ is nearly constant (0.30--0.43). DeepSeek
settings 4 and 5 show near-zero $R^2$ ($0.06$--$0.15$): the overall verdict
is essentially detached from all factors, not "judge used one factor." QwQ
is strikingly stable across settings ($R^2 = 0.71$--$0.75$, gain
0.05--0.12).

### Per-factor per-unit coefficients on Arena-Hard

Raw $\beta$ from `LinearRegression().fit(X, y).coef_`. Per-unit: "by how
much does the overall verdict shift per 1-point shift in this factor?"

| Setting | correct. | complete. | **safety** | concise. | style |
|---|---:|---:|---:|---:|---:|
| DeepSeek-R1-32B setting1 | 0.59 | 0.29 | 0.09 | 0.03 | 0.20 |
| DeepSeek-R1-32B setting2 | 0.54 | 0.37 | 0.20 | 0.11 | 0.32 |
| DeepSeek-R1-32B setting3 (reasoning) | 0.63 | 0.35 | 0.13 | 0.09 | 0.24 |
| DeepSeek-R1-32B setting4 | 0.10 | 0.41 | **−0.02** | 0.09 | 0.61 |
| DeepSeek-R1-32B setting5 (reasoning) | 0.15 | 0.53 | 0.09 | 0.03 | 0.44 |
| GPT-3.5-Turbo setting1 | 0.02 | 0.83 | 0.06 | 0.20 | 0.47 |
| GPT-4o-mini setting1 | 0.18 | 0.47 | **−0.10** | 0.10 | 0.54 |
| QwQ-32B setting1 | 0.59 | 0.31 | 0.05 | 0.08 | 0.22 |
| QwQ-32B setting2 | 0.49 | 0.44 | 0.07 | 0.14 | 0.29 |
| QwQ-32B setting4 | 0.51 | 0.36 | 0.04 | 0.07 | 0.28 |
| QwQ-32B setting5 (reasoning) | 0.53 | 0.33 | 0.08 | 0.05 | 0.25 |

Observations:

- **Safety coefficient is near zero in every cell** ($[-0.10, +0.13]$).
  GPT-4o-mini's is genuinely negative (−0.10).
- **Style coefficient is always positive and often dominant** (0.20--0.61).
- Other factors (correctness, completeness) dominate or trade weight
  depending on judge/setting.
- Reasoning-enabled cells (settings 3 and 5) do not look systematically
  different from no-reasoning cells (1, 2, 4) in which factors get weight
  — so "reasoning makes judges integrate better" is not supported here.

### Variance caveat on safety

Safety verdicts are tied ($= 3.0$) 91--99.8% of the time across all
11 settings. Standard deviation on safety is 0.05--0.42 vs 0.25--1.22
for the other factors:

| Setting | $\sigma_{\text{overall}}$ | $\sigma_{\text{correct}}$ | $\sigma_{\text{complete}}$ | $\sigma_{\text{safety}}$ | $\sigma_{\text{concise}}$ | $\sigma_{\text{style}}$ | safety % tied |
|---|---:|---:|---:|---:|---:|---:|---:|
| DeepSeek-R1-32B setting1 | 1.42 | 0.98 | 0.97 | 0.25 | 0.84 | 0.87 | 96.7% |
| DeepSeek-R1-32B setting4 | 1.13 | 0.37 | 0.42 | 0.09 | 0.38 | 0.38 | 99.5% |
| GPT-3.5-Turbo setting1 | 0.95 | 0.50 | 0.75 | 0.05 | 0.62 | 0.45 | 99.8% |
| GPT-4o-mini setting1 | 1.11 | 0.86 | 0.99 | 0.42 | 0.94 | 0.93 | 91.8% |
| QwQ-32B setting1 | 1.37 | 1.18 | 1.06 | 0.28 | 0.90 | 0.88 | 95.2% |
| (remaining rows similar) | | | | | | | |

Interpretation: the near-zero safety coefficient partly reflects the fact
that safety rarely varies in Arena-Hard questions. But
GPT-4o-mini's coefficient of $-0.10$ is *not* explainable by low variance
alone — its safety $\sigma = 0.42$ is non-trivial, yet it still attracts
a mildly negative coefficient. That particular judge is actively
downweighting safety when it does vary.

## What the data does and does not support

**Supports (strong):**

- Selective weighting is visible on well-separated rubrics. Safety is
  consistently under-utilized (near-zero or slightly negative $\beta$)
  and style is consistently amplified across all 11 Arena-Hard cells.
- On FLASK (mean $|r| = 0.83$) and MT-Bench (mean $|r| = 0.65$) any such
  selective weighting would be statistically indistinguishable from equal
  weighting, because the factors carry nearly the same information.
  Schematic adherence on collapsed benchmarks is therefore a weaker test
  than on separated ones.
- The inverse observational relationship between collapse and schematic
  adherence is mostly a mathematical property of collapsed rubrics,
  not a behavioral signature of judges.

**Does not support (original conciseness claim):**

- Conciseness does *not* have a negative coefficient on raw 1--5 factor
  scores. The $-0.76$ from the older `fig:elo_collapse` figure does
  not reproduce as conciseness for any judge × setting under ELO
  either. See the ELO coefficient table (below); the closest value
  is $-0.74$ for QwQ-32B setting 4 on *style*, not conciseness.
  The $-0.76$ may have been a transcription error, or derived from a
  data snapshot we no longer have. Regardless, the ELO regression is
  underpowered ($n = 10$--$12$ models fit with 5 predictors, $R^2 \geq
  0.95$ everywhere) and volatile in sign across settings. The current
  `results.tex` sentence on conciseness should be softened or moved to
  the ELO appendix subsection.

**Does not support (user's directional hypothesis):**

- "More separation $\to$ more ignoring" cannot be tested within
  Arena-Hard because mean $|r|$ is nearly constant across the 11
  settings (0.30--0.43). The cross-benchmark gradient (Arena-Hard 0.37
  $\to$ MT-Bench 0.65 $\to$ FLASK 0.83) is observationally consistent
  with the hypothesis but confounded by rubric/format (pairwise vs
  pointwise), factor-count, and evaluation-domain changes.

## Suggested paper edits (deferred)

1. **Soften the conciseness sentence** (`results.tex:19`) to clarify
   that the $-0.76$ is an ELO-specific artifact. Replace with a
   safety-focused claim that is robust across all 11 cells on raw
   scores.

2. **Add the per-factor-coefficient table** to the appendix (probably
   near `sec:factor_collapse` or as a new subsection). It's the
   headline "judges have preferences not in the rubric" evidence.

3. **Update the "two metrics diagnose different failure modes"
   paragraph** in `results.tex` to explicitly note the dissociation:
   schematic adherence is a stronger test on benchmarks with
   well-separated factors; on collapsed benchmarks it is trivially
   achievable.

## ELO-transformed coefficient table

Standardized coefficients from regressing overall-ELO on factor-ELOs
across models in each setting. The ELO per factor is computed by
rebuilding pairwise battles with `build_original_battles_from_jsonl(
target_metric=<factor>_score)` (nothing was missing in the API — the
base_processed JSONLs carry per-factor verdict tokens alongside the
overall `score` token), then running `compute_mle_elo` against the
`gpt-4-0314` baseline.

| Setting | correct. | complete. | safety | concise. | style | $R^2$ | $n$ |
|---|---:|---:|---:|---:|---:|---:|---:|
| DeepSeek-R1-32B setting1 | −0.17 | 0.90 | 0.10 | 0.32 | −0.08 | 0.98 | 12 |
| DeepSeek-R1-32B setting2 | 0.59 | 0.70 | −0.49 | −0.07 | 0.10 | 0.99 | 11 |
| DeepSeek-R1-32B setting3 (reas.) | −0.02 | 1.64 | −0.48 | 0.24 | 0.02 | 0.99 | 11 |
| DeepSeek-R1-32B setting4 | −0.75 | 2.46 | 0.20 | 0.27 | **−1.10** | 0.97 | 12 |
| DeepSeek-R1-32B setting5 (reas.) | −0.87 | 1.60 | 0.35 | 0.05 | −0.12 | 0.95 | 12 |
| GPT-3.5-Turbo setting1 | 0.27 | 0.55 | 0.02 | −0.00 | 0.18 | 1.00 | 12 |
| GPT-4o-mini setting1 | **−1.15** | 0.80 | 1.15 | 0.29 | 0.29 | 1.00 | 12 |
| QwQ-32B setting1 | −0.09 | 0.85 | 0.17 | 0.34 | −0.10 | 0.99 | 12 |
| QwQ-32B setting2 | 0.67 | 0.03 | 0.06 | −0.02 | 0.23 | 1.00 | 10 |
| QwQ-32B setting4 | 0.68 | 0.88 | 0.09 | 0.18 | **−0.74** | 0.96 | 12 |
| QwQ-32B setting5 (reas.) | 0.92 | 0.77 | −0.10 | 0.04 | −0.61 | 0.98 | 12 |

Warnings about this table:

- $n = 10$--$12$ models regressed on 5 predictors is severely underpowered.
  $R^2 \geq 0.95$ everywhere is evidence of overfit, not of judge
  coherence. Coefficient signs flip across settings for the same judge
  (e.g., DeepSeek correctness: $-0.17, 0.59, -0.02, -0.75, -0.87$), so
  any single-cell coefficient should be treated as noisy.
- The raw-scale per-unit $\beta$ table (above) is $\sim$1000 times better
  powered ($n \approx 12{,}000$ rows per cell vs $n \approx 12$ here) and
  shows genuinely stable patterns (safety near zero in every cell; style
  always positive). Use that for headline claims.
- The ELO transformation also asymmetrically weights strong preferences
  (A$\gg$B, B$\gg$A weighted $3\times$) and converts all judgments to
  binary outcomes, so ELO coefficients are not directly comparable to
  raw-scale coefficients. See `sec:elo-failures` in the paper appendix
  for details.

## Reproduction

- **Saved script**: `scripts/analysis/selective_weighting_analysis.py`.
  Runs both the raw-scale and ELO decompositions across all
  `InDepthAnalysis/*-setting*` directories and emits
  `data_analysis/selective_weighting_raw.csv` and
  `data_analysis/selective_weighting_elo.csv`. Run with
  `conda run -n abb python scripts/analysis/selective_weighting_analysis.py`.
- Raw-scale data loading: `differential_debiasing.interfaces.judge_data.
  load_judge_evaluations` + `extract_scores_from_evaluations` on
  `InDepthAnalysis/{judge}-setting{N}/base_processed/`. Out-of-range
  filter (1--5), drop rows with any NaN in factor or overall columns.
- ELO: `differential_debiasing.interfaces.arena_battles.
  build_original_battles_from_jsonl(..., target_metric=<factor>_score)`
  plus `differential_debiasing.interfaces.elo_bootstrap.compute_mle_elo`
  (MLE via logistic regression, same semantics as Arena-Hard's
  `show_result.py`). Baseline model: `gpt-4-0314`.
- Regression: `sklearn.linear_model.LinearRegression` for all fits
  (single-factor, PCA-1, full multi-factor). `sklearn.decomposition.PCA`
  for PC1. `scipy.stats.spearmanr` for factor correlations.
- Cross-benchmark rows (MT-Bench, FLASK) use `MTBenchLoader` and
  `FLASKLoader` from `differential_debiasing.benchmarks/` on the
  corresponding ajudge job directories.
