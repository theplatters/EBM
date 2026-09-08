# Plot artifact audit

Audited on 2026-09-06 after the CapabilityModel rewrite, its 30-seed risk
aversion experiment, and the completed age-50 avoidance experiment.

This directory contains generated Traffic artifacts only. AssetMarket,
ElFasol, and Sugarscape keep their outputs in their model directories. PNG and
MP4 files are presentation artifacts; CSV files are the run-level or sampled
data used to reproduce the corresponding comparisons. Do not edit generated
data manually.

## Current 5,000-tick capability outputs

| Artifact group | Generator | Replication and uncertainty |
|---|---|---|
| `risk_aversion_runs.csv`, `risk_aversion_distributions.csv`, `risk_aversion_comparison.png`, `risk_aversion_distribution.png` | `notebooks/run_risk_aversion_experiment.jl` | 30 paired seeds; 150 runs and 36,000 distribution rows; means ± between-run SD |
| `uniform_risk_capability_runs.csv`, `uniform_risk_capability_comparison.png`, `uniform_risk_capability_composition.png` | `notebooks/run_uniform_risk_capability_comparison.jl` | 30 paired seeds; 8 conditions (5 static-entry capability + 1 mixed evolutionary + 2 sequential reference); replicate points and 95% mean intervals |
| `activation_habit_*` | `notebooks/run_activation_habit_experiment.jl`, `verify_activation_habit_results.jl` | 30 paired seeds; bootstrap intervals |
| `heterogeneous_strategy/*` | `notebooks/generate_heterogeneous_strategy.jl` | 30 paired seeds; mean ± SD |
| `uniform_risk_mixture_dynamics.csv`, `uniform_risk_mixture_ensemble_dynamics.png` | `notebooks/run_mixture_ensemble_dynamics.jl` | 30 paired seeds; 201 samples per condition (12,060 rows = 30 seeds × 2 scenarios × 201 sampled steps); ensemble mean ± 1 SD |
| `age50_avoidance_mixture_dynamics.csv`, `age50_avoidance_mixture_ensemble_dynamics.png` | `notebooks/run_age50_avoidance_experiment.jl` | 30 paired seeds; 201 samples per condition (12,060 rows = 30 seeds × 2 scenarios × 201 sampled steps); ensemble mean ± 1 SD |
| `examples/*` | `notebooks/generate_traffic_examples.jl` | 5,000-tick seed-42 illustrations |

The uniform-risk capability comparison fixes every car's risk aversion as an
independent, non-heritable Uniform(0,1) draw: five synchronous treatments use
static `EntryDrawReplacement` (capabilities redrawn from entry shares, no
evolution), one synchronous treatment uses `EvolutionaryReplacement` (capabilities
and quantitative traits inherit and mutate, but the temporarily inherited risk is
overwritten by a fresh entry draw), and the two sequential references carry no
`RiskAversion` component. The evolutionary treatment uses capability mutation
rate 0.02 and quantitative-trait mutation scale 0.05.

In the uniform-risk capability comparison and both mixed-capability dynamics
artifacts, traffic throughput is `completed_cells_per_car_step`: the sum of
`CapabilityTickDiagnostics.realized_speed_total` divided by all configured
car-steps. Collided movements therefore contribute zero, and newborn cars do
not inflate throughput through their initial maximum-speed state. The dynamics
CSVs retain `mean_speed` as a separate state measure and aggregate completed
movement over the same 25-tick windows as replacement pressure. The sequential
references in the capability comparison use the analogous share of successful
unit movements, deliberately counting fatal movements as zero. Step-zero rows
in the dynamics CSVs use zero as the pre-simulation window sentinel.

Current capability experiments use lookahead 20. The `run_mixture_ensemble_dynamics.jl`
generator is now current: it samples the uniform-risk static and evolutionary
mixtures described above — still independently entry-drawn, non-heritable
Uniform(0,1) risk with no risk evolution — across 30 paired seeds and reports
ensemble mean ± 1 SD trajectories (the dashed line marks the 1,000-tick burn-in).
The social-habit, animation, and `generate_capability_scenario`/`generate_no_convention`
scripts remain historical/diagnostic generators, not sources for the current
retained evidence.

The age-50 avoidance experiment (`notebooks/run_age50_avoidance_experiment.jl`)
uses the same 30 paired seeds and static mixed-capability entry design in both
conditions. Both use `EntryDrawReplacement`, an
independently entry-drawn, non-heritable Uniform(0,1) lifetime risk, and mixed
H/C/S entry shares of 0.5. The only difference is a threshold treatment that,
at Step age >= 50, suppresses the score contributions of
`SameDirectionResponse`, `OppositeDirectionResponse`, and `NearFieldAvoidance`
without removing those components from the cars. The baseline keeps every
reactive score term active for the full run. The type-level `RiskAversion`
component and the old uniform-risk evolutionary artifact above remain separate,
current retained evidence; this experiment adds a conditional-suppression
treatment on top of the same replacement scheme rather than replacing that
earlier comparison.

## Current 2,000-tick avoidance shutdown-age robustness

| Artifact group | Generator | Replication and uncertainty |
|---|---|---|
| `avoidance_age_robustness.csv`, `avoidance_age_robustness.png` | `notebooks/run_avoidance_age_robustness.jl` | 30 paired seeds; 1,500 paired seed × shutdown-age rows; mean treatment-minus-baseline differences with normal 95% paired-mean intervals |

This sweep uses seeds 20260901:20260930 and compares a no-shutdown baseline
with every shutdown threshold from age 1 through 50. At threshold `N`, the
`SameDirectionResponse`, `OppositeDirectionResponse`, and
`NearFieldAvoidance` score contributions stop when a driver's Step age reaches
`N`. All conditions use the static mixed-capability `EntryDrawReplacement`
design and independently entry-drawn, non-heritable lifetime risk. Runs cover
2,000 ticks with a 1,000-tick burn-in, 120 cars on a 2 × 300 ring, and lookahead
20. Convention strength and speed are sampled every 25 post-burn-in ticks;
replacement rates use every post-burn-in tick. The CSV contains one paired row
per seed and threshold, and the plot reports treatment-minus-same-seed-baseline
means with normal 95% paired-mean intervals.

## Historical retained artifacts

`speed_sensitivity_*`, `lane_first/*`, old `social_habit_*`, old
`capability_*`, and `no_convention_*` are retained historical results from the
removed speed-first/lane-first semantics. They must not be presented as
current evidence or regenerated as current studies. The old speed generator
is removed and its report is no longer reproducible against the current model.

Historical regeneration is permitted only into `plots/historical_regenerated`
(the default for the historical generators, overridable via `TRAFFIC_OUTPUT_DIR`)
and remains a current-semantics diagnostic, never a replacement for the
retained historical evidence above.

The main ensemble scripts accept `TRAFFIC_REPLICATES`, `TRAFFIC_STEPS`,
`TRAFFIC_BURN_IN`, `TRAFFIC_LOOKAHEAD`, `TRAFFIC_PLOT_MAX_STEP`, and
`TRAFFIC_OUTPUT_DIR`. The dynamics plots default to ticks 0–2,000 for the
presentation while their source CSVs retain all 5,000 ticks. The avoidance
experiment additionally accepts `TRAFFIC_AVOIDANCE_DISABLE_AGE`, which sets the
Step age at which the reactive score terms are suppressed (the published
treatment defaults to 50); the paired baseline always keeps them
active for the full run. Development runs may override these values, but
checked-in artifacts must be regenerated at the design documented in their
report. A generator should
write both its plot and source CSV when the plot represents a replicated
comparison. The shutdown-age robustness sweep additionally accepts
`TRAFFIC_MAX_AVOIDANCE_DISABLE_AGE`, which sets the upper bound of the
threshold ladder (the published sweep covers ages 1:50); the baseline keeps
every reactive term active and each threshold treatment suppresses all three
reactive score contributions from the first Step at or above that age.

## Retained fixed-design benchmarks

- `strategy_analysis/*` is the separate 100-tick, 100-replicate occupancy-
  forecast strategy and sensitivity benchmark documented in
  `notebooks/strategy_analysis.md`. The capability changes do not alter that
  legacy forecast path, so its explicit short horizon is retained rather than
  relabeled as a 5,000-tick experiment.
- `mean_age.png`, `habitus.png`, `no_habit.png`, `stay_ratio.png`, and
  `sweeps.png` are the depth-100 presentation sweep generated by
  `notebooks/sweeps.jl` and referenced by the Typst presentation. They are
  retained as presentation/reference outputs, not current capability results.

## Removed relics

- Tick-300 capability and no-convention torus snapshots/animations were
  replaced by 5,000-tick analytical dynamics and replicated comparisons.
- The orphaned 150-tick heterogeneous data and torus image were replaced by a
  reproducible 30-seed long-run benchmark.
- `sweep.png` was removed because it was byte-for-byte identical to the
  canonical `sweeps.png`.
- `random.mkv` was an unreferenced sweep animation and is no longer generated.
