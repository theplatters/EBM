# CapabilityModel risk-aversion experiment

## Design

The completed rewrite was evaluated with 30 paired seeds,
`20260901:20260930`, 5,000 ticks, and a 1,000-tick burn-in. Each run has 120
cars on a 2 × 300 periodic road, lookahead 20, mixed Habit/Convention/SocialHabit
capabilities, and maximum speed 3. Evolution uses mutation rate 0.02 and
`trait_mutation_scale = 0.05`.

The treatments are fixed risk aversion 0, 0.5, and 1; uniform `EntryDraw`; and
uniform initial risk with evolutionary inheritance and mutation. Fixed values
are reset before each decision without consuming RNG. This intervention detail
does not change the core Uniform(0, 1) `EntryDraw` semantics.

## Results

Entries are means across runs; parentheses are between-run standard deviations.
Convention and coordinated time are post-burn-in outcomes, completed is
`completed_cells_per_car_step`: total successfully realized distance of
survivors divided by all car-steps (not simply mean realized speed), and
replacement is replacement pressure. “Danger acceptance” is acceptance of a
dangerous max-speed proposal.

| Treatment | Convention | Coordinated | Completed | Proposed | Danger acceptance | Replacement |
|---|---:|---:|---:|---:|---:|---:|
| Fixed 0 | .966679 (.000750) | .999908 (.000250) | 2.937057 (.001295) | 3.000000 | 1.000000 | .020981 (.000432) |
| Fixed .5 | .959085 (.000941) | .999317 (.001762) | 2.904827 (.002111) | 2.966806 (.000788) | .499664 (.003876) | .025192 (.000570) |
| Fixed 1 | .960429 (.000850) | .999842 (.000222) | 2.875526 (.002553) | 2.922054 (.001562) | 0 | .023403 (.000523) |
| Uniform entry | .958756 (.001031) | .999217 (.001228) | 2.903665 (.002200) | 2.966306 (.000904) | .496310 (.004066) | .025498 (.000559) |
| Evolutionary | .968572 (.002694) | .999883 (.000269) | 2.920262 (.016952) | 2.972313 (.019701) | .546015 (.271928) | .020948 (.001332) |

Population risk summaries are .499355 (.003858) for uniform entry and
.448111 (.273357) for evolution. Pooled within-population dispersion is
.288652 (.001313) and .193157 (.041785), respectively. Final evolutionary
mean risk is .406011 (.300784), versus final uniform-entry mean .492992
(.028302); final SD is .155941 (.052009) versus .291470 (.011897).

## Interpretation

These results do not support a universal claim that selection favors lower
risk. Fixed 0 has the highest throughput and lower replacement pressure than
fixed .5 or 1, while replacement is nonmonotone and worst at .5. Evolution
improves average convention, throughput, and replacement pressure relative to
uniform entry, but evolved risk is highly path dependent: between-seed
variation is large and outcomes often sit near low- or high-risk regimes. The
aggregate evolved final mean is below .5, but that difference is dominated by
path dependence.

## Artifacts and reproduction

The current artifacts are `../plots/risk_aversion_runs.csv` (150 rows plus
header, 31 columns), `../plots/risk_aversion_distributions.csv` (36,000 rows
plus header, 5 columns), `../plots/risk_aversion_comparison.png`, and
`../plots/risk_aversion_distribution.png`.

```sh
julia --project=. -t auto notebooks/run_risk_aversion_experiment.jl
```
