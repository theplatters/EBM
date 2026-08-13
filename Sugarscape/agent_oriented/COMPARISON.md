# ECS and Agents.jl Sugarscape comparison

This comparison covers three outcome-equivalent implementations of the repository's
single-resource Sugarscape: the Ark.jl ECS implementation with either movement mode,
`AgentSequential`, and `AgentSynchronous`. The equivalence suite compares every citizen
field exposed by `CitizenState`, the complete landscape matrix, and every logger series
at initialization and after every step. It also forces synchronous contention,
old-age replacement, starvation without replacement, disease transmission and
recovery, and reproduction. The results below therefore compare implementation
architectures executing the same model semantics, not merely similar variants.

## Performance

The benchmark measures a complete setup plus 100 model steps. Each row is the median
of 20 independently initialized runs after one unmeasured compilation/warm-up run.
Each ECS/Agents.jl pair uses the same parameters and seed. Measurements were made on
2026-08-13 with Julia 1.12.6, BenchmarkTools 1.7.0, one Julia thread, and an AMD Ryzen
7 5825U CPU on Linux.

The baseline uses a 50 × 50 grid, 400 initial citizens, seed 20260813, and the default
replacement regime. The extended case uses the same size, population, seed, and
horizon, with replacement disabled, reproduction enabled at probability 0.02, and an
initial infection probability of 0.05; all other parameters retain their defaults.

| Scenario | Implementation | Median time (ms) | Median memory (MiB) | Allocations |
|---|---|---:|---:|---:|
| Baseline | ECS sequential | 120.924 | 135.556 | 489,443 |
| Baseline | Agents.jl sequential | 119.718 | 124.537 | 484,400 |
| Baseline | ECS synchronous | 112.364 | 146.288 | 557,279 |
| Baseline | Agents.jl synchronous | 125.198 | 139.187 | 554,892 |
| Extended | ECS sequential | 83.899 | 97.969 | 406,127 |
| Extended | Agents.jl sequential | 88.513 | 86.031 | 407,501 |
| Extended | ECS synchronous | 86.141 | 90.732 | 370,966 |
| Extended | Agents.jl synchronous | 78.498 | 79.850 | 375,332 |

There is no single winner in these measurements. Sequential baseline time is nearly
tied: Agents.jl is about 1.0% faster. ECS is about 10.3% faster in the synchronous
baseline and about 5.2% faster in the extended sequential case. Agents.jl is about
8.9% faster in the extended synchronous case. The Agents.jl runs use less median
allocated memory in all four comparisons (about 4.9–12.2%), while allocation counts
are close. The extended scenario is faster than the baseline because its no-replacement
population trajectory changes the amount of later work; comparisons should be made
within a scenario and movement mode, not between scenarios.

These are measurements of the checked-in code on one machine, not universal framework
performance claims. Runtime differences of this size can change with Julia, package,
hardware, garbage-collector, and model-parameter changes. Reproduce the table with:

```sh
julia --project=. Sugarscape/agent_oriented/benchmark.jl
```

The script accepts `SUGARSCAPE_BENCH_SEED`, `SUGARSCAPE_BENCH_STEPS`,
`SUGARSCAPE_BENCH_WIDTH`, `SUGARSCAPE_BENCH_HEIGHT`,
`SUGARSCAPE_BENCH_POPULATION`, and `SUGARSCAPE_BENCH_SAMPLES` overrides.

## Lines of code

The count below is source lines of code (SLOC): nonblank lines whose first
non-whitespace character is not `#`. It includes imports, declarations, docstrings,
and module boilerplate, and excludes tests, plotting, interactive visualization,
documentation, and this benchmark. This deliberately simple definition is reproducible
without a language-specific counter:

```sh
awk 'NF && $1 !~ /^#/ {n++} END {print n}' \
  Sugarscape/components/*.jl Sugarscape/core/*.jl Sugarscape/systems/*.jl \
  Sugarscape/simulation/*.jl Sugarscape/analysis/logger.jl \
  Sugarscape/analysis/runner.jl
awk 'NF && $1 !~ /^#/ {n++} END {print n}' \
  Sugarscape/agent_oriented/sequential.jl
awk 'NF && $1 !~ /^#/ {n++} END {print n}' \
  Sugarscape/agent_oriented/synchronous.jl
```

| Implementation scope | SLOC | Interpretation |
|---|---:|---|
| ECS computational model, both movement modes | 994 | Shared components, resources, features, logging, setup, and runner |
| Agents.jl sequential | 514 | Self-contained sequential model |
| Agents.jl synchronous | 604 | Self-contained synchronous model |
| Both Agents.jl files | 1,118 | Includes duplicated setup, disease, lifecycle, reproduction, and logging |

The fairest conclusion depends on the unit of comparison. One Agents.jl variant is
roughly half the SLOC of the dual-mode ECS implementation. Providing both standalone
Agents.jl variants, however, takes 124 more SLOC than the shared dual-mode ECS model.
The ECS movement file itself has 79 shared SLOC, 32 sequential-only SLOC, and 73
synchronous-only SLOC. Thus dividing all 994 ECS lines between the modes would
double-count shared infrastructure, while comparing 994 against only one Agents.jl
file would charge ECS for its second movement mode. The table reports both views
instead of assigning shared lines arbitrarily.

## Architecture

### State representation

The ECS citizen is an entity assembled from narrow components. Identity, position,
proposal, vision, metabolism, sugar, age, maximum age, and initial endowment are
separate value types. Sex is represented by mutually exclusive `Female`/`Male` tag
components. Infection is structural: an infected citizen has an `Infection` component,
and recovery removes it. Persistent world state lives in resources: parameters, the
seeded RNG, clock, next citizen ID, landscape, occupancy grid, step events, and logger.

Each Agents.jl citizen is instead one mutable `@agent` record containing spatial state,
traits, wealth, age, sex as a `Symbol`, immunity, and
`infection::Union{Nothing, Infection}`. Infection changes a field value rather than the
agent's structure. Parameters, landscape, event counters, clock, and logger are model
properties. `AgentSequential` delegates occupancy to `GridSpace`; `AgentSynchronous`
also maintains an explicit occupancy matrix because all proposals must observe the same
pre-movement state.

This makes per-citizen state locally readable in the agent-oriented versions: one agent
object shows nearly everything about that citizen. The ECS representation makes data
roles and structural heterogeneity explicit and lets systems query only the components
they require. Its cost is more wrapper types, resources, queries, and conversion between
component records and analysis snapshots.

### Behavior and schedule

ECS behavior is split by concern across `systems/growback.jl`, `movement.jl`,
`disease.jl`, `lifecycle.jl`, and `reproduction.jl`. `simulation/step.jl` is the explicit
orchestrator. Systems query component columns, stage structural changes, and commit
entity additions/removals outside active queries.

The Agents.jl versions organize the same phases as functions inside each movement-mode
module. Their step functions retain the same order:

```text
growback -> movement -> disease -> lifecycle -> reproduction -> clock -> logger
```

This preserves outcome equivalence, but feature code is duplicated between the two
standalone modules. The duplication makes each model independently readable and
modifiable; the ECS version centralizes shared features and selects only the movement
system within one schedule.

### Sequential and synchronous coordination

For sequential movement, both versions sort citizens by stable ID, shuffle that list
with the seeded model RNG, and immediately expose each move and harvest to later
citizens. ECS updates an `OccupancyGrid` resource and later applies component values;
Agents.jl uses `move_agent!` and the grid's position index.

For synchronous movement, both versions first select proposals from the same occupancy
and landscape, group contenders by destination, sort destinations and IDs, use the
seeded RNG to select conflict winners, and then commit moves. ECS stores proposals in a
`ProposedPosition` component. `AgentSynchronous` stores `proposed_pos` in every mutable
agent and uses its explicit occupancy matrix. This is the main reason the synchronous
Agents.jl model is longer than the sequential one: synchronous semantics require an
agent-oriented implementation to add an explicit proposal/commit protocol rather than
relying on ordinary one-agent-at-a-time activation.

### Practical trade-off

The agent-oriented representation is compact for a single movement regime and keeps
all individual state together. The ECS representation is more decomposed and has a
larger fixed vocabulary, but shares features cleanly across movement regimes and makes
optional structure and system data dependencies visible. In this implementation,
architecture choice is more consequential for organization and extensibility than for
speed: warmed end-to-end timings are close and reverse ordering across the two tested
scenarios.
